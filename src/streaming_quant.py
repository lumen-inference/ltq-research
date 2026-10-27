"""
Layer-major quantizer and evaluator for models that do not fit in VRAM
(Qwen3-8B dense, Qwen3-30B-A3B MoE).

A single pass over layers:
  - layer weights are read from disk (safetensors) directly to the GPU; the full model is never materialized;
  - the reference hidden-state stream of the evaluation traces goes through the BF16 layer;
  - for each method the layer weights are quantized (`v2:<format>`, using the method's own calibration stream, sequentially
    q/k/v -> o_proj -> MLP or experts -> down) or reconstructed from an EXL3 directory (`exl3:<dir>`), and the method's
    calibration stream and its trace stream pass through the quantized layer;
  - at the end, KLD/PPL on the final states, using the same formulas as `eval_kld.py`.
MoE expert Hessians are accumulated over routed tokens with shrinkage towards the shared layer Hessian.
No full Fisher, DP or polish: uniform v2 and EXL3 formats are compared (EXL3 has no polish either).

    python -m src.streaming_quant \\
        --model_dir models/Qwen3-8B --data_dir data/qwen3 --methods v2:LIQK3_V2 exl3:exl3_out/qwen3_8b_3.44
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

import src.liqk_v2 as lv2
from src.eval_kld import get_scored_positions_and_targets, load_eval_trace


class TensorStore:
    """Access to checkpoint tensors by name without loading the whole model."""

    def __init__(self, model_dir: Path):
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
        else:
            weight_map = {}
            for f in sorted(model_dir.glob("*.safetensors")):
                with safe_open(str(f), framework="pt") as h:
                    weight_map.update({k: f.name for k in h.keys()})
        self.files = {k: model_dir / v for k, v in weight_map.items()}
        self._handles = {}

    def has(self, name: str) -> bool:
        return name in self.files

    def keys(self):
        return self.files.keys()

    def _handle(self, name: str):
        path = self.files[name]
        if path not in self._handles:
            self._handles[path] = safe_open(str(path), framework="pt", device="cpu")
        return self._handles[path]

    def get(self, name: str, device) -> torch.Tensor:
        return self._handle(name).get_tensor(name).to(device, non_blocking=True)

    def shape(self, name: str) -> list:
        return self._handle(name).get_slice(name).get_shape()


def load_module_weights(mod: nn.Module, prefix: str, store: TensorStore, device):
    """Materializes module parameters from the checkpoint; fused experts are assembled from per-expert tensors."""
    mod.to_empty(device=device)
    for name, p in mod.named_parameters():
        key = f"{prefix}.{name}"
        if store.has(key):
            p.data.copy_(store.get(key, device))
        elif name.endswith("experts.gate_up_proj"):
            base = key[: -len("gate_up_proj")]
            p.data.copy_(torch.stack([
                torch.cat([store.get(f"{base}{e}.gate_proj.weight", device), store.get(f"{base}{e}.up_proj.weight", device)])
                for e in range(p.shape[0])
            ]))
        elif name.endswith("experts.down_proj"):
            base = key[: -len("down_proj")]
            p.data.copy_(torch.stack([store.get(f"{base}{e}.down_proj.weight", device) for e in range(p.shape[0])]))
        else:
            raise KeyError(f"{key} not found in checkpoint")


def build_skeleton(model_dir: Path, device):
    config = AutoConfig.from_pretrained(str(model_dir))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    store = TensorStore(model_dir)
    load_module_weights(model.model.embed_tokens, "model.embed_tokens", store, device)
    load_module_weights(model.model.norm, "model.norm", store, device)
    if getattr(config, "tie_word_embeddings", False):
        model.lm_head.to_empty(device=device)
        model.lm_head.weight.data.copy_(model.model.embed_tokens.weight.data)
    else:
        load_module_weights(model.lm_head, "lm_head", store, device)
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config, device=device)
    return model, store, config


# ------------------------------------------------------------------------------------------------
# Layer quantization with a v2 format
# ------------------------------------------------------------------------------------------------

def _seed(l_idx: int, name: str) -> int:
    return 1000 + l_idx * 1009 + (sum(map(ord, name)) * 31) % 997


FISHER: dict | None = None  # {(layer, matrix): row weights} from `--fisher` (src/fisher_offload.py)
_CUR_LAYER = [0]


def _row_w(full: str, device):
    if FISHER is None:
        return None
    return FISHER[(_CUR_LAYER[0], full)].to(device)


def _quantize_linear_weight(W: torch.Tensor, H: torch.Tensor, spec, seed: int, damp: float, full: str | None = None) -> torch.Tensor:
    rw = _row_w(full, W.device) if full else None
    _, w_hat, _ = lv2.quantize_matrix_liqk_v2(W, H, spec, seed=seed, damp=damp, row_weights=rw, cd_sweeps=2)
    return w_hat.to(W.dtype)


def _cov_over(batches, fn) -> torch.Tensor:
    acc = lv2._CovAccumulator()
    for hb in batches:
        acc.add(fn(hb))
    return acc.get()


def _attn(layer, hb, pos_emb):
    return layer.self_attn(hidden_states=layer.input_layernorm(hb), position_embeddings=pos_emb, attention_mask=None)[0]


@torch.no_grad()
def quantize_layer_v2(layer, cal: list, pos_emb, spec, l_idx: int, damp: float = 0.02, shrink: float = 256.0,
                      attn: str | None = None) -> dict:
    """Quantizes the layer in place, updates the calibration stream `cal` and returns statistics.
    `attn` is a separate format for q/k/v/o (defaults to `spec`). `spec` may be a function
    of the matrix name ("self_attn.q_proj", ...) -> format or row slices [(r0, r1, format)] (DP assignment)."""
    attn_spec = attn or spec  # the name attn is used below for the attention module
    if callable(spec):
        sp = spec
    else:
        sp = lambda full: attn_spec if full.startswith("self_attn.") else spec
    spec_mlp = spec
    stats = {}
    attn = layer.self_attn

    # q/k/v: shared input input_layernorm
    H = _cov_over(cal, layer.input_layernorm)
    for name in ("q_proj", "k_proj", "v_proj"):
        mod = getattr(attn, name)
        mod.weight.data.copy_(_quantize_linear_weight(mod.weight.data, H, sp("self_attn." + name), _seed(l_idx, name), damp,
                                                      "self_attn." + name))
    del H

    # o_proj: input is captured by a hook inside self_attn
    acc = lv2._CovAccumulator()
    h = attn.o_proj.register_forward_pre_hook(lambda m, args: acc.add(args[0]))
    for hb in cal:
        _attn(layer, hb, pos_emb)
    h.remove()
    H = acc.get()
    del acc  # the fp64 K x K accumulator is freed before quantization
    attn.o_proj.weight.data.copy_(_quantize_linear_weight(attn.o_proj.weight.data, H, sp("self_attn.o_proj"), _seed(l_idx, "o_proj"), damp,
                                                          "self_attn.o_proj"))
    del H
    for i in range(len(cal)):
        cal[i] = cal[i] + _attn(layer, cal[i], pos_emb)

    spec = spec_mlp
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj"):
        H = _cov_over(cal, layer.post_attention_layernorm)
        for name in ("gate_proj", "up_proj"):
            mod = getattr(mlp, name)
            mod.weight.data.copy_(_quantize_linear_weight(mod.weight.data, H, sp("mlp." + name), _seed(l_idx, name), damp,
                                                          "mlp." + name))
        del H
        acc = lv2._CovAccumulator()
        h = mlp.down_proj.register_forward_pre_hook(lambda m, args: acc.add(args[0]))
        for hb in cal:
            mlp(layer.post_attention_layernorm(hb))
        h.remove()
        H = acc.get()
        del acc
        mlp.down_proj.weight.data.copy_(
            _quantize_linear_weight(mlp.down_proj.weight.data, H, sp("mlp.down_proj"), _seed(l_idx, "down_proj"), damp * 1.5,
                                    "mlp.down_proj")
        )
        del H
    else:
        stats.update(quantize_moe_experts_v2(mlp, [layer.post_attention_layernorm(hb) for hb in cal], spec, l_idx, damp, shrink))

    for i in range(len(cal)):
        cal[i] = cal[i] + mlp(layer.post_attention_layernorm(cal[i]))
    return stats


@torch.no_grad()
def _moe_experts_pass(mlp, xs: list, shrink: float, on_gate_up, on_down) -> dict:
    """
    Hessians of the MoE block experts: for expert e, (S_e + shrink * H_layer) / (n_e + shrink), where S_e is the sum of x x^T over
    tokens routed to it (cold experts fall back on the shared layer Hessian). `on_gate_up(e, H)` is called
    for all experts before the down_proj Hessians are collected (if it replaces gate_up_proj with the quantized one, down sees the quantized
    input), then `on_down(e, H)`.
    """
    experts = mlp.experts
    E, H_dim, I_dim = experts.num_experts, experts.hidden_dim, experts.intermediate_dim
    dev = experts.gate_up_proj.device
    routes = []
    S = torch.zeros(E, H_dim, H_dim, dtype=torch.float32, device=dev)
    n = torch.zeros(E, dtype=torch.float64)
    acc_all = lv2._CovAccumulator()
    for x in xs:
        x2 = x.reshape(-1, H_dim)
        _, _, sel = mlp.gate(x2)
        routes.append(sel)
        acc_all.add(x2)
        for e in torch.unique(sel).tolist():
            xe = x2[(sel == e).any(dim=-1)]
            S[e] += torch.mm(xe.T, xe, out_dtype=torch.float32)
            n[e] += xe.shape[0]
    H_layer = acc_all.get()
    for e in range(E):
        on_gate_up(e, (S[e] + shrink * H_layer) / (float(n[e]) + shrink))
    del S

    S_d = torch.zeros(E, I_dim, I_dim, dtype=torch.float32, device=dev)
    n_d = torch.zeros(E, dtype=torch.float64)
    acc_mid = lv2._CovAccumulator()
    for x, sel in zip(xs, routes):
        x2 = x.reshape(-1, H_dim)
        for e in torch.unique(sel).tolist():
            xe = x2[(sel == e).any(dim=-1)]
            gate, up = F.linear(xe, experts.gate_up_proj[e]).chunk(2, dim=-1)
            mid = experts.act_fn(gate) * up
            S_d[e] += torch.mm(mid.T, mid, out_dtype=torch.float32)
            n_d[e] += mid.shape[0]
            acc_mid.add(mid)
    H_mid = acc_mid.get()
    for e in range(E):
        on_down(e, (S_d[e] + shrink * H_mid) / (float(n_d[e]) + shrink))
    return {"tokens_per_expert_min": float(n.min()), "tokens_per_expert_median": float(n.median()), "tokens_per_expert_max": float(n.max())}


def quantize_moe_experts_v2(mlp, xs: list, spec, l_idx: int, damp: float, shrink: float) -> dict:
    """Quantizes the experts of an MoE block. `spec` is a format or a function of the matrix name (`mlp.experts.{e}.gate_up_proj` /
    `.down_proj`) -> format or row slices (DP assignment)."""
    experts = mlp.experts
    sp = spec if callable(spec) else (lambda full: spec)

    def gate_up(e, He):
        full = f"mlp.experts.{e}.gate_up_proj"
        experts.gate_up_proj.data[e] = _quantize_linear_weight(experts.gate_up_proj.data[e], He, sp(full), _seed(l_idx, f"gu{e}"),
                                                               damp, full)

    def down(e, He):
        full = f"mlp.experts.{e}.down_proj"
        experts.down_proj.data[e] = _quantize_linear_weight(experts.down_proj.data[e], He, sp(full), _seed(l_idx, f"dn{e}"),
                                                            damp * 1.5, full)

    return _moe_experts_pass(mlp, xs, shrink, gate_up, down)


def _weight_of(layer, full: str) -> torch.Tensor:
    """Layer matrix by name, including MoE experts (`mlp.experts.{e}.gate_up_proj` / `.down_proj`)."""
    if full.startswith("mlp.experts."):
        _, _, e, name = full.split(".")
        return getattr(layer.mlp.experts, name)[int(e)]
    return layer.get_submodule(full).weight


# ------------------------------------------------------------------------------------------------
# Layer weights from EXL3
# ------------------------------------------------------------------------------------------------

class Exl3Store:
    def __init__(self, exl3_dir: Path, device):
        self.store = TensorStore(exl3_dir)
        self.device = device

    def reconstruct(self, prefix: str, M: int, K: int) -> torch.Tensor:
        from exllamav3.ext import exllamav3_ext as ext
        from exllamav3.modules.quant.exl3 import LinearEXL3

        def get(k):
            return self.store.get(f"{prefix}.{k}", self.device) if self.store.has(f"{prefix}.{k}") else None

        lin = LinearEXL3(
            config=None, in_features=K, out_features=M, scale=None, su=get("su"), sv=get("sv"), suh=get("suh"),
            svh=get("svh"), trellis=get("trellis"), mcg=get("mcg"), mul1=get("mul1"), out_dtype=torch.float16, key=prefix,
        )
        w_t = torch.empty((K, M), dtype=torch.float16, device=self.device)
        ext.reconstruct_had_slice(w_t, lin.trellis, lin.suh, lin.svh, lin.K, lin.mcg, lin.mul1, 0)
        return w_t.T.contiguous()

    def trellis_bits(self, prefix: str) -> int:
        return math.prod(self.store.shape(f"{prefix}.trellis")) * 16


@torch.no_grad()
def load_layer_exl3(layer, l_idx: int, ex: Exl3Store) -> tuple[int, int]:
    """Replaces linear layer weights with those reconstructed from EXL3; returns (bits, params)."""
    bits = params = 0
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and not name.endswith("mlp.gate"):
            prefix = f"model.layers.{l_idx}.{name}"
            M, K = mod.weight.shape
            mod.weight.data.copy_(ex.reconstruct(prefix, M, K).to(mod.weight.dtype))
            bits += ex.trellis_bits(prefix) + M + K
            params += M * K
    experts = getattr(layer.mlp, "experts", None)
    if experts is not None and hasattr(experts, "gate_up_proj"):
        I_dim, H_dim = experts.intermediate_dim, experts.hidden_dim
        for e in range(experts.num_experts):
            base = f"model.layers.{l_idx}.mlp.experts.{e}"
            g = ex.reconstruct(f"{base}.gate_proj", I_dim, H_dim)
            u = ex.reconstruct(f"{base}.up_proj", I_dim, H_dim)
            experts.gate_up_proj.data[e] = torch.cat([g, u]).to(experts.gate_up_proj.dtype)
            experts.down_proj.data[e] = ex.reconstruct(f"{base}.down_proj", H_dim, I_dim).to(experts.down_proj.dtype)
            for sub, (M, K) in (("gate_proj", (I_dim, H_dim)), ("up_proj", (I_dim, H_dim)), ("down_proj", (H_dim, I_dim))):
                bits += ex.trellis_bits(f"{base}.{sub}") + M + K
                params += M * K
    return bits, params


# ------------------------------------------------------------------------------------------------
# Metrics on final states (as in evaluate_model_on_cached_trace)
# ------------------------------------------------------------------------------------------------

@torch.no_grad()
def trace_metrics(model, ref_final: list, q_final: list, targets: list, device, n_boot=2000, seed=42,
                  logits_fn=None, chunk: int = 1024) -> dict:
    """`logits_fn(h)` gives logits from the final hidden state (default: norm + lm_head); positions are processed in chunks."""
    if logits_fn is None:
        logits_fn = lambda h: model.lm_head(model.model.norm(h))
    seq_kld, seq_cnt, seq_nll_q, all_kld = [], [], [], []
    top1 = tot = 0
    nll_ref_sum = nll_q_sum = 0.0
    for hr_all, hq_all, tg_all in zip(ref_final, q_final, targets):
        s_kld = s_nq = 0.0
        for c0 in range(0, len(tg_all), chunk):
            hr, hq, tg = hr_all[c0 : c0 + chunk], hq_all[c0 : c0 + chunk], tg_all[c0 : c0 + chunk]
            lr = logits_fn(hr.to(device)).float()
            lq = logits_fn(hq.to(device)).float()
            tgt = torch.tensor(tg, device=device)
            lpr, lpq = F.log_softmax(lr, -1), F.log_softmax(lq, -1)
            kld = (lpr.exp() * (lpr - lpq)).sum(-1).clamp(min=0)
            s_kld += float(kld.sum())
            all_kld.append(kld.cpu().numpy())
            top1 += int((lr.argmax(-1) == lq.argmax(-1)).sum())
            nr = -lpr.gather(-1, tgt[:, None]).squeeze(-1)
            nq = -lpq.gather(-1, tgt[:, None]).squeeze(-1)
            nll_ref_sum += float(nr.sum())
            nll_q_sum += float(nq.sum())
            s_nq += float(nq.sum())
            del lr, lq, lpr, lpq
        seq_kld.append(s_kld)
        seq_cnt.append(len(tg_all))
        tot += len(tg_all)
        seq_nll_q.append(s_nq)
    s, c, nq = np.array(seq_kld), np.array(seq_cnt, dtype=np.float64), np.array(seq_nll_q)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    boot_ppl = np.exp(nq[idx].sum(1) / c[idx].sum(1))
    flat = np.concatenate(all_kld)
    return {
        "mean_kld": float(s.sum() / c.sum()), "ci95_lo": float(np.percentile(boot, 2.5)), "ci95_hi": float(np.percentile(boot, 97.5)),
        "median_kld": float(np.median(flat)), "p99_kld": float(np.percentile(flat, 99)), "top1_match": top1 / tot,
        "ppl_ref": math.exp(nll_ref_sum / tot), "ppl_quant": math.exp(nll_q_sum / tot),
        "ppl_ci95_lo": float(np.percentile(boot_ppl, 2.5)), "ppl_ci95_hi": float(np.percentile(boot_ppl, 97.5)),
        "n_tokens": tot, "per_seq_kld_sums": seq_kld, "per_seq_tok_counts": seq_cnt, "per_seq_nll_q_sums": seq_nll_q,
    }


@torch.no_grad()
def profile_matrix(slices: list, l_idx: int, full: str, W: torch.Tensor, H: torch.Tensor, seed: int, step: int):
    """Matrix row slices for the DP profile: suh rotation, diagonal LIQK3_V2 -> Fisher-weighted (if given) slice error
    under H. Appends one entry to `slices` per slice of `step` rows."""
    s_row = FISHER[(l_idx, full)].to(W.device).float() if FISHER is not None else None
    W_rot, H_rot, _ = lv2.compute_optimal_suh(W.float(), H.float(), seed=seed, row_weights=s_row)
    diag_h = torch.diagonal(H_rot).clamp(min=1e-8)
    diff = W_rot - lv2.quantize_liqk_v2_diag_rot(W_rot, diag_h, "LIQK3_V2")
    row_err = ((diff @ H_rot) * diff).sum(dim=1).clamp(min=1e-12)
    M, K = W.shape
    if s_row is not None:
        row_err = row_err * s_row
    for i, v in enumerate((0.5 * row_err).view(M // step, step).sum(dim=1).tolist()):
        slices.append({"layer": l_idx, "subname": full, "r0": i * step, "r1": (i + 1) * step, "params": step * K, "raw_kld3": v})


def finalize_profile(slices: list) -> dict:
    """Slice format costs via `SPEC_DISTORTION_RATIO`, with winsorization of the per-weight sensitivity to
    [0.2; 4.5] x mean. The median-based variant ([0.1; 1e4] x median, `LIQK_WINSOR=median`) separates slices much more strongly,
    but is worse in KLD (Gemma 4 E4B 3.0: +38 %, Qwen3-30B 3.0 on the multilingual trace: +55 %): sensitivity measured on calibration
    does not transfer to other data, and the smoothing of the original winsorization acts as regularization."""
    kpp = np.array([x["raw_kld3"] / x["params"] for x in slices])
    if os.environ.get("LIQK_WINSOR", "mean") == "mean":
        lo, hi = 0.20 * float(kpp.mean()), 4.5 * float(kpp.mean())
    else:
        lo, hi = 0.1 * float(np.median(kpp)), 1e4 * float(np.median(kpp))
    for x in slices:
        base = min(max(x["raw_kld3"] / x["params"], lo), hi) * x["params"]
        x["costs"] = {sp: base * r for sp, r in lv2.SPEC_DISTORTION_RATIO.items()}
    return {"slices": slices}


@torch.no_grad()
def build_stream_profile(model, store, cal: list, pos_cal, n_layers: int, device, attn_rows: int = 256, mlp_rows: int = 512) -> dict:
    """DP profile streamed layer by layer (like `liqk_v2.build_liqk_v2_fisher_slice_profile`, without MLP permutations): covariances on
    the bf16 stream, each matrix profiled with `profile_matrix`."""
    slices = []

    def add(l_idx, full, W, H, seed, step=None):
        profile_matrix(slices, l_idx, full, W, H, seed,
                       step or min(mlp_rows if full in ("mlp.gate_proj", "mlp.up_proj") else attn_rows, W.shape[0]))

    for l_idx in range(n_layers):
        layer = model.model.layers[l_idx]
        load_module_weights(layer, f"model.layers.{l_idx}", store, device)
        at, mlp = layer.self_attn, layer.mlp
        H = _cov_over(cal, layer.input_layernorm)
        for name in ("q_proj", "k_proj", "v_proj"):
            add(l_idx, "self_attn." + name, getattr(at, name).weight.data, H, _seed(l_idx, name))
        del H
        acc = lv2._CovAccumulator()
        h = at.o_proj.register_forward_pre_hook(lambda m, args: acc.add(args[0]))
        for hb in cal:
            _attn(layer, hb, pos_cal)
        h.remove()
        add(l_idx, "self_attn.o_proj", at.o_proj.weight.data, acc.get(), _seed(l_idx, "o_proj"))
        del acc
        for i in range(len(cal)):
            cal[i] = cal[i] + _attn(layer, cal[i], pos_cal)
        ex = getattr(mlp, "experts", None)
        if ex is not None and hasattr(ex, "gate_up_proj"):  # MoE: each expert matrix is a single slice
            _moe_experts_pass(
                mlp, [layer.post_attention_layernorm(hb) for hb in cal], 256.0,
                lambda e, He: add(l_idx, f"mlp.experts.{e}.gate_up_proj", ex.gate_up_proj.data[e], He, _seed(l_idx, f"gu{e}"),
                                  step=ex.gate_up_proj.shape[1]),
                lambda e, He: add(l_idx, f"mlp.experts.{e}.down_proj", ex.down_proj.data[e], He, _seed(l_idx, f"dn{e}"),
                                  step=ex.down_proj.shape[1]))
            for i in range(len(cal)):
                cal[i] = cal[i] + mlp(layer.post_attention_layernorm(cal[i]))
            model.model.layers[l_idx] = layer.to("meta")
            torch.cuda.empty_cache()
            continue
        H = _cov_over(cal, layer.post_attention_layernorm)
        for name in ("gate_proj", "up_proj"):
            add(l_idx, "mlp." + name, getattr(mlp, name).weight.data, H, _seed(l_idx, name))
        del H
        acc = lv2._CovAccumulator()
        h = mlp.down_proj.register_forward_pre_hook(lambda m, args: acc.add(args[0]))
        for hb in cal:
            mlp(layer.post_attention_layernorm(hb))
        h.remove()
        add(l_idx, "mlp.down_proj", mlp.down_proj.weight.data, acc.get(), _seed(l_idx, "down_proj"))
        del acc
        for i in range(len(cal)):
            cal[i] = cal[i] + mlp(layer.post_attention_layernorm(cal[i]))
        model.model.layers[l_idx] = layer.to("meta")
        torch.cuda.empty_cache()
    return finalize_profile(slices)


# ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, type=Path)
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--methods", nargs="+", required=True, help="v2:<format>[:shrink=λ][:attn=<attention format>], exl3:<dir> or bf16: (sanity check, KLD = 0)")
    ap.add_argument("--cal_rows", type=int, default=64)
    ap.add_argument("--cal_batch", type=int, default=4)
    ap.add_argument("--max_layers", type=int, default=0, help="for smoke tests: only the first N layers")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fisher", type=Path, default=None, help="Fisher row weights (src/fisher_offload.py)")
    ap.add_argument("--profile_rows", type=int, default=16, help="calibration sequences for the DP profile")
    args = ap.parse_args()
    global FISHER
    if args.fisher:
        FISHER = torch.load(args.fisher)

    device = torch.device("cuda")
    t_start = time.perf_counter()
    model, store, config = build_skeleton(args.model_dir, device)
    n_layers = config.num_hidden_layers if args.max_layers <= 0 else args.max_layers

    # Traces: tokens, positions and targets; hidden-state streams live on the CPU in bf16
    traces = {}
    for ds in ("nemotron", "multilingual"):
        seqs = load_eval_trace(args.data_dir / f"{ds}_trace.json")
        items = []
        for sq in seqs:
            pos, tg = get_scored_positions_and_targets(sq)
            if pos:
                items.append((sq["token_ids"][:8192], pos, tg))
        traces[ds] = items
    with torch.no_grad():
        emb = {ds: [model.model.embed_tokens(torch.tensor([t], device=device)).cpu() for t, _, _ in items]
               for ds, items in traces.items()}

    cal_tokens = load_file(str(args.data_dir / "cal_trace.safetensors"))["input_ids"][: args.cal_rows]
    seq_len = cal_tokens.shape[1]
    with torch.no_grad():
        cal0 = [model.model.embed_tokens(cal_tokens[i : i + args.cal_batch].to(device)) for i in range(0, args.cal_rows, args.cal_batch)]
        pos_cal = model.model.rotary_emb(cal0[0][:1], torch.arange(seq_len, device=device).unsqueeze(0))

    dp_methods = [m for m in args.methods if m.startswith("dp:")]
    profile = None
    if dp_methods:  # DP profile: a separate pass over layers before quantization
        t0 = time.perf_counter()
        # as in liqk_v2.build_liqk_v2_fisher_slice_profile: 16 calibration sequences are enough for the profile
        n_prof = max(1, args.profile_rows // args.cal_batch)
        profile = build_stream_profile(model, store, [c.clone() for c in cal0[:n_prof]], pos_cal, n_layers, device)
        print(f"DP profile: {len(profile['slices'])} slices, {time.perf_counter() - t0:.0f} s", flush=True)
        model, store, config = build_skeleton(args.model_dir, device)  # layers were moved to meta, rebuild
    methods = {}
    for m in args.methods:
        kind, arg = m.split(":", 1)
        if kind == "dp":  # dp:<menu from equal_bpw.SETS>:<target bitrate>
            from src.equal_bpw import SETS
            menu, target = arg.split(":")
            assignment = lv2.solve_fisher_slice_dp(profile, target_bpw=float(target), candidate_specs=SETS[menu])
            methods[m] = {"kind": "dp", "arg": arg, "opts": {}, "assignment": assignment, "quant_sec": 0.0, "bits": 0,
                          "params": 0, "streams": {ds: [e.clone() for e in v] for ds, v in emb.items()},
                          "cal": [c.clone() for c in cal0]}
            continue
        opts = {}
        if kind == "v2" and ":" in arg:  # v2:<format>:shrink=<λ>: strength of shrinking expert Hessians towards the layer Hessian
            arg, *kv = arg.split(":")
            opts = {k: (v if k == "attn" else float(v)) for k, v in (x.split("=") for x in kv)}
        st = {"kind": kind, "arg": arg, "opts": opts, "quant_sec": 0.0, "bits": 0, "params": 0, "streams": {ds: [e.clone() for e in v] for ds, v in emb.items()}}
        if kind == "v2":
            st["cal"] = [c.clone() for c in cal0]
        elif kind == "exl3":
            st["exl3"] = Exl3Store(root_path(arg), device)
        methods[m] = st
    ref_streams = emb
    del cal0

    def run_streams(layer, streams):
        for ds, lst in streams.items():
            for i, h in enumerate(lst):
                hd = h.to(device)
                pos = model.model.rotary_emb(hd, torch.arange(hd.shape[1], device=device).unsqueeze(0))
                lst[i] = layer(hd, position_embeddings=pos).cpu()

    t_load_total = 0.0
    for l_idx in range(n_layers):
        t0 = time.perf_counter()
        layer = model.model.layers[l_idx]
        load_module_weights(layer, f"model.layers.{l_idx}", store, device)
        torch.cuda.synchronize()
        t_load = time.perf_counter() - t0
        t_load_total += t_load
        with torch.no_grad():
            run_streams(layer, ref_streams)
        line = [f"layer {l_idx:2d}: load {t_load:.1f} s"]
        _CUR_LAYER[0] = l_idx
        for name, st in methods.items():
            lq = copy.deepcopy(layer)
            t0 = time.perf_counter()
            if st["kind"] == "v2":
                # the method's calibration stream is on the GPU only while it is being quantized (Llama-8B: down_proj 14336)
                st["cal"] = [c.to(device) for c in st["cal"]]
                stats = quantize_layer_v2(lq, st["cal"], pos_cal, st["arg"], l_idx, **st["opts"])
                st["cal"] = [c.cpu() for c in st["cal"]]
                bpw_mlp = lv2.LIQK_V2_SPECS[st["arg"]].bpw
                bpw_attn = lv2.LIQK_V2_SPECS[st["opts"].get("attn", st["arg"])].bpw
                p = 0
                for n_, mod in lq.named_modules():
                    if isinstance(mod, nn.Linear) and not n_.endswith("mlp.gate"):
                        st["bits"] += mod.weight.numel() * (bpw_attn if n_.startswith("self_attn.") else bpw_mlp)
                        p += mod.weight.numel()
                ex = getattr(lq.mlp, "experts", None)
                if ex is not None and hasattr(ex, "gate_up_proj"):
                    st["bits"] += (ex.gate_up_proj.numel() + ex.down_proj.numel()) * bpw_mlp
                    p += ex.gate_up_proj.numel() + ex.down_proj.numel()
                st["params"] += p
                st.setdefault("moe_stats", []).append(stats)
            elif st["kind"] == "dp":
                st["cal"] = [c.to(device) for c in st["cal"]]
                asg = st["assignment"]
                quantize_layer_v2(lq, st["cal"], pos_cal, lambda full: asg[(l_idx, full)], l_idx)
                st["cal"] = [c.cpu() for c in st["cal"]]
                for (l_, full), slc in asg.items():
                    if l_ != l_idx:
                        continue
                    K_in = _weight_of(lq, full).shape[1]
                    for r0, r1, sp_ in slc:
                        st["bits"] += (r1 - r0) * K_in * lv2.LIQK_V2_SPECS[sp_].bpw
                        st["params"] += (r1 - r0) * K_in
            elif st["kind"] == "exl3":
                b, p = load_layer_exl3(lq, l_idx, st["exl3"])
                st["bits"] += b
                st["params"] += p
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            st["quant_sec"] += dt
            with torch.no_grad():
                run_streams(lq, st["streams"])
            del lq
            torch.cuda.empty_cache()
            line.append(f"{name} {dt:.1f} s")
        model.model.layers[l_idx] = layer.to("meta")
        print(" | ".join(line), flush=True)

    results = {"model": str(args.model_dir), "n_layers": n_layers, "load_sec_total": t_load_total, "methods": {}}
    for name, st in methods.items():
        res = {"kind": st["kind"], "arg": st["arg"], "quant_sec": st["quant_sec"], "linear_bpw": st["bits"] / max(st["params"], 1)}
        if st.get("moe_stats"):
            res["moe_stats"] = st["moe_stats"]
        for ds, items in traces.items():
            ref_f = [h[0, pos] for h, (_, pos, _) in zip(ref_streams[ds], items)]
            q_f = [h[0, pos] for h, (_, pos, _) in zip(st["streams"][ds], items)]
            res[ds] = trace_metrics(model, ref_f, q_f, [tg for _, _, tg in items], device)
        results["methods"][name] = res
        print(f"[{name}] bpw {res['linear_bpw']:.4f} | quant {st['quant_sec']:.0f} s | KLD nem {res['nemotron']['mean_kld']:.5f} "
              f"mul {res['multilingual']['mean_kld']:.5f} | PPL nem {res['nemotron']['ppl_quant']:.4f} mul {res['multilingual']['ppl_quant']:.4f}",
              flush=True)
    results["wall_sec"] = time.perf_counter() - t_start
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


def root_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else Path(__file__).resolve().parent.parent / path


if __name__ == "__main__":
    main()
