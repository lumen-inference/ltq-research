"""
Layer-by-layer quantizer and evaluator for architectures with non-standard layers: Gemma 4 E4B (PLE: per-layer inputs from
a separate table, shared KV in late layers, sliding/full attention) and Qwen3.5 (Gated DeltaNet linear attention
plus full attention). Unlike `streaming_quant.py`, it knows nothing about the layer internals:
  - the arguments of layer l (positions, masks, PLE per-layer input, ...) are taken from the stock forward of the text model:
    layers are replaced with "traps"; the trap of layer l records its arguments and stops the pass;
  - linear layers inside a layer are discovered with hooks in execution order and grouped by shared input
    (q/k/v, gate/up, ...); groups are quantized in turn, each using the Hessian of its input after quantizing
    the previous groups (like q/k/v -> o_proj -> gate/up -> down in the main pipeline);
  - shared KV (Gemma 4: late layers read K/V of an earlier one) is stored on the CPU for each stream and sequence.
Methods: `v2:<format>`; `dp:<menu>:<bits>` assigns menu formats (`equal_bpw.SETS`) to DP row slices
(profile from a separate pass over layers) with Fisher row weights (`--fisher`, see `src/fisher_full.py`); `exl3q:<bits>` is the EXL3 quantizer (trellis, mul1 codebook, uniform bitrate) applied per matrix
on the same Hessians, since the stock convert.py does not support Gemma 4 PLE models; `gguf:<file>` and `exl3:<dir>` are ready-made
llama.cpp and EXL3 quants (linear layers are reconstructed to BF16, other tensors come from the BF16 checkpoint); `bf16:` is a sanity check (KLD = 0).
The bitrate `q_bpw` covers only the quantized matrices (the same set for all methods), with all format fields: for v2 / dp
plus d_row and suh (16 · (M + K)), for EXL3 suh and svh, for GGUF the bytes of the whole tensor.
Embeddings, the PLE table, per_layer_model_projection and lm_head stay BF16 for all methods; small matrices
(output < 128 rows, e.g. in_proj_a/b in Qwen3.5) also stay BF16 and are counted as 16 bits for all methods.

    python -m src.hybrid_quant \\
        --model_dir models/gemma-4-E4B-it --data_dir data/gemma4 --methods v2:LIQK3_V2 exl3q:3.5 --out results/x.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

import src.gguf_utils as gguf_utils
import src.liqk_v2 as lv2
from src.eval_kld import get_scored_positions_and_targets, load_eval_trace
import src.streaming_quant as sqm
from src.streaming_quant import TensorStore, _quantize_linear_weight, _seed, finalize_profile, profile_matrix, trace_metrics

EXL3_DEBUG_DIR = os.environ.get("LUMEN_EXL3_DEBUG_DIR", os.path.join(tempfile.gettempdir(), "exl3_debug"))
PACK_WRITERS = {}  # --pack <method>=<file>: method -> {"w": LtqWriter, "prefix": layer prefix, "bytes": total matrix bytes}
PACK_WRITER = {}  # writer of the method currently being quantized (empty = no packing)


class _Stop(Exception):
    pass


class _Catcher(nn.Module):
    """Replaces layer l: when `owner.want == l`, records the call arguments and stops the forward."""

    def __init__(self, owner, idx: int):
        super().__init__()
        self.owner, self.idx = [owner], idx

    def forward(self, *args, **kwargs):
        if self.idx == self.owner[0].want:
            self.owner[0].caught = (args, kwargs)
            raise _Stop
        return args[0]


def _own(obj):
    """Copies view tensors (a slice of per_layer_inputs would keep the whole source tensor alive)."""
    if isinstance(obj, torch.Tensor):
        return obj.clone() if obj._base is not None else obj
    if isinstance(obj, (tuple, list)):
        return type(obj)(_own(o) for o in obj)
    return obj


def _load_module(mod: nn.Module, prefix: str, store: TensorStore, device):
    mod.to_empty(device=device)
    for name, t in list(mod.named_parameters()) + list(mod.named_buffers()):
        key = f"{prefix}.{name}"
        if store.has(key):
            t.data.copy_(store.get(key, device))
        elif name.endswith("embed_scale"):
            pass  # non-persistent buffer, set below
        else:
            raise KeyError(f"{key} not found in checkpoint")
    for m in mod.modules():
        if hasattr(m, "scalar_embed_scale"):
            m.embed_scale = torch.tensor(m.scalar_embed_scale, device=device)


class TextModel:
    """Text part of the model: embeddings/head on the GPU (PLE table on the CPU), layers are loaded one at a time."""

    def __init__(self, model_dir: Path, device):
        self.device = device
        config = AutoConfig.from_pretrained(str(model_dir))
        self.config = getattr(config, "text_config", config)
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(self.config, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.model.eval()
        self.store = TensorStore(model_dir)
        self.prefix = "model.language_model" if self.store.has("model.language_model.embed_tokens.weight") else "model"
        tm = self.tm = self.model.model
        for name, mod in tm.named_children():
            if name in ("layers", "rotary_emb"):
                continue
            _load_module(mod, f"{self.prefix}.{name}", self.store, "cpu" if name == "embed_tokens_per_layer" else device)
        with torch.device(device):
            tm.rotary_emb = type(tm.rotary_emb)(config=self.config)
        if getattr(self.config, "tie_word_embeddings", False):
            self.model.lm_head.weight = tm.embed_tokens.weight
        else:
            _load_module(self.model.lm_head, "lm_head", self.store, device)
        self.softcap = getattr(self.config, "final_logit_softcapping", None)

        ple = getattr(tm, "embed_tokens_per_layer", None)
        if ple is not None:  # PLE table on the CPU: rows are gathered there, the result goes to the GPU
            ple.forward = lambda ids, _m=ple: (F.embedding(ids.cpu(), _m.weight) * _m.embed_scale.to(_m.weight.dtype)).to(device)

        self.layers = list(tm.layers)
        tm.layers = nn.ModuleList([_Catcher(self, i) for i in range(len(self.layers))])
        self.want, self.caught = -1, None

    @torch.no_grad()
    def capture(self, ids: torch.Tensor, l_idx: int):
        """(hidden_states at the layer input with a BF16 prefix, other positional arguments, keyword arguments)."""
        self.want = l_idx
        try:
            self.tm(input_ids=ids, use_cache=False)
        except _Stop:
            pass
        args, kwargs = self.caught
        self.caught = None
        kwargs = {k: _own(v) for k, v in kwargs.items()}
        return args[0], tuple(_own(a) for a in args[1:]), kwargs

    def load_layer(self, l_idx: int) -> nn.Module:
        layer = self.layers[l_idx]
        _load_module(layer, f"{self.prefix}.layers.{l_idx}", self.store, self.device)
        return layer

    def release_layer(self, l_idx: int):
        self.layers[l_idx] = self.layers[l_idx].to("meta")

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        out = self.model.lm_head(self.tm.norm(h))
        if self.softcap:
            out = torch.tanh(out.float() / self.softcap) * self.softcap
        return out


def call_layer(layer, h, ctx, state: dict):
    rest, kwargs = ctx
    if "shared_kv_states" in kwargs:
        kwargs = dict(kwargs, shared_kv_states=state)
    out = layer(h, *rest, **kwargs)
    out = out[0] if isinstance(out, tuple) else out
    for k, v in list(state.items()):  # shared KV lives on the CPU; the layer moves it to the GPU itself when reading
        if isinstance(v, tuple) and v[0].is_cuda:
            state[k] = tuple(t.cpu() for t in v)
    return out


# ------------------------------------------------------------------------------------------------
# Layer quantization: groups of linear layers with a shared input, in execution order
# ------------------------------------------------------------------------------------------------

def quantizable(name: str, mod: nn.Module) -> bool:
    return (isinstance(mod, nn.Linear) and not name.endswith(("mlp.gate", "router.proj", "shared_expert_gate"))
            and mod.out_features >= 128 and mod.out_features % 128 == 0 and mod.in_features % 256 == 0)


@torch.no_grad()
def linear_groups(layer, run_once) -> list[list[str]]:
    seen = []
    hooks = [mod.register_forward_pre_hook(lambda m, a, n=name: seen.append((n, a[0])))
             for name, mod in layer.named_modules() if quantizable(name, mod)]
    run_once()
    for h in hooks:
        h.remove()
    groups, reps = [], []
    for name, x in seen:
        for g, r in zip(groups, reps):
            if r is x:
                if name not in g:
                    g.append(name)
                break
        else:
            if not any(name in g for g in groups):
                groups.append([name])
                reps.append(x)
    return groups


def exl3_quantize_weight(W: torch.Tensor, H: torch.Tensor, K: float, seed: int) -> torch.Tensor:
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    dev = W.device
    H_data = {"H": H.to(dev, torch.float32).clone(), "count": 1, "finalized": False, "first_key": f"m{seed}", "num_total": 0,
              "device": dev}
    qa = {"seed": seed, "K": K, "devices": [dev.index or 0], "device_ratios": None, "apply_out_scales": None,
          "debug_dir": EXL3_DEBUG_DIR, "mul1": True}
    qa["K"] = int(K) if float(K).is_integer() else float(K)  # integer bitrate as int (EXL3 kernels accept only int)
    w_q, _, _ = quantize_exl3(W.T.float().contiguous(), H_data, qa, return_weight_q=True)
    return w_q.T.to(W.dtype)


@torch.no_grad()
def group_hessians(layer, cal: list, cal_states: list, cal_ctx: list, on_group) -> list[list[str]]:
    """Groups of linear layers with a shared input, in execution order; for each, the Hessian of its input over the calibration stream
    and a call to `on_group(group, H)` (if it modifies the group's weights, subsequent groups see the modified layer)."""
    mods = dict(layer.named_modules())

    def run(i):
        return call_layer(layer, cal[i], cal_ctx[i], cal_states[i])

    groups = linear_groups(layer, lambda: run(0))
    for g in groups:
        acc = lv2._CovAccumulator()

        def grab(m, a):
            acc.add(a[0])
            raise _Stop

        h = mods[g[0]].register_forward_pre_hook(grab)
        for i in range(len(cal)):
            try:
                run(i)
            except _Stop:
                pass
        h.remove()
        on_group(g, acc.get())
    return groups


def slice_step(name: str, M: int) -> int:
    """DP profile row slice: 512 for gate/up, 256 for the rest (or the largest divisor of M among 256 / 128)."""
    for st in ((512, 256, 128) if name.endswith(("gate_proj", "up_proj")) else (256, 128)):
        if st <= M and M % st == 0:
            return st
    return M


@torch.no_grad()
def quantize_layer_generic(layer, l_idx: int, cal: list, cal_states: list, cal_ctx: list, method: str, arg,
                           damp: float = 0.02) -> tuple[float, int]:
    """Quantizes the layer in place using the method's calibration stream; returns (bits, params) of its linear layers.
    `arg` is the format (v2), bitrate (exl3q) or DP assignment {(layer, matrix): row slices} (dp)."""
    mods = dict(layer.named_modules())
    sqm._CUR_LAYER[0] = l_idx

    def on_group(g, H):
        for name in g:
            mod = mods[name]
            seed = _seed(l_idx, name)
            d = damp * 1.5 if name.endswith("down_proj") else damp
            if method == "v2":
                mod.weight.data.copy_(_quantize_linear_weight(mod.weight.data, H, arg, seed, d))
            elif method == "dp":
                mod.weight.data.copy_(_quantize_linear_weight(mod.weight.data, H, arg[(l_idx, name)], seed, d, name))
                if PACK_WRITER:
                    PACK_WRITER["bytes"] += PACK_WRITER["w"].add_matrix(f"{PACK_WRITER['prefix']}.layers.{l_idx}.{name}",
                                                                        lv2.PACK["matrix"])
            else:
                mod.weight.data.copy_(exl3_quantize_weight(mod.weight.data, H, float(arg), seed))

    groups = group_hessians(layer, cal, cal_states, cal_ctx, on_group)
    done = {n for g in groups for n in g}
    bits = params = q_bits = q_params = 0
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and not name.endswith(("mlp.gate", "router.proj", "shared_expert_gate")):
            M, K = mod.weight.shape
            if name in done and method == "dp":
                b = sum((r1 - r0) * K * lv2.LIQK_V2_SPECS[sp].bpw for r0, r1, sp in arg[(l_idx, name)]) + 16 * (M + K)
            elif name in done:
                b = M * K * (lv2.LIQK_V2_SPECS[arg].bpw if method == "v2" else float(arg)) + 16 * (M + K)
            else:
                b = 16 * M * K
            if name in done:
                q_bits, q_params = q_bits + b, q_params + M * K
            bits, params = bits + b, params + M * K
    return bits, params, q_bits, q_params


# ------------------------------------------------------------------------------------------------
# Ready-made quants: GGUF (llama.cpp) and EXL3; linear layers are reconstructed to BF16
# ------------------------------------------------------------------------------------------------

GGUF_NAMES = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
              "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


def _rel_err(W: torch.Tensor, ref: torch.Tensor) -> float:
    return float((W.float() - ref.float()).norm() / ref.float().norm())


class GgufSource:
    """Matrices from GGUF. For Qwen3.5 linear attention, the llama.cpp converter reorders V heads from "grouped by K" order
    to tiled order (for ggml broadcast): the V part of in_proj_qkv and in_proj_z by rows, out_proj by columns. Here both
    layouts are compared against BF16: the correct one gives the quantization error, the wrong one about √2; the choice must be unambiguous."""

    def __init__(self, path: Path, config):
        self.data_off, self.tensors = gguf_utils.parse_gguf_tensors(path)
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.nk, self.nv = getattr(config, "linear_num_key_heads", 1), getattr(config, "linear_num_value_heads", 1)
        self.dk, self.dv = getattr(config, "linear_key_head_dim", 0), getattr(config, "linear_value_head_dim", 0)
        self.log = {}

    def _grouped(self, W: torch.Tensor, dim: int) -> torch.Tensor:
        """Tiled V head order [v_per_k, nk, d] -> grouped by K [nk, v_per_k, d] along dim."""
        n = W.shape[dim]
        idx = torch.arange(n).reshape(self.nv // self.nk, self.nk, self.dv).transpose(0, 1).reshape(n)
        return W.index_select(dim, idx.to(W.device))

    def weight(self, l_idx: int, name: str, ref: torch.Tensor) -> tuple[torch.Tensor, int]:
        key = f"blk.{l_idx}.{GGUF_NAMES[name]}.weight"
        info = self.tensors[key]
        o = self.data_off + info["offset"]
        W = gguf_utils.dequantize_gguf_tensor(self.mm[o : o + info["n_bytes"]], info["ggml_type"], tuple(ref.shape)).to(ref.device)
        cands = [W]
        if self.nv != self.nk:
            if name == "linear_attn.in_proj_qkv":
                q = 2 * self.nk * self.dk
                cands.append(torch.cat([W[:q], self._grouped(W[q:], 0)]))
            elif name == "linear_attn.in_proj_z":
                cands.append(self._grouped(W, 0))
            elif name == "linear_attn.out_proj":
                cands.append(self._grouped(W, 1))
        errs = [_rel_err(c, ref) for c in cands]
        best = min(range(len(cands)), key=errs.__getitem__)
        self.log[key] = {"ggml_type": info["ggml_type"], "rel_err": errs, "layout": ["file", "v_grouped"][best]}
        if errs[best] > 0.8 or (len(errs) > 1 and max(errs) < 1.0):
            raise RuntimeError(f"{key}: layout cannot be determined unambiguously, errors {errs}")
        return cands[best], info["n_bytes"] * 8


class Exl3Source:
    """Matrices from a ready-made EXL3 quant (trellis + suh / svh), reconstructed with the exllamav3 kernel."""

    def __init__(self, exl3_dir: Path, prefix: str, device):
        self.ex = sqm.Exl3Store(exl3_dir, device)
        self.prefix = prefix
        self.log = {}

    def weight(self, l_idx: int, name: str, ref: torch.Tensor) -> tuple[torch.Tensor, int]:
        p = f"{self.prefix}.layers.{l_idx}.{name}"
        M, K = ref.shape
        W = self.ex.reconstruct(p, M, K)
        self.log[p] = {"rel_err": [_rel_err(W, ref)]}
        if self.log[p]["rel_err"][0] > 0.8:
            raise RuntimeError(f"{p}: reconstruction error {self.log[p]['rel_err'][0]:.3f}")
        return W, self.ex.trellis_bits(p) + 16 * (M + K) + 32


@torch.no_grad()
def load_layer_external(layer, l_idx: int, src) -> tuple[float, int, float, int]:
    """Loads the ready-made quant's matrices into the layer (quantized ones only, the same set as for the other methods)."""
    bits = params = q_bits = q_params = 0
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and not name.endswith(("mlp.gate", "router.proj", "shared_expert_gate")):
            M, K = mod.weight.shape
            if quantizable(name, mod):
                W, b = src.weight(l_idx, name, mod.weight.data)
                mod.weight.data.copy_(W.to(mod.weight.dtype))
                q_bits, q_params = q_bits + b, q_params + M * K
            else:
                b = 16 * M * K
            bits, params = bits + b, params + M * K
    return bits, params, q_bits, q_params


def overhead_bpw(tmod: TextModel, n_layers: int) -> float:
    """The d_row and suh fields (16 · (M + K) per matrix) in bits per weight of the quantized matrices; they are not included in the DP format bitrates."""
    over = par = 0
    for l_idx in range(n_layers):
        for name, mod in tmod.layers[l_idx].named_modules():
            if quantizable(name, mod):
                over, par = over + 16 * (mod.out_features + mod.in_features), par + mod.out_features * mod.in_features
    return over / par


@torch.no_grad()
def build_profile(tmod: TextModel, cal_ids: list, n_layers: int) -> dict:
    """DP profile from a separate pass over layers on the BF16 calibration stream: each quantized matrix is split into row slices
    (`profile_matrix`) using the Hessian of its group."""
    slices = []
    cal = [tmod.capture(ids, 0)[0] for ids in cal_ids]
    states = [{} for _ in cal]
    for l_idx in range(n_layers):
        layer = tmod.load_layer(l_idx)
        ctx = [tmod.capture(ids, l_idx)[1:] for ids in cal_ids]
        mods = dict(layer.named_modules())

        def on_group(g, H):
            for name in g:
                W = mods[name].weight.data
                profile_matrix(slices, l_idx, name, W, H, _seed(l_idx, name), slice_step(name, W.shape[0]))

        group_hessians(layer, cal, states, ctx, on_group)
        for i in range(len(cal)):
            cal[i] = call_layer(layer, cal[i], ctx[i], states[i])
        tmod.release_layer(l_idx)
        torch.cuda.empty_cache()
    return finalize_profile(slices)


# ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, type=Path)
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--methods", nargs="+", required=True,
                    help="v2:<format>, dp:<menu>:<bits> (bits include the d_row / suh fields), exl3q:<bits>, gguf:<file>, exl3:<dir> or bf16:")
    ap.add_argument("--fisher", type=Path, default=None, help="Fisher row weights (src/fisher_full.py) for dp")
    ap.add_argument("--profile_rows", type=int, default=16, help="calibration sequences for the DP profile")
    ap.add_argument("--cal_rows", type=int, default=64)
    ap.add_argument("--cal_batch", type=int, default=4)
    ap.add_argument("--max_layers", type=int, default=0, help="for smoke tests: only the first N layers")
    ap.add_argument("--max_seqs", type=int, default=0, help="for smoke tests: only the first N dialogues of each trace")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save_assignments", type=Path, default=None, help="directory for DP assignments (for packing into .ltq)")
    ap.add_argument("--pack", action="append", default=[], metavar="METHOD=FILE",
                    help="write the quant of a dp / dpfile method to .ltq (with unquantized tensors and metadata); may be repeated")
    args = ap.parse_args()

    device = torch.device("cuda")
    t_start = time.perf_counter()
    tmod = TextModel(args.model_dir, device)
    n_layers = len(tmod.layers) if args.max_layers <= 0 else args.max_layers

    traces = {}
    for ds in ("nemotron", "multilingual"):
        items = []
        for sq in load_eval_trace(args.data_dir / f"{ds}_trace.json"):
            pos, tg = get_scored_positions_and_targets(sq)
            if pos:
                items.append((torch.tensor([sq["token_ids"][:8192]]), pos, tg))
        traces[ds] = items[: args.max_seqs] if args.max_seqs > 0 else items
    cal_tokens = load_file(str(args.data_dir / "cal_trace.safetensors"))["input_ids"][: args.cal_rows]
    cal_ids = [cal_tokens[i : i + args.cal_batch].to(device) for i in range(0, args.cal_rows, args.cal_batch)]

    with torch.no_grad():
        emb = {ds: [tmod.capture(ids.to(device), 0)[0].cpu() for ids, _, _ in items] for ds, items in traces.items()}
        cal0 = [tmod.capture(ids, 0)[0] for ids in cal_ids]

    if args.fisher:
        sqm.FISHER = torch.load(args.fisher)
    profile = None
    if args.pack:
        from src.equal_bpw import SETS
        from src.ltq_file import LtqWriter
        from src.ltq_format import load_formats
        for spec in args.pack:
            m, path = spec.split("=", 1)
            assert m in args.methods and m.startswith(("dp:", "dpfile:")), f"packing is only for dp methods from --methods: {m}"
            PACK_WRITERS[m] = {"w": LtqWriter(Path(path), list(load_formats(SETS["TCQPQ"]).values())), "prefix": tmod.prefix,
                               "bytes": 0, "path": Path(path)}
    if any(m.startswith("dp:") for m in args.methods):
        t0 = time.perf_counter()
        profile = build_profile(tmod, cal_ids[: max(1, args.profile_rows // args.cal_batch)], n_layers)
        print(f"DP profile: {len(profile['slices'])} slices, {time.perf_counter() - t0:.0f} s", flush=True)

    methods = {}
    over = overhead_bpw(tmod, n_layers)
    for m in args.methods:
        kind, arg = m.split(":", 1)
        if kind == "dp":  # dp:<menu from equal_bpw.SETS>:<target bitrate including the d_row / suh fields>
            from src.equal_bpw import SETS
            menu, target = arg.split(":")
            arg = lv2.solve_fisher_slice_dp(profile, target_bpw=float(target) - over, candidate_specs=SETS[menu])
            if args.save_assignments:
                args.save_assignments.mkdir(parents=True, exist_ok=True)
                torch.save(arg, args.save_assignments / f"{m.replace(':', '_')}.pt")
        elif kind == "dpfile":  # dpfile:<assignment from --save_assignments>
            kind, arg = "dp", torch.load(arg)
        elif kind == "gguf":
            arg = GgufSource(Path(arg), tmod.config)
        elif kind == "exl3":
            arg = Exl3Source(Path(arg), tmod.prefix, device)
        st = {"kind": kind, "arg": arg, "quant_sec": 0.0, "bits": 0.0, "params": 0, "q_bits": 0.0, "q_params": 0,
              "streams": {ds: [e.clone() for e in v] for ds, v in emb.items()},
              "states": {ds: [{} for _ in v] for ds, v in emb.items()}}
        if kind in ("v2", "dp", "exl3q"):
            st["cal"] = [c.clone() for c in cal0]
            st["cal_states"] = [{} for _ in cal0]
        methods[m] = st
    ref = {"streams": emb, "states": {ds: [{} for _ in v] for ds, v in emb.items()}}
    del cal0

    for l_idx in range(n_layers):
        t0 = time.perf_counter()
        layer = tmod.load_layer(l_idx)
        cal_ctx = [tmod.capture(ids, l_idx)[1:] for ids in cal_ids] if any("cal" in s for s in methods.values()) else []
        line = [f"layer {l_idx:2d}: load+context {time.perf_counter() - t0:.1f} s"]
        qlayers = {}
        for name, st in methods.items():
            t0 = time.perf_counter()
            if st["kind"] == "bf16":
                qlayers[name] = layer
                continue
            lq = copy.deepcopy(layer)
            if st["kind"] in ("gguf", "exl3"):
                b, p, qb, qp = load_layer_external(lq, l_idx, st["arg"])
            else:
                PACK_WRITER.clear()
                PACK_WRITER.update(PACK_WRITERS.get(name, {}))
                lv2.PACK = {} if PACK_WRITER else None
                b, p, qb, qp = quantize_layer_generic(lq, l_idx, st["cal"], st["cal_states"], cal_ctx, st["kind"], st["arg"])
                if PACK_WRITER:
                    PACK_WRITERS[name]["bytes"] = PACK_WRITER["bytes"]
                PACK_WRITER.clear()
                lv2.PACK = None
                with torch.no_grad():
                    for i in range(len(st["cal"])):
                        st["cal"][i] = call_layer(lq, st["cal"][i], cal_ctx[i], st["cal_states"][i])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            st["quant_sec"] += dt
            st["bits"] += b
            st["params"] += p
            st["q_bits"] += qb
            st["q_params"] += qp
            qlayers[name] = lq
            line.append(f"{name} {dt:.1f} s")
        del cal_ctx

        t0 = time.perf_counter()
        with torch.no_grad():
            for ds, items in traces.items():
                for i, (ids, _, _) in enumerate(items):
                    _, *ctx = tmod.capture(ids.to(device), l_idx)
                    ctx = tuple(ctx)
                    for streams, states, lay in [(ref["streams"], ref["states"], layer)] + [
                            (st["streams"], st["states"], qlayers[nm]) for nm, st in methods.items()]:
                        streams[ds][i] = call_layer(lay, streams[ds][i].to(device), ctx, states[ds][i]).cpu()
        line.append(f"traces {time.perf_counter() - t0:.1f} s")
        del qlayers
        tmod.release_layer(l_idx)
        torch.cuda.empty_cache()
        print(" | ".join(line), flush=True)

    results = {"model": str(args.model_dir), "n_layers": n_layers, "methods": {}}
    for name, st in methods.items():
        res = {"kind": st["kind"], "arg": name.split(":", 1)[1], "quant_sec": st["quant_sec"],
               "linear_bpw": st["bits"] / st["params"] if st["params"] else 16.0,
               "q_bpw": st["q_bits"] / st["q_params"] if st["q_params"] else 16.0}
        if st["kind"] in ("gguf", "exl3"):
            res["tensors"] = st["arg"].log
        if st["kind"] == "dp":
            sh = {}
            for (l, sub), slices in st["arg"].items():
                K = tmod.layers[l].get_submodule(sub).in_features
                for r0, r1, sp in slices:
                    sh[sp] = sh.get(sp, 0) + (r1 - r0) * K
            res["shares_pct"] = {k: round(100 * v / sum(sh.values()), 2) for k, v in sorted(sh.items())}
        for ds, items in traces.items():
            ref_f = [h[0, pos] for h, (_, pos, _) in zip(ref["streams"][ds], items)]
            q_f = [h[0, pos] for h, (_, pos, _) in zip(st["streams"][ds], items)]
            res[ds] = trace_metrics(tmod.model, ref_f, q_f, [tg for _, _, tg in items], device, logits_fn=tmod.logits)
        results["methods"][name] = res
        print(f"[{name}] q_bpw {res['q_bpw']:.4f} bpw {res['linear_bpw']:.4f} | quant {st['quant_sec']:.0f} s | KLD nem {res['nemotron']['mean_kld']:.5f} "
              f"mul {res['multilingual']['mean_kld']:.5f} | PPL nem {res['nemotron']['ppl_quant']:.4f} mul {res['multilingual']['ppl_quant']:.4f}",
              flush=True)
    for m_name, pw in PACK_WRITERS.items():
        from src.ltq_file import format_json, source_metadata
        w = pw["w"]
        done = {f"{p}.weight" for p in w.quantized}
        for k in sorted(tmod.store.keys()):
            if k not in done:
                w.add_tensor(k, tmod.store.get(k, "cpu"))
        res = results["methods"][m_name]
        w.finalize(dict(source_metadata(args.model_dir), quantization={
            "quant_method": "ltq", "version": 1, "formats": [format_json(f) for f in w.formats], "row_tables": {},
            "modules_not_quantized": sorted({k.rsplit(".", 1)[0] for k in tmod.store.keys() if k not in done and
                                             k.endswith(".weight") and len(tmod.store.shape(k)) == 2}),
            "bits_per_weight": round(res["q_bpw"], 4),
            "calibration": {"rows": args.cal_rows, "seq_len": 2048, "fisher": str(args.fisher), "menu": "TCQPQ",
                            "kld": {ds: res[ds]["mean_kld"] for ds in traces}}}))
        res["pack"] = {"path": str(pw["path"]), "file_bytes": pw["path"].stat().st_size, "quantized_matrices": len(w.quantized),
                       "matrix_bytes": pw["bytes"]}
        print(f"wrote {pw['path']}: {pw['path'].stat().st_size / 1e9:.2f} GB, {len(w.quantized)} matrices", flush=True)
    results["wall_sec"] = time.perf_counter() - t_start
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
