"""
Experiment B: where the quality gap between `.liqk v2` and EXL3 arises.

For each linear matrix, measures on the same BF16 Hessians (calibration 64x2048):
  - rel  = tr(dW H dW^T) / tr(W H W^T)       — relative layer proxy error;
  - fkl  = 0.5 * sum_i s_i (dW H dW^T)_ii    — the same error weighted by per-row Fisher
                                               (predicted contribution of the matrix to KLD; this is what DP minimizes).
Quants:
  - `.liqk v2` uniform LIQK2_V2 / LIQK3_V2 / LIQK4_V2 without polish — the v2 rate-distortion curve
    for each matrix;
  - `.liqk v2` Mixed DP before and after polish at the target bitrate;
  - EXL3 mul1 at several bitrates, with the actual bitrate of each tensor.
For each EXL3 matrix, the v2 curve (linear interpolation of log(rel) over bpw) gives the bitrate
at which v2 reaches the same error: `bit_gap = bpw_v2_equiv - bpw_exl3`. bit_gap > 0 means
the v2 quantizer needs more bits than EXL3 on this matrix.

    python -m src.proxy_error_comparison --model 3b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

import src.liqk_v2 as lv2
from src.quantize_model_iqk import LINEAR_SUBMODULES
from src.run_1b_exl3_experiments import load_exl3_into_hf_model
from src.run_1b_liqk_v2_experiments import restore_clean_model_weights, save_clean_model_weights

EXL3_PYTHON = os.environ.get("LUMEN_EXL3_PYTHON", sys.executable)
EXL3_CONVERT = os.environ.get("LUMEN_EXL3_CONVERT", os.path.expanduser("~/lm-inference/ablation-sigmoid/exl3/convert.py"))

PROFILES = {
    "1b": {"model_dir": "Llama-3.2-1B-Instruct", "polish_steps": 100, "lr": {3.4375: 5e-4, 4.5: 3e-4}},
    "3b": {"model_dir": "Llama-3.2-3B-Instruct", "polish_steps": 80, "lr": {3.4375: 4e-4, 4.5: 2.5e-4}},
}
UNIFORM_SPECS = ["LIQK2_V2", "LIQK3_V2", "LIQK4_V2"]
# Input of each matrix: q/k/v share an input, gate/up share an input
HESS_GROUP = {
    "self_attn.q_proj": "qkv",
    "self_attn.k_proj": "qkv",
    "self_attn.v_proj": "qkv",
    "self_attn.o_proj": "o",
    "mlp.gate_proj": "up",
    "mlp.up_proj": "up",
    "mlp.down_proj": "down",
}


@torch.no_grad()
def collect_bf16_hessians(model, cal_tokens: torch.Tensor, batch_size: int = 8) -> dict:
    """
    BF16 input Hessians of all linear matrices (on CPU, fp32). Layers are processed one at a time; each layer's Hessians
    go to CPU immediately: all fp64 accumulators of 3B at once would take ~21 GB of VRAM.
    """
    dev = next(model.parameters()).device
    H_batches = [model.model.embed_tokens(cal_tokens[i : i + batch_size].to(dev)) for i in range(0, cal_tokens.shape[0], batch_size)]
    pos_emb = model.model.rotary_emb(H_batches[0][:1], torch.arange(cal_tokens.shape[1], device=dev).unsqueeze(0))
    out = {}
    for l_idx, layer in enumerate(model.model.layers):
        accs, hooks = {}, []
        for sub, grp in (("self_attn.q_proj", "qkv"), ("self_attn.o_proj", "o"), ("mlp.gate_proj", "up"), ("mlp.down_proj", "down")):
            acc = accs[grp] = lv2._CovAccumulator()
            hooks.append(layer.get_submodule(sub).register_forward_hook(lambda m, inp, o, acc=acc: acc.add(inp[0])))
        for b_i in range(len(H_batches)):
            H_batches[b_i] = layer(H_batches[b_i], position_embeddings=pos_emb)
        for h in hooks:
            h.remove()
        for grp, acc in accs.items():
            out[(l_idx, grp)] = acc.get().cpu()
        del accs
    return out


@torch.no_grad()
def matrix_errors(model, clean: dict, hess: dict, fisher: dict, mlp_perms: dict | None) -> dict:
    """rel and fkl for each matrix of the model's current (quantized) weights."""
    dev = next(model.parameters()).device
    out = {}
    for l_idx, layer in enumerate(model.model.layers):
        inv = torch.argsort(mlp_perms[l_idx]).to(dev) if mlp_perms and l_idx in mlp_perms else None
        for sub in LINEAR_SUBMODULES:
            w_hat = layer.get_submodule(sub).weight.data.float()
            if inv is not None and sub in ("mlp.gate_proj", "mlp.up_proj"):
                w_hat = w_hat[inv]
            elif inv is not None and sub == "mlp.down_proj":
                w_hat = w_hat[:, inv]
            w = clean[(l_idx, sub)].to(dev).float()
            H = hess[(l_idx, HESS_GROUP[sub])].to(dev)
            d = w_hat - w
            row_err = ((d @ H) * d).sum(dim=1)
            base = ((w @ H) * w).sum()
            s_row = fisher[(l_idx, sub)].to(dev).float()
            out[f"{l_idx}.{sub}"] = {
                "rel": float(row_err.sum() / base),
                "fkl": float(0.5 * (s_row * row_err).sum()),
            }
    return out


def v2_matrix_bpw(model, spec_or_assignment) -> dict:
    out = {}
    for l_idx, layer in enumerate(model.model.layers):
        for sub in LINEAR_SUBMODULES:
            M, _ = layer.get_submodule(sub).weight.shape
            slices = [(0, M, spec_or_assignment)] if isinstance(spec_or_assignment, str) else spec_or_assignment[(l_idx, sub)]
            out[f"{l_idx}.{sub}"] = sum((r1 - r0) * lv2.LIQK_V2_SPECS[sp].bpw for r0, r1, sp in slices) / M
    return out


def exl3_matrix_bpw(exl3_dir: Path, model) -> dict:
    """Actual bitrate of each EXL3 tensor from safetensors headers: trellis (int16) + su/sv signs."""
    numel = {}
    for f in exl3_dir.glob("*.safetensors"):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for k, v in header.items():
            if k.endswith(".trellis"):
                numel[k[: -len(".trellis")]] = math.prod(v["shape"])
    out = {}
    for l_idx, layer in enumerate(model.model.layers):
        for sub in LINEAR_SUBMODULES:
            M, K = layer.get_submodule(sub).weight.shape
            out[f"{l_idx}.{sub}"] = (numel[f"model.layers.{l_idx}.{sub}"] * 16 + (M + K)) / (M * K)
    return out


def bit_gap(rate_points: list[tuple[float, float]], bpw_target: float, rel_target: float) -> float | None:
    """Bitrate at which v2 reaches rel_target (interpolation of log(rel) over bpw), minus bpw_target."""
    pts = sorted(rate_points)
    xs = np.array([p[0] for p in pts])
    ys = np.log(np.array([p[1] for p in pts]))
    y = math.log(rel_target)
    # log(rel) decreases as bpw grows; extrapolate along the outermost segment
    for i in range(len(xs) - 1):
        if ys[i] >= y >= ys[i + 1] or i == len(xs) - 2 or (i == 0 and y > ys[0]):
            x = xs[i] + (y - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
            return float(x - bpw_target)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PROFILES), default="3b")
    ap.add_argument("--mixed_bpw", type=float, default=3.4375)
    ap.add_argument("--exl3_bpw", type=float, nargs="*", default=[3.4375, 4.5])
    ap.add_argument("--uniform_specs", nargs="*", default=UNIFORM_SPECS, help="v2 curve for bit_gap")
    ap.add_argument("--variant_specs", nargs="*", default=[], help="formats for quantizer variants")
    ap.add_argument("--variants", nargs="*", default=[], choices=["out_rot"], help="quantizer variants")
    ap.add_argument("--skip_mixed", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    prof = PROFILES[args.model]
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "models" / prof["model_dir"]
    cal_path = root / "data" / "cal_trace.safetensors"
    out_json = root / "results" / f"proxy_error_{args.model}{args.tag}.json"
    device = torch.device("cuda")

    results = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"methods": {}}

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device).eval()
    clean = save_clean_model_weights(model)
    cal64 = load_file(str(cal_path))["input_ids"][:64]

    print("BF16 Hessians and per-row Fisher...", flush=True)
    hess = collect_bf16_hessians(model, cal64)
    fisher = lv2.compute_exact_row_fisher_sensitivities(model, cal64, n_Menu_seqs=8, seq_len=1024)
    slice_profile = lv2.build_liqk_v2_fisher_slice_profile(model, cal64, batch_size=8, fisher_out=fisher)
    mlp_perms = slice_profile["mlp_perms"]

    def record(label: str, bpw: dict, errs: dict, extra: dict | None = None):
        results["methods"][label] = {"bpw": bpw, "errors": errs, **(extra or {})}
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        tot_fkl = sum(e["fkl"] for e in errs.values())
        mean_bpw = sum(bpw.values()) / len(bpw)
        print(f"[{label:28s}] mean bpw over matrices {mean_bpw:.3f} | total fkl {tot_fkl:.5f}", flush=True)

    runs = [(f"{spec}-NoPolish", spec, {}) for spec in args.uniform_specs]
    for variant in args.variants:
        runs += [(f"{spec}-{variant}-NoPolish", spec, {variant: True}) for spec in args.variant_specs]
    for label, spec, kwargs in runs:
        if label in results["methods"]:
            continue
        restore_clean_model_weights(model, clean)
        q_info = lv2.quantize_llama_model_liqk_v2(
            model, cal64, spec, mlp_perms=mlp_perms, fisher_out=slice_profile["fisher_out"],
            batch_size=8, cd_sweeps=2, polish_steps=0, **kwargs,
        )
        record(label, v2_matrix_bpw(model, spec), matrix_errors(model, clean, hess, fisher, mlp_perms),
               {"quant_sec": q_info["quant_sec"]})

    mixed_label = f"LIQK_V2-{args.mixed_bpw:.2f}bpw"
    if not args.skip_mixed and f"{mixed_label}-Polish" not in results["methods"]:
        # Reference hidden states for polish must be computed on the BF16 weights
        restore_clean_model_weights(model, clean)
        ref_hidden = lv2.precompute_ref_final_hidden_states(model, cal64, n_seqs=64, seq_len=512, batch_size=2)
        assignment = lv2.solve_fisher_slice_dp(slice_profile, target_bpw=args.mixed_bpw)
        lv2.quantize_llama_model_liqk_v2(
            model, cal64, assignment, mlp_perms=mlp_perms, fisher_out=slice_profile["fisher_out"],
            batch_size=8, cd_sweeps=2, polish_steps=0,
        )
        bpw = v2_matrix_bpw(model, assignment)
        record(f"{mixed_label}-NoPolish", bpw, matrix_errors(model, clean, hess, fisher, mlp_perms))
        with torch.enable_grad():
            lv2.polish_model_scales_kld(
                model, cal64, ref_hidden, n_steps=prof["polish_steps"], lr=prof["lr"][args.mixed_bpw],
                batch_size=2, reg_wd=0.02, ce_weight=0.0,
            )
        record(f"{mixed_label}-Polish", bpw, matrix_errors(model, clean, hess, fisher, mlp_perms))
        del ref_hidden

    tmp_root = root / f"exl3_tmp_{args.model}"
    for bpw_target in args.exl3_bpw:
        label = f"EXL3-mul1-{bpw_target:.2f}bpw"
        if label in results["methods"]:
            continue
        work_dir, out_dir = tmp_root / f"proxy_work_{bpw_target:.3f}", tmp_root / f"proxy_out_{bpw_target:.3f}"
        for d in (work_dir, out_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        model.cpu()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        subprocess.run(
            [EXL3_PYTHON, EXL3_CONVERT, "-i", str(model_dir), "-o", str(out_dir), "-w", str(work_dir),
             "-b", str(bpw_target), "-hb", "16", "-eb", "16", "-cd", str(cal_path), "-cr", "64", "-cc", "2048",
             "-cb", "mul1"],
            check=True,
        )
        quant_sec = time.perf_counter() - t0
        model.to(device)
        restore_clean_model_weights(model, clean)
        load_exl3_into_hf_model(model, out_dir, device)
        record(label, exl3_matrix_bpw(out_dir, model), matrix_errors(model, clean, hess, fisher, None),
               {"quant_sec": quant_sec})
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)

    # ---- Analysis: bit gap along the v2 curve ----
    methods = results["methods"]
    curve_specs = [s for s in UNIFORM_SPECS if f"{s}-NoPolish" in methods]
    if len(curve_specs) < 2:
        print("bit_gap needs a curve of at least two uniform v2 formats")
        return
    curve = {
        key: [(methods[f"{s}-NoPolish"]["bpw"][key], methods[f"{s}-NoPolish"]["errors"][key]["rel"]) for s in curve_specs]
        for key in methods[f"{curve_specs[0]}-NoPolish"]["errors"]
    }
    print("\nbit_gap > 0 — method beats v2 (v2 needs more bits for the same error), bit_gap < 0 — worse than v2")
    analysis = {}
    for label, m in methods.items():
        if label in {f"{s}-NoPolish" for s in curve_specs}:
            continue
        gaps = {key: bit_gap(curve[key], m["bpw"][key], m["errors"][key]["rel"]) for key in m["errors"]}
        by_type = {}
        for key, g in gaps.items():
            by_type.setdefault(key.split(".", 1)[1], []).append(g)
        analysis[label] = {
            "bit_gap_by_matrix": gaps,
            "bit_gap_median_by_type": {t: float(np.median(v)) for t, v in by_type.items()},
            "bit_gap_median_all": float(np.median(list(gaps.values()))),
            "fkl_total": sum(e["fkl"] for e in m["errors"].values()),
        }
        print(f"\n{label}: median bit_gap {analysis[label]['bit_gap_median_all']:+.3f} bits, total fkl {analysis[label]['fkl_total']:.5f}")
        for t, g in analysis[label]["bit_gap_median_by_type"].items():
            print(f"  {t:18s} {g:+.3f}")
    results["analysis"] = analysis
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
