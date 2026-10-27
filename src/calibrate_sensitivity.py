"""
Sensitivity calibration for Fisher-Slice DP from the actual KLD increase.

Currently DP estimates the cost of a slice as `0.5 * sum_i s_i * row_err_i * ratio(format)` (diagonal Fisher,
per-layer error on the BF16 Hessian, independent summation). Experiment B showed that for uniform quants
the actual KLD is consistently ~0.16 x the prediction, but for the DP bit allocation it is 0.38 x: the relative slice weights are wrong.

The script:
  1. Quantizes each linear matrix to LIQK3_V2 separately (the rest stay BF16) and measures the actual
     KLD increase on held-out calibration rows (not on the evaluation traces).
  2. Compares with the DP formula's prediction for the same matrix -> a per-matrix correction factor c_g.
  3. Checks additivity: sum of increases vs. KLD of the fully quantized model (LIQK3_V2).
  4. Solves DP with calibrated costs (the measured matrix sensitivity is distributed over its slices
     in proportion to the prediction) and compares the full quant (original settings, polish without CE) with the uncalibrated one
     in a single session on the evaluation traces.

    python -m src.calibrate_sensitivity
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

import src.liqk_v2 as lv2
from src.eval_kld import (
    evaluate_model_on_cached_trace,
    load_eval_trace,
    paired_bootstrap_diff,
    paired_bootstrap_ppl_diff,
    precompute_ref_trace_data,
)
from src.proxy_error_comparison import HESS_GROUP, collect_bf16_hessians
from src.quantize_model_iqk import LINEAR_SUBMODULES
from src.run_1b_liqk_v2_experiments import restore_clean_model_weights, save_clean_model_weights

PROFILES = {
    "1b": {"model_dir": "Llama-3.2-1B-Instruct", "polish_steps": 100, "lr": 5e-4},
    "3b": {"model_dir": "Llama-3.2-3B-Instruct", "polish_steps": 80, "lr": 4e-4},
}


@torch.no_grad()
def precompute_sens_ref(model, tokens: torch.Tensor) -> list:
    """Reference logits of the held-out calibration rows (bf16, on CPU: 4.2 GB for 8 rows of 3B do not fit alongside quantization)."""
    return [
        model(tokens[i : i + 1].to(model.device), use_cache=False).logits[0].to(torch.bfloat16).cpu()
        for i in range(tokens.shape[0])
    ]


@torch.no_grad()
def sens_kld(model, tokens: torch.Tensor, ref: list, chunk: int = 512) -> float:
    """Token-averaged KL(P_BF16 || P_q) on the held-out rows."""
    tot, n = 0.0, 0
    for i in range(tokens.shape[0]):
        h = model.model(tokens[i : i + 1].to(model.device), use_cache=False).last_hidden_state[0]
        ref_i = ref[i].to(model.device, non_blocking=True)
        for c0 in range(0, h.shape[0], chunk):
            lq = F.log_softmax(model.lm_head(h[c0 : c0 + chunk]).float(), dim=-1)
            lr = F.log_softmax(ref_i[c0 : c0 + chunk].float(), dim=-1)
            tot += float((lr.exp() * (lr - lq)).sum(dim=-1).clamp(min=0).sum())
            n += lq.shape[0]
    return tot / n


@torch.no_grad()
def quantize_single_matrix(model, l_idx: int, sub: str, clean: dict, hess: dict, slice_profile: dict, spec: str):
    """Quantizes a single matrix the same way as the main loop (with MLP permutation) and writes W_hat into the model."""
    dev = model.device
    w = clean[(l_idx, sub)].to(dev)
    H = hess[(l_idx, HESS_GROUP[sub])].to(dev)
    perm = slice_profile["mlp_perms"][l_idx].to(dev)
    inv = torch.argsort(perm)
    if sub in ("mlp.gate_proj", "mlp.up_proj"):
        w = w[perm]
    elif sub == "mlp.down_proj":
        w = w[:, perm]
        H = H[perm][:, perm]
    M = w.shape[0]
    idx_s = LINEAR_SUBMODULES.index(sub)
    _, w_hat, _ = lv2.quantize_matrix_liqk_v2(
        w, H, [(0, M, spec)], seed=1000 + l_idx * 10 + idx_s, damp=0.02 * (1.5 if sub == "mlp.down_proj" else 1.0),
        row_weights=slice_profile["fisher_out"][(l_idx, sub)].to(dev), cd_sweeps=2,
    )
    if sub in ("mlp.gate_proj", "mlp.up_proj"):
        w_hat = w_hat[inv]
    elif sub == "mlp.down_proj":
        w_hat = w_hat[:, inv]
    mod = model.model.layers[l_idx].get_submodule(sub)
    mod.weight.data.copy_(w_hat.to(mod.weight.dtype))


def calibrated_profile(slice_profile: dict, real: dict) -> dict:
    """Slice cost = measured KLD increase of its matrix x the slice's share of that matrix's prediction."""
    prof = copy.copy(slice_profile)
    prof["slices"] = [dict(s) for s in slice_profile["slices"]]
    raw_sum: dict = {}
    for s in prof["slices"]:
        key = f"{s['layer']}.{s['subname']}"
        raw_sum[key] = raw_sum.get(key, 0.0) + s["raw_kld3"]
    floor = 1e-3 * float(np.median([v for v in real.values() if v > 0]))
    for s in prof["slices"]:
        key = f"{s['layer']}.{s['subname']}"
        base = max(real[key], floor) * s["raw_kld3"] / raw_sum[key]
        s["costs"] = {sp: base * ratio for sp, ratio in lv2.SPEC_DISTORTION_RATIO.items()}
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PROFILES), default="3b")
    ap.add_argument("--target", type=float, default=3.4375)
    ap.add_argument("--sens_rows", type=int, default=8)
    args = ap.parse_args()

    prof = PROFILES[args.model]
    root = Path(__file__).resolve().parent.parent
    out_json = root / "results" / f"sensitivity_{args.model}.json"
    device = torch.device("cuda")
    results = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}

    model = AutoModelForCausalLM.from_pretrained(str(root / "models" / prof["model_dir"]), dtype=torch.bfloat16).to(device).eval()
    clean = save_clean_model_weights(model)
    cal_all = load_file(str(root / "data" / "cal_trace.safetensors"))["input_ids"]
    cal64 = cal_all[:64]
    sens_tokens = cal_all[64 : 64 + args.sens_rows]

    nem_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "nemotron_trace.json"), device)
    mul_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "multilingual_trace.json"), device)
    ref_hidden = lv2.precompute_ref_final_hidden_states(model, cal64, n_seqs=64, seq_len=512, batch_size=2)
    hess = collect_bf16_hessians(model, cal64)
    fisher = lv2.compute_exact_row_fisher_sensitivities(model, cal64, n_Menu_seqs=8, seq_len=1024)
    slice_profile = lv2.build_liqk_v2_fisher_slice_profile(model, cal64, batch_size=8, fisher_out=fisher)
    sens_ref = precompute_sens_ref(model, sens_tokens)

    proxy_raw, proxy_wins = {}, {}
    for s in slice_profile["slices"]:
        key = f"{s['layer']}.{s['subname']}"
        proxy_raw[key] = proxy_raw.get(key, 0.0) + s["raw_kld3"]
        proxy_wins[key] = proxy_wins.get(key, 0.0) + s["costs"]["LIQK3_V2"]

    # ---- 1. Actual KLD increase from quantizing each matrix separately ----
    real = results.get("real_single", {})
    for l_idx in range(model.config.num_hidden_layers):
        for sub in LINEAR_SUBMODULES:
            key = f"{l_idx}.{sub}"
            if key in real:
                continue
            quantize_single_matrix(model, l_idx, sub, clean, hess, slice_profile, "LIQK3_V2")
            real[key] = sens_kld(model, sens_tokens, sens_ref)
            mod = model.model.layers[l_idx].get_submodule(sub)
            mod.weight.data.copy_(clean[(l_idx, sub)].to(device=device, dtype=mod.weight.dtype))
        results["real_single"] = real
        results["proxy_raw"] = proxy_raw
        results["proxy_winsorized"] = proxy_wins
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"layer {l_idx}: total KLD increase over 7 matrices {sum(real[f'{l_idx}.{s}'] for s in LINEAR_SUBMODULES):.5f}", flush=True)

    keys = list(real)
    r = np.array([real[k] for k in keys])
    p = np.array([proxy_raw[k] for k in keys])
    pw = np.array([proxy_wins[k] for k in keys])

    def spearman(a, b):
        return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

    c = r / p
    by_type = {sub: c[[k.endswith(sub) for k in keys]] for sub in LINEAR_SUBMODULES}
    summary = {
        "spearman_real_vs_proxy_raw": spearman(r, p),
        "spearman_real_vs_proxy_winsorized": spearman(r, pw),
        "c_ratio_p5_p50_p95": [float(np.percentile(c / np.median(c), q)) for q in (5, 50, 95)],
        "c_median_by_type_rel": {sub: float(np.median(v) / np.median(c)) for sub, v in by_type.items()},
    }
    print("\nSpearman correlation of actual increase with prediction: raw %.3f, winsorized %.3f"
          % (summary["spearman_real_vs_proxy_raw"], summary["spearman_real_vs_proxy_winsorized"]))
    print("Correction c_g relative to the median, p5/p50/p95:", ["%.2f" % v for v in summary["c_ratio_p5_p50_p95"]])
    print("Median correction by type (relative to the overall median):",
          {k.split(".")[-1]: round(v, 2) for k, v in summary["c_median_by_type_rel"].items()})

    # ---- 2. Additivity ----
    if "full_uniform_kld" not in results:
        restore_clean_model_weights(model, clean)
        lv2.quantize_llama_model_liqk_v2(
            model, cal64, "LIQK3_V2", mlp_perms=slice_profile["mlp_perms"], fisher_out=slice_profile["fisher_out"],
            batch_size=8, cd_sweeps=2, polish_steps=0,
        )
        results["full_uniform_kld"] = sens_kld(model, sens_tokens, sens_ref)
    summary["sum_single"] = float(r.sum())
    summary["full_uniform"] = results["full_uniform_kld"]
    print(f"Additivity: sum of 196 individual increases {r.sum():.5f}, fully quantized model {results['full_uniform_kld']:.5f}")

    # ---- 3. DP with calibrated costs vs. the original ----
    del sens_ref
    torch.cuda.empty_cache()
    variants = {
        "uncalibrated": slice_profile,
        "calibrated": calibrated_profile(slice_profile, real),
    }
    evals = {}
    for name, prof_v in variants.items():
        assignment = lv2.solve_fisher_slice_dp(prof_v, target_bpw=args.target)
        restore_clean_model_weights(model, clean)
        q_info = lv2.quantize_llama_model_liqk_v2(
            model, cal64, assignment, mlp_perms=slice_profile["mlp_perms"], fisher_out=slice_profile["fisher_out"],
            batch_size=8, cd_sweeps=2, polish_steps=0,
        )
        with torch.enable_grad():
            lv2.polish_model_scales_kld(
                model, cal64, ref_hidden, n_steps=prof["polish_steps"], lr=prof["lr"], batch_size=2, reg_wd=0.02, ce_weight=0.0
            )
        evals[name] = {
            "linear_bpw": q_info["linear_bpw"],
            "nemotron": evaluate_model_on_cached_trace(model, nem_ref, device),
            "multilingual": evaluate_model_on_cached_trace(model, mul_ref, device),
        }
        print(f"[{name:12s}] bpw {q_info['linear_bpw']:.4f} | KLD nem {evals[name]['nemotron']['mean_kld']:.5f} "
              f"mul {evals[name]['multilingual']['mean_kld']:.5f} | PPL nem {evals[name]['nemotron']['ppl_quant']:.4f} "
              f"mul {evals[name]['multilingual']['ppl_quant']:.4f}", flush=True)
    for ds in ("nemotron", "multilingual"):
        dk = paired_bootstrap_diff(evals["calibrated"][ds], evals["uncalibrated"][ds])
        dp = paired_bootstrap_ppl_diff(evals["calibrated"][ds], evals["uncalibrated"][ds])
        summary[f"calibrated_vs_uncalibrated_{ds}"] = {"kld": dk, "ppl": dp}
        print(f"{ds:12s} calib. - uncalib.: ΔKLD {dk['mean_diff']:+.5f} [{dk['ci95_lo']:+.5f}, {dk['ci95_hi']:+.5f}] "
              f"({100 * dk['mean_diff'] / evals['uncalibrated'][ds]['mean_kld']:+.1f}%), "
              f"ΔPPL {dp['mean_diff']:+.4f} [{dp['ci95_lo']:+.4f}, {dp['ci95_hi']:+.4f}]")

    results["summary"] = summary
    results["dp_eval"] = evals
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
