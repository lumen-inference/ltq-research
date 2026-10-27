"""
Validation script for the `.liqk v2` algorithm on Llama-3.2-3B-Instruct (28 layers, 3.21B parameters):
  1. Uniform `.liqk v2` quants:
     - `LIQK3_V2-Uniform` (3.4375 bpw)
     - `LIQK4_V2-Uniform` (4.5000 bpw)
  2. Mixed `.liqk v2` quants (Fisher-Slice Knapsack DP + half-bit adaptive formats +
     lossless MLP channel permutation + STE scale polishing) at the budgets of stock quants and EXL3:
     - `LIQK_V2-3.44bpw` (3.4375 bpw, IQ3_K / EXL3-3.44 budget)
     - `LIQK_V2-3.85bpw` (3.8471 bpw, Q3_K_M / DP-LIQK-3.88bpw budget)
     - `LIQK_V2-4.25bpw` (4.2500 bpw, IQ4_XS / DP-LIQK-4.27bpw budget)
     - `LIQK_V2-4.50bpw` (4.5000 bpw, IQ4_K / EXL3-4.50 budget)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from src.eval_kld import (
    evaluate_model_on_cached_trace,
    load_eval_trace,
    precompute_ref_trace_data,
)
from src.liqk_v2 import (
    build_liqk_v2_fisher_slice_profile,
    compute_exact_row_fisher_sensitivities,
    polish_model_scales_kld,
    precompute_ref_final_hidden_states,
    quantize_llama_model_liqk_v2,
    solve_fisher_slice_dp,
)
from src.run_1b_liqk_v2_experiments import (
    restore_clean_model_weights,
    save_clean_model_weights,
    summarize_slice_assignment,
)


def main():
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "models" / "Llama-3.2-3B-Instruct"
    out_json = root / "results" / "liqk_v2_3b.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    print("1. Loading the reference BF16 Llama-3.2-3B-Instruct model on GPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device)
    model.eval()

    clean_weights = save_clean_model_weights(model)

    print("2. Precomputing reference BF16 3B logits on the Nemotron and Multilingual traces...", flush=True)
    nem_seqs = load_eval_trace(root / "data" / "nemotron_trace.json")
    mul_seqs = load_eval_trace(root / "data" / "multilingual_trace.json")
    nem_ref = precompute_ref_trace_data(model, nem_seqs, device)
    mul_ref = precompute_ref_trace_data(model, mul_seqs, device)

    print("3. Loading calibration trace cal_trace.safetensors (64x2048 = 131072 tokens)...", flush=True)
    cal_tokens = load_file(str(root / "data" / "cal_trace.safetensors"))["input_ids"][:64]

    print("4. Caching reference BF16 3B hidden states and per-row Fisher information...", flush=True)
    ref_hidden_cpu = precompute_ref_final_hidden_states(
        model, cal_tokens, n_seqs=64, seq_len=512, batch_size=2
    )
    t_f0 = time.perf_counter()
    fisher_out = compute_exact_row_fisher_sensitivities(model, cal_tokens, n_Menu_seqs=8, seq_len=1024)
    fisher_sec = time.perf_counter() - t_f0
    print(f"  -> Exact per-row Fisher information (3B) computed in {fisher_sec:.2f} s", flush=True)

    print("5. Building the row-slice sensitivity profile for 3B Fisher-Slice DP...", flush=True)
    t_dp0 = time.perf_counter()
    slice_profile = build_liqk_v2_fisher_slice_profile(
        model, cal_tokens, batch_size=8, fisher_out=fisher_out
    )
    mlp_perms = slice_profile["mlp_perms"]
    fisher_out_perm = slice_profile["fisher_out"]
    dp_profile_sec = time.perf_counter() - t_dp0
    print(f"  -> 3B slice profile built in {dp_profile_sec:.1f} s", flush=True)

    results = {}
    if out_json.exists():
        try:
            results = json.loads(out_json.read_text(encoding="utf-8"))
            print(f"  -> Loaded existing 3B results: {list(results.keys())}", flush=True)
        except Exception:
            results = {}

    def record_and_save(label: str, q_info: dict, extra_meta: dict | None = None):
        torch.cuda.empty_cache()
        res_nem = evaluate_model_on_cached_trace(model, nem_ref, device)
        res_mul = evaluate_model_on_cached_trace(model, mul_ref, device)
        torch.cuda.empty_cache()
        info_clean = dict(q_info)
        if extra_meta:
            info_clean.update(extra_meta)
        results[label] = {
            "info": info_clean,
            "nemotron": res_nem,
            "multilingual": res_mul,
        }
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"[3B {label:24s}] bpw={info_clean['linear_bpw']:.4f} | time={info_clean['quant_sec']:.1f}s | "
            f"Nem KLD={res_nem['mean_kld']:.5f} PPL={res_nem['ppl_quant']:.4f} | "
            f"Mul KLD={res_mul['mean_kld']:.5f} PPL={res_mul['ppl_quant']:.4f}",
            flush=True,
        )

    # 1. Uniform LIQK3_V2 and LIQK4_V2 quants on 3B
    for base_label, spec_name in [("LIQK3_V2", "LIQK3_V2"), ("LIQK4_V2", "LIQK4_V2")]:
        lbl_nopol = f"{base_label}-NoPolish"
        lbl_pol = f"{base_label}-Uniform"
        if lbl_nopol in results and lbl_pol in results:
            print(f"  -> Skipping already computed {lbl_nopol} and {lbl_pol}", flush=True)
            continue

        restore_clean_model_weights(model, clean_weights)
        q_info = quantize_llama_model_liqk_v2(
            model,
            cal_tokens,
            spec_name,
            mlp_perms=mlp_perms,
            fisher_out=fisher_out_perm,
            batch_size=8,
            cd_sweeps=2,
            polish_steps=0,
        )
        record_and_save(lbl_nopol, q_info, {"spec": spec_name, "mode": "liqk_v2_uniform_nopolish"})

        with torch.enable_grad():
            pol_sec = polish_model_scales_kld(
                model,
                cal_tokens,
                ref_hidden_cpu,
                n_steps=80,
                lr=4e-4 if spec_name == "LIQK3_V2" else 2.5e-4,
                batch_size=2,
                reg_wd=0.02,
            )
        q_info_pol = dict(q_info)
        q_info_pol["polish_sec"] = pol_sec
        q_info_pol["quant_sec"] = q_info["quant_sec"] + pol_sec
        record_and_save(lbl_pol, q_info_pol, {"spec": spec_name, "mode": "liqk_v2_uniform_polished"})

    # 2. Fisher-Slice DP mixed bitrate on 3B
    dp_budgets = [
        ("LIQK_V2-3.44bpw", 3.4375),
        ("LIQK_V2-3.85bpw", 3.8471),
        ("LIQK_V2-4.25bpw", 4.2500),
        ("LIQK_V2-4.50bpw", 4.5000),
    ]

    for base_label, target_bpw in dp_budgets:
        lbl_nopol = f"{base_label}-NoPolish"
        lbl_pol = base_label

        restore_clean_model_weights(model, clean_weights)
        assignment = solve_fisher_slice_dp(slice_profile, target_bpw=target_bpw)
        summary = summarize_slice_assignment(assignment, model)
        print(
            f"  -> 3B {base_label} (target={target_bpw:.4f}): parameter shares (%) = {summary['param_shares_pct']}",
            flush=True,
        )

        q_info = quantize_llama_model_liqk_v2(
            model,
            cal_tokens,
            assignment,
            mlp_perms=mlp_perms,
            fisher_out=fisher_out_perm,
            batch_size=8,
            cd_sweeps=2,
            polish_steps=0,
        )
        record_and_save(
            lbl_nopol,
            q_info,
            {
                "target_bpw": target_bpw,
                "mode": "liqk_v2_mixed_nopolish",
                "param_shares_pct": summary["param_shares_pct"],
                "dp_profile_sec": dp_profile_sec,
            },
        )

        with torch.enable_grad():
            pol_sec = polish_model_scales_kld(
                model,
                cal_tokens,
                ref_hidden_cpu,
                n_steps=80,
                lr=4e-4 if target_bpw < 4.0 else 2.5e-4,
                batch_size=2,
                reg_wd=0.02,
            )
        q_info_pol = dict(q_info)
        q_info_pol["polish_sec"] = pol_sec
        q_info_pol["quant_sec"] = q_info["quant_sec"] + pol_sec
        record_and_save(
            lbl_pol,
            q_info_pol,
            {
                "target_bpw": target_bpw,
                "mode": "liqk_v2_mixed_polished",
                "param_shares_pct": summary["param_shares_pct"],
                "dp_profile_sec": dp_profile_sec,
            },
        )


if __name__ == "__main__":
    main()

