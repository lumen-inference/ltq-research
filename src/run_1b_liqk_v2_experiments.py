"""
Runner for the full series of `.liqk v2` experiments on Llama-3.2-1B-Instruct:
  1. Building an exact per-row Fisher information profile (960 row slices of 256/512 channels)
     and lossless permutations of the SwiGLU MLP intermediate channels (`mlp_perms`).
  2. Uniform `.liqk v2` quants:
     - `LIQK2_V2` (2.3750 bpw): before and after calibration STE scale polishing
     - `LIQK3_V2` (3.4375 bpw): before and after calibration STE scale polishing
     - `LIQK4_V2` (4.5000 bpw): before and after calibration STE scale polishing
  3. Mixed `.liqk v2` quants (Fisher-Slice Knapsack DP + half-bit adaptive formats
     `LIQK3_0_V2` / `LIQK3_5_V2` + MLP channel permutation + STE scale polishing)
     at exactly 5 target bit budgets:
     - 2.3750 bpw (IQ2_K / EXL3-2.38 budget)
     - 3.4375 bpw (IQ3_K / EXL3-3.44 budget)
     - 3.8416 bpw (Q3_K_M / EXL3-3.84 budget)
     - 4.2716 bpw (IQ4_XS / EXL3-4.27 budget)
     - 4.5000 bpw (IQ4_K / EXL3-4.50 budget)
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
    LINEAR_SUBMODULES,
    LIQK_V2_SPECS,
    build_liqk_v2_fisher_slice_profile,
    compute_exact_row_fisher_sensitivities,
    polish_model_scales_kld,
    precompute_ref_final_hidden_states,
    quantize_llama_model_liqk_v2,
    solve_fisher_slice_dp,
)


def save_clean_model_weights(model: torch.nn.Module) -> dict:
    """Saves a copy of all 112 linear matrices and all RMSNorm weights in CPU RAM."""
    saved = {}
    for l_idx, layer in enumerate(model.model.layers):
        saved[(l_idx, "input_layernorm")] = layer.input_layernorm.weight.data.cpu().clone()
        saved[(l_idx, "post_attention_layernorm")] = layer.post_attention_layernorm.weight.data.cpu().clone()
        for subname in LINEAR_SUBMODULES:
            mod = layer.get_submodule(subname)
            saved[(l_idx, subname)] = mod.weight.data.cpu().clone()
    saved[(-1, "model.norm")] = model.model.norm.weight.data.cpu().clone()
    return saved


def restore_clean_model_weights(model: torch.nn.Module, saved: dict):
    """Instantly restores the reference BF16 weights of all linear layers and RMSNorm."""
    for l_idx, layer in enumerate(model.model.layers):
        layer.input_layernorm.weight.data.copy_(
            saved[(l_idx, "input_layernorm")].to(
                device=layer.input_layernorm.weight.device, dtype=layer.input_layernorm.weight.dtype
            )
        )
        layer.post_attention_layernorm.weight.data.copy_(
            saved[(l_idx, "post_attention_layernorm")].to(
                device=layer.post_attention_layernorm.weight.device, dtype=layer.post_attention_layernorm.weight.dtype
            )
        )
        for subname in LINEAR_SUBMODULES:
            mod = layer.get_submodule(subname)
            mod.weight.data.copy_(saved[(l_idx, subname)].to(device=mod.weight.device, dtype=mod.weight.dtype))
    model.model.norm.weight.data.copy_(
        saved[(-1, "model.norm")].to(device=model.model.norm.weight.device, dtype=model.model.norm.weight.dtype)
    )


def summarize_slice_assignment(assignment: dict, model: torch.nn.Module) -> dict:
    """Returns the parameter distribution across .liqk v2 formats (as percentages and parameter counts)."""
    param_counts = {}
    total_params = 0
    for (l_idx, subname), slices in assignment.items():
        mod = model.model.layers[l_idx].get_submodule(subname)
        _, K = mod.weight.shape
        for r0, r1, sp in slices:
            n_p = (r1 - r0) * K
            param_counts[sp] = param_counts.get(sp, 0) + n_p
            total_params += n_p
    shares = {sp: round(100.0 * cnt / max(total_params, 1), 2) for sp, cnt in sorted(param_counts.items())}
    return {"param_counts": param_counts, "param_shares_pct": shares}


def main():
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "models" / "Llama-3.2-1B-Instruct"
    out_json = root / "results" / "liqk_v2_1b.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    print("1. Loading the reference BF16 Llama-3.2-1B-Instruct model onto the GPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device)
    model.eval()

    clean_weights = save_clean_model_weights(model)

    print("2. Precomputing reference BF16 logits on the Nemotron and Multilingual traces...", flush=True)
    nem_seqs = load_eval_trace(root / "data" / "nemotron_trace.json")
    mul_seqs = load_eval_trace(root / "data" / "multilingual_trace.json")
    nem_ref = precompute_ref_trace_data(model, nem_seqs, device)
    mul_ref = precompute_ref_trace_data(model, mul_seqs, device)

    print("3. Loading the calibration trace cal_trace.safetensors (64x2048 = 131072 tokens)...", flush=True)
    cal_all = load_file(str(root / "data" / "cal_trace.safetensors"))["input_ids"]
    cal_tokens = cal_all[:64]

    print("4. Caching reference BF16 hidden states and per-row Fisher information...", flush=True)
    ref_hidden_cpu = precompute_ref_final_hidden_states(
        model, cal_tokens, n_seqs=64, seq_len=512, batch_size=4
    )
    t_f0 = time.perf_counter()
    fisher_out = compute_exact_row_fisher_sensitivities(model, cal_tokens, n_Menu_seqs=8, seq_len=1024)
    fisher_sec = time.perf_counter() - t_f0
    print(f"  -> Exact per-row Fisher information computed in {fisher_sec:.2f} s", flush=True)

    print("5. Building the row-slice sensitivity profile for Fisher-Slice DP...", flush=True)
    t_dp0 = time.perf_counter()
    slice_profile = build_liqk_v2_fisher_slice_profile(
        model, cal_tokens, batch_size=8, fisher_out=fisher_out
    )
    mlp_perms = slice_profile["mlp_perms"]
    fisher_out_perm = slice_profile["fisher_out"]
    dp_profile_sec = time.perf_counter() - t_dp0
    print(f"  -> Slice profile built in {dp_profile_sec:.1f} s", flush=True)

    results = {}

    def record_and_save(label: str, q_info: dict, extra_meta: dict | None = None):
        res_nem = evaluate_model_on_cached_trace(model, nem_ref, device)
        res_mul = evaluate_model_on_cached_trace(model, mul_ref, device)
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
            f"[{label:26s}] bpw={info_clean['linear_bpw']:.4f} | time={info_clean['quant_sec']:.1f}s | "
            f"Nem KLD={res_nem['mean_kld']:.5f} PPL={res_nem['ppl_quant']:.4f} | "
            f"Mul KLD={res_mul['mean_kld']:.5f} PPL={res_mul['ppl_quant']:.4f}",
            flush=True,
        )

    # ==========================================================================
    # Part A: Uniform .liqk v2 quants (LIQK2_V2, LIQK3_V2, LIQK4_V2)
    # ==========================================================================
    uniform_configs = [
        ("LIQK2_V2", "LIQK2_V2"),
        ("LIQK3_V2", "LIQK3_V2"),
        ("LIQK4_V2", "LIQK4_V2"),
    ]

    for base_label, spec_name in uniform_configs:
        lbl_nopol = f"{base_label}-NoPolish"
        lbl_pol = f"{base_label}-Uniform"

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
                n_steps=100,
                lr=5e-4 if spec_name != "LIQK4_V2" else 3e-4,
                batch_size=2,
                reg_wd=0.02,
            )
        q_info_pol = dict(q_info)
        q_info_pol["polish_sec"] = pol_sec
        q_info_pol["quant_sec"] = q_info["quant_sec"] + pol_sec
        record_and_save(lbl_pol, q_info_pol, {"spec": spec_name, "mode": "liqk_v2_uniform_polished"})

    # ==========================================================================
    # Part B: Mixed bitrate over row slices (Fisher-Slice DP + MLP Perm)
    # ==========================================================================
    dp_budgets = [
        ("LIQK_V2-2.38bpw", 2.3750),
        ("LIQK_V2-3.44bpw", 3.4375),
        ("LIQK_V2-3.84bpw", 3.8416),
        ("LIQK_V2-4.27bpw", 4.2716),
        ("LIQK_V2-4.50bpw", 4.5000),
    ]

    for base_label, target_bpw in dp_budgets:
        lbl_nopol = f"{base_label}-NoPolish"
        lbl_pol = base_label

        restore_clean_model_weights(model, clean_weights)
        assignment = solve_fisher_slice_dp(slice_profile, target_bpw=target_bpw)
        summary = summarize_slice_assignment(assignment, model)
        print(
            f"  -> {base_label} (target={target_bpw:.4f}): parameter shares (%) = {summary['param_shares_pct']}",
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
                n_steps=100,
                lr=5e-4 if target_bpw < 4.0 else 3e-4,
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
