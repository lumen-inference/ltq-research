"""
Runner for a series of experiments on Llama-3.2-1B-Instruct:
  1. Uniform quants IQ2_K (2.375 bpw), IQ3_K (3.4375 bpw), IQ4_K (4.500 bpw):
     - LDLQ-IQ*_K (no Hadamard, exported to valid .gguf files for llama-bench / llama-perplexity)
     - LIQK-IQ*_K (block Hadamard rotation suh/svh + LDLQ + Post-LDLQ Scale Refit)
     - GSQ-LIQK-IQ*_K (Gumbel-Softmax + Lion refinement on top of LIQK for IQ2_K and IQ3_K)
  2. Mixed bitrate via dynamic programming (DP-LDLQ and DP-LIQK)
     at exactly the budgets of stock quants:
     - 3.4375 bpw (uniform IQ3_K budget)
     - 3.8416 bpw (GGUF Q3_K_M budget)
     - 4.2716 bpw (GGUF IQ4_XS budget)
     - 4.5000 bpw (uniform IQ4_K budget)
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from src.eval_kld import (
    evaluate_model_on_cached_trace,
    load_eval_trace,
    paired_bootstrap_diff,
    precompute_ref_trace_data,
)
from src.gguf_utils import patch_gguf_linear_tensors
from src.quantize_model_iqk import (
    LINEAR_SUBMODULES,
    build_dp_sensitivity_profile,
    quantize_llama_model_sequential,
    solve_dp_bit_allocation,
)


def save_clean_linear_weights(model: torch.nn.Module) -> dict:
    """Saves a copy of all 112 transformer linear weights in CPU RAM (~1.95 GB for 1B)."""
    saved = {}
    for l_idx, layer in enumerate(model.model.layers):
        for subname in LINEAR_SUBMODULES:
            mod = layer.get_submodule(subname)
            saved[(l_idx, subname)] = mod.weight.data.cpu().clone()
    return saved


def restore_clean_linear_weights(model: torch.nn.Module, saved: dict):
    """Instantly (~0.08 s) restores the reference BF16 linear-layer weights."""
    for l_idx, layer in enumerate(model.model.layers):
        for subname in LINEAR_SUBMODULES:
            mod = layer.get_submodule(subname)
            mod.weight.data.copy_(saved[(l_idx, subname)].to(device=mod.weight.device, dtype=mod.weight.dtype))


def main():
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "models" / "Llama-3.2-1B-Instruct"
    gguf_dir = root / "gguf_1b"
    out_json = root / "results" / "iqk_liqk_1b.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    print("1. Loading the reference BF16 Llama-3.2-1B-Instruct model onto the GPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device)
    model.eval()

    clean_weights = save_clean_linear_weights(model)

    print("2. Precomputing reference BF16 logits on the Nemotron and Multilingual traces...", flush=True)
    nem_seqs = load_eval_trace(root / "data" / "nemotron_trace.json")
    mul_seqs = load_eval_trace(root / "data" / "multilingual_trace.json")
    nem_ref = precompute_ref_trace_data(model, nem_seqs, device)
    mul_ref = precompute_ref_trace_data(model, mul_seqs, device)

    print("3. Loading the calibration trace cal_trace.safetensors (64x2048 = 131072 tokens)...", flush=True)
    cal_all = load_file(str(root / "data" / "cal_trace.safetensors"))["input_ids"]
    cal_tokens = cal_all[:64]  # 64 x 2048 = 131072 tokens (EXL3 / GPTQ standard)

    results = {}
    if out_json.exists():
        try:
            results = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    def record_and_save(label: str, q_info: dict, extra_meta: dict | None = None):
        res_nem = evaluate_model_on_cached_trace(model, nem_ref, device)
        res_mul = evaluate_model_on_cached_trace(model, mul_ref, device)
        info_clean = {k: v for k, v in q_info.items() if k != "packed_gguf"}
        if extra_meta:
            info_clean.update(extra_meta)
        results[label] = {
            "info": info_clean,
            "nemotron": res_nem,
            "multilingual": res_mul,
        }
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"[{label:22s}] bpw={info_clean['linear_bpw']:.4f} | time={info_clean['quant_sec']:.1f}s | "
            f"Nem KLD={res_nem['mean_kld']:.5f} (top1={res_nem['top1_match']*100:.2f}%) | "
            f"Mul KLD={res_mul['mean_kld']:.5f} (top1={res_mul['top1_match']*100:.2f}%)",
            flush=True,
        )

    # ==========================================================================
    # Part A: Uniform quants (IQ2_K, IQ3_K, IQ4_K)
    # ==========================================================================
    uniform_configs = [
        ("LDLQ-IQ2_K", "IQ2_K", "ldlq", True),
        ("LIQK-IQ2_K", "IQ2_K", "liqk", False),
        ("GSQ-LIQK-IQ2_K", "IQ2_K", "liqk_gsq", False),
        ("LDLQ-IQ3_K", "IQ3_K", "ldlq", True),
        ("LIQK-IQ3_K", "IQ3_K", "liqk", False),
        ("GSQ-LIQK-IQ3_K", "IQ3_K", "liqk_gsq", False),
        ("LDLQ-IQ4_K", "IQ4_K", "ldlq", True),
        ("LIQK-IQ4_K", "IQ4_K", "liqk", False),
    ]

    for label, spec_name, mode, do_gguf_export in uniform_configs:
        if label in results:
            print(f"Skipping {label} (already computed)", flush=True)
            continue
        restore_clean_linear_weights(model, clean_weights)
        q_info = quantize_llama_model_sequential(
            model,
            cal_tokens,
            spec_name,
            mode=mode,
            batch_size=8,
            export_gguf_blocks=do_gguf_export,
            gsq_steps=60,
        )
        if do_gguf_export and q_info["packed_gguf"]:
            tmpl_gguf = gguf_dir / f"Llama-3.2-1B-{spec_name}-imatrix.gguf"
            out_gguf = gguf_dir / f"Llama-3.2-1B-{label}.gguf"
            if tmpl_gguf.exists():
                patch_gguf_linear_tensors(
                    tmpl_gguf,
                    out_gguf,
                    q_info["packed_gguf"],
                    n_heads=model.config.num_attention_heads,
                    n_kv_heads=model.config.num_key_value_heads,
                )
                print(f"  -> Exported native GGUF: {out_gguf.name}", flush=True)
        record_and_save(label, q_info, {"spec": spec_name, "mode": mode})

    # ==========================================================================
    # Part B: Global bit allocation via DP (Mixed-Precision)
    # ==========================================================================
    print("4. Building the layer sensitivity table for the DP allocator...", flush=True)
    restore_clean_linear_weights(model, clean_weights)
    t_dp0 = time.perf_counter()
    layer_costs_had, layer_params = build_dp_sensitivity_profile(
        model, cal_tokens, use_hadamard=True, batch_size=8
    )
    dp_profile_sec = time.perf_counter() - t_dp0
    print(f"  -> Sensitivity profile built in {dp_profile_sec:.1f} s", flush=True)

    dp_budgets = [
        ("DP-LIQK-3.44bpw", 3.4375, "liqk"),
        ("DP-LDLQ-3.44bpw", 3.4375, "ldlq"),
        ("DP-LIQK-3.84bpw", 3.8416, "liqk"),
        ("DP-LDLQ-3.84bpw", 3.8416, "ldlq"),
        ("DP-LIQK-4.27bpw", 4.2716, "liqk"),
        ("DP-LIQK-4.50bpw", 4.5000, "liqk"),
    ]

    for label, target_bpw, mode in dp_budgets:
        if label in results:
            print(f"Skipping {label} (already computed)", flush=True)
            continue
        restore_clean_linear_weights(model, clean_weights)
        assignment = solve_dp_bit_allocation(layer_costs_had, layer_params, target_bpw=target_bpw)
        counts = dict(Counter(assignment.values()))
        print(f"  -> {label} (target={target_bpw:.4f}): matrix allocation = {counts}", flush=True)
        q_info = quantize_llama_model_sequential(
            model,
            cal_tokens,
            assignment,
            mode=mode,
            batch_size=8,
            export_gguf_blocks=False,
        )
        record_and_save(
            label,
            q_info,
            {"target_bpw": target_bpw, "mode": mode, "matrix_counts": counts, "dp_profile_sec": dp_profile_sec},
        )


if __name__ == "__main__":
    main()

