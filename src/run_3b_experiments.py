"""
Validation script for the key quantization algorithms on the larger Llama-3.2-3B-Instruct model (28 layers, 3.21B parameters):
  1. Conversion to BF16 .gguf and computing imatrix.dat via llama-imatrix.
  2. Stock llama.cpp quants: Q3_K_M, IQ4_XS, Q4_K_M, and PR #30718 IQ2_K, IQ3_K, IQ4_K (+ imatrix).
  3. Our algorithms:
     - LDLQ-IQ3_K (3.4375 bpw, no Hadamard, 100% .gguf-compatible)
     - LIQK-IQ3_K (3.4375 bpw, Hadamard + LDLQ + Post-LDLQ Scale Refit)
     - DP-LIQK-3.44bpw (global DP bitrate allocation within a 3.4375 bpw budget)
     - DP-LIQK-3.84bpw (global DP bitrate allocation within the Q3_K_M budget = 3.88 bpw)
     - LIQK-IQ4_K (4.5000 bpw, Hadamard + LDLQ + Post-LDLQ Scale Refit)
  4. EXL3 (mul1) at the same budgets of 3.4375 bpw and 4.5000 bpw on the same calibration trace.
"""

from __future__ import annotations

import os

from collections import Counter
import json
import shutil
import subprocess
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
from src.gguf_utils import load_gguf_into_hf_model, patch_gguf_linear_tensors
from src.quantize_model_iqk import (
    build_dp_sensitivity_profile,
    quantize_llama_model_sequential,
    solve_dp_bit_allocation,
)
from src.run_1b_exl3_experiments import load_exl3_into_hf_model
from src.run_1b_iqk_experiments import (
    restore_clean_linear_weights,
    save_clean_linear_weights,
)


def prepare_3b_gguf_baselines(root: Path, model_dir: Path, gguf_dir: Path):
    """Creates the BF16 GGUF, computes imatrix and quantizes the baseline GGUFs for 3B if they do not exist yet."""
    gguf_dir.mkdir(parents=True, exist_ok=True)
    python_bin = os.path.expanduser("~/lm-inference/ablation-sigmoid/venv/bin/python")
    conv_script = root / "llama-iqk" / "convert_hf_to_gguf.py"
    imat_bin = root / "llama-iqk" / "build" / "bin" / "llama-imatrix"
    q_bin = root / "llama-iqk" / "build" / "bin" / "llama-quantize"

    bf16_gguf = gguf_dir / "Llama-3.2-3B-Instruct-bf16.gguf"
    imat_file = gguf_dir / "imatrix.dat"
    calib_txt = root / "data" / "calib_imatrix.txt"

    if not bf16_gguf.exists():
        print("Converting Llama-3.2-3B-Instruct to BF16 GGUF...", flush=True)
        subprocess.run(
            [python_bin, str(conv_script), str(model_dir), "--outfile", str(bf16_gguf), "--outtype", "bf16"],
            check=True,
        )

    if not imat_file.exists():
        print("Computing imatrix.dat for Llama-3.2-3B-Instruct...", flush=True)
        subprocess.run(
            [
                str(imat_bin),
                "-m", str(bf16_gguf),
                "-f", str(calib_txt),
                "-o", str(imat_file),
                "-ngl", "99",
                "-c", "512",
                "-b", "512",
                "--chunks", "128",
            ],
            check=True,
        )

    for qtype in ["IQ3_K", "Q3_K_M", "IQ4_XS", "IQ4_K", "Q4_K_M"]:
        out_q = gguf_dir / f"Llama-3.2-3B-{qtype}-imatrix.gguf"
        if not out_q.exists():
            print(f"Quantizing stock 3B {qtype}...", flush=True)
            subprocess.run(
                [str(q_bin), "--imatrix", str(imat_file), str(bf16_gguf), str(out_q), qtype, "16"],
                check=True,
            )


def main():
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "models" / "Llama-3.2-3B-Instruct"
    gguf_dir = root / "gguf_3b"
    out_json = root / "results" / "all_3b_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    prepare_3b_gguf_baselines(root, model_dir, gguf_dir)

    device = torch.device("cuda")
    print("Loading the reference BF16 Llama-3.2-3B-Instruct model on GPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device)
    model.eval()
    clean_weights = save_clean_linear_weights(model)

    print("Precomputing reference BF16 3B logits on the Nemotron and Multilingual traces...", flush=True)
    nem_seqs = load_eval_trace(root / "data" / "nemotron_trace.json")
    mul_seqs = load_eval_trace(root / "data" / "multilingual_trace.json")
    nem_ref = precompute_ref_trace_data(model, nem_seqs, device)
    mul_ref = precompute_ref_trace_data(model, mul_seqs, device)

    results = {}
    if out_json.exists():
        try:
            results = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    def record(label: str, info: dict):
        res_nem = evaluate_model_on_cached_trace(model, nem_ref, device)
        res_mul = evaluate_model_on_cached_trace(model, mul_ref, device)
        info_clean = {k: v for k, v in info.items() if k != "packed_gguf"}
        results[label] = {"info": info_clean, "nemotron": res_nem, "multilingual": res_mul}
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"[3B {label:20s}] bpw={info_clean['linear_bpw']:.4f} | "
            f"Nem KLD={res_nem['mean_kld']:.5f} (top1={res_nem['top1_match']*100:.2f}%) | "
            f"Mul KLD={res_mul['mean_kld']:.5f} (top1={res_mul['top1_match']*100:.2f}%)",
            flush=True,
        )

    # 1. Evaluating stock GGUFs
    for qtype in ["IQ3_K", "Q3_K_M", "IQ4_XS", "IQ4_K", "Q4_K_M"]:
        lbl = f"GGUF-{qtype}-imatrix"
        if lbl in results:
            continue
        gpath = gguf_dir / f"Llama-3.2-3B-{qtype}-imatrix.gguf"
        info = load_gguf_into_hf_model(model, gpath, include_embeddings=False)
        record(lbl, info)

    # Delete the heavy 6.4 GB BF16 .gguf after generating the quants to save disk space
    (gguf_dir / "Llama-3.2-3B-Instruct-bf16.gguf").unlink(missing_ok=True)

    # 2. Our LDLQ / LIQK / DP-LIQK methods on 3B
    cal_tokens = load_file(str(root / "data" / "cal_trace.safetensors"))["input_ids"][:64]

    for lbl, sp, mode, exp_gguf in [
        ("LDLQ-IQ3_K", "IQ3_K", "ldlq", True),
        ("LIQK-IQ3_K", "IQ3_K", "liqk", False),
        ("LDLQ-IQ4_K", "IQ4_K", "ldlq", False),
        ("LIQK-IQ4_K", "IQ4_K", "liqk", False),
    ]:
        if lbl in results:
            continue
        restore_clean_linear_weights(model, clean_weights)
        q_info = quantize_llama_model_sequential(
            model, cal_tokens, sp, mode=mode, batch_size=8, export_gguf_blocks=exp_gguf
        )
        if exp_gguf and q_info["packed_gguf"]:
            tmpl = gguf_dir / f"Llama-3.2-3B-{sp}-imatrix.gguf"
            out_g = gguf_dir / f"Llama-3.2-3B-{lbl}.gguf"
            if tmpl.exists():
                patch_gguf_linear_tensors(
                    tmpl,
                    out_g,
                    q_info["packed_gguf"],
                    n_heads=model.config.num_attention_heads,
                    n_kv_heads=model.config.num_key_value_heads,
                )
        record(lbl, q_info)

    # DP-LIQK on 3B (at the Q3_K_M and IQ4_XS budgets)
    if "DP-LIQK-3.88bpw" not in results or "DP-LIQK-4.27bpw" not in results:
        restore_clean_linear_weights(model, clean_weights)
        layer_costs_had, layer_params = build_dp_sensitivity_profile(
            model, cal_tokens, use_hadamard=True, batch_size=8
        )
        q3_bpw = results["GGUF-Q3_K_M-imatrix"]["info"]["linear_bpw"]
        iq4xs_bpw = results["GGUF-IQ4_XS-imatrix"]["info"]["linear_bpw"]
        for lbl, t_bpw in [("DP-LIQK-3.88bpw", q3_bpw), ("DP-LIQK-4.27bpw", iq4xs_bpw)]:
            if lbl in results:
                continue
            restore_clean_linear_weights(model, clean_weights)
            assign = solve_dp_bit_allocation(layer_costs_had, layer_params, target_bpw=t_bpw)
            counts = dict(Counter(assign.values()))
            q_info = quantize_llama_model_sequential(model, cal_tokens, assign, mode="liqk", batch_size=8)
            q_info["matrix_counts"] = counts
            record(lbl, q_info)

    # 3. EXL3 (mul1) on 3B at the 3.4375 and 4.5000 bpw budgets
    python_bin = os.path.expanduser("~/lm-inference/ablation-sigmoid/venv/bin/python")
    convert_script = os.path.expanduser("~/lm-inference/ablation-sigmoid/exl3/convert.py")
    tmp_base = root / "tmp_exl3_3b"
    tmp_base.mkdir(parents=True, exist_ok=True)

    for lbl, t_bits in [("EXL3-mul1-3.44bpw", 3.4375), ("EXL3-mul1-4.50bpw", 4.5000)]:
        if lbl in results:
            continue
        w_dir = tmp_base / f"work_{lbl}"
        o_dir = tmp_base / f"out_{lbl}"
        shutil.rmtree(w_dir, ignore_errors=True)
        shutil.rmtree(o_dir, ignore_errors=True)
        cmd = [
            python_bin,
            convert_script,
            "-i", str(model_dir),
            "-w", str(w_dir),
            "-o", str(o_dir),
            "-b", str(t_bits),
            "-hb", "16",
            "-eb", "16",
            "-cd", str(root / "data" / "cal_trace.safetensors"),
            "-cr", "64",
            "-cc", "2048",
            "-cb", "mul1",
        ]
        model.cpu()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        q_sec = time.perf_counter() - t0
        model.to(device)
        restore_clean_linear_weights(model, clean_weights)
        info = load_exl3_into_hf_model(model, o_dir, device)
        torch.cuda.empty_cache()
        info["quant_sec"] = q_sec
        record(lbl, info)
        shutil.rmtree(w_dir, ignore_errors=True)
        shutil.rmtree(o_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

