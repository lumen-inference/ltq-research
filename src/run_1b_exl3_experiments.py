"""
Conversion and fair evaluation of EXL3 quants (mul1 and mcg codebooks)
on Llama-3.2-1B-Instruct at exactly the same bitrates (bpw) and on exactly the same
calibration set cal_trace.safetensors (64x2048 = 131072 tokens) as our
IQ2_K..IQ6_K / LDLQ / .liqk experiments.

For 100% mathematically identical conditions:
  - Input embeddings and the output head (embed_tokens / lm_head) are kept at 16 bits (--head_bits 16 --embed_bits 16).
  - Quantized LinearEXL3 matrices are reconstructed into the original basis (via the built-in kernel
    reconstruct_had_slice / get_weight_tensor) and substituted into the same reference
    LlamaForCausalLM graph to compute Teacher-Forced KLD on the Nemotron and Multilingual traces.
"""

from __future__ import annotations

import os

import glob
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from src.eval_kld import (
    evaluate_model_on_cached_trace,
    load_eval_trace,
    precompute_ref_trace_data,
)
from src.quantize_model_iqk import LINEAR_SUBMODULES
from src.run_1b_iqk_experiments import (
    restore_clean_linear_weights,
    save_clean_linear_weights,
)


def load_exl3_into_hf_model(
    model: torch.nn.Module,
    exl3_dir: str | Path,
    device: torch.device,
) -> dict:
    """
    Reads compiled LinearEXL3 tensors from the exl3_dir folder,
    reconstructs the effective weights W_hat in the original basis and copies them into model.
    Returns the exact size of the linear tensors (trellis + suh + svh) and the average bitrate bpw.
    """
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3 import LinearEXL3

    st_files = sorted(glob.glob(str(Path(exl3_dir) / "*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {exl3_dir}")

    # Index which .safetensors file holds each tensor
    tensor_to_file: Dict[str, str] = {}
    for stf in st_files:
        with safe_open(stf, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensor_to_file[k] = stf

    def get_t(k: str) -> torch.Tensor | None:
        if k not in tensor_to_file:
            return None
        with safe_open(tensor_to_file[k], framework="pt", device=str(device)) as f:
            return f.get_tensor(k)

    n_layers = model.config.num_hidden_layers
    total_params = 0
    total_bytes = 0
    bitrates_counts: Dict[str, int] = {}

    for l_idx in range(n_layers):
        layer = model.model.layers[l_idx]
        for subname in LINEAR_SUBMODULES:
            prefix = f"model.layers.{l_idx}.{subname}"
            trellis = get_t(f"{prefix}.trellis")
            if trellis is None:
                raise KeyError(f"Tensor {prefix}.trellis not found in {exl3_dir}")
            suh = get_t(f"{prefix}.suh")
            svh = get_t(f"{prefix}.svh")
            su = get_t(f"{prefix}.su")
            sv = get_t(f"{prefix}.sv")
            mcg = get_t(f"{prefix}.mcg")
            mul1 = get_t(f"{prefix}.mul1")

            mod = layer.get_submodule(subname)
            M, K = mod.weight.shape

            lin_exl3 = LinearEXL3(
                config=None,
                in_features=K,
                out_features=M,
                scale=None,
                su=su,
                sv=sv,
                suh=suh,
                svh=svh,
                trellis=trellis,
                mcg=mcg,
                mul1=mul1,
                out_dtype=torch.float16,
                key=prefix,
            )

            # Reconstruct the weight matrix in the original basis via the native CUDA kernel reconstruct_had_slice
            w_recon_t = torch.empty((K, M), dtype=torch.float16, device=device)
            ext.reconstruct_had_slice(
                w_recon_t,
                lin_exl3.trellis,
                lin_exl3.suh,
                lin_exl3.svh,
                lin_exl3.K,
                lin_exl3.mcg,
                lin_exl3.mul1,
                0,
            )
            # On the first layer, check that the reconstructed matrix gives exactly the same output as LinearEXL3.forward
            if l_idx == 0 and subname == "self_attn.q_proj":
                x_test = torch.randn((4, K), dtype=torch.float16, device=device)
                y_ref = lin_exl3.forward(x_test, {"reconstruct": False})
                y_rec = x_test @ w_recon_t
                rel_err = (y_ref - y_rec).float().norm() / y_ref.float().norm().clamp(min=1e-6)
                if rel_err > 5e-3:
                    raise RuntimeError(f"EXL3 reconstruction verification failed: rel_err={rel_err:.6f}")

            mod.weight.data.copy_(w_recon_t.T.to(dtype=mod.weight.dtype, device=mod.weight.device))

            n_p = M * K
            # Count actual EXL3 storage bytes: trellis (int16) + packed su/sv signs (1 bit per channel = (K+M)/8 bytes)
            b_trellis = trellis.numel() * 2
            b_signs = (K + M) // 8
            total_params += n_p
            total_bytes += b_trellis + b_signs
            k_str = f"{lin_exl3.K}b"
            bitrates_counts[k_str] = bitrates_counts.get(k_str, 0) + 1

    return {
        "linear_params": total_params,
        "linear_bytes": total_bytes,
        "linear_bpw": (total_bytes * 8.0) / max(total_params, 1),
        "bitrates_counts": bitrates_counts,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "models" / "Llama-3.2-1B-Instruct"
    cal_path = root / "data" / "cal_trace.safetensors"
    out_json = root / "results" / "exl3_1b.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    python_bin = os.path.expanduser("~/lm-inference/ablation-sigmoid/venv/bin/python")
    convert_script = os.path.expanduser("~/lm-inference/ablation-sigmoid/exl3/convert.py")

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

    results = {}
    if out_json.exists():
        try:
            results = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    # Free VRAM during conversion in the external subprocess if needed (though 1B takes only 2.5 GB of 16 GB)
    configs = [
        ("EXL3-mul1-2.38bpw", 2.375, "mul1"),
        ("EXL3-mcg-2.38bpw", 2.375, "mcg"),
        ("EXL3-mul1-3.44bpw", 3.4375, "mul1"),
        ("EXL3-mcg-3.44bpw", 3.4375, "mcg"),
        ("EXL3-mul1-3.84bpw", 3.8416, "mul1"),
        ("EXL3-mul1-4.27bpw", 4.2716, "mul1"),
        ("EXL3-mul1-4.50bpw", 4.5000, "mul1"),
        ("EXL3-mcg-4.50bpw", 4.5000, "mcg"),
    ]

    tmp_base = root / "tmp_exl3_1b"
    tmp_base.mkdir(parents=True, exist_ok=True)

    for label, target_bits, cb_name in configs:
        if label in results:
            print(f"Skipping {label} (already computed)", flush=True)
            continue

        work_dir = tmp_base / f"work_{label}"
        out_dir = tmp_base / f"out_{label}"
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)

        cmd = [
            python_bin,
            convert_script,
            "-i", str(model_dir),
            "-w", str(work_dir),
            "-o", str(out_dir),
            "-b", str(target_bits),
            "-hb", "16",
            "-eb", "16",
            "-cd", str(cal_path),
            "-cr", "64",
            "-cc", "2048",
            "-cb", cb_name,
        ]
        print(f"--- Converting {label} (bits={target_bits}, codebook={cb_name}) ---", flush=True)
        model.cpu()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        quant_sec = time.perf_counter() - t0
        model.to(device)

        restore_clean_linear_weights(model, clean_weights)
        info = load_exl3_into_hf_model(model, out_dir, device)
        info["quant_sec"] = quant_sec
        info["target_bits"] = target_bits
        info["codebook"] = cb_name

        res_nem = evaluate_model_on_cached_trace(model, nem_ref, device)
        res_mul = evaluate_model_on_cached_trace(model, mul_ref, device)

        results[label] = {
            "info": info,
            "nemotron": res_nem,
            "multilingual": res_mul,
        }
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"[{label:22s}] bpw={info['linear_bpw']:.4f} | time={quant_sec:.1f}s | "
            f"Nem KLD={res_nem['mean_kld']:.5f} (top1={res_nem['top1_match']*100:.2f}%) | "
            f"Mul KLD={res_mul['mean_kld']:.5f} (top1={res_mul['top1_match']*100:.2f}%)",
            flush=True,
        )

        # Delete temporary checkpoints after evaluation (keep only EXL3-mul1-3.44bpw and EXL3-mul1-4.50bpw for speed benchmarks!)
        shutil.rmtree(work_dir, ignore_errors=True)
        if label not in ("EXL3-mul1-3.44bpw", "EXL3-mul1-4.50bpw"):
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

