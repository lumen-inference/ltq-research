"""
Reproducibility audit of the `.liqk v2` results on Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct.

Checks three things the README conclusions depend on:
  1. Determinism: a repeated run with the `run_extended_bpw_experiments.py` settings
     (cal 32 rows, polish 80 steps, ce_weight=0) must match `results/liqk_v2_{1b,3b}.json`.
  2. Contribution of the polish CE term: settings of the original `run_{1b,3b}_liqk_v2_experiments.py`
     (cal 64 rows, polish 100/80 steps) with ce_weight=0.08 and without it. CE trains the model on
     Qwen-generated calibration texts, and the evaluation traces are also written by Qwen,
     so CE can lower trace PPL without bringing the model closer to BF16.
  3. Neutral domain: KLD and PPL on WikiText-2 (human text unrelated to either the calibration
     or the traces) for the same quants and for EXL3, reconverted right here.

All PPLs come with bootstrap CIs over dialogues (traces) or 2048-token chunks (WikiText-2).

    python -m src.audit_reproducibility \\
        --model 3b --wiki data/wikitext2_test.raw
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval_kld import (
    evaluate_model_on_cached_trace,
    load_eval_trace,
    paired_bootstrap_diff,
    paired_bootstrap_ppl_diff,
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
from src.run_1b_exl3_experiments import load_exl3_into_hf_model
from src.run_1b_liqk_v2_experiments import restore_clean_model_weights, save_clean_model_weights

EXL3_PYTHON = os.environ.get("LUMEN_EXL3_PYTHON", sys.executable)
EXL3_CONVERT = os.environ.get("LUMEN_EXL3_CONVERT", os.path.expanduser("~/lm-inference/ablation-sigmoid/exl3/convert.py"))

# targets: (label, target_bpw, lr from run_extended_bpw_experiments.py, lr from the original run_*_liqk_v2_experiments.py)
PROFILES = {
    "1b": {
        "model_dir": "Llama-3.2-1B-Instruct",
        "orig_polish_steps": 100,
        "hidden_batch": 4,
        "targets": [("LIQK_V2-3.44bpw", 3.4375, 3e-4, 5e-4), ("LIQK_V2-4.50bpw", 4.5000, 2e-4, 3e-4)],
    },
    "3b": {
        "model_dir": "Llama-3.2-3B-Instruct",
        "orig_polish_steps": 80,
        "hidden_batch": 2,
        "targets": [("LIQK_V2-3.44bpw", 3.4375, 3e-4, 4e-4), ("LIQK_V2-4.50bpw", 4.5000, 2e-4, 2.5e-4)],
    },
}
EXL3_TARGETS = [("EXL3-mul1-3.44bpw", 3.4375), ("EXL3-mul1-4.50bpw", 4.5000)]


def load_wiki_chunks(tok, path: Path, chunk_len: int, n_chunks: int) -> torch.Tensor:
    ids = tok.encode(path.read_text(encoding="utf-8"), add_special_tokens=False)
    n_avail = len(ids) // chunk_len
    n = min(n_avail, n_chunks)
    bos = tok.bos_token_id
    rows = [[bos] + ids[i * chunk_len : (i + 1) * chunk_len - 1] for i in range(n)]
    print(f"WikiText-2: {len(ids)} tokens, using {n} chunks of {chunk_len}", flush=True)
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def precompute_ref_wiki_logits(model, chunks: torch.Tensor, device) -> list:
    """Reference logits of the second half of each chunk in bf16 on CPU (16.8 GB for 64 chunks)."""
    half = chunks.shape[1] // 2
    return [
        model(input_ids=x.unsqueeze(0).to(device), use_cache=False).logits[0, half - 1 : -1].to(torch.bfloat16).cpu()
        for x in chunks
    ]


@torch.no_grad()
def eval_wiki(model, ref_logits: list, chunks: torch.Tensor, device, n_boot: int = 2000, seed: int = 42) -> dict:
    """KLD(P_ref || P_q) and PPL on the second half of each chunk (as in llama-perplexity)."""
    half = chunks.shape[1] // 2
    kld_sums, nll_q_sums, nll_ref_sums = [], [], []
    for x, lr in zip(chunks, ref_logits):
        x = x.unsqueeze(0).to(device)
        tgt = x[0, half:]
        lq = model(input_ids=x, use_cache=False).logits[0, half - 1 : -1].float()
        lr = lr.to(device=device, dtype=torch.float32)
        lpq = F.log_softmax(lq, dim=-1)
        lpr = F.log_softmax(lr, dim=-1)
        kld_sums.append(float((lpr.exp() * (lpr - lpq)).sum(dim=-1).clamp(min=0).sum()))
        nll_q_sums.append(float(-lpq.gather(-1, tgt[:, None]).sum()))
        nll_ref_sums.append(float(-lpr.gather(-1, tgt[:, None]).sum()))
        del lq, lr, lpq, lpr
    n_tok = (chunks.shape[1] - half) * len(chunks)
    kld = np.array(kld_sums)
    nll = np.array(nll_q_sums)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kld), size=(n_boot, len(kld)))
    per = chunks.shape[1] - half
    boot_kld = kld[idx].sum(axis=1) / (per * len(kld))
    boot_ppl = np.exp(nll[idx].sum(axis=1) / (per * len(kld)))
    return {
        "mean_kld": float(kld.sum() / n_tok),
        "ci95_lo": float(np.percentile(boot_kld, 2.5)),
        "ci95_hi": float(np.percentile(boot_kld, 97.5)),
        "ppl_ref": math.exp(sum(nll_ref_sums) / n_tok),
        "ppl_quant": math.exp(nll.sum() / n_tok),
        "ppl_ci95_lo": float(np.percentile(boot_ppl, 2.5)),
        "ppl_ci95_hi": float(np.percentile(boot_ppl, 97.5)),
        "n_tokens": n_tok,
        "per_seq_kld_sums": kld_sums,
        "per_seq_nll_q_sums": nll_q_sums,
        "per_seq_tok_counts": [per] * len(kld),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PROFILES), default="1b")
    ap.add_argument("--wiki", required=True, type=Path)
    ap.add_argument("--wiki_chunks", type=int, default=64)
    ap.add_argument("--skip_exl3", action="store_true")
    args = ap.parse_args()

    prof = PROFILES[args.model]
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "models" / prof["model_dir"]
    cal_path = root / "data" / "cal_trace.safetensors"
    out_json = root / "results" / f"audit_{args.model}.json"
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device).eval()
    clean_weights = save_clean_model_weights(model)

    nem_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "nemotron_trace.json"), device)
    mul_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "multilingual_trace.json"), device)
    wiki = load_wiki_chunks(tok, args.wiki, 2048, args.wiki_chunks)
    wiki_ref = precompute_ref_wiki_logits(model, wiki, device)

    cal64 = load_file(str(cal_path))["input_ids"][:64]
    cal32 = cal64[:32]

    results = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}

    def record(label: str, info: dict):
        torch.cuda.empty_cache()
        res = {
            "info": info,
            "nemotron": evaluate_model_on_cached_trace(model, nem_ref, device),
            "multilingual": evaluate_model_on_cached_trace(model, mul_ref, device),
            "wikitext2": eval_wiki(model, wiki_ref, wiki, device),
        }
        results[label] = res
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        n, m, w = res["nemotron"], res["multilingual"], res["wikitext2"]
        print(
            f"[{label:34s}] KLD nem={n['mean_kld']:.5f} mul={m['mean_kld']:.5f} wiki={w['mean_kld']:.5f} | "
            f"PPL nem={n['ppl_quant']:.4f} mul={m['ppl_quant']:.4f} wiki={w['ppl_quant']:.4f}",
            flush=True,
        )

    if "BF16" not in results:
        record("BF16", {"linear_bpw": 16.0})

    restore_clean_model_weights(model, clean_weights)
    ref_hidden_cpu = precompute_ref_final_hidden_states(
        model, cal64, n_seqs=64, seq_len=512, batch_size=prof["hidden_batch"]
    )
    t0 = time.perf_counter()
    fisher_out = compute_exact_row_fisher_sensitivities(model, cal64, n_Menu_seqs=8, seq_len=1024)
    slice_profile = build_liqk_v2_fisher_slice_profile(model, cal64, batch_size=8, fisher_out=fisher_out)
    profile_sec = time.perf_counter() - t0
    mlp_perms = slice_profile["mlp_perms"]
    fisher_out_perm = slice_profile["fisher_out"]

    # (suffix, calibration for quantization, polish steps, which lr, ce_weight)
    configs = [
        ("ext", cal32, 80, "ext", 0.0),
        ("ext-rerun", cal32, 80, "ext", 0.0),
        ("orig-ce0.08", cal64, prof["orig_polish_steps"], "orig", 0.08),
        ("orig-ce0", cal64, prof["orig_polish_steps"], "orig", 0.0),
    ]
    for base, target_bpw, lr_ext, lr_orig in prof["targets"]:
        assignment = solve_fisher_slice_dp(slice_profile, target_bpw=target_bpw)
        for suffix, cal_q, n_steps, lr_kind, ce in configs:
            label = f"{base}-{suffix}"
            if label in results:
                continue
            restore_clean_model_weights(model, clean_weights)
            q_info = quantize_llama_model_liqk_v2(
                model, cal_q, assignment, mlp_perms=mlp_perms, fisher_out=fisher_out_perm,
                batch_size=8, cd_sweeps=2, polish_steps=0,
            )
            if suffix == "ext":
                record(f"{base}-NoPolish", dict(q_info, profile_sec=profile_sec))
            lr = lr_ext if lr_kind == "ext" else lr_orig
            with torch.enable_grad():
                pol_sec = polish_model_scales_kld(
                    model, cal64, ref_hidden_cpu, n_steps=n_steps, lr=lr, batch_size=2, reg_wd=0.02, ce_weight=ce
                )
            record(label, dict(q_info, polish_sec=pol_sec, profile_sec=profile_sec, cal_rows=int(cal_q.shape[0]),
                               polish_steps=n_steps, polish_lr=lr, ce_weight=ce))

    if not args.skip_exl3:
        tmp_root = root / f"exl3_tmp_{args.model}"
        for label, bpw in EXL3_TARGETS:
            if label in results:
                continue
            work_dir, out_dir = tmp_root / f"audit_work_{bpw:.3f}", tmp_root / f"audit_out_{bpw:.3f}"
            for d in (work_dir, out_dir):
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
            model.cpu()
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            subprocess.run(
                [EXL3_PYTHON, EXL3_CONVERT, "-i", str(model_dir), "-o", str(out_dir), "-w", str(work_dir),
                 "-b", str(bpw), "-hb", "16", "-eb", "16", "-cd", str(cal_path), "-cr", "64", "-cc", "2048",
                 "-cb", "mul1"],
                check=True,
            )
            quant_sec = time.perf_counter() - t0
            model.to(device)
            restore_clean_model_weights(model, clean_weights)
            info = load_exl3_into_hf_model(model, out_dir, device)
            record(label, dict(info, quant_sec=quant_sec))
            shutil.rmtree(work_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    # Paired comparisons: CE vs. no CE and v2 vs. EXL3 at the same bitrate
    pairs = []
    for base, _, _, _ in prof["targets"]:
        pairs.append((f"{base}-orig-ce0.08", f"{base}-orig-ce0"))
        pairs.append((f"{base}-ext", f"{base}-ext-rerun"))
    pairs += [("LIQK_V2-3.44bpw-ext", "EXL3-mul1-3.44bpw"), ("LIQK_V2-4.50bpw-ext", "EXL3-mul1-4.50bpw"),
              ("LIQK_V2-3.44bpw-orig-ce0.08", "EXL3-mul1-3.44bpw"), ("LIQK_V2-4.50bpw-orig-ce0.08", "EXL3-mul1-4.50bpw")]
    comparisons = {}
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        comparisons[f"{a} vs {b}"] = {
            ds: {
                "kld": paired_bootstrap_diff(results[a][ds], results[b][ds]),
                "ppl": paired_bootstrap_ppl_diff(results[a][ds], results[b][ds]),
            }
            for ds in ("nemotron", "multilingual", "wikitext2")
        }
    (root / "results" / f"audit_{args.model}_comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print("Done:", out_json, flush=True)


if __name__ == "__main__":
    main()
