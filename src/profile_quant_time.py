"""
Per-stage breakdown of `.liqk v2` quant creation time (experiment A before scaling up).

Repeats the pipeline of the original `run_*_liqk_v2_experiments.py` (cal 64 rows, polish without CE) for a single
target bitrate and times each stage with `torch.cuda.synchronize()`. Timers are nested:
for each stage the inclusive (incl) and self (self, excluding nested timers) time is printed.
Everything not covered by timers inside the per-layer loop (forward passes, CPU <-> GPU copies,
Python overhead) ends up in the self time of `quantize_llama_model_liqk_v2`.

`torch.profiler` is deliberately not used here: over the whole quantization stage it keeps a record
of each of the millions of small LDLQ CUDA ops and ate 58 GB of RAM (OOM on 1B).

    python -m src.profile_quant_time --model 3b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

import src.liqk_v2 as lv2

PROFILES = {
    "1b": {"model_dir": "Llama-3.2-1B-Instruct", "hidden_batch": 4, "polish_steps": 100, "lr": {3.4375: 5e-4, 4.5: 3e-4}},
    "3b": {"model_dir": "Llama-3.2-3B-Instruct", "hidden_batch": 2, "polish_steps": 80, "lr": {3.4375: 4e-4, 4.5: 2.5e-4}},
}

STACK: list[str] = []
STATS: dict[str, dict] = defaultdict(lambda: {"n": 0, "incl": 0.0, "child": 0.0})


def timed(name: str, fn):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        STACK.append(name)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            STACK.pop()
            STATS[" > ".join(STACK + [name])]["n"] += 1
            STATS[" > ".join(STACK + [name])]["incl"] += dt
            if STACK:
                STATS[" > ".join(STACK)]["child"] += dt

    return wrapper


def stage(name: str, fn, *args, **kwargs):
    return timed(name, fn)(*args, **kwargs)


def install_timers():
    # Module functions call each other through src.liqk_v2 globals, so patching the module
    # also intercepts internal calls
    for fn_name in [
        "compute_optimal_suh",
        "quantize_liqk_v2_diag_rot",
        "_quantize_rows_lut_ldlq",
        "quantize_matrix_liqk_v2",
        "fwht_inplace",
        "get_warp_register_bank",
    ]:
        setattr(lv2, fn_name, timed(fn_name, getattr(lv2, fn_name)))
    lv2._CovAccumulator.add = timed("_CovAccumulator.add", lv2._CovAccumulator.add)
    torch.linalg.cholesky = timed("linalg.cholesky", torch.linalg.cholesky)
    torch.cholesky_inverse = timed("cholesky_inverse", torch.cholesky_inverse)
    torch.linalg.solve_triangular = timed("linalg.solve_triangular", torch.linalg.solve_triangular)
    torch.linalg.solve = timed("linalg.solve", torch.linalg.solve)


def print_tree(total_wall: float):
    rows = []
    for path, st in STATS.items():
        depth = path.count(" > ")
        rows.append((path, depth, st["n"], st["incl"], st["incl"] - st["child"]))
    rows.sort(key=lambda r: r[0])
    print(f"\n{'stage':74s} {'calls':>8s} {'incl, s':>9s} {'self, s':>9s} {'% wall':>7s}")
    for path, depth, n, incl, self_t in rows:
        name = "  " * depth + path.split(" > ")[-1]
        print(f"{name:74s} {n:8d} {incl:9.2f} {self_t:9.2f} {100 * incl / total_wall:6.1f}%")
    print(f"{'TOTAL wall-clock':74s} {'':8s} {total_wall:9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PROFILES), default="3b")
    ap.add_argument("--target", type=float, default=3.4375)
    ap.add_argument("--eval", action="store_true", help="KLD/PPL on traces after polish (reference outside timers)")
    ap.add_argument("--tag", default="", help="suffix for result file names")
    args = ap.parse_args()

    prof = PROFILES[args.model]
    root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda")
    tag = f"{args.model}_{args.target:.2f}{args.tag}"
    dmon_path = root / "results" / f"profile_time_{tag}_dmon.log"
    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "ut", "-d", "1", "-o", "T"], stdout=open(dmon_path, "w"), stderr=subprocess.DEVNULL
    )
    install_timers()
    marks = []

    def mark(name):
        torch.cuda.synchronize()
        marks.append((name, time.strftime("%H:%M:%S")))

    t_wall0 = time.perf_counter()
    mark("load")
    model = stage(
        "load_model",
        lambda: AutoModelForCausalLM.from_pretrained(str(root / "models" / prof["model_dir"]), dtype=torch.bfloat16)
        .to(device)
        .eval(),
    )
    cal64 = load_file(str(root / "data" / "cal_trace.safetensors"))["input_ids"][:64]

    excluded_sec = 0.0
    if args.eval:
        from src.eval_kld import load_eval_trace, precompute_ref_trace_data

        t0 = time.perf_counter()
        nem_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "nemotron_trace.json"), device)
        mul_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "multilingual_trace.json"), device)
        torch.cuda.synchronize()
        excluded_sec = time.perf_counter() - t0

    mark("ref_hidden")
    ref_hidden = stage(
        "precompute_ref_final_hidden_states",
        lv2.precompute_ref_final_hidden_states,
        model, cal64, n_seqs=64, seq_len=512, batch_size=prof["hidden_batch"],
    )
    mark("fisher")
    fisher_out = stage(
        "compute_exact_row_fisher_sensitivities",
        lv2.compute_exact_row_fisher_sensitivities,
        model, cal64, n_Menu_seqs=8, seq_len=1024,
    )
    mark("slice_profile")
    slice_profile = stage(
        "build_liqk_v2_fisher_slice_profile",
        lv2.build_liqk_v2_fisher_slice_profile,
        model, cal64, batch_size=8, fisher_out=fisher_out,
    )
    mark("dp")
    assignment = stage("solve_fisher_slice_dp", lv2.solve_fisher_slice_dp, slice_profile, target_bpw=args.target)

    def run_quant():
        return lv2.quantize_llama_model_liqk_v2(
            model, cal64, assignment, mlp_perms=slice_profile["mlp_perms"], fisher_out=slice_profile["fisher_out"],
            batch_size=8, cd_sweeps=2, polish_steps=0,
        )

    mark("quantize")
    q_info = stage("quantize_llama_model_liqk_v2", run_quant)

    mark("polish")
    with torch.enable_grad():
        stage(
            "polish_model_scales_kld",
            lv2.polish_model_scales_kld,
            model, cal64, ref_hidden, n_steps=prof["polish_steps"], lr=prof["lr"][args.target],
            batch_size=2, reg_wd=0.02, ce_weight=0.0,
        )
    mark("end")
    total_wall = time.perf_counter() - t_wall0 - excluded_sec
    dmon.terminate()

    print(f"\n=== {args.model} @ {args.target} bpw (actual bpw {q_info['linear_bpw']:.4f}) ===")
    print_tree(total_wall)

    out = {
        "model": args.model,
        "target_bpw": args.target,
        "total_wall_sec": total_wall,
        "stages": {k: {"n": v["n"], "incl": v["incl"], "self": v["incl"] - v["child"]} for k, v in STATS.items()},
        "stage_marks": marks,
        "dmon_log": str(dmon_path),
    }

    if args.eval:
        from src.eval_kld import evaluate_model_on_cached_trace

        for ds, ref in (("nemotron", nem_ref), ("multilingual", mul_ref)):
            r = evaluate_model_on_cached_trace(model, ref, device)
            out[ds] = {k: r[k] for k in ("mean_kld", "ci95_lo", "ci95_hi", "ppl_quant", "top1_match")}
            out[ds]["per_seq_kld_sums"] = r["per_seq_kld_sums"]
            out[ds]["per_seq_tok_counts"] = r["per_seq_tok_counts"]
            print(f"{ds:13s} KLD {r['mean_kld']:.5f} [{r['ci95_lo']:.5f}, {r['ci95_hi']:.5f}]  PPL {r['ppl_quant']:.4f}")

    (root / "results" / f"profile_time_{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
