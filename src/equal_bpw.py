"""
Comparison of v2, TCQ and EXL3 at the same average bitrate (1B / 3B), in a single session.

Each method uses its own bit allocation at exactly the same target bitrate:
  - v2:   DP over row slices from v2 formats (default candidates for the budget);
  - TCQ:  the same DP over TCQ2 / TCQ3 / TCQ4 (+ LIQK5_V2 on top to reach 4.5: TCQ4 is 4.164 bpw);
  - EXL3: stock convert.py conversion at the target bitrate (mul1, same calibration 64 x 2048).
Polish is off for all (EXL3 has none); with --polish, v2 and TCQ are additionally refined (separate rows).
Comparisons use a paired bootstrap over trace dialogues.

    python -m src.equal_bpw --model 1b
"""

from __future__ import annotations

import argparse
import json
import os

# fragmentation with parallel matrices (LIQK_PAR) caused OOM: expandable segments under both variable names
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import shutil
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

import src.liqk_v2 as lv2
from src.audit_reproducibility import EXL3_CONVERT, EXL3_PYTHON
from src.eval_kld import evaluate_model_on_cached_trace, load_eval_trace, paired_bootstrap_diff, precompute_ref_trace_data
from src.run_1b_exl3_experiments import load_exl3_into_hf_model
from src.run_1b_liqk_v2_experiments import restore_clean_model_weights, save_clean_model_weights

PROFILES = {
    "1b": {"model_dir": "Llama-3.2-1B-Instruct", "polish_steps": 100, "lr": 5e-4, "hidden_batch": 4},
    "3b": {"model_dir": "Llama-3.2-3B-Instruct", "polish_steps": 80, "lr": 4e-4, "hidden_batch": 2},
}
TCQ_CANDIDATES = ["TCQ2", "TCQ3", "TCQ4", "LIQK5_V2"]
TCQR_CANDIDATES = ["TCQ2R", "TCQ25R", "TCQ3R", "TCQ35R", "TCQ4R", "LIQK5_V2"]
V2_ALL = [k for k in lv2.SPEC_DISTORTION_RATIO if k.startswith("LIQK") and k.endswith("_V2")]
TCQH_CANDIDATES = ["TCQ2R", "TCQ3R", "TCQ3RH", "TCQ4R", "TCQ4RH", "TCQ4RH3", "LIQK5_V2"]
# night 2: tail-biting + extended labels (tables <= 32 entries); TCQX16 — TCQ3X16 (64 entries) instead of TCQ3X
TCQX_CANDIDATES = ["TCQ2X0", "TCQ2X", "TCQ3X1", "TCQ3X", "TCQ4X1", "TCQ4X2", "TCQ4X", "LIQK5_V2"]
TCQX16_CANDIDATES = ["TCQ2X0", "TCQ2X", "TCQ3X1", "TCQ3X16", "TCQ4X1", "TCQ4X2", "TCQ4X", "LIQK5_V2"]
# 64-entry tables for K >= 3 (K = 3 — 16 subsets, K = 4 — 8), including h = 0 (3.03 / 4.03 bits/weight)
TCQY_CANDIDATES = ["TCQ2X0", "TCQ2X", "TCQ3Y0", "TCQ3Y1", "TCQ3Y2", "TCQ4Y0", "TCQ4Y1", "TCQ4Y2", "LIQK5_V2"]
TCQYH_CANDIDATES = ["TCQ2X0", "TCQ2X", "TCQ3Y1", "TCQ3Y2", "TCQ4Y1", "TCQ4Y2", "LIQK5_V2"]  # TCQY without h = 0 for K >= 3
# TCQX + K = 5 (h = 3 / 4) instead of LIQK5_V2 (on Gaussian 0.177 / 0.175 at 5.21 / 5.28 bits/weight vs. 0.411 at 5.5)
TCQX5_CANDIDATES = ["TCQ2X0", "TCQ2X", "TCQ3X1", "TCQ3X", "TCQ4X1", "TCQ4X2", "TCQ4X", "TCQ5X3", "TCQ5X"]
# + header per 32 weights (TCQ3W2 / TCQ4W2 — same bitrate as h = 1 per 16, TCQ4W3 — 4.12 bits/weight)
TCQX5W_CANDIDATES = TCQX5_CANDIDATES + ["TCQ3W2", "TCQ4W2", "TCQ4W3"]
# final (no h = 0: the proxy overrates them, worse on the model by mul): TCQ2X .. TCQ5X + headers per 32 weights
TCQF_CANDIDATES = ["TCQ2X", "TCQ3X1", "TCQ3W2", "TCQ3X", "TCQ4X1", "TCQ4W2", "TCQ4W3", "TCQ4X2", "TCQ4X", "TCQ5X3", "TCQ5X"]
SETS = {"v2": None, "TCQ": TCQ_CANDIDATES, "TCQR": TCQR_CANDIDATES, "TCQR+v2": TCQR_CANDIDATES + V2_ALL,
        "TCQH": TCQH_CANDIDATES, "TCQH+v2": TCQH_CANDIDATES + V2_ALL, "TCQX": TCQX_CANDIDATES, "TCQX16": TCQX16_CANDIDATES,
        "TCQY": TCQY_CANDIDATES, "TCQYH": TCQYH_CANDIDATES, "TCQX5": TCQX5_CANDIDATES,
        "TCQX5W": TCQX5W_CANDIDATES, "TCQF": TCQF_CANDIDATES, "TCQX5K4W": TCQX5_CANDIDATES + ["TCQ4W2", "TCQ4W3"],
        # + half-integer (pair parity): bitrate menu step is half a bit, mixing penalty from ~1.25x down to ~1.06x
        "TCQP": TCQX5_CANDIDATES + ["TCQ4W2", "TCQ4W3", "TCQ25X", "TCQ35X1", "TCQ35X", "TCQ45X"],
        # + K = 4.5 with h = 3 and 64-entry half-integer (K = 2.5 / 16 subsets, K = 3.5 / 8)
        # + quarter (D4: K - 0.25 on the K table)
        "TCQPQ": TCQX5_CANDIDATES + ["TCQ4W2", "TCQ4W3", "TCQ25X", "TCQ35X1", "TCQ35X", "TCQ45X", "TCQ275X", "TCQ275X2",
                                     "TCQ375X", "TCQ475X"],
        # + K + 0.25 (all low bits of a group of 4 are equal)
        "TCQPQE": TCQX5_CANDIDATES + ["TCQ4W2", "TCQ4W3", "TCQ25X", "TCQ35X1", "TCQ35X", "TCQ45X", "TCQ275X", "TCQ275X2",
                                      "TCQ375X", "TCQ475X", "TCQ225X", "TCQ325X", "TCQ425X"],
        # + high-precision formats for hypersensitive slices (Gemma 4 E4B: per_layer_projection of layer 21)
        "TCQPQH": TCQX5_CANDIDATES + ["TCQ4W2", "TCQ4W3", "TCQ25X", "TCQ35X1", "TCQ35X", "TCQ45X", "TCQ275X", "TCQ275X2",
                                      "TCQ375X", "TCQ475X", "LIQK6_V2", "LIQK8_V2"],
        "TCQPY": TCQX5_CANDIDATES + ["TCQ4W2", "TCQ4W3", "TCQ25X", "TCQ35X1", "TCQ35X", "TCQ45X", "TCQ45X3",
                                     "TCQ25Y", "TCQ35Y1", "TCQ35Y"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PROFILES), default="1b")
    ap.add_argument("--targets", type=float, nargs="*", default=[2.5, 3.0, 3.4375, 4.0, 4.5])
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--sens_power", type=float, default=1.0, help="exponent of per-slice specific sensitivity in DP")
    ap.add_argument("--tag", default="", help="run label suffix (e.g. -nc4 with TCQ_NCAND=4)")
    ap.add_argument("--sets", nargs="*", default=["v2", "TCQ"], choices=sorted(SETS),
                    help="DP candidate sets: v2, TCQ (TCQ2/3/4), TCQR (R variants and half-integer), TCQH (R + 2/3-bit header), +v2 — together with v2 formats")
    args = ap.parse_args()

    prof = PROFILES[args.model]
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "models" / prof["model_dir"]
    cal_path = root / "data" / "cal_trace.safetensors"
    out_json = root / "results" / f"equal_bpw_{args.model}.json"
    device = torch.device("cuda")
    results = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"runs": {}}

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device).eval()
    clean = save_clean_model_weights(model)
    cal64 = load_file(str(cal_path))["input_ids"][:64]
    nem_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "nemotron_trace.json"), device)
    mul_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "multilingual_trace.json"), device)
    ref_hidden = lv2.precompute_ref_final_hidden_states(model, cal64, n_seqs=64, seq_len=512, batch_size=prof["hidden_batch"]) if args.polish else None
    fisher = lv2.compute_exact_row_fisher_sensitivities(model, cal64, n_Menu_seqs=8, seq_len=1024)
    t0 = time.perf_counter()
    profile = lv2.build_liqk_v2_fisher_slice_profile(model, cal64, batch_size=8, fisher_out=fisher)
    profile_sec = time.perf_counter() - t0

    def evaluate(label: str, info: dict):
        res = {"info": info, "nemotron": evaluate_model_on_cached_trace(model, nem_ref, device),
               "multilingual": evaluate_model_on_cached_trace(model, mul_ref, device)}
        results["runs"][label] = res
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[{label:24s}] bpw {info['linear_bpw']:.4f} | time {info.get('quant_sec', 0):.0f} s | "
              f"KLD nem {res['nemotron']['mean_kld']:.5f} mul {res['multilingual']['mean_kld']:.5f}", flush=True)

    def shares(assignment):
        sh = {}
        for (l, sub), slices in assignment.items():
            K = model.model.layers[l].get_submodule(sub).weight.shape[1]
            for r0, r1, sp in slices:
                sh[sp] = sh.get(sp, 0) + (r1 - r0) * K
        tot = sum(sh.values())
        return {k: round(100 * v / tot, 1) for k, v in sorted(sh.items())}

    for target in args.targets:
        for name in args.sets:
            cands = SETS[name]
            label = f"{name}-{target:.4f}{args.tag}"
            if label in results["runs"] and (not args.polish or f"{label}-Polish" in results["runs"]):
                continue
            restore_clean_model_weights(model, clean)
            assignment = lv2.solve_fisher_slice_dp(profile, target_bpw=target, candidate_specs=cands, sens_power=args.sens_power)
            q = lv2.quantize_llama_model_liqk_v2(model, cal64, assignment, mlp_perms=profile["mlp_perms"],
                                                  fisher_out=profile["fisher_out"], batch_size=8, cd_sweeps=2, polish_steps=0)
            info = {"linear_bpw": q["linear_bpw"], "quant_sec": q["quant_sec"], "profile_sec": profile_sec,
                    "target_bpw": target, "shares_pct": shares(assignment)}
            evaluate(label, info)
            if args.polish:
                with torch.enable_grad():
                    pol = lv2.polish_model_scales_kld(model, cal64, ref_hidden, n_steps=prof["polish_steps"], lr=prof["lr"],
                                                      batch_size=2, reg_wd=0.02, ce_weight=0.0)
                evaluate(f"{label}-Polish", dict(info, polish_sec=pol))

        label = f"EXL3-{target:.4f}"
        if label not in results["runs"]:
            tmp = root / f"exl3_tmp_{args.model}"
            work_dir, out_dir = tmp / f"eq_work_{target:.4f}", tmp / f"eq_out_{target:.4f}"
            for d in (work_dir, out_dir):
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
            model.cpu()
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            subprocess.run([EXL3_PYTHON, EXL3_CONVERT, "-i", str(model_dir), "-o", str(out_dir), "-w", str(work_dir),
                            "-b", str(target), "-hb", "16", "-eb", "16", "-cd", str(cal_path), "-cr", "64", "-cc", "2048",
                            "-cb", "mul1"], check=True, stdout=subprocess.DEVNULL)
            quant_sec = time.perf_counter() - t0
            model.to(device)
            restore_clean_model_weights(model, clean)
            info = load_exl3_into_hf_model(model, out_dir, device)
            evaluate(label, dict(info, quant_sec=quant_sec, target_bpw=target))
            shutil.rmtree(work_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    comps = {}
    for target in args.targets:
        t = f"{target:.4f}"
        pairs = [(f"{n}-{t}", f"EXL3-{t}") for n in SETS] + [(f"{n}-{t}", f"v2-{t}") for n in SETS if n != "v2"]
        pairs += [(f"{n}-{t}-Polish", f"EXL3-{t}") for n in SETS]
        if args.tag:
            pairs += [(f"{n}-{t}{args.tag}", f"{n}-{t}") for n in SETS] + [(f"{n}-{t}{args.tag}", f"EXL3-{t}") for n in SETS]
        for a, b in pairs:
            if a in results["runs"] and b in results["runs"]:
                comps[f"{a} vs {b}"] = {ds: paired_bootstrap_diff(results["runs"][a][ds], results["runs"][b][ds])
                                        for ds in ("nemotron", "multilingual")}
                d = comps[f"{a} vs {b}"]
                print(f"{a} - {b}: ΔKLD nem {d['nemotron']['mean_diff']:+.5f} [{d['nemotron']['ci95_lo']:+.5f}, {d['nemotron']['ci95_hi']:+.5f}]"
                      f" | mul {d['multilingual']['mean_diff']:+.5f} [{d['multilingual']['ci95_lo']:+.5f}, {d['multilingual']['ci95_hi']:+.5f}]")
    results["comparisons"] = comps
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
