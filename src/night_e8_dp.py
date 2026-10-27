"""
Night stage 1: validating E8 and Ungerboeck TCQ on real KLD (1B / 3B).

In a single session (shared slice profile, shared references):
  - uniform quants: E8_2 (2.125), E8_2G4 (2.1875), LIQK2_25_V2 (2.25), LIQK2_V2 (2.375), before and after polish;
  - DP at 2.375 and 2.90 bpw: v2 formats vs v2 formats + E8_2 / E8_2G4.
Polish uses the original settings without CE. Comparisons: paired bootstrap over trace dialogues.

    python -m src.night_e8_dp --model 3b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

import src.liqk_v2 as lv2
from src.eval_kld import evaluate_model_on_cached_trace, load_eval_trace, paired_bootstrap_diff, precompute_ref_trace_data
from src.run_1b_liqk_v2_experiments import restore_clean_model_weights, save_clean_model_weights

PROFILES = {
    "1b": {"model_dir": "Llama-3.2-1B-Instruct", "polish_steps": 100, "lr": 5e-4, "hidden_batch": 4},
    "3b": {"model_dir": "Llama-3.2-3B-Instruct", "polish_steps": 80, "lr": 4e-4, "hidden_batch": 2},
}
E8_SPECS = ["E8_2", "E8_2G4"]
DP_SPECS = {
    2.375: ["LIQK2_25_V2", "LIQK2_0_V2", "LIQK2_V2", "LIQK2_5_V2", "LIQK3_25_V2", "LIQK3_V2"],
    2.90: ["LIQK2_V2", "LIQK2_5_V2", "LIQK2_7_V2", "LIQK3_25_V2", "LIQK3_38_V2", "LIQK3_V2", "LIQK3_5_V2"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PROFILES), default="3b")
    ap.add_argument("--uniform", nargs="*", default=["E8_2", "E8_2G4", "LIQK2_25_V2", "LIQK2_V2"])
    ap.add_argument("--budgets", type=float, nargs="*", default=[2.375, 2.90])
    args = ap.parse_args()

    prof = PROFILES[args.model]
    root = Path(__file__).resolve().parent.parent
    out_json = root / "results" / f"night_e8_dp_{args.model}.json"
    device = torch.device("cuda")
    results = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"runs": {}}

    model = AutoModelForCausalLM.from_pretrained(str(root / "models" / prof["model_dir"]), dtype=torch.bfloat16).to(device).eval()
    clean = save_clean_model_weights(model)
    cal64 = load_file(str(root / "data" / "cal_trace.safetensors"))["input_ids"][:64]
    nem_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "nemotron_trace.json"), device)
    mul_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "multilingual_trace.json"), device)
    ref_hidden = lv2.precompute_ref_final_hidden_states(model, cal64, n_seqs=64, seq_len=512, batch_size=prof["hidden_batch"])
    fisher = lv2.compute_exact_row_fisher_sensitivities(model, cal64, n_Menu_seqs=8, seq_len=1024)
    t0 = time.perf_counter()
    slice_profile = lv2.build_liqk_v2_fisher_slice_profile(model, cal64, batch_size=8, fisher_out=fisher)
    profile_sec = time.perf_counter() - t0

    def evaluate(label: str, info: dict):
        res = {"info": info, "nemotron": evaluate_model_on_cached_trace(model, nem_ref, device),
               "multilingual": evaluate_model_on_cached_trace(model, mul_ref, device)}
        results["runs"][label] = res
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[{label:30s}] bpw {info['linear_bpw']:.4f} | quant {info['quant_sec']:.0f} s | KLD nem {res['nemotron']['mean_kld']:.5f} "
              f"mul {res['multilingual']['mean_kld']:.5f} | PPL nem {res['nemotron']['ppl_quant']:.4f} mul {res['multilingual']['ppl_quant']:.4f}",
              flush=True)

    def run(label: str, spec_or_assignment, extra: dict):
        if f"{label}-Polish" in results["runs"]:
            return
        restore_clean_model_weights(model, clean)
        q_info = lv2.quantize_llama_model_liqk_v2(
            model, cal64, spec_or_assignment, mlp_perms=slice_profile["mlp_perms"], fisher_out=slice_profile["fisher_out"],
            batch_size=8, cd_sweeps=2, polish_steps=0,
        )
        info = {"linear_bpw": q_info["linear_bpw"], "quant_sec": q_info["quant_sec"], "profile_sec": profile_sec, **extra}
        evaluate(f"{label}-NoPolish", info)
        with torch.enable_grad():
            pol = lv2.polish_model_scales_kld(model, cal64, ref_hidden, n_steps=prof["polish_steps"], lr=prof["lr"],
                                              batch_size=2, reg_wd=0.02, ce_weight=0.0)
        evaluate(f"{label}-Polish", dict(info, polish_sec=pol))

    for spec in args.uniform:
        run(f"U-{spec}", spec, {"spec": spec})

    for budget in args.budgets:
        base = DP_SPECS[budget]
        for name, cands in (("v2", base), ("v2+E8", base + E8_SPECS)):
            assignment = lv2.solve_fisher_slice_dp(slice_profile, target_bpw=budget, candidate_specs=cands)
            shares = {}
            for (l, sub), slices in assignment.items():
                K = model.model.layers[l].get_submodule(sub).weight.shape[1]
                for r0, r1, sp in slices:
                    shares[sp] = shares.get(sp, 0) + (r1 - r0) * K
            tot = sum(shares.values())
            run(f"DP{budget:.2f}-{name}", assignment, {"target_bpw": budget, "shares_pct": {k: round(100 * v / tot, 1) for k, v in sorted(shares.items())}})

    # Paired comparisons
    comps = {}
    pairs = [("U-E8_2-Polish", "U-LIQK2_25_V2-Polish"), ("U-E8_2G4-Polish", "U-LIQK2_25_V2-Polish"), ("U-E8_2-Polish", "U-LIQK2_V2-Polish")]
    pairs += [(f"DP{b:.2f}-v2+E8-Polish", f"DP{b:.2f}-v2-Polish") for b in args.budgets]
    # Ungerboeck TCQ vs v2 of the nearest bitrate (TCQ2 2.16 / TCQ3 3.16 / TCQ4 4.16 bpw)
    for tcq, ref in (("TCQ2", "LIQK2_V2"), ("TCQ2", "E8_2"), ("TCQ3", "LIQK3_V2"), ("TCQ4", "LIQK4_V2"), ("TCQ4", "LIQK3_5_V2")):
        for pol in ("NoPolish", "Polish"):
            pairs.append((f"U-{tcq}-{pol}", f"U-{ref}-{pol}"))
    for a, b in pairs:
        if a in results["runs"] and b in results["runs"]:
            comps[f"{a} vs {b}"] = {ds: paired_bootstrap_diff(results["runs"][a][ds], results["runs"][b][ds]) for ds in ("nemotron", "multilingual")}
            for ds, d in comps[f"{a} vs {b}"].items():
                print(f"{a} - {b} [{ds}]: ΔKLD {d['mean_diff']:+.5f} [{d['ci95_lo']:+.5f}, {d['ci95_hi']:+.5f}]")
    results["comparisons"] = comps
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
