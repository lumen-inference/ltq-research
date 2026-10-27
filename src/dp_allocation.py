"""
Where DP (Fisher-Slice) spends bits in the streaming pipeline: mean bitrate by matrix type (attention / MoE experts /
dense MLP), by layer and by expert sensitivity. The profile is built as in `streaming_quant` (16 sequences).

    python -m src.dp_allocation --model_dir models/Qwen3-30B-A3B-Instruct-2507 --data_dir data/qwen3 \
        --fisher data/qwen3/fisher_30b.pt --targets 3.0 4.5 --out results/dp_allocation_30b.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import src.liqk_v2 as lv2
import src.streaming_quant as sq
from src.equal_bpw import SETS


def kind_of(sub: str) -> str:
    if sub.startswith("self_attn."):
        return sub
    if sub.startswith("mlp.experts."):
        return "experts." + sub.rsplit(".", 1)[1]
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--fisher", type=Path, required=True)
    ap.add_argument("--targets", type=float, nargs="+", default=[3.0, 4.5])
    ap.add_argument("--menu", default="TCQPQ")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sq.FISHER = torch.load(args.fisher)
    dev = torch.device("cuda")
    model, store, config = sq.build_skeleton(args.model_dir, dev)
    cal = load_file(str(args.data_dir / "cal_trace.safetensors"))["input_ids"][:16]
    with torch.no_grad():
        cal0 = [model.model.embed_tokens(cal[i : i + 4].to(dev)) for i in range(0, 16, 4)]
        pos = model.model.rotary_emb(cal0[0][:1], torch.arange(cal.shape[1], device=dev).unsqueeze(0))
    prof = sq.build_stream_profile(model, store, cal0, pos, config.num_hidden_layers, dev)
    slices = prof["slices"]
    bpw = {sp: lv2.LIQK_V2_SPECS[sp].bpw for sp in SETS[args.menu]}
    res = {}
    for tgt in args.targets:
        asg = lv2.solve_fisher_slice_dp(prof, target_bpw=tgt, candidate_specs=SETS[args.menu])
        by_kind, by_layer = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
        expert = []  # (per-parameter sensitivity, bits/weight) of expert matrices
        for s in slices:
            key = (s["layer"], s["subname"])
            K_in = s["params"] // (s["r1"] - s["r0"])
            b = sum((min(r1, s["r1"]) - max(r0, s["r0"])) * K_in * bpw[sp] for r0, r1, sp in asg[key]
                    if r1 > s["r0"] and r0 < s["r1"])
            k = kind_of(s["subname"])
            by_kind[k][0] += b; by_kind[k][1] += s["params"]
            if k.startswith("self_attn."):
                by_layer[s["layer"]][0] += b; by_layer[s["layer"]][1] += s["params"]
            if k.startswith("experts."):
                expert.append((s["raw_kld3"] / s["params"], b / s["params"]))
        tot_b = sum(v[0] for v in by_kind.values()); tot_p = sum(v[1] for v in by_kind.values())
        attn_b = sum(v[0] for k, v in by_kind.items() if k.startswith("self_attn.")); attn_p = sum(v[1] for k, v in by_kind.items() if k.startswith("self_attn."))
        r = {"total_bpw": tot_b / tot_p, "attn_bpw": attn_b / attn_p, "attn_param_share": attn_p / tot_p,
             "attn_bit_share": attn_b / tot_b, "by_kind": {k: v[0] / v[1] for k, v in sorted(by_kind.items())},
             "attn_by_layer": {l: v[0] / v[1] for l, v in sorted(by_layer.items())}}
        if expert:
            e = np.array(expert); order = np.argsort(e[:, 0]); q = len(order) // 4
            r["experts_bpw_by_sensitivity_quartile"] = [float(e[order[i * q : (i + 1) * q], 1].mean()) for i in range(4)]
            r["experts_bpw_percentiles"] = {p: float(np.percentile(e[:, 1], p)) for p in (5, 25, 50, 75, 95)}
        res[str(tgt)] = r
        print(f"\n=== {tgt} bits/weight: total {r['total_bpw']:.3f}; attention {r['attn_bpw']:.2f} bits/weight "
              f"({100 * r['attn_param_share']:.1f} % of parameters, {100 * r['attn_bit_share']:.1f} % of bits)")
        print("  by type: " + "  ".join(f"{k} {v:.2f}" for k, v in r["by_kind"].items()))
        if expert:
            print("  experts by sensitivity quartile (cold to important): "
                  + " / ".join(f"{v:.2f}" for v in r["experts_bpw_by_sensitivity_quartile"]))
            print("  experts, percentiles 5/25/50/75/95: " + " / ".join(f"{v:.2f}" for v in r["experts_bpw_percentiles"].values()))
        al = list(r["attn_by_layer"].values())
        print(f"  attention by layer: min {min(al):.2f}, median {sorted(al)[len(al) // 2]:.2f}, max {max(al):.2f}; "
              f"first 4: {' '.join(f'{v:.2f}' for v in al[:4])}; last 4: {' '.join(f'{v:.2f}' for v in al[-4:])}")
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
