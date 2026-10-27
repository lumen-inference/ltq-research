"""
Summary of the LTQ vs GSQ-RCO (GGUF) and EXL3 comparison on Qwen3.8-27B (`hybrid_quant`, results/q27_compare.json): a
per-method table and a paired bootstrap of LTQ vs the competitor at the same bitrate over the quantized matrices.

    python -m src.q27_report results/q27_compare.json
"""

import json
import sys

from src.eval_kld import paired_bootstrap_diff

PAIRS = [("dp:TCQPQ:2.9409", "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"), ("dp:TCQPQ:3.0043", "exl3_SC_3.00bpw_H4"),
         ("dp:TCQPQ:3.4794", "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"), ("dp:TCQPQ:3.5038", "exl3_3.50bpw")]


def short(name: str) -> str:
    return name.rsplit("/", 1)[-1] if "/" in name else name


def main():
    r = json.load(open(sys.argv[1]))
    ms = {short(k): v for k, v in r["methods"].items()}
    print(f"{'method':36s} {'bpw':>7s} | {'KLD nem [95 %]':>26s} | {'KLD mul [95 %]':>26s} | top1 nem/mul | PPL nem/mul")
    for k, v in sorted(ms.items(), key=lambda kv: kv[1]["q_bpw"]):
        n, m = v["nemotron"], v["multilingual"]
        print(f"{k:36s} {v['q_bpw']:7.4f} | {n['mean_kld']:.5f} [{n['ci95_lo']:.5f}, {n['ci95_hi']:.5f}] | "
              f"{m['mean_kld']:.5f} [{m['ci95_lo']:.5f}, {m['ci95_hi']:.5f}] | {n['top1_match']:.4f} / {m['top1_match']:.4f} | "
              f"{n['ppl_quant']:.4f} / {m['ppl_quant']:.4f}")
    print(f"PPL BF16: nem {next(iter(ms.values()))['nemotron']['ppl_ref']:.4f} mul {next(iter(ms.values()))['multilingual']['ppl_ref']:.4f}")
    print()
    for a, b in PAIRS:
        if a in ms and b in ms:
            line = [f"{a} ({ms[a]['q_bpw']:.4f}) − {b} ({ms[b]['q_bpw']:.4f}):"]
            for ds in ("nemotron", "multilingual"):
                d = paired_bootstrap_diff(ms[a][ds], ms[b][ds])
                rel = d["mean_diff"] / ms[b][ds]["mean_kld"]
                line.append(f"{ds[:3]} {d['mean_diff']:+.5f} [{d['ci95_lo']:+.5f}, {d['ci95_hi']:+.5f}] ({rel:+.1%})")
            print(" ".join(line))


if __name__ == "__main__":
    main()
