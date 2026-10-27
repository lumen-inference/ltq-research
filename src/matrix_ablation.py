"""
Why TCQ lags behind EXL3 on certain matrices (down_proj / gate_proj): ablations on real weights and Hessians.
The Hessian for quantization comes from the first half of the calibration, the error tr(dW H dW^T) / tr(W H W^T) from the second
(otherwise weak damping wins by overfitting to the same sample). Grid: damping (fraction of the mean
diagonal; EXL3 uses sigma_reg 0.025), output rotation (out_rot, like sv in EXL3), bitrate.

    python -m src.matrix_ablation --model_dir models/Qwen3-8B --data_dir data/qwen3 \
        --out results/matrix_ablation_8b.json
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

import src.liqk_v2 as lv2
import src.streaming_quant as sq
from src.hybrid_quant import exl3_quantize_weight

INPUTS = {"self_attn.q_proj": "self_attn.q_proj", "self_attn.o_proj": "self_attn.o_proj", "mlp.gate_proj": "mlp.gate_proj",
          "mlp.up_proj": "mlp.gate_proj", "mlp.down_proj": "mlp.down_proj"}  # matrix -> module whose input gives H
SPEC = {2: "TCQ2X", 3: "TCQ3X", 4: "TCQ4X"}


def collect(model, store, cal, layers, dev, batch=4):
    """bf16 forward pass layer by layer; for layers in `layers`, input Hessians over the two calibration halves."""
    n = cal.shape[0]
    with torch.no_grad():
        hs = [model.model.embed_tokens(cal[i : i + batch].to(dev)) for i in range(0, n, batch)]
        pos = model.model.rotary_emb(hs[0][:1], torch.arange(cal.shape[1], device=dev).unsqueeze(0))
        out = {}
        for l in range(max(layers) + 1):
            layer = model.model.layers[l]
            sq.load_module_weights(layer, f"model.layers.{l}", store, dev)
            if l in layers:
                Hs = {}
                cur = {"half": 0}

                def hook(name):
                    def f(m, a):
                        x = a[0].reshape(-1, a[0].shape[-1]).float()
                        key = (name, cur["half"])
                        Hs[key] = Hs.get(key, 0) + x.T @ x
                    return f

                hk = [layer.get_submodule(m).register_forward_pre_hook(hook(m)) for m in set(INPUTS.values())]
                for i, h in enumerate(hs):
                    cur["half"] = int(i >= len(hs) // 2)
                    layer(h, position_embeddings=pos)
                for x in hk:
                    x.remove()
                out[l] = {"H": {k: v / v.diagonal().mean() for k, v in Hs.items()},
                          "W": {m: layer.get_submodule(m).weight.data.float().clone() for m in INPUTS}}
            hs = [layer(h, position_embeddings=pos) for h in hs]
            model.model.layers[l] = layer.to("meta")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 12, 24, 33])
    ap.add_argument("--K", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--damps", type=float, nargs="+", default=[0.025, 0.05, 0.1, 0.2])
    ap.add_argument("--n_cal", type=int, default=64)
    args = ap.parse_args()
    os.environ["TCQ_DAMP_MULT"] = "1"  # damping is set explicitly
    dev = torch.device("cuda")
    model, store, _ = sq.build_skeleton(args.model_dir, dev)
    cal = load_file(str(args.data_dir / "cal_trace.safetensors"))["input_ids"][: args.n_cal]
    t0 = time.perf_counter()
    data = collect(model, store, cal, args.layers, dev)
    print(f"Hessians: {time.perf_counter() - t0:.0f} s", flush=True)

    rows = []
    for l, d in data.items():
        for name, src in INPUTS.items():
            W = d["W"][name]
            Hq, He = d["H"][(src, 0)], d["H"][(src, 1)]
            refs = {h: float(torch.einsum("ij,jk,ik->", W, H, W)) for h, H in (("q", Hq), ("e", He))}

            def err(Wq):
                D = W - Wq.float()
                return {h: float(torch.einsum("ij,jk,ik->", D, H, D)) / refs[h] for h, H in (("q", Hq), ("e", He))}

            for K in args.K:
                e_x = err(exl3_quantize_weight(W, Hq, K, 7))
                b_x = K + 0.023
                rows.append({"layer": l, "matrix": name, "K": K, "method": "exl3", "bpw": b_x, "err": e_x})
                sp = SPEC[K]
                b_t = lv2.LIQK_V2_SPECS[sp].bpw
                for out_rot in (False, True):
                    for damp in args.damps:
                        _, wq, _ = lv2.quantize_matrix_liqk_v2(W, Hq, sp, seed=7, damp=damp, cd_sweeps=2, out_rot=out_rot)
                        e_t = err(wq)
                        adv = {h: (b_x - b_t) + 0.5 * math.log2(e_x[h] / e_t[h]) for h in ("q", "e")}
                        rows.append({"layer": l, "matrix": name, "K": K, "method": "tcq", "spec": sp, "damp": damp,
                                     "out_rot": out_rot, "bpw": b_t, "err": e_t, "adv_bits": adv})
                        print(f"L{l:2d} {name:17s} K={K} out_rot={int(out_rot)} damp={damp:<5} | EXL3 {e_x['e']:.5f} "
                              f"TCQ {e_t['e']:.5f} | advantage, bits: held-out {adv['e']:+.3f} (own {adv['q']:+.3f})", flush=True)
            args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    del data

    # summary: mean advantage (on the held-out half) by matrix type
    print("\nmedian over layers of advantage over EXL3, bits (held-out calibration half):")
    keys = sorted({(r["out_rot"], r["damp"]) for r in rows if r["method"] == "tcq"})
    print("matrix            K | " + " | ".join(f"rot{int(o)} d{dm}" for o, dm in keys))
    for name in INPUTS:
        for K in args.K:
            vals = []
            for o, dm in keys:
                a = [r["adv_bits"]["e"] for r in rows if r["method"] == "tcq" and r["matrix"] == name and r["K"] == K
                     and r["out_rot"] == o and r["damp"] == dm]
                vals.append(sorted(a)[len(a) // 2] if len(a) % 2 else sum(sorted(a)[len(a) // 2 - 1 : len(a) // 2 + 1]) / 2)
            print(f"{name:17s} {K} | " + " | ".join(f"{v:+.3f}".rjust(len(f'rot{int(o)} d{dm}')) for v, (o, dm) in zip(vals, keys)))


if __name__ == "__main__":
    main()
