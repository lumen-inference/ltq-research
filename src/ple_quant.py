"""
Quantizing a row table with LTQ rings (`RowCodec::LtqRing`, candidate for v1): the Gemma 4 E4B PLE table
(`embed_tokens_per_layer`, [vocab, n_layers * 256]) is cut into rings of 256 values — one row chunk per (token, layer);
each ring is an LTQ tail-biting trellis (same Viterbi and same formats as for matrices), no Hessian (MSE), with an fp16 scale
per ring. Optionally a Hadamard-256 rotation of the ring with random signs. Everything else is BF16; KLD vs. BF16 on traces
is computed with a full forward pass (all layers on GPU, PLE table on CPU, as in `hybrid_quant.TextModel`).

    python -m src.ple_quant --model_dir models/gemma-4-E4B-it --data_dir data/gemma4 \
        --variants int8 TCQ3X TCQ4X TCQ4X:rot --out results/ple_quant_gemma4.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.liqk_v2 as lv2
from src.eval_kld import get_scored_positions_and_targets, load_eval_trace
from src.hybrid_quant import TextModel
from src.ltq_format import load_formats
from src.tcq_proto import viterbi_ub_cuda

RING = 256


def hadamard_signs(seed: int = 4242) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (RING,), generator=g) * 2 - 1).float()


@torch.no_grad()
def quantize_rings(W: torch.Tensor, spec: str, rot: bool, chunk: int = 1 << 17, scales=(0.9, 1.0, 1.12)) -> tuple[torch.Tensor, float]:
    """W [rows, n*256] (bf16, CPU) -> reconstructed table (bf16, CPU) and bits per value. The ring scale is fp16,
    the MSE-best of `scales` x rms x format multiplier."""
    fmt = load_formats([spec])[spec]
    cfg = lv2.TCQ_CONFIG[spec]
    lut = torch.tensor(fmt.lut, device="cuda")
    gains = torch.tensor(fmt.gain_tab, device="cuda")
    signs = hadamard_signs().cuda()
    rings = W.view(-1, RING)
    out = torch.empty_like(rings)
    for i in range(0, rings.shape[0], chunk):
        x = rings[i : i + chunk].cuda().float()
        if rot:
            x = lv2.fwht_inplace((x * signs).contiguous(), block_size=RING)
        rms = x.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-8) * cfg.get("scale", 1.0)
        best, best_err = None, None
        for sc in scales:
            d = (rms * sc).half().float()
            rec = viterbi_ub_cuda(x / d, lut, fmt.K, 10, fmt.masks[0], gains, sub=fmt.sub, tail_biting=True,
                                  m_extra=fmt.masks[1:], pair_parity=fmt.frac, n_cand=cfg.get("n_cand", 0)) * d
            err = (rec - x).square().sum(dim=1)
            if best is None:
                best, best_err = rec, err
            else:
                better = err < best_err
                best = torch.where(better.unsqueeze(1), rec, best)
                best_err = torch.where(better, err, best_err)
        if rot:
            best = lv2.fwht_inplace(best.contiguous(), block_size=RING) * signs
        out[i : i + chunk] = best.to(out.dtype).cpu()
    return out.view_as(W), fmt.tile_bytes() * 8 / RING + 16 / RING


@torch.no_grad()
def int8_rows(W: torch.Tensor, chunk: int = 1 << 17) -> tuple[torch.Tensor, float]:
    """int8 with an fp16 scale per ring of 256 values (absmax)."""
    rings = W.view(-1, RING)
    out = torch.empty_like(rings)
    for i in range(0, rings.shape[0], chunk):
        x = rings[i : i + chunk].cuda().float()
        d = (x.abs().amax(dim=1, keepdim=True) / 127).half().float().clamp(min=1e-10)
        out[i : i + chunk] = ((x / d).round().clamp(-127, 127) * d).to(out.dtype).cpu()
    return out.view_as(W), 8 + 16 / RING


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", required=True, help="int8 or <TCQ format>[:rot]")
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    tmod = TextModel(args.model_dir, dev)
    for l in range(len(tmod.layers)):
        tmod.load_layer(l)
    tmod.tm.layers = nn.ModuleList(tmod.layers)
    ple = tmod.tm.embed_tokens_per_layer
    W0 = ple.weight.data
    print(f"PLE table: {tuple(W0.shape)} {W0.dtype}, {W0.numel() * 2 / 2**30:.2f} GB in bf16", flush=True)

    tables, info = {"bf16": W0}, {}
    for v in args.variants:
        t0 = time.perf_counter()
        if v == "int8":
            Wq, bpv = int8_rows(W0)
        else:
            spec, *opt = v.split(":")
            Wq, bpv = quantize_rings(W0, spec, rot="rot" in opt)
        err = float(((Wq.float() - W0.float()).square().sum() / W0.float().square().sum()).item()) if W0.numel() < 4e9 else float("nan")
        tables[v] = Wq
        info[v] = {"bits_per_value": bpv, "size_gb": W0.numel() * bpv / 8 / 2**30, "rel_mse": err, "quant_sec": time.perf_counter() - t0}
        print(f"{v}: {bpv:.3f} bits/value ({info[v]['size_gb']:.2f} GB), rel. MSE {err:.2e}, {info[v]['quant_sec']:.0f} s", flush=True)

    res = {}
    for ds in ("nemotron", "multilingual"):
        items = []
        for sq in load_eval_trace(args.data_dir / f"{ds}_trace.json"):
            pos, tg = get_scored_positions_and_targets(sq)
            if pos:
                items.append((torch.tensor([sq["token_ids"][:8192]]), pos))
        if args.max_seqs:
            items = items[: args.max_seqs]
        kl = {v: [] for v in args.variants}
        for ids, pos in items:
            logp = {}
            for v in ["bf16"] + args.variants:
                ple.weight.data = tables[v]
                h = tmod.tm(input_ids=ids.to(dev), use_cache=False).last_hidden_state[0, pos]
                logp[v] = F.log_softmax(tmod.logits(h).float(), dim=-1)
            p = logp["bf16"].exp()
            for v in args.variants:
                kl[v].append((p * (logp["bf16"] - logp[v])).sum(dim=-1).cpu())
        for v in args.variants:
            k = torch.cat(kl[v]).numpy()
            boot = np.random.default_rng(0).choice(len(k), (1000, len(k))).astype(np.int64)
            mb = k[boot].mean(axis=1)
            res.setdefault(v, {})[ds] = {"mean_kld": float(k.mean()), "ci95_lo": float(np.percentile(mb, 2.5)),
                                         "ci95_hi": float(np.percentile(mb, 97.5)), "n_tokens": int(len(k))}
    ple.weight.data = W0
    for v in args.variants:
        res[v].update(info[v])
        print(f"[{v}] {info[v]['bits_per_value']:.3f} bits/value | KLD nem {res[v]['nemotron']['mean_kld']:.5f} "
              f"mul {res[v]['multilingual']['mean_kld']:.5f}", flush=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
