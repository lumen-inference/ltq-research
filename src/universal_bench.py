"""
Universal loss map on the Gaussian rig, K = 2..8 bits per weight: gap to the R-D bound (bits, overhead included).

After FWHT-256 the weights are close to i.i.d. Gaussian, so the gap `gap = bpw - 0.5 * log2(1 / D)` on N(0, 1)
shows how many bits a format loses relative to an ideal quantizer at any bitrate.

  - tcq: Ungerboeck TCQ (nu = 10, m = 359, tail-biting), merged table of 2^(K+1) levels (4 subsets),
    header of h bits per 16 weights (2^h gains), d per 256 weights; overhead as in R formats: h / 16 + 7 / 256. The table is trained by Lloyd
    along the Viterbi path (CUDA).
  - tcq-pwl: same, but the table is piecewise linear in the merged index I: v = base[I >> s] + slope[I >> s] * (I & mask),
    32 segments — two tables of 32 entries (pshufb) for any K.
  - exl3: EXL3 (mul1) per matrix on an i.i.d. Gaussian matrix, H = I; overhead 0.023 bits.

    python -m src.universal_bench --K 2 3 4 5 6 8
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.tcq_proto import make_gains, viterbi_ub_cuda

NU, MASK, SUB, T = 10, 359, 16, 256
OUT = Path(__file__).resolve().parent.parent / "results" / "universal_gaussian.json"


def pwl_project(levels: torch.Tensor, weights: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Weighted least-squares projection of the sorted levels (by merged index I) onto a piecewise-linear function of I."""
    L = levels.numel()
    if L <= n_seg:
        return levels
    seg = L // n_seg
    i = torch.arange(seg, dtype=torch.float64, device=levels.device)
    out = torch.empty_like(levels)
    for s in range(n_seg):
        y, w = levels[s * seg:(s + 1) * seg].double(), weights[s * seg:(s + 1) * seg].double() + 1e-3
        A = torch.stack([torch.ones_like(i), i], dim=1) * w.sqrt().unsqueeze(1)
        coef = torch.linalg.lstsq(A, (y * w.sqrt()).unsqueeze(1)).solution.squeeze(1)
        out[s * seg:(s + 1) * seg] = (coef[0] + coef[1] * i).to(levels.dtype)
    return out


def union_to_tab(levels: torch.Tensor, K: int) -> torch.Tensor:
    """Sorted 2^(K+1) levels -> kernel table [4 subsets, U]: subset j is levels j, j + 4, ..."""
    return levels.view(1 << (K - 1), 4).T.contiguous().flatten()


def tab_to_union(tab: torch.Tensor, K: int) -> torch.Tensor:
    return tab.view(4, 1 << (K - 1)).T.contiguous().flatten()


@torch.no_grad()
def train_tcq(K: int, h: int, step: float, n_seg: int = 0, iters: int = 16, n_tiles: int = 4096, seed: int = 0,
              spread: float = math.sqrt(3.0)):
    dev = "cuda"
    # initial levels are quantiles of N(0, spread^2): spread = sqrt(3) gives density ~ p^(1/3) (Panter–Dite); N(0, 1) quantiles
    # are squeezed toward the center at high K. Lloyd with headers gets stuck in local minima, so both are tried
    n = 1 << (K + 1)
    levels = (spread * torch.special.ndtri((torch.arange(n, dtype=torch.float64) + 0.5) / n)).float().to(dev)
    gains = make_gains(h, step, dev) if h > 0 else torch.ones(1, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    n_cand = 4 if gains.numel() > 4 else 0
    for _ in range(iters):
        x = torch.randn((n_tiles, T), generator=g, device=dev)
        x = x / x.square().mean(dim=1, keepdim=True).sqrt()
        tab = union_to_tab(levels, K)
        _, gi, lev = viterbi_ub_cuda(x, tab, K, NU, MASK, gains, sub=SUB, n_cand=n_cand, return_code=True, tail_biting=True)
        # kernel level index (label * U + u) -> merged index 4u + label
        U = 1 << (K - 1)
        j = ((lev % U) * 4 + lev // U).flatten().long()
        gv, xf = gains[gi.flatten().long()], x.flatten()
        num = torch.zeros_like(levels).index_add_(0, j, gv * xf)
        den = torch.zeros_like(levels).index_add_(0, j, gv * gv)
        new = torch.where(den > 0, num / den.clamp(min=1e-12), levels)
        new, _ = new.sort()
        levels = pwl_project(new, den, n_seg) if n_seg else new
    return union_to_tab(levels, K), gains, n_cand


@torch.no_grad()
def train_tcq_x(K: int, h: int, step: float, m_extra: tuple, iters: int = 16, n_tiles: int = 4096, seed: int = 0,
                spread: float = 1.0, additive: bool = False, sub: int = SUB, pp: bool = False):
    """TCQ with extended labels: nl = 4 << len(m_extra) subsets of 2^(K-1) levels, table [nl, U] in kernel layout.
    Start: quantiles of N(0, spread^2) at nl * U points; subset j is points j, j + nl, ..."""
    dev = "cuda"
    nl, U = 4 << len(m_extra), 1 << (K - 1)
    n = nl * U
    q = (spread * torch.special.ndtri((torch.arange(n, dtype=torch.float64) + 0.5) / n)).float().to(dev)
    tab = q.view(U, nl).T.contiguous().flatten()
    gains = make_gains(h, step, dev) if h > 0 else torch.ones(1, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    n_cand = 4 if gains.numel() > 4 else 0
    for _ in range(iters):
        x = torch.randn((n_tiles, T), generator=g, device=dev)
        x = x / x.square().mean(dim=1, keepdim=True).sqrt()
        _, gi, lev = viterbi_ub_cuda(x, tab, K, NU, MASK, gains, sub=sub, n_cand=n_cand, return_code=True,
                                     tail_biting=True, m_extra=m_extra, pair_parity=pp)
        j, gv, xf = lev.flatten().long(), gains[gi.flatten().long()], x.flatten()
        num = torch.zeros_like(tab).index_add_(0, j, gv * xf)
        den = torch.zeros_like(tab).index_add_(0, j, gv * gv)
        tab = torch.where(den > 0, num / den.clamp(min=1e-12), tab)
        if additive:  # tab[lb, u] = base[u] + off[lb]: weighted least squares over centroids (two tables, pshufb + add)
            c, wgt = tab.view(nl, U), den.view(nl, U) + 1e-6
            off = torch.zeros(nl, device=dev)
            for _ in range(20):
                base = ((c - off[:, None]) * wgt).sum(0) / wgt.sum(0)
                off = ((c - base[None, :]) * wgt).sum(1) / wgt.sum(1)
            tab = (base[None, :] + off[:, None]).flatten()
        else:
            tab = tab.view(nl, U).sort(dim=1).values.flatten()
    return tab, gains, n_cand


@torch.no_grad()
def gap_tcq(K: int, tab: torch.Tensor, gains: torch.Tensor, h: int, n_cand: int, n_tiles: int = 8192, seed: int = 123,
            m_extra: tuple = (), sub: int = SUB, pp: bool = False) -> float:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn((n_tiles, T), generator=g, device="cuda")
    d = w.square().mean(dim=1, keepdim=True).sqrt().half().float()
    # fair: tail-biting (no free choice of the initial state) and a single scale multiplier for the whole set
    # (previously the minimum over the multiplier was taken per tile — a free hint worth ~0.02 bits)
    D = min(float((w - viterbi_ub_cuda(w / (d * sc), tab, K, NU, MASK, gains, sub=sub, n_cand=n_cand, tail_biting=True,
                                       m_extra=m_extra, pair_parity=pp)
                   * d * sc).square().mean()) for sc in (0.75, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25))
    return K - {0: 0.0, 1: 0.5, 2: 0.25, 3: 0.75}[int(pp)] + h / sub + 7 / 256 - 0.5 * math.log2(1.0 / D)


def gap_exl3(K: int, M: int = 256, Kin: int = 4096, seed: int = 7) -> float:
    from src.hybrid_quant import exl3_quantize_weight
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = torch.randn((M, Kin), generator=g, device="cuda")
    H = torch.eye(Kin, device="cuda")
    Wq = exl3_quantize_weight(W, H, K, seed).float()
    D = float((W - Wq).square().mean())
    return K + 0.023 - 0.5 * math.log2(1.0 / D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="*", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--h", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--methods", nargs="*", default=["exl3", "tcq", "tcq-pwl"])
    ap.add_argument("--n_seg", type=int, default=32)
    ap.add_argument("--m_extra", type=int, nargs="*", default=[], help="masks of extra label parities (tcq-x)")
    args = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for K in args.K:
        if "exl3" in args.methods and f"exl3_K{K}" not in res:
            res[f"exl3_K{K}"] = {"gap": gap_exl3(K)}
            print(f"K={K} exl3: {res[f'exl3_K{K}']['gap']:.3f}", flush=True)
        for meth in [m for m in args.methods if m.startswith("tcq")]:
            if meth == "tcq-pwl" and (1 << (K + 1)) <= args.n_seg:
                continue  # the table fits entirely anyway
            mx = tuple(args.m_extra) if meth.startswith("tcq-x") else ()
            additive = meth == "tcq-xa"
            if mx:
                meth = f"tcq-x{'a' if additive else ''}{4 << len(mx)}" + "".join(f"_{v}" for v in mx)
            for h in args.h:
                key = f"{meth}_K{K}_h{h}"
                if key in res:
                    continue
                best = None
                steps = {0: (0.0,), 1: (0.4,), 2: (0.22, 0.3), 3: (0.1, 0.15), 4: (0.06, 0.09)}[h]
                for step in steps:
                    for spread in ((1.0, math.sqrt(3.0)) if h <= 1 else (1.0,)):
                        if mx:
                            tab, gains, nc = train_tcq_x(K, h, step, mx, spread=spread, additive=additive)
                        else:
                            tab, gains, nc = train_tcq(K, h, step, n_seg=args.n_seg if meth == "tcq-pwl" else 0, spread=spread)
                        gp = gap_tcq(K, tab, gains, h, nc, m_extra=mx)
                        if best is None or gp < best["gap"]:
                            best = {"gap": gp, "step": step, "spread": spread, "tab": tab.tolist()}
                res[key] = best
                print(f"K={K} {meth} h={h}: {best['gap']:.3f} (step {best['step']}, spread {best['spread']:.2f})", flush=True)
                OUT.write_text(json.dumps(res, indent=1))
        OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
