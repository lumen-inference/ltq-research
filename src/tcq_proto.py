"""
TCQ (trellis-coded quantization) prototype with a short bitshift trellis and a trainable codebook table.

The state of weight i is a window of L bits of the stream: s_i = ((s_{i-1} << K) | b_i) mod 2^L, reconstruction is lut[s_i].
Decode: funnel shift + mask + one lookup in a table of 2^L values (IQ*_K class). Encoding is Viterbi over 2^L
states with 2^K transitions (cost 256 * 2^L * 2^K per tile of 256 weights; EXL3 has L = 16).
Tile bitrate: 256*K + (L-K) bits (free initial state) + fp16 scale d.

Here: a check on a Gaussian signal: gap to the R-D bound, in bits, in the same setup as E8/v2 earlier today
(scale d = tile rms, overhead included).

    python -m src.tcq_proto
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch


@torch.no_grad()
def viterbi_tcq(x: torch.Tensor, w: torch.Tensor | None, lut: torch.Tensor, K: int, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """x [N, T] -> (reconstruction [N, T], states [N, T]) with minimal weighted error sum w_t (x_t - lut[s_t])^2."""
    N, T = x.shape
    S, Sl = 1 << L, 1 << (L - K)
    if w is None:
        w = torch.ones_like(x)
    elif w.dim() == 1:
        w = w.view(1, T).expand(N, T)
    dp = w[:, :1] * (x[:, :1] - lut.view(1, S)).square()
    bp = torch.empty((T, N, Sl), dtype=torch.uint8, device=x.device)
    for t in range(1, T):
        m, arg = dp.view(N, 1 << K, Sl).min(dim=1)
        bp[t] = arg.to(torch.uint8)
        dp = m.repeat_interleave(1 << K, dim=1) + w[:, t : t + 1] * (x[:, t : t + 1] - lut.view(1, S)).square()
    states = torch.empty((N, T), dtype=torch.long, device=x.device)
    s = dp.argmin(dim=1)
    ar = torch.arange(N, device=x.device)
    for t in range(T - 1, -1, -1):
        states[:, t] = s
        if t > 0:
            low = s >> K
            s = bp[t][ar, low].long() * Sl + low
    return lut[states], states


def init_lut(K: int, L: int, seed: int = 0, device="cuda") -> torch.Tensor:
    """N(0,1) quantiles in random order: the 2^K successors of each state cover the whole range."""
    S = 1 << L
    q = torch.special.ndtri((torch.arange(S, dtype=torch.float64) + 0.5) / S).float()
    g = torch.Generator().manual_seed(seed)
    return q[torch.randperm(S, generator=g)].to(device)


def additive_init(K: int, L: int, seed: int = 0, device="cuda") -> torch.Tensor:
    """Structured table L = 8: lut[s] = A[s >> 4] + B[s & 15] (CPU decode is two 16-entry pshufb/tbl)."""
    assert L == 8
    g = torch.Generator().manual_seed(seed)
    q = torch.special.ndtri((torch.arange(16, dtype=torch.float64) + 0.5) / 16).float()
    A, B = q[torch.randperm(16, generator=g)] * 0.6, q[torch.randperm(16, generator=g)] * 0.8
    return (A.view(16, 1) + B.view(1, 16)).flatten().to(device)


def coset_index(K: int, L: int, nb: int, seed: int = 0, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ungerboeck-style TCQ within a bitshift window: value = small[label(s), sym(s)], where sym is the low K bits of the window
    (the new symbol), label is nb parity bits of the old L-K window bits by fixed masks. Long memory with
    a small table of 2^(K+nb) (16-32 entries: pshufb/tbl on CPU). Returns (index into small for each
    state [2^L], initial small [2^(K+nb)] — N(0,1) quantiles interleaved across subsets).
    """
    g = torch.Generator().manual_seed(1000 + seed)
    old_bits = L - K
    masks = []
    while len(masks) < nb:
        m = int(torch.randint(1, 1 << old_bits, (1,), generator=g))
        if m not in masks:
            masks.append(m)
    st = torch.arange(1 << L)
    label = torch.zeros_like(st)
    for j, m in enumerate(masks):
        v = (st >> K) & m
        par = torch.zeros_like(v)
        while bool(v.any()):
            par ^= v & 1
            v = v >> 1
        label |= par << j
    idx = label * (1 << K) + (st & ((1 << K) - 1))
    n = 1 << (K + nb)
    q = torch.special.ndtri((torch.arange(n, dtype=torch.float64) + 0.5) / n).float()
    small = q.view(1 << K, 1 << nb).T.contiguous().flatten()  # small[label * 2^K + sym] = q[sym * 2^nb + label]
    return idx.to(device), small.to(device)


@torch.no_grad()
def train_coset(K: int, L: int, nb: int, iters: int = 8, n_tiles: int = 4096, T: int = 256, seed: int = 0, device="cuda"):
    idx, small = coset_index(K, L, nb, seed, device)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    for _ in range(iters):
        x = torch.randn((n_tiles, T), generator=g, device=device)
        x = x / x.square().mean(dim=1, keepdim=True).sqrt()
        _, st = viterbi_tcq(x, None, small[idx], K, L)
        j, xf = idx[st.flatten()], x.flatten()
        sums = torch.zeros_like(small).index_add_(0, j, xf)
        cnt = torch.zeros_like(small).index_add_(0, j, torch.ones_like(xf))
        small = torch.where(cnt > 0, sums / cnt.clamp(min=1), small)
    return small[idx], small


@torch.no_grad()
def train_lut(K: int, L: int, iters: int = 8, n_tiles: int = 4096, T: int = 256, seed: int = 0, device="cuda",
              structure: str = "full") -> torch.Tensor:
    """Lloyd over Viterbi assignments on Gaussian tiles (normalized by their own rms, as in quantization).
    structure="add": lut = A[hi] + B[lo], update is alternating means over the assignments."""
    lut = init_lut(K, L, seed, device) if structure == "full" else additive_init(K, L, seed, device)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    S = 1 << L
    for _ in range(iters):
        x = torch.randn((n_tiles, T), generator=g, device=device)
        x = x / x.square().mean(dim=1, keepdim=True).sqrt()
        _, st = viterbi_tcq(x, None, lut, K, L)
        st, xf = st.flatten(), x.flatten()
        if structure == "full":
            sums = torch.zeros(S, device=device).index_add_(0, st, xf)
            cnt = torch.zeros(S, device=device).index_add_(0, st, torch.ones_like(xf))
            lut = torch.where(cnt > 0, sums / cnt.clamp(min=1), lut)
        else:
            hi, lo = st >> 4, st & 15
            A, B = lut.view(16, 16)[:, 0].clone(), torch.zeros(16, device=device)
            B = lut.view(16, 16)[0, :] - A[0]
            for _ in range(10):
                cnt_h = torch.zeros(16, device=device).index_add_(0, hi, torch.ones_like(xf)).clamp(min=1)
                A = torch.zeros(16, device=device).index_add_(0, hi, xf - B[lo]) / cnt_h
                cnt_l = torch.zeros(16, device=device).index_add_(0, lo, torch.ones_like(xf)).clamp(min=1)
                B = torch.zeros(16, device=device).index_add_(0, lo, xf - A[hi]) / cnt_l
            lut = (A.view(16, 1) + B.view(1, 16)).flatten()
    return lut


@torch.no_grad()
def gaussian_gap(K: int, L: int, lut: torch.Tensor, n_tiles: int = 8192, T: int = 256, seed: int = 123, device="cuda") -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn((n_tiles, T), generator=g, device=device)
    d = w.square().mean(dim=1, keepdim=True).sqrt().half().float()
    # tile scale: multiplier search over a grid (like d_cand/refit in v2), fp16
    best = None
    for sc in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1):
        rec, _ = viterbi_tcq(w / (d * sc), None, lut, K, L)
        mse = ((w - rec * d * sc).square().mean(dim=1))
        best = mse if best is None else torch.minimum(best, mse)
    D = float(best.mean())
    bpw = K + (L - K + 16) / T
    return {"K": K, "L": L, "bpw": bpw, "mse": D, "gap_bits": bpw - 0.5 * math.log2(1.0 / D)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="*", default=[2, 3, 4])
    ap.add_argument("--L", type=int, nargs="*", default=[6, 8, 10, 12])
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--structure", choices=["full", "add", "coset"], default="full")
    ap.add_argument("--nb", type=int, default=1, help="coset: subset label bits")
    ap.add_argument("--seeds", type=int, default=1, help="coset: number of parity-mask sets (the best is taken)")
    args = ap.parse_args()
    if args.structure == "coset":
        out = Path(__file__).resolve().parent.parent / "results" / "tcq_gaussian.json"
        res = json.loads(out.read_text()) if out.exists() else {}
        for K in args.K:
            for L in args.L:
                gaps = []
                for sd in range(args.seeds):
                    lut, small = train_coset(K, L, args.nb, iters=args.iters, seed=sd)
                    gaps.append((gaussian_gap(K, L, lut), sd, small.tolist()))
                best = min(gaps, key=lambda r: r[0]["gap_bits"])
                r = dict(best[0], seed=best[1], small_lut=best[2], all_gaps=[g[0]["gap_bits"] for g in gaps], nb=args.nb)
                res[f"K{K}_L{L}_coset{args.nb}"] = r
                out.write_text(json.dumps(res, indent=2))
                print(f"coset nb={args.nb} K={K} L={L:2d}: table {1 << (K + args.nb)} | bpw {r['bpw']:.4f} | gap "
                      f"{r['gap_bits']:.3f} bits (per mask: {', '.join(f'{x:.3f}' for x in r['all_gaps'])})", flush=True)
        return
    out = Path(__file__).resolve().parent.parent / "results" / "tcq_gaussian.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for K in args.K:
        for L in args.L:
            if L <= K:
                continue
            t0 = time.perf_counter()
            lut = train_lut(K, L, iters=args.iters, structure=args.structure)
            t_train = time.perf_counter() - t0
            r0 = gaussian_gap(K, L, init_lut(K, L) if args.structure == "full" else additive_init(K, L))
            t0 = time.perf_counter()
            r = gaussian_gap(K, L, lut)
            r["gap_bits_untrained_lut"] = r0["gap_bits"]
            r["train_sec"], r["eval_sec_6scales"] = t_train, time.perf_counter() - t0
            res[f"K{K}_L{L}" + ("" if args.structure == "full" else "_add")] = r
            out.write_text(json.dumps(res, indent=2))
            print(f"{args.structure} K={K} L={L:2d}: bpw {r['bpw']:.4f} | gap to R-D {r['gap_bits']:.3f} bits "
                  f"(untrained table {r0['gap_bits']:.3f}) | training {t_train:.0f} s", flush=True)



# ---------------------------------------------------------------------------------------------------------
# Hybrid: a sub-block header (16 weights) picks a gain from 2^h values, as in v2/E8; the trellis uses a small table.
# Header and trellis path are chosen jointly: Viterbi over (gain, state) pairs, the gain may change only
# at a sub-block boundary. AVX2 decode: 16-32 entry table (pshufb) x sub-block scale (like GGUF block scales).
# ---------------------------------------------------------------------------------------------------------

@torch.no_grad()
def viterbi_tcq_gain(x: torch.Tensor, lut: torch.Tensor, K: int, L: int, gains: torch.Tensor, sub: int = 16):
    """x [N, T] -> (reconstruction, states, per-sub-block gain indices [N, T // sub])."""
    N, T = x.shape
    G, S, Sl = gains.shape[0], 1 << L, 1 << (L - K)
    vals = gains.view(G, 1) * lut.view(1, S)  # [G, S]
    dp = (x[:, :1, None] - vals.view(1, G, S)).square()
    bp = torch.empty((T, N, G, Sl), dtype=torch.uint8, device=x.device)
    gsel = torch.empty((T // sub, N, S), dtype=torch.uint8, device=x.device)
    for t in range(1, T):
        if t % sub == 0:
            m_g, a_g = dp.min(dim=1)
            gsel[t // sub] = a_g.to(torch.uint8)
            dp = m_g.unsqueeze(1).expand(N, G, S)
        m, arg = dp.reshape(N, G, 1 << K, Sl).min(dim=2)
        bp[t] = arg.to(torch.uint8)
        dp = m.repeat_interleave(1 << K, dim=2) + (x[:, t : t + 1, None] - vals.view(1, G, S)).square()
    flat = dp.reshape(N, G * S).argmin(dim=1)
    g, s = flat // S, flat % S
    ar = torch.arange(N, device=x.device)
    states = torch.empty((N, T), dtype=torch.long, device=x.device)
    gidx = torch.empty((N, T // sub), dtype=torch.long, device=x.device)
    for t in range(T - 1, -1, -1):
        states[:, t] = s
        if t % sub == 0:
            gidx[:, t // sub] = g
        if t > 0:
            low = s >> K
            s = bp[t][ar, g, low].long() * Sl + low
            if t % sub == 0:
                g = gsel[t // sub][ar, s].long()
    rec = gains[gidx].repeat_interleave(sub, dim=1) * lut[states]
    return rec, states, gidx


def make_gains(h: int, step: float, device="cuda") -> torch.Tensor:
    G = 1 << h
    return 2.0 ** ((torch.arange(G, dtype=torch.float32, device=device) - (G - 1) / 2) * step)


@torch.no_grad()
def train_gain(K: int, L: int, nb: int, h: int, step: float, iters: int = 6, n_tiles: int = 2048, T: int = 256, seed: int = 0,
               device="cuda"):
    """nb = 0: full 2^L table (for comparison only), otherwise an Ungerboeck table of 2^(K+nb)."""
    if nb > 0:
        idx, small = coset_index(K, L, nb, seed, device)
    else:
        idx, small = torch.arange(1 << L, device=device), init_lut(K, L, seed, device)
    gains = make_gains(h, step, device)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    for _ in range(iters):
        x = torch.randn((n_tiles, T), generator=g, device=device)
        x = x / x.square().mean(dim=1, keepdim=True).sqrt()
        _, st, gi = viterbi_tcq_gain(x, small[idx], K, L, gains)
        gv = gains[gi].repeat_interleave(16, dim=1).flatten()
        j, xf = idx[st.flatten()], x.flatten()
        num = torch.zeros_like(small).index_add_(0, j, gv * xf)
        den = torch.zeros_like(small).index_add_(0, j, gv * gv)
        small = torch.where(den > 0, num / den.clamp(min=1e-12), small)
    return small[idx], gains


@torch.no_grad()
def gaussian_gap_gain(K: int, L: int, lut: torch.Tensor, gains: torch.Tensor, h: int, n_tiles: int = 4096, T: int = 256,
                      seed: int = 123, device="cuda") -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn((n_tiles, T), generator=g, device=device)
    d = w.square().mean(dim=1, keepdim=True).sqrt().half().float()
    best = None
    for sc in (0.9, 1.0, 1.1):
        rec, _, _ = viterbi_tcq_gain(w / (d * sc), lut, K, L, gains)
        mse = (w - rec * d * sc).square().mean(dim=1)
        best = mse if best is None else torch.minimum(best, mse)
    D = float(best.mean())
    bpw = K + (L - K + 16) / T + h / 16
    return {"K": K, "L": L, "h": h, "bpw": bpw, "mse": D, "gap_bits": bpw - 0.5 * math.log2(1.0 / D)}


def main_gain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="*", default=[2, 3, 4])
    ap.add_argument("--L", type=int, nargs="*", default=[10])
    ap.add_argument("--nb", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--h", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--steps", type=float, nargs="*", default=[0.25, 0.4])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    out = Path(__file__).resolve().parent.parent / "results" / "tcq_gaussian.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for K in args.K:
        for L in args.L:
            for nb in args.nb:
                for h in args.h:
                    cands = []
                    for step in args.steps:
                        for sd in range(args.seeds if nb > 0 else 1):
                            lut, gains = train_gain(K, L, nb, h, step, seed=sd)
                            cands.append((gaussian_gap_gain(K, L, lut, gains, h), step, sd))
                    r, step, sd = min(cands, key=lambda c: c[0]["gap_bits"])
                    r.update(nb=nb, step=step, seed=sd, all_gaps=[c[0]["gap_bits"] for c in cands])
                    res[f"K{K}_L{L}_nb{nb}_h{h}"] = r
                    out.write_text(json.dumps(res, indent=2))
                    tab = (1 << L) if nb == 0 else (1 << (K + nb))
                    print(f"hybrid K={K} L={L} nb={nb} h={h}: table {tab} | bpw {r['bpw']:.4f} | gap {r['gap_bits']:.3f} bits "
                          f"(step {step}, masks {sd}; all: {', '.join(f'{x:.3f}' for x in r['all_gaps'])})", flush=True)



# ---------------------------------------------------------------------------------------------------------
# Classic Ungerboeck TCQ: 1 code bit per weight goes into the trellis memory (nu bits), the remaining K-1 bits select
# the level within the subset (parallel transitions). Subset label (2 bits) = (parity of the old memory bits by
# mask m, new code bit). Table of 4 x 2^(K-1) = 2^(K+1) entries: K=3 gives 16 (one pshufb), K=4 gives 32 (two).
# AVX2 decode: the tile's code bits fit in one 256-bit register, the label is an XOR of its shifts by the mask.
# ---------------------------------------------------------------------------------------------------------

def _ub_step_impl(dp, xt, wt, vals, lab_of):
    """One Viterbi step: dp [N, G, S] -> (dp', best predecessor [N, G, S/2] uint8)."""
    N, G, S = dp.shape
    mn, arg = dp.reshape(N, G, 2, S // 2).min(dim=2)
    e = (xt.view(N, 1, 1, 1) - vals.unsqueeze(0)).square().amin(dim=3) * wt.view(N, 1, 1)   # [N, G, 4]
    return mn.repeat_interleave(2, dim=2) + e[:, :, lab_of], arg.to(torch.uint8)


_ub_step_compiled = None


def _ub_steps(dp, xs, ws, vals, lab_of):
    """Sub-block steps: xs, ws [N, n] -> (dp, predecessors [n, N, G, S/2]). The step is fused by torch.compile
    (~2x faster; TCQ_NO_COMPILE=1 disables compilation)."""
    global _ub_step_compiled
    import os
    step = _ub_step_impl
    if os.environ.get("TCQ_NO_COMPILE") != "1":
        if _ub_step_compiled is None:
            _ub_step_compiled = torch.compile(_ub_step_impl, dynamic=True)
        step = _ub_step_compiled
    args = []
    for i in range(xs.shape[1]):
        dp, a = step(dp, xs[:, i], ws[:, i], vals, lab_of)
        args.append(a)
    return dp, torch.stack(args)


def _parity(v: torch.Tensor) -> torch.Tensor:
    p = torch.zeros_like(v)
    while bool(v.any()):
        p ^= v & 1
        v = v >> 1
    return p


@torch.no_grad()
def viterbi_ub(x: torch.Tensor, small: torch.Tensor, K: int, nu: int, m: int, gains: torch.Tensor | None = None, sub: int = 16,
               wts: torch.Tensor | None = None):
    """x [N, T] -> (reconstruction, labels [N, T], indices within the subset [N, T], sub-block gains [N, T//sub]).
    wts [N, T] or [T]: per-position error weights (LDLQ diagonal)."""
    N, T = x.shape
    S, U = 1 << nu, 1 << (K - 1)
    dev = x.device
    st = torch.arange(S, device=dev)
    lab_of = (_parity((st >> 1) & m) << 1) | (st & 1)          # [S]
    tab = small.view(4, U)
    G = 1 if gains is None else gains.shape[0]
    gv = torch.ones(1, device=dev) if gains is None else gains
    vals = gv.view(G, 1, 1) * tab.view(1, 4, U)                   # [G, 4, U]

    if wts is not None and wts.dim() == 1:
        wts = wts.view(1, T).expand(N, T)

    def branch_err(t):  # -> (error [N, G, S], best u [N, G, 4])
        e = (x[:, t].view(N, 1, 1, 1) - vals.view(1, G, 4, U)).square()
        em, eu = e.min(dim=3)                                     # [N, G, 4]
        if wts is not None:
            em = em * wts[:, t].view(N, 1, 1)
        return em[:, :, lab_of], eu

    dp, _ = branch_err(0)
    bp = torch.empty((T, N, G, S // 2), dtype=torch.uint8, device=dev)
    gsel = torch.empty((max(T // sub, 1), N, S), dtype=torch.uint8, device=dev)
    w_all = wts if wts is not None else torch.ones_like(x)
    for t0 in range(0, T, sub):
        if G > 1 and t0 > 0:
            m_g, a_g = dp.min(dim=1)
            gsel[t0 // sub] = a_g.to(torch.uint8)
            dp = m_g.unsqueeze(1).expand(N, G, S).contiguous()
        lo = max(t0, 1)
        dp, args = _ub_steps(dp, x[:, lo : t0 + sub], w_all[:, lo : t0 + sub], vals, lab_of)
        bp[lo : t0 + sub] = args
    flat = dp.reshape(N, G * S).argmin(dim=1)
    g, s = flat // S, flat % S
    ar = torch.arange(N, device=dev)
    states = torch.empty((N, T), dtype=torch.long, device=dev)
    gidx = torch.zeros((N, max(T // sub, 1)), dtype=torch.long, device=dev)
    for t in range(T - 1, -1, -1):
        states[:, t] = s
        if t % sub == 0:
            gidx[:, t // sub] = g
        if t > 0:
            low = s >> 1
            s = bp[t][ar, g, low].long() * (S // 2) + low
            if G > 1 and t % sub == 0:
                g = gsel[t // sub][ar, s].long()
    labels = lab_of[states]
    gfull = gv[gidx].repeat_interleave(sub, dim=1) if G > 1 else torch.ones_like(x)
    cand = (x.unsqueeze(-1) - gfull.unsqueeze(-1) * tab[labels]).square()   # [N, T, U]
    u = cand.argmin(dim=-1)
    rec = gfull * tab[labels, u]
    return rec, labels, u, gidx


def ub_init(K: int, device="cuda") -> torch.Tensor:
    n = 1 << (K + 1)
    q = torch.special.ndtri((torch.arange(n, dtype=torch.float64) + 0.5) / n).float()
    return q.view(1 << (K - 1), 4).T.contiguous().flatten().to(device)  # subset D_j = levels j, j+4, ...


@torch.no_grad()
def train_ub(K: int, nu: int, m: int, h: int = 0, step: float = 0.3, iters: int = 6, n_tiles: int = 2048, T: int = 256,
             seed: int = 0, device="cuda", sub: int = 16):
    small = ub_init(K, device)
    gains = make_gains(h, step, device) if h > 0 else None
    g = torch.Generator(device=device).manual_seed(seed + 1)
    for _ in range(iters):
        x = torch.randn((n_tiles, T), generator=g, device=device)
        x = x / x.square().mean(dim=1, keepdim=True).sqrt()
        _, lab, u, gi = viterbi_ub(x, small, K, nu, m, gains, sub=sub)
        gv = gains[gi].repeat_interleave(sub, dim=1).flatten() if gains is not None else torch.ones(x.numel(), device=device)
        j, xf = (lab * (1 << (K - 1)) + u).flatten(), x.flatten()
        num = torch.zeros_like(small).index_add_(0, j, gv * xf)
        den = torch.zeros_like(small).index_add_(0, j, gv * gv)
        small = torch.where(den > 0, num / den.clamp(min=1e-12), small)
    return small, gains


@torch.no_grad()
def gaussian_gap_ub(K, nu, m, small, gains, h, n_tiles=4096, T=256, seed=123, device="cuda", sub: int = 16) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn((n_tiles, T), generator=g, device=device)
    d = w.square().mean(dim=1, keepdim=True).sqrt().half().float()
    best = None
    for sc in (0.9, 0.95, 1.0, 1.05, 1.1):
        rec, *_ = viterbi_ub(w / (d * sc), small, K, nu, m, gains, sub=sub)
        mse = (w - rec * d * sc).square().mean(dim=1)
        best = mse if best is None else torch.minimum(best, mse)
    D = float(best.mean())
    bpw = K + (nu + 16) / T + h / sub
    return {"K": K, "nu": nu, "m": m, "h": h, "sub": sub, "bpw": bpw, "mse": D, "gap_bits": bpw - 0.5 * math.log2(1.0 / D)}


def main_ub():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="*", default=[2, 3, 4])
    ap.add_argument("--nu", type=int, nargs="*", default=[3, 6, 8])
    ap.add_argument("--h", type=int, nargs="*", default=[0])
    ap.add_argument("--n_masks", type=int, default=8)
    args = ap.parse_args()
    out = Path(__file__).resolve().parent.parent / "results" / "tcq_gaussian.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for nu in args.nu:
        # masks: enumeration (all for small nu, otherwise random), chosen at K=3 without a header on a small sample
        all_m = list(range(1, 1 << (nu - 1)))
        gen = torch.Generator().manual_seed(nu)
        masks = all_m if len(all_m) <= args.n_masks else [all_m[i] for i in torch.randperm(len(all_m), generator=gen)[: args.n_masks].tolist()]
        scores = []
        for m in masks:
            sm, _ = train_ub(3, nu, m, iters=3, n_tiles=1024)
            scores.append((gaussian_gap_ub(3, nu, m, sm, None, 0, n_tiles=1024)["gap_bits"], m))
        m_best = min(scores)[1]
        print(f"nu={nu}: masks {', '.join(f'{m}:{g:.3f}' for g, m in sorted(scores)[:4])} -> {m_best}", flush=True)
        for K in args.K:
            for h in args.h:
                best = None
                for step in ((0.3,) if h == 0 else (0.25, 0.4)):
                    sm, gains = train_ub(K, nu, m_best, h, step)
                    r = gaussian_gap_ub(K, nu, m_best, sm, gains, h)
                    if best is None or r["gap_bits"] < best["gap_bits"]:
                        best = dict(r, step=step, small_lut=sm.tolist())
                res[f"UB_K{K}_nu{nu}_h{h}"] = best
                out.write_text(json.dumps(res, indent=2))
                print(f"TCQ-Ungerboeck K={K} nu={nu} h={h}: table {1 << (K + 1)} | bpw {best['bpw']:.4f} | gap "
                      f"{best['gap_bits']:.3f} bits", flush=True)


if __name__ == "__main__":
    import sys
    if "--ub" in sys.argv:
        sys.argv.remove("--ub")
        main_ub()
    elif "--gain" in sys.argv:
        sys.argv.remove("--gain")
        main_gain()
    else:
        main()


# ---------------------------------------------------------------------------------------------------------
# CUDA Viterbi (src/cuda/tcq_viterbi.cu, built by nvcc into a .so and called via ctypes: torch extensions
# don't build — PyTorch is built for CUDA 12.8, the system has nvcc 13). Same algorithm and order of operations as viterbi_ub.
# ---------------------------------------------------------------------------------------------------------

_TCQ_LIB = None


def _tcq_lib():
    global _TCQ_LIB
    if _TCQ_LIB is None:
        import ctypes
        import subprocess
        import hashlib
        src = Path(__file__).with_name("cuda") / "tcq_viterbi.cu"
        # name includes the source hash: a rebuild doesn't overwrite a .so already loaded into running processes
        so = src.with_name(f"libtcq_viterbi_{hashlib.sha1(src.read_bytes()).hexdigest()[:10]}.so")
        if not so.exists():
            subprocess.run(["/usr/local/cuda/bin/nvcc", "-O3", "-arch=sm_120", "-shared", "-Xcompiler", "-fPIC",
                            "-o", str(so), str(src)], check=True)
        lib = ctypes.CDLL(str(so))
        lib.tcq_viterbi.restype = ctypes.c_int
        lib.tcq_viterbi.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int] * 18 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.lut_subs_quant.restype = ctypes.c_int
        lib.lut_subs_quant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                                                               ctypes.c_void_p]
        _TCQ_LIB = lib
    return _TCQ_LIB


@torch.no_grad()
def lut_subs_quant_cuda(x_subs: torch.Tensor, dh: torch.Tensor, lut_val: torch.Tensor, inv_step: float, half_b: float):
    """Like liqk_v2.quantize_subs_lut_chunked with 64 candidates: x_subs, dh [S, 16] -> [S, 16] (single kernel)."""
    assert lut_val.shape[1] == 64
    x = x_subs.float().contiguous()
    d = dh.float().reshape(x.shape).contiguous()
    out = torch.empty_like(x)
    rc = _tcq_lib().lut_subs_quant(x.data_ptr(), d.data_ptr(), lut_val.float().contiguous().data_ptr(), out.data_ptr(),
                                   x.shape[0], inv_step, half_b, lut_val.shape[0], torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"lut_subs_quant: CUDA error code {rc}")
    return out


@torch.no_grad()
def gain_candidates(x, w, tab, gains, n_cand, sub, tab_hi=None, gains_hi=None, mix=0) -> torch.Tensor:
    """[N, T / sub, n_cand] uint8: gains with the lowest weighted scalar-quantization error of the sub-block
    using the merged table (all 4 subsets, no trellis constraints)."""
    N, T = x.shape
    def err(tb, gn):  # [N, T, G]
        q = gn.view(-1, 1) * tb.view(1, -1)  # [G, L]
        return (x.unsqueeze(-1).unsqueeze(-1) - q).square().amin(dim=-1)
    e = err(tab, gains)
    if tab_hi is not None:
        hi = ((torch.arange(T, device=x.device) // mix) % 2 == 1).view(1, T, 1)
        e = torch.where(hi, err(tab_hi, gains_hi), e)
    e = (e * w.unsqueeze(-1)).view(N, T // sub, sub, -1).sum(dim=2)
    return e.topk(n_cand, dim=-1, largest=False).indices.to(torch.uint8).contiguous()


@torch.no_grad()
def viterbi_ub_cuda(x: torch.Tensor, small: torch.Tensor, K: int, nu: int, m: int, gains: torch.Tensor | None = None,
                    sub: int = 16, wts: torch.Tensor | None = None, init_state: torch.Tensor | None = None,
                    return_end_state: bool = False, small_hi: torch.Tensor | None = None, gains_hi: torch.Tensor | None = None,
                    mix: int = 0, n_cand: int = 0, return_code: bool = False, tail_biting: bool = False,
                    tb_passes: int = 2, tb_cand1: int = 0, tb_window: int = 0, m_extra: tuple = (), pair_parity: bool = False,
                    _mid: int = 0, _tail: bool = False, _steps: int = 0, _rot: int = 0):
    """Reconstruction like viterbi_ub(...)[0]; x [N, T] float32 on GPU. init_state [N] int32: initial state
    of the tile (carried over from the previous tile of the same rows, < 0 = free); return_end_state: also return final states.
    n_cand > 0: joint Viterbi over only n_cand gains per sub-block, selected by the merged-table error.
    tail_biting: the tile's path starts and ends in the same state (the decoder takes it from the last nu code bits
    of the tile, no side information). Two passes, as in QTIP: a free pass yields the path state in the middle of the tile;
    the tile is cyclically shifted so the middle becomes the boundary, and the second pass runs with start = end.
    m_extra: masks of extra label parities (0–3 of them) — 8 / 16 / 32 subsets instead of 4, table small [nl, U].
    pair_parity: 1 (True) — rate K - 0.5 (K = bit width of table small): in a pair (2i, 2i+1) the low bits of the non-code
    indices are equal (D2 lattice) — 2K - 1 bits per pair instead of 2K; 2 — rate K - 0.25: the sum of the low bits of a group
    (4i .. 4i+3) is even (D4 lattice); 3 — rate K - 0.75: the low bits of a group of 4 are equal."""
    if tail_biting:
        N, T = x.shape
        h = T // 2
        kw = dict(sub=sub, small_hi=small_hi, gains_hi=gains_hi, mix=mix, n_cand=n_cand, m_extra=m_extra,
                  pair_parity=pair_parity)
        w_full = None if wts is None else (wts.view(1, T).expand(N, T) if wts.dim() == 1 else wts).float().contiguous()
        x = x.float().contiguous()
        # the first pass is only needed for the middle state: tb_cand1 > 0 makes it cheap, over tb_cand1 gains;
        # tb_window restricts it to the window [h - tb_window, h + tb_window) (paths merge within a few nu steps).
        # The kernel takes the shift and window from _rot / _steps — no copies.
        kw1 = dict(kw, n_cand=tb_cand1) if tb_cand1 else kw
        if tb_window:
            _, mid = viterbi_ub_cuda(x, small, K, nu, m, gains, wts=w_full, _mid=tb_window, _steps=2 * tb_window,
                                     _rot=h - tb_window, **kw1)
        else:
            _, mid = viterbi_ub_cuda(x, small, K, nu, m, gains, wts=w_full, _mid=h, **kw1)
        shift = 0  # shift of the current sequence relative to the original
        for p in range(1, tb_passes):
            shift = (shift + h) % T
            if p + 1 < tb_passes:  # intermediate pass: tail-biting path and its middle state
                _, mid = viterbi_ub_cuda(x, small, K, nu, m, gains, wts=w_full, init_state=mid, _tail=True, _mid=h,
                                         _rot=shift, **kw)
            else:  # the kernel writes rec / code to the original positions
                out = viterbi_ub_cuda(x, small, K, nu, m, gains, wts=w_full, init_state=mid, return_code=return_code,
                                      _tail=True, _rot=shift, **kw)
        return out
    N, ld = x.shape  # ld = row stride; the kernel walks T = _steps (or ld) samples (t + _rot) % ld
    T = _steps or ld
    S, U = 1 << nu, 1 << (K - 1)
    dev = x.device
    x = x.float().contiguous()
    if wts is None:
        w = torch.ones_like(x)
    else:
        w = (wts.view(1, ld).expand(N, ld) if wts.dim() == 1 else wts).float().contiguous()
    g = (gains if gains is not None else torch.ones(1, device=dev)).float().contiguous()
    tab = small.float().contiguous()
    rec = torch.empty_like(x)
    G_run = n_cand if 0 < n_cand < g.numel() else g.numel()  # Viterbi runs over candidates: pointers only for them
    bp = torch.empty((N, T, G_run, S // 32), dtype=torch.int32, device=dev)
    gsel = torch.empty((N, max(T // sub, 1), S), dtype=torch.uint8, device=dev)
    # half-integer bitrate: positions t with (t // mix) % 2 == 1 use table small_hi (K + 1 bits) and gains gains_hi
    tab_hi = small_hi.float().contiguous() if small_hi is not None else tab
    g_hi = gains_hi.float().contiguous() if gains_hi is not None else g
    init = init_state.to(torch.int32).contiguous() if init_state is not None else None
    end = torch.empty(N, dtype=torch.int32, device=dev) if return_end_state else None
    Gfull, cand = g.numel(), None
    best = torch.empty(N, dtype=torch.int32, device=dev)
    code = torch.empty((N, ld), dtype=torch.int32, device=dev) if return_code else None
    mid = torch.empty(N, dtype=torch.int32, device=dev) if _mid else None
    if int(pair_parity) >= 2:  # groups of 4 (even sum / all equal): 9 bit planes per group
        pbits = torch.empty((N, T // 4, G_run, 9, S // 32), dtype=torch.int32, device=dev)
    else:
        pbits = torch.empty((N, T // 2, G_run, S // 32), dtype=torch.int32, device=dev) if pair_parity else None
    nl = 4 << len(m_extra)
    if 0 < n_cand < Gfull:
        cand = torch.empty((N, T // sub, n_cand), dtype=torch.uint8, device=dev)  # filled by the kernel
    rc = _tcq_lib().tcq_viterbi(x.data_ptr(), w.data_ptr(), tab.data_ptr(), g.data_ptr(), rec.data_ptr(), bp.data_ptr(),
                                gsel.data_ptr(), init.data_ptr() if init is not None else None,
                                end.data_ptr() if end is not None else None, tab_hi.data_ptr(), g_hi.data_ptr(),
                                cand.data_ptr() if cand is not None else None, best.data_ptr(),
                                code.data_ptr() if code is not None else None,
                                mid.data_ptr() if mid is not None else None, _mid, int(_tail),
                                N, T, S, U, n_cand if cand is not None else Gfull, Gfull, m, sub, tab_hi.numel() // nl,
                                mix if small_hi is not None else 0, *(list(m_extra) + [0, 0, 0])[:3], nl, ld, _rot,
                                pbits.data_ptr() if pbits is not None else None, int(pair_parity),
                                torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"tcq_viterbi: CUDA error code {rc}")
    if _mid:
        return rec, mid
    if return_code:  # (rec, gain index [N, T], level index in the table [N, T]); with n_cand the index is into Gfull
        return rec, code >> 16, code & 0xffff
    return (rec, end) if return_end_state else rec
