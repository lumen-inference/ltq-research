"""
Reference implementation of the LTQ v1 format (lumen-rs, docs/EN/ltq-format.md): 16x16 tile packing and decoding per the
spec, independent of the quantizer. Self-test (`python -m src.ltq_format`): for each format of the TCQPQ menu, quantizes
Gaussian tiles with CUDA Viterbi (tail-biting, extended labels, fractional modes), checks that the quantizer's labels match
the labels computed from the code bits and spec masks, that the fractional-mode constraints hold, packs the tile fields
and checks the decode against the quantizer's reconstruction bit for bit.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

NU = 10  # trellis memory length: state = last 10 code bits
TILE = 256
MASK_M = 359  # primary Ungerboeck label mask


@dataclass(frozen=True)
class LtqFormat:
    name: str
    K: int            # bits/weight before fractional mode: 1 coded + (K - 1) uncoded
    frac: int         # 0 = none, 1 = D2 (K - 0.5), 2 = D4 (K - 0.25)
    masks: tuple      # label masks over (s >> 1): m, then extra masks (8 / 16 subsets)
    h: int            # gain index bits
    step: float       # gain step (log2)
    sub: int          # weights per gain index: 16 or 32
    lut: np.ndarray   # float32 [n_labels * 2^(K-1)], label-major
    gain_tab: np.ndarray  # float32 [2^h]: gain values are stored explicitly (2^((g - (2^h - 1) / 2) * step) rounds differently
                          # on different devices; the decoder must use exactly the same numbers as the quantizer)

    @property
    def n_labels(self) -> int:
        return 2 ** len(self.masks) * 2

    @property
    def U(self) -> int:
        return 1 << (self.K - 1)

    def lsb_words(self, L: int = TILE) -> int:  # u16 words in the lowest uncoded bit plane
        return ({0: L, 1: L // 2, 2: 3 * L // 4}[self.frac] + 15) // 16

    def gain_words(self, L: int = TILE) -> int:
        return (L // self.sub * self.h + 15) // 16

    def tile_bytes(self, L: int = TILE) -> int:
        """Bytes per tile (L = 256) or per ring of length L (excluding the ring's fp16 scale)."""
        return 2 * (L // 16 + L // 16 * (self.K - 2) + (self.lsb_words(L) if self.K > 1 else 0) + self.gain_words(L))

    def bpw(self, k_in: int) -> float:
        """Bits/weight for a matrix of width k_in: tiles + scale exponent (int8 per row and 256-block) + d_row (fp16 per row)."""
        return self.tile_bytes() * 8 / TILE + 8 / 256 + 16 / k_in


def load_formats(menu: list[str]) -> dict[str, LtqFormat]:
    import src.liqk_v2 as lv2
    from src.tcq_proto import make_gains
    out = {}
    for name in menu:
        c = lv2.TCQ_CONFIG[name]
        out[name] = LtqFormat(name, c["K"], int(c.get("pp", 0)), (c["m"],) + tuple(c.get("m_extra", ())), c["h"],
                              float(c["step"]), c.get("sub", 16), np.asarray(c["lut"], dtype=np.float32),
                              make_gains(c["h"], c["step"], "cuda").cpu().numpy())
    return out


def parity(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    for s in (16, 8, 4, 2, 1):
        x ^= x >> s
    return x & 1


def states_from_codes(c: np.ndarray) -> np.ndarray:
    """c [L] code bits of a tile (L = 256, position t = 16 r + col) or of a row ring -> states:
    s_t = sum_j c[(t - j) mod L] << j, j = 0..9 (tail-biting: the start continues the end)."""
    L = len(c)
    t = np.arange(L)
    s = np.zeros(L, dtype=np.uint32)
    for j in range(NU):
        s |= c[(t - j) % L].astype(np.uint32) << j
    return s


def labels_from_states(s: np.ndarray, fmt: LtqFormat) -> np.ndarray:
    lb = (s & 1) | (parity((s >> 1) & fmt.masks[0]) << 1)
    for i, m in enumerate(fmt.masks[1:]):
        lb |= parity((s >> 1) & m) << (2 + i)
    return lb


def _bits_to_words(bits: np.ndarray, n_words: int) -> np.ndarray:
    out = np.zeros(n_words * 16, dtype=np.uint16)
    out[: len(bits)] = bits
    return (out.reshape(n_words, 16).astype(np.uint32) << np.arange(16, dtype=np.uint32)).sum(axis=1).astype(np.uint16)


def _words_to_bits(words: np.ndarray, n_bits: int) -> np.ndarray:
    return ((words.astype(np.uint32)[:, None] >> np.arange(16, dtype=np.uint32)) & 1).reshape(-1)[:n_bits].astype(np.uint32)


def pack_tile(labels: np.ndarray, u: np.ndarray, gidx: np.ndarray, fmt: LtqFormat) -> dict:
    """Tile fields: codes [16] u16 (word r = tile row, bit col = position 16 r + col), planes [K - 2, 16] u16
    (uncoded bits 1..K-2, same layout), lsb [16 | 8 | 12] u16 (lowest uncoded bit: all 256 / one bit per pair /
    3 bits per quad), gains [ceil(256 / sub * h / 16)] u16 (gain indices back to back, LSB first).
    For a row ring of length L, the same with L instead of 256."""
    L = len(labels)
    W = L // 16
    codes = _bits_to_words(labels & 1, W)
    planes = np.stack([_bits_to_words((u >> p) & 1, W) for p in range(1, fmt.K - 1)]) if fmt.K > 2 else np.zeros((0, W), np.uint16)
    lsb_all = u & 1
    if fmt.frac == 0:
        lsb = _bits_to_words(lsb_all, fmt.lsb_words(L))
    elif fmt.frac == 1:
        lsb = _bits_to_words(lsb_all[0::2], fmt.lsb_words(L))
    else:
        lsb = _bits_to_words(lsb_all.reshape(-1, 4)[:, :3].reshape(-1), fmt.lsb_words(L))
    g = gidx[:: fmt.sub].astype(np.uint32)
    gbits = ((g[:, None] >> np.arange(fmt.h, dtype=np.uint32)) & 1).reshape(-1) if fmt.h else np.zeros(0, np.uint32)
    return {"L": L, "codes": codes, "planes": planes, "lsb": lsb,
            "gains": _bits_to_words(gbits, fmt.gain_words(L)) if fmt.h else np.zeros(0, np.uint16)}


def decode_tile(f: dict, fmt: LtqFormat) -> np.ndarray:
    """Unscaled values of a tile (L = 256) or row ring: gain[g] * lut[label * U + u], [L] float32."""
    L = f.get("L", TILE)
    c = _words_to_bits(f["codes"], L)
    lb = labels_from_states(states_from_codes(c), fmt)
    if fmt.frac == 0:
        lsb = _words_to_bits(f["lsb"], L)
    elif fmt.frac == 1:
        lsb = np.repeat(_words_to_bits(f["lsb"], L // 2), 2)
    else:
        b3 = _words_to_bits(f["lsb"], 3 * L // 4).reshape(-1, 3)
        lsb = np.concatenate([b3, (b3.sum(axis=1, keepdims=True) & 1)], axis=1).reshape(-1)  # even sum over the quad
    u = lsb.copy()
    for p in range(1, fmt.K - 1):
        u |= _words_to_bits(f["planes"][p - 1], L) << p
    if fmt.h:
        gb = _words_to_bits(f["gains"], L // fmt.sub * fmt.h).reshape(-1, fmt.h)
        g = (gb << np.arange(fmt.h, dtype=np.uint32)).sum(axis=1)
        gain = fmt.gain_tab[np.repeat(g, fmt.sub)]
    else:
        gain = np.ones(L, dtype=np.float32)
    return (gain * fmt.lut[lb * fmt.U + u]).astype(np.float32)


def self_test(n_tiles: int = 512, seed: int = 0, L: int = TILE):
    from src.equal_bpw import SETS
    import src.liqk_v2 as lv2
    from src.tcq_proto import viterbi_ub_cuda
    torch.manual_seed(seed)
    fmts = load_formats(SETS["TCQPQ"])
    res = {}
    for name, fmt in fmts.items():
        if L != TILE and fmt.sub != 16:  # row rings: only formats with a gain per 16 values (spec, section 9)
            continue
        x = torch.randn(n_tiles, L, device="cuda")
        gains = torch.tensor(fmt.gain_tab, device="cuda")
        rec, gi, idx = viterbi_ub_cuda(x, torch.tensor(fmt.lut, device="cuda"), fmt.K, NU, fmt.masks[0], gains, sub=fmt.sub,
                                       return_code=True, tail_biting=True, m_extra=fmt.masks[1:], pair_parity=fmt.frac,
                                       n_cand=lv2.TCQ_CONFIG[name].get("n_cand", 0))  # as in the pipeline (h > 2: 4 candidates)
        rec, gi, idx = rec.cpu().numpy(), gi.cpu().numpy().astype(np.uint32), idx.cpu().numpy().astype(np.uint32)
        bad_lab = bad_frac = bad_dec = 0
        for n in range(n_tiles):
            lab, u = idx[n] // fmt.U, idx[n] % fmt.U
            if not np.array_equal(labels_from_states(states_from_codes(lab & 1), fmt), lab):
                bad_lab += 1
            if fmt.frac == 1 and np.any((u[0::2] & 1) != (u[1::2] & 1)):
                bad_frac += 1
            if fmt.frac == 2 and np.any((u.reshape(-1, 4) & 1).sum(axis=1) & 1):
                bad_frac += 1
            if not np.array_equal(decode_tile(pack_tile(lab, u, gi[n], fmt), fmt), rec[n]):
                bad_dec += 1
        res[name] = {"tile_bytes": fmt.tile_bytes(L), "bpw_k4096": round(fmt.bpw(4096), 4), "label_mismatch": bad_lab,
                     "frac_violations": bad_frac, "decode_mismatch": bad_dec}
        print(f"{name:9s} L={L} {fmt.tile_bytes(L):3d} B, {fmt.bpw(4096):.4f} bits/weight (K_in 4096) | mismatches: labels {bad_lab}, "
              f"fractional mode {bad_frac}, decode {bad_dec} of {n_tiles}", flush=True)
    return res


if __name__ == "__main__":
    r = {"tiles_256": self_test(), "rings_160": self_test(n_tiles=256, L=160)}  # 160 = n-gram row of Qwen3.8-Flash-Next
    Path(__file__).resolve().parent.parent.joinpath("results", "ltq_format_selftest.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
