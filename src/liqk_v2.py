"""
Implementation of the second-generation quantization format and algorithm `.liqk v2` (Lumen-RS Quantization v2),
free of the structural constraints of `.gguf` headers.

The four key mathematical components of `.liqk v2`:
1. A universal family of 16 `Warp-Register VQ Bank` formats (`L = 4, 5, 6, 8, 10, 11, 16, 22, 32`)
   with `LUT-Decoupled Block16-LDLQ`:
   - Replaces the redundant low scales `|ls| in {1..9}` (used 0.0% of the time after the
     Walsh-Hadamard rotation FWHT-256) with a K-means-trained, sign-canonicalized
     bank of signed `int8` grids `ls_vec[s] * cb_int8[s, 0..L-1]` specialized to the 16-dimensional
     sub-block statistics (amplitude, peak asymmetry and tail kurtosis).
   - Instead of destructive bit masks (`LIQK3_0_V2` / the old `LIQK3_5_V2`, which dropped half of
     a superblock to 2 or 3 bits) it uses compact state headers (`hdr = 3, 4, 5, 6, 7` bits:
     `LIQK3_25_V2`, `LIQK3_31_V2`, `LIQK3_38_V2`, `LIQK4_31_V2`, `LIQK4_38_V2`) and native
     fractional-bit radix-`L` lattices (`LIQK2_5_V2` `L=5` `5^2<=32`, `LIQK3_3_V2` `L=10` `10^3<=1024`,
     `LIQK3_5_V2` `L=11` `11^2<=128`, `LIQK4_5_V2` `L=22` `22^2<=512`).
   - Precomputing the 1D table `lut_val[8192, C_all]` decouples the search for the optimal state
     `O(M * 16 * C_all)` from level quantization `O(M * 16 * L)`, speeding up matrix quantization **15–40x**!
   - Decodes on GPU in 1 cycle via intra-warp register exchange (`__shfl_sync`) +
     `__byte_perm` + `__dp4a`; on CPU (AVX-512) via `vpermb` in ZMM registers.

2. Cauchy-Schwarz scaling `suh` + fast block Cholesky `Block16-LDLQ` + joint
   global refinement of the output scales `c_row` and the input vector `suh[K]`:
   - Before the Hadamard transform, input channels are scaled by the analytic Cauchy-Schwarz optimum
     with Fisher row weights: `suh_k = sign_k * col_rms_k^0.5 * (H_kk^0.5)^-0.25`.
   - The block Cholesky pass `Block16-LDLQ` with precomputed `U_blks_inv` and analytic
     Schur-complement recomputation of `d` takes ~0.04–0.35 s per matrix.

3. Exact bitrate allocation over sub-matrices (Row-Slice Knapsack DP) based on the diagonal
   Fisher information of the final KLD + balanced intra-slice permutation of SwiGLU MLP channels:
   - In 1.2 s on the calibration sample, the exact per-row Fisher sensitivity is computed
     `s_out[l, m, i] = E[(d log p(y|x) / d z_{l,m,i})^2]`.
   - Intermediate MLP channels (`gate_proj`, `up_proj`, `down_proj`) are sorted by the Fisher
     sensitivity of `gate_proj/up_proj` into 512-row slices, interleaving pairs of 256-channel superblocks
     within each slice (`perm_bal2`), which preserves slice differentiation for `gate/up` and
     prevents outliers from concentrating in `down_proj`.

4. Post-quantization calibration polish of the continuous `fp16` scales (`d`, `suh`) with
   the combined loss `Forward KL + 0.12 * Reverse KL + 0.08 * Cross-Entropy`:
   - With `RMSNorm` weights and discrete indices frozen, in ~10–15 s it lowers both
     teacher-forced KLD and perplexity (PPL), baking the corrections directly into the `fp16` fields `d` and `suh`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from src.iqk_format import (
    NUM_SUBBLOCKS,
    QK_K,
    SUBBLOCK_SIZE,
    fwht_inplace,
    make_rademacher_signs,
)
from src.quantize_model_iqk import LINEAR_SUBMODULES

N_LUT_BUCKETS = 8192
# Packing mode for .ltq (src/ltq_file.py): if not None, the TCQ quantizer stores the matrix's tile codes, d_row, e and suh,
# and rebuilds the reconstructed matrix from the same fp16 fields that go into the file (PACK["matrix"] after each matrix)
PACK: Optional[dict] = None


@dataclass(frozen=True)
class LiqkV2Spec:
    name: str
    bits: float
    n_levels: int
    n_pos_states: int  # number of positive states (total number of states = 2 * n_pos_states)
    block_bytes: int
    is_half_bit: bool = False
    low_spec_name: Optional[str] = None
    high_spec_name: Optional[str] = None

    @property
    def bpw(self) -> float:
        return (self.block_bytes * 8.0) / QK_K


LIQK_V2_SPECS: Dict[str, LiqkV2Spec] = {
    # Ultra-low-bit ternary family (L = 3, 3^5 <= 256, 1.60 bits/weight + 3..6-bit headers)
    "LIQK1_88_V2": LiqkV2Spec(name="LIQK1_88_V2", bits=1.58, n_levels=3, n_pos_states=8,  block_bytes=60),  # 1.8750 bpw
    "LIQK2_00_V2": LiqkV2Spec(name="LIQK2_00_V2", bits=1.58, n_levels=3, n_pos_states=32, block_bytes=64),  # 2.0000 bpw
    "LIQK2_06_V2": LiqkV2Spec(name="LIQK2_06_V2", bits=1.65, n_levels=3, n_pos_states=48, block_bytes=66),  # 2.0625 bpw
    # 2-bit family (L = 4, 2 bits/weight + 3, 4, 5-bit headers per sub-block)
    "LIQK2_25_V2": LiqkV2Spec(name="LIQK2_25_V2", bits=2.0, n_levels=4, n_pos_states=4, block_bytes=72),   # 2.2500 bpw
    "LIQK2_0_V2":  LiqkV2Spec(name="LIQK2_0_V2",  bits=2.0, n_levels=4, n_pos_states=8, block_bytes=74),   # 2.3125 bpw
    "LIQK2_V2":    LiqkV2Spec(name="LIQK2_V2",    bits=2.0, n_levels=4, n_pos_states=16, block_bytes=76),  # 2.3750 bpw
    # Fractional-bit 2.32-bit and 2.58-bit formats (L = 5, 5^2 <= 32 and L = 6)
    "LIQK2_5_V2":  LiqkV2Spec(name="LIQK2_5_V2",  bits=2.32, n_levels=5, n_pos_states=16, block_bytes=92), # 2.8750 bpw
    "LIQK2_7_V2":  LiqkV2Spec(name="LIQK2_7_V2",  bits=2.58, n_levels=6, n_pos_states=16, block_bytes=98), # 3.0625 bpw
    # 3-bit family (L = 8, 3 bits/weight + 3, 4, 5, 6-bit headers per sub-block; 0% 2-bit weights!)
    "LIQK3_25_V2": LiqkV2Spec(name="LIQK3_25_V2", bits=3.0, n_levels=8, n_pos_states=4, block_bytes=104),  # 3.2500 bpw
    "LIQK3_31_V2": LiqkV2Spec(name="LIQK3_31_V2", bits=3.0, n_levels=8, n_pos_states=8, block_bytes=106),  # 3.3125 bpw
    "LIQK3_38_V2": LiqkV2Spec(name="LIQK3_38_V2", bits=3.0, n_levels=8, n_pos_states=16, block_bytes=108), # 3.3750 bpw
    "LIQK3_V2":    LiqkV2Spec(name="LIQK3_V2",    bits=3.0, n_levels=8, n_pos_states=32, block_bytes=110), # 3.4375 bpw
    # Native fractional-bit 3.32-bit and 3.46-bit formats (L = 10, 10^3 <= 1024 and L = 11, 11^2 <= 128)
    "LIQK3_3_V2":  LiqkV2Spec(name="LIQK3_3_V2",  bits=3.32, n_levels=10, n_pos_states=16, block_bytes=119), # 3.71875 bpw
    "LIQK3_5_V2":  LiqkV2Spec(name="LIQK3_5_V2",  bits=3.46, n_levels=11, n_pos_states=32, block_bytes=126), # 3.9375 bpw
    # 4-bit family (L = 16, 4 bits/weight + 4, 5, 7-bit headers per sub-block; 0% of weights below 4 bits!)
    "LIQK4_31_V2": LiqkV2Spec(name="LIQK4_31_V2", bits=4.0, n_levels=16, n_pos_states=8, block_bytes=138),  # 4.3125 bpw
    "LIQK4_38_V2": LiqkV2Spec(name="LIQK4_38_V2", bits=4.0, n_levels=16, n_pos_states=16, block_bytes=140), # 4.3750 bpw
    "LIQK4_V2":    LiqkV2Spec(name="LIQK4_V2",    bits=4.0, n_levels=16, n_pos_states=64, block_bytes=144), # 4.5000 bpw
    # Native 4.46-bit format (L = 22, 22^2 <= 512) and 5..8-bit formats (L = 32, 45, 64, 128, 180, 255)
    "LIQK4_5_V2":  LiqkV2Spec(name="LIQK4_5_V2",  bits=4.46, n_levels=22,  n_pos_states=32, block_bytes=158), # 4.9375 bpw
    "LIQK5_31_V2": LiqkV2Spec(name="LIQK5_31_V2", bits=5.0,  n_levels=32,  n_pos_states=8,  block_bytes=170), # 5.3125 bpw
    "LIQK5_38_V2": LiqkV2Spec(name="LIQK5_38_V2", bits=5.0,  n_levels=32,  n_pos_states=16, block_bytes=172), # 5.3750 bpw
    "LIQK5_V2":    LiqkV2Spec(name="LIQK5_V2",    bits=5.0,  n_levels=32,  n_pos_states=64, block_bytes=176), # 5.5000 bpw
    "LIQK5_5_V2":  LiqkV2Spec(name="LIQK5_5_V2",  bits=5.49, n_levels=45,  n_pos_states=32, block_bytes=190), # 5.9375 bpw
    "LIQK6_31_V2": LiqkV2Spec(name="LIQK6_31_V2", bits=6.0,  n_levels=64,  n_pos_states=8,  block_bytes=202), # 6.3125 bpw
    "LIQK6_V2":    LiqkV2Spec(name="LIQK6_V2",    bits=6.0,  n_levels=64,  n_pos_states=64, block_bytes=212), # 6.6250 bpw
    "LIQK7_V2":    LiqkV2Spec(name="LIQK7_V2",    bits=7.0,  n_levels=128, n_pos_states=32, block_bytes=240), # 7.5000 bpw
    "LIQK8_00_V2": LiqkV2Spec(name="LIQK8_00_V2", bits=7.49, n_levels=180, n_pos_states=32, block_bytes=256), # 8.0000 bpw
    "LIQK8_V2":    LiqkV2Spec(name="LIQK8_V2",    bits=8.0,  n_levels=255, n_pos_states=32, block_bytes=272), # 8.5000 bpw
    # Experimental light-header formats: one codebook per block, only 1 mirror-sign bit per
    # sub-block. Negative result (proxy, 1B): 0.15..0.26 bits worse than v2; bank headers pay for themselves
    "LIQK2_HL_V2": LiqkV2Spec(name="LIQK2_HL_V2", bits=2.0, n_levels=4,  n_pos_states=1, block_bytes=68),  # 2.1250 bpw
    "LIQK3_HL_V2": LiqkV2Spec(name="LIQK3_HL_V2", bits=3.0, n_levels=8,  n_pos_states=1, block_bytes=100), # 3.1250 bpw
    "LIQK4_HL_V2": LiqkV2Spec(name="LIQK4_HL_V2", bits=4.0, n_levels=16, n_pos_states=1, block_bytes=132), # 4.1250 bpw
    # Experimental formats with richer banks (testing the "more states is better" trend)
    "LIQK2_S64_V2":  LiqkV2Spec(name="LIQK2_S64_V2",  bits=2.0, n_levels=4,  n_pos_states=64,  block_bytes=80),  # 2.5000 bpw
    "LIQK3_S128_V2": LiqkV2Spec(name="LIQK3_S128_V2", bits=3.0, n_levels=8,  n_pos_states=128, block_bytes=114), # 3.5625 bpw
    "LIQK3_S256_V2": LiqkV2Spec(name="LIQK3_S256_V2", bits=3.0, n_levels=8,  n_pos_states=256, block_bytes=116), # 3.6250 bpw
    "LIQK4_S256_V2": LiqkV2Spec(name="LIQK4_S256_V2", bits=4.0, n_levels=16, n_pos_states=256, block_bytes=148), # 4.6250 bpw
}

# Convex relative distortion ratios of the formats w.r.t. the baseline LIQK3_V2 = 1.000
SPEC_DISTORTION_RATIO: Dict[str, float] = {
    "LIQK1_88_V2": 9.850,
    "LIQK2_00_V2": 8.450,
    "LIQK2_06_V2": 8.150,
    "LIQK2_25_V2": 5.950,
    "LIQK2_0_V2":  5.250,
    "LIQK2_V2":    4.880,
    "LIQK2_5_V2":  2.970,
    "LIQK2_7_V2":  1.990,
    "LIQK3_25_V2": 1.385,
    "LIQK3_31_V2": 1.215,
    "LIQK3_38_V2": 1.091,
    "LIQK3_V2":    1.000,
    "LIQK3_3_V2":  0.710,
    "LIQK3_5_V2":  0.544,
    "LIQK4_31_V2": 0.365,
    "LIQK4_38_V2": 0.320,
    "LIQK4_V2":    0.242,
    "LIQK4_5_V2":  0.155,
    "LIQK5_31_V2": 0.105,
    "LIQK5_38_V2": 0.088,
    "LIQK5_V2":    0.065,
    "LIQK5_5_V2":  0.040,
    "LIQK6_31_V2": 0.028,
    "LIQK6_V2":    0.020,
    "LIQK7_V2":    0.006,
    "LIQK8_00_V2": 0.0025,
    "LIQK8_V2":    0.0012,
}


def _hl_index_bits(n_levels: int) -> float:
    """Bits/weight with the best radix packing of indices in groups of up to 8 weights."""
    return min(math.ceil(g * math.log2(n_levels) - 1e-9) / g for g in range(1, 9))


def _hl_block_bytes(n_levels: int) -> int:
    g = min(range(1, 9), key=lambda g: math.ceil(g * math.log2(n_levels) - 1e-9) / g)
    idx_bits = math.ceil(QK_K / g) * math.ceil(g * math.log2(n_levels) - 1e-9)
    return math.ceil(idx_bits / 8) + 2 + NUM_SUBBLOCKS // 8  # indices + d fp16 + 1 sign bit per sub-block


# HL family (light header): one codebook per block, 1 mirror-sign bit per sub-block.
# Negative result (proxy, 1B): radix formats HL5 / HL11 / HL22 are 0.21..0.26 bits worse than v2.
# Relative distortion for DP: log2 interpolation over index bits through the HL2/HL3/HL4 points
# measured by the proxy rig on 1B (9.77 / 2.07 / 0.505 relative to LIQK3_V2), with the end slopes extrapolated.
_HL_LEVELS = [3, 4, 5, 6, 8, 10, 11, 12, 16, 20, 22, 24, 32]
_HL_RATIO_POINTS = [(2.0, math.log2(9.77)), (3.0, math.log2(2.07)), (4.0, math.log2(0.505))]


def _hl_ratio(idx_bits: float) -> float:
    pts = _HL_RATIO_POINTS
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if idx_bits <= x1 or (x1, y1) == pts[-1]:
            return 2.0 ** (y0 + (idx_bits - x0) * (y1 - y0) / (x1 - x0))
    return 2.0 ** pts[-1][1]


for _L in _HL_LEVELS:
    _name = f"LIQK_HL{_L}"
    LIQK_V2_SPECS[_name] = LiqkV2Spec(
        name=_name, bits=_hl_index_bits(_L), n_levels=_L, n_pos_states=1, block_bytes=_hl_block_bytes(_L)
    )
    SPEC_DISTORTION_RATIO[_name] = _hl_ratio(_hl_index_bits(_L))
HL_SPECS = [f"LIQK_HL{_L}" for _L in _HL_LEVELS]

class _CovAccumulator:
    """
    Accumulates H = (1/N) sum X^T X directly on the batch's device without storing X itself.
    For bf16/fp16 inputs X^T X is computed by a half-precision GEMM with fp32 output (~2.8x faster than fp32 GEMM
    on RTX 5070 Ti, relative error of H ~5e-5 vs damping of 2e-2); batches are summed in fp64.
    """

    def __init__(self):
        self.H: Optional[torch.Tensor] = None
        self.n = 0

    def add(self, x: torch.Tensor):
        x_2d = x.detach().reshape(-1, x.shape[-1])
        if x_2d.dtype in (torch.bfloat16, torch.float16):
            gram = torch.mm(x_2d.T, x_2d, out_dtype=torch.float32)
        else:
            x_2d = x_2d.float()
            gram = x_2d.T @ x_2d
        if self.H is None:
            self.H = torch.zeros((x_2d.shape[1],) * 2, dtype=torch.float64, device=x_2d.device)
        self.H.add_(gram.to(torch.float64))
        self.n += x_2d.shape[0]

    def get(self) -> torch.Tensor:
        return (self.H / max(self.n, 1)).to(torch.float32)


# Global per-device cache of trained register grid banks
_WARP_BANK_CACHE: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}

# Precomputed banks of all formats (part of the format spec): `export_warp_register_banks()`
WARP_BANK_FILE = Path(__file__).resolve().parent.parent / "data" / "liqk_v2_banks.safetensors"
_WARP_BANK_FILE_TENSORS: Optional[Dict[str, torch.Tensor]] = None


@torch.no_grad()
def _train_warp_register_bank(
    spec_name: str, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Trains (~1 s per format) the universal Warp-Register VQ Bank for format `spec_name`
    on the reference Gaussian ensemble after FWHT-256. Returns `(grids_all, ls_all, cb_all, max_grid)`.
    """
    spec = LIQK_V2_SPECS[spec_name]
    n_levels = spec.n_levels
    n_pos_states = spec.n_pos_states

    if n_levels > 64:
        # For 7..8-bit formats (L = 128, 180, 255), an analytic dense scale bank
        # [0.22..1.005] with soft companding p=1.01 eliminates codebook integer-rounding
        # error and initializes in 0.002 s (MSE = 0.000018 at 8 bits)
        C_all = 2 * n_pos_states
        ls_vec = torch.linspace(0.22, 1.005, n_pos_states, device=device, dtype=torch.float32) * 63.0
        ls_all = torch.cat([-ls_vec, ls_vec], dim=0)
        t_unif = torch.linspace(-1.0, 1.0, n_levels, device=device, dtype=torch.float32).unsqueeze(0)
        t_comp = (t_unif.sign() * (t_unif.abs() ** 1.01)) * 127.0
        cb_all = t_comp.expand(C_all, n_levels).contiguous()
        grids_all = ls_all.abs().unsqueeze(1) * cb_all
        max_g = float(grids_all.abs().max().item()) / 1.005
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(42 + n_levels * 100 + n_pos_states)
        n_train_rows = 2048 if n_levels >= 32 else 8192
        max_cb = 1023.0 if n_levels >= 32 else 127.0
        X_train = torch.randn((n_train_rows, QK_K), generator=gen, device=device, dtype=torch.float32)
        d_row = (X_train.abs().amax(dim=1, keepdim=True) / (max_cb * 63.0)).clamp(min=1e-6)
        X_sub = (X_train / d_row).reshape(-1, SUBBLOCK_SIZE)

        max_pos = X_sub.amax(dim=1, keepdim=True)
        min_neg = X_sub.amin(dim=1, keepdim=True)
        flip = (min_neg.abs() > max_pos).float() * -2.0 + 1.0
        X_sub_canon = X_sub * flip
        B_sub = X_sub_canon.shape[0]

        feat = torch.stack(
            [
                X_sub_canon.amax(dim=1),
                X_sub_canon.amin(dim=1),
                X_sub_canon.std(dim=1),
                X_sub_canon.mean(dim=1),
            ],
            dim=1,
        )
        perm = torch.randperm(B_sub, generator=gen, device=device)[:n_pos_states]
        centers = feat[perm].clone()
        for _ in range(15):
            assign = torch.cdist(feat, centers).argmin(dim=1)
            for s in range(n_pos_states):
                m = assign == s
                if m.any():
                    centers[s] = feat[m].mean(dim=0)

        grids_pos = torch.empty((n_pos_states, n_levels), device=device, dtype=torch.float32)
        probs = torch.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels, device=device)
        clip_q = 0.0005 if n_levels >= 32 else (0.005 if n_levels >= 10 else 0.015)
        for s in range(n_pos_states):
            m = assign == s
            pts = X_sub_canon[m].reshape(-1) if m.any() else X_sub_canon.reshape(-1)
            grids_pos[s] = torch.quantile(pts, probs)
            grids_pos[s, 0] = torch.quantile(pts, torch.tensor(clip_q, device=device))
            grids_pos[s, -1] = torch.quantile(pts, torch.tensor(1.0 - clip_q, device=device))

        n_iters = 15 if n_levels >= 32 else 25
        chunk_k = 512 if n_levels >= 32 else 1024
        for _ in range(n_iters):
            ls_vec = (grids_pos.abs().amax(dim=1) / max_cb).clamp(min=1.0).round().clamp(1, 63)
            cb_int8 = (grids_pos / ls_vec.unsqueeze(1)).round().clamp(-max_cb, max_cb)
            grids_q = ls_vec.unsqueeze(1) * cb_int8
            grids_all = torch.cat([-grids_q.flip(1), grids_q], dim=0)

            best_c_all = torch.empty((B_sub,), dtype=torch.long, device=device)
            best_q_all = torch.empty((B_sub, SUBBLOCK_SIZE), dtype=torch.long, device=device)
            for i0 in range(0, B_sub, chunk_k):
                xs = X_sub[i0 : i0 + chunk_k]
                diff = xs[:, None, :, None] - grids_all[None, :, None, :]
                err2, q_idx = (diff * diff).min(dim=-1)
                best_c = err2.sum(dim=-1).argmin(dim=-1)
                ar = torch.arange(xs.shape[0], device=device)
                best_c_all[i0 : i0 + chunk_k] = best_c
                best_q_all[i0 : i0 + chunk_k] = q_idx[ar, best_c, :]

            for s in range(n_pos_states):
                m_pos = best_c_all == (s + n_pos_states)
                m_neg = best_c_all == s
                xs_p = X_sub[m_pos]
                qp = best_q_all[m_pos]
                xs_n = -X_sub[m_neg]
                qn = (n_levels - 1) - best_q_all[m_neg]
                xs_comb = torch.cat([xs_p, xs_n], dim=0)
                q_comb = torch.cat([qp, qn], dim=0)
                if xs_comb.numel() > 0:
                    for l in range(n_levels):
                        ml = q_comb == l
                        if ml.any():
                            grids_pos[s, l] = xs_comb[ml].mean()
            grids_pos, _ = torch.sort(grids_pos, dim=1)

        ls_vec = (grids_pos.abs().amax(dim=1) / max_cb).clamp(min=1.0).round().clamp(1, 63)
        cb_int8 = (grids_pos / ls_vec.unsqueeze(1)).round().clamp(-max_cb, max_cb)
        grids_q = ls_vec.unsqueeze(1) * cb_int8
        grids_all = torch.cat([-grids_q.flip(1), grids_q], dim=0)
        ls_all = torch.cat([-ls_vec, ls_vec], dim=0)
        cb_all = torch.cat([-cb_int8.flip(1), cb_int8], dim=0)
        max_g = float(grids_all.abs().max().item())
    return grids_all, ls_all, cb_all, max_g


def _stored_warp_register_bank(
    spec_name: str, device: torch.device
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:
    global _WARP_BANK_FILE_TENSORS
    if _WARP_BANK_FILE_TENSORS is None:
        _WARP_BANK_FILE_TENSORS = load_file(str(WARP_BANK_FILE)) if WARP_BANK_FILE.exists() else {}
    if f"{spec_name}.grids_all" not in _WARP_BANK_FILE_TENSORS:
        return None
    t = {k: _WARP_BANK_FILE_TENSORS[f"{spec_name}.{k}"] for k in ("grids_all", "ls_all", "cb_all", "max_grid")}
    return t["grids_all"].to(device), t["ls_all"].to(device), t["cb_all"].to(device), float(t["max_grid"].item())


def export_warp_register_banks(path: Path = WARP_BANK_FILE, device: torch.device = torch.device("cuda")):
    """Trains the banks of all `LIQK_V2_SPECS` formats and saves them to `path`."""
    tensors: Dict[str, torch.Tensor] = {}
    for spec_name in LIQK_V2_SPECS:
        grids_all, ls_all, cb_all, max_g = _train_warp_register_bank(spec_name, device)
        tensors[f"{spec_name}.grids_all"] = grids_all.cpu().contiguous()
        tensors[f"{spec_name}.ls_all"] = ls_all.cpu().contiguous()
        tensors[f"{spec_name}.cb_all"] = cb_all.cpu().contiguous()
        tensors[f"{spec_name}.max_grid"] = torch.tensor(max_g, dtype=torch.float64)
    save_file(tensors, str(path))


@torch.no_grad()
def get_warp_register_bank(spec_name: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Returns the Warp-Register VQ Bank of format `spec_name` from the cache or `WARP_BANK_FILE`, or trains it,
    including the 1D fast lookup table `lut_val[8192, C_all]`.
    """
    key = (spec_name, str(device))
    if key in _WARP_BANK_CACHE:
        return _WARP_BANK_CACHE[key]

    stored = _stored_warp_register_bank(spec_name, device)
    if stored is not None:
        grids_all, ls_all, cb_all, max_g = stored
    else:
        grids_all, ls_all, cb_all, max_g = _train_warp_register_bank(spec_name, device)

    z = torch.linspace(-max_g, max_g, N_LUT_BUCKETS, device=device, dtype=torch.float32)
    q_idx = (z[:, None, None] - grids_all[None, :, :]).abs().argmin(dim=-1)
    lut_val = torch.gather(grids_all.unsqueeze(0).expand(N_LUT_BUCKETS, -1, -1), 2, q_idx.unsqueeze(-1)).squeeze(-1).contiguous()

    bank = {
        "grids_all": grids_all,
        "ls_all": ls_all,
        "cb_all": cb_all,
        "lut_val": lut_val,
        "max_grid": max_g,
    }
    _WARP_BANK_CACHE[key] = bank
    return bank


@torch.no_grad()
def compute_optimal_suh(
    W: torch.Tensor,
    H: torch.Tensor,
    seed: int = 1337,
    row_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Cauchy-Schwarz-optimal (analytic) preconditioning input vector `suh[K]`
    applied before the block Walsh-Hadamard transform (FWHT-256), taking Fisher row weights `row_weights` into account.
    """
    device = W.device
    M, K = W.shape
    signs = make_rademacher_signs(K, seed=seed, device=device)

    if row_weights is not None:
        rw = row_weights.float().clamp(min=float(row_weights.mean().item()) * 0.1)
        rw = (rw / rw.mean().clamp(min=1e-12)).clamp(0.25, 4.0)
        col_rms = ((W.float().square() * rw.unsqueeze(1)).mean(dim=0)).sqrt().clamp(min=1e-6)
    else:
        col_rms = W.float().square().mean(dim=0).sqrt().clamp(min=1e-6)
    col_rms = col_rms / col_rms.mean().clamp(min=1e-6)

    h_std = torch.diagonal(H.float()).clamp(min=1e-8).sqrt()
    h_std = h_std / h_std.mean().clamp(min=1e-6)

    scale = ((col_rms ** 0.5) * (h_std ** -0.25)).clamp(0.25, 4.0)
    suh = (signs * scale).to(torch.float16).to(torch.float32)

    W_rot = fwht_inplace(W.float() / suh.unsqueeze(0), block_size=QK_K)
    H_tmp = fwht_inplace(H.float() * suh.unsqueeze(0), block_size=QK_K)
    H_rot = fwht_inplace(H_tmp.T * suh.unsqueeze(0), block_size=QK_K).T
    return W_rot, H_rot, suh


@torch.no_grad()
def quantize_subs_lut_chunked(
    x_subs: torch.Tensor,
    dhs: torch.Tensor,
    lut_val: torch.Tensor,
    inv_step: float,
    half_b: float,
) -> torch.Tensor:
    """Vectorized optimal-state search and sub-block quantization via the 1D LUT, with a VRAM limit."""
    S_tot = x_subs.shape[0]
    G_out = torch.empty((S_tot, SUBBLOCK_SIZE), dtype=torch.float32, device=x_subs.device)
    chunk = 32768
    for i0 in range(0, S_tot, chunk):
        i1 = min(i0 + chunk, S_tot)
        xs = x_subs[i0:i1]
        dh = dhs[i0:i1]
        b_idx = (xs * inv_step + half_b).round().long().clamp(0, N_LUT_BUCKETS - 1)
        g_cand = lut_val[b_idx]  # [c_len, 16, C_all]
        err2 = (xs.unsqueeze(-1) - g_cand).square()
        bc = (err2 * dh).sum(dim=1).argmin(dim=-1)
        ar = torch.arange(i1 - i0, device=x_subs.device)
        G_out[i0:i1] = g_cand[ar, :, bc]
    return G_out


@torch.no_grad()
def quantize_liqk_v2_diag_rot(
    W_rot: torch.Tensor,
    diag_h: torch.Tensor,
    spec_name: str,
) -> torch.Tensor:
    """
    Fast vectorized diagonal quantization in the rotated basis `W_rot [M, K]`
    via the precomputed table `lut_val[8192, C_all]` (~0.004 s per matrix).
    """
    device = W_rot.device
    M, K = W_rot.shape
    nblocks = K // QK_K
    bank = get_warp_register_bank(spec_name, device)
    lut_val = bank["lut_val"]
    max_g = bank["max_grid"]
    inv_step = (N_LUT_BUCKETS - 1) / (2.0 * max_g)
    half_b = (N_LUT_BUCKETS - 1) * 0.5

    w_blks = W_rot.float().contiguous().view(M * nblocks, QK_K)
    # contiguous: with a single block (K = 256), reshape after expand stays a stride-0 view and the view below is impossible
    dh_blks = diag_h.float().view(1, nblocks, QK_K).expand(M, nblocks, QK_K).reshape(M * nblocks, QK_K).contiguous()
    B_tot = M * nblocks
    S_tot = B_tot * NUM_SUBBLOCKS
    d_cur = (w_blks.abs().amax(dim=1) / max_g).to(torch.float16).to(torch.float32).clamp(min=1e-8)
    dhs = dh_blks.view(S_tot, SUBBLOCK_SIZE, 1)

    for _ in range(int(os.environ.get("LIQK_PROFILE_ITERS", "2"))):  # d-fitting iterations (DP profile)
        x_subs = (w_blks / d_cur.unsqueeze(1)).view(S_tot, SUBBLOCK_SIZE)
        if lut_val.shape[1] == 64 and os.environ.get("LIQK_PROFILE_CUDA", "1") == "1":  # fused kernel (DP profile)
            from src.tcq_proto import lut_subs_quant_cuda
            G_blk = lut_subs_quant_cuda(x_subs, dh_blks.view(S_tot, SUBBLOCK_SIZE), lut_val, inv_step, half_b).view(B_tot, QK_K)
        else:
            G_blk = quantize_subs_lut_chunked(x_subs, dhs, lut_val, inv_step, half_b).view(B_tot, QK_K)
        num = (dh_blks * w_blks * G_blk).sum(dim=1)
        den = (dh_blks * G_blk * G_blk).sum(dim=1).clamp(min=1e-12)
        d_cur = (num / den).to(torch.float16).to(torch.float32).clamp(min=1e-8)
    return (d_cur.unsqueeze(1) * G_blk).view(M, K)


@torch.no_grad()
def _quantize_rows_lut_ldlq(
    W_rot: torch.Tensor,
    U: torch.Tensor,
    U_blks_inv: torch.Tensor,
    diag_w: torch.Tensor,
    spec_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast block Cholesky algorithm `Block16-LDLQ` with decoupled state search via `lut_val`
    and analytic Schur-complement recomputation of `d`.
    """
    device = W_rot.device
    bank = get_warp_register_bank(spec_name, device)
    grids_all = bank["grids_all"]
    lut_val = bank["lut_val"]
    max_g = bank["max_grid"]
    inv_step = (N_LUT_BUCKETS - 1) / (2.0 * max_g)
    half_b = (N_LUT_BUCKETS - 1) * 0.5

    M, K = W_rot.shape
    nblocks = K // QK_K
    W_work = W_rot.float().clone()
    W_hat_rot = torch.empty_like(W_work)
    d_full = torch.empty((M, nblocks), dtype=torch.float16, device=device)

    for ibl in range(nblocks):
        c0, c1 = ibl * QK_K, (ibl + 1) * QK_K
        w_blk_start = W_work[:, c0:c1].clone()
        U_blk = U[c0:c1, c0:c1]
        U_blk_inv = U_blks_inv[ibl]
        diag_w_blk = diag_w[c0:c1]
        A_chol = w_blk_start @ U_blk_inv

        d_cand = (w_blk_start.abs().amax(dim=1) / max_g).to(torch.float16).to(torch.float32).clamp(min=1e-8)
        x_blk = (w_blk_start / d_cand.unsqueeze(1)).view(M * NUM_SUBBLOCKS, SUBBLOCK_SIZE)
        dhs = diag_w_blk.view(1, NUM_SUBBLOCKS, SUBBLOCK_SIZE).expand(M, -1, -1).reshape(M * NUM_SUBBLOCKS, SUBBLOCK_SIZE, 1)
        G_init = quantize_subs_lut_chunked(x_blk, dhs, lut_val, inv_step, half_b).view(M, QK_K)

        num0 = (diag_w_blk.unsqueeze(0) * w_blk_start * G_init).sum(dim=1)
        den0 = (diag_w_blk.unsqueeze(0) * G_init * G_init).sum(dim=1).clamp(min=1e-12)
        d_cand = torch.where(den0 > 0, num0 / den0, d_cand).to(torch.float16).to(torch.float32).clamp(min=1e-8)

        best_loss_row = torch.full((M,), float("inf"), device=device)
        best_d_row = d_cand.clone()
        best_what_row = torch.zeros((M, QK_K), dtype=torch.float32, device=device)

        for _ in range(2):
            w_sub_work = w_blk_start.clone()
            G_pass = torch.empty((M, QK_K), dtype=torch.float32, device=device)
            inv_d = 1.0 / d_cand.unsqueeze(1)

            for ib in range(NUM_SUBBLOCKS):
                s0, s1 = ib * 16, (ib + 1) * 16
                w_sub_orig = w_sub_work[:, s0:s1]
                dw_sub = diag_w_blk[s0:s1]

                x_sub = w_sub_orig * inv_d
                b_idx = (x_sub * inv_step + half_b).round().long().clamp(0, N_LUT_BUCKETS - 1)
                g_cand = lut_val[b_idx]
                err2 = (x_sub.unsqueeze(-1) - g_cand).square()
                tot_e = torch.matmul(dw_sub, err2)
                bc = tot_e.argmin(dim=-1)

                grids_win = grids_all[bc]
                qi = (x_sub.unsqueeze(-1) - grids_win.unsqueeze(1)).abs().argmin(dim=-1)
                g_sub = torch.gather(grids_win, 1, qi)
                w_hat_sub = d_cand.unsqueeze(1) * g_sub

                G_pass[:, s0:s1] = g_sub
                if s1 < QK_K:
                    err_sub = (w_sub_orig - w_hat_sub) @ U_blk_inv[s0:s1, s0:s1]
                    w_sub_work[:, s1:] -= err_sub @ U_blk[s0:s1, s1:]

            B_chol = G_pass @ U_blk_inv
            num = (A_chol * B_chol).sum(dim=1)
            den = (B_chol * B_chol).sum(dim=1).clamp(min=1e-12)
            d_refit = torch.where(den > 0, num / den, d_cand).to(torch.float16).to(torch.float32).clamp(min=1e-8)

            for d_try in (d_cand, d_refit):
                w_hat_try = d_try.unsqueeze(1) * G_pass
                err_try = A_chol - d_try.unsqueeze(1) * B_chol
                loss_try = (err_try * err_try).sum(dim=1)
                imp = loss_try < best_loss_row
                if imp.any():
                    best_loss_row = torch.where(imp, loss_try, best_loss_row)
                    best_d_row = torch.where(imp, d_try, best_d_row)
                    best_what_row = torch.where(imp.unsqueeze(1), w_hat_try, best_what_row)
            d_cand = d_refit

        W_hat_rot[:, c0:c1] = best_what_row
        d_full[:, ibl] = best_d_row.to(torch.float16)
        if c1 < K:
            err_blk = (w_blk_start - best_what_row) @ U_blk_inv
            W_work[:, c1:] -= err_blk @ U[c0:c1, c1:]

    return d_full, W_hat_rot


# ---------------------------------------------------------------------------------------------------------
# Experimental E8 quantizer (night of 2026-10-25): instead of scalar levels, E8 lattice points in a ball
# of radius sqrt(r2) (r2 = 10: 56 881 points, stored as a 16-bit code per 8 weights; r2 = 2: 241 points, 8 bits).
# Structure as in Block16-LDLQ v2: scale d per (row, 256-block), the sub-block header (16 weights = 2 vectors)
# picks a gain from a bank of 2^h values, LDLQ feedback between sub-blocks. 3-4 bits: a second
# (residual) E8 stage with its own scale beta, like RVQ in QuIP#. Decode cost: one lookup per 8 weights.
# ---------------------------------------------------------------------------------------------------------

E8_CONFIG: Dict[str, dict] = {
    # stages: [(r2, beta)], gains: 2^h gains spaced gain_step octaves apart, scale: d = rms(block row) * scale
    # Parameters tuned on a Gaussian signal (gap to the R-D bound, bits): E8_2 G2 0.353, G4 0.366, G8 0.409;
    # v2 on the same: 0.38..0.45. Two-stage E8_3 / E8_4 are worse than v2 (0.55 / 0.53); kept as a negative result.
    "E8_2": {"stages": [(10.0, 1.0)], "gains": 2, "gain_step": 0.35, "scale": 1.0},
    "E8_2G4": {"stages": [(10.0, 1.0)], "gains": 4, "gain_step": 0.30, "scale": 1.0},
    "E8_2G8": {"stages": [(10.0, 1.0)], "gains": 8, "gain_step": 0.20, "scale": 1.0},
    "E8_3": {"stages": [(10.0, 1.0), (2.0, 0.38)], "gains": 4, "gain_step": 0.30, "scale": 1.1},
    "E8_4": {"stages": [(10.0, 1.0), (10.0, 0.25)], "gains": 4, "gain_step": 0.30, "scale": 1.1},
}
# Bytes per 256 weights: 32 vectors x (16 bits per r2=10 stage | 8 bits per r2=2 stage) + d fp16 + 3 bits x 16 headers
# Relative distortion for DP (proxy rig 1B, relative to LIQK3_V2): E8_2 6.75, E8_2G4 6.26
SPEC_DISTORTION_RATIO["E8_2"] = 6.75
SPEC_DISTORTION_RATIO["E8_2G4"] = 6.26
for _name, _cfg in E8_CONFIG.items():
    _idx_bytes = sum(64 if r2 > 2.0 else 32 for r2, _ in _cfg["stages"])
    _hdr_bytes = NUM_SUBBLOCKS * int(math.log2(_cfg["gains"])) // 8
    LIQK_V2_SPECS[_name] = LiqkV2Spec(
        name=_name, bits=_idx_bytes * 8 / QK_K, n_levels=0, n_pos_states=_cfg["gains"],
        block_bytes=_idx_bytes + 2 + _hdr_bytes,
    )


def _nearest_d8(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of the D8 lattice (integer vectors with even sum) along the last dimension."""
    f = torch.round(x)
    err = x - f
    odd = f.sum(dim=-1).remainder(2.0) != 0
    idx = err.abs().argmax(dim=-1, keepdim=True)
    step = torch.where(err.gather(-1, idx) >= 0, 1.0, -1.0)
    f_alt = f.scatter_add(-1, idx, step)
    return torch.where(odd.unsqueeze(-1), f_alt, f)


def _nearest_e8(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of E8 = D8 U (D8 + 1/2) (Conway-Sloane)."""
    a = _nearest_d8(x)
    b = _nearest_d8(x - 0.5) + 0.5
    da = (x - a).square().sum(dim=-1, keepdim=True)
    db = (x - b).square().sum(dim=-1, keepdim=True)
    return torch.where(da <= db, a, b)


_E8_SHRINK = (0.92, 0.84, 0.76, 0.68, 0.6, 0.5, 0.4, 0.3, 0.0)


def _nearest_e8_ball(x: torch.Tensor, r2: float) -> torch.Tensor:
    """
    E8 point with norm^2 <= r2: the nearest one if it lies in the ball, otherwise the nearest to the vector shrunk by the first
    suitable factor from `_E8_SHRINK`. No data-dependent branches, so the whole function compiles.
    """
    q = _nearest_e8(x)
    done = q.square().sum(dim=-1) <= r2 + 1e-6
    for t in _E8_SHRINK:
        qt = _nearest_e8(x * t)
        ok = (qt.square().sum(dim=-1) <= r2 + 1e-6) & ~done
        q = torch.where(ok.unsqueeze(-1), qt, q)
        done = done | ok
    return q


def _e8_quant_subblock_impl(x: torch.Tensor, gains: torch.Tensor, dw: torch.Tensor, stages: tuple) -> torch.Tensor:
    """
    Quantizes sub-blocks x [M, 16] (already divided by d) with two E8 vectors at all bank gains at once
    and returns the reconstruction with the gain that has the lowest weighted error.
    """
    G, M = gains.shape[0], x.shape[0]
    res = (x.unsqueeze(0) / gains.view(-1, 1, 1)).view(G, M, 2, 8)
    rec = torch.zeros_like(res)
    for r2, beta in stages:
        q = _nearest_e8_ball(res / beta, r2) * beta
        rec = rec + q
        res = res - q
    rec = rec.view(G, M, SUBBLOCK_SIZE) * gains.view(-1, 1, 1)
    err = ((x.unsqueeze(0) - rec).square() * dw).sum(dim=-1)
    best = err.argmin(dim=0)
    return rec.gather(0, best.view(1, M, 1).expand(1, M, SUBBLOCK_SIZE)).squeeze(0)


# torch.compile fuses ~200 small kernels per sub-block into a few (~10x faster, compilation ~15 s per process).
# LIQK_E8_NO_COMPILE=1 disables compilation.
_e8_quant_subblock_compiled = None


def _e8_quant_subblock(x: torch.Tensor, gains: torch.Tensor, stages, dw: torch.Tensor) -> torch.Tensor:
    global _e8_quant_subblock_compiled
    stages = tuple((float(r2), float(beta)) for r2, beta in stages)
    if os.environ.get("LIQK_E8_NO_COMPILE") == "1":
        return _e8_quant_subblock_impl(x, gains, dw, stages)
    if _e8_quant_subblock_compiled is None:
        _e8_quant_subblock_compiled = torch.compile(_e8_quant_subblock_impl, dynamic=True)
    return _e8_quant_subblock_compiled(x, gains, dw, stages)


@torch.no_grad()
def _quantize_rows_e8_ldlq(
    W_rot: torch.Tensor,
    U: torch.Tensor,
    U_blks_inv: torch.Tensor,
    diag_w: torch.Tensor,
    spec_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block16-LDLQ as in `_quantize_rows_lut_ldlq`, but each sub-block is quantized with two E8 points."""
    cfg = E8_CONFIG[spec_name]
    device = W_rot.device
    G = cfg["gains"]
    gains = 2.0 ** ((torch.arange(G, dtype=torch.float32, device=device) - (G - 1) / 2) * cfg["gain_step"])
    M, K = W_rot.shape
    nblocks = K // QK_K
    W_work = W_rot.float().clone()
    W_hat_rot = torch.empty_like(W_work)
    d_full = torch.empty((M, nblocks), dtype=torch.float16, device=device)

    for ibl in range(nblocks):
        c0, c1 = ibl * QK_K, (ibl + 1) * QK_K
        w_blk_start = W_work[:, c0:c1].clone()
        U_blk = U[c0:c1, c0:c1]
        U_blk_inv = U_blks_inv[ibl]
        diag_w_blk = diag_w[c0:c1]
        A_chol = w_blk_start @ U_blk_inv

        d_cand = (w_blk_start.square().mean(dim=1).sqrt() * cfg["scale"]).to(torch.float16).to(torch.float32).clamp(min=1e-8)
        best_loss_row = torch.full((M,), float("inf"), device=device)
        best_d_row = d_cand.clone()
        best_what_row = torch.zeros((M, QK_K), dtype=torch.float32, device=device)

        for _ in range(2):
            w_sub_work = w_blk_start.clone()
            G_pass = torch.empty((M, QK_K), dtype=torch.float32, device=device)
            inv_d = 1.0 / d_cand.unsqueeze(1)
            for ib in range(NUM_SUBBLOCKS):
                s0, s1 = ib * SUBBLOCK_SIZE, (ib + 1) * SUBBLOCK_SIZE
                w_sub_orig = w_sub_work[:, s0:s1]
                g_sub = _e8_quant_subblock(w_sub_orig * inv_d, gains, cfg["stages"], diag_w_blk[s0:s1])
                G_pass[:, s0:s1] = g_sub
                if s1 < QK_K:
                    err_sub = (w_sub_orig - d_cand.unsqueeze(1) * g_sub) @ U_blk_inv[s0:s1, s0:s1]
                    w_sub_work[:, s1:] -= err_sub @ U_blk[s0:s1, s1:]

            B_chol = G_pass @ U_blk_inv
            num = (A_chol * B_chol).sum(dim=1)
            den = (B_chol * B_chol).sum(dim=1).clamp(min=1e-12)
            d_refit = torch.where(den > 0, num / den, d_cand).to(torch.float16).to(torch.float32).clamp(min=1e-8)
            for d_try in (d_cand, d_refit):
                loss_try = (A_chol - d_try.unsqueeze(1) * B_chol).square().sum(dim=1)
                imp = loss_try < best_loss_row
                best_loss_row = torch.where(imp, loss_try, best_loss_row)
                best_d_row = torch.where(imp, d_try, best_d_row)
                best_what_row = torch.where(imp.unsqueeze(1), d_try.unsqueeze(1) * G_pass, best_what_row)
            d_cand = d_refit

        W_hat_rot[:, c0:c1] = best_what_row
        d_full[:, ibl] = best_d_row.to(torch.float16)
        if c1 < K:
            err_blk = (w_blk_start - best_what_row) @ U_blk_inv
            W_work[:, c1:] -= err_blk @ U[c0:c1, c1:]

    return d_full, W_hat_rot


# ---------------------------------------------------------------------------------------------------------
# Ungerboeck TCQ (src/tcq_proto.py: viterbi_ub): tile of 16 rows x 16 sub-block columns = 256 weights, row by row.
# 1 code bit per weight goes into the trellis memory (nu bits), K-1 bits pick the level within the subset; a table of 2^(K+1)
# entries (8/16/32: pshufb on AVX2); an h-bit header per 16 weights (tile row) picks the gain; scale d per
# (row, 256-block), as in v2. Tables and gains are trained on a Gaussian signal (results/tcq_gaussian.json, nu = 10).
# Bits per 256 weights: 256*K + nu (initial tile state) + 16 (d) + 16*h (headers).
# scale (d = rms * scale) tuned on a Gaussian signal in LDLQ: gap to R-D 0.215 / 0.251 / 0.283 bits
# (v2 on the same: 0.448 @2.375 / 0.384 @3.44 / 0.420 @4.5).
# ---------------------------------------------------------------------------------------------------------

TCQ_CONFIG: Dict[str, dict] = {
    "TCQ2": {"K": 2, "nu": 10, "m": 359, "h": 1, "scale": 1.1, "step": 0.4,
             "lut": [-1.7691, 0.1581, -0.9582, 0.5299, -0.5294, 0.9609, -0.1571, 1.7674]},
    "TCQ3": {"K": 3, "nu": 10, "m": 359, "h": 1, "scale": 1.15, "step": 0.4,
             "lut": [-2.1129, -0.5914, 0.0734, 0.7905, -1.4204, -0.4043, 0.2426, 1.0537, -1.0546, -0.2453, 0.4021, 1.4203,
                     -0.7924, -0.0751, 0.5916, 2.1165]},
    "TCQ4": {"K": 4, "nu": 10, "m": 359, "h": 1, "scale": 1.25, "step": 0.25,
             "lut": [-2.4689, -1.0897, -0.6307, -0.2814, 0.0376, 0.3596, 0.7259, 1.2576, -1.7803, -0.9513, -0.5342, -0.1954,
                     0.1209, 0.448, 0.8359, 1.47, -1.4681, -0.8349, -0.4492, -0.12, 0.1953, 0.5335, 0.9513, 1.7807, -1.2568,
                     -0.7259, -0.36, -0.0359, 0.281, 0.6301, 1.0894, 2.4651]},
}
# R variants: one scale d per row (instead of d per (row, 256-block); c_row does the per-row refinement anyway) and the
# trellis state carries over to the next tile of the same rows (no nu bits of initial state per tile); one pass instead of two.
# Overhead: 16*h bits + 5 bits of scale correction per (row, 256-block) + d fp16 per row (~0.008 bits/weight at K >= 2048,
# counted as 2 bits per 256).
# scale tuned on a Gaussian signal: gap to R-D 0.197 / 0.236 / 0.269 bits (TCQ 0.215 / 0.251 / 0.283; EXL3 0.086 / 0.099 / 0.115)
for _k, _sc in ((2, 1.05), (3, 1.15), (4, 1.25)):
    TCQ_CONFIG[f"TCQ{_k}R"] = dict(TCQ_CONFIG[f"TCQ{_k}"], row_scale=True, carry=True, scale=_sc)
# Header h = 2 (4 gains per 16 weights): on a Gaussian signal (tcq_proto rig) K=3 0.216 vs 0.261 at h = 1,
# K=4 0.244 vs 0.327, K=2 no gain; via LDLQ (R scales): TCQ3RH 0.193 (TCQ3R 0.236), TCQ4RH 0.223 (TCQ4R 0.269).
TCQ_CONFIG["TCQ3RH"] = dict(TCQ_CONFIG["TCQ3R"], h=2, step=0.3, scale=1.2, lut=[
    -1.9627, -0.5848, 0.0759, 0.7802, -1.3757, -0.401, 0.2391, 1.0302, -1.0293, -0.24, 0.4021, 1.3758, -0.78, -0.0751, 0.5836, 1.9601])
TCQ_CONFIG["TCQ4RH"] = dict(TCQ_CONFIG["TCQ4R"], h=2, step=0.3, scale=1.3, lut=[
    -2.2634, -1.0835, -0.626, -0.2776, 0.0375, 0.3583, 0.724, 1.2492, -1.6723, -0.9483, -0.5319, -0.1947, 0.1186, 0.446,
    0.8312, 1.4464, -1.4469, -0.8308, -0.4461, -0.1175, 0.1947, 0.5318, 0.9474, 1.6697, -1.2506, -0.7245, -0.359, -0.037,
    0.2785, 0.6272, 1.0835, 2.2575])
# h = 3 (8 gains) for K=4: on a Gaussian signal 0.205 vs 0.244 at h = 2 (no gain for K=3); via LDLQ 0.180 (EXL3 K=4: 0.115).
# n_cand=4: Viterbi over 4 of 8 gains per sub-block (preselected by scalar error): 2x faster, KLD on 3B unchanged
TCQ_CONFIG["TCQ4RH3"] = dict(TCQ_CONFIG["TCQ4R"], h=3, step=0.15, scale=1.2, n_cand=4, lut=[-2.2018, -1.0823, -0.629, -0.2785, 0.0376, 0.3586, 0.723, 1.2422, -1.6809, -0.947, -0.5322, -0.1953, 0.1197, 0.446, 0.8324, 1.4363, -1.4341, -0.832, -0.4471, -0.1185, 0.1953, 0.5317, 0.9493, 1.6801, -1.2404, -0.7251, -0.3584, -0.0366, 0.2786, 0.6283, 1.0822, 2.2035])
# Half-integer TCQ25R / TCQ35R: tile rows alternate, even rows with K bits (table TCQ_K), odd rows with K + 1 (table
# TCQ_{K+1}); the trellis is shared (1 code bit per weight), each half has its own scale and gains (tables premultiplied by
# scale, d = rms without the factor). DP needs these to avoid mixing formats one whole bit apart: with D ~ 2^(-2R) a 50/50
# mix of R - 0.5 / R + 0.5 gives 1.25x more distortion than a uniform bitrate R.
for _k in (2, 3):
    TCQ_CONFIG[f"TCQ{_k}5R"] = {"K": _k, "K_hi": _k + 1, "nu": 10, "m": 359, "h": 1, "row_scale": True, "carry": True,
                                "scale": 1.0, "lo": f"TCQ{_k}R", "hi": f"TCQ{_k + 1}R"}
# Night 2: tail-biting (the tile path is closed, initial state = last nu code bits of the tile, no side information)
# and extended labels (m_extra: extra state parities, 8 / 16 subsets, table [nl, 2^(K-1)]). Tables are trained
# on the Gaussian rig for tail-biting (`src/train_tcq_x.py` -> results/tcq_x_tables.json). Overhead as for the R formats.
# TCQ_X_SCALE is the factor in d = rms * scale, tuned through the full pipeline on a Gaussian matrix.
TCQ_X_SCALE: Dict[str, float] = {"TCQ2X0": 1.0, "TCQ2X": 1.0, "TCQ3X1": 1.1, "TCQ3X": 1.1, "TCQ3X3": 1.1, "TCQ3X16": 1.1,
                                 "TCQ4X1": 1.2, "TCQ4X2": 1.2, "TCQ4X": 1.2,
                                 "TCQ3Y0": 1.1, "TCQ3Y1": 1.1, "TCQ3Y2": 1.1, "TCQ4Y0": 1.2, "TCQ4Y1": 1.1, "TCQ4Y2": 1.1,
                                 "TCQ4Y3": 1.1, "TCQ5X3": 1.2, "TCQ5X": 1.1, "TCQ3W2": 1.1, "TCQ3W3": 1.1, "TCQ4W2": 1.2,
                                 "TCQ4W3": 1.2, "TCQ4W4": 1.2, "TCQ25X": 1.0, "TCQ35X1": 1.1, "TCQ35X": 1.1, "TCQ45X": 1.2,
                                 "TCQ45X3": 1.1, "TCQ25Y": 1.0, "TCQ35Y1": 1.1, "TCQ35Y": 1.1, "TCQ275X": 1.1,
                                 "TCQ275X2": 1.1, "TCQ375X": 1.1, "TCQ475X": 1.2, "TCQ225X": 1.0, "TCQ325X": 1.1,
                                 "TCQ425X": 1.2}
_X_TABLES = Path(__file__).resolve().parent.parent / "results" / "tcq_x_tables.json"
if _X_TABLES.exists():
    for _name, _t in json.loads(_X_TABLES.read_text()).items():
        TCQ_CONFIG[_name] = {"K": _t["K"], "nu": 10, "m": 359, "h": _t["h"], "step": _t["step"], "lut": _t["lut"],
                             "sub": _t.get("sub", SUBBLOCK_SIZE), "pp": _t.get("pp", False),
                             "scale": TCQ_X_SCALE.get(_name, 1.0), "m_extra": tuple(_t["m_extra"]), "row_scale": True,
                             "tb": True, **({"n_cand": 4} if _t["h"] > 2 else {})}
# Relative distortion for DP (proxy rig 1B, sum of fkl relative to LIQK3_V2; E8_2 was calibrated the same way)
SPEC_DISTORTION_RATIO.update({"TCQ2": 5.03, "TCQ3": 1.21, "TCQ4": 0.308,
                              "TCQ2R": 5.78, "TCQ25R": 3.20, "TCQ3R": 1.354, "TCQ35R": 0.786, "TCQ4R": 0.345,
                              "TCQ3RH": 1.164, "TCQ4RH": 0.296, "TCQ4RH3": 0.256})
# night 2: proxy rig 3B (sum of fkl relative to LIQK3_V2 = 0.2445, results/proxy_error_3b_night2.json); the old TCQs on 3B
# nearly match 1B (TCQ2R 5.56, TCQ3RH 1.148, TCQ4RH3 0.257)
SPEC_DISTORTION_RATIO.update({"TCQ2X0": 5.443, "TCQ2X": 4.904, "TCQ3X1": 1.197, "TCQ3X": 1.063, "TCQ3X16": 1.040,
                              "TCQ4X1": 0.332, "TCQ4X2": 0.284, "TCQ4X": 0.251,
                              "TCQ3Y0": 1.280, "TCQ3Y1": 1.145, "TCQ3Y2": 1.036, "TCQ4Y0": 0.343, "TCQ4Y1": 0.297,
                              "TCQ4Y2": 0.262, "TCQ5X3": 0.0633, "TCQ5X": 0.0581,
                              "TCQ3W2": 1.166, "TCQ4W2": 0.316, "TCQ4W3": 0.292,
                              "TCQ25X": 2.487, "TCQ35X1": 0.663, "TCQ35X": 0.575, "TCQ45X": 0.145,
                              "TCQ45X3": 0.1319, "TCQ25Y": 2.452, "TCQ35Y1": 0.611, "TCQ35Y": 0.545,
                              "TCQ275X": 1.725, "TCQ275X2": 1.554, "TCQ375X": 0.405, "TCQ475X": 0.102,
                              # K + 0.25: proxy with the new (unbiased) Fisher, mapped to the old one's scale via neighbors
                              # (fkl / rel of the neighboring format per matrix): fkl sums under different Fishers are not comparable
                              "TCQ225X": 3.549, "TCQ325X": 0.801, "TCQ425X": 0.2007})
# Recalibration (night 2, 13:37): all TCQPQ menu formats in a single 3B proxy run with the unbiased Fisher
# (results/proxy_error_3b_night2_fisher2.json); TCQ_RATIOS_V1=1 restores the previous values
if os.environ.get("TCQ_RATIOS_V1", "0") != "1":
    SPEC_DISTORTION_RATIO.update({"TCQ2X0": 5.5586, "TCQ2X": 5.0328, "TCQ3X1": 1.2457, "TCQ3X": 1.1072, "TCQ4X1": 0.3473, "TCQ4X2": 0.2973, "TCQ4X": 0.2635, "TCQ4W2": 0.3304, "TCQ4W3": 0.306, "TCQ5X3": 0.0664, "TCQ5X": 0.0609, "TCQ25X": 2.5181, "TCQ35X1": 0.6695, "TCQ35X": 0.5806, "TCQ45X": 0.1464, "TCQ275X": 1.7468, "TCQ275X2": 1.5742, "TCQ375X": 0.4094, "TCQ475X": 0.1033})
for _name, _cfg in TCQ_CONFIG.items():
    if _cfg.get("row_scale"):
        _bits = (QK_K * (_cfg["K"] + 0.5 * ("K_hi" in _cfg) - {0: 0.0, 1: 0.5, 2: 0.25, 3: 0.75}[int(_cfg.get("pp", 0))])
                 + QK_K // _cfg.get("sub", SUBBLOCK_SIZE) * _cfg["h"] + 5 + 2)
    else:
        _bits = QK_K * _cfg["K"] + _cfg["nu"] + 16 + 16 * _cfg["h"]
    LIQK_V2_SPECS[_name] = LiqkV2Spec(name=_name, bits=float(_cfg["K"]), n_levels=1 << (_cfg["K"] + 1),
                                      n_pos_states=1 << _cfg["h"], block_bytes=_bits / 8)


@torch.no_grad()
def _quantize_rows_tcq_ldlq(
    W_rot: torch.Tensor,
    U: torch.Tensor,
    U_blks_inv: torch.Tensor,
    diag_w: torch.Tensor,
    spec_name: str,
    n_pass: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block16-LDLQ as in `_quantize_rows_e8_ldlq`, but sub-block [M, 16] is quantized in 16x16 tiles via TCQ."""
    from src.tcq_proto import make_gains, viterbi_ub, viterbi_ub_cuda

    cfg = TCQ_CONFIG[spec_name]
    device = W_rot.device
    hi_kw = {}
    if "K_hi" in cfg:  # half-integer bitrate: each half's tables are premultiplied by its scale, d has no factor
        lo, hi = TCQ_CONFIG[cfg["lo"]], TCQ_CONFIG[cfg["hi"]]
        lut = torch.tensor(lo["lut"], dtype=torch.float32, device=device) * lo["scale"]
        gains = make_gains(lo["h"], lo["step"], device)
        hi_kw = {"small_hi": torch.tensor(hi["lut"], dtype=torch.float32, device=device) * hi["scale"],
                 "gains_hi": make_gains(hi["h"], hi["step"], device), "mix": SUBBLOCK_SIZE}
    else:
        lut = torch.tensor(cfg["lut"], dtype=torch.float32, device=device)
        gains = make_gains(cfg["h"], cfg["step"], device)
    M, K = W_rot.shape
    assert M % 16 == 0, "TCQ: number of rows must be divisible by 16"
    nt = M // 16
    nblocks = K // QK_K
    W_work = W_rot.float().clone()
    W_hat_rot = torch.empty_like(W_work)
    d_full = torch.empty((M, nblocks), dtype=torch.float16, device=device)

    carry = {"st": None}
    # n_cand: Viterbi only over n_cand gain candidates per sub-block (TCQ_NCAND overrides the config)
    n_cand = int(os.environ.get("TCQ_NCAND", cfg.get("n_cand", 0)))
    pk = None
    if PACK is not None:
        assert cfg.get("tb") and cfg.get("row_scale"), f"{spec_name}: only formats with tail-biting and d_row are written to .ltq"
        pk = {"idx": torch.empty((M, K), dtype=torch.int16, device=device), "gi": torch.empty((M, K), dtype=torch.uint8, device=device),
              "G": torch.empty((M, K), dtype=torch.float32, device=device), "cur": None}

    def quant_sub(xs: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:  # xs [M, 16] (already divided by d)
        tiles = xs.view(nt, 16 * SUBBLOCK_SIZE)
        tb = os.environ.get("TCQ_TB", "")  # "", "full" or the first-pass window width (e.g. "64")
        if cfg.get("tb"):  # night-2 formats: tail-biting and extended labels; TCQ_TB_WIN = first-pass window (0 = whole tile)
            rec = viterbi_ub_cuda(tiles, lut, cfg["K"], cfg["nu"], cfg["m"], gains, sub=cfg.get("sub", SUBBLOCK_SIZE),
                                  wts=dw.repeat(16), n_cand=n_cand, tail_biting=True,
                                  tb_window=int(os.environ.get("TCQ_TB_WIN", "32")),
                                  m_extra=cfg.get("m_extra", ()), pair_parity=cfg.get("pp", False), return_code=pk is not None)
            if pk is not None:  # tile codes: position t = 16 r + c -> sub-block row r, column c
                rec, gi, ix = rec
                pk["cur"] = (gi.reshape(M, SUBBLOCK_SIZE), ix.reshape(M, SUBBLOCK_SIZE))
        elif tb and cfg.get("carry"):  # tail-biting: closed path in each tile instead of carrying the state
            rec = viterbi_ub_cuda(tiles, lut, cfg["K"], cfg["nu"], cfg["m"], gains, sub=SUBBLOCK_SIZE, wts=dw.repeat(16),
                                  n_cand=n_cand, tail_biting=True, tb_window=0 if tb == "full" else int(tb), **hi_kw)
        elif cfg.get("carry"):  # trellis state carries over to the next tile of the same rows
            rec, carry["st"] = viterbi_ub_cuda(tiles, lut, cfg["K"], cfg["nu"], cfg["m"], gains, sub=SUBBLOCK_SIZE,
                                               wts=dw.repeat(16), init_state=carry["st"], return_end_state=True, n_cand=n_cand,
                                               **hi_kw)
        elif os.environ.get("TCQ_CUDA", "1") == "1":  # CUDA Viterbi (bit-exact with the torch version), TCQ_CUDA=0 = torch
            rec = viterbi_ub_cuda(tiles, lut, cfg["K"], cfg["nu"], cfg["m"], gains, sub=SUBBLOCK_SIZE, wts=dw.repeat(16),
                                  n_cand=n_cand)
        else:
            rec, *_ = viterbi_ub(tiles, lut, cfg["K"], cfg["nu"], cfg["m"], gains, sub=SUBBLOCK_SIZE, wts=dw.repeat(16))
        return rec.view(M, SUBBLOCK_SIZE)

    if cfg.get("row_scale"):
        # d(row, block) = d_row * 2^(0.25 * e), e is a 5-bit correction per (row, 256-block) in the range [-16, 15]
        # (outlier input channels make individual row blocks of down_proj / o_proj much larger than the rest);
        # d_row is fp16 per row (geometric mean of the row's block rms)
        rms_rb = W_rot.float().view(M, nblocks, QK_K).square().mean(dim=2).sqrt().clamp(min=1e-10) * cfg.get("scale", 1.0)
        d_row = torch.exp(torch.log(rms_rb).mean(dim=1)).to(torch.float16).to(torch.float32).clamp(min=1e-8)
        e_step, e_max = cfg.get("e_step", 0.25), 1 << (cfg.get("e_bits", 5) - 1)
        e_rb = torch.round(torch.log2(rms_rb / d_row.unsqueeze(1)) / e_step).clamp(-e_max, e_max - 1)
        d_rb_all = d_row.unsqueeze(1) * torch.exp2(e_step * e_rb)

    for ibl in range(nblocks):
        c0, c1 = ibl * QK_K, (ibl + 1) * QK_K
        w_blk_start = W_work[:, c0:c1].clone()
        U_blk = U[c0:c1, c0:c1]
        U_blk_inv = U_blks_inv[ibl]
        diag_w_blk = diag_w[c0:c1]
        A_chol = w_blk_start @ U_blk_inv

        if cfg.get("row_scale"):  # single pass with a rank-1 scale
            d_rb = d_rb_all[:, ibl].clamp(min=1e-8)
            w_sub_work = w_blk_start.clone()
            G_pass = torch.empty((M, QK_K), dtype=torch.float32, device=device)
            inv_d = 1.0 / d_rb.unsqueeze(1)
            for ib in range(NUM_SUBBLOCKS):
                s0, s1 = ib * SUBBLOCK_SIZE, (ib + 1) * SUBBLOCK_SIZE
                w_sub_orig = w_sub_work[:, s0:s1]
                g_sub = quant_sub(w_sub_orig * inv_d, diag_w_blk[s0:s1])
                G_pass[:, s0:s1] = g_sub
                if pk is not None:
                    pk["gi"][:, c0 + s0 : c0 + s1] = pk["cur"][0].to(torch.uint8)
                    pk["idx"][:, c0 + s0 : c0 + s1] = pk["cur"][1].to(torch.int16)
                if s1 < QK_K:
                    err_sub = (w_sub_orig - d_rb.unsqueeze(1) * g_sub) @ U_blk_inv[s0:s1, s0:s1]
                    w_sub_work[:, s1:] -= err_sub @ U_blk[s0:s1, s1:]
            best_what_row = d_rb.unsqueeze(1) * G_pass
            W_hat_rot[:, c0:c1] = best_what_row
            if pk is not None:
                pk["G"][:, c0:c1] = G_pass
            d_full[:, ibl] = d_rb.to(torch.float16)
            if c1 < K:
                err_blk = (w_blk_start - best_what_row) @ U_blk_inv
                W_work[:, c1:] -= err_blk @ U[c0:c1, c1:]
            continue

        d_cand = (w_blk_start.square().mean(dim=1).sqrt() * cfg.get("scale", 1.0)).to(torch.float16).to(torch.float32).clamp(min=1e-8)
        best_loss_row = torch.full((M,), float("inf"), device=device)
        best_d_row = d_cand.clone()
        best_what_row = torch.zeros((M, QK_K), dtype=torch.float32, device=device)

        for _ in range(n_pass):
            w_sub_work = w_blk_start.clone()
            G_pass = torch.empty((M, QK_K), dtype=torch.float32, device=device)
            inv_d = 1.0 / d_cand.unsqueeze(1)
            for ib in range(NUM_SUBBLOCKS):
                s0, s1 = ib * SUBBLOCK_SIZE, (ib + 1) * SUBBLOCK_SIZE
                w_sub_orig = w_sub_work[:, s0:s1]
                g_sub = quant_sub(w_sub_orig * inv_d, diag_w_blk[s0:s1])
                G_pass[:, s0:s1] = g_sub
                if s1 < QK_K:
                    err_sub = (w_sub_orig - d_cand.unsqueeze(1) * g_sub) @ U_blk_inv[s0:s1, s0:s1]
                    w_sub_work[:, s1:] -= err_sub @ U_blk[s0:s1, s1:]

            B_chol = G_pass @ U_blk_inv
            num = (A_chol * B_chol).sum(dim=1)
            den = (B_chol * B_chol).sum(dim=1).clamp(min=1e-12)
            d_refit = torch.where(den > 0, num / den, d_cand).to(torch.float16).to(torch.float32).clamp(min=1e-8)
            for d_try in (d_cand, d_refit):
                loss_try = (A_chol - d_try.unsqueeze(1) * B_chol).square().sum(dim=1)
                imp = loss_try < best_loss_row
                best_loss_row = torch.where(imp, loss_try, best_loss_row)
                best_d_row = torch.where(imp, d_try, best_d_row)
                best_what_row = torch.where(imp.unsqueeze(1), d_try.unsqueeze(1) * G_pass, best_what_row)
            d_cand = d_refit

        W_hat_rot[:, c0:c1] = best_what_row
        d_full[:, ibl] = best_d_row.to(torch.float16)
        if c1 < K:
            err_blk = (w_blk_start - best_what_row) @ U_blk_inv
            W_work[:, c1:] -= err_blk @ U[c0:c1, c1:]

    if pk is not None:
        PACK["groups"].append({"spec": spec_name, "d_row": d_row, "e": e_rb.to(torch.int8), "idx": pk["idx"], "gi": pk["gi"],
                               "G": pk["G"]})
    return d_full, W_hat_rot


def _fwht_rows(X: torch.Tensor) -> torch.Tensor:
    """Block normalized FWHT-256 over rows (along the first dimension)."""
    return fwht_inplace(X.T.contiguous(), block_size=QK_K).T.contiguous()


@torch.no_grad()
def quantize_matrix_liqk_v2(
    W: torch.Tensor,
    H: torch.Tensor,
    spec_or_row_specs: str | List[Tuple[int, int, str]],
    seed: int = 1337,
    damp: float = 0.02,
    row_weights: Optional[torch.Tensor] = None,
    cd_sweeps: int = 2,
    out_rot: bool = False,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float]:
    """
    Quantizes matrix `W [M, K]` with input Hessian `H [K, K]` into the `.liqk v2` format.
    Supports both a single format `spec_name` for the whole matrix and a per-block assignment
    of formats over row slices `[(r0, r1, spec_name), ...]`, using a shared rotation `suh[K]`,
    LUT-accelerated `Block16-LDLQ` and joint final refinement of `c_row` and `suh[K]`.

    `out_rot=True` additionally rotates rows with a block FWHT-256 with signs `s_out[M]` (at inference:
    one inverse FWHT-256 per output vector). The proxy error tr(dW H dW^T) is unchanged by this;
    Fisher row weights are averaged per 256-block, row slices (multiples of 256) stay independent.
    """
    if out_rot:
        s_out = make_rademacher_signs(W.shape[0], seed=seed + 7919, device=W.device)
        rw = None
        if row_weights is not None:
            rw = row_weights.float().view(-1, QK_K).mean(dim=1, keepdim=True).expand(-1, QK_K).reshape(-1)
        q_res, w_hat_r, rel = quantize_matrix_liqk_v2(
            _fwht_rows(W.float() * s_out.unsqueeze(1)), H, spec_or_row_specs, seed=seed, damp=damp,
            row_weights=rw, cd_sweeps=cd_sweeps,
        )
        q_res["s_out"] = s_out.to(torch.int8)
        return q_res, _fwht_rows(w_hat_r) * s_out.unsqueeze(1), rel

    W_f32 = W.float()
    H_f32 = H.float()
    M, K = W_f32.shape
    nblocks = K // QK_K
    # TCQ needs stronger Hessian damping than v2 (3B, night 2: 0.2 instead of 0.02 gives 4.0 bpw −17 % KLD, 2.5 mul −4 %,
    # 4.5 −5…−7 %; 0.4 is already worse at 3.0 / 4.0). TCQ_DAMP_MULT=1 restores the previous behavior.
    specs_used = [spec_or_row_specs] if isinstance(spec_or_row_specs, str) else [sp for _, _, sp in spec_or_row_specs]
    if any(sp in TCQ_CONFIG for sp in specs_used):
        damp *= float(os.environ.get("TCQ_DAMP_MULT", "10"))

    W_rot, H_rot, suh = compute_optimal_suh(W_f32, H_f32, seed=seed, row_weights=row_weights)

    H_reg = H_rot.float().clone()
    diag_mean = H_reg.diagonal().mean().clamp(min=1e-8)
    H_reg.diagonal().add_(damp * diag_mean)
    # U is the upper Cholesky factor of H^-1 (U^T U = H^-1) without explicit inversion: with J H J = L' L'^T,
    # U = J L'^-1 J (J = index reversal); one factorization and one triangular inversion instead of two factorizations
    L_flip = torch.linalg.cholesky(H_reg.flip(0, 1))
    del H_rot, H_reg  # K x K matrices: on Llama-8B down_proj (K = 14336) that is ~0.8 GB VRAM each
    U = torch.linalg.solve_triangular(L_flip, torch.eye(K, dtype=L_flip.dtype, device=L_flip.device), upper=False)
    U = U.flip(0, 1)
    del L_flip
    diag_w = 1.0 / (torch.diagonal(U) ** 2).clamp(min=1e-12)

    eye256 = torch.eye(QK_K, dtype=torch.float32, device=W.device).unsqueeze(0).expand(nblocks, -1, -1)
    U_blks = torch.stack([U[b * QK_K : (b + 1) * QK_K, b * QK_K : (b + 1) * QK_K] for b in range(nblocks)], dim=0)
    U_blks_inv = torch.linalg.solve_triangular(U_blks, eye256, upper=True)

    if isinstance(spec_or_row_specs, str):
        row_slices = [(0, M, spec_or_row_specs)]
    else:
        row_slices = spec_or_row_specs

    by_spec: Dict[str, List[Tuple[int, int]]] = {}
    for r0, r1, sp in row_slices:
        by_spec.setdefault(sp, []).append((r0, r1))

    W_hat_rot = torch.empty_like(W_rot)
    d_full = torch.empty((M, nblocks), dtype=torch.float16, device=W.device)
    if PACK is not None:
        PACK["groups"] = []
        pm = {"idx": torch.empty((M, K), dtype=torch.int16, device=W.device), "gi": torch.empty((M, K), dtype=torch.uint8, device=W.device),
              "G": torch.empty((M, K), dtype=torch.float32, device=W.device), "d_row": torch.empty(M, device=W.device),
              "e": torch.empty((M, nblocks), dtype=torch.int8, device=W.device)}

    def quant_spec(sp: str, ranges: List[Tuple[int, int]]):
        rows_idx = torch.cat([torch.arange(r0, r1, device=W.device) for r0, r1 in ranges], dim=0)
        quant_rows = (_quantize_rows_e8_ldlq if sp in E8_CONFIG else _quantize_rows_tcq_ldlq if sp in TCQ_CONFIG
                      else _quantize_rows_lut_ldlq)
        d_sub, Wh_sub = quant_rows(W_rot[rows_idx], U, U_blks_inv, diag_w, sp)
        W_hat_rot[rows_idx] = Wh_sub
        d_full[rows_idx] = d_sub
        if PACK is not None:
            g = PACK["groups"].pop()
            for k in ("idx", "gi", "G", "d_row", "e"):
                pm[k][rows_idx] = g[k].to(pm[k].dtype)

    if len(by_spec) > 1 and W.is_cuda and os.environ.get("LIQK_PAR_SPEC", "0") == "1":
        # row groups of different formats are independent: their LDLQ loops (small Viterbi calls) overlap in their own CUDA streams
        main = torch.cuda.current_stream(W.device)
        streams = [torch.cuda.Stream(W.device) for _ in by_spec]

        def job(i_item):
            i, (sp, ranges) = i_item
            streams[i].wait_stream(main)
            with torch.cuda.stream(streams[i]):
                quant_spec(sp, ranges)
            streams[i].synchronize()

        with ThreadPoolExecutor(len(by_spec)) as ex:
            list(ex.map(job, enumerate(by_spec.items())))
    else:
        for sp, ranges in by_spec.items():
            quant_spec(sp, ranges)

    if PACK is not None:
        assert not (len(by_spec) > 1 and os.environ.get("LIQK_PAR_SPEC", "0") == "1"), "packing requires LIQK_PAR_SPEC to be off"
    U_base = fwht_inplace(W_hat_rot.clone(), block_size=QK_K)
    W_hat_eff = U_base * suh.unsqueeze(0)
    c_pack = torch.ones((M,), dtype=torch.float32, device=W.device)

    HW = W_f32 @ H_f32
    HQ = W_hat_eff @ H_f32
    c_row = (W_hat_eff * HW).sum(dim=1) / (W_hat_eff * HQ).sum(dim=1).clamp(min=1e-12)
    c_row = torch.where(torch.isfinite(c_row) & (c_row > 0.5) & (c_row < 2.0), c_row, torch.ones_like(c_row))
    d_new = (d_full.float() * c_row.unsqueeze(1)).to(torch.float16)
    c_eff = (d_new.float() / d_full.float().clamp(min=1e-12)).mean(dim=1, keepdim=True)
    U_scaled = U_base * c_eff

    if row_weights is not None:
        rw = row_weights.float().clamp(min=float(row_weights.mean().item()) * 0.1)
        rw = (rw / rw.mean().clamp(min=1e-12)).clamp(0.25, 4.0)
        U_w = U_scaled * rw.unsqueeze(1)
    else:
        rw = torch.ones((M,), dtype=torch.float32, device=W.device)
        U_w = U_scaled

    A_in = (U_w.T @ U_scaled) * H_f32
    b_in = (U_w * HW).sum(dim=0)
    A_in.diagonal().add_(1e-3 * A_in.diagonal().mean().clamp(min=1e-12))
    try:
        suh_cand = torch.linalg.solve(A_in, b_in).to(torch.float16).to(torch.float32)
        valid_s = torch.isfinite(suh_cand) & (suh_cand.abs() > 0.05) & (suh_cand.abs() < 10.0)
        suh_cand = torch.where(valid_s, suh_cand, suh)
        W_hat_cand = U_scaled * suh_cand.unsqueeze(0)
        HQ_cand = W_hat_cand @ H_f32
        # (W - W_hat) H = HW - W_hat H: no extra M x K x K products
        diff_base = W_f32 - W_hat_eff
        diff_cand = W_f32 - W_hat_cand
        loss_base = float((((HW - HQ) * diff_base).sum(dim=1) * rw).sum().item())
        loss_cand = float((((HW - HQ_cand) * diff_cand).sum(dim=1) * rw).sum().item())
        if loss_cand < loss_base:
            W_hat_eff = W_hat_cand
            d_full = d_new
            suh = suh_cand
            HQ = HQ_cand
            c_pack = c_eff.squeeze(1)
    except RuntimeError:
        pass

    if PACK is not None:  # matrix exactly from the file fields: fp16 d_row (with c_row refinement), e, fp16 suh and tile values
        d_row_p = (pm["d_row"] * c_pack).to(torch.float16)
        suh_p = suh.to(torch.float16)
        scale = d_row_p.float().unsqueeze(1) * torch.exp2(0.25 * pm["e"].float())
        Wq_rot = (scale.unsqueeze(2) * pm["G"].view(M, nblocks, QK_K)).view(M, K)
        W_hat_eff = fwht_inplace(Wq_rot, block_size=QK_K) * suh_p.float().unsqueeze(0)
        HQ = W_hat_eff @ H_f32
        PACK["matrix"] = {"row_slices": row_slices, "idx": pm["idx"], "gi": pm["gi"], "G": pm["G"], "d_row": d_row_p,
                          "e": pm["e"], "suh": suh_p}

    diff = W_f32 - W_hat_eff
    num_loss = float(((HW - HQ) * diff).sum().item())
    den_loss = float((HW * W_f32).sum().clamp(min=1e-12).item())
    rel_loss = num_loss / den_loss

    qdata_out = {"d": d_full, "suh": suh.to(torch.float16), "row_slices": row_slices}
    return qdata_out, W_hat_eff, rel_loss


def compute_exact_row_fisher_sensitivities(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    n_Menu_seqs: int = 8,
    seq_len: int = 1024,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """
    In ~1.2 seconds computes an unbiased estimate of the diagonal of the Fisher information matrix
    of the model's final KL divergence for each output channel `i = 0..M-1` of all linear matrices.
    """
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    if os.environ.get("LIQK_UNIT_FISHER", "0") == "1":  # experiment: row weights = 1 (as in the streaming pipeline for 8B)
        return {(l, sub): torch.ones(model.model.layers[l].get_submodule(sub).out_features)
                for l in range(n_layers) for sub in LINEAR_SUBMODULES}
    fisher_out: Dict[Tuple[int, str], torch.Tensor] = {
        (l, sub): torch.zeros(
            model.model.layers[l].get_submodule(sub).out_features, dtype=torch.float64, device=device
        )
        for l in range(n_layers)
        for sub in LINEAR_SUBMODULES
    }
    total_tokens = 0

    hooks = []
    for l in range(n_layers):
        layer = model.model.layers[l]
        for sub in LINEAR_SUBMODULES:
            mod = layer.get_submodule(sub)

            def make_hook(key):
                def fwd_hook(module, inp, out):
                    if out.requires_grad:
                        def bwd_hook(grad):
                            g2 = grad.detach().float().square().sum(dim=(0, 1)).to(torch.float64)
                            fisher_out[key].add_(g2)
                        out.register_hook(bwd_hook)
                return fwd_hook

            hooks.append(mod.register_forward_hook(make_hook((l, sub))))

    emb_hook = model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    n_use = min(n_Menu_seqs, cal_tokens.shape[0])
    with torch.enable_grad():
        for i in range(n_use):
            tok = cal_tokens[i : i + 1, :seq_len].to(device)
            out = model(tok, use_cache=False)
            logits = out.logits.reshape(-1, out.logits.shape[-1])
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=-1)
                if os.environ.get("LIQK_FISHER_STABLE", "1") == "1" and os.environ.get("LIQK_FISHER_SAMPLE", "1") == "1":
                    # inverse CDF with fixed uniform numbers: small numerical discrepancies in the
                    # logits (non-deterministic attention backward) barely change the sampled tokens, so the DP profile
                    # is reproducible across processes (with multinomial, discrepancies changed samples, MLP permutations and DP)
                    u = torch.rand((probs.shape[0], 2), generator=torch.Generator().manual_seed(1000 + i)).to(device)
                    cdf = probs.cumsum(dim=-1)
                    y_samp = torch.searchsorted(cdf, (u * cdf[:, -1:]).contiguous()).clamp(max=probs.shape[-1] - 1)
                else:
                    y_samp = torch.multinomial(probs, num_samples=2, generator=gen)
            loss = 0.5 * (
                F.cross_entropy(logits, y_samp[:, 0], reduction="sum")
                + F.cross_entropy(logits, y_samp[:, 1], reduction="sum")
            )
            loss.backward()
            total_tokens += tok.numel()
            del out, logits, probs, y_samp, loss

    emb_hook.remove()
    for h in hooks:
        h.remove()

    result: Dict[Tuple[int, str], torch.Tensor] = {}
    for k, v in fisher_out.items():
        result[k] = (v / max(total_tokens, 1)).to(torch.float32).cpu()
    del fisher_out
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def build_liqk_v2_fisher_slice_profile(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    candidate_specs: Optional[List[str]] = None,
    batch_size: int = 8,
    attn_slice_rows: int = 256,
    mlp_slice_rows: int = 512,
    fisher_out: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    row_stride: int = 1,
) -> dict:
    """
    Builds the Fisher distortion profile `dKLD(slice, spec)` over row slices (256/512 rows)
    of all linear matrices of the model and computes a lossless, intra-slice-balanced
    permutation `mlp_perms[l]` of the SwiGLU MLP intermediate neurons in each layer.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_layers = model.config.num_hidden_layers

    if candidate_specs is None:
        candidate_specs = list(SPEC_DISTORTION_RATIO.keys())

    if fisher_out is None:
        fisher_out = compute_exact_row_fisher_sensitivities(model, cal_tokens, n_Menu_seqs=8, seq_len=1024)

    n_seqs = min(16, cal_tokens.shape[0])
    H_batches: List[torch.Tensor] = [
        model.model.embed_tokens(cal_tokens[i : i + batch_size].to(device)) for i in range(0, n_seqs, batch_size)
    ]

    seq_len = cal_tokens.shape[1]
    dummy_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    pos_emb = model.model.rotary_emb(H_batches[0][:1], dummy_pos)

    slices_info = []
    mlp_perms: Dict[int, torch.Tensor] = {}
    fisher_out_perm: Dict[Tuple[int, str], torch.Tensor] = {k: v.clone() for k, v in fisher_out.items()}

    for l_idx in range(n_layers):
        layer = model.model.layers[l_idx]
        cov_acc = {name: _CovAccumulator() for name in ("qkv", "o", "up", "down")}

        def make_cap(name):
            def fn(m, inp, out):
                cov_acc[name].add(inp[0])
            return fn

        h0 = layer.self_attn.q_proj.register_forward_hook(make_cap("qkv"))
        h1 = layer.self_attn.o_proj.register_forward_hook(make_cap("o"))
        h2 = layer.mlp.gate_proj.register_forward_hook(make_cap("up"))
        h3 = layer.mlp.down_proj.register_forward_hook(make_cap("down"))

        for b_i in range(len(H_batches)):
            H_batches[b_i] = layer(H_batches[b_i], position_embeddings=pos_emb)

        h0.remove()
        h1.remove()
        h2.remove()
        h3.remove()

        H_qkv = cov_acc["qkv"].get()
        H_o = cov_acc["o"].get()
        H_up = cov_acc["up"].get()
        H_down = cov_acc["down"].get()
        del cov_acc

        wg = layer.mlp.gate_proj.weight.data.float()
        wu = layer.mlp.up_proj.weight.data.float()
        sg = fisher_out[(l_idx, "mlp.gate_proj")].to(device).float()
        su = fisher_out[(l_idx, "mlp.up_proj")].to(device).float()
        diag_up = torch.diagonal(H_up).clamp(min=1e-8)
        eg = (wg * wg * diag_up.unsqueeze(0)).sum(dim=1)
        eu = (wu * wu * diag_up.unsqueeze(0)).sum(dim=1)
        mlp_channel_imp = sg * eg + su * eu
        if os.environ.get("LIQK_FISHER_STABLE", "1") == "1" and os.environ.get("LIQK_FISHER_PERM", "1") == "1":
            # near-equal importances must not get reordered by backward noise: 3 significant digits, stable sort
            mlp_channel_imp = torch.round(torch.log10(mlp_channel_imp.clamp(min=1e-30)) * 1000.0)
            perm_desc = torch.sort(mlp_channel_imp, descending=True, stable=True).indices
        else:
            perm_desc = torch.argsort(mlp_channel_imp, descending=True)

        # Interleave 2 superblocks of 256 within each 512-channel slice:
        # gate_proj/up_proj slices stay strictly sorted by Fisher sensitivity,
        # while paired down_proj superblocks get a balanced distribution of outliers!
        Kd = wg.shape[0]
        perm_bal2 = perm_desc.view(Kd // 512, QK_K, 2).transpose(1, 2).reshape(Kd)
        mlp_perms[l_idx] = perm_bal2.cpu()

        fisher_out_perm[(l_idx, "mlp.gate_proj")] = sg[perm_bal2].cpu()
        fisher_out_perm[(l_idx, "mlp.up_proj")] = su[perm_bal2].cpu()
        H_down_perm = H_down[perm_bal2][:, perm_bal2]

        hess_map = {
            "self_attn.q_proj": H_qkv,
            "self_attn.k_proj": H_qkv,
            "self_attn.v_proj": H_qkv,
            "self_attn.o_proj": H_o,
            "mlp.gate_proj": H_up,
            "mlp.up_proj": H_up,
            "mlp.down_proj": H_down_perm,
        }

        for idx_s, subname in enumerate(LINEAR_SUBMODULES):
            mod = layer.get_submodule(subname)
            W_f32 = mod.weight.data.float()
            if subname in ("mlp.gate_proj", "mlp.up_proj"):
                W_f32 = W_f32[perm_bal2]
            elif subname == "mlp.down_proj":
                W_f32 = W_f32[:, perm_bal2]

            M, K = W_f32.shape
            H_mat = hess_map[subname].float()
            s_row = fisher_out_perm[(l_idx, subname)].to(device).float()

            W_rot, H_rot, _ = compute_optimal_suh(
                W_f32, H_mat, seed=1000 + l_idx * 10 + idx_s, row_weights=s_row
            )
            diag_h = torch.diagonal(H_rot).clamp(min=1e-8)

            slice_step = mlp_slice_rows if subname in ("mlp.gate_proj", "mlp.up_proj") else attn_slice_rows
            slice_step = min(slice_step, M)
            n_slices = M // slice_step

            # With row_stride > 1, only every `row_stride`-th row is quantized and the sum is scaled back.
            # Off by default: on 3B with stride=4 the slice estimate error reaches p95 48% (outlier rows
            # drop out of the sample), and DP changes the format of 34% of parameters at 3.44 bpw
            stride = row_stride if slice_step % row_stride == 0 else 1
            rows = torch.arange(0, M, stride, device=device)
            W_rot_s = W_rot[rows]
            diff_3 = W_rot_s - quantize_liqk_v2_diag_rot(W_rot_s, diag_h, "LIQK3_V2")
            row_err_3 = ((diff_3 @ H_rot) * diff_3).sum(dim=1).clamp(min=1e-12)
            slice_klds_3 = (0.5 * s_row[rows] * row_err_3).view(n_slices, slice_step // stride).sum(dim=1) * stride

            for s_i in range(n_slices):
                r0, r1 = s_i * slice_step, (s_i + 1) * slice_step
                n_p = (r1 - r0) * K
                slices_info.append(
                    {
                        "layer": l_idx,
                        "subname": subname,
                        "r0": r0,
                        "r1": r1,
                        "params": n_p,
                        "raw_kld3": float(slice_klds_3[s_i].item()),
                    }
                )

    # Soft winsorization of per-parameter slice sensitivity to [0.20x, 4.5x] of the mean:
    # preserves the exact Fisher ranking (1B and 3B) while protecting the knapsack from isolated spikes
    kpp = np.array([s["raw_kld3"] / s["params"] for s in slices_info], dtype=np.float64)
    mean_kpp = float(np.mean(kpp))
    for s in slices_info:
        kpp_clamped = min(max(s["raw_kld3"] / s["params"], 0.20 * mean_kpp), 4.5 * mean_kpp)
        base_c3 = kpp_clamped * s["params"]
        s["costs"] = {sp: base_c3 * SPEC_DISTORTION_RATIO[sp] for sp in candidate_specs}

    return {
        "slices": slices_info,
        "mlp_perms": mlp_perms,
        "fisher_out": {k: v.cpu() for k, v in fisher_out_perm.items()},
    }


def solve_fisher_slice_dp(
    profile: dict,
    target_bpw: float,
    candidate_specs: Optional[List[str]] = None,
    sens_power: float = 1.0,
) -> Dict[Tuple[int, str], List[Tuple[int, int, str]]]:
    """
    Solves the global knapsack problem (Multiple-Choice Knapsack DP) over all row slices of the model,
    minimizing the total Fisher-predicted KLD increase subject to the budget `<= target_bpw`.
    `sens_power` < 1 smooths per-parameter slice sensitivity: the cost is multiplied by (kpp / mean)^(p - 1).
    """
    slices = profile["slices"]
    if candidate_specs is None:
        if target_bpw <= 2.15:
            candidate_specs = [
                "LIQK1_88_V2",
                "LIQK2_00_V2",
                "LIQK2_06_V2",
                "LIQK2_25_V2",
                "LIQK2_0_V2",
                "LIQK2_V2",
                "LIQK2_5_V2",
            ]
        elif target_bpw <= 2.45:
            candidate_specs = ["LIQK2_25_V2", "LIQK2_0_V2", "LIQK2_V2", "LIQK2_5_V2", "LIQK3_25_V2", "LIQK3_V2"]
        elif target_bpw <= 3.15:
            candidate_specs = [
                "LIQK2_V2",
                "LIQK2_5_V2",
                "LIQK2_7_V2",
                "LIQK3_25_V2",
                "LIQK3_38_V2",
                "LIQK3_V2",
                "LIQK3_5_V2",
            ]
        elif target_bpw <= 3.55:
            candidate_specs = [
                "LIQK3_25_V2",
                "LIQK3_31_V2",
                "LIQK3_38_V2",
                "LIQK3_V2",
                "LIQK3_3_V2",
                "LIQK3_5_V2",
                "LIQK4_31_V2",
                "LIQK4_V2",
            ]
        elif target_bpw <= 4.05:
            candidate_specs = [
                "LIQK3_38_V2",
                "LIQK3_V2",
                "LIQK3_5_V2",
                "LIQK4_31_V2",
                "LIQK4_38_V2",
                "LIQK4_V2",
                "LIQK4_5_V2",
                "LIQK5_V2",
            ]
        elif target_bpw <= 4.35:
            candidate_specs = [
                "LIQK3_5_V2",
                "LIQK4_31_V2",
                "LIQK4_38_V2",
                "LIQK4_V2",
                "LIQK4_5_V2",
                "LIQK5_V2",
            ]
        elif target_bpw <= 4.65:
            candidate_specs = [
                "LIQK4_38_V2",
                "LIQK4_V2",
                "LIQK4_5_V2",
            ]
        elif target_bpw <= 5.15:
            candidate_specs = [
                "LIQK4_V2",
                "LIQK4_5_V2",
                "LIQK5_31_V2",
                "LIQK5_38_V2",
                "LIQK5_V2",
                "LIQK5_5_V2",
            ]
        elif target_bpw <= 5.85:
            candidate_specs = [
                "LIQK4_5_V2",
                "LIQK5_31_V2",
                "LIQK5_38_V2",
                "LIQK5_V2",
                "LIQK5_5_V2",
                "LIQK6_31_V2",
                "LIQK6_V2",
            ]
        elif target_bpw <= 6.85:
            candidate_specs = [
                "LIQK5_5_V2",
                "LIQK6_31_V2",
                "LIQK6_V2",
                "LIQK7_V2",
                "LIQK8_00_V2",
            ]
        else:
            candidate_specs = [
                "LIQK6_V2",
                "LIQK7_V2",
                "LIQK8_00_V2",
                "LIQK8_V2",
            ]

    total_params = sum(s["params"] for s in slices)
    max_total_bytes = int(round(total_params * target_bpw / 8.0))
    sp0 = candidate_specs[0]
    kpp = np.array([s["costs"][sp0] / (s["params"] * SPEC_DISTORTION_RATIO[sp0]) for s in slices], dtype=np.float64)
    sens_mult = (kpp / kpp.mean()) ** (sens_power - 1.0)
    min_bytes_per_slice = []
    items = []
    for s, mult in zip(slices, sens_mult):
        n_p = s["params"]
        opts = []
        for sp in candidate_specs:
            b_cost = int(round(n_p * LIQK_V2_SPECS[sp].bpw / 8.0))
            d_cost = s["costs"][sp] * float(mult)
            opts.append((sp, b_cost, d_cost))
        opts.sort(key=lambda x: (x[1], x[2]))
        # among formats of the same size keep only the lowest-distortion one: otherwise a step down with zero savings
        # gets stuck and the budget may be exceeded
        opts = [o for i, o in enumerate(opts) if i == 0 or o[1] != opts[i - 1][1]]
        b_min = opts[0][1]
        min_bytes_per_slice.append(b_min)
        items.append((s, [(sp, b_c - b_min, d_c) for sp, b_c, d_c in opts]))

    base_bytes = sum(min_bytes_per_slice)
    rem_bytes = max(0, max_total_bytes - base_bytes)
    unit_bytes = max(512, rem_bytes // 5000)
    max_units = rem_bytes // unit_bytes + 50

    n_items = len(items)
    dp = np.full((n_items + 1, max_units + 1), np.inf, dtype=np.float64)
    choice = np.zeros((n_items + 1, max_units + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    for i, (_, opts) in enumerate(items, start=1):
        for c_idx, (sp, delta_b, d_cost) in enumerate(opts):
            u_cost = int(round(delta_b / float(unit_bytes))) if delta_b > 0 else 0
            if u_cost <= max_units:
                prev_vals = dp[i - 1, : max_units + 1 - u_cost] + d_cost
                target_slice = dp[i, u_cost:]
                mask = prev_vals < target_slice
                target_slice[mask] = prev_vals[mask]
                choice[i, u_cost:][mask] = c_idx

    target_u = min(max_units, rem_bytes // unit_bytes)
    best_u = int(np.argmin(dp[n_items, : target_u + 1]))
    chosen_indices = [0] * n_items
    cur_u = best_u
    for i in range(n_items, 0, -1):
        _, opts = items[i - 1]
        c_idx = int(choice[i, cur_u])
        chosen_indices[i - 1] = c_idx
        _, delta_b, _ = opts[c_idx]
        u_cost = int(round(delta_b / float(unit_bytes))) if delta_b > 0 else 0
        cur_u -= u_cost

    cur_total_bytes = base_bytes + sum(items[idx][1][chosen_indices[idx]][1] for idx in range(n_items))

    while cur_total_bytes > max_total_bytes:
        best_down = None
        best_ratio = float("inf")
        for idx in range(n_items):
            c_cur = chosen_indices[idx]
            if c_cur > 0:
                opts = items[idx][1]
                db_save = opts[c_cur][1] - opts[c_cur - 1][1]
                dkld_loss = max(0.0, opts[c_cur - 1][2] - opts[c_cur][2])
                if db_save > 0 and (dkld_loss / db_save) < best_ratio:
                    best_ratio = dkld_loss / db_save
                    best_down = idx
        if best_down is None:
            break
        c_cur = chosen_indices[best_down]
        opts = items[best_down][1]
        cur_total_bytes -= opts[c_cur][1] - opts[c_cur - 1][1]
        chosen_indices[best_down] = c_cur - 1

    while True:
        best_up = None
        best_gain_ratio = 0.0
        rem = max_total_bytes - cur_total_bytes
        for idx in range(n_items):
            c_cur = chosen_indices[idx]
            opts = items[idx][1]
            if c_cur + 1 < len(opts):
                db_add = opts[c_cur + 1][1] - opts[c_cur][1]
                dkld_gain = opts[c_cur][2] - opts[c_cur + 1][2]
                if 0 < db_add <= rem and (dkld_gain / db_add) > best_gain_ratio:
                    best_gain_ratio = dkld_gain / db_add
                    best_up = idx
        if best_up is None:
            break
        c_cur = chosen_indices[best_up]
        opts = items[best_up][1]
        cur_total_bytes += opts[c_cur + 1][1] - opts[c_cur][1]
        chosen_indices[best_up] = c_cur + 1

    assignment: Dict[Tuple[int, str], List[Tuple[int, int, str]]] = {}
    for idx in range(n_items):
        s, opts = items[idx]
        sp, _, _ = opts[chosen_indices[idx]]
        key = (s["layer"], s["subname"])
        assignment.setdefault(key, []).append((s["r0"], s["r1"], sp))

    for key in assignment:
        assignment[key].sort(key=lambda x: x[0])
        merged: List[Tuple[int, int, str]] = []
        for r0, r1, sp in assignment[key]:
            if merged and merged[-1][2] == sp and merged[-1][1] == r0:
                merged[-1] = (merged[-1][0], r1, sp)
            else:
                merged.append((r0, r1, sp))
        assignment[key] = merged

    return assignment


@torch.no_grad()
def precompute_ref_final_hidden_states(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    n_seqs: int = 64,
    seq_len: int = 512,
    batch_size: int = 4,
) -> torch.Tensor:
    """
    Caches the output hidden states of the reference BF16 model before `lm_head`
    on `n_seqs x seq_len` calibration tokens in CPU RAM (~134 MB for 64x512).
    """
    device = next(model.parameters()).device
    model.eval()
    n_use = min(n_seqs, cal_tokens.shape[0])
    out_list = []
    for i in range(0, n_use, batch_size):
        xb = cal_tokens[i : i + batch_size, :seq_len].to(device)
        h = model.model(xb, use_cache=False).last_hidden_state.to(torch.bfloat16).cpu()
        out_list.append(h)
    return torch.cat(out_list, dim=0)


def polish_model_scales_kld(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    ref_hidden_cpu: torch.Tensor,
    n_steps: int = 100,
    lr: float = 5e-4,
    batch_size: int = 2,
    reg_wd: float = 0.02,
    rev_kl_weight: float = 0.12,
    ce_weight: float = 0.0,
) -> float:
    """
    Performs a fast (`~11 s` on 1B, `~20 s` on 3B) end-to-end refinement of the continuous `fp16` row
    scales `d` (`alpha`) and input `fp16` scales `suh` (`beta`) with `RMSNorm` weights frozen,
    minimizing `Forward KL + 0.12 * Reverse KL + 0.08 * Cross-Entropy` to lower both
    KLD and perplexity (PPL) without a single extra bit in storage/inference.
    """
    t0 = time.perf_counter()
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_seqs, seq_len, _ = ref_hidden_cpu.shape
    cal_pol = cal_tokens[:n_seqs, :seq_len].to(device)

    scales_alpha: Dict[Tuple[int, str], nn.Parameter] = {}
    scales_beta: Dict[Tuple[int, str], nn.Parameter] = {}
    scale_params: List[nn.Parameter] = []
    hooks = []

    for l in range(model.config.num_hidden_layers):
        layer = model.model.layers[l]
        for sub in LINEAR_SUBMODULES:
            mod = layer.get_submodule(sub)
            M, K = mod.weight.shape
            alpha = nn.Parameter(torch.ones(M, dtype=torch.float32, device=device))
            beta = nn.Parameter(torch.ones(K, dtype=torch.float32, device=device))
            scales_alpha[(l, sub)] = alpha
            scales_beta[(l, sub)] = beta
            scale_params.extend([alpha, beta])

            def make_hook(a, b):
                def pre_hook(m, args):
                    return (args[0] * b.to(args[0].dtype),)

                def post_hook(m, args, out):
                    return out * a.to(out.dtype)

                return pre_hook, post_hook

            pre_h, post_h = make_hook(alpha, beta)
            hooks.append(mod.register_forward_pre_hook(pre_h))
            hooks.append(mod.register_forward_hook(post_h))

    opt = torch.optim.AdamW(scale_params, lr=lr, weight_decay=0.0)

    for step in range(n_steps):
        progress = step / max(n_steps - 1, 1)
        cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        idx = (step * batch_size) % n_seqs
        x_b = cal_pol[idx : idx + batch_size]
        h_ref_b = ref_hidden_cpu[idx : idx + batch_size].to(device=device, dtype=model.dtype)

        opt.zero_grad()
        h_q_b = model.model(x_b, use_cache=False).last_hidden_state
        h_ref_flat = h_ref_b[:, :-1, :].reshape(-1, h_ref_b.shape[-1])
        h_q_flat = h_q_b[:, :-1, :].reshape(-1, h_q_b.shape[-1])
        targets_flat = x_b[:, 1:].reshape(-1)
        n_tok = h_q_flat.shape[0]
        grad_h_q = torch.empty_like(h_q_flat)

        chunk_sz = 256
        for c0 in range(0, n_tok, chunk_sz):
            c1 = min(c0 + chunk_sz, n_tok)
            w_fraction = (c1 - c0) / float(n_tok)
            q_chunk = h_q_flat[c0:c1].detach().requires_grad_(True)
            t_chunk = targets_flat[c0:c1]
            with torch.no_grad():
                ref_logits = model.lm_head(h_ref_flat[c0:c1]).float()
                ref_logp = F.log_softmax(ref_logits, dim=-1)
                ref_p = ref_logp.exp()
            q_logits = model.lm_head(q_chunk).float()
            q_logp = F.log_softmax(q_logits, dim=-1)
            q_p = q_logp.exp()

            fwd_kl = (ref_p * (ref_logp - q_logp)).sum(dim=-1).mean()
            rev_kl = (q_p * (q_logp - ref_logp)).sum(dim=-1).mean()
            ce_loss = F.nll_loss(q_logp, t_chunk, reduction="mean")
            loss_chunk = (fwd_kl + rev_kl_weight * rev_kl + ce_weight * ce_loss) * w_fraction
            loss_chunk.backward()
            grad_h_q[c0:c1] = q_chunk.grad

        h_q_b[:, :-1, :].reshape(-1, h_q_b.shape[-1]).backward(grad_h_q)
        if reg_wd > 0.0:
            for p in scale_params:
                if p.grad is not None:
                    p.grad.add_(p.data - 1.0, alpha=reg_wd)
        opt.step()

    for h in hooks:
        h.remove()

    with torch.no_grad():
        for (l, sub), alpha in scales_alpha.items():
            beta = scales_beta[(l, sub)]
            mod = model.model.layers[l].get_submodule(sub)
            a_fp16 = alpha.to(torch.float16).float()
            b_fp16 = beta.to(torch.float16).float()
            mod.weight.data.copy_(
                (mod.weight.data.float() * a_fp16.unsqueeze(1) * b_fp16.unsqueeze(0)).to(mod.weight.dtype)
            )
    for p in model.parameters():
        p.requires_grad_(False)

    del opt, scale_params, scales_alpha, scales_beta, cal_pol
    torch.cuda.empty_cache()
    return time.perf_counter() - t0


@torch.no_grad()
def quantize_llama_model_liqk_v2(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    spec_or_assignment: str | Dict[Tuple[int, str], List[Tuple[int, int, str]]],
    mlp_perms: Optional[Dict[int, torch.Tensor]] = None,
    fisher_out: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    batch_size: int = 8,
    damp: float = 0.02,
    cd_sweeps: int = 2,
    polish_steps: int = 80,
    polish_lr: float = 5e-4,
    out_rot: bool = False,
) -> dict:
    """
    Sequential layer-by-layer transformer quantizer into the `.liqk v2` format:
      1. If `mlp_perms` is given, applies the lossless permutation of MLP intermediate channels.
      2. In each layer, collects exact Hessians on the already-quantized inputs and quantizes each matrix
         (or its row slices) with `LUT-Decoupled Warp-Register Bank + CS-suh + Block16-LDLQ`.
      3. Optionally runs a fast end-to-end polish of the continuous `fp16` scales (`d`, `suh`).
    """
    t_start = time.perf_counter()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()
    damp = float(os.environ.get("LIQK_DAMP", damp))  # damping of the LDLQ Hessians (experiment)
    # TF32 for quantization fp32 GEMMs (LDLQ feedback, scale fitting): error ~1e-3 vs quantization noise ~1e-1
    tf32_prev = torch.backends.cuda.matmul.allow_tf32
    if os.environ.get("LIQK_TF32", "0") == "1":
        torch.backends.cuda.matmul.allow_tf32 = True

    ref_hidden_cpu = None
    if polish_steps > 0:
        ref_hidden_cpu = precompute_ref_final_hidden_states(
            model, cal_tokens, n_seqs=64, seq_len=512, batch_size=4
        )

    if fisher_out is None:
        fisher_out = compute_exact_row_fisher_sensitivities(model, cal_tokens, n_Menu_seqs=8, seq_len=1024)

    n_seqs, seq_len = cal_tokens.shape
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_heads
    n_rep = n_heads // n_kv_heads

    # The calibration residual stream lives on the GPU for the whole quantization. Matrix inputs (RMSNorm, attention,
    # SwiGLU) are recomputed from it in batches and H is accumulated in place: intermediate activations
    # are neither stored nor shuttled through the CPU (that used to be ~2/3 of the loop time on 3B).
    H_batches: List[torch.Tensor] = [
        model.model.embed_tokens(cal_tokens[i : i + batch_size].to(device)) for i in range(0, n_seqs, batch_size)
    ]

    dummy_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    pos_emb = model.model.rotary_emb(H_batches[0][:1], dummy_pos)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    total_params = 0
    total_bits = 0.0
    suh_dict: Dict[Tuple[int, str], torch.Tensor] = {}
    cache_act = os.environ.get("LIQK_CACHE_ACT", "0") == "1"  # gave no speedup (19.3 vs 19.7 s), and LIQK_PAR needs the memory
    par = os.environ.get("LIQK_PAR", "0") == "1"  # OOM on 16 GB: each stream has its own allocator pool
    bits_lock = threading.Lock()
    _linalg_warm = [False]

    def _cache_fits(batches: List[torch.Tensor], width_ratio: float) -> bool:
        """Whether an activation cache of width width_ratio * hidden fits in free GPU memory with a 2 GB margin."""
        need = sum(b.numel() * b.element_size() for b in batches) * width_ratio
        return torch.cuda.mem_get_info(device)[0] > need + (2 << 30)

    def quantize_linear(l_idx: int, subname: str, H: torch.Tensor, seed: int, mat_damp: float):
        nonlocal total_params, total_bits
        mod = model.model.layers[l_idx].get_submodule(subname)
        M, K = mod.weight.shape
        if isinstance(spec_or_assignment, str):
            r_specs = [(0, M, spec_or_assignment)]
        else:
            r_specs = spec_or_assignment[(l_idx, subname)]
        q_res, w_hat, _ = quantize_matrix_liqk_v2(
            mod.weight.data,
            H,
            r_specs,
            seed=seed,
            damp=mat_damp,
            row_weights=fisher_out[(l_idx, subname)].to(device),
            cd_sweeps=cd_sweeps,
            out_rot=out_rot,
        )
        suh_dict[(l_idx, subname)] = q_res["suh"].detach().cpu()
        mod.weight.data.copy_(w_hat.to(dtype=dtype, device=device))
        with bits_lock:
            for r0, r1, sp in r_specs:
                n_p = (r1 - r0) * K
                total_params += n_p
                total_bits += n_p * LIQK_V2_SPECS[sp].bpw

    def quantize_group(l_idx: int, names: List[str], H: torch.Tensor, seed0: int, mat_damp: float):
        """Matrices sharing H (q / k / v, gate / up) are independent: with LIQK_PAR=1 they are quantized in parallel threads
        with their own CUDA streams; small Viterbi calls overlap on the GPU, the result is the same."""
        if not par or len(names) == 1:
            for i, name in enumerate(names):
                quantize_linear(l_idx, name, H, seed0 + i, mat_damp)
            return
        main = torch.cuda.current_stream(device)
        streams = [torch.cuda.Stream(device) for _ in names]
        if not _linalg_warm[0]:  # lazy init of torch's linalg backend is not thread-safe, so warm it up in the main thread
            e = torch.eye(8, device=device)
            torch.linalg.solve_triangular(torch.linalg.cholesky(e), e, upper=False)
            torch.linalg.solve(e, e)
            torch.cholesky_inverse(e)
            torch.linalg.cholesky(e, upper=True)
            _linalg_warm[0] = True

        def job(i: int):
            streams[i].wait_stream(main)
            with torch.cuda.stream(streams[i]):
                quantize_linear(l_idx, names[i], H, seed0 + i, mat_damp)
            streams[i].synchronize()

        with ThreadPoolExecutor(len(names)) as ex:
            list(ex.map(job, range(len(names))))

    def attn_core(layer: nn.Module, x_norm: torch.Tensor) -> torch.Tensor:
        """o_proj input: q/k/v -> RoPE -> causal SDPA."""
        bsz, q_len, _ = x_norm.shape
        q = layer.self_attn.q_proj(x_norm).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        k = layer.self_attn.k_proj(x_norm).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(x_norm).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)
        ao = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return ao.transpose(1, 2).contiguous().view(bsz, q_len, -1)

    def mlp_mid(layer: nn.Module, hb: torch.Tensor) -> torch.Tensor:
        """down_proj input: post_attention_layernorm -> SwiGLU."""
        x_norm2 = layer.post_attention_layernorm(hb)
        return F.silu(layer.mlp.gate_proj(x_norm2)) * layer.mlp.up_proj(x_norm2)

    for l_idx in range(n_layers):
        layer = model.model.layers[l_idx]

        if mlp_perms is not None and l_idx in mlp_perms:
            perm_l = mlp_perms[l_idx].to(device)
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[perm_l].contiguous()
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[perm_l].contiguous()
            layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, perm_l].contiguous()

        # --- Sub-block A: q_proj, k_proj, v_proj ---
        acc = _CovAccumulator()
        for hb in H_batches:
            acc.add(layer.input_layernorm(hb))
        H_qkv = acc.get()
        quantize_group(l_idx, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], H_qkv, 1000 + l_idx * 10, damp)
        del acc, H_qkv

        # --- Sub-block B: SDPA -> o_proj, then residual update ---
        # The o_proj input does not depend on o_proj quantization: cache it between collecting H_o and the residual update
        # (otherwise attn_core is computed twice) if the GPU has enough memory.
        acc = _CovAccumulator()
        ao_cache = [] if cache_act and _cache_fits(H_batches, 1.0) else None
        for hb in H_batches:
            ao = attn_core(layer, layer.input_layernorm(hb))
            acc.add(ao)
            if ao_cache is not None:
                ao_cache.append(ao)
        H_o = acc.get()
        quantize_linear(l_idx, "self_attn.o_proj", H_o, 1000 + l_idx * 10 + 3, damp)
        del acc, H_o
        for b_i, hb in enumerate(H_batches):
            ao = ao_cache[b_i] if ao_cache is not None else attn_core(layer, layer.input_layernorm(hb))
            H_batches[b_i] = hb + layer.self_attn.o_proj(ao)
        del ao_cache

        # --- Sub-block C: gate_proj, up_proj ---
        acc = _CovAccumulator()
        for hb in H_batches:
            acc.add(layer.post_attention_layernorm(hb))
        H_up = acc.get()
        quantize_group(l_idx, ["mlp.gate_proj", "mlp.up_proj"], H_up, 1000 + l_idx * 10 + 4, damp)
        del acc, H_up

        # --- Sub-block D: SwiGLU -> down_proj, then residual update (the down_proj input is cached the same way) ---
        acc = _CovAccumulator()
        inter = layer.mlp.gate_proj.weight.shape[0] / layer.mlp.gate_proj.weight.shape[1]
        mid_cache = [] if cache_act and _cache_fits(H_batches, inter) else None
        for hb in H_batches:
            mid = mlp_mid(layer, hb)
            acc.add(mid)
            if mid_cache is not None:
                mid_cache.append(mid)
        H_down = acc.get()
        quantize_linear(l_idx, "mlp.down_proj", H_down, 1000 + l_idx * 10 + 6, damp * 1.5)
        del acc, H_down
        for b_i, hb in enumerate(H_batches):
            mid = mid_cache[b_i] if mid_cache is not None else mlp_mid(layer, hb)
            H_batches[b_i] = hb + layer.mlp.down_proj(mid)
        del mid_cache

    del H_batches
    model._liqk_suh_dict = suh_dict
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = tf32_prev

    polish_sec = 0.0
    if polish_steps > 0 and ref_hidden_cpu is not None:
        with torch.enable_grad():
            polish_sec = polish_model_scales_kld(
                model,
                cal_tokens,
                ref_hidden_cpu,
                n_steps=polish_steps,
                lr=polish_lr,
                batch_size=2,
            )

    total_sec = time.perf_counter() - t_start
    return {
        "linear_params": total_params,
        "linear_bytes": int(round(total_bits / 8.0)),
        "linear_bpw": total_bits / max(total_params, 1),
        "quant_sec": total_sec,
        "polish_sec": polish_sec,
    }
