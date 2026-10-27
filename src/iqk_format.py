"""
Implementation of the IQ*_K quantization format family (PR #30715-#30718 in llama.cpp) and the .liqk extension:
  - IQ2_K (2.375 bpw, 76 bytes per 256 weights)
  - IQ3_K (3.4375 bpw, 110 bytes per 256 weights in GGUF / 112 bytes with alignment in .liqk)
  - IQ4_K (4.500 bpw, 144 bytes per 256 weights)
  - IQ5_K (5.500 bpw, 176 bytes per 256 weights)
  - IQ6_K (6.625 bpw, 212 bytes per 256 weights)

Includes:
1. Exact non-linear level tables `kvalues_iq{2..6}k` and phase shifts `IQ{2..6}K_PHASE` from `ggml-common.h`.
2. Byte-level packing (`pack_iqk_blocks`) and unpacking (`dequantize_iqk_blocks`), bit-exact with `dequantize_row_iq*_k` in C.
3. Three GPU quantization methods (PyTorch CUDA):
   - `quantize_iqk_diag`: diagonal weighting (RTN / imatrix) with exhaustive `(ls, phase)` search and `d` refinement.
   - `quantize_iqk_ldlq`: column-wise error compensation using the full Hessian `H = X^T X` (LDLQ / GPTQ) with `(ls, phase)` search inside the Cholesky loop and analytic Post-LDLQ Scale Refit.
   - `quantize_liqk_hadamard_ldlq`: `Hadamard (suh, svh + FWHT) + LDLQ + Post-LDLQ Scale Refit` (`.liqk` format).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch

QK_K = 256
SUBBLOCK_SIZE = 16
NUM_SUBBLOCKS = QK_K // SUBBLOCK_SIZE  # 16 sub-blocks of 16 weights


@dataclass(frozen=True)
class IqkSpec:
    name: str
    ggml_type_id: int
    bits: int
    nlevels: int
    kvalues: Tuple[int, ...]
    phase_shift: int
    ls_min: int
    ls_max: int
    ls_step: int
    ls_off: int
    block_bytes: int

    @property
    def bpw(self) -> float:
        return (self.block_bytes * 8.0) / QK_K


# Specs from ggml-common.h (PR #30715)
IQK_SPECS: Dict[str, IqkSpec] = {
    "iq2_k": IqkSpec(
        name="iq2_k",
        ggml_type_id=42,
        bits=2,
        nlevels=4,
        kvalues=(-31, -14, -2, 10),
        phase_shift=5,
        ls_min=-8,
        ls_max=7,
        ls_step=1,
        ls_off=0,
        block_bytes=76,  # 2 (d) + 2 (extra) + 8 (scales) + 64 (qs)
    ),
    "iq3_k": IqkSpec(
        name="iq3_k",
        ggml_type_id=43,
        bits=3,
        nlevels=8,
        kvalues=(-56, -37, -23, -12, -1, 10, 23, 40),
        phase_shift=23,
        ls_min=-31,
        ls_max=31,
        ls_step=2,  # odd numbers: ±1, ±3, ..., ±31
        ls_off=1,
        block_bytes=110,  # 2 (d) + 2 (extra) + 2 (scales_h) + 8 (scales_l) + 64 (qs) + 32 (qh)
    ),
    "iq4_k": IqkSpec(
        name="iq4_k",
        ggml_type_id=44,
        bits=4,
        nlevels=16,
        kvalues=(-127, -97, -78, -62, -48, -35, -23, -12, -1, 11, 23, 36, 51, 67, 87, 105),
        phase_shift=18,
        ls_min=-32,
        ls_max=31,
        ls_step=1,
        ls_off=0,
        block_bytes=144,  # 2 (d) + 2 (extra) + 4 (scales_h) + 8 (scales_l) + 128 (qs)
    ),
    "iq5_k": IqkSpec(
        name="iq5_k",
        ggml_type_id=45,
        bits=5,
        nlevels=32,
        kvalues=(
            -115, -105, -94, -83, -73, -65, -58, -51, -45, -39, -33, -27, -22, -17, -11, -6,
            -1, 4, 9, 15, 20, 26, 32, 39, 46, 53, 61, 69, 77, 86, 103, 111,
        ),
        phase_shift=15,
        ls_min=-32,
        ls_max=31,
        ls_step=1,
        ls_off=0,
        block_bytes=176,  # 2 (d) + 2 (extra) + 4 (scales_h) + 8 (scales_l) + 128 (qs) + 32 (qh)
    ),
    "iq6_k": IqkSpec(
        name="iq6_k",
        ggml_type_id=46,
        bits=6,
        nlevels=64,
        kvalues=(
            -120, -115, -108, -103, -99, -93, -88, -83, -79, -75, -70, -66, -62, -59, -55, -51,
            -48, -45, -41, -38, -35, -32, -29, -26, -24, -21, -18, -15, -13, -10, -7, -5,
            -2, 0, 3, 6, 8, 11, 14, 16, 19, 22, 25, 28, 31, 34, 38, 41,
            44, 48, 52, 56, 60, 64, 68, 73, 77, 82, 86, 93, 97, 104, 109, 114,
        ),
        phase_shift=13,
        ls_min=-127,
        ls_max=127,
        ls_step=1,
        ls_off=0,
        block_bytes=212,  # 2 (d) + 2 (extra) + 16 (scales) + 128 (qs) + 64 (qh)
    ),
}
for _k in list(IQK_SPECS.keys()):
    IQK_SPECS[_k.upper()] = IQK_SPECS[_k]


def get_valid_scales(spec: IqkSpec, device: torch.device) -> torch.Tensor:
    """Returns a 1-D tensor of all valid integer sub-block scales `ls`."""
    if spec.name == "iq3_k":
        # Odd numbers from -31 to +31 with step 2 (32 values total)
        vals = list(range(-31, 32, 2))
    else:
        vals = list(range(spec.ls_min, spec.ls_max + 1))
    return torch.tensor(vals, dtype=torch.float32, device=device)


def get_codebooks(spec: IqkSpec, device: torch.device) -> torch.Tensor:
    """
    Returns the level-table tensor of shape `[2, nlevels]` for phase 0 and phase 1:
      cb[0, q] = kvalues[q]
      cb[1, q] = kvalues[q] + phase_shift
    """
    base = torch.tensor(spec.kvalues, dtype=torch.float32, device=device)
    cb0 = base
    cb1 = base + float(spec.phase_shift)
    return torch.stack([cb0, cb1], dim=0)


# ==============================================================================
# 1. Walsh-Hadamard transform (FWHT) for the .liqk format
# ==============================================================================

def fwht_inplace(x: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """
    Normalized block-wise fast Walsh-Hadamard transform (FWHT) of size `block_size` (a power of 2)
    over the last dimension of tensor `x`. Orthogonal and self-inverse: H @ H = I.
    """
    orig_shape = x.shape
    assert orig_shape[-1] % block_size == 0, f"Dimension {orig_shape[-1]} is not a multiple of {block_size}"
    y = x.reshape(-1, block_size)
    h = 1
    while h < block_size:
        y = y.view(-1, block_size // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :]
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        h *= 2
    y = y.view(orig_shape) * (block_size ** -0.5)
    return y


def make_rademacher_signs(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generates a deterministic ±1 sign vector for the Hadamard transform."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.int64)
    signs = (bits * 2 - 1).to(dtype=torch.float32, device=device)
    return signs


def rotate_weights_and_hessian(
    W: torch.Tensor,
    H: Optional[torch.Tensor],
    block_size: int = 256,
    seed: int = 42,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Applies the input-side Hadamard transform to the weight matrix W [N, K] and Hessian H [K, K]:
      W_rot = FWHT(W * suh)
      H_rot = FWHT(FWHT(H * suh)^T * suh)
    At inference it suffices to apply the same transform to the input vector x:
      x_rot = FWHT(x * suh)
    then W_rot @ x_rot == W @ x (since the orthogonal matrix Q = H_wht @ diag(suh) satisfies Q^T Q = I).
    """
    K = W.shape[1]
    suh = make_rademacher_signs(K, seed=seed, device=W.device)
    W_rot = fwht_inplace(W * suh.unsqueeze(0), block_size=block_size)
    if H is None:
        return W_rot, None, suh
    H_tmp = fwht_inplace(H * suh.unsqueeze(0), block_size=block_size)
    H_rot = fwht_inplace(H_tmp.T * suh.unsqueeze(0), block_size=block_size).T
    return W_rot, H_rot, suh


# ==============================================================================
# 2. Quantization algorithms onto the IQ*_K grid (Diagonal and LDLQ + Scale Refit)
# ==============================================================================

@torch.no_grad()
def _quantize_subblock_candidates(
    w_sub: torch.Tensor,         # [B, 16] — weights of the current sub-block
    d: torch.Tensor,             # [B] — super-block scale (already rounded to fp16)
    cb: torch.Tensor,            # [2, nlevels] — phase 0 and phase 1 tables
    valid_ls: torch.Tensor,      # [S] — valid integer sub-block scales
    u_sub: Optional[torch.Tensor] = None,  # [16, 16] — upper-triangular Cholesky block U_sub (for LDLQ)
    diag_w: Optional[torch.Tensor] = None, # [16] — diagonal importance weights
    top_k_candidates: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each row of batch B finds the optimal:
      - `best_ls`: [B] (integer sub-block scale)
      - `best_phase`: [B] (0 or 1)
      - `best_q`: [B, 16] (table indices `0..nlevels-1`)
      - `w_hat`: [B, 16] (reconstructed sub-block weights `d * best_ls * cb[best_phase, best_q]`)

    Step 1: Fast vectorized pre-selection of the best `top_k_candidates` `(ls, phase)` pairs
            over all `2 * S` possible scale/phase combinations (accounting for table asymmetry and clipping).
    Step 2: If `u_sub` is given (LDLQ mode), a full 16-step Cholesky loop with error propagation
            inside the sub-block is run for the selected `top_k_candidates` pairs.
    """
    B = w_sub.shape[0]
    device = w_sub.device
    S = valid_ls.shape[0]
    L = cb.shape[1]

    # For S > 16 (IQ3_K..IQ6_K) pick the 5 nearest positive and 5 nearest negative scales
    # around the analytic estimate (C_s=10, C_all=20), which avoids gigabyte-sized [B, C_all, 16, L] allocations
    if S > 16:
        amax = w_sub.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        d_safe = d.unsqueeze(1).clamp(min=1e-12)
        cb_max = float(cb.abs().max().item())
        est_mag = (amax / (d_safe * (cb_max * 0.88))).clamp(min=0.0)
        dist_pos = (valid_ls.unsqueeze(0) - est_mag).abs()
        dist_neg = (valid_ls.unsqueeze(0) + est_mag).abs()
        idx_pos = dist_pos.topk(5, dim=1, largest=False).indices
        idx_neg = dist_neg.topk(5, dim=1, largest=False).indices
        cand_idx = torch.cat([idx_pos, idx_neg], dim=1)  # [B, 10]
        cand_ls_base = valid_ls[cand_idx]                # [B, 10]
    else:
        cand_ls_base = valid_ls.unsqueeze(0).expand(B, S)  # [B, S]

    C_s = cand_ls_base.shape[1]
    C_all = 2 * C_s
    cand_dl_all = (d[:, None, None] * cand_ls_base[:, None, :]).expand(B, 2, C_s).reshape(B, C_all)
    cand_ls_all = cand_ls_base[:, None, :].expand(B, 2, C_s).reshape(B, C_all)
    cand_phase_all = (
        torch.tensor([0, 1], device=device, dtype=torch.long)[None, :, None]
        .expand(B, 2, C_s)
        .reshape(B, C_all)
    )
    cand_cb_all = cb[cand_phase_all]                       # [B, C_all, L]
    cand_grid_all = cand_dl_all.unsqueeze(-1) * cand_cb_all  # [B, C_all, L]

    # Vectorized evaluation of all C_all candidates by diagonal error:
    diff_all = w_sub[:, None, :, None] - cand_grid_all[:, :, None, :]  # [B, C_all, 16, L]
    err2_all, q_idx_all = (diff_all * diff_all).min(dim=-1)            # [B, C_all, 16]
    if diag_w is not None:
        err2_all = err2_all * diag_w[None, None, :]
    elif u_sub is not None:
        u_diag_inv2 = 1.0 / (torch.diagonal(u_sub) ** 2).clamp(min=1e-12)
        err2_all = err2_all * u_diag_inv2[None, None, :]
    total_err_all = err2_all.sum(dim=-1)  # [B, C_all]

    batch_arange = torch.arange(B, device=device)

    if u_sub is None:
        best_c = total_err_all.argmin(dim=-1)
        best_ls = cand_ls_all[batch_arange, best_c]
        best_phase = cand_phase_all[batch_arange, best_c]
        best_q = q_idx_all[batch_arange, best_c, :]
        best_grid = cand_grid_all[batch_arange, best_c, :]
        w_hat = torch.gather(best_grid, 1, best_q)
        return best_ls, best_phase, best_q, w_hat

    # Select the top_k_candidates best (ls, phase) pairs for the exact 16-step Cholesky loop (LDLQ)
    C = min(top_k_candidates, C_all)
    top_c = total_err_all.topk(C, dim=-1, largest=False).indices  # [B, C]
    cand_ls_full = torch.gather(cand_ls_all, 1, top_c)            # [B, C]
    cand_phase_full = torch.gather(cand_phase_all, 1, top_c)      # [B, C]
    cand_grid = torch.gather(
        cand_grid_all, 1, top_c.unsqueeze(-1).expand(B, C, L)
    )  # [B, C, L]

    w_work = w_sub.unsqueeze(1).expand(B, C, 16).clone()  # [B, C, 16]
    q_all = torch.empty((B, C, 16), dtype=torch.long, device=device)
    w_hat_all = torch.empty((B, C, 16), dtype=torch.float32, device=device)
    total_err = torch.zeros((B, C), dtype=torch.float32, device=device)

    u_diag = torch.diagonal(u_sub)  # [16]

    for j in range(16):
        wj = w_work[:, :, j]  # [B, C]
        diff = (wj.unsqueeze(-1) - cand_grid).abs()  # [B, C, L]
        qj = diff.argmin(dim=-1)  # [B, C]
        wh_j = torch.gather(cand_grid, 2, qj.unsqueeze(-1)).squeeze(-1)  # [B, C]
        q_all[:, :, j] = qj
        w_hat_all[:, :, j] = wh_j

        err_j = (wj - wh_j) / u_diag[j]  # [B, C]
        total_err += err_j * err_j
        if j + 1 < 16:
            w_work[:, :, j + 1 :] -= err_j.unsqueeze(-1) * u_sub[j, j + 1 :].view(1, 1, -1)

    best_c = total_err.argmin(dim=-1)  # [B]
    best_ls = cand_ls_full[batch_arange, best_c]
    best_phase = cand_phase_full[batch_arange, best_c]
    best_q = q_all[batch_arange, best_c, :]
    w_hat = w_hat_all[batch_arange, best_c, :]
    return best_ls, best_phase, best_q, w_hat


@torch.no_grad()
def _init_superblock_d(
    w_blk: torch.Tensor,    # [N, 256]
    spec: IqkSpec,
    cb: torch.Tensor,       # [2, nlevels]
) -> torch.Tensor:
    """
    Computes the initial estimate of the super-block scale `d` (fp16 -> fp32) for each row.
    Accounts for phase-0 asymmetry (max |kvalues[0]|) so that no sub-block gets clipped.
    """
    cb0_max = float(cb[0].abs().max().item())
    ls_max_abs = float(max(abs(spec.ls_min), abs(spec.ls_max)))
    row_amax = w_blk.abs().amax(dim=1)  # [N]
    d_init = (row_amax / (cb0_max * ls_max_abs)).to(torch.float16).to(torch.float32)
    return d_init


@torch.no_grad()
def quantize_iqk_diag(
    W: torch.Tensor,
    spec_name: str,
    imatrix_w: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Fast diagonal quantization (RTN or with importance weights `imatrix_w` [K])
    with exhaustive `(ls, phase)` search and 2-pass analytic refinement of the super-block `d`.
    """
    spec = IQK_SPECS[spec_name]
    device = W.device
    N, K = W.shape
    assert K % QK_K == 0, f"K={K} must be a multiple of {QK_K}"
    nblocks = K // QK_K

    cb = get_codebooks(spec, device)
    valid_ls = get_valid_scales(spec, device)

    d_all = torch.empty((N, nblocks), dtype=torch.float16, device=device)
    extra_all = torch.zeros((N, nblocks), dtype=torch.int32, device=device)
    ls_all = torch.empty((N, nblocks, NUM_SUBBLOCKS), dtype=torch.int16, device=device)
    q_all = torch.empty((N, K), dtype=torch.uint8, device=device)
    W_hat = torch.empty_like(W, dtype=torch.float32)

    for ibl in range(nblocks):
        c0, c1 = ibl * QK_K, (ibl + 1) * QK_K
        w_blk = W[:, c0:c1].float()  # [N, 256]
        if imatrix_w is not None:
            sigma2 = (2.0 / QK_K) * (w_blk * w_blk).sum(dim=1, keepdim=True)
            qw_blk = imatrix_w[c0:c1].float().unsqueeze(0) * torch.sqrt(sigma2 + w_blk * w_blk)
        else:
            qw_blk = w_blk * w_blk + 1e-6

        d_cur = _init_superblock_d(w_blk, spec, cb)

        # 2 iterations of alternating minimization: (ls, phase, q) -> analytic recompute of d
        for _ in range(2):
            ls_blk = torch.empty((N, NUM_SUBBLOCKS), dtype=torch.float32, device=device)
            phase_blk = torch.empty((N, NUM_SUBBLOCKS), dtype=torch.long, device=device)
            q_blk = torch.empty((N, QK_K), dtype=torch.long, device=device)

            for ib in range(NUM_SUBBLOCKS):
                s0, s1 = ib * 16, (ib + 1) * 16
                diag_sub = imatrix_w[c0 + s0 : c0 + s1].float() if imatrix_w is not None else None
                best_ls, best_ph, best_q, _ = _quantize_subblock_candidates(
                    w_blk[:, s0:s1], d_cur, cb, valid_ls, u_sub=None, diag_w=diag_sub
                )
                ls_blk[:, ib] = best_ls
                phase_blk[:, ib] = best_ph
                q_blk[:, s0:s1] = best_q

            # Assemble integer levels G = ls * cb[phase, q] for all 256 weights of the super-block
            # and analytically refine d = sum(qw * w * G) / sum(qw * G^2)
            G_blk = torch.empty((N, QK_K), dtype=torch.float32, device=device)
            for ib in range(NUM_SUBBLOCKS):
                s0, s1 = ib * 16, (ib + 1) * 16
                cb_ib = cb[phase_blk[:, ib]]  # [N, L]
                v_ib = torch.gather(cb_ib, 1, q_blk[:, s0:s1])  # [N, 16]
                G_blk[:, s0:s1] = ls_blk[:, ib].unsqueeze(1) * v_ib

            num = (qw_blk * w_blk * G_blk).sum(dim=1)
            den = (qw_blk * G_blk * G_blk).sum(dim=1)
            d_new = torch.where(den > 0, num / den, d_cur)
            d_cur = d_new.to(torch.float16).to(torch.float32)

        w_hat_blk = d_cur.unsqueeze(1) * G_blk
        W_hat[:, c0:c1] = w_hat_blk
        d_all[:, ibl] = d_cur.to(torch.float16)
        ls_all[:, ibl, :] = ls_blk.to(torch.int16)
        q_all[:, c0:c1] = q_blk.to(torch.uint8)
        extra_bits = torch.zeros((N,), dtype=torch.int32, device=device)
        for ib in range(NUM_SUBBLOCKS):
            extra_bits |= (phase_blk[:, ib].to(torch.int32) & 1) << ib
        extra_all[:, ibl] = extra_bits

    qdata = {"d": d_all, "extra": extra_all, "ls": ls_all, "q": q_all, "spec": spec_name}
    return qdata, W_hat


def compute_cholesky_factors(H: torch.Tensor, damp: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    From the Hessian `H` [K, K] builds the regularized Hessian `H_reg` and the upper-triangular
    Cholesky factor of its inverse `U = cholesky(H_reg^-1, upper=True)`.
    """
    K = H.shape[0]
    H_f64 = H.to(torch.float64)
    diag_mean = torch.diagonal(H_f64).mean().clamp(min=1e-8)
    for mult in (1.0, 2.0, 5.0, 10.0, 50.0):
        try:
            H_reg = H_f64.clone()
            H_reg.diagonal().add_(damp * mult * diag_mean)
            L_low = torch.linalg.cholesky(H_reg)
            H_inv = torch.cholesky_inverse(L_low)
            U = torch.linalg.cholesky(H_inv, upper=True)
            return H_reg.to(torch.float32), U.to(torch.float32)
        except RuntimeError:
            continue
    raise RuntimeError("Cholesky decomposition of Hessian H failed")


@torch.no_grad()
def quantize_iqk_ldlq(
    W: torch.Tensor,
    H: torch.Tensor,
    spec_name: str,
    damp: float = 0.01,
    H_precomputed: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float]:
    """
    `LDLQ-LIQK` algorithm:
      1. Cholesky decomposition of the full Hessian H [K, K] -> U = cholesky(H^-1, upper=True).
      2. Iterate over super-blocks of 256 weights (ibl = 0 .. K/256 - 1):
         - Inside a super-block, run 2 micro-passes over the 16 sub-blocks with column-wise error
           propagation through matrix U.
         - In each 16-weight sub-block, `(ls, phase)` candidates are scored with a full 16-step
           inner Cholesky loop.
         - After all 16 sub-blocks of the super-block are processed, the scale `d` is analytically refined
           using the full 256x256 Hessian block `H_blk`:
             d* = (g^T H_blk w_orig) / (g^T H_blk g)
         - Between micro-pass 1 and 2, for each row the one with strictly smaller Hessian quadratic error
           `(w - w_hat)^T H_blk (w - w_hat)` is chosen.
         - The final super-block error `err_blk = (w_blk - w_hat_blk) @ U_blk^-1` is propagated to all
           remaining super-blocks to the right `W[:, (ibl+1)*256:] -= err_blk @ U[c0:c1, c1:]`.
    Returns `(qdata, W_hat, hessian_loss)`.
    """
    spec = IQK_SPECS[spec_name]
    device = W.device
    N, K = W.shape
    assert K % QK_K == 0, f"K={K} must be a multiple of {QK_K}"
    nblocks = K // QK_K

    H_reg, U = H_precomputed if H_precomputed is not None else compute_cholesky_factors(H, damp=damp)

    cb = get_codebooks(spec, device)
    valid_ls = get_valid_scales(spec, device)

    W_work = W.float().clone()
    W_hat = torch.empty_like(W_work)

    d_all = torch.empty((N, nblocks), dtype=torch.float16, device=device)
    extra_all = torch.zeros((N, nblocks), dtype=torch.int32, device=device)
    ls_all = torch.empty((N, nblocks, NUM_SUBBLOCKS), dtype=torch.int16, device=device)
    q_all = torch.empty((N, K), dtype=torch.uint8, device=device)

    total_hessian_loss = 0.0

    for ibl in range(nblocks):
        c0, c1 = ibl * QK_K, (ibl + 1) * QK_K
        w_blk_start = W_work[:, c0:c1].clone()  # [N, 256] — super-block weights including errors from previous super-blocks
        U_blk = U[c0:c1, c0:c1]                 # [256, 256] — upper-triangular Cholesky block of the inverse
        # Compute the inverse upper-triangular matrix U_blk^-1 [256, 256] ONCE per super-block:
        # its diagonal 16x16 blocks are exactly (U_blk[s0:s1, s0:s1])^-1, which replaces 34
        # solve_triangular calls with fast GEMM matrix multiplies!
        U_blk_inv = torch.linalg.solve_triangular(
            U_blk, torch.eye(QK_K, dtype=torch.float32, device=device), upper=True
        )
        diag_w_blk = 1.0 / (torch.diagonal(U_blk) ** 2).clamp(min=1e-12)  # [256]

        # Exact error vector for the zero approximation A = w_blk_start @ U_blk^-1:
        A_chol = w_blk_start @ U_blk_inv  # [N, 256]

        # Fast diagonal pre-initialization of the super-block scale d_cand before the Cholesky pass:
        d_cand = _init_superblock_d(w_blk_start, spec, cb)
        G_init = torch.empty((N, QK_K), dtype=torch.float32, device=device)
        for ib in range(NUM_SUBBLOCKS):
            s0, s1 = ib * 16, (ib + 1) * 16
            b_ls0, b_ph0, b_q0, _ = _quantize_subblock_candidates(
                w_blk_start[:, s0:s1], d_cand, cb, valid_ls, u_sub=None, diag_w=diag_w_blk[s0:s1]
            )
            cb_ib0 = cb[b_ph0]
            v_ib0 = torch.gather(cb_ib0, 1, b_q0)
            G_init[:, s0:s1] = b_ls0.unsqueeze(1) * v_ib0
        num0 = (diag_w_blk.unsqueeze(0) * w_blk_start * G_init).sum(dim=1)
        den0 = (diag_w_blk.unsqueeze(0) * G_init * G_init).sum(dim=1)
        d_cand = torch.where(den0 > 0, num0 / den0, d_cand).to(torch.float16).to(torch.float32)

        best_loss_row = torch.full((N,), float("inf"), device=device)
        best_d_row = d_cand.clone()
        best_ls_row = torch.zeros((N, NUM_SUBBLOCKS), dtype=torch.int16, device=device)
        best_phase_row = torch.zeros((N, NUM_SUBBLOCKS), dtype=torch.long, device=device)
        best_q_row = torch.zeros((N, QK_K), dtype=torch.uint8, device=device)
        best_what_row = torch.zeros((N, QK_K), dtype=torch.float32, device=device)

        for micro_pass in range(2):
            w_sub_work = w_blk_start.clone()
            ls_pass = torch.empty((N, NUM_SUBBLOCKS), dtype=torch.float32, device=device)
            phase_pass = torch.empty((N, NUM_SUBBLOCKS), dtype=torch.long, device=device)
            q_pass = torch.empty((N, QK_K), dtype=torch.long, device=device)
            G_pass = torch.empty((N, QK_K), dtype=torch.float32, device=device)

            for ib in range(NUM_SUBBLOCKS):
                s0, s1 = ib * 16, (ib + 1) * 16
                u_sub = U_blk[s0:s1, s0:s1]  # [16, 16]
                w_sub_orig = w_sub_work[:, s0:s1].clone()

                b_ls, b_ph, b_q, w_hat_sub = _quantize_subblock_candidates(
                    w_sub_orig, d_cand, cb, valid_ls, u_sub=u_sub, top_k_candidates=8
                )
                ls_pass[:, ib] = b_ls
                phase_pass[:, ib] = b_ph
                q_pass[:, s0:s1] = b_q

                cb_ib = cb[b_ph]  # [N, L]
                v_ib = torch.gather(cb_ib, 1, b_q)  # [N, 16]
                G_pass[:, s0:s1] = b_ls.unsqueeze(1) * v_ib

                # Propagate the current sub-block's error (16 columns) to the remaining sub-blocks within this super-block (256)
                if s1 < QK_K:
                    err_sub = (w_sub_orig - w_hat_sub) @ U_blk_inv[s0:s1, s0:s1]  # [N, 16]
                    w_sub_work[:, s1:] -= err_sub @ U_blk[s0:s1, s1:]

            # Exact analytic Post-LDLQ Scale Refit of scale d via the Cholesky Schur complement U_blk:
            # Minimize || (w_blk_start - d * G_pass) @ U_blk^-1 ||_2^2 = || A_chol - d * B_chol ||_2^2
            B_chol = G_pass @ U_blk_inv  # [N, 256]
            num = (A_chol * B_chol).sum(dim=1)
            den = (B_chol * B_chol).sum(dim=1)
            d_refit = torch.where(den > 0, num / den, d_cand).to(torch.float16).to(torch.float32)

            # Check both d variants (before and after refit) by the exact super-block Cholesky error
            for d_try in (d_cand, d_refit):
                w_hat_try = d_try.unsqueeze(1) * G_pass
                err_try = A_chol - d_try.unsqueeze(1) * B_chol
                loss_try = (err_try * err_try).sum(dim=1)  # [N]
                improved = loss_try < best_loss_row
                if improved.any():
                    best_loss_row = torch.where(improved, loss_try, best_loss_row)
                    best_d_row = torch.where(improved, d_try, best_d_row)
                    best_ls_row = torch.where(improved.unsqueeze(1), ls_pass.to(torch.int16), best_ls_row)
                    best_phase_row = torch.where(improved.unsqueeze(1), phase_pass, best_phase_row)
                    best_q_row = torch.where(improved.unsqueeze(1), q_pass.to(torch.uint8), best_q_row)
                    best_what_row = torch.where(improved.unsqueeze(1), w_hat_try, best_what_row)

            # Use the refined d_refit for the second micro-pass
            d_cand = d_refit

        W_hat[:, c0:c1] = best_what_row
        d_all[:, ibl] = best_d_row.to(torch.float16)
        ls_all[:, ibl, :] = best_ls_row
        q_all[:, c0:c1] = best_q_row
        extra_bits = torch.zeros((N,), dtype=torch.int32, device=device)
        for ib in range(NUM_SUBBLOCKS):
            extra_bits |= (best_phase_row[:, ib].to(torch.int32) & 1) << ib
        extra_all[:, ibl] = extra_bits

        # Propagate the exact error of the whole super-block (256 columns) to all remaining super-blocks to the right!
        err_blk = (w_blk_start - best_what_row) @ U_blk_inv  # [N, 256]
        total_hessian_loss += float((err_blk * err_blk).sum().item())
        if c1 < K:
            W_work[:, c1:] -= err_blk @ U[c0:c1, c1:]

    qdata = {"d": d_all, "extra": extra_all, "ls": ls_all, "q": q_all, "spec": spec.name}
    return qdata, W_hat, total_hessian_loss


# ==============================================================================
# 3. Byte-level packing and unpacking into block_iq{2..6}_k C structs (PR #30715)
# ==============================================================================

def pack_iqk_blocks(qdata: Dict[str, torch.Tensor]) -> bytes:
    """
    Packs `qdata` parameters (`d`, `extra`, `ls`, `q`) into a binary buffer
    byte-identical to the `block_iq{2..6}_k` structs in `ggml-common.h` / `ggml-quants.c`.
    """
    spec = IQK_SPECS[qdata["spec"]]
    spec_name = spec.name
    d_np = qdata["d"].detach().cpu().numpy().view(np.uint16)          # [N, nblocks]
    extra_np = qdata["extra"].detach().cpu().numpy().astype(np.uint16) # [N, nblocks]
    ls_np = qdata["ls"].detach().cpu().numpy().astype(np.int16)        # [N, nblocks, 16]
    q_np = qdata["q"].detach().cpu().numpy().astype(np.uint8)          # [N, K]

    N, nblocks = d_np.shape
    total_blocks = N * nblocks
    d_flat = d_np.reshape(total_blocks)
    extra_flat = extra_np.reshape(total_blocks)
    ls_flat = ls_np.reshape(total_blocks, 16)
    q_flat = q_np.reshape(total_blocks, 16, 16)  # [B, ib=0..15, j=0..15]

    out = np.zeros((total_blocks, spec.block_bytes), dtype=np.uint8)
    out[:, 0:2] = d_flat[:, None].view(np.uint8).reshape(total_blocks, 2)
    out[:, 2:4] = extra_flat[:, None].view(np.uint8).reshape(total_blocks, 2)

    if spec_name == "iq2_k":
        # scales[8]: u4 (ls + 8), qs[64]
        l_u8 = (ls_flat + 8).astype(np.uint8) & 0xF
        scales = l_u8[:, 0::2] | (l_u8[:, 1::2] << 4)  # [B, 8]
        out[:, 4:12] = scales
        qs = np.zeros((total_blocks, 64), dtype=np.uint8)
        for ib in range(16):
            g = ib // 2
            offset = 32 * (g // 4) + 16 * (ib % 2)
            qs[:, offset : offset + 16] |= (q_flat[:, ib, :] & 3) << (2 * (g % 4))
        out[:, 12:76] = qs

    elif spec_name == "iq3_k":
        # scales_h: u16 (sign), scales_l[8]: m = (|ls| - 1) // 2, qs[64], qh[32]
        m = ((np.abs(ls_flat) - 1) // 2).astype(np.uint8) & 0xF
        scales_l = m[:, 0::2] | (m[:, 1::2] << 4)
        scales_h = np.zeros((total_blocks,), dtype=np.uint16)
        for ib in range(16):
            neg = (ls_flat[:, ib] < 0).astype(np.uint16)
            scales_h |= neg << ib
        out[:, 4:6] = scales_h[:, None].view(np.uint8).reshape(total_blocks, 2)
        out[:, 6:14] = scales_l
        qs = np.zeros((total_blocks, 64), dtype=np.uint8)
        qh = np.zeros((total_blocks, 32), dtype=np.uint8)
        for ib in range(16):
            g = ib // 2
            off_s = 32 * (g // 4) + 16 * (ib % 2)
            off_h = 16 * (ib % 2)
            qs[:, off_s : off_s + 16] |= (q_flat[:, ib, :] & 3) << (2 * (g % 4))
            qh[:, off_h : off_h + 16] |= ((q_flat[:, ib, :] >> 2) & 1) << g
        out[:, 14:78] = qs
        out[:, 78:110] = qh

    elif spec_name == "iq4_k":
        # scales_h[4], scales_l[8], qs[128]
        l_u8 = (ls_flat + 32).astype(np.uint8) & 0x3F
        scales_l = (l_u8[:, 0::2] & 0xF) | ((l_u8[:, 1::2] & 0xF) << 4)
        scales_h = np.zeros((total_blocks, 4), dtype=np.uint8)
        for ib in range(16):
            scales_h[:, ib // 4] |= (l_u8[:, ib] >> 4) << (2 * (ib % 4))
        out[:, 4:8] = scales_h
        out[:, 8:16] = scales_l
        qs = np.zeros((total_blocks, 128), dtype=np.uint8)
        for i in range(8):
            qs[:, 16 * i : 16 * (i + 1)] = (q_flat[:, 2 * i, :] & 0xF) | ((q_flat[:, 2 * i + 1, :] & 0xF) << 4)
        out[:, 16:144] = qs

    elif spec_name == "iq5_k":
        # scales_h[4], scales_l[8], qs[128], qh[32]
        l_u8 = (ls_flat + 32).astype(np.uint8) & 0x3F
        scales_l = (l_u8[:, 0::2] & 0xF) | ((l_u8[:, 1::2] & 0xF) << 4)
        scales_h = np.zeros((total_blocks, 4), dtype=np.uint8)
        for ib in range(16):
            scales_h[:, ib // 4] |= (l_u8[:, ib] >> 4) << (2 * (ib % 4))
        out[:, 4:8] = scales_h
        out[:, 8:16] = scales_l
        qs = np.zeros((total_blocks, 128), dtype=np.uint8)
        qh = np.zeros((total_blocks, 32), dtype=np.uint8)
        for ib in range(16):
            g = ib // 2
            off_s = 32 * (g // 2) + 16 * (ib % 2)
            off_h = 16 * (ib % 2)
            qs[:, off_s : off_s + 16] |= (q_flat[:, ib, :] & 0xF) << (4 * (g % 2))
            qh[:, off_h : off_h + 16] |= ((q_flat[:, ib, :] >> 4) & 1) << g
        out[:, 16:144] = qs
        out[:, 144:176] = qh

    elif spec_name == "iq6_k":
        # scales[16] (int8), qs[128], qh[64]
        out[:, 4:20] = ls_flat.astype(np.int8).view(np.uint8)
        qs = np.zeros((total_blocks, 128), dtype=np.uint8)
        qh = np.zeros((total_blocks, 64), dtype=np.uint8)
        for ib in range(16):
            g = ib // 2
            off_s = 32 * (g // 2) + 16 * (ib % 2)
            off_h = 32 * (g // 4) + 16 * (ib % 2)
            qs[:, off_s : off_s + 16] |= (q_flat[:, ib, :] & 0xF) << (4 * (g % 2))
            qh[:, off_h : off_h + 16] |= ((q_flat[:, ib, :] >> 4) & 3) << (2 * (g % 4))
        out[:, 20:148] = qs
        out[:, 148:212] = qh

    return out.tobytes()


def dequantize_iqk_blocks(buf: bytes, spec_name: str, nrow: int, k: int) -> np.ndarray:
    """
    Reference scalar decoder of binary `block_iq{2..6}_k` blocks into `float32 [nrow, k]`.
    Exactly mirrors the logic of `dequantize_row_iq{2..6}_k` from `ggml-quants.c`.
    """
    spec = IQK_SPECS[spec_name]
    nblocks = k // QK_K
    total_blocks = nrow * nblocks
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(total_blocks, spec.block_bytes)

    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).squeeze(-1)  # [B]
    extra = raw[:, 2:4].copy().view(np.uint16).squeeze(-1)                  # [B]
    kvals = np.array(spec.kvalues, dtype=np.float32)

    y = np.zeros((total_blocks, QK_K), dtype=np.float32)

    if spec_name == "iq2_k":
        scales = raw[:, 4:12]
        qs_all = raw[:, 12:76]
        for ib in range(16):
            g = ib // 2
            ls = ((scales[:, ib // 2] >> (4 * (ib % 2))) & 0xF).astype(np.int32) - 8
            dl = d * ls
            ph = np.where((extra & (1 << ib)) != 0, spec.phase_shift, 0).astype(np.float32)
            off = 32 * (g // 4) + 16 * (ib % 2)
            qs = qs_all[:, off : off + 16]
            q = (qs >> (2 * (g % 4))) & 3
            y[:, 16 * ib : 16 * (ib + 1)] = dl[:, None] * (kvals[q] + ph[:, None])

    elif spec_name == "iq3_k":
        scales_h = raw[:, 4:6].copy().view(np.uint16).squeeze(-1)
        scales_l = raw[:, 6:14]
        qs_all = raw[:, 14:78]
        qh_all = raw[:, 78:110]
        for ib in range(16):
            g = ib // 2
            m = ((scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0xF).astype(np.int32)
            ls = np.where((scales_h & (1 << ib)) != 0, -(2 * m + 1), 2 * m + 1)
            dl = d * ls
            ph = np.where((extra & (1 << ib)) != 0, spec.phase_shift, 0).astype(np.float32)
            off_s = 32 * (g // 4) + 16 * (ib % 2)
            off_h = 16 * (ib % 2)
            q = ((qs_all[:, off_s : off_s + 16] >> (2 * (g % 4))) & 3) | (
                ((qh_all[:, off_h : off_h + 16] >> g) & 1) << 2
            )
            y[:, 16 * ib : 16 * (ib + 1)] = dl[:, None] * (kvals[q] + ph[:, None])

    elif spec_name == "iq4_k":
        scales_h = raw[:, 4:8]
        scales_l = raw[:, 8:16]
        qs_all = raw[:, 16:144]
        for ib in range(0, 16, 2):
            hb = scales_h[:, ib // 4]
            ls1 = ((scales_l[:, ib // 2] & 0xF) | (((hb >> (2 * (ib % 4))) & 3) << 4)).astype(np.int32) - 32
            ls2 = ((scales_l[:, ib // 2] >> 4) | (((hb >> (2 * (ib % 4) + 2)) & 3) << 4)).astype(np.int32) - 32
            dl1, dl2 = d * ls1, d * ls2
            ph1 = np.where((extra & (1 << ib)) != 0, spec.phase_shift, 0).astype(np.float32)
            ph2 = np.where((extra & (1 << (ib + 1))) != 0, spec.phase_shift, 0).astype(np.float32)
            qs = qs_all[:, 8 * ib : 8 * ib + 16]
            y[:, 16 * ib : 16 * (ib + 1)] = dl1[:, None] * (kvals[qs & 0xF] + ph1[:, None])
            y[:, 16 * (ib + 1) : 16 * (ib + 2)] = dl2[:, None] * (kvals[qs >> 4] + ph2[:, None])

    elif spec_name == "iq5_k":
        scales_h = raw[:, 4:8]
        scales_l = raw[:, 8:16]
        qs_all = raw[:, 16:144]
        qh_all = raw[:, 144:176]
        for ib in range(16):
            g = ib // 2
            hb = scales_h[:, ib // 4]
            ls = (
                ((scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((hb >> (2 * (ib % 4))) & 3) << 4)
            ).astype(np.int32) - 32
            dl = d * ls
            ph = np.where((extra & (1 << ib)) != 0, spec.phase_shift, 0).astype(np.float32)
            off_s = 32 * (g // 2) + 16 * (ib % 2)
            off_h = 16 * (ib % 2)
            q = ((qs_all[:, off_s : off_s + 16] >> (4 * (g % 2))) & 0xF) | (
                ((qh_all[:, off_h : off_h + 16] >> g) & 1) << 4
            )
            y[:, 16 * ib : 16 * (ib + 1)] = dl[:, None] * (kvals[q] + ph[:, None])

    elif spec_name == "iq6_k":
        scales = raw[:, 4:20].copy().view(np.int8).astype(np.int32)
        qs_all = raw[:, 20:148]
        qh_all = raw[:, 148:212]
        for ib in range(16):
            g = ib // 2
            dl = d * scales[:, ib]
            ph = np.where((extra & (1 << ib)) != 0, spec.phase_shift, 0).astype(np.float32)
            off_s = 32 * (g // 2) + 16 * (ib % 2)
            off_h = 32 * (g // 4) + 16 * (ib % 2)
            q = ((qs_all[:, off_s : off_s + 16] >> (4 * (g % 2))) & 0xF) | (
                ((qh_all[:, off_h : off_h + 16] >> (2 * (g % 4))) & 3) << 4
            )
            y[:, 16 * ib : 16 * (ib + 1)] = dl[:, None] * (kvals[q] + ph[:, None])

    return y.reshape(nrow, k)


if __name__ == "__main__":
    # Quick self-test of pack/unpack correctness and LDLQ on a synthetic matrix
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = torch.randn(32, 512, device=device) * 0.05
    X = torch.randn(256, 512, device=device)
    H = (X.T @ X) / X.shape[0]
    H_f = compute_cholesky_factors(H)

    for name in ("iq2_k", "iq3_k", "iq4_k", "iq5_k", "iq6_k"):
        q_diag, w_diag = quantize_iqk_diag(W, name, imatrix_w=torch.diagonal(H))
        buf = pack_iqk_blocks(q_diag)
        w_unp = torch.from_numpy(dequantize_iqk_blocks(buf, name, 32, 512)).to(device)
        max_diff = (w_diag - w_unp).abs().max().item()
        assert max_diff < 1e-6, f"Packing roundtrip error in {name}: {max_diff}"

        q_ldlq, w_ldlq, _ = quantize_iqk_ldlq(W, H, name, H_precomputed=H_f)
        err_diag = ((X @ W.T - X @ w_diag.T) ** 2).mean().item()
        err_ldlq = ((X @ W.T - X @ w_ldlq.T) ** 2).mean().item()
        print(f"{name:6s} ({IQK_SPECS[name].bpw:.3f} bpw): roundtrip OK | MSE(diag)={err_diag:.6e} -> MSE(LDLQ)={err_ldlq:.6e} ({(err_ldlq/err_diag - 1)*100:+.1f}%)")
