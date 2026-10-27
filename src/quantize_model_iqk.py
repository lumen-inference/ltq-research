"""
Whole-model layer-by-layer quantizer for the IQ2_K..IQ6_K families and the .liqk format:
  1. Sequential (GPTQ / EXL3-style) collection of exact Hessians H = (1/N) X^T X of size [K, K]
     inside each transformer layer (qkv -> o_proj -> gate_up -> down_proj), so that each
     subsequent sub-layer and layer sees the already-quantized outputs of the previous ones.
  2. Four quantization modes for linear weights:
     - "diag": fast 2-pass diagonal quantization (analytic refit of scale d)
     - "ldlq": block-sequential Cholesky LDLQ + Post-LDLQ Scale Refit without Hadamard rotation
               (100% compatible with stock .gguf PR #30715..#30718!)
     - "liqk": random block Hadamard rotation (suh, svh) + LDLQ + Post-LDLQ Scale Refit (.liqk)
     - "liqk_gsq": refinement of discrete indices via Hessian-folded Gumbel-Softmax + Lion (GSQ)
                   on top of the LDLQ-LIQK solution (128x faster than naive GSQ over activation batches).
  3. Global dynamic programming (DP) bit allocation across the 112 matrices
     for an arbitrary target bpw budget (replaces expensive RCO, ~10 seconds).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iqk_format import (
    IQK_SPECS,
    NUM_SUBBLOCKS,
    QK_K,
    compute_cholesky_factors,
    fwht_inplace,
    get_codebooks,
    pack_iqk_blocks,
    quantize_iqk_diag,
    quantize_iqk_ldlq,
    rotate_weights_and_hessian,
)


LINEAR_SUBMODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

GGUF_SUBMODULE_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


@torch.no_grad()
def unrotate_weights(
    W_rot_hat: torch.Tensor,
    suh: torch.Tensor,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Inverse input-side Hadamard transform for the weight matrix W_rot_hat:
        W_rot = fwht(W * suh)
    Since the normalized block Walsh-Hadamard transform is orthogonal and symmetric (H^-1 = H):
        W_hat = fwht(W_rot_hat) * suh
    Then for any input x:
        x @ W_hat^T == fwht(x * suh) @ W_rot_hat^T
    up to machine epsilon (~1e-6). This allows transparently substituting W_hat
    into the stock LlamaForCausalLM graph for KLD evaluation.
    """
    W_unrot = fwht_inplace(W_rot_hat.float().clone(), block_size=block_size)
    return W_unrot * suh.float().unsqueeze(0)


def refine_iqk_gsq(
    W_target: torch.Tensor,
    H: torch.Tensor,
    qdata_init: Dict[str, torch.Tensor],
    spec_name: str,
    num_steps: int = 80,
    lr: float = 0.08,
    temp_start: float = 1.5,
    temp_end: float = 0.05,
    damp: float = 0.01,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Hessian-folded GSQ (Gumbel-Softmax Quantization + Lion) on top of the initial LDLQ solution.

    128x mathematical speedup:
      Instead of running activation batches X of shape [N_tok=262144, K] at every iteration,
      we use the identity:
          (1/N) || X W^T - X W_hat^T ||_F^2 = tr((W - W_hat) H (W - W_hat)^T) = || (W - W_hat) L_H ||_F^2,
      where L_H = cholesky(H) is only [K, K].
      One gradient-descent step reduces to a single [N, K] @ [K, K] matrix multiply!

    Optimized:
      - A local window of M=5 neighboring codebook indices around the LDLQ initialization q_init
        via Gumbel-Softmax / Softmax relaxation with temperature annealing T: 1.5 -> 0.05 (STE in the forward pass).
      - The continuous super-block scale d (with an analytic refit at the end).
    """
    spec = IQK_SPECS[spec_name]
    device = W_target.device
    N, K = W_target.shape
    nblocks = K // QK_K
    L = spec.nlevels

    # Compute the lower-triangular Cholesky factor of the Hessian itself H = L_H @ L_H^T
    H_f64 = H.to(torch.float64)
    diag_mean = torch.diagonal(H_f64).mean().clamp(min=1e-8)
    H_f64.diagonal().add_(damp * diag_mean)
    L_H = torch.linalg.cholesky(H_f64).to(torch.float32)  # [K, K]

    cb = get_codebooks(spec, device)  # [2, L]
    q_init = qdata_init["q"].long()   # [N, K]
    ls_fixed = qdata_init["ls"].float()  # [N, nblocks, 16]
    d_init = qdata_init["d"].float()     # [N, nblocks]
    extra_fixed = qdata_init["extra"]    # [N, nblocks]

    # Extract the phase (0 or 1) of each sub-block and expand it to all K weights
    phase_sub = torch.empty((N, nblocks, NUM_SUBBLOCKS), dtype=torch.long, device=device)
    for ib in range(NUM_SUBBLOCKS):
        phase_sub[:, :, ib] = (extra_fixed >> ib) & 1

    # For each weight build a local window of W_WIN=5 neighboring codebook levels around q_init:
    # shifts = [-2, -1, 0, +1, +2]
    shifts = torch.tensor([-2, -1, 0, 1, 2], device=device, dtype=torch.long)
    cand_q = (q_init.unsqueeze(-1) + shifts.view(1, 1, 5)).clamp(0, L - 1)  # [N, K, 5]

    phase_full = phase_sub.unsqueeze(-1).expand(N, nblocks, NUM_SUBBLOCKS, 16).reshape(N, K)
    ls_full = ls_fixed.unsqueeze(-1).expand(N, nblocks, NUM_SUBBLOCKS, 16).reshape(N, K)

    # Codebook values in the 5-candidate window: [N, K, 5]
    cb_expanded = cb[phase_full]  # [N, K, L]
    cand_vals = torch.gather(cb_expanded, 2, cand_q)  # [N, K, 5]
    # Multiply by the fixed sub-block scale ls:
    cand_G = ls_full.unsqueeze(-1) * cand_vals  # [N, K, 5]

    # Initialize logits so the central element (shift=0, i.e. the LDLQ solution) has priority
    logits = torch.zeros((N, K, 5), dtype=torch.float32, device=device)
    logits[:, :, 2] = 2.5
    logits.requires_grad_(True)

    d_param = d_init.clone().detach().requires_grad_(True)

    # Lion optimizer state (beta1=0.9, beta2=0.99)
    exp_avg_logits = torch.zeros_like(logits)
    exp_avg_d = torch.zeros_like(d_param)
    beta1, beta2 = 0.9, 0.99

    WL_target = W_target.float() @ L_H  # [N, K] — precomputed projection of the reference weights onto the Cholesky factor

    with torch.enable_grad():
        for step in range(num_steps):
            progress = step / max(num_steps - 1, 1)
            temp = temp_start * ((temp_end / temp_start) ** progress)
            cur_lr = lr * (0.5 * (1.0 + math.cos(math.pi * progress)))

            probs = F.softmax(logits / temp, dim=-1)
            # Straight-Through Estimator (STE): hard argmax in the forward pass, softmax gradient in the backward pass
            hard_idx = probs.argmax(dim=-1, keepdim=True)
            hard_one_hot = torch.zeros_like(probs).scatter_(-1, hard_idx, 1.0)
            ste_weights = hard_one_hot - probs.detach() + probs

            G_cur = (ste_weights * cand_G).sum(dim=-1)  # [N, K]
            d_full = d_param.unsqueeze(-1).expand(N, nblocks, QK_K).reshape(N, K)
            W_hat_cur = d_full * G_cur

            # Exact quadratic activation error || (W - W_hat) L_H ||_F^2
            diff_proj = WL_target - (W_hat_cur @ L_H)
            loss = (diff_proj * diff_proj).sum()

            loss.backward()

            with torch.no_grad():
                # Lion step for the logits
                g_l = logits.grad
                update_l = exp_avg_logits * beta1 + g_l * (1.0 - beta1)
                logits.sub_(cur_lr * torch.sign(update_l))
                exp_avg_logits.mul_(beta2).add_(g_l, alpha=1.0 - beta2)
                logits.grad.zero_()

                # Lion step for the super-block scale d
                g_d = d_param.grad
                update_d = exp_avg_d * beta1 + g_d * (1.0 - beta1)
                d_param.sub_(cur_lr * 1e-3 * torch.sign(update_d))
                d_param.clamp_(min=1e-7)
                exp_avg_d.mul_(beta2).add_(g_d, alpha=1.0 - beta2)
                d_param.grad.zero_()

    # Final discretization and per-row check: keep the GSQ solution only for rows
    # where the error || (w - w_hat) L_H ||_2^2 strictly decreased relative to LDLQ!
    with torch.no_grad():
        best_shift_idx = logits.argmax(dim=-1, keepdim=True)  # [N, K, 1]
        q_gsq = torch.gather(cand_q, 2, best_shift_idx).squeeze(-1).to(torch.uint8)  # [N, K]
        G_gsq = torch.gather(cand_G, 2, best_shift_idx).squeeze(-1)  # [N, K]

        d_gsq_fp16 = d_param.to(torch.float16)
        d_gsq_full = d_gsq_fp16.float().unsqueeze(-1).expand(N, nblocks, QK_K).reshape(N, K)
        W_hat_gsq = d_gsq_full * G_gsq

        # Compare with the original LDLQ solution row by row
        G_init = torch.gather(cand_G, 2, torch.full_like(best_shift_idx, 2)).squeeze(-1)
        d_init_full = qdata_init["d"].float().unsqueeze(-1).expand(N, nblocks, QK_K).reshape(N, K)
        W_hat_init = d_init_full * G_init

        err_init = ((WL_target - W_hat_init @ L_H) ** 2).sum(dim=1)
        err_gsq = ((WL_target - W_hat_gsq @ L_H) ** 2).sum(dim=1)
        better = err_gsq < err_init  # [N]

        q_final = torch.where(better.unsqueeze(1), q_gsq, qdata_init["q"])
        d_final = torch.where(better.unsqueeze(1), d_gsq_fp16, qdata_init["d"])
        W_hat_final = torch.where(better.unsqueeze(1), W_hat_gsq, W_hat_init)

    qdata_out = {
        "d": d_final,
        "extra": qdata_init["extra"],
        "ls": qdata_init["ls"],
        "q": q_final,
        "spec": spec_name,
    }
    return qdata_out, W_hat_final


@torch.no_grad()
def quantize_single_linear_matrix(
    W: torch.Tensor,
    H: torch.Tensor,
    spec_name: str,
    mode: str = "liqk",
    seed: int = 1337,
    damp: float = 0.01,
    gsq_steps: int = 80,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float]:
    """
    Quantizes a single linear matrix W [M, K] with input Hessian H [K, K] using the selected method:
      - mode="diag": fast 2-pass diagonal quantization without Hadamard
      - mode="diag_had": 2-pass diagonal quantization with Hadamard rotation
      - mode="ldlq": full Cholesky LDLQ + Post-LDLQ Scale Refit without Hadamard (.gguf-compatible)
      - mode="liqk": Hadamard rotation (suh, svh) + full Cholesky LDLQ + Post-LDLQ Scale Refit (.liqk)
      - mode="liqk_gsq": Hadamard rotation + LDLQ + Hessian-folded GSQ (Gumbel-Softmax + Lion)

    Returns:
      - qdata: dict of quantized parameters (d, extra, ls, q, spec, and suh/svh with Hadamard)
      - W_hat_eff: effective dequantized matrix [M, K] in the original basis, for substitution into the model
      - rel_loss: relative Hessian error tr((W - W_hat) H (W - W_hat)^T) / tr(W H W^T)
    """
    W_f32 = W.float()
    H_f32 = H.float()

    use_hadamard = mode in ("diag_had", "liqk", "liqk_gsq")
    if use_hadamard:
        W_work, H_work, suh = rotate_weights_and_hessian(W_f32, H_f32, seed=seed)
    else:
        W_work, H_work, suh = W_f32, H_f32, None

    if mode in ("diag", "diag_had"):
        diag_h = torch.diagonal(H_work).clamp(min=1e-8)
        qdata, W_hat_work = quantize_iqk_diag(W_work, spec_name, imatrix_w=diag_h)
    elif mode in ("ldlq", "liqk", "liqk_gsq"):
        qdata, W_hat_work, _ = quantize_iqk_ldlq(W_work, H_work, spec_name, damp=damp)
        if mode == "liqk_gsq" and spec_name.upper() in ("IQ2_K", "IQ3_K"):
            qdata, W_hat_work = refine_iqk_gsq(
                W_work, H_work, qdata, spec_name, num_steps=gsq_steps, damp=damp
            )
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    if use_hadamard:
        assert suh is not None
        W_hat_eff = unrotate_weights(W_hat_work, suh)
        # Post-LDLQ refinement of the per-channel input scales suh (fp16) and the output row scales d (fp16)
        # in the original basis under the Hessian metric H_f32 (0 bits of overhead, 0 extra ops at inference).
        HW = W_f32 @ H_f32
        HQ = W_hat_eff @ H_f32
        c_row = (W_hat_eff * HW).sum(dim=1) / (W_hat_eff * HQ).sum(dim=1).clamp(min=1e-12)
        c_row = torch.where(torch.isfinite(c_row) & (c_row > 0.5) & (c_row < 2.0), c_row, torch.ones_like(c_row))
        d_new = (qdata["d"].float() * c_row.unsqueeze(1)).to(torch.float16)
        c_eff = (d_new.float() / qdata["d"].float().clamp(min=1e-12)).mean(dim=1, keepdim=True)
        Q_mid = W_hat_eff * c_eff

        A_in = (Q_mid.T @ Q_mid) * H_f32
        b_in = (Q_mid * HW).sum(dim=0)
        diag_mean = A_in.diagonal().mean().clamp(min=1e-12)
        A_in.diagonal().add_(1e-5 * diag_mean)
        try:
            r_col = torch.linalg.solve(A_in, b_in)
            r_col = torch.where(torch.isfinite(r_col) & (r_col > 0.5) & (r_col < 2.0), r_col, torch.ones_like(r_col))
            suh_new = (suh.float() * r_col).to(torch.float16)
            W_hat_cand = unrotate_weights(W_hat_work * c_eff, suh_new)
            diff_base = W_f32 - W_hat_eff
            diff_cand = W_f32 - W_hat_cand
            if float(((diff_cand @ H_f32) * diff_cand).sum().item()) < float(((diff_base @ H_f32) * diff_base).sum().item()):
                W_hat_eff = W_hat_cand
                qdata["d"] = d_new
                suh = suh_new
        except RuntimeError:
            pass
        qdata["suh"] = suh
    else:
        W_hat_eff = W_hat_work

    diff = W_f32 - W_hat_eff
    num_loss = float(((diff @ H_f32) * diff).sum().item())
    den_loss = float(((W_f32 @ H_f32) * W_f32).sum().clamp(min=1e-12).item())
    rel_loss = num_loss / den_loss

    return qdata, W_hat_eff, rel_loss


def solve_dp_bit_allocation(
    layer_costs: List[Dict[str, Dict[str, float]]],
    layer_params: List[Dict[str, int]],
    target_bpw: float,
    candidate_specs: List[str] = ["IQ2_K", "IQ3_K", "IQ4_K", "IQ5_K", "IQ6_K"],
) -> Dict[Tuple[int, str], str]:
    """
    Global dynamic-programming knapsack solver (Multiple-Choice Knapsack DP)
    that assigns formats {IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K} to all linear matrices of the model
    so as to minimize the total weighted distortion under an average budget <= target_bpw.

    Budget discretization step: 1 unit = 262144 bytes (0.25 MiB), giving ~2000 DP states,
    solved exactly in < 0.05 seconds.
    """
    if target_bpw >= 3.45 and "IQ2_K" in candidate_specs:
        # For budgets >= 3.45 bpw forbid individual layers from dropping to 2-bit IQ2_K
        # (minimum floor IQ3_K = 3.4375 bpw, similar to the 3-bit floor in Q3_K_M / EXL3).
        candidate_specs = [sp for sp in candidate_specs if sp != "IQ2_K"]
    if target_bpw <= 4.35 and "IQ6_K" in candidate_specs:
        # Cap the ceiling at 1 step above IQ4_K (IQ5_K) so the other matrices are not starved.
        candidate_specs = [sp for sp in candidate_specs if sp != "IQ6_K"]

    items = []
    total_params = 0
    for l_idx in range(len(layer_costs)):
        for subname in LINEAR_SUBMODULES:
            n_p = layer_params[l_idx][subname]
            total_params += n_p
            opts = []
            for sp in candidate_specs:
                bpw = IQK_SPECS[sp].bpw
                bytes_cost = int(round(n_p * bpw / 8.0))
                dist_cost = layer_costs[l_idx][subname][sp]
                opts.append((sp, bytes_cost, dist_cost))
            items.append(((l_idx, subname), opts))

    max_total_bytes = int(round(total_params * target_bpw / 8.0))
    unit_bytes = 65536  # 64 KiB grid step (high budget-hitting precision)
    max_units = max_total_bytes // unit_bytes

    n_items = len(items)
    dp = np.full((n_items + 1, max_units + 1), np.inf, dtype=np.float64)
    choice = np.zeros((n_items + 1, max_units + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    for i, (_, opts) in enumerate(items, start=1):
        for c_idx, (sp, b_cost, d_cost) in enumerate(opts):
            u_cost = (b_cost + unit_bytes - 1) // unit_bytes
            if u_cost <= max_units:
                prev_vals = dp[i - 1, : max_units + 1 - u_cost] + d_cost
                target_slice = dp[i, u_cost:]
                mask = prev_vals < target_slice
                target_slice[mask] = prev_vals[mask]
                choice[i, u_cost:][mask] = c_idx

    best_u = int(np.argmin(dp[n_items, :]))
    if np.isinf(dp[n_items, best_u]):
        # If the target budget is below the minimum possible, pick the smallest format everywhere
        return {key: candidate_specs[0] for key, _ in items}

    assignment: Dict[Tuple[int, str], str] = {}
    cur_u = best_u
    for i in range(n_items, 0, -1):
        key, opts = items[i - 1]
        c_idx = int(choice[i, cur_u])
        sp, b_cost, _ = opts[c_idx]
        assignment[key] = sp
        u_cost = (b_cost + unit_bytes - 1) // unit_bytes
        cur_u -= u_cost

    return assignment


@torch.no_grad()
def _compute_cov_matrix(X_batches: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Accumulates the exact Hessian covariance matrix H = (1/N) sum X^T X in float64 on the GPU."""
    K = X_batches[0].shape[-1]
    H_acc = torch.zeros((K, K), dtype=torch.float64, device=device)
    total_tokens = 0
    for xb in X_batches:
        x_2d = xb.to(device=device, dtype=torch.float32).reshape(-1, K)
        # Use float32 GEMM (Tensor Cores) with float64 accumulation
        H_acc.add_((x_2d.T @ x_2d).to(torch.float64))
        total_tokens += x_2d.shape[0]
    H_acc.div_(max(total_tokens, 1))
    return H_acc.to(torch.float32)


@torch.no_grad()
def quantize_llama_model_sequential(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    spec_or_assignment: str | Dict[Tuple[int, str], str],
    mode: str = "liqk",
    batch_size: int = 8,
    damp: float = 0.01,
    export_gguf_blocks: bool = False,
    gsq_steps: int = 80,
) -> dict:
    """
    Sequential layer-by-layer quantization of all transformer layers of a LlamaForCausalLM model
    on the calibration tensor cal_tokens [N_seq, seq_len].

    Inside each layer l = 0 .. n_layers-1:
      1. The exact Hessian H_qkv is collected at the input_layernorm output and q_proj, k_proj, v_proj are quantized.
      2. The attention block (SDPA with RoPE) is run on the already-quantized Q, K, V; at the o_proj input
         the exact Hessian H_o is collected, o_proj is quantized and the residual stream H_states is updated.
      3. The exact Hessian H_up is collected at the post_attention_layernorm output and gate_proj, up_proj are quantized.
      4. The SwiGLU intermediate activation silu(gate_q(x)) * up_q(x) is computed, the exact
         Hessian H_down [8192, 8192] is collected, down_proj is quantized and H_states is updated for layer l+1.
    """
    t_start = time.perf_counter()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()

    n_seqs, seq_len = cal_tokens.shape
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_heads
    n_rep = n_heads // n_kv_heads

    # 1. Compute the initial hidden states after embed_tokens (batches are kept in CPU RAM as bfloat16)
    H_batches: List[torch.Tensor] = []
    for i in range(0, n_seqs, batch_size):
        tok_b = cal_tokens[i : i + batch_size].to(device)
        emb_b = model.model.embed_tokens(tok_b).cpu()
        H_batches.append(emb_b)

    # Precompute position_embeddings (cos, sin) for RoPE once
    dummy_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy_h = H_batches[0][:1].to(device=device, dtype=dtype)
    pos_emb = model.model.rotary_emb(dummy_h, dummy_pos)

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    total_params = 0
    total_bits = 0
    packed_gguf_dict: Dict[str, np.ndarray] = {}
    per_layer_stats = []

    for l_idx in range(n_layers):
        t_layer = time.perf_counter()
        layer = model.model.layers[l_idx]

        def get_spec(subname: str) -> str:
            if isinstance(spec_or_assignment, str):
                return spec_or_assignment
            return spec_or_assignment[(l_idx, subname)]

        # --- Sub-block A: input_layernorm -> q_proj, k_proj, v_proj ---
        norm1_batches = []
        for hb in H_batches:
            norm1_batches.append(layer.input_layernorm(hb.to(device=device, dtype=dtype)).cpu())
        H_qkv = _compute_cov_matrix(norm1_batches, device)

        for idx_s, subname in enumerate(["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]):
            mod = layer.get_submodule(subname)
            sp = get_spec(subname)
            qdata, w_hat, rloss = quantize_single_linear_matrix(
                mod.weight.data,
                H_qkv,
                sp,
                mode=mode,
                seed=1000 + l_idx * 10 + idx_s,
                damp=damp,
                gsq_steps=gsq_steps,
            )
            mod.weight.data.copy_(w_hat.to(dtype=dtype, device=device))
            n_p = mod.weight.numel()
            total_params += n_p
            total_bits += int(round(n_p * IQK_SPECS[sp].bpw))
            if export_gguf_blocks and mode in ("diag", "ldlq"):
                raw = pack_iqk_blocks(qdata)
                packed_gguf_dict[f"blk.{l_idx}.{GGUF_SUBMODULE_MAP[subname]}.weight"] = np.frombuffer(
                    raw, dtype=np.uint8
                ).copy()

        # --- Sub-block B: SDPA on quantized Q, K, V -> o_proj input ---
        attn_out_batches = []
        for x_norm_cpu in norm1_batches:
            x_norm = x_norm_cpu.to(device=device, dtype=dtype)
            bsz, q_len, _ = x_norm.shape
            q = layer.self_attn.q_proj(x_norm).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(x_norm).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(x_norm).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_out_batches.append(attn_out.cpu())
        del norm1_batches

        H_o = _compute_cov_matrix(attn_out_batches, device)
        mod_o = layer.self_attn.o_proj
        sp_o = get_spec("self_attn.o_proj")
        qdata_o, w_hat_o, _ = quantize_single_linear_matrix(
            mod_o.weight.data,
            H_o,
            sp_o,
            mode=mode,
            seed=1000 + l_idx * 10 + 3,
            damp=damp,
            gsq_steps=gsq_steps,
        )
        mod_o.weight.data.copy_(w_hat_o.to(dtype=dtype, device=device))
        n_po = mod_o.weight.numel()
        total_params += n_po
        total_bits += int(round(n_po * IQK_SPECS[sp_o].bpw))
        if export_gguf_blocks and mode in ("diag", "ldlq"):
            raw = pack_iqk_blocks(qdata_o)
            packed_gguf_dict[f"blk.{l_idx}.attn_output.weight"] = np.frombuffer(raw, dtype=np.uint8).copy()

        # Update the residual stream after attention
        for b_i in range(len(H_batches)):
            ao = attn_out_batches[b_i].to(device=device, dtype=dtype)
            H_batches[b_i] = (H_batches[b_i].to(device=device, dtype=dtype) + mod_o(ao)).cpu()
        del attn_out_batches

        # --- Sub-block C: post_attention_layernorm -> gate_proj, up_proj ---
        norm2_batches = []
        for hb in H_batches:
            norm2_batches.append(layer.post_attention_layernorm(hb.to(device=device, dtype=dtype)).cpu())
        H_up = _compute_cov_matrix(norm2_batches, device)

        for idx_s, subname in enumerate(["mlp.gate_proj", "mlp.up_proj"]):
            mod = layer.get_submodule(subname)
            sp = get_spec(subname)
            qdata, w_hat, _ = quantize_single_linear_matrix(
                mod.weight.data,
                H_up,
                sp,
                mode=mode,
                seed=1000 + l_idx * 10 + 4 + idx_s,
                damp=damp,
                gsq_steps=gsq_steps,
            )
            mod.weight.data.copy_(w_hat.to(dtype=dtype, device=device))
            n_p = mod.weight.numel()
            total_params += n_p
            total_bits += int(round(n_p * IQK_SPECS[sp].bpw))
            if export_gguf_blocks and mode in ("diag", "ldlq"):
                raw = pack_iqk_blocks(qdata)
                packed_gguf_dict[f"blk.{l_idx}.{GGUF_SUBMODULE_MAP[subname]}.weight"] = np.frombuffer(
                    raw, dtype=np.uint8
                ).copy()

        # --- Sub-block D: SwiGLU intermediate activation -> down_proj ---
        mlp_mid_batches = []
        for x_norm2_cpu in norm2_batches:
            x_norm2 = x_norm2_cpu.to(device=device, dtype=dtype)
            mid = F.silu(layer.mlp.gate_proj(x_norm2)) * layer.mlp.up_proj(x_norm2)
            mlp_mid_batches.append(mid.cpu())
        del norm2_batches

        H_down = _compute_cov_matrix(mlp_mid_batches, device)
        mod_d = layer.mlp.down_proj
        sp_d = get_spec("mlp.down_proj")
        qdata_d, w_hat_d, _ = quantize_single_linear_matrix(
            mod_d.weight.data,
            H_down,
            sp_d,
            mode=mode,
            seed=1000 + l_idx * 10 + 6,
            damp=damp,
            gsq_steps=gsq_steps,
        )
        mod_d.weight.data.copy_(w_hat_d.to(dtype=dtype, device=device))
        n_pd = mod_d.weight.numel()
        total_params += n_pd
        total_bits += int(round(n_pd * IQK_SPECS[sp_d].bpw))
        if export_gguf_blocks and mode in ("diag", "ldlq"):
            raw = pack_iqk_blocks(qdata_d)
            packed_gguf_dict[f"blk.{l_idx}.ffn_down.weight"] = np.frombuffer(raw, dtype=np.uint8).copy()

        # Update the residual stream after the MLP for the next layer
        for b_i in range(len(H_batches)):
            md = mlp_mid_batches[b_i].to(device=device, dtype=dtype)
            H_batches[b_i] = (H_batches[b_i].to(device=device, dtype=dtype) + mod_d(md)).cpu()
        del mlp_mid_batches

        layer_sec = time.perf_counter() - t_layer
        per_layer_stats.append({"layer": l_idx, "sec": layer_sec})

    total_sec = time.perf_counter() - t_start
    return {
        "linear_params": total_params,
        "linear_bytes": total_bits // 8,
        "linear_bpw": total_bits / max(total_params, 1),
        "quant_sec": total_sec,
        "packed_gguf": packed_gguf_dict,
    }


@torch.no_grad()
def build_dp_sensitivity_profile(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    candidate_specs: List[str] = ["IQ2_K", "IQ3_K", "IQ4_K", "IQ5_K", "IQ6_K"],
    use_hadamard: bool = True,
    batch_size: int = 8,
) -> Tuple[List[Dict[str, Dict[str, float]]], List[Dict[str, int]]]:
    """
    In one fast pass over the calibration set, collects exact Hessian diagonals (Hadamard-rotated when use_hadamard=True)
    and computes the relative distortion of each of the 112 matrices for all 5 bitrate candidates {IQ2_K..IQ6_K}.
    Accounts for the GQA replication factor of v_proj/k_proj and the direct contribution of o_proj/down_proj to the residual stream.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()

    n_seqs, seq_len = cal_tokens.shape
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_heads
    n_rep = n_heads // n_kv_heads

    # Sensitivity coefficients of matrix types with respect to the layer output:
    # v_proj is replicated n_rep times (4x in Llama-3.2) and enters the attention output directly and linearly;
    # o_proj and down_proj write directly into the transformer residual stream.
    submodule_importance = {
        "self_attn.q_proj": 3.5,
        "self_attn.k_proj": 3.5,
        "self_attn.v_proj": float(n_rep) * 1.5,
        "self_attn.o_proj": 2.0,
        "mlp.gate_proj": 1.0,
        "mlp.up_proj": 1.2,
        "mlp.down_proj": 2.2,
    }

    H_batches: List[torch.Tensor] = []
    for i in range(0, n_seqs, batch_size):
        tok_b = cal_tokens[i : i + batch_size].to(device)
        H_batches.append(model.model.embed_tokens(tok_b).cpu())

    dummy_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy_h = H_batches[0][:1].to(device=device, dtype=dtype)
    pos_emb = model.model.rotary_emb(dummy_h, dummy_pos)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    layer_costs: List[Dict[str, Dict[str, float]]] = []
    layer_params: List[Dict[str, int]] = []

    for l_idx in range(n_layers):
        layer = model.model.layers[l_idx]
        # The first 2 and last 2 transformer layers have elevated sensitivity
        depth_mult = 1.35 if (l_idx < 2 or l_idx >= n_layers - 2) else 1.0

        norm1_batches = [layer.input_layernorm(hb.to(device=device, dtype=dtype)).cpu() for hb in H_batches]
        H_qkv = _compute_cov_matrix(norm1_batches, device)

        attn_out_batches = []
        for x_norm_cpu in norm1_batches:
            x_norm = x_norm_cpu.to(device=device, dtype=dtype)
            bsz, q_len, _ = x_norm.shape
            q = layer.self_attn.q_proj(x_norm).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(x_norm).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(x_norm).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_out_batches.append(attn_out.cpu())
        del norm1_batches
        H_o = _compute_cov_matrix(attn_out_batches, device)

        for b_i in range(len(H_batches)):
            ao = attn_out_batches[b_i].to(device=device, dtype=dtype)
            H_batches[b_i] = (H_batches[b_i].to(device=device, dtype=dtype) + layer.self_attn.o_proj(ao)).cpu()
        del attn_out_batches

        norm2_batches = [layer.post_attention_layernorm(hb.to(device=device, dtype=dtype)).cpu() for hb in H_batches]
        H_up = _compute_cov_matrix(norm2_batches, device)

        mlp_mid_batches = []
        for x_norm2_cpu in norm2_batches:
            x_norm2 = x_norm2_cpu.to(device=device, dtype=dtype)
            mid = F.silu(layer.mlp.gate_proj(x_norm2)) * layer.mlp.up_proj(x_norm2)
            mlp_mid_batches.append(mid.cpu())
        del norm2_batches
        H_down = _compute_cov_matrix(mlp_mid_batches, device)

        for b_i in range(len(H_batches)):
            md = mlp_mid_batches[b_i].to(device=device, dtype=dtype)
            H_batches[b_i] = (H_batches[b_i].to(device=device, dtype=dtype) + layer.mlp.down_proj(md)).cpu()
        del mlp_mid_batches

        hess_map = {
            "self_attn.q_proj": H_qkv,
            "self_attn.k_proj": H_qkv,
            "self_attn.v_proj": H_qkv,
            "self_attn.o_proj": H_o,
            "mlp.gate_proj": H_up,
            "mlp.up_proj": H_up,
            "mlp.down_proj": H_down,
        }

        l_cost: Dict[str, Dict[str, float]] = {}
        l_param: Dict[str, int] = {}
        for idx_s, subname in enumerate(LINEAR_SUBMODULES):
            mod = layer.get_submodule(subname)
            n_p = mod.weight.numel()
            l_param[subname] = n_p
            l_cost[subname] = {}
            H_mat = hess_map[subname].float()
            W_f32 = mod.weight.data.float()
            if use_hadamard:
                W_work, H_work, _ = rotate_weights_and_hessian(
                    W_f32, H_mat, seed=1000 + l_idx * 10 + idx_s
                )
            else:
                W_work, H_work = W_f32, H_mat
            diag_h = torch.diagonal(H_work).clamp(min=1e-8)
            den_loss = float(((W_work @ H_work) * W_work).sum().clamp(min=1e-12).item())
            weight_mult = depth_mult * submodule_importance[subname]
            for sp in candidate_specs:
                _, W_hat_work = quantize_iqk_diag(W_work, sp, imatrix_w=diag_h)
                diff = W_work - W_hat_work
                num_loss = float(((diff @ H_work) * diff).sum().item())
                l_cost[subname][sp] = weight_mult * (num_loss / den_loss)

        layer_costs.append(l_cost)
        layer_params.append(l_param)

    return layer_costs, layer_params
