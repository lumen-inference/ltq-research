"""
Quality evaluation of quantized models via Teacher-Forced KL-Divergence (KLD).

Metrics computed on traces of real dialogues (nemotron_trace.json and multilingual_trace.json),
only over the generated assistant responses (score_spans):
1. mean_kld: mean Kullback-Leibler divergence KL(P_ref || P_quant) in nats over all response tokens.
2. ci95_lo, ci95_hi: 95% confidence interval (paired bootstrap over dialogues, B=2000, seed=42).
3. median_kld, p90_kld, p99_kld: quantiles of the per-token KLD distribution.
4. top1_match: fraction of tokens where argmax P_quant(v) matches argmax P_ref(v) (1 - flip_rate).
5. ppl_ref, ppl_quant: perplexity of the BF16 reference and the quantized model on the trace's response tokens.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_eval_trace(trace_path: str | Path) -> List[dict]:
    """Loads a JSON trace and normalizes the rows into a list of dicts {id, token_ids, score_spans}."""
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "sequences" in data:
        return data["sequences"]
    rows = data["rows"]
    out = []
    for idx, r in enumerate(rows):
        t_ids = r["input_ids"] + r["response_ids"]
        out.append(
            {
                "id": r.get("uuid") or f"row_{idx}",
                "token_ids": t_ids,
                "score_spans": r["score_spans"],
            }
        )
    return out


def get_scored_positions_and_targets(seq: dict, max_seq_len: int = 8192) -> Tuple[List[int], List[int]]:
    """
    Given the list of score_spans intervals, returns:
      - pos_indices: indices of logit positions (t - 1) that predict target token t
      - target_tokens: the target tokens themselves, token_ids[t]
    """
    token_ids = seq["token_ids"][:max_seq_len]
    n_tok = len(token_ids)
    pos_indices: List[int] = []
    target_tokens: List[int] = []
    for s, e in seq["score_spans"]:
        s_clamped = max(1, int(s))
        e_clamped = min(n_tok, int(e))
        for t in range(s_clamped, e_clamped):
            pos_indices.append(t - 1)
            target_tokens.append(token_ids[t])
    return pos_indices, target_tokens


@torch.no_grad()
def precompute_ref_trace_data(
    ref_model: torch.nn.Module,
    sequences: List[dict],
    device: torch.device,
    max_seq_len: int = 8192,
) -> List[dict]:
    """
    Runs the BF16 reference model over the trace and caches in CPU RAM (as bfloat16)
    the logits only at the scored positions of the assistant responses.
    For 18k–27k tokens with a 128256 vocabulary this takes ~4.7–7.0 GB of 64 GB RAM
    and makes evaluating all subsequent quants 2x faster.
    """
    ref_model.eval()
    cached = []
    for seq in sequences:
        pos_idx, targets = get_scored_positions_and_targets(seq, max_seq_len=max_seq_len)
        if not pos_idx:
            continue
        input_ids = torch.tensor([seq["token_ids"][:max_seq_len]], dtype=torch.long, device=device)
        pos_t = torch.tensor(pos_idx, dtype=torch.long, device=device)
        if hasattr(ref_model, "model") and hasattr(ref_model, "lm_head"):
            hidden = ref_model.model(input_ids=input_ids, use_cache=False).last_hidden_state
            ref_logits_cpu = ref_model.lm_head(hidden[0, pos_t, :]).to(dtype=torch.bfloat16).cpu()
            del hidden
        else:
            out = ref_model(input_ids=input_ids, use_cache=False)
            ref_logits_cpu = out.logits[0, pos_t, :].to(dtype=torch.bfloat16).cpu()
            del out
        cached.append(
            {
                "id": seq["id"],
                "token_ids": seq["token_ids"][:max_seq_len],
                "pos_idx": pos_idx,
                "targets": targets,
                "ref_logits_bf16": ref_logits_cpu,
            }
        )
    return cached


@torch.no_grad()
def evaluate_model_on_cached_trace(
    q_model: torch.nn.Module,
    cached_trace: List[dict],
    device: torch.device,
    n_boot: int = 2000,
    seed: int = 42,
    forward_fn=None,
) -> dict:
    """
    Evaluates a quantized model (or an arbitrary function forward_fn(input_ids) -> logits)
    against the precomputed reference logits in cached_trace.

    All probabilities and logarithms are computed strictly in float32 over the full vocabulary V=128256:
        KL(P_ref || P_q) = sum_v P_ref(v) * (log P_ref(v) - log P_q(v))
    """
    if q_model is not None:
        q_model.eval()

    seq_kld_sums = []
    seq_tok_counts = []
    all_token_klds = []
    top1_matches = 0
    total_tokens = 0
    nll_ref_sum = 0.0
    nll_q_sum = 0.0
    seq_nll_ref_sums = []
    seq_nll_q_sums = []

    t0 = time.perf_counter()
    for item in cached_trace:
        input_ids = torch.tensor([item["token_ids"]], dtype=torch.long, device=device)
        if forward_fn is not None:
            logits_full = forward_fn(input_ids)
            pos_t = torch.tensor(item["pos_idx"], dtype=torch.long, device=logits_full.device)
            q_logits = logits_full[0, pos_t, :].to(device=device, dtype=torch.float32)
            del logits_full
        elif hasattr(q_model, "model") and hasattr(q_model, "lm_head"):
            pos_t = torch.tensor(item["pos_idx"], dtype=torch.long, device=device)
            hidden = q_model.model(input_ids=input_ids, use_cache=False).last_hidden_state
            q_logits = q_model.lm_head(hidden[0, pos_t, :]).to(device=device, dtype=torch.float32)
            del hidden
        else:
            logits_full = q_model(input_ids=input_ids, use_cache=False).logits
            pos_t = torch.tensor(item["pos_idx"], dtype=torch.long, device=logits_full.device)
            q_logits = logits_full[0, pos_t, :].to(device=device, dtype=torch.float32)
            del logits_full
        ref_logits = item["ref_logits_bf16"].to(device=device, dtype=torch.float32)
        targets_t = torch.tensor(item["targets"], dtype=torch.long, device=device)

        log_p_ref = F.log_softmax(ref_logits, dim=-1)
        log_p_q = F.log_softmax(q_logits, dim=-1)
        p_ref = log_p_ref.exp()

        # Exact KL(P_ref || P_q) for each response token
        tok_kld = (p_ref * (log_p_ref - log_p_q)).sum(dim=-1)
        tok_kld = torch.clamp(tok_kld, min=0.0)

        tok_kld_np = tok_kld.cpu().numpy()
        all_token_klds.append(tok_kld_np)
        seq_kld_sums.append(float(tok_kld_np.sum()))
        seq_tok_counts.append(len(tok_kld_np))

        # Top-1 agreement with the reference
        ref_top1 = ref_logits.argmax(dim=-1)
        q_top1 = q_logits.argmax(dim=-1)
        top1_matches += int((ref_top1 == q_top1).sum().item())
        total_tokens += len(tok_kld_np)

        # NLL / Perplexity on the ground-truth trace tokens
        nll_ref = -log_p_ref.gather(-1, targets_t.unsqueeze(-1)).squeeze(-1)
        nll_q = -log_p_q.gather(-1, targets_t.unsqueeze(-1)).squeeze(-1)
        nll_ref_sum += float(nll_ref.sum().item())
        nll_q_sum += float(nll_q.sum().item())
        seq_nll_ref_sums.append(float(nll_ref.sum().item()))
        seq_nll_q_sums.append(float(nll_q.sum().item()))

    elapsed = time.perf_counter() - t0
    all_klds_flat = np.concatenate(all_token_klds, axis=0)
    seq_kld_sums_arr = np.array(seq_kld_sums, dtype=np.float64)
    seq_tok_counts_arr = np.array(seq_tok_counts, dtype=np.float64)

    mean_kld = float(seq_kld_sums_arr.sum() / seq_tok_counts_arr.sum())

    # Bootstrap 95% confidence interval over dialogues
    rng = np.random.default_rng(seed)
    n_seqs = len(seq_kld_sums_arr)
    boot_Idx = rng.integers(0, n_seqs, size=(n_boot, n_seqs))
    boot_means = (
        seq_kld_sums_arr[boot_Idx].sum(axis=1) / seq_tok_counts_arr[boot_Idx].sum(axis=1)
    )
    ci95_lo = float(np.percentile(boot_means, 2.5))
    ci95_hi = float(np.percentile(boot_means, 97.5))

    # The same per-dialogue bootstrap for the quantized model's perplexity
    seq_nll_q_arr = np.array(seq_nll_q_sums, dtype=np.float64)
    boot_ppl = np.exp(seq_nll_q_arr[boot_Idx].sum(axis=1) / seq_tok_counts_arr[boot_Idx].sum(axis=1))

    return {
        "mean_kld": mean_kld,
        "ci95_lo": ci95_lo,
        "ci95_hi": ci95_hi,
        "median_kld": float(np.median(all_klds_flat)),
        "p90_kld": float(np.percentile(all_klds_flat, 90.0)),
        "p99_kld": float(np.percentile(all_klds_flat, 99.0)),
        "top1_match": top1_matches / max(total_tokens, 1),
        "flip_rate": 1.0 - (top1_matches / max(total_tokens, 1)),
        "ppl_ref": math.exp(nll_ref_sum / max(total_tokens, 1)),
        "ppl_quant": math.exp(nll_q_sum / max(total_tokens, 1)),
        "ppl_ci95_lo": float(np.percentile(boot_ppl, 2.5)),
        "ppl_ci95_hi": float(np.percentile(boot_ppl, 97.5)),
        "n_sequences": n_seqs,
        "n_tokens": total_tokens,
        "eval_sec": elapsed,
        "per_seq_kld_sums": seq_kld_sums,
        "per_seq_tok_counts": seq_tok_counts,
        "per_seq_nll_ref_sums": seq_nll_ref_sums,
        "per_seq_nll_q_sums": seq_nll_q_sums,
    }


def paired_bootstrap_diff(res_a: dict, res_b: dict, n_boot: int = 2000, seed: int = 42) -> dict:
    """
    Paired bootstrap of the KLD difference between method A and method B: delta = KLD(A) - KLD(B).
    If ci95_hi < 0, method A is statistically significantly better than method B at p < 0.05.
    """
    sums_a = np.array(res_a["per_seq_kld_sums"], dtype=np.float64)
    sums_b = np.array(res_b["per_seq_kld_sums"], dtype=np.float64)
    counts = np.array(res_a["per_seq_tok_counts"], dtype=np.float64)

    n_seqs = len(counts)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seqs, size=(n_boot, n_seqs))

    tot_tok = counts[idx].sum(axis=1)
    mean_a = sums_a[idx].sum(axis=1) / tot_tok
    mean_b = sums_b[idx].sum(axis=1) / tot_tok
    diff = mean_a - mean_b

    return {
        "mean_diff": float(res_a["mean_kld"] - res_b["mean_kld"]),
        "ci95_lo": float(np.percentile(diff, 2.5)),
        "ci95_hi": float(np.percentile(diff, 97.5)),
        "p_a_better": float((diff < 0).mean()),
    }


def paired_bootstrap_ppl_diff(res_a: dict, res_b: dict, n_boot: int = 2000, seed: int = 42) -> dict:
    """
    Paired bootstrap over dialogues of the perplexity difference delta = PPL(A) - PPL(B).
    Requires per_seq_nll_q_sums in both results.
    """
    nll_a = np.array(res_a["per_seq_nll_q_sums"], dtype=np.float64)
    nll_b = np.array(res_b["per_seq_nll_q_sums"], dtype=np.float64)
    counts = np.array(res_a["per_seq_tok_counts"], dtype=np.float64)

    n_seqs = len(counts)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seqs, size=(n_boot, n_seqs))

    tot_tok = counts[idx].sum(axis=1)
    diff = np.exp(nll_a[idx].sum(axis=1) / tot_tok) - np.exp(nll_b[idx].sum(axis=1) / tot_tok)

    return {
        "mean_diff": float(res_a["ppl_quant"] - res_b["ppl_quant"]),
        "ci95_lo": float(np.percentile(diff, 2.5)),
        "ci95_hi": float(np.percentile(diff, 97.5)),
        "frac_a_better": float((diff < 0).mean()),
    }
