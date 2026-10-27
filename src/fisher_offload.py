"""
Per-row Fisher (like `liqk_v2.compute_exact_row_fisher_sensitivities`) for models that do not fit on the GPU: weights stay in RAM,
layers are moved to the GPU one at a time. The forward pass stores layer inputs on the CPU; the backward pass goes over layers in reverse
order: each layer is recomputed from its stored input and the gradient is propagated through it (manual checkpointing).
Sampling is as in liqk_v2 (inverse CDF with fixed random numbers, 2 samples per position).
Output: {(layer, matrix): row weights} in a .pt file for `streaming_quant --fisher`.

    python -m src.fisher_offload --model_dir models/Qwen3-8B --data_dir data/qwen3 --out data/qwen3/fisher.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from src.quantize_model_iqk import LINEAR_SUBMODULES

ATTN_SUBMODULES = [n for n in LINEAR_SUBMODULES if n.startswith("self_attn.")]


def layer_keys(layer) -> list[tuple[str, int]]:
    """Layer matrices and their output lengths: dense MLP as in LINEAR_SUBMODULES; MoE: attention plus every matrix
    of every expert (`mlp.experts.{e}.gate_up_proj` / `.down_proj`; the `mlp.gate` router is not quantized)."""
    ex = getattr(layer.mlp, "experts", None)
    if ex is None or not hasattr(ex, "gate_up_proj"):
        return [(n, layer.get_submodule(n).out_features) for n in LINEAR_SUBMODULES]
    keys = [(n, layer.get_submodule(n).out_features) for n in ATTN_SUBMODULES]
    for e in range(ex.num_experts):
        keys += [(f"mlp.experts.{e}.gate_up_proj", ex.gate_up_proj.shape[1]), (f"mlp.experts.{e}.down_proj", ex.down_proj.shape[1])]
    return keys


def experts_forward_with_hooks(experts, add):
    """Qwen3-MoE experts forward (as in transformers), but the outputs of each expert's matrices pass their gradient to `add(key, g)`."""
    def forward(hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=experts.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit[:, 0].tolist():
            pos, tok = torch.where(mask[e])
            gu = F.linear(hidden_states[tok], experts.gate_up_proj[e])
            if gu.requires_grad:
                gu.register_hook(lambda g, e=e: add(f"mlp.experts.{e}.gate_up_proj", g))
            gate, up = gu.chunk(2, dim=-1)
            dn = F.linear(experts.act_fn(gate) * up, experts.down_proj[e])
            if dn.requires_grad:
                dn.register_hook(lambda g, e=e: add(f"mlp.experts.{e}.down_proj", g))
            out.index_add_(0, tok, (dn * top_k_weights[tok, pos, None]).to(out.dtype))
        return out
    return forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_seqs", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=1024)
    args = ap.parse_args()
    dev = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), dtype=torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    inner = model.model
    inner.embed_tokens.to(dev)
    inner.norm.to(dev)
    inner.rotary_emb.to(dev)
    model.lm_head.to(dev)
    layers = inner.layers
    n_layers = len(layers)
    fisher = {(l, sub): torch.zeros(n, dtype=torch.float64, device=dev) for l in range(n_layers) for sub, n in layer_keys(layers[l])}
    cal = load_file(str(args.data_dir / "cal_trace.safetensors"))["input_ids"][:64]
    n = min(args.n_seqs, cal.shape[0])
    toks = [cal[i : i + 1, : args.seq_len].to(dev) for i in range(n)]
    total = sum(t.numel() for t in toks)
    pos = inner.rotary_emb(torch.zeros(1, device=dev, dtype=torch.bfloat16), torch.arange(toks[0].shape[1], device=dev).unsqueeze(0))
    # layers are the outer loop, sequences the inner one: each layer is moved to the GPU once per pass
    with torch.no_grad():
        hs = [inner.embed_tokens(t) for t in toks]
        inputs = []
        for l in range(n_layers):
            inputs.append([h.cpu() for h in hs])
            layers[l].to(dev)
            hs = [layers[l](h, position_embeddings=pos) for h in hs]
            layers[l].to("cpu")
    gs = []
    for i, h in enumerate(hs):
        h = h.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model.lm_head(inner.norm(h)).reshape(-1, model.config.vocab_size)
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=-1)
                u = torch.rand((probs.shape[0], 2), generator=torch.Generator().manual_seed(1000 + i)).to(dev)
                cdf = probs.cumsum(dim=-1)
                y = torch.searchsorted(cdf, (u * cdf[:, -1:]).contiguous()).clamp(max=probs.shape[-1] - 1)
                del probs, cdf
            loss = 0.5 * (F.cross_entropy(logits.float(), y[:, 0], reduction="sum")
                          + F.cross_entropy(logits.float(), y[:, 1], reduction="sum"))
            gs.append(torch.autograd.grad(loss, h)[0])
        del logits, loss
    del hs
    for l in reversed(range(n_layers)):
        layer = layers[l].to(dev)
        hooks = []
        ex = getattr(layer.mlp, "experts", None)
        moe = ex is not None and hasattr(ex, "gate_up_proj")
        for sub in (ATTN_SUBMODULES if moe else LINEAR_SUBMODULES):
            def fwd(m, inp, out, key=(l, sub)):
                def acc(gr, key=key):  # the gradient hook must return None (otherwise it replaces the gradient)
                    fisher[key].add_(gr.detach().float().square().sum(dim=(0, 1)).double())
                out.register_hook(acc)
            hooks.append(layer.get_submodule(sub).register_forward_hook(fwd))
        if moe:
            def add(sub, g, l=l):  # g [expert tokens, out]; normalized over all tokens, as for dense matrices
                fisher[(l, sub)].add_(g.detach().float().square().sum(dim=0).double())
            ex.forward = experts_forward_with_hooks(ex, add)
        for i in range(n):
            x = inputs[l][i].to(dev).requires_grad_(True)
            with torch.enable_grad():
                out = layer(x, position_embeddings=pos)
                out = out[0] if isinstance(out, tuple) else out
                gs[i] = torch.autograd.grad(out, x, grad_outputs=gs[i])[0]
            del x, out
        for hk in hooks:
            hk.remove()
        if moe:
            del ex.forward  # revert to the class forward
        layers[l].to("cpu")
        inputs[l] = None
        if l % 8 == 0:
            print(f"layer {l}: done", flush=True)
    res = {k: (v / max(total, 1)).float().cpu() for k, v in fisher.items()}
    torch.save(res, args.out)
    print(f"Fisher: {len(res)} matrices -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
