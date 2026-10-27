"""
Per-row Fisher for models that fit entirely on the GPU, with any layer architecture (Gemma 4 with PLE and shared KV,
Qwen3.5 with Gated DeltaNet): all layers are loaded, a regular backward pass over the whole model, the gradient is taken at the output
of every quantized linear matrix (`hybrid_quant.quantizable`). Unlike the layer-by-layer `fisher_offload`, the gradient
also flows through the shared KV (late Gemma 4 layers read K/V of earlier ones). Sampling is as in liqk_v2 (inverse CDF
with fixed random numbers, 2 samples per position). Output: {(layer, matrix name within the layer): row weights}
for `hybrid_quant --fisher`.

    python -m src.fisher_full --model_dir models/Qwen3.5-4B --data_dir data/qwen35 --out data/qwen35/fisher.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from src.hybrid_quant import TextModel, quantizable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_seqs", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=1024)
    args = ap.parse_args()
    dev = torch.device("cuda")
    tmod = TextModel(args.model_dir, dev)
    for l in range(len(tmod.layers)):
        tmod.load_layer(l)
    tm = tmod.tm
    tm.layers = nn.ModuleList(tmod.layers)
    for p in tmod.model.parameters():
        p.requires_grad_(False)
    # weights are frozen: the graph starts at the embedding output
    tm.embed_tokens.register_forward_hook(lambda m, a, out: out.detach().requires_grad_(True))

    fisher = {}
    for l, layer in enumerate(tmod.layers):
        for name, mod in layer.named_modules():
            if not quantizable(name, mod):
                continue
            key = (l, name)
            fisher[key] = torch.zeros(mod.out_features, dtype=torch.float64, device=dev)

            def fwd(m, inp, out, key=key):
                def acc(g):  # the gradient hook must return None (otherwise it replaces the gradient)
                    fisher[key].add_(g.detach().float().square().reshape(-1, g.shape[-1]).sum(dim=0).double())
                out.register_hook(acc)
            mod.register_forward_hook(fwd)

    cal = load_file(str(args.data_dir / "cal_trace.safetensors"))["input_ids"]
    total = 0
    for i in range(min(args.n_seqs, cal.shape[0])):
        tok = cal[i : i + 1, : args.seq_len].to(dev)
        with torch.enable_grad():
            h = tm(input_ids=tok, use_cache=False).last_hidden_state
            logits = tmod.logits(h).reshape(-1, tmod.config.vocab_size)
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=-1)
                u = torch.rand((probs.shape[0], 2), generator=torch.Generator().manual_seed(1000 + i)).to(dev)
                cdf = probs.cumsum(dim=-1)
                y = torch.searchsorted(cdf, (u * cdf[:, -1:]).contiguous()).clamp(max=probs.shape[-1] - 1)
                del probs, cdf
            loss = 0.5 * (F.cross_entropy(logits.float(), y[:, 0], reduction="sum")
                          + F.cross_entropy(logits.float(), y[:, 1], reduction="sum"))
            loss.backward()
        del h, logits, loss
        total += tok.numel()
        print(f"sequence {i + 1}: done", flush=True)
    res = {k: (v / max(total, 1)).float().cpu() for k, v in fisher.items()}
    torch.save(res, args.out)
    print(f"Fisher: {len(res)} matrices -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
