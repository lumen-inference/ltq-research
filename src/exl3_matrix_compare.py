import math, torch
from pathlib import Path
from safetensors.torch import load_file
import src.streaming_quant as sq
import src.liqk_v2 as lv2
from src.hybrid_quant import exl3_quantize_weight
dev = torch.device("cuda"); L = 12
model, store, config = sq.build_skeleton(Path("models/Qwen3-8B"), dev)
cal = load_file("data/qwen3/cal_trace.safetensors")["input_ids"][:8]
with torch.no_grad():
    hs = [model.model.embed_tokens(cal[i:i + 2].to(dev)) for i in range(0, 8, 2)]
    pos = model.model.rotary_emb(hs[0][:1], torch.arange(cal.shape[1], device=dev).unsqueeze(0))
    for l in range(L):
        layer = model.model.layers[l]; sq.load_module_weights(layer, f"model.layers.{l}", store, dev)
        hs = [layer(h, position_embeddings=pos) for h in hs]
        model.model.layers[l] = layer.to("meta")
    layer = model.model.layers[L]; sq.load_module_weights(layer, f"model.layers.{L}", store, dev)
    Hs = {}
    def hook(name):
        def f(m, a):
            x = a[0].reshape(-1, a[0].shape[-1]).float()
            Hs[name] = Hs.get(name, 0) + x.T @ x
        return f
    hk = [layer.get_submodule(n).register_forward_pre_hook(hook(n)) for n in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")]
    for h in hs:
        layer(h, position_embeddings=pos)
    for x in hk: x.remove()
pairs = [(2, "TCQ2X"), (2.5, "TCQ25X"), (3, "TCQ3X"), (3.5, "TCQ35X"), (4, "TCQ4X"), (5, "TCQ5X")]
print("matrix       K    | EXL3 rel.error | TCQ format(bpw) rel.error | our advantage, bits")
for name, H in [(k, v) for k, v in Hs.items() if k != "self_attn.q_proj"]:
    W = layer.get_submodule(name).weight.data.float()
    H = H / H.diagonal().mean()
    ref = float(torch.einsum("ij,jk,ik->", W, H, W))
    err = lambda Wq: float(torch.einsum("ij,jk,ik->", W - Wq.float(), H, W - Wq.float())) / ref
    damp = 0.02 * (1.5 if name.endswith("down_proj") else 1.0)
    for K, sp in pairs:
        e_x = err(exl3_quantize_weight(W, H, K, 7))
        _, wq, _ = lv2.quantize_matrix_liqk_v2(W, H, sp, seed=7, damp=damp, cd_sweeps=2)
        e_t = err(wq); b_t = lv2.LIQK_V2_SPECS[sp].bpw; b_x = K + 0.023
        adv = (b_x - b_t) + 0.5 * math.log2(e_x / e_t)
        print(f"{name:18s} {K:<4} | {e_x:.5f} | {sp:7s}({b_t:.2f}) {e_t:.5f} | {adv:+.3f}", flush=True)
