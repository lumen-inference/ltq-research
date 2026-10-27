"""
Utilities for working with .gguf files (including the new IQ2_K..IQ6_K types from PR #30715):
1. Decoding (dequantizing) any GGUF tensor via the native libggml-base.so library.
2. Loading quantized linear weights from .gguf directly into a PyTorch LlamaForCausalLM model
   (with correct inverse unpermute_qk permutation for q_proj and k_proj weights).
3. Patching .gguf files with our own packed blocks (e.g. LDLQ-IQ*_K)
   for direct testing in llama-bench and llama-perplexity.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ggml_type codes from ggml.h (including PR #30715)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30
GGML_TYPE_Q1_0 = 41
GGML_TYPE_Q2_0 = 42
# 43 and 44 are reserved for IQ2_NL and IQ3_NL in ggml.h
GGML_TYPE_IQ2_K = 45
GGML_TYPE_IQ3_K = 46
GGML_TYPE_IQ4_K = 47
GGML_TYPE_IQ5_K = 48
GGML_TYPE_IQ6_K = 49

# Mapping ggml_type -> (block size in elements, block size in bytes, name of the C dequantization function)
GGML_QUANT_INFO: Dict[int, Tuple[int, int, Optional[str]]] = {
    GGML_TYPE_F32: (1, 4, None),
    GGML_TYPE_F16: (1, 2, None),
    GGML_TYPE_BF16: (1, 2, None),
    GGML_TYPE_Q4_0: (32, 18, "dequantize_row_q4_0"),
    GGML_TYPE_Q4_1: (32, 20, "dequantize_row_q4_1"),
    GGML_TYPE_Q5_0: (32, 22, "dequantize_row_q5_0"),
    GGML_TYPE_Q5_1: (32, 24, "dequantize_row_q5_1"),
    GGML_TYPE_Q8_0: (32, 34, "dequantize_row_q8_0"),
    GGML_TYPE_Q2_K: (256, 84, "dequantize_row_q2_K"),
    GGML_TYPE_Q3_K: (256, 110, "dequantize_row_q3_K"),
    GGML_TYPE_Q4_K: (256, 144, "dequantize_row_q4_K"),
    GGML_TYPE_Q5_K: (256, 176, "dequantize_row_q5_K"),
    GGML_TYPE_Q6_K: (256, 210, "dequantize_row_q6_K"),
    GGML_TYPE_IQ2_XXS: (256, 66, "dequantize_row_iq2_xxs"),
    GGML_TYPE_IQ2_XS: (256, 74, "dequantize_row_iq2_xs"),
    GGML_TYPE_IQ3_XXS: (256, 98, "dequantize_row_iq3_xxs"),
    GGML_TYPE_IQ4_NL: (32, 18, "dequantize_row_iq4_nl"),
    GGML_TYPE_IQ3_S: (256, 110, "dequantize_row_iq3_s"),
    GGML_TYPE_IQ2_S: (256, 82, "dequantize_row_iq2_s"),
    GGML_TYPE_IQ4_XS: (256, 136, "dequantize_row_iq4_xs"),
    GGML_TYPE_IQ1_S: (256, 50, "dequantize_row_iq1_s"),
    GGML_TYPE_IQ1_M: (256, 56, "dequantize_row_iq1_m"),
    GGML_TYPE_IQ2_K: (256, 76, "dequantize_row_iq2_k"),
    GGML_TYPE_IQ3_K: (256, 110, "dequantize_row_iq3_k"),
    GGML_TYPE_IQ4_K: (256, 144, "dequantize_row_iq4_k"),
    GGML_TYPE_IQ5_K: (256, 176, "dequantize_row_iq5_k"),
    GGML_TYPE_IQ6_K: (256, 212, "dequantize_row_iq6_k"),
}

_LIBGGML: Optional[ctypes.CDLL] = None


def get_libggml(lib_path: Optional[str] = None) -> ctypes.CDLL:
    """Loads the compiled libggml-base.so library with IQ2_K..IQ6_K support."""
    global _LIBGGML
    if _LIBGGML is None:
        if lib_path is None:
            lib_path = os.environ.get("LUMEN_LIBGGML")
        if lib_path is None:
            default_path = (
                Path(__file__).resolve().parent.parent
                / "llama-iqk"
                / "build"
                / "bin"
                / "libggml-base.so"
            )
            lib_path = str(default_path)
        _LIBGGML = ctypes.CDLL(lib_path)
    return _LIBGGML


def permute_qk(weights: torch.Tensor, n_head: int) -> torch.Tensor:
    """
    Forward Q/K row permutation (from HuggingFace Llama layout to GGUF layout).
    In HF the head_dim halves are split (first half, second half); in GGUF they are interleaved in pairs (even, odd).
    """
    return (
        weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
        .swapaxes(1, 2)
        .reshape(weights.shape)
        .contiguous()
    )


def unpermute_qk(weights: torch.Tensor, n_head: int) -> torch.Tensor:
    """
    Inverse Q/K row permutation (from GGUF layout back to HuggingFace Llama layout).
    """
    return (
        weights.reshape(n_head, weights.shape[0] // n_head // 2, 2, *weights.shape[1:])
        .swapaxes(1, 2)
        .reshape(weights.shape)
        .contiguous()
    )


def _read_gguf_string(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _skip_gguf_value(f, vtype: int):
    # GGUF value types:
    # 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32, 6=FLOAT32, 7=BOOL,
    # 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
    fixed_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in fixed_sizes:
        f.seek(fixed_sizes[vtype], 1)
    elif vtype == 8:
        length = struct.unpack("<Q", f.read(8))[0]
        f.seek(length, 1)
    elif vtype == 9:
        elem_type, count = struct.unpack("<IQ", f.read(12))
        if elem_type in fixed_sizes:
            f.seek(fixed_sizes[elem_type] * count, 1)
        else:
            for _ in range(count):
                _skip_gguf_value(f, elem_type)
    else:
        raise ValueError(f"Unknown GGUF metadata type: {vtype}")


def parse_gguf_tensors(gguf_path: str | Path) -> Tuple[int, Dict[str, dict]]:
    """
    Fast GGUF V3 header parser (supports any new tensor types, including IQ2_K..IQ6_K=42..46,
    which the stock Python gguf package may not know).

    Returns:
        data_offset: absolute offset of the start of the data section in the file
        tensors: dict {name: {"dims": tuple, "ggml_type": int, "offset": int, "n_elements": int, "n_bytes": int}}
    """
    path = Path(gguf_path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"File {path} is not GGUF (magic={magic!r})")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        alignment = 32

        for _ in range(n_kv):
            key = _read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            if key == "general.alignment" and vtype == 4:
                alignment = struct.unpack("<I", f.read(4))[0]
            else:
                _skip_gguf_value(f, vtype)

        tensors = {}
        for _ in range(n_tensors):
            name = _read_gguf_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = tuple(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            ggml_type, offset = struct.unpack("<IQ", f.read(12))
            n_elements = 1
            for d in dims:
                n_elements *= d
            if ggml_type not in GGML_QUANT_INFO:
                raise ValueError(f"Tensor {name} has unsupported ggml_type={ggml_type}")
            blck_size, type_size, _ = GGML_QUANT_INFO[ggml_type]
            n_bytes = (n_elements // blck_size) * type_size
            tensors[name] = {
                "dims": dims,  # In GGUF dims = (ne[0]=K, ne[1]=M, ...)
                "ggml_type": ggml_type,
                "offset": offset,
                "n_elements": n_elements,
                "n_bytes": n_bytes,
            }

        pos = f.tell()
        data_offset = (pos + alignment - 1) // alignment * alignment

    return data_offset, tensors


def dequantize_gguf_tensor(
    raw_bytes: bytes | np.ndarray,
    ggml_type: int,
    shape_pytorch: Tuple[int, ...],
) -> torch.Tensor:
    """
    Dequantizes a tensor from raw GGUF bytes into a float32 PyTorch tensor of the given shape.
    """
    n_elements = 1
    for s in shape_pytorch:
        n_elements *= s

    if ggml_type == GGML_TYPE_F32:
        arr = np.frombuffer(raw_bytes, dtype=np.float32, count=n_elements).copy()
        return torch.from_numpy(arr).reshape(shape_pytorch)
    if ggml_type == GGML_TYPE_F16:
        arr = np.frombuffer(raw_bytes, dtype=np.float16, count=n_elements).astype(np.float32)
        return torch.from_numpy(arr).reshape(shape_pytorch)
    if ggml_type == GGML_TYPE_BF16:
        u16 = np.frombuffer(raw_bytes, dtype=np.uint16, count=n_elements)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32).copy()
        return torch.from_numpy(arr).reshape(shape_pytorch)

    _, _, func_name = GGML_QUANT_INFO[ggml_type]
    assert func_name is not None
    lib = get_libggml()
    fn = getattr(lib, func_name)
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    fn.restype = None

    if isinstance(raw_bytes, np.ndarray):
        src_buf = np.ascontiguousarray(raw_bytes, dtype=np.uint8)
        src_ptr = src_buf.ctypes.data_as(ctypes.c_void_p)
    else:
        src_ptr = ctypes.cast(ctypes.c_char_p(raw_bytes), ctypes.c_void_p)

    out = np.empty(n_elements, dtype=np.float32)
    fn(src_ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_elements)
    return torch.from_numpy(out).reshape(shape_pytorch)


def load_gguf_into_hf_model(
    model: torch.nn.Module,
    gguf_path: str | Path,
    include_embeddings: bool = False,
) -> dict:
    """
    Loads quantized linear-layer weights from a .gguf file into a HuggingFace LlamaForCausalLM model.
    By default (include_embeddings=False) only the 112 linear matrices of the transformer blocks are updated
    (for a fair comparison of linear-layer quantization with identical BF16 embedding/lm_head).
    If include_embeddings=True, token_embd.weight (and output.weight, if present) is dequantized as well.

    Returns statistics for the loaded layers and the average bitrate of the linear weights (bpw).
    """
    data_offset, tensors = parse_gguf_tensors(gguf_path)
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    n_layers = model.config.num_hidden_layers

    mapping = {
        "attn_q": ("self_attn.q_proj", n_heads),
        "attn_k": ("self_attn.k_proj", n_kv_heads),
        "attn_v": ("self_attn.v_proj", None),
        "attn_output": ("self_attn.o_proj", None),
        "ffn_gate": ("mlp.gate_proj", None),
        "ffn_up": ("mlp.up_proj", None),
        "ffn_down": ("mlp.down_proj", None),
    }

    total_linear_params = 0
    total_linear_bits = 0

    with open(gguf_path, "rb") as f:
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            for gguf_suffix, (hf_subpath, permute_heads) in mapping.items():
                tname = f"blk.{layer_idx}.{gguf_suffix}.weight"
                info = tensors[tname]
                f.seek(data_offset + info["offset"])
                raw = f.read(info["n_bytes"])

                # In GGUF dims = (K=in_features, M=out_features), in PyTorch shape = (M, K)
                K, M = info["dims"][0], info["dims"][1]
                w_f32 = dequantize_gguf_tensor(raw, info["ggml_type"], (M, K))

                if permute_heads is not None:
                    w_f32 = unpermute_qk(w_f32, permute_heads)

                target_mod = layer.get_submodule(hf_subpath)
                target_mod.weight.data.copy_(w_f32.to(dtype=target_mod.weight.dtype, device=target_mod.weight.device))

                total_linear_params += info["n_elements"]
                total_linear_bits += info["n_bytes"] * 8

        if include_embeddings and "token_embd.weight" in tensors:
            info = tensors["token_embd.weight"]
            f.seek(data_offset + info["offset"])
            raw = f.read(info["n_bytes"])
            K, M = info["dims"][0], info["dims"][1]
            w_f32 = dequantize_gguf_tensor(raw, info["ggml_type"], (M, K))
            model.model.embed_tokens.weight.data.copy_(
                w_f32.to(dtype=model.model.embed_tokens.weight.dtype, device=model.model.embed_tokens.weight.device)
            )
            if getattr(model.config, "tie_word_embeddings", False):
                model.lm_head.weight = model.model.embed_tokens.weight
            elif "output.weight" in tensors:
                info_out = tensors["output.weight"]
                f.seek(data_offset + info_out["offset"])
                raw_out = f.read(info_out["n_bytes"])
                Ko, Mo = info_out["dims"][0], info_out["dims"][1]
                wo_f32 = dequantize_gguf_tensor(raw_out, info_out["ggml_type"], (Mo, Ko))
                model.lm_head.weight.data.copy_(
                    wo_f32.to(dtype=model.lm_head.weight.dtype, device=model.lm_head.weight.device)
                )

    return {
        "linear_params": total_linear_params,
        "linear_bytes": total_linear_bits // 8,
        "linear_bpw": total_linear_bits / max(total_linear_params, 1),
    }


def patch_gguf_linear_tensors(
    template_gguf_path: str | Path,
    out_gguf_path: str | Path,
    packed_tensors: Dict[str, np.ndarray],
    n_heads: int,
    n_kv_heads: int,
):
    """
    Creates a new valid .gguf file from a template (of the same bitrate, e.g. Llama-3.2-1B-IQ3_K-imatrix.gguf),
    replacing the binary blocks of the transformer linear tensors with our packed blocks (e.g. LDLQ-IQ3_K).
    Automatically applies the forward permute_qk row permutation for attn_q and attn_k.

    Parameters:
        template_gguf_path: source .gguf file with the same tensor-type layout
        out_gguf_path: output .gguf file
        packed_tensors: dict {"blk.{i}.{attn_q|...}.weight": uint8 array of shape (n_blocks, block_bytes)}
                        in HuggingFace row order
    """
    shutil.copyfile(template_gguf_path, out_gguf_path)
    data_offset, tensors = parse_gguf_tensors(out_gguf_path)

    with open(out_gguf_path, "r+b") as f:
        for tname, packed in packed_tensors.items():
            if tname not in tensors:
                raise KeyError(f"Tensor {tname} is missing from template {template_gguf_path}")
            info = tensors[tname]
            K, M = info["dims"][0], info["dims"][1]
            row_bytes = info["n_bytes"] // M
            packed_rows = packed.reshape(M, row_bytes)

            if tname.endswith(".attn_q.weight"):
                t_rows = torch.from_numpy(packed_rows)
                packed_rows = permute_qk(t_rows, n_heads).numpy()
            elif tname.endswith(".attn_k.weight"):
                t_rows = torch.from_numpy(packed_rows)
                packed_rows = permute_qk(t_rows, n_kv_heads).numpy()

            raw_bytes = np.ascontiguousarray(packed_rows, dtype=np.uint8).tobytes()
            if len(raw_bytes) != info["n_bytes"]:
                raise ValueError(
                    f"Size mismatch for {tname}: got {len(raw_bytes)} bytes, expected {info['n_bytes']}"
                )
            f.seek(data_offset + info["offset"])
            f.write(raw_bytes)
