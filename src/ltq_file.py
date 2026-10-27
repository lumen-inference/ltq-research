"""
The .ltq container (lumen-rs, docs/EN/ltq-format.md, sections 3, 5, 6): writing, reading and full matrix decode on GPU.

In packing mode (`liqk_v2.PACK`) the quantizer emits, for each matrix, the tile codes (label * U + u and gain index
per position), tile values G, fp16 d_row, int8 e and fp16 suh; here they are laid out into the spec's tile fields
(SoA per row slice: codes, planes, lsb, gains, each region on a 64-byte boundary) and written to a single file with
a JSON header. `decode_matrix` is a vectorized decoder following the spec (not the quantizer): it verifies every
written matrix and also serves as the reference for `lumen-format`.

    python -m src.ltq_file --check model.ltq   # header, sizes, full decode of a few matrices
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from src.iqk_format import fwht_inplace
from src.ltq_format import NU, TILE, LtqFormat

MAGIC = b"LTQ\0"
VERSION = 1
DTYPES = {"f32": (np.float32, torch.float32), "f16": (np.float16, torch.float16), "bf16": (np.uint16, torch.bfloat16),
          "i32": (np.int32, torch.int32), "i8": (np.int8, torch.int8), "u8": (np.uint8, torch.uint8)}
TORCH_TO_LTQ = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16", torch.int32: "i32", torch.int8: "i8",
                torch.uint8: "u8"}
FRAC_NAMES = ["none", "d2", "d4"]


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def format_json(f: LtqFormat) -> dict:
    """Format description for the header: lut and gain as lossless float32 (the repr of a float32 value determines it uniquely)."""
    return {"name": f.name, "K": f.K, "frac": FRAC_NAMES[f.frac], "masks": [int(m) for m in f.masks], "h": f.h, "sub": f.sub,
            "lut": [float(v) for v in np.asarray(f.lut, np.float32)], "gain": [float(v) for v in np.asarray(f.gain_tab, np.float32)]}


def format_from_json(d: dict) -> LtqFormat:
    return LtqFormat(d["name"], d["K"], FRAC_NAMES.index(d["frac"]), tuple(d["masks"]), d["h"], 0.0, d["sub"],
                     np.asarray(d["lut"], np.float32), np.asarray(d["gain"], np.float32))


# ------------------------------------------------------------------------------------------------
# Tile fields
# ------------------------------------------------------------------------------------------------

def _words(bits: torch.Tensor) -> torch.Tensor:
    """[T, n] bits -> [T, ceil(n / 16)] u16 (int32): bit j of word w is element 16 w + j."""
    T, n = bits.shape
    n16 = _align(n, 16)
    if n16 != n:
        bits = torch.cat([bits, bits.new_zeros(T, n16 - n)], dim=1)
    sh = torch.arange(16, device=bits.device, dtype=torch.int32)
    return (bits.view(T, -1, 16).to(torch.int32) << sh).sum(dim=2)


def _bits(words: torch.Tensor, n: int) -> torch.Tensor:
    sh = torch.arange(16, device=words.device, dtype=torch.int32)
    return ((words.to(torch.int32).unsqueeze(2) >> sh) & 1).reshape(words.shape[0], -1)[:, :n]


def _tiles(x: torch.Tensor) -> torch.Tensor:
    """[R, K] -> [T, 256]: tiles by row group, by kb within a group; position t = 16 r + c."""
    R, K = x.shape
    return x.reshape(R // 16, 16, K // 16, 16).permute(0, 2, 1, 3).reshape(-1, TILE)


def _untiles(t: torch.Tensor, R: int, K: int) -> torch.Tensor:
    return t.reshape(R // 16, K // 16, 16, 16).permute(0, 2, 1, 3).reshape(R, K)


def pack_slice(idx: torch.Tensor, gi: torch.Tensor, f: LtqFormat) -> bytes:
    """Row slice data [R, K] (label * U + u and gain index per position): four SoA regions on 64-byte boundaries."""
    ix, g = _tiles(idx.to(torch.int32)), _tiles(gi.to(torch.int32))
    T = ix.shape[0]
    lab, u = ix // f.U, ix % f.U
    regions = [_words(lab & 1)]
    regions.append(torch.stack([_words((u >> p) & 1) for p in range(1, f.K - 1)], dim=1).reshape(T, -1) if f.K > 2
                   else ix.new_zeros(T, 0))
    lsb = u & 1
    if f.frac == 1:
        lsb = lsb[:, 0::2]
    elif f.frac == 2:
        lsb = lsb.view(T, -1, 4)[:, :, :3].reshape(T, -1)
    regions.append(_words(lsb))
    if f.h:
        gs = g[:, :: f.sub]
        regions.append(_words(((gs.unsqueeze(2) >> torch.arange(f.h, device=gs.device)) & 1).reshape(T, -1)))
    else:
        regions.append(ix.new_zeros(T, 0))
    out = bytearray()
    for r in regions:
        out += bytes(_align(len(out), 64) - len(out))
        out += r.to(torch.int32).cpu().numpy().astype(np.uint16).tobytes()
    return bytes(out)


def _label_table(f: LtqFormat) -> torch.Tensor:
    """Label for state s (10 bits): position bit and parities of (s >> 1) & m_i."""
    s = np.arange(1 << NU)
    lab = s & 1
    for i, m in enumerate(f.masks):
        v = (s >> 1) & m
        par = np.zeros_like(v)
        while v.any():
            par ^= v & 1
            v >>= 1
        lab |= par << (i + 1)
    return torch.tensor(lab, dtype=torch.int64)


def decode_slice(data: memoryview | bytes, R: int, K: int, f: LtqFormat, device) -> torch.Tensor:
    """Unscaled slice values [R, K]: gain[g] * lut[label * U + u] per the spec (sections 2–3)."""
    T = (R // 16) * (K // 16)
    sizes = [16, 16 * (f.K - 2), f.lsb_words(), f.gain_words()]
    regions, off = [], 0
    for n in sizes:
        off = _align(off, 64)
        regions.append(torch.frombuffer(bytearray(data[off : off + 2 * T * n]), dtype=torch.int16).to(device).view(T, n)
                       if n else None)
        off += 2 * T * n
    c = _bits(regions[0], TILE)
    s = sum(torch.roll(c, j, dims=1) << j for j in range(NU))  # s_t = sum_j c_{(t - j) mod 256} 2^j
    lab = _label_table(f).to(device)[s]
    lb = _bits(regions[2], {0: TILE, 1: TILE // 2, 2: 3 * TILE // 4}[f.frac])
    if f.frac == 1:
        lb = lb.repeat_interleave(2, dim=1)
    elif f.frac == 2:
        b3 = lb.view(T, -1, 3)
        lb = torch.cat([b3, b3.sum(dim=2, keepdim=True) & 1], dim=2).reshape(T, TILE)
    u = lb
    for p in range(1, f.K - 1):
        u = u | (_bits(regions[1][:, 16 * (p - 1) : 16 * p], TILE) << p)
    if f.h:
        gb = _bits(regions[3], TILE // f.sub * f.h).view(T, -1, f.h)
        g = (gb << torch.arange(f.h, device=device)).sum(dim=2).repeat_interleave(f.sub, dim=1)
        gain = torch.tensor(f.gain_tab, device=device)[g]
    else:
        gain = torch.ones((T, TILE), device=device)
    vals = gain * torch.tensor(f.lut, device=device)[lab * f.U + u]
    return _untiles(vals, R, K)


def assemble(G: torch.Tensor, d_row: torch.Tensor, e: torch.Tensor, suh: torch.Tensor) -> torch.Tensor:
    """W = H256(d · q) ⊙ suh over columns, d[m, b] = d_row[m] · 2^(e[m, b] / 4) (section 1)."""
    M, K = G.shape
    scale = d_row.float().unsqueeze(1) * torch.exp2(0.25 * e.float())
    W = (scale.unsqueeze(2) * G.view(M, K // TILE, TILE)).view(M, K)
    return fwht_inplace(W, block_size=TILE) * suh.float().unsqueeze(0)


# ------------------------------------------------------------------------------------------------
# Writing
# ------------------------------------------------------------------------------------------------

class LtqWriter:
    """Writes data straight to the file: `reserve` bytes are reserved for the header (data starts at 16 + reserve, a multiple
    of 4096); `finalize` writes the header, space-padded to the same length."""

    def __init__(self, path: Path, formats: list[LtqFormat], reserve: int = (64 << 20) - 16):
        assert (16 + reserve) % 4096 == 0
        self.path, self.reserve = Path(path), reserve
        self.formats = formats
        self.fmt_index = {f.name: i for i, f in enumerate(formats)}
        self.f = open(self.path, "wb")
        self.f.write(bytes(16 + reserve))
        self.cur = 0
        self.tensors = {}
        self.quantized = []

    def _write(self, name: str, raw: bytes, dtype: str, shape: list, align: int = 64):
        pad = _align(self.cur, align) - self.cur
        if pad:
            self.f.write(bytes(pad))
            self.cur += pad
        self.tensors[name] = {"dtype": dtype, "shape": [int(s) for s in shape], "offset": self.cur}
        self.f.write(raw)
        self.cur += len(raw)

    def add_tensor(self, name: str, t: torch.Tensor):
        t = t.detach().cpu().contiguous()
        dt = TORCH_TO_LTQ[t.dtype]
        raw = t.view(torch.int16).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes()
        self._write(name, raw, dt, list(t.shape))

    def add_matrix(self, p: str, m: dict, verify: bool = True) -> int:
        """Matrix from `liqk_v2.PACK["matrix"]`; returns the byte size of its fields. verify: the spec decode of each slice
        from the written bytes must match the quantizer's tile values bit for bit."""
        M, K = m["idx"].shape
        rows, data = [], bytearray()
        for r0, r1, sp in m["row_slices"]:
            f = self.formats[self.fmt_index[sp]]
            data += bytes(_align(len(data), 64) - len(data))
            rows.append([r0, r1, self.fmt_index[sp], len(data)])
            raw = pack_slice(m["idx"][r0:r1], m["gi"][r0:r1], f)
            if verify:
                G = m["G"][r0:r1]
                bad = int((decode_slice(raw, r1 - r0, K, f, G.device) != G).sum())
                if bad:
                    raise RuntimeError(f"{p} [{r0}:{r1}] {sp}: decode disagrees with the quantizer in {bad} weights")
            data += raw
        self.add_tensor(f"{p}.ltq.slices", torch.tensor(rows, dtype=torch.int32))
        self._write(f"{p}.ltq.data", bytes(data), "u8", [len(data)])
        self.add_tensor(f"{p}.ltq.d_row", m["d_row"].to(torch.float16))
        self.add_tensor(f"{p}.ltq.e", m["e"].to(torch.int8))
        self.add_tensor(f"{p}.ltq.suh", m["suh"].to(torch.float16))
        self.quantized.append(p)
        return len(data) + 16 * len(rows) + 2 * M + M * (K // TILE) + 2 * K

    def finalize(self, header: dict):
        header = dict(header, tensors=self.tensors)
        raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        assert len(raw) <= self.reserve, f"header of {len(raw)} B does not fit in the {self.reserve} B reserve"
        self.f.seek(0)
        self.f.write(MAGIC + struct.pack("<IQ", VERSION, self.reserve) + raw + b" " * (self.reserve - len(raw)))
        self.f.close()


def source_metadata(model_dir: Path) -> dict:
    """Source model metadata for the header: config, generation_config, tokenizer(_config), chat template, processor."""
    model_dir = Path(model_dir)
    out = {}
    for key, fn in (("config", "config.json"), ("generation_config", "generation_config.json"), ("tokenizer", "tokenizer.json"),
                    ("tokenizer_config", "tokenizer_config.json"), ("processor_config", "preprocessor_config.json"),
                    ("video_processor_config", "video_preprocessor_config.json")):
        if (model_dir / fn).exists():
            out[key] = json.loads((model_dir / fn).read_text(encoding="utf-8"))
    if (model_dir / "chat_template.jinja").exists():
        out["chat_template"] = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    elif "chat_template" in out.get("tokenizer_config", {}):
        out["chat_template"] = out["tokenizer_config"]["chat_template"]
    return out


# ------------------------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------------------------

class LtqFile:
    def __init__(self, path: Path):
        with open(path, "rb") as f:
            magic, (ver, H) = f.read(4), struct.unpack("<IQ", f.read(12))
            assert magic == MAGIC and ver == VERSION, (magic, ver)
            self.header = json.loads(f.read(H).decode("utf-8"))
        self.data_off = 16 + H
        assert self.data_off % 4096 == 0
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.formats = [format_from_json(d) for d in self.header["quantization"]["formats"]]

    def raw(self, name: str) -> memoryview:
        t = self.header["tensors"][name]
        n = int(np.prod(t["shape"])) * np.dtype(DTYPES[t["dtype"]][0]).itemsize
        o = self.data_off + t["offset"]
        return memoryview(self.mm[o : o + n])

    def tensor(self, name: str) -> torch.Tensor:
        t = self.header["tensors"][name]
        np_dt, th_dt = DTYPES[t["dtype"]]
        arr = torch.from_numpy(np.frombuffer(self.raw(name), dtype=np_dt).copy()).reshape(t["shape"])
        return arr.view(torch.bfloat16) if t["dtype"] == "bf16" else arr

    def decode_matrix(self, p: str, device="cuda") -> torch.Tensor:
        """Matrix W [M, K] float32, as reconstructed by the engine."""
        d_row, e, suh = (self.tensor(f"{p}.ltq.{k}").to(device) for k in ("d_row", "e", "suh"))
        M, K = d_row.shape[0], suh.shape[0]
        data = self.raw(f"{p}.ltq.data")
        G = torch.empty((M, K), device=device)
        for r0, r1, fi, off in self.tensor(f"{p}.ltq.slices").tolist():
            G[r0:r1] = decode_slice(data[off:], r1 - r0, K, self.formats[fi], device)
        return assemble(G, d_row, e, suh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4, help="how many matrices to decode in full")
    args = ap.parse_args()
    lf = LtqFile(args.check)
    q = lf.header["quantization"]
    print(f"{args.check}: {args.check.stat().st_size / 1e9:.2f} GB, tensors {len(lf.header['tensors'])}, formats "
          f"{len(q['formats'])}, bits/weight {q.get('bits_per_weight')}")
    names = sorted({k[: -len(".ltq.data")] for k in lf.header["tensors"] if k.endswith(".ltq.data")})
    for p in names[:: max(1, len(names) // args.n)][: args.n]:
        W = lf.decode_matrix(p)
        print(f"{p}: {tuple(W.shape)}, rms {W.square().mean().sqrt():.4g}, finite {bool(torch.isfinite(W).all())}")


if __name__ == "__main__":
    main()
