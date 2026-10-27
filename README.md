# ltq-research

The research that LTQ (*Lumen Trellis Quant*) grew out of — the weight quantization format of the [lumen-rs](https://github.com/lumen-inference/lumen-rs) engine: Python and CUDA quantizer prototypes, comparison rigs against EXL3 and GGUF, raw results. The format specification is `docs/EN/ltq-format.md` in lumen-rs; the rationale and summary tables are in `docs/EN/ltq-design.md` there.

The metric is the same everywhere: mean KL divergence from BF16 with teacher-forced tokens on two traces that take no part in calibration — English Nemotron dialogues (`nem`) and a multilingual ru / zh / de / fr / es trace (`mul`); differences are computed with a paired bootstrap over dialogues.

## How the format was derived

Constraints from the very start: decoding on GPU **and** on CPU with AVX2 (level tables ≤ 32 entries — `pshufb`), an operation count per 256 weights on par with GGUF, quantization noticeably faster than EXL3 conversion, and quality no worse than EXL3 at the same bitrate.

1. **GGUF block quants without changing the format** (`IQ2_K..IQ6_K`, `.liqk v1`). Block-sequential Cholesky error compensation (LDLQ) and scale refitting on top of stock blocks: −34…−49 % KLD while staying byte-compatible with `.gguf`. The limit is the GGUF block headers themselves.
2. **A custom `.liqk v2` container.** Vector grid banks in the sub-block header, half-bit formats, Hadamard-256 rotation, bit allocation over row slices via a DP knapsack with Fisher row weights, scale polishing: another −21…−44 %. But on dense 8B models EXL3 was still ahead by +0.23…0.35 equivalent bits, and by +0.45 on MoE.
3. **MoE: it's not the quantizer, it's the bit allocation.** Ablating expert Hessians had no effect; +0.03 bits/weight on attention (3 % of MoE parameters) cuts KLD by 36–43 %. Hence DP across the whole model instead of uniform formats.
4. **Searching for the quantization core.** E8 gives +0.06…0.10 bits only around 2 bits; lightweight headers and large bitshift-trellis tables were dropped (AVX2). The winner was **Ungerboeck TCQ**: one code bit per weight, 10-bit trellis memory, the level-subset label is the masked parity of the memory, plus a gain index per 16 weights in a joint Viterbi search: +0.23 / +0.12 / +0.09 bits over v2 on Gaussian data at K = 2 / 3 / 4 and +0.15…0.25 equivalent bits on real models.
5. **CUDA Viterbi** (`src/cuda/tcq_viterbi.cu`): 1B quantization 580 → 41 s; then radix-4, in-block traceback, gain candidates — 3–5.6x faster than EXL3 on 3B.
6. **Closing the gap with EXL3 at equal bitrate.**
   - Scale: fp16 per row and a 5-bit exponent per (row, 256-block) — a single per-row scale is not enough for down / o_proj with outlier channels.
   - A 2–4 bit gain header; its optimum grows with K.
   - **Tail-biting trellis**: −0.017…−0.020 bits on all formats with no side information.
   - **Extended labels** (8 / 16 subsets): at K = 2 the gap to R(D) goes 0.176 → 0.118 bits.
   - Hessian damping of 0.2 instead of 0.02 (the old value had been tuned for v2).
   - Rejected: 64-entry tables (−20 % CPU decode speed for noise-level gains), a header per 32 weights at K = 3 (the two tile rows are different output channels), headerless formats, output rotation, half-integer formats built from rows of different bitrates.
7. **The key finding — the mixing penalty.** EXL3 natively quantizes every tensor at exactly 2.5 / 3 / 4 bits, while DP mixed formats in one-bit steps: a 2 + 3 bit mix loses up to 1.25x KLD. The fix is **fractional rates via parity of the low non-code bits**: D2 (pairs, K − 0.5) and D4 (quads, K − 0.25), supported inside Viterbi. With them — parity with or better than EXL3 on 1B and 3B at 2.5–4.5 bits/weight.
8. **Reproducibility.** Fisher sampled tokens randomly, and DP assignments drifted by ±5 % KLD between processes; sampling via the inverse CDF with fixed numbers brought it down to 1–2 %. Without Fisher weights the same menu is twice as bad.
9. **Final TCQPQ menu** — 19 formats, K = 2…5, integer, D2 and D4, 2.04–5.29 bits/weight. Against EXL3 at the same bitrate: Qwen3-8B −10…−31 %, Qwen3-30B-A3B −52 % at 3.0 (attention gets 4.4 bits at the cost of 0.05 bits on experts), Qwen3.5-4B and Gemma 4 E4B −14…−26 % (against per-matrix EXL3, since the stock converter does not support them). Rejected after testing: K + 0.25, median-based winsorization (+38…+55 % KLD), a single-iteration DP profile.
10. **Row tables** (embeddings, n-grams, PLE) — rings of the same trellis with Hadamard rotation: Gemma 4 E4B PLE at 4.25 bits — KLD 0.0010 at 1.39 GiB versus 0.0006 for int8 at 2.65 GiB.
11. **The format is frozen** by the specification and the reference `src/ltq_format.py`: a self-test of 19 tile formats and 17 ring formats — 0 mismatches with the quantizer (`results/ltq_format_selftest.json`).
12. **Check against third-party quants** (Qwen3.8-27B, `results/q27_report.txt`): at the same bitrate of the decoder linear layers, −46…−49 % KLD vs GSQ-RCO (GGUF, IQ3_XXS / IQ3_S) and −32…−33 % vs EXL3 3.5; against self-calibrated EXL3 3.0 — within noise.

Caveat for items 9 and 12: the calibration (Qwen3.8-Flash-Next self-calibration, chat and agents) is closer in domain to the evaluation traces than the competitors' calibration; a check with neutral calibration and an external trace has not been done yet.

## What's where

| Path | Contents |
| --- | --- |
| `src/liqk_v2.py` | quantizer: formats (`TCQ_CONFIG`, `LIQK_V2_SPECS`), LDLQ, profile and DP over row slices, `.ltq` packing mode (`PACK`) |
| `src/tcq_proto.py`, `src/cuda/tcq_viterbi.cu` | Viterbi (torch reference and CUDA), table training, Gaussian rig |
| `src/ltq_format.py`, `src/ltq_file.py` | format reference (tile, decode, self-test) and the `.ltq` container (write, read, full decode) |
| `src/hybrid_quant.py`, `src/streaming_quant.py` | layer-wise quantizers and evaluators for models that don't fit on the GPU, with any layer types; comparison with off-the-shelf GGUF and EXL3 |
| `src/fisher_full.py`, `src/fisher_offload.py` | Fisher row weights |
| `src/equal_bpw.py` | equal-bitrate comparison (1B, 3B) |
| `src/cpu/`, `src/cuda/ltq_matvec_bench.cu`, `src/kbench/` | decode speed on CPU and GPU; CUDA C++ vs cuda-oxide |
| `results/` | raw results (JSON) and summaries |

## Running

Python with torch (CUDA) and transformers; the CUDA Viterbi needs `nvcc` in `/usr/local/cuda/bin` (built on first call). EXL3 comparisons need exllamav3 1.6.x (`LUMEN_EXL3_PYTHON`, `LUMEN_EXL3_CONVERT`), GGUF comparisons need a built `libggml-base.so` (`LUMEN_LIBGGML`). Paths to local models and third-party builds default to `~/lm-inference`.

`data/` is not part of the repository: traces and calibration sets are derived from external datasets and are built by `src/prepare_traces.py`; Fisher weights are recomputed by `src/fisher_full.py` / `src/fisher_offload.py`. Models and quants are local as well.

```bash
python -m src.ltq_format        # format self-test (requires a GPU)
python -m src.hybrid_quant --model_dir <bf16> --data_dir data/<model> --fisher data/<model>/fisher.pt \
    --methods dp:TCQPQ:3.5 --out results/<name>.json
```
