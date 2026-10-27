"""
Script for measuring inference speed (Prompt Processing pp512 and Token Generation tg128)
and native perplexity (llama-perplexity) for all formats under study:
  - CPU: AMD Ryzen 9 9950X (Zen 5 AVX-512 VNNI/VBMI) at 1, 8 and 16 threads (-ngl 0)
  - GPU: NVIDIA GeForce RTX 5070 Ti 16 GB (-ngl 99)
  - Comparison of native .gguf files: stock llama-quantize (PR #30718 + imatrix)
    vs. our patched Llama-3.2-1B-LDLQ-IQ{2,3,4}_K.gguf in llama-perplexity!
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def run_llama_bench(
    bench_bin: Path,
    models: List[tuple[str, Path]],
    ngl: int,
    threads: str = "16",
    repetitions: int = 3,
) -> List[dict]:
    """Runs llama-bench and returns the parsed JSON list of results."""
    cmd = [
        str(bench_bin),
        "-ngl", str(ngl),
        "-t", threads,
        "-p", "512",
        "-n", "128",
        "-r", str(repetitions),
        "-o", "json",
    ]
    for _, mpath in models:
        cmd.extend(["-m", str(mpath)])

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # llama-bench prints a JSON array to stdout
    stdout = res.stdout.strip()
    start_idx = stdout.find("[")
    if start_idx == -1:
        raise RuntimeError(f"Could not find JSON in llama-bench output:\n{stdout[:500]}")
    data = json.loads(stdout[start_idx:])

    # Map model paths to readable labels
    path_to_label = {str(p): lbl for lbl, p in models}
    simplified = []
    for row in data:
        mfile = row.get("model_filename", "")
        label = path_to_label.get(mfile, Path(mfile).stem)
        n_prompt = row.get("n_prompt", 0)
        n_gen = row.get("n_gen", 0)
        mode = f"pp{n_prompt}" if n_prompt > 0 else f"tg{n_gen}"
        simplified.append(
            {
                "label": label,
                "mode": mode,
                "n_threads": row.get("n_threads"),
                "n_gpu_layers": row.get("n_gpu_layers"),
                "avg_ts": row.get("avg_ts"),
                "stddev_ts": row.get("stddev_ts"),
                "model_size_bytes": row.get("model_size"),
                "model_n_params": row.get("model_n_params"),
            }
        )
    return simplified


def run_llama_perplexity(
    ppl_bin: Path,
    model_path: Path,
    calib_txt: Path,
    chunks: int = 64,
) -> dict:
    """Runs native llama-perplexity on GPU (-ngl 99) and extracts the final PPL."""
    cmd = [
        str(ppl_bin),
        "-m", str(model_path),
        "-f", str(calib_txt),
        "-ngl", "99",
        "-c", "512",
        "-b", "512",
        "--chunks", str(chunks),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    elapsed = time.perf_counter() - t0
    combined = res.stdout + "\n" + res.stderr
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", combined)
    if not m:
        raise RuntimeError(f"Could not extract PPL from llama-perplexity output:\n{combined[-1000:]}")
    return {
        "ppl": float(m.group(1)),
        "ppl_stderr": float(m.group(2)),
        "sec": elapsed,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    bin_dir = root / "llama-iqk" / "build" / "bin"
    bench_bin = bin_dir / "llama-bench"
    ppl_bin = bin_dir / "llama-perplexity"
    gguf_dir = root / "gguf_1b"
    calib_txt = root / "data" / "calib_imatrix.txt"
    out_json = root / "results" / "speed_and_ppl_1b.json"

    models_bench = [
        ("BF16", gguf_dir / "Llama-3.2-1B-Instruct-bf16.gguf"),
        ("IQ2_K", gguf_dir / "Llama-3.2-1B-IQ2_K-imatrix.gguf"),
        ("IQ3_K", gguf_dir / "Llama-3.2-1B-IQ3_K-imatrix.gguf"),
        ("Q3_K_M", gguf_dir / "Llama-3.2-1B-Q3_K_M-imatrix.gguf"),
        ("IQ4_XS", gguf_dir / "Llama-3.2-1B-IQ4_XS-imatrix.gguf"),
        ("IQ4_K", gguf_dir / "Llama-3.2-1B-IQ4_K-imatrix.gguf"),
        ("Q4_K_M", gguf_dir / "Llama-3.2-1B-Q4_K_M-imatrix.gguf"),
        ("IQ5_K", gguf_dir / "Llama-3.2-1B-IQ5_K-imatrix.gguf"),
        ("IQ6_K", gguf_dir / "Llama-3.2-1B-IQ6_K-imatrix.gguf"),
    ]
    if (gguf_dir / "Llama-3.2-1B-LDLQ-IQ3_K.gguf").exists():
        models_bench.append(("LDLQ-IQ3_K", gguf_dir / "Llama-3.2-1B-LDLQ-IQ3_K.gguf"))

    print("1. Measuring GPU speed (RTX 5070 Ti, -ngl 99)...", flush=True)
    gpu_bench = run_llama_bench(bench_bin, models_bench, ngl=99, threads="16", repetitions=5)
    for r in gpu_bench:
        print(f"  [GPU] {r['label']:12s} | {r['mode']:6s} | {r['avg_ts']:8.2f} +- {r['stddev_ts']:5.2f} tok/s", flush=True)

    print("2. Measuring CPU speed (Ryzen 9 9950X, -ngl 0, threads=8,16)...", flush=True)
    cpu_bench = run_llama_bench(bench_bin, models_bench, ngl=0, threads="8,16", repetitions=3)
    for r in cpu_bench:
        print(
            f"  [CPU t={r['n_threads']:2d}] {r['label']:12s} | {r['mode']:6s} | {r['avg_ts']:8.2f} +- {r['stddev_ts']:5.2f} tok/s",
            flush=True,
        )

    print("3. Native llama-perplexity check on stock GGUFs vs. our LDLQ-IQ*_K.gguf...", flush=True)
    ppl_models = [
        ("BF16", gguf_dir / "Llama-3.2-1B-Instruct-bf16.gguf"),
        ("Stock-IQ2_K-imatrix", gguf_dir / "Llama-3.2-1B-IQ2_K-imatrix.gguf"),
        ("LDLQ-IQ2_K-gguf", gguf_dir / "Llama-3.2-1B-LDLQ-IQ2_K.gguf"),
        ("Stock-IQ3_K-imatrix", gguf_dir / "Llama-3.2-1B-IQ3_K-imatrix.gguf"),
        ("LDLQ-IQ3_K-gguf", gguf_dir / "Llama-3.2-1B-LDLQ-IQ3_K.gguf"),
        ("Stock-Q3_K_M-imatrix", gguf_dir / "Llama-3.2-1B-Q3_K_M-imatrix.gguf"),
        ("Stock-IQ4_XS-imatrix", gguf_dir / "Llama-3.2-1B-IQ4_XS-imatrix.gguf"),
        ("Stock-IQ4_K-imatrix", gguf_dir / "Llama-3.2-1B-IQ4_K-imatrix.gguf"),
        ("LDLQ-IQ4_K-gguf", gguf_dir / "Llama-3.2-1B-LDLQ-IQ4_K.gguf"),
        ("Stock-Q4_K_M-imatrix", gguf_dir / "Llama-3.2-1B-Q4_K_M-imatrix.gguf"),
    ]
    ppl_results: Dict[str, dict] = {}
    for lbl, mp in ppl_models:
        if not mp.exists():
            continue
        pres = run_llama_perplexity(ppl_bin, mp, calib_txt, chunks=64)
        ppl_results[lbl] = pres
        print(f"  [PPL] {lbl:22s}: {pres['ppl']:.4f} +/- {pres['ppl_stderr']:.4f} ({pres['sec']:.1f}s)", flush=True)

    out_data = {
        "gpu_bench": gpu_bench,
        "cpu_bench": cpu_bench,
        "native_gguf_ppl": ppl_results,
    }
    out_json.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"Saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()

