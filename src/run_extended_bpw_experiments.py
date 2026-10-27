"""
Full end-to-end quantization run over the 2.00 to 8.50 bpw range
on Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct for all families:
  1. Stock GGUF + imatrix:
     IQ2_XXS (2.06 bpw), IQ2_K (2.38 bpw), IQ3_XXS (3.06 bpw), IQ3_K (3.44 bpw),
     Q3_K_M (3.84/3.85 bpw), IQ4_XS (4.27/4.25 bpw), IQ4_K (4.50 bpw), Q4_K_M (4.80/4.79 bpw),
     IQ5_K (5.50 bpw), IQ6_K (6.625 bpw), Q8_0 (8.50 bpw).
  2. LDLQ-IQ*_K (no Hadamard, 100% .gguf-compatible) and .liqk v1 (LIQK-IQ*_K / DP-LIQK):
     2.38, 3.44, 3.84/3.85, 4.27/4.25, 4.50, 5.50, 6.625 bpw.
  3. Our .liqk v2 Uniform (LIQK2_00_V2, LIQK2_V2, LIQK3_V2, LIQK4_V2, LIQK5_V2, LIQK6_V2, LIQK8_00_V2)
     and .liqk v2 Mixed (2.06, 2.38, 2.90, 3.44, 3.84/3.85, 4.27/4.25, 4.50, 5.00, 5.50, 6.625, 8.00 bpw)
     with a single pure-KLD scale polish (ce_weight = 0.0) across the whole 2..8 bit range.
  4. EXL3 (mul1 codebook):
     2.06, 2.38, 3.44, 3.84/3.85, 4.27/4.25, 4.50, 5.50, 6.625, 8.00 bpw.
"""

from __future__ import annotations

import os

import json
import shutil
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from src.eval_kld import (
    evaluate_model_on_cached_trace,
    load_eval_trace,
    precompute_ref_trace_data,
)
from src.gguf_utils import load_gguf_into_hf_model
from src.liqk_v2 import (
    build_liqk_v2_fisher_slice_profile,
    compute_exact_row_fisher_sensitivities,
    polish_model_scales_kld,
    precompute_ref_final_hidden_states,
    quantize_llama_model_liqk_v2,
    solve_fisher_slice_dp,
)
from src.quantize_model_iqk import quantize_llama_model_sequential
from src.run_1b_exl3_experiments import load_exl3_into_hf_model
from src.run_1b_iqk_experiments import (
    restore_clean_linear_weights,
    save_clean_linear_weights,
)
from src.run_1b_liqk_v2_experiments import (
    restore_clean_model_weights,
    save_clean_model_weights,
    summarize_slice_assignment,
)


def run_exl3_convert_and_eval(
    model: torch.nn.Module,
    model_dir: Path,
    cal_path: Path,
    nem_ref: list,
    mul_ref: list,
    bpw: float,
    cb_flag: str,
    device: torch.device,
    tmp_root: Path,
) -> dict:
    python_bin = os.path.expanduser("~/lm-inference/ablation-sigmoid/venv/bin/python")
    convert_script = os.path.expanduser("~/lm-inference/ablation-sigmoid/exl3/convert.py")
    work_dir = tmp_root / f"work_{bpw:.3f}_{cb_flag}"
    out_dir = tmp_root / f"out_{bpw:.3f}_{cb_flag}"
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin,
        convert_script,
        "-i", str(model_dir),
        "-o", str(out_dir),
        "-w", str(work_dir),
        "-b", str(bpw),
        "-hb", "16",
        "-eb", "16",
        "-cd", str(cal_path),
        "-cr", "64",
        "-cc", "2048",
        "-cb", cb_flag,
    ]

    model.cpu()
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    quant_sec = time.perf_counter() - t0
    model.to(device)

    q_info = load_exl3_into_hf_model(model, out_dir, device)
    q_info["quant_sec"] = quant_sec
    q_info["target_bpw"] = bpw
    q_info["codebook"] = cb_flag

    res_nem = evaluate_model_on_cached_trace(model, nem_ref, device)
    res_mul = evaluate_model_on_cached_trace(model, mul_ref, device)
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    return {"info": q_info, "nemotron": res_nem, "multilingual": res_mul}


def ensure_and_eval_ggufs(
    model: torch.nn.Module,
    model_dir: Path,
    gguf_dir: Path,
    qtypes: list[str],
    res_dict: dict,
    res_json: Path,
    key_fmt: str,
    nem_ref: list,
    mul_ref: list,
    device: torch.device,
    root: Path,
):
    missing = [q for q in qtypes if key_fmt.format(q=q) not in res_dict]
    if not missing:
        return

    python_bin = os.path.expanduser("~/lm-inference/ablation-sigmoid/venv/bin/python")
    conv_script = root / "llama-iqk" / "convert_hf_to_gguf.py"
    q_bin = root / "llama-iqk" / "build" / "bin" / "llama-quantize"
    bf16_gguf = gguf_dir / f"{model_dir.name}-bf16.gguf"
    imat_file = gguf_dir / "imatrix.dat"

    need_build = [q for q in missing if not (gguf_dir / f"{model_dir.name}-{q}-imatrix.gguf").exists()]
    if need_build:
        if not bf16_gguf.exists():
            print(f"Converting {model_dir.name} to BF16 GGUF...", flush=True)
            subprocess.run([python_bin, str(conv_script), str(model_dir), "--outfile", str(bf16_gguf), "--outtype", "bf16"], check=True)
        for qtype in need_build:
            out_q = gguf_dir / f"{model_dir.name}-{qtype}-imatrix.gguf"
            print(f"Quantizing stock {model_dir.name} {qtype}...", flush=True)
            if qtype == "Q8_0":
                subprocess.run([str(q_bin), str(bf16_gguf), str(out_q), qtype, "16"], check=True)
            else:
                subprocess.run([str(q_bin), "--imatrix", str(imat_file), str(bf16_gguf), str(out_q), qtype, "16"], check=True)
        if model_dir.name.endswith("3B-Instruct"):
            bf16_gguf.unlink(missing_ok=True)

    for qtype in missing:
        lbl = key_fmt.format(q=qtype)
        gpath = gguf_dir / f"{model_dir.name}-{qtype}-imatrix.gguf"
        if not gpath.exists():
            # Check a possible alternative name for 1B
            alt = gguf_dir / f"Llama-3.2-1B-{qtype}-imatrix.gguf"
            if alt.exists():
                gpath = alt
        info = load_gguf_into_hf_model(model, gpath, include_embeddings=False)
        info_clean = {k: v for k, v in info.items() if k != "packed_gguf"}
        res_nem = evaluate_model_on_cached_trace(model, nem_ref, device)
        res_mul = evaluate_model_on_cached_trace(model, mul_ref, device)
        res_dict[lbl] = {"info": info_clean, "nemotron": res_nem, "multilingual": res_mul}
        res_json.write_text(json.dumps(res_dict, indent=2), encoding="utf-8")
        print(
            f"[{model_dir.name[:12]} {lbl:24s}] bpw={info_clean['linear_bpw']:.4f} | "
            f"Nem KLD={res_nem['mean_kld']:.5f} PPL={res_nem['ppl_quant']:.4f} | "
            f"Mul KLD={res_mul['mean_kld']:.5f} PPL={res_mul['ppl_quant']:.4f}",
            flush=True,
        )
        if qtype in ("IQ2_XXS", "IQ3_XXS", "Q8_0") and gpath.exists():
            gpath.unlink(missing_ok=True)
        elif model_dir.name.endswith("3B-Instruct") and gpath.exists():
            gpath.unlink(missing_ok=True)


def run_1b_full_2_to_8(root: Path, device: torch.device):
    print("\n================ 1B FULL BENCHMARK (2.00..8.50 BPW) ================", flush=True)
    model_dir = root / "models" / "Llama-3.2-1B-Instruct"
    gguf_dir = root / "gguf_1b"
    cal_path = root / "data" / "cal_trace.safetensors"
    cal_tokens = load_file(str(cal_path))["input_ids"][:64]
    cal_tokens_fast = cal_tokens[:32]

    gguf_json = root / "results" / "gguf_stock_1b.json"
    iqk_json = root / "results" / "iqk_liqk_1b.json"
    v2_json = root / "results" / "liqk_v2_1b.json"
    exl3_json = root / "results" / "exl3_1b.json"

    gguf_res = json.loads(gguf_json.read_text(encoding="utf-8"))
    iqk_res = json.loads(iqk_json.read_text(encoding="utf-8"))
    v2_res = json.loads(v2_json.read_text(encoding="utf-8"))
    exl3_res = json.loads(exl3_json.read_text(encoding="utf-8"))

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device)
    model.eval()
    clean_weights = save_clean_model_weights(model)

    nem_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "nemotron_trace.json"), device)
    mul_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "multilingual_trace.json"), device)

    # 1. Stock GGUFs on 1B: adding IQ2_XXS (2.0625 bpw), IQ3_XXS (3.0625 bpw), Q8_0 (8.5000 bpw)
    ensure_and_eval_ggufs(
        model,
        model_dir,
        gguf_dir,
        ["IQ2_XXS", "IQ2_K", "IQ3_XXS", "IQ3_K", "Q3_K_M", "IQ4_XS", "IQ4_K", "Q4_K_M", "IQ5_K", "IQ6_K", "Q8_0"],
        gguf_res,
        gguf_json,
        "GGUF-{q}-imatrix",
        nem_ref,
        mul_ref,
        device,
        root,
    )

    # 2. Full recomputation of .liqk v2 on 1B (2.00 .. 8.00 bpw) with a single pure-KLD polish (ce_weight=0.0)
    restore_clean_model_weights(model, clean_weights)
    ref_hidden_cpu = precompute_ref_final_hidden_states(model, cal_tokens, n_seqs=64, seq_len=512, batch_size=4)
    fisher_out = compute_exact_row_fisher_sensitivities(model, cal_tokens, n_Menu_seqs=8, seq_len=1024)
    t_dp0 = time.perf_counter()
    slice_profile = build_liqk_v2_fisher_slice_profile(model, cal_tokens, batch_size=8, fisher_out=fisher_out)
    mlp_perms = slice_profile["mlp_perms"]
    fisher_out_perm = slice_profile["fisher_out"]
    dp_profile_sec = time.perf_counter() - t_dp0

    def record_v2_1b(label: str, q_info: dict, extra_meta: dict | None = None):
        torch.cuda.empty_cache()
        res_nem = evaluate_model_on_cached_trace(model, nem_ref, device)
        res_mul = evaluate_model_on_cached_trace(model, mul_ref, device)
        info_clean = dict(q_info)
        if extra_meta:
            info_clean.update(extra_meta)
        info_clean["ce_weight_clean"] = 0.0
        v2_res[label] = {"info": info_clean, "nemotron": res_nem, "multilingual": res_mul}
        v2_json.write_text(json.dumps(v2_res, indent=2), encoding="utf-8")
        print(
            f"[1B {label:24s}] bpw={info_clean['linear_bpw']:.4f} | time={info_clean['quant_sec']:.1f}s | "
            f"Nem KLD={res_nem['mean_kld']:.5f} PPL={res_nem['ppl_quant']:.4f} | "
            f"Mul KLD={res_mul['mean_kld']:.5f} PPL={res_mul['ppl_quant']:.4f}",
            flush=True,
        )

    uniform_specs_1b = [
        ("LIQK2_00_V2", "LIQK2_00_V2", 5e-4),
        ("LIQK2_V2",    "LIQK2_V2",    5e-4),
        ("LIQK3_V2",    "LIQK3_V2",    3e-4),
        ("LIQK4_V2",    "LIQK4_V2",    2e-4),
        ("LIQK5_V2",    "LIQK5_V2",    1.5e-4),
        ("LIQK6_V2",    "LIQK6_V2",    1.0e-4),
        ("LIQK8_00_V2", "LIQK8_00_V2", 5e-5),
    ]
    for base_label, spec_name, pol_lr in uniform_specs_1b:
        lbl_nopol = f"{base_label}-NoPolish"
        lbl_pol = f"{base_label}-Uniform"
        if lbl_pol in v2_res and v2_res[lbl_pol].get("info", {}).get("ce_weight_clean") == 0.0:
            continue
        restore_clean_model_weights(model, clean_weights)
        q_info = quantize_llama_model_liqk_v2(
            model, cal_tokens_fast, spec_name, mlp_perms=mlp_perms, fisher_out=fisher_out_perm, batch_size=8, cd_sweeps=2, polish_steps=0
        )
        record_v2_1b(lbl_nopol, q_info, {"spec": spec_name, "mode": "liqk_v2_uniform_nopolish"})
        with torch.enable_grad():
            pol_sec = polish_model_scales_kld(
                model, cal_tokens, ref_hidden_cpu, n_steps=80, lr=pol_lr, batch_size=2, reg_wd=0.02, ce_weight=0.0
            )
        q_info_pol = dict(q_info, polish_sec=pol_sec, quant_sec=q_info["quant_sec"] + pol_sec)
        record_v2_1b(lbl_pol, q_info_pol, {"spec": spec_name, "mode": "liqk_v2_uniform_polished"})

    mixed_targets_1b = [
        ("LIQK_V2-2.06bpw", 2.0625, 5e-4),
        ("LIQK_V2-2.38bpw", 2.3750, 5e-4),
        ("LIQK_V2-2.90bpw", 2.9000, 4e-4),
        ("LIQK_V2-3.44bpw", 3.4375, 3e-4),
        ("LIQK_V2-3.84bpw", 3.8416, 2.5e-4),
        ("LIQK_V2-4.27bpw", 4.2716, 2e-4),
        ("LIQK_V2-4.50bpw", 4.5000, 2e-4),
        ("LIQK_V2-5.00bpw", 5.0000, 1.5e-4),
        ("LIQK_V2-5.50bpw", 5.5000, 1.5e-4),
        ("LIQK_V2-6.62bpw", 6.6250, 1.0e-4),
        ("LIQK_V2-8.00bpw", 8.0000, 5e-5),
    ]
    for base_label, target_bpw, pol_lr in mixed_targets_1b:
        lbl_nopol = f"{base_label}-NoPolish"
        lbl_pol = base_label
        if lbl_pol in v2_res and v2_res[lbl_pol].get("info", {}).get("ce_weight_clean") == 0.0:
            continue
        restore_clean_model_weights(model, clean_weights)
        assignment = solve_fisher_slice_dp(slice_profile, target_bpw=target_bpw)
        summary = summarize_slice_assignment(assignment, model)
        print(f"  -> 1B {base_label} (target={target_bpw:.4f}): shares={summary['param_shares_pct']}", flush=True)
        q_info = quantize_llama_model_liqk_v2(
            model, cal_tokens_fast, assignment, mlp_perms=mlp_perms, fisher_out=fisher_out_perm, batch_size=8, cd_sweeps=2, polish_steps=0
        )
        record_v2_1b(
            lbl_nopol,
            q_info,
            {"target_bpw": target_bpw, "mode": "liqk_v2_mixed_nopolish", "param_shares_pct": summary["param_shares_pct"], "dp_profile_sec": dp_profile_sec},
        )
        with torch.enable_grad():
            pol_sec = polish_model_scales_kld(
                model, cal_tokens, ref_hidden_cpu, n_steps=80, lr=pol_lr, batch_size=2, reg_wd=0.02, ce_weight=0.0
            )
        q_info_pol = dict(q_info, polish_sec=pol_sec, quant_sec=q_info["quant_sec"] + pol_sec)
        record_v2_1b(
            lbl_pol,
            q_info_pol,
            {"target_bpw": target_bpw, "mode": "liqk_v2_mixed_polished", "param_shares_pct": summary["param_shares_pct"], "dp_profile_sec": dp_profile_sec},
        )

    # 3. EXL3 on 1B: adding 2.06 bpw, 6.625 bpw, 8.00 bpw
    for lbl, bpw in [
        ("EXL3-mul1-2.06bpw", 2.0625),
        ("EXL3-mul1-6.62bpw", 6.6250),
        ("EXL3-mul1-8.00bpw", 8.0000),
    ]:
        if lbl in exl3_res:
            continue
        restore_clean_model_weights(model, clean_weights)
        exl3_res[lbl] = run_exl3_convert_and_eval(
            model, model_dir, cal_path, nem_ref, mul_ref, bpw, "mul1", device, root / "exl3_tmp_1b"
        )
        exl3_json.write_text(json.dumps(exl3_res, indent=2), encoding="utf-8")
        r = exl3_res[lbl]
        print(
            f"[1B {lbl:24s}] bpw={r['info']['linear_bpw']:.4f} | "
            f"Nem KLD={r['nemotron']['mean_kld']:.5f} PPL={r['nemotron']['ppl_quant']:.4f} | "
            f"Mul KLD={r['multilingual']['mean_kld']:.5f} PPL={r['multilingual']['ppl_quant']:.4f}",
            flush=True,
        )

    del model, nem_ref, mul_ref, clean_weights, ref_hidden_cpu
    torch.cuda.empty_cache()


def run_3b_full_2_to_8(root: Path, device: torch.device):
    print("\n================ 3B FULL BENCHMARK (2.00..8.50 BPW) ================", flush=True)
    model_dir = root / "models" / "Llama-3.2-3B-Instruct"
    gguf_dir = root / "gguf_3b"
    cal_path = root / "data" / "cal_trace.safetensors"
    cal_tokens = load_file(str(cal_path))["input_ids"][:64]
    cal_tokens_fast = cal_tokens[:32]

    res3b_json = root / "results" / "all_3b_results.json"
    v2_3b_json = root / "results" / "liqk_v2_3b.json"

    res_3b = json.loads(res3b_json.read_text(encoding="utf-8"))
    v2_3b = json.loads(v2_3b_json.read_text(encoding="utf-8"))

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device)
    model.eval()
    clean_weights = save_clean_model_weights(model)
    clean_lin = save_clean_linear_weights(model)

    nem_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "nemotron_trace.json"), device)
    mul_ref = precompute_ref_trace_data(model, load_eval_trace(root / "data" / "multilingual_trace.json"), device)

    # 1. Stock GGUFs on 3B: adding IQ2_XXS (2.0625 bpw), IQ3_XXS (3.0625 bpw), Q8_0 (8.5000 bpw)
    ensure_and_eval_ggufs(
        model,
        model_dir,
        gguf_dir,
        ["IQ2_XXS", "IQ2_K", "IQ3_XXS", "IQ3_K", "Q3_K_M", "IQ4_XS", "IQ4_K", "Q4_K_M", "IQ5_K", "IQ6_K", "Q8_0"],
        res_3b,
        res3b_json,
        "GGUF-{q}-imatrix",
        nem_ref,
        mul_ref,
        device,
        root,
    )

    # 2. LDLQ and .liqk v1 on 3B: adding IQ6_K (6.625 bpw)
    for lbl, sp, mode in [
        ("LDLQ-IQ6_K", "IQ6_K", "ldlq"),
        ("LIQK-IQ6_K", "IQ6_K", "liqk"),
    ]:
        if lbl in res_3b:
            continue
        restore_clean_linear_weights(model, clean_lin)
        q_info = quantize_llama_model_sequential(model, cal_tokens_fast, sp, mode=mode, batch_size=8, export_gguf_blocks=False)
        info_clean = {k: v for k, v in q_info.items() if k != "packed_gguf"}
        res_nem = evaluate_model_on_cached_trace(model, nem_ref, device)
        res_mul = evaluate_model_on_cached_trace(model, mul_ref, device)
        res_3b[lbl] = {"info": info_clean, "nemotron": res_nem, "multilingual": res_mul}
        res3b_json.write_text(json.dumps(res_3b, indent=2), encoding="utf-8")
        print(
            f"[3B {lbl:24s}] bpw={info_clean['linear_bpw']:.4f} | "
            f"Nem KLD={res_nem['mean_kld']:.5f} PPL={res_nem['ppl_quant']:.4f} | "
            f"Mul KLD={res_mul['mean_kld']:.5f} PPL={res_mul['ppl_quant']:.4f}",
            flush=True,
        )

    # 3. Full recomputation of .liqk v2 on 3B (2.00 .. 8.00 bpw) with a single pure-KLD polish (ce_weight=0.0)
    restore_clean_model_weights(model, clean_weights)
    ref_hidden_cpu = precompute_ref_final_hidden_states(model, cal_tokens, n_seqs=64, seq_len=512, batch_size=2)
    fisher_out = compute_exact_row_fisher_sensitivities(model, cal_tokens, n_Menu_seqs=8, seq_len=1024)
    t_dp0 = time.perf_counter()
    slice_profile = build_liqk_v2_fisher_slice_profile(model, cal_tokens, batch_size=8, fisher_out=fisher_out)
    mlp_perms = slice_profile["mlp_perms"]
    fisher_out_perm = slice_profile["fisher_out"]
    dp_profile_sec = time.perf_counter() - t_dp0

    def record_v2_3b(label: str, q_info: dict, extra_meta: dict | None = None):
        torch.cuda.empty_cache()
        res_nem = evaluate_model_on_cached_trace(model, nem_ref, device)
        res_mul = evaluate_model_on_cached_trace(model, mul_ref, device)
        info_clean = dict(q_info)
        if extra_meta:
            info_clean.update(extra_meta)
        info_clean["ce_weight_clean"] = 0.0
        v2_3b[label] = {"info": info_clean, "nemotron": res_nem, "multilingual": res_mul}
        v2_3b_json.write_text(json.dumps(v2_3b, indent=2), encoding="utf-8")
        print(
            f"[3B {label:24s}] bpw={info_clean['linear_bpw']:.4f} | time={info_clean['quant_sec']:.1f}s | "
            f"Nem KLD={res_nem['mean_kld']:.5f} PPL={res_nem['ppl_quant']:.4f} | "
            f"Mul KLD={res_mul['mean_kld']:.5f} PPL={res_mul['ppl_quant']:.4f}",
            flush=True,
        )

    uniform_specs_3b = [
        ("LIQK2_00_V2", "LIQK2_00_V2", 5e-4),
        ("LIQK2_V2",    "LIQK2_V2",    5e-4),
        ("LIQK3_V2",    "LIQK3_V2",    3e-4),
        ("LIQK4_V2",    "LIQK4_V2",    2e-4),
        ("LIQK5_V2",    "LIQK5_V2",    1.5e-4),
        ("LIQK6_V2",    "LIQK6_V2",    1.0e-4),
        ("LIQK8_00_V2", "LIQK8_00_V2", 5e-5),
    ]
    for base_label, spec_name, pol_lr in uniform_specs_3b:
        lbl_nopol = f"{base_label}-NoPolish"
        lbl_pol = f"{base_label}-Uniform"
        if lbl_pol in v2_3b and v2_3b[lbl_pol].get("info", {}).get("ce_weight_clean") == 0.0:
            continue
        restore_clean_model_weights(model, clean_weights)
        q_info = quantize_llama_model_liqk_v2(
            model, cal_tokens_fast, spec_name, mlp_perms=mlp_perms, fisher_out=fisher_out_perm, batch_size=8, cd_sweeps=2, polish_steps=0
        )
        record_v2_3b(lbl_nopol, q_info, {"spec": spec_name, "mode": "liqk_v2_uniform_nopolish"})
        with torch.enable_grad():
            pol_sec = polish_model_scales_kld(
                model, cal_tokens, ref_hidden_cpu, n_steps=80, lr=pol_lr, batch_size=2, reg_wd=0.02, ce_weight=0.0
            )
        q_info_pol = dict(q_info, polish_sec=pol_sec, quant_sec=q_info["quant_sec"] + pol_sec)
        record_v2_3b(lbl_pol, q_info_pol, {"spec": spec_name, "mode": "liqk_v2_uniform_polished"})

    mixed_targets_3b = [
        ("LIQK_V2-2.06bpw", 2.0625, 5e-4),
        ("LIQK_V2-2.38bpw", 2.3750, 5e-4),
        ("LIQK_V2-2.90bpw", 2.9000, 4e-4),
        ("LIQK_V2-3.44bpw", 3.4375, 3e-4),
        ("LIQK_V2-3.85bpw", 3.8471, 2.5e-4),
        ("LIQK_V2-4.25bpw", 4.2500, 2e-4),
        ("LIQK_V2-4.50bpw", 4.5000, 2e-4),
        ("LIQK_V2-5.00bpw", 5.0000, 1.5e-4),
        ("LIQK_V2-5.50bpw", 5.5000, 1.5e-4),
        ("LIQK_V2-6.62bpw", 6.6250, 1.0e-4),
        ("LIQK_V2-8.00bpw", 8.0000, 5e-5),
    ]
    for base_label, target_bpw, pol_lr in mixed_targets_3b:
        lbl_nopol = f"{base_label}-NoPolish"
        lbl_pol = base_label
        if lbl_pol in v2_3b and v2_3b[lbl_pol].get("info", {}).get("ce_weight_clean") == 0.0:
            continue
        restore_clean_model_weights(model, clean_weights)
        assignment = solve_fisher_slice_dp(slice_profile, target_bpw=target_bpw)
        summary = summarize_slice_assignment(assignment, model)
        print(f"  -> 3B {base_label} (target={target_bpw:.4f}): shares={summary['param_shares_pct']}", flush=True)
        q_info = quantize_llama_model_liqk_v2(
            model, cal_tokens_fast, assignment, mlp_perms=mlp_perms, fisher_out=fisher_out_perm, batch_size=8, cd_sweeps=2, polish_steps=0
        )
        record_v2_3b(
            lbl_nopol,
            q_info,
            {"target_bpw": target_bpw, "mode": "liqk_v2_mixed_nopolish", "param_shares_pct": summary["param_shares_pct"], "dp_profile_sec": dp_profile_sec},
        )
        with torch.enable_grad():
            pol_sec = polish_model_scales_kld(
                model, cal_tokens, ref_hidden_cpu, n_steps=80, lr=pol_lr, batch_size=2, reg_wd=0.02, ce_weight=0.0
            )
        q_info_pol = dict(q_info, polish_sec=pol_sec, quant_sec=q_info["quant_sec"] + pol_sec)
        record_v2_3b(
            lbl_pol,
            q_info_pol,
            {"target_bpw": target_bpw, "mode": "liqk_v2_mixed_polished", "param_shares_pct": summary["param_shares_pct"], "dp_profile_sec": dp_profile_sec},
        )

    # 4. EXL3 on 3B: adding 2.06, 3.85, 4.25, 6.625, 8.00 bpw
    for lbl, bpw in [
        ("EXL3-mul1-2.06bpw", 2.0625),
        ("EXL3-mul1-3.85bpw", 3.8500),
        ("EXL3-mul1-4.25bpw", 4.2500),
        ("EXL3-mul1-6.62bpw", 6.6250),
        ("EXL3-mul1-8.00bpw", 8.0000),
    ]:
        if lbl in res_3b:
            continue
        restore_clean_model_weights(model, clean_weights)
        res_3b[lbl] = run_exl3_convert_and_eval(
            model, model_dir, cal_path, nem_ref, mul_ref, bpw, "mul1", device, root / "exl3_tmp_3b"
        )
        res3b_json.write_text(json.dumps(res_3b, indent=2), encoding="utf-8")
        r = res_3b[lbl]
        print(
            f"[3B {lbl:24s}] bpw={r['info']['linear_bpw']:.4f} | "
            f"Nem KLD={r['nemotron']['mean_kld']:.5f} PPL={r['nemotron']['ppl_quant']:.4f} | "
            f"Mul KLD={r['multilingual']['mean_kld']:.5f} PPL={r['multilingual']['ppl_quant']:.4f}",
            flush=True,
        )


def main():
    root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda")
    run_1b_full_2_to_8(root, device)
    run_3b_full_2_to_8(root, device)


if __name__ == "__main__":
    main()
