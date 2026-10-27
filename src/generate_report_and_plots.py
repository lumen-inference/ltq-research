"""
Generates the final plots: Pareto fronts (KLD vs BPW and PPL vs BPW over the full 2.38..6.625 bpw range),
the ablation waterfall, speed plots (CPU Ryzen 9 9950X / GPU RTX 5070 Ti)
and reproducibility audit plots (`results/audit_{1b,3b}.json`, see `src/audit_reproducibility.py`).

The `.liqk v2` points on the Pareto plots come from `results/liqk_v2_{1b,3b}.json` (fast settings of
`run_extended_bpw_experiments.py`, polish without the CE term). Hollow stars on top of them are audit points
with the original `run_*_liqk_v2_experiments.py` settings, also without the CE term.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Dark theme for all plots
plt.style.use("dark_background")
plt.rcParams.update({"figure.facecolor": "#0d1117", "savefig.facecolor": "#0d1117", "axes.facecolor": "#161b22",
                     "axes.edgecolor": "#8b949e", "grid.color": "#8b949e", "legend.facecolor": "#161b22",
                     "legend.edgecolor": "#30363d"})


def _linear_decimal_y(ax):
    """Linear Y axis starting at zero, with plain decimal tick labels (0.005, 0.01, ...) instead of 10^k."""
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.4f}".rstrip("0").rstrip(".")))
import numpy as np

from src.eval_kld import paired_bootstrap_ppl_diff

# Polish configurations from the audit: (label suffix, caption, color, hatch)
AUDIT_CONFIGS = [
    ("ext", "v2: JSON settings\n(cal 32, no CE)", "#82e0aa", ""),
    ("orig-ce0", "v2: original settings\n(cal 64, no CE)", "#27ae60", ""),
    ("orig-ce0.08", "v2: original settings\n+ CE 0.08 (as in README)", "#f4d03f", "//"),
]
AUDIT_DATASETS = [("nemotron", "Nemotron"), ("multilingual", "Multilingual"), ("wikitext2", "WikiText-2 (neutral)")]


def load_json_safe(p: Path) -> dict:
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


LTQ_LABEL = "LTQ: TCQ + fractional rates + Fisher-Slice DP (final TCQPQ menu, 2.5..4.5 bpw)"
LTQ_STYLE, LTQ_COLOR = "*-", "#f7dc6f"
EQ_TAG = "-r2"  # run of the final menu with recalibrated DP coefficients (results/equal_bpw_*.json)
EQ_BPW = ["2.5000", "3.0000", "3.4375", "4.0000", "4.5000"]


def eq_runs(path: Path) -> dict:
    """Equal-bitrate runs (src/equal_bpw.py), prefixed with "EQ:" to avoid clashing with keys from other files."""
    return {"EQ:" + k: v for k, v in load_json_safe(path).get("runs", {}).items()}


def ltq_series(eq: dict):
    return (LTQ_LABEL, eq, [f"EQ:TCQPQ-{b}{EQ_TAG}" for b in EQ_BPW], LTQ_STYLE, LTQ_COLOR)


def with_eq_exl3(src: dict, keys: list, eq: dict) -> tuple[dict, list]:
    """Adds to the EXL3 series its runs at the same bitrates as LTQ (2.5 / 3.0 / 4.0 are missing from the main set)."""
    merged = dict(src)
    merged.update(eq)
    have = {round(src[k]["info"]["linear_bpw"], 2) for k in keys if k in src}
    extra = [f"EQ:EXL3-{b}" for b in EQ_BPW if f"EQ:EXL3-{b}" in eq and round(eq[f"EQ:EXL3-{b}"]["info"]["linear_bpw"], 2) not in have]
    return merged, keys + extra


def get_1b_series(gguf_res: dict, iqk_res: dict, liqk_v2_res: dict, exl3_res: dict, eq: dict | None = None):
    iqk_v1_merged = dict(gguf_res)
    iqk_v1_merged.update(iqk_res)
    series = [
        (
            "Classic GGUF (IQ2_XXS, IQ3_XXS, Q3_K_M, IQ4_XS, Q4_K_M, Q8_0 + imatrix)",
            iqk_v1_merged,
            [
                "GGUF-IQ2_XXS-imatrix",
                "GGUF-IQ3_XXS-imatrix",
                "GGUF-Q3_K_M-imatrix",
                "GGUF-IQ4_XS-imatrix",
                "GGUF-Q4_K_M-imatrix",
                "GGUF-Q8_0-imatrix",
            ],
            "o--",
            "#aab7b8",
        ),
        (
            "Stock PR #30718 (IQ2_K..IQ6_K + imatrix)",
            iqk_v1_merged,
            [
                "GGUF-IQ2_K-imatrix",
                "GGUF-IQ3_K-imatrix",
                "GGUF-IQ4_K-imatrix",
                "GGUF-IQ5_K-imatrix",
                "GGUF-IQ6_K-imatrix",
            ],
            "s--",
            "#e67e22",
        ),
        (
            "Our LDLQ-IQ*_K (no Hadamard, 100% .gguf compatible)",
            iqk_v1_merged,
            ["LDLQ-IQ2_K", "LDLQ-IQ3_K", "LDLQ-IQ4_K", "LDLQ-IQ5_K", "LDLQ-IQ6_K"],
            "^-.",
            "#5dade2",
        ),
        (
            "Our .liqk v1 (Hadamard + LDLQ + per-matrix bit DP)",
            iqk_v1_merged,
            [
                "LIQK-IQ2_K",
                "LIQK-IQ3_K",
                "DP-LIQK-3.84bpw",
                "DP-LIQK-4.27bpw",
                "LIQK-IQ4_K",
                "LIQK-IQ5_K",
                "LIQK-IQ6_K",
            ],
            "D-.",
            "#c39bd3",
        ),
        (
            "Our .liqk v2 Uniform (Warp-Reg VQ + CS-suh + Block16-LDLQ + Polish, 2.0..8.0 bpw)",
            liqk_v2_res,
            [
                "LIQK2_00_V2-Uniform",
                "LIQK2_V2-Uniform",
                "LIQK3_V2-Uniform",
                "LIQK4_V2-Uniform",
                "LIQK5_V2-Uniform",
                "LIQK6_V2-Uniform",
                "LIQK8_00_V2-Uniform",
            ],
            "P--",
            "#48c9b0",
        ),
        (
            "Our .liqk v2 Mixed (Full pipeline + Fisher-Slice DP + MLP Perm, 2.06..8.0 bpw)",
            liqk_v2_res,
            [
                "LIQK_V2-2.06bpw",
                "LIQK_V2-2.38bpw",
                "LIQK_V2-2.90bpw",
                "LIQK_V2-3.44bpw",
                "LIQK_V2-3.84bpw",
                "LIQK_V2-4.27bpw",
                "LIQK_V2-4.50bpw",
                "LIQK_V2-5.00bpw",
                "LIQK_V2-5.50bpw",
                "LIQK_V2-6.62bpw",
                "LIQK_V2-8.00bpw",
            ],
            "*-",
            "#27ae60",
        ),
        (
            "EXL3 (mul1 codebook, 16x16 trellis + 2-sided Hadamard, 2.06..8.0 bpw)",
            exl3_res,
            [
                "EXL3-mul1-2.06bpw",
                "EXL3-mul1-2.38bpw",
                "EXL3-mul1-3.44bpw",
                "EXL3-mul1-3.84bpw",
                "EXL3-mul1-4.27bpw",
                "EXL3-mul1-4.50bpw",
                "EXL3-mul1-5.50bpw",
                "EXL3-mul1-6.62bpw",
                "EXL3-mul1-8.00bpw",
            ],
            "v:",
            "#ff6b5b",
        ),
        (
            "EXL3 (mcg codebook, 16x16 trellis + 2-sided Hadamard)",
            exl3_res,
            ["EXL3-mcg-2.38bpw", "EXL3-mcg-3.44bpw", "EXL3-mcg-4.50bpw"],
            "^:",
            "#f5b7b1",
        ),
    ]
    if eq:
        series = [(lbl, *with_eq_exl3(src, keys, eq), st, c) if lbl.startswith("EXL3 (mul1") else (lbl, src, keys, st, c)
                  for lbl, src, keys, st, c in series]
        series.append(ltq_series(eq))
    return series


def get_3b_series(merged_3b: dict, eq: dict | None = None):
    series = [
        (
            "Classic GGUF (IQ2_XXS, IQ3_XXS, Q3_K_M, IQ4_XS, Q4_K_M, Q8_0 + imatrix)",
            merged_3b,
            [
                "GGUF-IQ2_XXS-imatrix",
                "GGUF-IQ3_XXS-imatrix",
                "GGUF-Q3_K_M-imatrix",
                "GGUF-IQ4_XS-imatrix",
                "GGUF-Q4_K_M-imatrix",
                "GGUF-Q8_0-imatrix",
            ],
            "o--",
            "#aab7b8",
        ),
        (
            "Stock PR #30718 (IQ2_K..IQ6_K + imatrix)",
            merged_3b,
            [
                "GGUF-IQ2_K-imatrix",
                "GGUF-IQ3_K-imatrix",
                "GGUF-IQ4_K-imatrix",
                "GGUF-IQ5_K-imatrix",
                "GGUF-IQ6_K-imatrix",
            ],
            "s--",
            "#e67e22",
        ),
        (
            "Our LDLQ-IQ*_K (no Hadamard, 100% .gguf compatible)",
            merged_3b,
            ["LDLQ-IQ2_K", "LDLQ-IQ3_K", "LDLQ-IQ4_K", "LDLQ-IQ5_K", "LDLQ-IQ6_K"],
            "^-.",
            "#5dade2",
        ),
        (
            "Our .liqk v1 (Hadamard + LDLQ + per-matrix bit DP)",
            merged_3b,
            ["LIQK-IQ2_K", "LIQK-IQ3_K", "DP-LIQK-3.88bpw", "DP-LIQK-4.27bpw", "LIQK-IQ4_K", "LIQK-IQ5_K", "LIQK-IQ6_K"],
            "D-.",
            "#c39bd3",
        ),
        (
            "Our .liqk v2 Uniform (Warp-Reg VQ + CS-suh + Block16-LDLQ + Polish, 2.0..8.0 bpw)",
            merged_3b,
            [
                "LIQK2_00_V2-Uniform",
                "LIQK2_V2-Uniform",
                "LIQK3_V2-Uniform",
                "LIQK4_V2-Uniform",
                "LIQK5_V2-Uniform",
                "LIQK6_V2-Uniform",
                "LIQK8_00_V2-Uniform",
            ],
            "P--",
            "#48c9b0",
        ),
        (
            "Our .liqk v2 Mixed (Warp-Reg VQ + Block16-LDLQ + Fisher-Slice DP + Polish, 2.06..8.0 bpw)",
            merged_3b,
            [
                "LIQK_V2-2.06bpw",
                "LIQK_V2-2.38bpw",
                "LIQK_V2-2.90bpw",
                "LIQK_V2-3.44bpw",
                "LIQK_V2-3.85bpw",
                "LIQK_V2-4.25bpw",
                "LIQK_V2-4.50bpw",
                "LIQK_V2-5.00bpw",
                "LIQK_V2-5.50bpw",
                "LIQK_V2-6.62bpw",
                "LIQK_V2-8.00bpw",
            ],
            "*-",
            "#27ae60",
        ),
        (
            "EXL3 (mul1 codebook, 16x16 trellis, 2.06..8.0 bpw)",
            merged_3b,
            [
                "EXL3-mul1-2.06bpw",
                "EXL3-mul1-2.38bpw",
                "EXL3-mul1-3.44bpw",
                "EXL3-mul1-3.85bpw",
                "EXL3-mul1-4.25bpw",
                "EXL3-mul1-4.50bpw",
                "EXL3-mul1-5.50bpw",
                "EXL3-mul1-6.62bpw",
                "EXL3-mul1-8.00bpw",
            ],
            "v:",
            "#ff6b5b",
        ),
    ]
    if eq:
        series = [(lbl, *with_eq_exl3(src, keys, eq), st, c) if lbl.startswith("EXL3 (mul1") else (lbl, src, keys, st, c)
                  for lbl, src, keys, st, c in series]
        series.append(ltq_series(eq))
    return series


def _overlay_audit_points(ax, audit_res: dict, trace_key: str, show_label: bool):
    """Plots v2 Mixed audit points with the original settings (no CE) as hollow stars."""
    pts = []
    for k, item in audit_res.items():
        if k.startswith("LIQK_V2-") and k.endswith("-orig-ce0"):
            m = item[trace_key]
            pts.append((item["info"]["linear_bpw"], m["mean_kld"], m["mean_kld"] - m["ci95_lo"], m["ci95_hi"] - m["mean_kld"]))
    if not pts:
        return
    pts.sort()
    ax.errorbar(
        [p[0] for p in pts],
        [p[1] for p in pts],
        yerr=[[p[2] for p in pts], [p[3] for p in pts]],
        fmt="*",
        markersize=15,
        markerfacecolor="none",
        markeredgecolor="#abebc6",
        markeredgewidth=1.6,
        ecolor="#abebc6",
        capsize=3,
        linestyle="none",
        label="Audit: .liqk v2 Mixed, original settings, no CE" if show_label else None,
    )


ZOOM_BPW = (2.4, 4.6)  # zoomed region: the LTQ range where the comparison with EXL3 is decided

# KLD plots are for outside readers: short names, main series only, jargon-free labels
SHORT_NAMES = [  # (series label prefix, short name); series not listed here are omitted from the KLD plots
    ("Classic GGUF", "GGUF (Q3_K_M, IQ4_XS, Q4_K_M, Q8_0, etc.)"),
    ("Stock PR #30718", "GGUF IQ2_K…IQ6_K"),
    ("EXL3 (mul1", "EXL3"),
    (LTQ_LABEL, "LTQ (ours)"),
]
TRACES = [("nemotron", "English text: chat and reasoning"), ("multilingual", "Other languages: ru, zh, de, fr, es")]
XLABEL = "Bits per weight (average)"
YLABEL = "Difference from the original model\n(KL divergence, lower is better)"
CAPTION = ("Difference from the original model: how much the compressed model changes next-word probabilities (0 = no change).\n"
           "Mean over ~20k tokens of English dialogues and ~27k tokens in other languages; vertical bars are 95% confidence intervals.")


def _short(series_defs: list) -> list:
    out = []
    for label, src, keys, style, color in series_defs:
        name = next((n for pre, n in SHORT_NAMES if label.startswith(pre)), None)
        if name:
            out.append((name, src, keys, style, color))
    return out


def _plot_pareto_kld(series_defs: list, model: str, out_path: Path):
    """Top row: full bitrate range; bottom row: the ZOOM_BPW region (on the full range, 3..4.5 bpw points are squashed near zero)."""
    series_defs = _short(series_defs)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.4), dpi=160)
    for col, (trace_key, title) in enumerate(TRACES):
        zoom_max = 0.0
        for row in (0, 1):
            ax = axes[row, col]
            for label, src_dict, keys, style, color in series_defs:
                pts = []
                for k in keys:
                    if k not in src_dict:
                        continue
                    item = src_dict[k]
                    m = item[trace_key]
                    kld = m["mean_kld"]
                    pts.append((item["info"]["linear_bpw"], kld, max(0.0, kld - m["ci95_lo"]), max(0.0, m["ci95_hi"] - kld)))
                pts.sort(key=lambda x: x[0])
                if not pts:
                    continue
                main = label == "LTQ (ours)"
                ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[[p[2] for p in pts], [p[3] for p in pts]], fmt=style,
                            color=color, label=label, linewidth=3.0 if main else 2.0, markersize=11.0 if main else 7.0, capsize=3)
                if main or label == "EXL3":
                    zoom_max = max([zoom_max] + [p[1] + p[3] for p in pts if ZOOM_BPW[0] <= p[0] <= ZOOM_BPW[1]])
            _linear_decimal_y(ax)
            if row == 1:
                ax.set_xlim(*ZOOM_BPW)
                ax.set_ylim(0, zoom_max * 1.1)
            ax.set_xlabel(XLABEL, fontsize=11)
            ax.set_ylabel(YLABEL, fontsize=11)
            ax.set_title(title if row == 0 else f"{title} — zoomed, {ZOOM_BPW[0]}…{ZOOM_BPW[1]} bits", fontsize=12, fontweight="bold")
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.042))
    fig.text(0.5, 0.008, CAPTION, ha="center", fontsize=10, color="#aab7b8")
    fig.suptitle(f"{model}: how much compression degrades the model", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.975))
    fig.savefig(out_path)
    plt.close(fig)


def plot_pareto_1b(gguf_res: dict, iqk_res: dict, liqk_v2_res: dict, exl3_res: dict, out_dir: Path, audit_res: dict | None = None,
                   eq: dict | None = None):
    _plot_pareto_kld(get_1b_series(gguf_res, iqk_res, liqk_v2_res, exl3_res, eq), "Llama-3.2-1B", out_dir / "pareto_1b_kld.png")


def plot_pareto_3b(res_3b: dict, liqk_v2_3b: dict, out_dir: Path, audit_res: dict | None = None, eq: dict | None = None):
    if not res_3b:
        return
    merged_3b = dict(res_3b)
    merged_3b.update(liqk_v2_3b)
    _plot_pareto_kld(get_3b_series(merged_3b, eq), "Llama-3.2-3B", out_dir / "pareto_3b_kld.png")


def _plot_ppl_two_rows(series_defs: list, suptitle: str, ref_nem: float, ref_mul: float, out_path: Path):
    """
    Builds an honest 2-row perplexity plot (PPL vs BPW) without log(ΔPPL -> 0) artifacts:
      - Top row: full range (2.38 .. 6.625 bpw, including extreme 2-bit quants).
      - Bottom row: close-up of the working range (3.35 .. 6.65 bpw) on a linear PPL scale with a dashed BF16 asymptote.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.6), dpi=160)

    trace_configs = [
        ("nemotron", f"Nemotron (BF16 = {ref_nem:.4f})", ref_nem),
        ("multilingual", f"Multilingual (BF16 = {ref_mul:.4f})", ref_mul),
    ]

    for col_idx, (trace_key, base_title, ppl_ref) in enumerate(trace_configs):
        ax_full = axes[0, col_idx]
        ax_zoom = axes[1, col_idx]

        ax_full.axhline(ppl_ref, color="#d5d8dc", linestyle="--", linewidth=1.6, label=f"BF16 reference ({ppl_ref:.4f})")
        ax_zoom.axhline(ppl_ref, color="#d5d8dc", linestyle="--", linewidth=1.6, label=f"BF16 reference ({ppl_ref:.4f})")

        zoom_ys = []
        for label, src_dict, keys, style, color in series_defs:
            pts = []
            for k in keys:
                if k not in src_dict:
                    continue
                item = src_dict[k]
                bpw = item["info"]["linear_bpw"]
                m = item[trace_key]
                ppl_q = m["ppl_quant"]
                pts.append((bpw, ppl_q, ppl_q - m.get("ppl_ci95_lo", ppl_q), m.get("ppl_ci95_hi", ppl_q) - ppl_q))
            pts.sort(key=lambda x: x[0])
            if not pts:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            lw = 2.6 if ".liqk v2 Mixed" in label else 1.9
            ms = 9.0 if ".liqk v2 Mixed" in label else 6.8

            yerr = [[p[2] for p in pts], [p[3] for p in pts]]
            ax_full.errorbar(xs, ys, yerr=yerr, fmt=style, color=color, label=label, linewidth=lw, markersize=ms, capsize=3)

            pts_zoom = [p for p in pts if p[0] >= 3.35]
            if pts_zoom:
                xz = [p[0] for p in pts_zoom]
                yz = [p[1] for p in pts_zoom]
                zoom_ys.extend(yz)
                yerr_z = [[p[2] for p in pts_zoom], [p[3] for p in pts_zoom]]
                ax_zoom.errorbar(xz, yz, yerr=yerr_z, fmt=style, color=color, label=label, linewidth=lw, markersize=ms, capsize=3)

        ax_full.set_title(f"{base_title} — Full range (2.00..8.50 bpw)", fontsize=11.5, fontweight="bold")
        ax_full.set_xlabel("Average bitrate of linear layers (BPW)", fontsize=10.5)
        ax_full.set_ylabel("Perplexity PPL (linear scale, lower = better)", fontsize=10.5)
        ax_full.grid(True, which="both", linestyle=":", alpha=0.6)
        if col_idx == 0:
            ax_full.legend(fontsize=7.8, loc="upper right")

        ax_zoom.set_title(f"{base_title} — Close-up (3.44..8.50 bpw)", fontsize=11.5, fontweight="bold")
        ax_zoom.set_xlabel("Average bitrate of linear layers (BPW)", fontsize=10.5)
        ax_zoom.set_ylabel("Perplexity PPL (linear scale, close-up)", fontsize=10.5)
        if zoom_ys:
            y_min = min(min(zoom_ys), ppl_ref) - 0.025
            y_max = max(zoom_ys) + 0.04
            ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.grid(True, which="both", linestyle=":", alpha=0.6)
        if col_idx == 0:
            ax_zoom.legend(fontsize=7.8, loc="upper right")

    fig.suptitle(suptitle, fontsize=13.5, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "PPL for all .liqk v2 points is measured after polish without the CE term (ce_weight = 0). The absolute per-dialogue 95% CI of PPL "
        "is about ±13%, so quants can only be compared with each other via paired differences (see audit_*.png).",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def plot_pareto_ppl_1b(gguf_res: dict, iqk_res: dict, liqk_v2_res: dict, exl3_res: dict, out_dir: Path):
    series_defs = get_1b_series(gguf_res, iqk_res, liqk_v2_res, exl3_res)
    _plot_ppl_two_rows(
        series_defs,
        "Llama-3.2-1B-Instruct: Perplexity PPL vs BPW (2.00..8.50 bpw, full overview and close-up 3.44..8.50 bpw)",
        6.0382,
        5.8156,
        out_dir / "pareto_1b_ppl.png",
    )


def plot_pareto_ppl_3b(res_3b: dict, liqk_v2_3b: dict, out_dir: Path):
    if not res_3b:
        return
    merged_3b = dict(res_3b)
    merged_3b.update(liqk_v2_3b)
    series_defs = get_3b_series(merged_3b)
    _plot_ppl_two_rows(
        series_defs,
        "Llama-3.2-3B-Instruct (3.21B): Perplexity PPL vs BPW (2.00..8.50 bpw, full overview and close-up 3.44..8.50 bpw)",
        4.7739,
        4.3443,
        out_dir / "pareto_3b_ppl.png",
    )


def plot_ablation_waterfall_344(gguf_res: dict, iqk_res: dict, liqk_v2_res: dict, exl3_res: dict, out_dir: Path):
    """Builds a bar chart of the step-by-step ablation at exactly the 3.4375 bpw budget on Llama-3.2-1B."""
    bars_spec = [
        ("1. Stock IQ3_K\n(PR #30718 + imat)", gguf_res.get("GGUF-IQ3_K-imatrix"), "#e67e22"),
        ("2. LDLQ-IQ3_K\n(100% .gguf compat.)", iqk_res.get("LDLQ-IQ3_K"), "#85c1e9"),
        ("3. .liqk v1\n(+ Hadamard + LDLQ)", iqk_res.get("LIQK-IQ3_K"), "#d2b4de"),
        ("4. + Warp-Reg & LDLQ\n(LIQK3_V2 NoPolish)", liqk_v2_res.get("LIQK3_V2-NoPolish"), "#5dade2"),
        ("5. + Scale Polish\n(LIQK3_V2 Uniform)", liqk_v2_res.get("LIQK3_V2-Uniform"), "#48c9b0"),
        ("6. + Fisher-Slice DP\n(Mixed NoPolish)", liqk_v2_res.get("LIQK_V2-3.44bpw-NoPolish"), "#2ecc71"),
        ("7. .liqk v2 Mixed\n(+ Polish)", liqk_v2_res.get("LIQK_V2-3.44bpw"), "#27ae60"),
        ("8. EXL3 (mul1)\n(16x16 trellis)", exl3_res.get("EXL3-mul1-3.44bpw"), "#ff6b5b"),
    ]
    bars_spec = [(lbl, item, col) for lbl, item, col in bars_spec if item is not None]
    if not bars_spec:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=160)
    for ax_idx, (tkey, tname) in enumerate([("nemotron", "Nemotron KLD"), ("multilingual", "Multilingual KLD")]):
        ax = axes[ax_idx]
        labels = [b[0] for b in bars_spec]
        vals = [b[1][tkey]["mean_kld"] for b in bars_spec]
        errs = [
            [v - b[1][tkey]["ci95_lo"] for v, b in zip(vals, bars_spec)],
            [b[1][tkey]["ci95_hi"] - v for v, b in zip(vals, bars_spec)],
        ]
        cols = [b[2] for b in bars_spec]
        x = np.arange(len(labels))
        rects = ax.bar(x, vals, yerr=errs, capsize=3, color=cols, width=0.64, edgecolor="black", linewidth=0.7)
        for rect, v, e_hi in zip(rects, vals, errs[1]):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                v + e_hi + max(vals) * 0.015,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.6,
                fontweight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.0, rotation=18, ha="right")
        ax.set_ylabel("Teacher-Forced KLD [nats] (lower = better)", fontsize=10.5)
        ax.set_title(f"{tname} (strictly at 3.4375 bpw)", fontsize=11.5, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.15)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(
        "Step-by-step ablation .liqk v1 -> .liqk v2 at 3.4375 bpw (Llama-3.2-1B-Instruct, per-dialogue 95% CI)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_344bpw.png")
    plt.close(fig)


def plot_audit(audit_res: dict, model_name: str, out_path: Path):
    """
    Audit plot: for each bitrate and each dataset
      - top row: KLD with 95% CI for the three polish configurations and EXL3;
      - bottom row: paired difference PPL(v2) - PPL(EXL3) with 95% CI (below zero = v2 is better).
    """
    if not audit_res:
        return
    bpws = sorted({k.split("-")[1] for k in audit_res if k.startswith("LIQK_V2-") and k.endswith("-orig-ce0")})
    bpws = [b for b in bpws if f"EXL3-mul1-{b}" in audit_res]
    if not bpws:
        return

    fig, axes = plt.subplots(2, len(AUDIT_DATASETS), figsize=(17.5, 10.2), dpi=160)
    group_w = 0.8
    n_bars_top = len(AUDIT_CONFIGS) + 1
    n_bars_bot = len(AUDIT_CONFIGS)

    for col, (ds, ds_name) in enumerate(AUDIT_DATASETS):
        ax_k, ax_p = axes[0, col], axes[1, col]
        for gi, bpw in enumerate(bpws):
            exl3 = audit_res[f"EXL3-mul1-{bpw}"]
            entries = [(audit_res.get(f"LIQK_V2-{bpw}-{suf}"), lbl, c, h) for suf, lbl, c, h in AUDIT_CONFIGS]
            entries.append((exl3, "EXL3 mul1", "#ff6b5b", ""))
            for bi, (item, lbl, c, h) in enumerate(entries):
                if item is None:
                    continue
                m = item[ds]
                xk = gi + (bi - (n_bars_top - 1) / 2) * group_w / n_bars_top
                ax_k.bar(
                    xk, m["mean_kld"], width=group_w / n_bars_top, color=c, hatch=h, edgecolor="black", linewidth=0.6,
                    yerr=[[m["mean_kld"] - m["ci95_lo"]], [m["ci95_hi"] - m["mean_kld"]]], capsize=2.5,
                    label=lbl if (gi == 0 and col == 0) else None,
                )
            for bi, (item, lbl, c, h) in enumerate(entries[:-1]):
                if item is None:
                    continue
                d = paired_bootstrap_ppl_diff(item[ds], exl3[ds])
                xp = gi + (bi - (n_bars_bot - 1) / 2) * group_w / n_bars_bot
                ax_p.bar(
                    xp, d["mean_diff"], width=group_w / n_bars_bot, color=c, hatch=h, edgecolor="black", linewidth=0.6,
                    yerr=[[d["mean_diff"] - d["ci95_lo"]], [d["ci95_hi"] - d["mean_diff"]]], capsize=2.5,
                )
                ax_p.annotate(
                    f"{d['mean_diff']:+.3f}", (xp, max(d["ci95_hi"], 0)), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5,
                )

        for ax in (ax_k, ax_p):
            ax.set_xticks(range(len(bpws)))
            ax.set_xticklabels(bpws, fontsize=10)
            ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax_k.set_title(f"{ds_name}: KLD, 95% CI", fontsize=11.5, fontweight="bold")
        ax_k.set_ylabel("KL(P_BF16 || P_quant) [nats]", fontsize=10)
        ax_p.axhline(0.0, color="#d5d8dc", linewidth=1.0)
        ax_p.set_title(f"{ds_name}: PPL(v2) − PPL(EXL3)", fontsize=11.5, fontweight="bold")
        ax_p.set_ylabel("ΔPPL, paired bootstrap (below zero = v2 is better)", fontsize=10)
        lo, hi = ax_p.get_ylim()
        ax_p.set_ylim(lo, hi + 0.08 * (hi - lo))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, [l.replace("\n", " ") for l in labels], loc="lower center", ncol=len(labels), fontsize=9.5, frameon=False)
    fig.suptitle(
        f"{model_name}: audit of .liqk v2 vs EXL3 at equal bitrate (top: KLD, bottom: paired ΔPPL vs EXL3)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def plot_speed_benchmarks(speed_res: dict, out_dir: Path):
    """Builds generation (tg128) and prefill (pp512) speed plots on CPU and GPU."""
    if not speed_res:
        return
    gpu_rows = speed_res.get("gpu_bench", [])
    cpu_rows = speed_res.get("cpu_bench", [])
    if not gpu_rows and not cpu_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=160)

    cpu_tg16 = [r for r in cpu_rows if r["mode"] == "tg128" and r["n_threads"] == 16]
    if cpu_tg16:
        ax = axes[0]
        lbls = [r["label"] for r in cpu_tg16]
        ts = [r["avg_ts"] for r in cpu_tg16]
        x = np.arange(len(lbls))
        rects = ax.bar(x, ts, color="#5dade2", width=0.6, edgecolor="black")
        for rect, v in zip(rects, ts):
            ax.text(rect.get_x() + rect.get_width() / 2, v + max(ts) * 0.015, f"{v:.1f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Generation speed tg128 [tok/s] (higher = better)", fontsize=10.5)
        ax.set_title("CPU: AMD Ryzen 9 9950X (16 threads, AVX-512 VNNI) — Decode tg128", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(ts) * 1.15)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    gpu_tg = [r for r in gpu_rows if r["mode"] == "tg128"]
    if gpu_tg:
        ax = axes[1]
        lbls = [r["label"] for r in gpu_tg]
        ts = [r["avg_ts"] for r in gpu_tg]
        x = np.arange(len(lbls))
        rects = ax.bar(x, ts, color="#27ae60", width=0.6, edgecolor="black")
        for rect, v in zip(rects, ts):
            ax.text(rect.get_x() + rect.get_width() / 2, v + max(ts) * 0.015, f"{v:.1f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Generation speed tg128 [tok/s] (higher = better)", fontsize=10.5)
        ax.set_title("GPU: NVIDIA RTX 5070 Ti (CUDA) — Decode tg128", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(ts) * 1.15)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle("Inference performance of GGUF quants and the IQ2_K..IQ6_K family (Llama-3.2-1B)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "speed_cpu_gpu.png")
    plt.close(fig)


def _plot_vs_exl3(models: list, suptitle: str, out_path: Path, exl3_label: str = "EXL3"):
    """One row per model: LTQ vs EXL3 at the same bitrate, annotated with the difference from EXL3 in percent.
    `models` is [(name, EXL3 points, LTQ points)]; points are `methods` entries of the results."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.4), dpi=160)
    for row, (model, exl3_items, ltq_items) in enumerate(models):
        for col, (trace_key, title) in enumerate(TRACES):
            ax = axes[row, col]
            pts_by = {}
            for label, items, style, color, main in ((exl3_label, exl3_items, "v:", "#ff6b5b", False), ("LTQ (ours)", ltq_items, LTQ_STYLE, LTQ_COLOR, True)):
                pts = sorted((it["linear_bpw"], it[trace_key]["mean_kld"], it[trace_key]["ci95_lo"], it[trace_key]["ci95_hi"])
                             for it in items if it)
                pts_by[label] = pts
                ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                            yerr=[[max(0.0, p[1] - p[2]) for p in pts], [max(0.0, p[3] - p[1]) for p in pts]],
                            fmt=style, color=color, label=label, linewidth=3.0 if main else 2.0, markersize=12.0 if main else 8.0, capsize=3)
            for bpw, kld, _, _ in pts_by["LTQ (ours)"]:  # EXL3 at the same bitrate (up to overhead bits)
                ref = next((y for x, y, _, _ in pts_by[exl3_label] if abs(x - bpw) < 0.05), None)
                if ref:
                    ax.annotate(f"{(kld / ref - 1) * 100:+.0f} % vs EXL3".replace("-", "−"), (bpw, kld), textcoords="offset points",
                                xytext=(10, -18), fontsize=11, color=LTQ_COLOR, fontweight="bold")
            _linear_decimal_y(ax)
            ax.set_xlim(2.85, 4.75)  # 3.0..4.5 bpw
            ax.set_xlabel(XLABEL, fontsize=11)
            ax.set_ylabel(YLABEL, fontsize=11)
            ax.set_title(f"{model} — {title}", fontsize=12, fontweight="bold")
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.042))
    fig.text(0.5, 0.008, CAPTION, ha="center", fontsize=10, color="#aab7b8")
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.975))
    fig.savefig(out_path)
    plt.close(fig)




def plot_qwen3(res_dir: Path):
    """Qwen3-8B and Qwen3-30B-A3B (MoE): LTQ vs EXL3 at the same bitrate, annotated with the difference from EXL3 in percent."""
    def m(fname, key):
        return load_json_safe(res_dir / fname).get("methods", {}).get(key)

    models = [
        ("Qwen3-8B", [m("night2_qwen3_8b_final.json", "exl3:exl3_out/qwen3_8b_3.0"), m("night_qwen3_8b.json", "exl3:exl3_out/qwen3_8b_3.4375"),
                      m("night_qwen3_8b.json", "exl3:exl3_out/qwen3_8b_4.5")],
         [m("night2_qwen3_8b_final.json", "dp:TCQPQ:3.0"), m("night2_qwen3_8b_dp_fisher.json", "dp:TCQPQ:3.4375"),
          m("night2_qwen3_8b_final.json", "dp:TCQPQ:4.5")]),
        ("Qwen3-30B-A3B (MoE)", [m("g4_qwen3_30b_dp.json", "exl3:exl3_out/qwen3_30b_3.0"), m("night_qwen3_30b.json", "exl3:exl3_out/qwen3_30b_3.4375"),
                                 m("g4_qwen3_30b_dp.json", "exl3:exl3_out/qwen3_30b_4.5")],
         [m("g4_qwen3_30b_dp.json", "dp:TCQPQ:3.0"), m("g4_qwen3_30b_dp.json", "dp:TCQPQ:4.5")]),
    ]
    _plot_vs_exl3(models, "Qwen3: LTQ vs EXL3 at equal size", res_dir / "kld_qwen3.png")


def plot_hybrid(res_dir: Path):
    """Hybrid architectures (Gemma 4 E4B: PLE and shared KV; Qwen3.5-4B: Gated DeltaNet): LTQ vs per-matrix EXL3
    on the same Hessians (the stock convert.py does not support Gemma 4 PLE models)."""
    models = []
    for name, d in (("Qwen3.5-4B (Gated DeltaNet)", "qwen35"), ("Gemma 4 E4B (PLE, shared KV)", "gemma4")):
        ms = load_json_safe(res_dir / f"g4_hybrid_{d}.json").get("methods", {})
        models.append((name, [v for k, v in ms.items() if k.startswith("exl3q:")], [v for k, v in ms.items() if k.startswith("dp:")]))
    _plot_vs_exl3(models, "Models with recurrent layers and PLE: LTQ vs EXL3", res_dir / "kld_hybrid.png",
                  "EXL3 (per-matrix, same Hessians)")


def main():
    root = Path(__file__).resolve().parent.parent
    res_dir = root / "results"
    gguf_res = load_json_safe(res_dir / "gguf_stock_1b.json")
    iqk_res = load_json_safe(res_dir / "iqk_liqk_1b.json")
    liqk_v2_1b = load_json_safe(res_dir / "liqk_v2_1b.json")
    exl3_res = load_json_safe(res_dir / "exl3_1b.json")
    speed_res = load_json_safe(res_dir / "speed_and_ppl_1b.json")
    res_3b = load_json_safe(res_dir / "all_3b_results.json")
    liqk_v2_3b = load_json_safe(res_dir / "liqk_v2_3b.json")
    audit_1b = load_json_safe(res_dir / "audit_1b.json")
    audit_3b = load_json_safe(res_dir / "audit_3b.json")

    eq_1b, eq_3b = eq_runs(res_dir / "equal_bpw_1b.json"), eq_runs(res_dir / "equal_bpw_3b.json")
    plot_pareto_1b(gguf_res, iqk_res, liqk_v2_1b, exl3_res, res_dir, audit_1b, eq_1b)
    plot_pareto_ppl_1b(gguf_res, iqk_res, liqk_v2_1b, exl3_res, res_dir)
    plot_ablation_waterfall_344(gguf_res, iqk_res, liqk_v2_1b, exl3_res, res_dir)
    plot_speed_benchmarks(speed_res, res_dir)
    plot_pareto_3b(res_3b, liqk_v2_3b, res_dir, audit_3b, eq_3b)
    plot_qwen3(res_dir)
    plot_hybrid(res_dir)
    plot_pareto_ppl_3b(res_3b, liqk_v2_3b, res_dir)
    plot_audit(audit_1b, "Llama-3.2-1B-Instruct", res_dir / "audit_1b.png")
    plot_audit(audit_3b, "Llama-3.2-3B-Instruct", res_dir / "audit_3b.png")
    print("Plots successfully generated in", res_dir)


if __name__ == "__main__":
    main()
