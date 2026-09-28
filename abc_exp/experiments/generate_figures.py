#!/usr/bin/env python3
"""
Generate publication-quality figures for NeurIPS paper on ABC (Alpha-Based
Calibration) debate method.

Usage:
    python generate_figures.py [--output-dir DIR] [--format {pdf,png,both}]

Figures produced:
    1. Alpha Distribution and Reliability (2-panel)
    2. ABC Method Overview (placeholder)
    3. ABC vs Standard Debate Accuracy (grouped bar chart)
    4. Multi-Benchmark Generalization (grouped bar chart)
    5. Ablation Studies (4-panel grid)
    6. Collapse Prevention (2-panel)
    Table 1: Main Results (LaTeX .tex file)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Optional imports -- degrade gracefully
# ---------------------------------------------------------------------------
try:
    from scipy import stats as sp_stats
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found; KDE overlay and some statistics will be "
                   "skipped.")

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

# ===========================================================================
# Configuration
# ===========================================================================

# NeurIPS column widths (inches)
SINGLE_COL = 3.25
DOUBLE_COL = 6.75

# Colorblind-friendly palette (Paul Tol's bright)
COLORS = {
    "blue":      "#4477AA",
    "cyan":      "#66CCEE",
    "green":     "#228833",
    "yellow":    "#CCBB44",
    "red":       "#EE6677",
    "purple":    "#AA3377",
    "grey":      "#BBBBBB",
    "dark_grey": "#555555",
}
# Ordered list for bar charts
PALETTE = [COLORS["blue"], COLORS["red"], COLORS["green"],
           COLORS["yellow"], COLORS["cyan"], COLORS["purple"],
           COLORS["grey"]]

# Matplotlib rcParams for NeurIPS style
NEURIPS_RC = {
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "legend.frameon":     False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,   # TrueType for editable text in PDF
    "ps.fonttype":        42,
}


# ===========================================================================
# Helpers
# ===========================================================================

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file, returning a list of dicts."""
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def try_load(path: str | Path, loader=load_jsonl) -> Any | None:
    """Return data or None if file does not exist."""
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return loader(p)
    return None


def savefig(fig: plt.Figure, stem: str, out_dir: Path,
            fmt: str = "both") -> None:
    """Save figure as PDF and/or PNG."""
    if fmt in ("pdf", "both"):
        fig.savefig(out_dir / f"{stem}.pdf")
    if fmt in ("png", "both"):
        fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  Saved {stem}")


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000,
                 ci: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) via bootstrap."""
    rng = np.random.default_rng(42)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([
        values[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)
    ])
    alpha_half = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha_half)
    hi = np.percentile(boot_means, 100 * (1 - alpha_half))
    return float(values.mean()), float(lo), float(hi)


def mcnemar_pvalue(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Two-sided McNemar test p-value (exact binomial)."""
    if not HAS_SCIPY:
        return float("nan")
    # Discordant pairs
    b_only = int(np.sum((~correct_a) & correct_b))
    a_only = int(np.sum(correct_a & (~correct_b)))
    n_disc = b_only + a_only
    if n_disc == 0:
        return 1.0
    # exact binomial test (binomtest replaces deprecated binom_test)
    result = sp_stats.binomtest(b_only, n_disc, 0.5)
    return float(result.pvalue)


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def annotate_significance(ax, x1: float, x2: float, y: float,
                          p: float, h: float = 0.01) -> None:
    """Draw a bracket with significance stars between two bar positions."""
    stars = significance_stars(p)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8,
            color=COLORS["dark_grey"])
    ax.text((x1 + x2) / 2, y + h, f"p={p:.3f} {stars}",
            ha="center", va="bottom", fontsize=7, color=COLORS["dark_grey"])


# ===========================================================================
# Figure 1: Alpha Distribution and Reliability
# ===========================================================================

def figure1(alpha_data: list[dict], retest_data: list[dict],
            summary: dict, out_dir: Path, fmt: str) -> None:
    """Panel A: Alpha histogram + KDE.  Panel B: Test-retest scatter."""
    print("[Figure 1] Alpha Distribution and Reliability")

    # -- Extract alpha values (flip-rate alpha = mean_flip_rate) --
    alphas_all: list[float] = []
    for rec in alpha_data:
        for agent_key, est in rec["alpha_estimates"].items():
            alphas_all.append(est["mean_flip_rate"])
    alphas_arr = np.array(alphas_all)

    # -- Test-retest pairs --
    test_map: dict[tuple[str, str], dict] = {}
    for rec in alpha_data:
        for agent_key, est in rec["alpha_estimates"].items():
            test_map[(rec["question_id"], agent_key)] = est

    retest_map: dict[tuple[str, str], dict] = {}
    for rec in retest_data:
        for agent_key, est in rec["alpha_estimates"].items():
            retest_map[(rec["question_id"], agent_key)] = est

    common_keys = sorted(set(test_map) & set(retest_map))
    test_vals = np.array([test_map[k]["mean_flip_rate"] for k in common_keys])
    retest_vals = np.array([retest_map[k]["mean_flip_rate"]
                            for k in common_keys])

    # Correctness for colouring: use test data initial_correct
    correctness: list[bool] = []
    # Build a question-level lookup
    q_correct_map: dict[str, dict[str, bool]] = {}
    for rec in alpha_data:
        for agent_info in rec.get("agents", []):
            q_correct_map.setdefault(
                rec["question_id"], {}
            )[agent_info["agent_id"]] = agent_info.get("initial_correct", True)
    for k in common_keys:
        qid, aid = k
        correctness.append(q_correct_map.get(qid, {}).get(aid, True))
    correct_mask = np.array(correctness)

    # -- Create figure --
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))

    # ---- Panel A: Histogram + KDE ----
    bins = np.linspace(0, 1, 21)
    ax_a.hist(alphas_arr, bins=bins, density=True, alpha=0.55,
              color=COLORS["blue"], edgecolor="white", linewidth=0.5,
              label="Histogram")

    if HAS_SCIPY:
        xs = np.linspace(0, 1, 200)
        try:
            kde = gaussian_kde(alphas_arr, bw_method=0.12)
            ax_a.plot(xs, kde(xs), color=COLORS["red"], lw=1.5, label="KDE")
        except Exception:
            pass

    mean_val = float(np.mean(alphas_arr))
    median_val = float(np.median(alphas_arr))
    ax_a.axvline(mean_val, color=COLORS["red"], ls="--", lw=1,
                 label=f"Mean = {mean_val:.2f}")
    ax_a.axvline(median_val, color=COLORS["green"], ls=":", lw=1,
                 label=f"Median = {median_val:.2f}")
    ax_a.set_xlabel("Flip-rate $\\alpha$ (revisability)")
    ax_a.set_ylabel("Density")
    ax_a.set_title("(A) Alpha Distribution", fontsize=11)
    ax_a.legend(fontsize=7, loc="upper center")
    ax_a.set_xlim(-0.02, 1.02)

    # ---- Panel B: Test-retest scatter ----
    ax_b.scatter(test_vals[correct_mask], retest_vals[correct_mask],
                 s=10, alpha=0.45, color=COLORS["blue"],
                 label="Initially correct", zorder=3, edgecolors="none")
    ax_b.scatter(test_vals[~correct_mask], retest_vals[~correct_mask],
                 s=10, alpha=0.45, color=COLORS["red"],
                 label="Initially incorrect", zorder=3, edgecolors="none")

    # Regression line
    if HAS_SCIPY and len(test_vals) > 2:
        slope, intercept, r_val, _, _ = sp_stats.linregress(test_vals,
                                                             retest_vals)
        xs = np.linspace(0, 1, 100)
        ax_b.plot(xs, intercept + slope * xs, color=COLORS["dark_grey"],
                  lw=1, ls="--")

    # ICC from summary
    icc_val = summary.get("icc", {}).get("icc", None)
    pearson_r = summary.get("icc", {}).get("pearson_r", None)
    annotation_parts = []
    if icc_val is not None:
        annotation_parts.append(f"ICC = {icc_val:.3f}")
    if pearson_r is not None:
        annotation_parts.append(f"$r$ = {pearson_r:.3f}")
    annotation_parts.append(f"$n$ = {len(common_keys)}")
    ax_b.text(0.05, 0.95, "\n".join(annotation_parts),
              transform=ax_b.transAxes, fontsize=8, va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey",
                        alpha=0.8))

    ax_b.plot([0, 1], [0, 1], color=COLORS["grey"], lw=0.8, ls=":")
    ax_b.set_xlabel("Test $\\alpha$ (flip-rate)")
    ax_b.set_ylabel("Retest $\\alpha$ (flip-rate)")
    ax_b.set_title("(B) Test-Retest Reliability", fontsize=11)
    ax_b.set_xlim(-0.02, 1.02)
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.set_aspect("equal")
    ax_b.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    savefig(fig, "fig1_alpha_reliability", out_dir, fmt)


# ===========================================================================
# Figure 2: ABC Method Overview (placeholder)
# ===========================================================================

def figure2(out_dir: Path, fmt: str) -> None:
    """Conceptual diagram placeholder -- to be manually designed."""
    print("[Figure 2] ABC Method Overview (placeholder)")
    # This figure will be a manually-designed architectural diagram.
    # We create a minimal placeholder with a text annotation.
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_axis_off()

    # Draw conceptual flow boxes
    boxes = [
        (1.0, 2.0, "1. Alpha\nEstimation\n(Probe Phase)"),
        (4.0, 2.0, "2. Revisability\nWeighting\n$w_i = f(\\alpha_i)$"),
        (7.0, 2.0, "3. Weighted\nDebate\nAggregation"),
    ]
    for cx, cy, txt in boxes:
        rect = plt.Rectangle((cx - 0.9, cy - 0.8), 1.8, 1.6,
                              fill=True, facecolor=COLORS["blue"],
                              edgecolor="white", alpha=0.15, lw=0)
        ax.add_patch(rect)
        rect_border = plt.Rectangle((cx - 0.9, cy - 0.8), 1.8, 1.6,
                                     fill=False, edgecolor=COLORS["blue"],
                                     lw=1.5)
        ax.add_patch(rect_border)
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=9,
                color=COLORS["dark_grey"])

    # Arrows
    for x1, x2 in [(1.9, 3.1), (4.9, 6.1)]:
        ax.annotate("", xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle="->", lw=1.5,
                                    color=COLORS["dark_grey"]))

    # Title
    ax.text(5.0, 3.6,
            "Figure 2: ABC Method Overview (replace with detailed diagram)",
            ha="center", va="center", fontsize=10, style="italic",
            color=COLORS["dark_grey"])

    fig.tight_layout()
    savefig(fig, "fig2_method_overview_placeholder", out_dir, fmt)


# ===========================================================================
# Figure 3: ABC vs Standard Debate Accuracy
# ===========================================================================

def figure3(baseline_data: list[dict], out_dir: Path, fmt: str) -> None:
    """Grouped bar chart with bootstrap 95% CI and McNemar tests."""
    print("[Figure 3] ABC vs Baselines Accuracy")

    methods = [
        ("abc_debate",            "ABC Debate"),
        ("standard_debate",       "Standard\nDebate"),
        ("confidence_weighted",   "Confidence\nWeighted"),
        ("self_consistency",      "Self-\nConsistency"),
        ("best_of_n",             "Best-of-N"),
        ("majority_vote",         "Majority\nVote"),
    ]

    # Collect correctness arrays
    correct_arrays: dict[str, np.ndarray] = {}
    for method_key, _ in methods:
        vals = []
        for rec in baseline_data:
            if method_key in rec:
                m = rec[method_key]
                if isinstance(m, dict):
                    val = m.get("final_correct", m.get("correct", False))
                    vals.append(bool(val))
        correct_arrays[method_key] = np.array(vals, dtype=bool)

    n_questions = len(baseline_data)

    # Bootstrap CIs
    means = []
    ci_lowers = []
    ci_uppers = []
    labels = []
    colors = []
    for i, (mk, ml) in enumerate(methods):
        arr = correct_arrays[mk].astype(float)
        m, lo, hi = bootstrap_ci(arr)
        means.append(m)
        ci_lowers.append(m - lo)
        ci_uppers.append(hi - m)
        labels.append(ml)
        colors.append(PALETTE[i % len(PALETTE)])

    means = np.array(means)
    errs = np.array([ci_lowers, ci_uppers])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.2))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=errs, capsize=3, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.65,
                  error_kw={"lw": 0.8, "capthick": 0.8})

    # Annotate accuracy inside bars (to avoid title overlap)
    for xi, mi in zip(x, means):
        ax.text(xi, mi - 0.04,
                f"{mi:.1%}", ha="center", va="top", fontsize=8,
                fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"ABC vs. Baselines ($n$={n_questions})", fontsize=11,
                 pad=8)
    # Set ylim to give headroom for error bars and any significance brackets
    top_val = max(means + errs[1])
    ax.set_ylim(0, min(1.12, top_val + 0.18))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # McNemar comparisons of ABC vs each other method
    abc_correct = correct_arrays["abc_debate"]
    y_annot = top_val + 0.04
    for i, (mk, ml) in enumerate(methods[1:], start=1):
        other_correct = correct_arrays[mk]
        if len(abc_correct) == len(other_correct) and len(abc_correct) > 0:
            p = mcnemar_pvalue(abc_correct, other_correct)
            if p < 0.1:  # only annotate marginally significant or better
                annotate_significance(ax, 0, i, y_annot, p, h=0.012)
                y_annot += 0.05

    fig.tight_layout()
    savefig(fig, "fig3_abc_vs_baselines", out_dir, fmt)


# ===========================================================================
# Figure 4: Multi-Benchmark Generalization
# ===========================================================================

def figure4(multi_data: list[dict], out_dir: Path, fmt: str) -> None:
    """Per-benchmark grouped bar chart: ABC vs Standard Debate."""
    print("[Figure 4] Multi-Benchmark Generalization")

    # Group data by benchmark
    benchmark_results: dict[str, dict[str, list[bool]]] = {}
    for rec in multi_data:
        bm = rec["benchmark"]
        benchmark_results.setdefault(bm, {"abc": [], "standard": []})
        abc_d = rec.get("abc_debate", {})
        std_d = rec.get("standard_debate", {})
        benchmark_results[bm]["abc"].append(
            bool(abc_d.get("final_correct", abc_d.get("correct", False))))
        benchmark_results[bm]["standard"].append(
            bool(std_d.get("final_correct", std_d.get("correct", False))))

    # Sort benchmarks; compute pooled overall
    bm_names = sorted(benchmark_results.keys())

    # If there is only one benchmark, still show it and add the overall
    all_abc = []
    all_std = []
    for bm in bm_names:
        all_abc.extend(benchmark_results[bm]["abc"])
        all_std.extend(benchmark_results[bm]["standard"])

    # Display names
    display_names = {
        "mmlu_pro": "MMLU-Pro",
        "gpqa": "GPQA",
        "arc_challenge": "ARC-C",
        "truthfulqa": "TruthfulQA",
        "gsm8k": "GSM8K",
    }
    labels = [display_names.get(bm, bm) for bm in bm_names]
    labels.append("Overall")

    abc_means = []
    abc_errs_lo = []
    abc_errs_hi = []
    std_means = []
    std_errs_lo = []
    std_errs_hi = []

    for bm in bm_names:
        arr_abc = np.array(benchmark_results[bm]["abc"], dtype=float)
        arr_std = np.array(benchmark_results[bm]["standard"], dtype=float)
        m_a, lo_a, hi_a = bootstrap_ci(arr_abc)
        m_s, lo_s, hi_s = bootstrap_ci(arr_std)
        abc_means.append(m_a)
        abc_errs_lo.append(m_a - lo_a)
        abc_errs_hi.append(hi_a - m_a)
        std_means.append(m_s)
        std_errs_lo.append(m_s - lo_s)
        std_errs_hi.append(hi_s - m_s)

    # Overall
    arr_abc = np.array(all_abc, dtype=float)
    arr_std = np.array(all_std, dtype=float)
    m_a, lo_a, hi_a = bootstrap_ci(arr_abc)
    m_s, lo_s, hi_s = bootstrap_ci(arr_std)
    abc_means.append(m_a)
    abc_errs_lo.append(m_a - lo_a)
    abc_errs_hi.append(hi_a - m_a)
    std_means.append(m_s)
    std_errs_lo.append(m_s - lo_s)
    std_errs_hi.append(hi_s - m_s)

    abc_means = np.array(abc_means)
    std_means = np.array(std_means)
    abc_errs = np.array([abc_errs_lo, abc_errs_hi])
    std_errs = np.array([std_errs_lo, std_errs_hi])

    n_groups = len(labels)
    x = np.arange(n_groups)
    w = 0.32

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))
    ax.bar(x - w / 2, abc_means, w, yerr=abc_errs, capsize=3,
           color=COLORS["blue"], edgecolor="white", linewidth=0.5,
           label="ABC Debate",
           error_kw={"lw": 0.8, "capthick": 0.8})
    ax.bar(x + w / 2, std_means, w, yerr=std_errs, capsize=3,
           color=COLORS["red"], edgecolor="white", linewidth=0.5,
           label="Standard Debate",
           error_kw={"lw": 0.8, "capthick": 0.8})

    # Value annotations inside bars to avoid clipping
    for xi, ma, ms in zip(x, abc_means, std_means):
        ax.text(xi - w / 2, ma - 0.04,
                f"{ma:.1%}", ha="center", va="top", fontsize=7,
                fontweight="bold", color="white")
        ax.text(xi + w / 2, ms - 0.04,
                f"{ms:.1%}", ha="center", va="top", fontsize=7,
                fontweight="bold", color="white")

    # Separator before "Overall"
    if n_groups > 1:
        ax.axvline(n_groups - 1.5, color=COLORS["grey"], ls=":", lw=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Multi-Benchmark Generalization", fontsize=11, pad=8)
    top_val = max(max(abc_means + abc_errs[1]), max(std_means + std_errs[1]))
    ax.set_ylim(0, min(1.12, top_val + 0.12))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    savefig(fig, "fig4_multi_benchmark", out_dir, fmt)


# ===========================================================================
# Figure 5: Ablation Studies (4-panel grid)
# ===========================================================================

def figure5(ablation_data: list[dict] | None, out_dir: Path,
            fmt: str) -> None:
    """Four-panel ablation grid.  Falls back to stub if no data."""
    print("[Figure 5] Ablation Studies")

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.4))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    if ablation_data is not None and len(ablation_data) > 0:
        # Parse ablation records by type
        by_type: dict[str, list[dict]] = {}
        for rec in ablation_data:
            t = rec.get("ablation_type", rec.get("type", "unknown"))
            by_type.setdefault(t, []).append(rec)

        # Panel A: Alpha estimation method comparison
        _ablation_panel_bar(ax_a, by_type.get("alpha_method", []),
                            key="method", val_key="accuracy",
                            title="(A) Alpha Estimation Method",
                            ylabel="Accuracy")

        # Panel B: Number of probes vs accuracy
        _ablation_panel_line(ax_b, by_type.get("n_probes", []),
                             x_key="n_probes", y_key="accuracy",
                             title="(B) Number of Probes",
                             xlabel="Number of probes",
                             ylabel="Accuracy")

        # Panel C: Weighting function comparison
        _ablation_panel_bar(ax_c, by_type.get("weighting_fn", []),
                            key="function", val_key="accuracy",
                            title="(C) Weighting Function",
                            ylabel="Accuracy")

        # Panel D: Social pressure impact
        _ablation_panel_bar(ax_d, by_type.get("social_pressure", []),
                            key="condition", val_key="accuracy",
                            title="(D) Social Pressure Impact",
                            ylabel="Accuracy")
    else:
        # Stub panels
        for ax_i, panel_label in zip(axes.flat,
                                      ["(A) Alpha Estimation Method",
                                       "(B) Number of Probes vs Accuracy",
                                       "(C) Weighting Function",
                                       "(D) Social Pressure Impact"]):
            ax_i.text(0.5, 0.5, f"{panel_label}\n[data pending]",
                      ha="center", va="center", fontsize=9,
                      color=COLORS["grey"], transform=ax_i.transAxes)
            ax_i.set_title(panel_label, fontsize=10)
            ax_i.set_xticks([])
            ax_i.set_yticks([])

    fig.tight_layout()
    savefig(fig, "fig5_ablations", out_dir, fmt)


def _ablation_panel_bar(ax, records: list[dict], key: str, val_key: str,
                         title: str, ylabel: str) -> None:
    """Helper: bar chart for an ablation dimension."""
    if not records:
        ax.text(0.5, 0.5, f"{title}\n[data pending]",
                ha="center", va="center", fontsize=9, color=COLORS["grey"],
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    labels_vals: list[tuple[str, float]] = []
    for r in records:
        labels_vals.append((str(r.get(key, "?")), float(r.get(val_key, 0))))
    labels_vals.sort(key=lambda lv: lv[1], reverse=True)

    xs = np.arange(len(labels_vals))
    vals = [v for _, v in labels_vals]
    lbls = [l for l, _ in labels_vals]

    ax.bar(xs, vals, color=[PALETTE[i % len(PALETTE)]
                            for i in range(len(xs))],
           edgecolor="white", linewidth=0.5, width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(lbls, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))


def _ablation_panel_line(ax, records: list[dict], x_key: str, y_key: str,
                          title: str, xlabel: str, ylabel: str) -> None:
    """Helper: line plot for an ablation dimension."""
    if not records:
        ax.text(0.5, 0.5, f"{title}\n[data pending]",
                ha="center", va="center", fontsize=9, color=COLORS["grey"],
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    points = sorted([(r[x_key], r[y_key]) for r in records])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    ax.plot(xs, ys, "o-", color=COLORS["blue"], markersize=5, lw=1.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))


# ===========================================================================
# Figure 6: Collapse Prevention
# ===========================================================================

def figure6(debate_data: list[dict], out_dir: Path, fmt: str) -> None:
    """Panel A: Collapse rate by alpha group.
       Panel B: Correction rate by alpha group."""
    print("[Figure 6] Collapse Prevention by Alpha Group")

    # Compute mean alpha per debate and assign to groups
    # Alpha groups: low (<0.33), mid (0.33-0.66), high (>=0.66)
    groups: dict[str, dict[str, list[bool]]] = {
        "Low":  {"collapse": [], "correction": []},
        "Mid":  {"collapse": [], "correction": []},
        "High": {"collapse": [], "correction": []},
    }
    group_keys = list(groups.keys())
    group_labels = [
        "Low\n($\\alpha<0.33$)",
        "Mid\n($0.33{\\leq}\\alpha{<}0.66$)",
        "High\n($\\alpha{\\geq}0.66$)",
    ]

    for rec in debate_data:
        # Compute mean alpha (flip-rate) across agents in this debate
        alphas_here = []
        for aid, est in rec.get("alpha_estimates", {}).items():
            alphas_here.append(est["mean_flip_rate"])
        if not alphas_here:
            continue
        mean_alpha = np.mean(alphas_here)

        if mean_alpha < 0.33:
            g = group_keys[0]
        elif mean_alpha < 0.66:
            g = group_keys[1]
        else:
            g = group_keys[2]

        outcome = rec.get("outcome", {})
        groups[g]["collapse"].append(outcome.get("collapse", False))
        groups[g]["correction"].append(outcome.get("correction", False))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    x = np.arange(len(group_keys))
    w = 0.55

    # Panel A: Collapse rate
    collapse_rates = []
    collapse_errs_lo = []
    collapse_errs_hi = []
    for g in group_keys:
        arr = np.array(groups[g]["collapse"], dtype=float)
        if len(arr) > 0:
            m, lo, hi = bootstrap_ci(arr)
        else:
            m, lo, hi = 0, 0, 0
        collapse_rates.append(m)
        collapse_errs_lo.append(m - lo)
        collapse_errs_hi.append(hi - m)

    ax_a.bar(x, collapse_rates, w,
             yerr=np.array([collapse_errs_lo, collapse_errs_hi]),
             capsize=4, color=[COLORS["red"], COLORS["yellow"],
                               COLORS["green"]],
             edgecolor="white", linewidth=0.5,
             error_kw={"lw": 0.8, "capthick": 0.8})
    for xi, m in zip(x, collapse_rates):
        n_g = len(groups[group_keys[int(xi)]]["collapse"])
        ax_a.text(xi, m + collapse_errs_hi[int(xi)] + 0.005,
                  f"{m:.1%}\n($n$={n_g})", ha="center", va="bottom",
                  fontsize=7)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(group_labels, fontsize=8)
    ax_a.set_ylabel("Collapse Rate")
    ax_a.set_title("(A) Collapse Rate by Revisability", fontsize=11)
    ax_a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_a.set_ylim(0, max(collapse_rates) * 2.0 + 0.02)

    # Panel B: Correction rate
    correction_rates = []
    correction_errs_lo = []
    correction_errs_hi = []
    for g in group_keys:
        arr = np.array(groups[g]["correction"], dtype=float)
        if len(arr) > 0:
            m, lo, hi = bootstrap_ci(arr)
        else:
            m, lo, hi = 0, 0, 0
        correction_rates.append(m)
        correction_errs_lo.append(m - lo)
        correction_errs_hi.append(hi - m)

    ax_b.bar(x, correction_rates, w,
             yerr=np.array([correction_errs_lo, correction_errs_hi]),
             capsize=4, color=[COLORS["red"], COLORS["yellow"],
                               COLORS["green"]],
             edgecolor="white", linewidth=0.5,
             error_kw={"lw": 0.8, "capthick": 0.8})
    for xi, m in zip(x, correction_rates):
        n_g = len(groups[group_keys[int(xi)]]["correction"])
        ax_b.text(xi, m + correction_errs_hi[int(xi)] + 0.005,
                  f"{m:.1%}\n($n$={n_g})", ha="center", va="bottom",
                  fontsize=7)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(group_labels, fontsize=8)
    ax_b.set_ylabel("Correction Rate")
    ax_b.set_title("(B) Correction Rate by Revisability", fontsize=11)
    ax_b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_b.set_ylim(0, max(correction_rates) * 1.5 + 0.02)

    fig.tight_layout()
    savefig(fig, "fig6_collapse_prevention", out_dir, fmt)


# ===========================================================================
# Table 1: Main Results (LaTeX)
# ===========================================================================

def table1(baseline_data: list[dict] | None,
           multi_data: list[dict] | None,
           out_dir: Path) -> None:
    """Generate a LaTeX table of main results."""
    print("[Table 1] Main Results (LaTeX)")

    methods = [
        ("abc_debate",          "ABC Debate (ours)"),
        ("standard_debate",     "Standard Debate"),
        ("confidence_weighted", "Confidence Weighted"),
        ("self_consistency",    "Self-Consistency"),
        ("best_of_n",           "Best-of-N"),
        ("majority_vote",       "Majority Vote"),
    ]
    benchmarks = ["mmlu_pro", "gpqa", "arc_challenge", "truthfulqa", "gsm8k"]
    bm_display = {
        "mmlu_pro":      "MMLU-Pro",
        "gpqa":          "GPQA",
        "arc_challenge": "ARC-C",
        "truthfulqa":    "TruthfulQA",
        "gsm8k":         "GSM8K",
    }

    # Build per-benchmark, per-method correctness arrays
    # From block1 (has all methods, may be single benchmark)
    results: dict[str, dict[str, list[bool]]] = {}
    all_sources = []
    if baseline_data:
        all_sources.extend(baseline_data)
    if multi_data:
        all_sources.extend(multi_data)

    for rec in all_sources:
        bm = rec.get("benchmark", "unknown")
        for mk, _ in methods:
            if mk in rec:
                m = rec[mk]
                val = m.get("final_correct", m.get("correct", False))
                results.setdefault(mk, {}).setdefault(bm, []).append(
                    bool(val)
                )

    # Also collect "Overall" across all benchmarks
    for mk, _ in methods:
        if mk in results:
            all_correct: list[bool] = []
            for bm_list in results[mk].values():
                all_correct.extend(bm_list)
            results[mk]["_overall"] = all_correct

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results: accuracy (\%) with standard "
                 r"errors across benchmarks.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\small")

    # Determine which benchmarks actually have data
    active_bm = []
    for bm in benchmarks:
        for mk, _ in methods:
            if bm in results.get(mk, {}):
                active_bm.append(bm)
                break
    col_spec = "l" + "c" * (len(active_bm) + 1)  # +1 for Overall
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = "Method"
    for bm in active_bm:
        header += f" & {bm_display.get(bm, bm)}"
    header += r" & Overall \\"
    lines.append(header)
    lines.append(r"\midrule")

    for mk, ml in methods:
        row = ml
        bold_prefix = r"\textbf{" if mk == "abc_debate" else ""
        bold_suffix = "}" if mk == "abc_debate" else ""
        for bm in active_bm + ["_overall"]:
            arr = results.get(mk, {}).get(bm, [])
            if arr:
                arr_np = np.array(arr, dtype=float)
                m = arr_np.mean() * 100
                se = arr_np.std(ddof=1) / np.sqrt(len(arr_np)) * 100 \
                    if len(arr_np) > 1 else 0
                cell = f"{bold_prefix}{m:.1f}{bold_suffix}"
                if se > 0:
                    cell += f"$\\pm${se:.1f}"
            else:
                cell = "--"
            row += f" & {cell}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = out_dir / "table1_main_results.tex"
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Saved table1_main_results.tex")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NeurIPS publication-quality figures for ABC "
                    "experiment.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures "
                             "(default: abc_exp/results/figures/)")
    parser.add_argument("--format", type=str, default="both",
                        choices=["pdf", "png", "both"],
                        help="Output format (default: both)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory (default: abc_exp/results/)")
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # abc_exp/

    results_dir = Path(args.results_dir) if args.results_dir else \
        project_root / "results"
    out_dir = Path(args.output_dir) if args.output_dir else \
        results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory : {results_dir}")
    print(f"Output directory  : {out_dir}")
    print(f"Output format     : {args.format}")
    print()

    # Apply NeurIPS style
    plt.rcParams.update(NEURIPS_RC)

    # ------------------------------------------------------------------
    # Load data (gracefully handle missing files)
    # ------------------------------------------------------------------
    alpha_data = try_load(results_dir / "block0_alpha.jsonl")
    retest_data = try_load(results_dir / "block0_alpha_retest.jsonl")
    debate_data = try_load(results_dir / "block0_debates.jsonl")
    summary = try_load(results_dir / "block0_analysis_summary.json",
                       loader=load_json)
    baseline_data = try_load(results_dir / "block1_abc_vs_baselines.jsonl")
    multi_data = try_load(results_dir / "block3_multi_benchmark.jsonl")
    ablation_data = try_load(results_dir / "block4_ablations.jsonl")

    generated = 0

    # ------------------------------------------------------------------
    # Figure 1: Alpha Distribution and Reliability
    # ------------------------------------------------------------------
    if alpha_data and retest_data and summary:
        figure1(alpha_data, retest_data, summary, out_dir, args.format)
        generated += 1
    else:
        print("[Figure 1] SKIPPED -- missing block0_alpha.jsonl, "
              "block0_alpha_retest.jsonl, or block0_analysis_summary.json")

    # ------------------------------------------------------------------
    # Figure 2: Method Overview (always generated -- placeholder)
    # ------------------------------------------------------------------
    figure2(out_dir, args.format)
    generated += 1

    # ------------------------------------------------------------------
    # Figure 3: ABC vs Baselines
    # ------------------------------------------------------------------
    if baseline_data:
        figure3(baseline_data, out_dir, args.format)
        generated += 1
    else:
        print("[Figure 3] SKIPPED -- missing block1_abc_vs_baselines.jsonl")

    # ------------------------------------------------------------------
    # Figure 4: Multi-Benchmark
    # ------------------------------------------------------------------
    if multi_data:
        figure4(multi_data, out_dir, args.format)
        generated += 1
    else:
        print("[Figure 4] SKIPPED -- missing block3_multi_benchmark.jsonl")

    # ------------------------------------------------------------------
    # Figure 5: Ablation Studies
    # ------------------------------------------------------------------
    figure5(ablation_data, out_dir, args.format)
    generated += 1

    # ------------------------------------------------------------------
    # Figure 6: Collapse Prevention
    # ------------------------------------------------------------------
    if debate_data:
        figure6(debate_data, out_dir, args.format)
        generated += 1
    else:
        print("[Figure 6] SKIPPED -- missing block0_debates.jsonl")

    # ------------------------------------------------------------------
    # Table 1: Main Results
    # ------------------------------------------------------------------
    if baseline_data or multi_data:
        table1(baseline_data, multi_data, out_dir)
        generated += 1
    else:
        print("[Table 1] SKIPPED -- no baseline or multi-benchmark data")

    print(f"\nDone. Generated {generated} figures/tables in {out_dir}")


if __name__ == "__main__":
    main()
