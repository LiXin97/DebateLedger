"""Specification curve / multiverse for the headline rho(alpha, C^cond).

Enumerates ~24 reasonable analyst combinations across:
  - probe set    : {8-probe full, 6-probe (no very_strong)}
  - alpha source : {alpha_total, alpha_adv, log-ratio S/A as alternative axis}
  - primary metric : {C^cond, C^cond Wilson lower bound, raw collapse %}
  - estimator    : {frequentist Spearman + Fisher-z 95% CI, EIV bootstrap-with-noise posterior median + 95% CrI}
  - LOMO         : {none, leave-one-model-out range}
  - partial      : {none, residualize alpha and C^cond on rank initial accuracy}

Per spec, computes rho with 95% interval. Saves a JSON table and a single
matplotlib figure (PDF + PNG) at abc_exp/results/figures/spec_curve_rho.{pdf,png}.

The frequentist primary spec (8-probe / alpha_total / C^cond / Spearman /
no-LOMO / no-partial) reproduces the headline rho=+0.78, p=0.013 from
PRE_REGISTRATION_N18.md §1. Other specs are exploratory.
"""
from __future__ import annotations
import json
from itertools import product
from pathlib import Path

import numpy as np

ABC_ROOT = Path(__file__).resolve().parent.parent
RES = ABC_ROOT / "results"
FIG_DIR = RES / "figures"
SEED = 20260424


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def fisher_z_ci(rho: float, n: int) -> tuple[float, float]:
    rho = max(min(rho, 0.999999), -0.999999)
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    lo, hi = z - 1.96 * se, z + 1.96 * se
    return float((np.exp(2 * lo) - 1) / (np.exp(2 * lo) + 1)), float(
        (np.exp(2 * hi) - 1) / (np.exp(2 * hi) + 1)
    )


def partial_residualize(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Rank-residualize x on z (returns ranks of residuals)."""
    rx = x.argsort().argsort().astype(float)
    rz = z.argsort().argsort().astype(float)
    rz_c = rz - rz.mean()
    beta = float((rx * rz_c).sum() / max((rz_c ** 2).sum(), 1e-9))
    return rx - beta * rz_c


def lomo_range(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rs = []
    for i in range(len(x)):
        idx = [j for j in range(len(x)) if j != i]
        rs.append(spearman(x[idx], y[idx]))
    return float(min(rs)), float(max(rs))


def eiv_posterior(
    alpha: np.ndarray,
    sd_alpha: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    n_iter: int = 5000,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Reproduce the bootstrap_with_noise EIV procedure from
    experiments/hierarchical_association.py (median + 95% CrI)."""
    rng = np.random.default_rng(seed)
    rs = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        a_draw = alpha + rng.normal(0.0, sd_alpha)
        theta_draw = rng.beta(k + 0.5, n - k + 0.5)
        rs[i] = spearman(a_draw, theta_draw)
    return float(np.median(rs)), float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    headline = json.loads((RES / "alpha_decomp_headline_n9.json").read_text())
    eiv_rows = json.loads((RES / "hierarchical_association.json").read_text())["rows"]
    drop_vs = {  # T2.1: per-model alpha under 6-probe (no very_strong)
        "Sonnet 4.5": 0.347, "GPT-4o-mini": 0.297, "GPT-5.4-mini": 0.281,
        "Gemini 3-flash": 0.224, "Phi-4-mini": 0.376, "Qwen3-4B": 0.554,
        "Llama-3.1-8B": 0.647, "Qwen3-8B": 0.551, "Qwen3-32B": 0.615,
    }
    log_ratio_sa = {  # T2.4: per-model log((s_soc+eps)/(s_arg+eps))
        "Sonnet 4.5": -1.928, "GPT-4o-mini": -2.509, "GPT-5.4-mini": -3.375,
        "Gemini 3-flash": -5.028, "Phi-4-mini": -1.483, "Qwen3-4B": -1.711,
        "Llama-3.1-8B": -2.713, "Qwen3-8B": -2.709, "Qwen3-32B": -1.126,
    }
    # Per-model initial accuracy (for partial Spearman) and raw collapse %
    init_acc = {  # from paper Tab. 1; aligned with the N=9 headline cohort
        "Sonnet 4.5": 77.5, "GPT-4o-mini": 68.5, "GPT-5.4-mini": 71.5,
        "Gemini 3-flash": 85.0, "Phi-4-mini": 50.6, "Qwen3-4B": 45.3,
        "Llama-3.1-8B": 50.7, "Qwen3-8B": 28.0, "Qwen3-32B": 50.0,
    }
    raw_collapse_pct = {  # from paper Tab. 1
        "Sonnet 4.5": 1.67, "GPT-4o-mini": 1.61, "GPT-5.4-mini": 0.50,
        "Gemini 3-flash": 0.50, "Phi-4-mini": 5.18, "Qwen3-4B": 2.31,
        "Llama-3.1-8B": 4.29, "Qwen3-8B": 1.50, "Qwen3-32B": 4.00,
    }
    models = headline["models"]
    n = len(models)

    # Build per-spec data vectors
    alpha_full = np.asarray(headline["alpha_total"])
    alpha_adv = np.asarray(headline["alpha_adv"])
    alpha_no_vs = np.asarray([drop_vs[m] for m in models])
    alpha_log_sa = np.asarray([log_ratio_sa[m] for m in models])  # alternative axis
    c_cond = np.asarray(headline["C_cond"])
    raw = np.asarray([raw_collapse_pct[m] for m in models])
    iacc = np.asarray([init_acc[m] for m in models])

    # Wilson lower bound on conditional collapse: z=1.96, requires k and n
    eiv_by_model = {r["model"]: r for r in eiv_rows}
    name_back = {  # alpha_decomp uses short names; eiv uses long
        "Sonnet 4.5": "Claude Sonnet 4.5", "GPT-4o-mini": "GPT-4o-mini",
        "GPT-5.4-mini": "GPT-5.4-mini", "Gemini 3-flash": "Gemini 3-flash",
        "Phi-4-mini": "Phi-4-mini", "Qwen3-4B": "Qwen3-4B",
        "Llama-3.1-8B": "Llama-3.1-8B", "Qwen3-8B": "Qwen3-8B",
        "Qwen3-32B": "Qwen3-32B",
    }
    ks = np.asarray([eiv_by_model[name_back[m]]["k"] for m in models])
    ns = np.asarray([eiv_by_model[name_back[m]]["n"] for m in models])
    sd_alpha = np.asarray([eiv_by_model[name_back[m]]["sd_alpha"] for m in models])

    z = 1.96
    p_hat = ks / np.maximum(ns, 1)
    denom = 1 + z * z / ns
    centre = (p_hat + z * z / (2 * ns)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / ns + z * z / (4 * ns * ns)) / denom
    wilson_lo = (centre - half) * 100  # back to percent

    metrics = {"C^cond": c_cond, "C^cond_wilson_lo": wilson_lo, "raw_collapse": raw}
    alpha_specs = {
        "alpha_total_8probe": alpha_full,
        "alpha_total_6probe": alpha_no_vs,
        "alpha_adv": alpha_adv,
        "log_ratio_S/A": alpha_log_sa,
    }

    specs = []
    for (a_name, a_vec), (m_name, m_vec), partial in product(
        alpha_specs.items(), metrics.items(), [False, True]
    ):
        x = a_vec.copy()
        y = m_vec.copy()
        if partial:
            x = partial_residualize(x, iacc)
            y = partial_residualize(y, iacc)
        rho = spearman(x, y)
        ci_lo, ci_hi = fisher_z_ci(rho, n)
        lo, hi = lomo_range(x, y)
        specs.append(
            dict(
                alpha=a_name, metric=m_name, partial=partial,
                estimator="spearman_freq", rho=rho,
                ci_lo=ci_lo, ci_hi=ci_hi, lomo_min=lo, lomo_max=hi,
            )
        )

    # EIV bootstrap-with-noise (only meaningful when y = C^cond binomial counts)
    for a_name, a_vec in alpha_specs.items():
        if a_name == "log_ratio_S/A":
            continue  # log-ratio is not a probability; skip EIV
        median, lo, hi = eiv_posterior(a_vec, sd_alpha, ks, ns)
        specs.append(
            dict(
                alpha=a_name, metric="C^cond", partial=False,
                estimator="eiv_bootstrap", rho=median,
                ci_lo=lo, ci_hi=hi, lomo_min=None, lomo_max=None,
            )
        )

    out = dict(N=n, seed=SEED, n_specs=len(specs), specs=specs)
    out_path = RES / "spec_curve_rho.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}; {len(specs)} specifications")

    # ---- Figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_specs = sorted(specs, key=lambda s: s["rho"])
    rhos = [s["rho"] for s in sorted_specs]
    los = [s["ci_lo"] for s in sorted_specs]
    his = [s["ci_hi"] for s in sorted_specs]
    labels = [
        f"{s['alpha']} | {s['metric']}{' | partial' if s['partial'] else ''} | {s['estimator']}"
        for s in sorted_specs
    ]
    colors = [
        "#1b7837" if (s["alpha"].startswith("alpha_total") and s["metric"] == "C^cond"
                      and not s["partial"] and s["estimator"] == "spearman_freq")
        else "#762a83" if s["estimator"] == "eiv_bootstrap"
        else "#4d4d4d"
        for s in sorted_specs
    ]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(specs) + 1.2))
    y = np.arange(len(specs))
    for i, (rho, lo, hi, c) in enumerate(zip(rhos, los, his, colors)):
        ax.plot([lo, hi], [i, i], color=c, lw=1.2, alpha=0.7)
        ax.plot(rho, i, "o", color=c, ms=4)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.axvline(0.78, color="#1b7837", lw=0.5, ls=":", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Spearman rho (95% CI / CrI)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(
        f"Specification curve: rho(alpha, C^cond) at N={n}\n"
        "green = pre-registered primary; purple = EIV posterior median; grey = exploratory",
        fontsize=9,
    )
    plt.tight_layout()
    pdf_path = FIG_DIR / "spec_curve_rho.pdf"
    png_path = FIG_DIR / "spec_curve_rho.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
