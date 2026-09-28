"""Zero-API family-level partial Spearman check for the N=14 cohort.

Reviewer concern: the paper reports a model-row partial Spearman after
residualizing alpha_tot and C^cond on initial-majority accuracy, but the
headline inferential unit is the model family. This script recomputes the
analogous check after collapsing the realized N=14 table to G=7 family means.

Output:
  abc_exp/results/family_partial_spearman.json
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "abc_exp" / "results"
SOURCE = RES / "reviewer_hardening_analyses.json"
OUT_JSON = RES / "family_partial_spearman.json"


def load_rows() -> list[dict[str, Any]]:
    data = json.loads(SOURCE.read_text())
    rows = data["per_model"]
    required = {"model", "family", "alpha_tot", "c_cond_pct", "init_acc"}
    missing = [r.get("model", "<unknown>") for r in rows if not required <= set(r)]
    if missing:
        raise ValueError(f"rows missing required keys: {missing}")
    if len(rows) != 14:
        raise ValueError(f"expected N=14 source rows, found {len(rows)}")
    return rows


def family_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)

    out = []
    for family in sorted(by_family):
        vals = by_family[family]
        out.append(
            {
                "family": family,
                "n_models": len(vals),
                "models": [str(v["model"]) for v in vals],
                "alpha_tot": float(np.mean([float(v["alpha_tot"]) for v in vals])),
                "c_cond_pct": float(np.mean([float(v["c_cond_pct"]) for v in vals])),
                "init_acc": float(np.mean([float(v["init_acc"]) for v in vals])),
            }
        )
    return out


def rank_residual_partial(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> dict[str, Any]:
    """Partial Spearman via Pearson correlation of rank residuals."""

    rx = stats.rankdata(x, method="average")
    ry = stats.rankdata(y, method="average")
    rz = np.column_stack([stats.rankdata(controls[:, j], method="average") for j in range(controls.shape[1])])
    design = np.column_stack([np.ones(len(x)), rz])

    beta_x = np.linalg.lstsq(design, rx, rcond=None)[0]
    beta_y = np.linalg.lstsq(design, ry, rcond=None)[0]
    resid_x = rx - design @ beta_x
    resid_y = ry - design @ beta_y

    rho = float(np.corrcoef(resid_x, resid_y)[0, 1])
    df = len(x) - controls.shape[1] - 2
    if abs(rho) >= 1:
        t_stat = math.copysign(math.inf, rho)
        p_two = 0.0
    else:
        t_stat = rho * math.sqrt(df / (1 - rho * rho))
        p_two = float(2 * stats.t.sf(abs(t_stat), df))

    return {
        "rho": rho,
        "df": df,
        "t_statistic": float(t_stat),
        "p_two_sided_t_approx": p_two,
        "rank_alpha": [float(v) for v in rx],
        "rank_c_cond": [float(v) for v in ry],
        "rank_controls": [[float(v) for v in row] for row in rz],
        "residual_alpha": [float(v) for v in resid_x],
        "residual_c_cond": [float(v) for v in resid_y],
    }


def exact_partial_permutation(x: np.ndarray, y: np.ndarray, controls: np.ndarray, observed: float) -> dict[str, Any]:
    null = []
    for permuted_y in itertools.permutations(y):
        stat = rank_residual_partial(x, np.asarray(permuted_y, dtype=float), controls)["rho"]
        null.append(float(stat))
    ge = sum(v >= observed - 1e-12 for v in null)
    two = sum(abs(v) >= abs(observed) - 1e-12 for v in null)
    return {
        "n_permutations": len(null),
        "one_sided_ge_count": ge,
        "p_one_sided_positive_exact": ge / len(null),
        "two_sided_abs_count": two,
        "p_two_sided_exact": two / len(null),
        "null_min": min(null),
        "null_max": max(null),
    }


def main() -> None:
    rows = load_rows()
    families = family_means(rows)
    x = np.asarray([r["alpha_tot"] for r in families], dtype=float)
    y = np.asarray([r["c_cond_pct"] for r in families], dtype=float)
    z = np.asarray([[r["init_acc"]] for r in families], dtype=float)

    marginal = stats.spearmanr(x, y)
    partial = rank_residual_partial(x, y, z)
    exact = exact_partial_permutation(x, y, z, partial["rho"])

    source_data = json.loads(SOURCE.read_text())
    out = {
        "description": "Family-level partial Spearman after collapsing the realized N=14 model table to G=7 family means.",
        "source": "abc_exp/results/reviewer_hardening_analyses.json:per_model",
        "aggregation": "simple mean over model rows within each family, matching the headline family aggregation",
        "controls": ["family_mean_initial_majority_accuracy"],
        "family_rows": families,
        "family_marginal_spearman": {
            "n": len(families),
            "rho": float(marginal.statistic),
            "p_two_sided_scipy": float(marginal.pvalue),
        },
        "family_partial_spearman": partial,
        "family_partial_exact_permutation": exact,
        "model_row_partial_for_comparison": source_data["partial_spearman"]["alpha_controlling_init_acc"],
        "p_value_note": "Exact permutation permutes family-level C^cond values while holding family alpha_tot and initial-majority accuracy fixed, then recomputes the rank-residualized partial Spearman for all 7! assignments.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")

    print(
        "family partial rho="
        f"{partial['rho']:+.4f}, exact p2={exact['p_two_sided_exact']:.6f}, "
        f"p1={exact['p_one_sided_positive_exact']:.6f}; wrote {OUT_JSON}"
    )


if __name__ == "__main__":
    main()
