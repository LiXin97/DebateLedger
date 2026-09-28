"""Zero-API leave-one-Qwen-row sensitivity for the N=14 alpha/Ccond view.

Reviewer concern: the N=14 model-row sensitivity is Qwen-heavy. This script
removes each Qwen row in turn, recomputes the model-row Spearman association
between alpha_tot and C^cond on N=13, and also recomputes family-level mean and
median aggregation as a design-aware companion.

Output:
  abc_exp/results/leave_one_qwen_row_sensitivity.json
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "abc_exp" / "results"
SOURCE = RES / "reviewer_hardening_analyses.json"
OUT_JSON = RES / "leave_one_qwen_row_sensitivity.json"


def load_rows() -> list[dict[str, Any]]:
    data = json.loads(SOURCE.read_text())
    rows = data["per_model"]
    required = {"model", "family", "alpha_tot", "c_cond_pct"}
    missing = [r.get("model", "<unknown>") for r in rows if not required <= set(r)]
    if missing:
        raise ValueError(f"rows missing required keys: {missing}")
    if len(rows) != 14:
        raise ValueError(f"expected N=14 source rows, found {len(rows)}")
    return rows


def spearman_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = [float(r["alpha_tot"]) for r in rows]
    y = [float(r["c_cond_pct"]) for r in rows]
    result = stats.spearmanr(x, y)
    rho = float(result.statistic)
    p_two = float(result.pvalue)
    return {
        "n": len(rows),
        "rho": rho,
        "p_two_sided_asymptotic": p_two,
        "p_one_sided_positive_asymptotic": p_two / 2 if rho >= 0 else 1 - p_two / 2,
    }


def aggregate_by_family(rows: list[dict[str, Any]], reducer_name: str) -> list[dict[str, Any]]:
    reducer = {"mean": mean, "median": median}[reducer_name]
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
                "alpha_tot": float(reducer(float(v["alpha_tot"]) for v in vals)),
                "c_cond_pct": float(reducer(float(v["c_cond_pct"]) for v in vals)),
            }
        )
    return out


def exact_family_spearman(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = [float(r["alpha_tot"]) for r in rows]
    y = [float(r["c_cond_pct"]) for r in rows]
    observed = float(stats.spearmanr(x, y).statistic)
    null = [float(stats.spearmanr(x, list(p)).statistic) for p in itertools.permutations(y)]
    ge = sum(v >= observed - 1e-12 for v in null)
    two = sum(abs(v) >= abs(observed) - 1e-12 for v in null)
    return {
        "n": len(rows),
        "rho": observed,
        "n_permutations": len(null),
        "one_sided_ge_count": ge,
        "p_one_sided_positive_exact": ge / len(null),
        "two_sided_abs_count": two,
        "p_two_sided_exact": two / len(null),
    }


def main() -> None:
    rows = load_rows()
    qwen_rows = [r for r in rows if r["family"] == "Qwen"]
    if not qwen_rows:
        raise ValueError("no Qwen rows found in N=14 source")

    baseline_family_mean = aggregate_by_family(rows, "mean")
    baseline_family_median = aggregate_by_family(rows, "median")

    leave_one = []
    for dropped in qwen_rows:
        kept = [r for r in rows if r["model"] != dropped["model"]]
        family_mean = aggregate_by_family(kept, "mean")
        family_median = aggregate_by_family(kept, "median")
        leave_one.append(
            {
                "dropped_model": dropped["model"],
                "dropped_alpha_tot": float(dropped["alpha_tot"]),
                "dropped_c_cond_pct": float(dropped["c_cond_pct"]),
                "model_row_spearman": spearman_summary(kept),
                "family_mean_exact_spearman": exact_family_spearman(family_mean),
                "family_median_exact_spearman": exact_family_spearman(family_median),
                "family_mean_rows": family_mean,
            }
        )

    model_rhos = [r["model_row_spearman"]["rho"] for r in leave_one]
    family_mean_rhos = [r["family_mean_exact_spearman"]["rho"] for r in leave_one]
    family_median_rhos = [r["family_median_exact_spearman"]["rho"] for r in leave_one]
    out = {
        "description": "Leave-one-Qwen-row N=14 sensitivity for alpha_tot versus conditional collapse Ccond.",
        "source": "abc_exp/results/reviewer_hardening_analyses.json:per_model",
        "n_source_rows": len(rows),
        "n_qwen_rows": len(qwen_rows),
        "baseline": {
            "model_row_spearman": spearman_summary(rows),
            "family_mean_exact_spearman": exact_family_spearman(baseline_family_mean),
            "family_median_exact_spearman": exact_family_spearman(baseline_family_median),
        },
        "leave_one_qwen_row": leave_one,
        "summary": {
            "model_row_rho_min": min(model_rhos),
            "model_row_rho_max": max(model_rhos),
            "model_row_all_positive": all(r > 0 for r in model_rhos),
            "family_mean_rho_min": min(family_mean_rhos),
            "family_mean_rho_max": max(family_mean_rhos),
            "family_median_rho_min": min(family_median_rhos),
            "family_median_rho_max": max(family_median_rhos),
            "worst_model_row_drop": min(leave_one, key=lambda r: r["model_row_spearman"]["rho"])["dropped_model"],
            "worst_family_mean_drop": min(leave_one, key=lambda r: r["family_mean_exact_spearman"]["rho"])["dropped_model"],
        },
        "p_value_note": "Model-row p-values use scipy.stats.spearmanr asymptotic p-values, matching the existing N=14 reviewer-hardening artifact. Family-level p-values enumerate all 7! permutations after each row removal.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")

    print(f"baseline N=14 rho={out['baseline']['model_row_spearman']['rho']:+.4f}")
    for row in leave_one:
        stats_row = row["model_row_spearman"]
        fam_row = row["family_mean_exact_spearman"]
        print(
            f"drop {row['dropped_model']}: "
            f"N=13 rho={stats_row['rho']:+.4f}, p2={stats_row['p_two_sided_asymptotic']:.6f}; "
            f"family-mean rho={fam_row['rho']:+.4f}, exact p2={fam_row['p_two_sided_exact']:.6f}"
        )
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
