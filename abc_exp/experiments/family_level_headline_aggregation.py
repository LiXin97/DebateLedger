"""Family-level aggregation for the N=14 headline alpha/collapse table.

This is a zero-cost reviewer diagnostic: collapse the per-model D5.2 headline
table to one row per model family, then recompute Spearman rho on family-level
mean and median aggregates of alpha_tot and C^cond.

Outputs:
  - results/family_level_headline_aggregation.json
  - results/FAMILY_LEVEL_HEADLINE_AGGREGATION.md

Run:
  python -m abc_exp.experiments.family_level_headline_aggregation
"""

from __future__ import annotations

import json
import re
import itertools
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SOURCE = RES / "D5.2_gate_decision.md"
OUT_JSON = RES / "family_level_headline_aggregation.json"
OUT_MD = RES / "FAMILY_LEVEL_HEADLINE_AGGREGATION.md"


TABLE_ROW_RE = re.compile(
    r"^(?P<model>\S+)\s+"
    r"(?P<family>\S+)\s+"
    r"(?P<alpha>[0-9.]+)\s+"
    r"(?P<ccond>[0-9.]+)%\s+"
    r"(?P<wilson>[0-9.]+)"
)


def parse_headline_table(path: Path) -> list[dict[str, object]]:
    """Parse the fenced per-model table in D5.2_gate_decision.md."""
    rows: list[dict[str, object]] = []
    in_target_section = False
    in_code_block = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        if line.startswith("##") and "Per-Model Inputs" in line:
            in_target_section = True
            continue
        if in_target_section and line.startswith("##") and "Sensitivity Sweep" in line:
            break
        if not in_target_section:
            continue
        if line.strip() == "```":
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            continue

        match = TABLE_ROW_RE.match(line.strip())
        if not match:
            continue

        rows.append(
            {
                "model": match.group("model"),
                "family": match.group("family"),
                "alpha_tot": float(match.group("alpha")),
                "c_cond_pct": float(match.group("ccond")),
                "wilson_hw": float(match.group("wilson")),
            }
        )

    if len(rows) != 14:
        raise ValueError(f"expected 14 headline rows in {path}, parsed {len(rows)}")
    return rows


def spearman_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    alpha = [float(r["alpha_tot"]) for r in rows]
    ccond = [float(r["c_cond_pct"]) for r in rows]
    result = spearmanr(alpha, ccond)
    rho = float(result.statistic)
    scipy_p_two_sided = float(result.pvalue)

    null = [float(spearmanr(alpha, list(p)).statistic) for p in itertools.permutations(ccond)]
    ge = sum(v >= rho - 1e-12 for v in null)
    two = sum(abs(v) >= abs(rho) - 1e-12 for v in null)
    return {
        "n": len(rows),
        "rho": rho,
        "n_permutations": len(null),
        "one_sided_ge_count": ge,
        "p_one_sided_positive_exact": ge / len(null),
        "two_sided_abs_count": two,
        "p_two_sided_exact": two / len(null),
        "p_two_sided_scipy": scipy_p_two_sided,
        "p_one_sided_positive_scipy": scipy_p_two_sided / 2 if rho >= 0 else 1 - scipy_p_two_sided / 2,
    }


def aggregate_by_family(rows: list[dict[str, object]], reducer_name: str) -> list[dict[str, object]]:
    reducer = {"mean": mean, "median": median}[reducer_name]
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)

    out = []
    for family in sorted(by_family):
        fam_rows = by_family[family]
        out.append(
            {
                "family": family,
                "n_models": len(fam_rows),
                "models": [str(r["model"]) for r in fam_rows],
                "alpha_tot": float(reducer(float(r["alpha_tot"]) for r in fam_rows)),
                "c_cond_pct": float(reducer(float(r["c_cond_pct"]) for r in fam_rows)),
            }
        )
    return out


def leave_one_family(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    families = sorted({str(r["family"]) for r in rows})
    out = []
    for family in families:
        kept = [r for r in rows if r["family"] != family]
        agg = aggregate_by_family(kept, "mean")
        summary = spearman_summary(agg)
        out.append(
            {
                "dropped_family": family,
                "n_families_remaining": summary["n"],
                "rho": summary["rho"],
                "p_two_sided_exact": summary["p_two_sided_exact"],
                "p_one_sided_positive_exact": summary["p_one_sided_positive_exact"],
            }
        )
    return out


def write_markdown(out: dict[str, object]) -> None:
    mean_stats = out["aggregations"]["mean"]["spearman"]  # type: ignore[index]
    median_stats = out["aggregations"]["median"]["spearman"]  # type: ignore[index]
    lofo = out["leave_one_family_mean"]  # type: ignore[assignment]
    worst = min(lofo, key=lambda r: r["rho"])  # type: ignore[index]

    lines = [
        "# Family-Level Headline Aggregation",
        "",
        "Zero-cost diagnostic computed from `D5.2_gate_decision.md` section 1. The N=14 per-model table is collapsed to one row per family before recomputing Spearman association between `alpha_tot` and `C^cond`.",
        "",
        "| Aggregation | G | Spearman rho | p (two-sided) | p (one-sided, positive) |",
        "|---|---:|---:|---:|---:|",
        f"| Family mean | {mean_stats['n']} | {mean_stats['rho']:+.4f} | {mean_stats['p_two_sided_exact']:.4f} | {mean_stats['p_one_sided_positive_exact']:.4f} |",
        f"| Family median | {median_stats['n']} | {median_stats['rho']:+.4f} | {median_stats['p_two_sided_exact']:.4f} | {median_stats['p_one_sided_positive_exact']:.4f} |",
        "",
        "## Family Aggregates",
        "",
        "| Family | n models | mean alpha_tot | mean C^cond | median alpha_tot | median C^cond |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    mean_rows = {r["family"]: r for r in out["aggregations"]["mean"]["families"]}  # type: ignore[index]
    median_rows = {r["family"]: r for r in out["aggregations"]["median"]["families"]}  # type: ignore[index]
    for family in sorted(mean_rows):
        mr = mean_rows[family]
        dr = median_rows[family]
        lines.append(
            f"| {family} | {mr['n_models']} | {mr['alpha_tot']:.4f} | {mr['c_cond_pct']:.4f}% | "
            f"{dr['alpha_tot']:.4f} | {dr['c_cond_pct']:.4f}% |"
        )

    lines.extend(
        [
            "",
            "## Leave-One-Family Sensitivity (Family Mean)",
            "",
            "| Dropped family | G remaining | rho | p (two-sided) | p (one-sided, positive) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in lofo:  # type: ignore[assignment]
        lines.append(
            f"| {row['dropped_family']} | {row['n_families_remaining']} | {row['rho']:+.4f} | "
            f"{row['p_two_sided_exact']:.4f} | {row['p_one_sided_positive_exact']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"Worst family-mean leave-one-family case: drop `{worst['dropped_family']}`, rho = {worst['rho']:+.4f}.",
            "",
            "Note: `C^cond` values are stored and reported in percentage points, matching the D5.2 source table.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = parse_headline_table(SOURCE)
    family_mean = aggregate_by_family(rows, "mean")
    family_median = aggregate_by_family(rows, "median")
    out = {
        "description": "Family-level mean/median aggregation of the D5.2 N=14 headline table.",
        "source": str(SOURCE.relative_to(ROOT.parent)),
        "n_models": len(rows),
        "n_families": len({r["family"] for r in rows}),
        "per_model_rows": rows,
        "aggregations": {
            "mean": {"families": family_mean, "spearman": spearman_summary(family_mean)},
            "median": {"families": family_median, "spearman": spearman_summary(family_median)},
        },
        "leave_one_family_mean": leave_one_family(rows),
        "p_value_note": "Exact permutation p-values over all family-label assignments; scipy.stats.spearmanr p-values are retained in JSON as *_scipy for provenance only.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    write_markdown(out)

    mean_stats = out["aggregations"]["mean"]["spearman"]
    median_stats = out["aggregations"]["median"]["spearman"]
    print(
        f"family mean: G={mean_stats['n']} rho={mean_stats['rho']:+.4f} "
        f"p_two_sided_exact={mean_stats['p_two_sided_exact']:.4f}"
    )
    print(
        f"family median: G={median_stats['n']} rho={median_stats['rho']:+.4f} "
        f"p_two_sided_exact={median_stats['p_two_sided_exact']:.4f}"
    )
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
