"""Rebuttal analyses for Reviewer TdYQ Q1/Q2 (2026-07-24).

Q1: enumerate all C(14,2)=91 model-row pairs and count re-orderings between
the 8-probe screen (alpha_tot; higher = riskier) and initial-majority accuracy
(lower = riskier), adjudicated against observed conditional collapse, with
two-sided Fisher-exact significance on each pair's collapse-count difference
and Holm adjustment over all re-ordered pairs.

Q2: two-sided Fisher-z planning approximations for the capability-adjusted
partial Spearman at the family level, using both the submitted-paper estimate
(rho=0.767) and the response-period extension estimate (rho=0.597).

Zero-API: reads only checked-in artifacts. Output:
abc_exp/results/rebuttal_q1_pairwise_enumeration.json
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

from scipy import stats as st

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "abc_exp/results/reviewer_hardening_analyses.json"
OUT = REPO_ROOT / "abc_exp/results/rebuttal_q1_pairwise_enumeration.json"


def fisher_z_power(rho: float, n_eff: int, k_controls: int, two_sided: bool = True,
                   alpha: float = 0.05) -> float:
    """Power for a (partial) correlation via the Fisher-z approximation.

    n_eff is the number of units (families or rows); the z statistic has
    se = 1/sqrt(n_eff - 3 - k_controls).
    """
    df = n_eff - 3 - k_controls
    if df <= 0:
        return 0.0
    z_mean = abs(math.atanh(rho)) * math.sqrt(df)
    zcrit = st.norm.ppf(1 - (alpha / 2 if two_sided else alpha))
    upper_tail = st.norm.sf(zcrit - z_mean)
    if not two_sided:
        return float(upper_tail)
    lower_tail = st.norm.cdf(-zcrit - z_mean)
    return float(upper_tail + lower_tail)


def minimum_n_for_power(rho: float, k_controls: int, target_power: float,
                        alpha: float = 0.05, max_n: int = 10_000) -> int:
    """Smallest independent-unit count reaching target two-sided power."""
    for n_eff in range(4 + k_controls, max_n + 1):
        if fisher_z_power(rho, n_eff, k_controls, True, alpha) >= target_power:
            return n_eff
    raise RuntimeError(f"target power not reached by n={max_n}")


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=p_values.__getitem__)
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted[idx] = float(min(1.0, running_max))
    return adjusted


def main() -> int:
    per_model = json.loads(SRC.read_text())["per_model"]
    rows = [
        {
            "model": v["model"],
            "alpha_tot": v["alpha_tot"],
            "init_acc": v["init_acc"],
            "c_cond_pct": v["c_cond_pct"],
            "k_collapses": v["n_collapses"],
            "n_init_correct": v["n_init_correct"],
        }
        for v in per_model
    ]
    assert len(rows) == 14, f"expected 14 cohort rows, got {len(rows)}"

    pairs = []
    n_reordered = 0
    adjudication = {"alpha_confirmed": 0, "init_acc_confirmed": 0, "ccond_tied": 0}
    for a, b in itertools.combinations(rows, 2):
        alpha_says_a_riskier = a["alpha_tot"] > b["alpha_tot"]
        cap_says_a_riskier = a["init_acc"] < b["init_acc"]
        reordered = alpha_says_a_riskier != cap_says_a_riskier
        rec = {
            "pair": [a["model"], b["model"]],
            "reordered": reordered,
        }
        if reordered:
            n_reordered += 1
            tab = [
                [a["k_collapses"], a["n_init_correct"] - a["k_collapses"]],
                [b["k_collapses"], b["n_init_correct"] - b["k_collapses"]],
            ]
            _, p = st.fisher_exact(tab)
            rec["fisher_p_two_sided_ccond_diff"] = p
            if a["c_cond_pct"] == b["c_cond_pct"]:
                adjudication["ccond_tied"] += 1
                rec["confirmed_by_observed_ccond"] = "tied"
            else:
                obs_a_riskier = a["c_cond_pct"] > b["c_cond_pct"]
                winner = "alpha" if alpha_says_a_riskier == obs_a_riskier else "init_acc"
                rec["confirmed_by_observed_ccond"] = winner
                adjudication[f"{winner}_confirmed"] += 1
        pairs.append(rec)

    reordered = [rec for rec in pairs if rec["reordered"]]
    holm_p = holm_adjust(
        [rec["fisher_p_two_sided_ccond_diff"] for rec in reordered]
    )
    for rec, p_adj in zip(reordered, holm_p):
        rec["holm_p_over_26_reordered"] = p_adj
        rec["holm_reject_0_05"] = bool(p_adj <= 0.05)

    fisher_sig = [
        {**rec} for rec in reordered
        if rec["fisher_p_two_sided_ccond_diff"] < 0.05
    ]
    holm_sig = [{**rec} for rec in reordered if rec["holm_reject_0_05"]]

    def confirmation_split(records: list[dict]) -> dict[str, int]:
        return {
            "alpha_confirmed": sum(
                rec["confirmed_by_observed_ccond"] == "alpha" for rec in records
            ),
            "init_acc_confirmed": sum(
                rec["confirmed_by_observed_ccond"] == "init_acc" for rec in records
            ),
            "ccond_tied": sum(
                rec["confirmed_by_observed_ccond"] == "tied" for rec in records
            ),
        }

    alpha = 0.05
    target_power = 0.80
    k_controls = 1
    planning_rho = 0.767
    extension_rho = 0.597
    q2 = {
        "method": "Fisher-z normal approximation for a partial correlation",
        "assumptions": {
            "independent_unit": "model family",
            "test": "two-sided",
            "alpha": alpha,
            "target_power": target_power,
            "k_controls": k_controls,
        },
        "submitted_paper_planning_effect": {
            "rho": planning_rho,
            "power_by_G": {
                G: round(fisher_z_power(planning_rho, G, k_controls, True, alpha), 3)
                for G in range(7, 31)
            },
            "minimum_G_for_target_power": minimum_n_for_power(
                planning_rho, k_controls, target_power, alpha
            ),
        },
        "extension_effect_sensitivity": {
            "rho": extension_rho,
            "power_by_G": {
                G: round(fisher_z_power(extension_rho, G, k_controls, True, alpha), 3)
                for G in range(7, 31)
            },
            "minimum_G_for_target_power": minimum_n_for_power(
                extension_rho, k_controls, target_power, alpha
            ),
        },
        "caveat": "Planning approximation only. It treats families as independent "
                  "and reuses noisy observed effects; it is neither exact-permutation "
                  "power nor retrospective achieved power.",
    }

    out = {
        "description": "TdYQ Q1 pairwise re-ordering enumeration + Q2 power projection",
        "source": str(SRC.relative_to(REPO_ROOT)),
        "n_pairs": len(pairs),
        "n_reordered": n_reordered,
        "reordered_pct": round(100 * n_reordered / len(pairs), 1),
        "adjudication_among_reordered": adjudication,
        "fisher_significant_reordered_pairs": fisher_sig,
        "n_fisher_significant": len(fisher_sig),
        "fisher_significant_split": {
            **confirmation_split(fisher_sig),
        },
        "multiplicity_adjustment": {
            "method": "Holm step-down",
            "family": "all 26 re-ordered pairwise two-sided Fisher tests",
            "familywise_alpha": 0.05,
        },
        "holm_significant_reordered_pairs": holm_sig,
        "n_holm_significant": len(holm_sig),
        "holm_significant_split": confirmation_split(holm_sig),
        "q2_power_projection": q2,
        "pairs": pairs,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"pairs={len(pairs)} reordered={n_reordered} ({out['reordered_pct']}%) "
          f"adjudication={adjudication} raw_sig={out['fisher_significant_split']} "
          f"holm_sig={out['holm_significant_split']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
