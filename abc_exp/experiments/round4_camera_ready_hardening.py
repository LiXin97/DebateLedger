"""Round-4 camera-ready hardening diagnostics.

This is a zero-API script. It reads existing result artifacts and writes a new
derived artifact rather than modifying sealed-vintage outputs.

Outputs:
  - abc_exp/results/round4_camera_ready_hardening.json
  - abc_exp/results/ROUND4_CAMERA_READY_HARDENING.md
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "abc_exp" / "results"
OUT_JSON = RES / "round4_camera_ready_hardening.json"
OUT_MD = RES / "ROUND4_CAMERA_READY_HARDENING.md"

SEED = 20260430
N_DRAWS = 20_000


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def family_model_rows() -> list[dict[str, Any]]:
    return read_json(RES / "reviewer_hardening_analyses.json")["per_model"]


def aggregate_family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    out = []
    for family, vals in sorted(by_family.items()):
        out.append(
            {
                "family": family,
                "n_models": len(vals),
                "models": [v["model"] for v in vals],
                "alpha_tot": float(mean(float(v["alpha_tot"]) for v in vals)),
                "c_cond_pct": float(mean(float(v["c_cond_pct"]) for v in vals)),
                "n_collapses": int(sum(int(v["n_collapses"]) for v in vals)),
                "n_init_correct": int(sum(int(v["n_init_correct"]) for v in vals)),
            }
        )
    return out


def spearman_exact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = [float(r["alpha_tot"]) for r in rows]
    y = [float(r["c_cond_pct"]) for r in rows]
    if len(rows) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(rows), "rho": None, "p_one_sided": None, "p_two_sided": None, "n_permutations": 0}
    observed = float(stats.spearmanr(x, y).statistic)
    null = [float(stats.spearmanr(x, perm).statistic) for perm in itertools.permutations(y)]
    ge = sum(v >= observed - 1e-12 for v in null)
    two = sum(abs(v) >= abs(observed) - 1e-12 for v in null)
    return {
        "n": len(rows),
        "rho": observed,
        "one_sided_ge_count": ge,
        "p_one_sided": ge / len(null),
        "two_sided_abs_count": two,
        "p_two_sided": two / len(null),
        "n_permutations": len(null),
    }


def leverage_sensitivity(family_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]]) -> dict[str, Any]:
    drop_sets = [
        ("drop_meta", ["Meta"]),
        ("drop_qwen", ["Qwen"]),
        ("drop_meta_qwen", ["Meta", "Qwen"]),
    ]
    rows = []
    for name, drops in drop_sets:
        kept = [r for r in family_rows if r["family"] not in drops]
        rows.append(
            {
                "name": name,
                "dropped_families": drops,
                "kept_families": [r["family"] for r in kept],
                "spearman_exact": spearman_exact(kept),
            }
        )
    counts = Counter(r["family"] for r in model_rows)
    n_models = sum(counts.values())
    effective_model_family_n = 1.0 / sum((c / n_models) ** 2 for c in counts.values())
    return {
        "description": "Leverage sensitivity requested by fresh reviewers; all rows are sensitivity diagnostics, not new primary tests.",
        "family_model_counts": dict(sorted(counts.items())),
        "model_row_effective_family_n_inverse_simpson": effective_model_family_n,
        "rows": rows,
    }


def sample_family_ccond_pct(model_rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, float]:
    samples_by_family: dict[str, list[float]] = {}
    for row in model_rows:
        k = int(row["n_collapses"])
        n = int(row["n_init_correct"])
        theta = rng.beta(k + 0.5, n - k + 0.5)
        samples_by_family.setdefault(row["family"], []).append(100.0 * float(theta))
    return {family: float(mean(vals)) for family, vals in samples_by_family.items()}


def eiv_lofo_prediction_intervals(model_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    family_alpha = {r["family"]: float(r["alpha_tot"]) for r in family_rows}
    observed = {r["family"]: float(r["c_cond_pct"]) for r in family_rows}
    families = [r["family"] for r in family_rows]

    per_family: list[dict[str, Any]] = []
    for held in families:
        predictive_samples = []
        train_families = [f for f in families if f != held]
        held_alpha = family_alpha[held]
        for _ in range(N_DRAWS):
            sampled_c = sample_family_ccond_pct(model_rows, rng)
            x = np.asarray([family_alpha[f] for f in train_families], dtype=float)
            y = np.asarray([sampled_c[f] for f in train_families], dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            beta_hat = np.linalg.lstsq(design, y, rcond=None)[0]
            pred = float(np.asarray([1.0, held_alpha]) @ beta_hat)
            resid = y - design @ beta_hat
            df = len(x) - 2
            sigma = math.sqrt(float(resid @ resid) / df) if df > 0 else 0.0
            xbar = float(np.mean(x))
            sxx = float(np.sum((x - xbar) ** 2))
            pred_se = sigma * math.sqrt(1.0 + 1.0 / len(x) + (held_alpha - xbar) ** 2 / sxx)
            predictive_samples.append(pred + float(rng.standard_t(df)) * pred_se)
        arr = np.asarray(predictive_samples, dtype=float)
        pi80 = [float(np.quantile(arr, 0.10)), float(np.quantile(arr, 0.90))]
        pi95 = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]
        pi80_clip = [max(0.0, pi80[0]), min(100.0, pi80[1])]
        pi95_clip = [max(0.0, pi95[0]), min(100.0, pi95[1])]
        obs = observed[held]
        per_family.append(
            {
                "heldout_family": held,
                "alpha_tot": held_alpha,
                "observed_c_cond_pct": obs,
                "eiv_pi80_pct": pi80,
                "eiv_pi95_pct": pi95,
                "eiv_pi80_pct_clipped": pi80_clip,
                "eiv_pi95_pct_clipped": pi95_clip,
                "covered_80_clipped": pi80_clip[0] <= obs <= pi80_clip[1],
                "covered_95_clipped": pi95_clip[0] <= obs <= pi95_clip[1],
            }
        )
    return {
        "description": "Jeffreys conditional-collapse uncertainty propagated through the six-family LoFO linear predictive fit; alpha is fixed at the measured family mean. Intervals are posterior predictive Monte Carlo quantiles, clipped to [0,100] for displayed coverage.",
        "seed": SEED,
        "n_draws_per_fold": N_DRAWS,
        "rows": per_family,
        "coverage_80_clipped": {"covered": int(sum(r["covered_80_clipped"] for r in per_family)), "total": len(per_family)},
        "coverage_95_clipped": {"covered": int(sum(r["covered_95_clipped"] for r in per_family)), "total": len(per_family)},
    }


def r5_panel_comparison() -> dict[str, Any]:
    panel_paths = sorted(RES.glob("r5_gemini_alpha_panel_202604*_day*.json"))
    panels = []
    for path in panel_paths:
        data = read_json(path)
        panels.append({"name": path.stem, "path": str(path.relative_to(ROOT)), **data})

    pairwise = []
    for i, a in enumerate(panels):
        qa = {r["question_id"]: r for r in a.get("per_question", [])}
        for b in panels[i + 1 :]:
            qb = {r["question_id"]: r for r in b.get("per_question", [])}
            shared = sorted(set(qa) & set(qb))
            xs, ys, deltas = [], [], []
            init_changes = 0
            for qid in shared:
                x = float(qa[qid]["alpha_mean"])
                y = float(qb[qid]["alpha_mean"])
                if math.isfinite(x) and math.isfinite(y):
                    xs.append(x)
                    ys.append(y)
                    deltas.append(abs(x - y))
                ma = Counter(qa[qid].get("initial_answers", [])).most_common(1)[0][0]
                mb = Counter(qb[qid].get("initial_answers", [])).most_common(1)[0][0]
                init_changes += int(ma != mb)
            if len(xs) >= 3 and len(set(xs)) >= 2 and len(set(ys)) >= 2:
                rho, p = stats.spearmanr(xs, ys)
                rho, p = float(rho), float(p)
            else:
                rho, p = None, None
            pairwise.append(
                {
                    "panel_a": a["name"],
                    "panel_b": b["name"],
                    "shared_questions": len(shared),
                    "alpha_pair_count": len(xs),
                    "alpha_mean_spearman": rho,
                    "alpha_mean_spearman_p_two_sided": p,
                    "mean_abs_alpha_mean_delta": float(mean(deltas)) if deltas else None,
                    "max_abs_alpha_mean_delta": float(max(deltas)) if deltas else None,
                    "initial_answer_majority_changes": init_changes,
                }
            )
    return {
        "description": "Closed-API same-prompt R5 panels found in the checkout. This is a drift audit only and is not an input to the headline association.",
        "panels": [
            {
                "name": p["name"],
                "path": p["path"],
                "n_records": p.get("n_records"),
                "n_questions": p.get("n_questions"),
                "questions_with_initial_answer_instability": p.get("questions_with_initial_answer_instability"),
                "mean_alpha_range_within_question": p.get("mean_alpha_range_within_question"),
                "max_alpha_range_within_question": p.get("max_alpha_range_within_question"),
                "probe_post_answer_none_rate": p.get("probe_post_answer_none_rate"),
                "total_cost": p.get("total_cost"),
            }
            for p in panels
        ],
        "pairwise": pairwise,
    }


def artifact_audit() -> dict[str, Any]:
    candidates = sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("**/*pilot*gated*lomo*.json"))
    candidates += sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("**/*lomo*.json"))
    r1_jsonl = RES / "per_debate_r1_features.jsonl"
    return {
        "pilot_gated_lomo_json_present": (RES / "pilot_gated_lomo.json").exists(),
        "candidate_lomo_json_files": candidates,
        "per_debate_r1_features_jsonl_bytes": r1_jsonl.stat().st_size if r1_jsonl.exists() else None,
        "strict_6525_lomo_matrix_status": "not recoverable from this checkout; only aggregate LOMO ledger and 1,255 saved-trace replay are available locally",
    }


def fmt_pct_interval(vals: list[float]) -> str:
    return f"[{vals[0]:+.2f}, {vals[1]:+.2f}]"


def write_markdown(out: dict[str, Any]) -> None:
    lines = [
        "# Round-4 Camera-Ready Hardening",
        "",
        "Zero-API diagnostics computed from checked-in artifacts. These rows are camera-ready hardening material, not new primary tests.",
        "",
        "## EIV-Propagated Held-Out-Family Prediction Intervals",
        "",
        out["eiv_lofo_prediction"]["description"],
        "",
        f"Coverage after clipping to [0,100]%: 80% {out['eiv_lofo_prediction']['coverage_80_clipped']['covered']}/{out['eiv_lofo_prediction']['coverage_80_clipped']['total']}; 95% {out['eiv_lofo_prediction']['coverage_95_clipped']['covered']}/{out['eiv_lofo_prediction']['coverage_95_clipped']['total']}.",
        "",
        "| Held-out family | observed C^cond | EIV 80% PI clipped | EIV 95% PI clipped | covered 80 | covered 95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in out["eiv_lofo_prediction"]["rows"]:
        lines.append(
            f"| {row['heldout_family']} | {row['observed_c_cond_pct']:.2f}% | "
            f"{fmt_pct_interval(row['eiv_pi80_pct_clipped'])} | {fmt_pct_interval(row['eiv_pi95_pct_clipped'])} | "
            f"{int(row['covered_80_clipped'])} | {int(row['covered_95_clipped'])} |"
        )
    lines += [
        "",
        "## Meta/Qwen Leverage Sensitivity",
        "",
        f"Model-row family-count inverse-Simpson effective family count: {out['leverage_sensitivity']['model_row_effective_family_n_inverse_simpson']:.2f} over 14 model rows. This describes the model-row imbalance only; the primary inferential table remains the seven-family exact test.",
        "",
        "| Sensitivity | G | rho | exact one-sided p | exact two-sided p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in out["leverage_sensitivity"]["rows"]:
        s = row["spearman_exact"]
        lines.append(
            f"| {', '.join(row['dropped_families'])} dropped | {s['n']} | {s['rho']:+.3f} | "
            f"{s['p_one_sided']:.4f} | {s['p_two_sided']:.4f} |"
        )
    lines += [
        "",
        "## R5 Closed-API Stability Panels",
        "",
        "R5 remains a drift/nondeterminism audit, not a headline input.",
        "",
        "| Panel | records | questions | init unstable | mean alpha range | max alpha range | post-None | cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for panel in out["r5_panel_comparison"]["panels"]:
        lines.append(
            f"| `{panel['name']}` | {panel['n_records']} | {panel['n_questions']} | "
            f"{panel['questions_with_initial_answer_instability']} | {panel['mean_alpha_range_within_question']:.4f} | "
            f"{panel['max_alpha_range_within_question']:.4f} | {panel['probe_post_answer_none_rate']:.2%} | "
            f"${panel['total_cost']:.4f} |"
        )
    if out["r5_panel_comparison"]["pairwise"]:
        lines += [
            "",
            "| Panel pair | shared q | Spearman | mean abs delta | max abs delta | init-answer changes |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for pair in out["r5_panel_comparison"]["pairwise"]:
            rho = "NA" if pair["alpha_mean_spearman"] is None else f"{pair['alpha_mean_spearman']:.4f}"
            lines.append(
                f"| `{pair['panel_a']}` vs `{pair['panel_b']}` | {pair['shared_questions']} | {rho} | "
                f"{pair['mean_abs_alpha_mean_delta']:.4f} | {pair['max_abs_alpha_mean_delta']:.4f} | "
                f"{pair['initial_answer_majority_changes']} |"
            )
    audit = out["artifact_audit"]
    lines += [
        "",
        "## 6,525-Row Matched-Tau DG Audit",
        "",
        f"`pilot_gated_lomo.json` present: {audit['pilot_gated_lomo_json_present']}.",
        f"Candidate LOMO JSON files found: {audit['candidate_lomo_json_files'] or 'none'}.",
        f"`per_debate_r1_features.jsonl` size: {audit['per_debate_r1_features_jsonl_bytes']} bytes.",
        "Conclusion: the strict 6,525-row matched-tau DisagreementGate/DRS comparison is not recoverable from this checkout and should remain explicitly disclaimed unless the gated matrix is restored.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    model_rows = family_model_rows()
    family_rows = aggregate_family_rows(model_rows)
    out = {
        "description": "Round-4 zero-API camera-ready hardening diagnostics.",
        "inputs": {
            "model_rows": "abc_exp/results/reviewer_hardening_analyses.json:per_model",
            "r5_panels_pattern": "abc_exp/results/r5_gemini_alpha_panel_202604*_day*.json",
        },
        "eiv_lofo_prediction": eiv_lofo_prediction_intervals(model_rows, family_rows),
        "leverage_sensitivity": leverage_sensitivity(family_rows, model_rows),
        "r5_panel_comparison": r5_panel_comparison(),
        "artifact_audit": artifact_audit(),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    write_markdown(out)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
