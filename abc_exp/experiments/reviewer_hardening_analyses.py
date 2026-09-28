"""Reviewer-hardening zero-cost analyses for the NeurIPS draft.

This script does not run new model calls. It consolidates existing N=14
headline rows with available debate-trace summaries, then writes a compact
confound/utility report that can be cited from the paper or appendix.

Outputs:
  - abc_exp/results/reviewer_hardening_analyses.json
  - abc_exp/results/REVIEWER_HARDENING_ANALYSES.md
"""

from __future__ import annotations

import json
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "abc_exp" / "results"
OUT_JSON = RES / "reviewer_hardening_analyses.json"
OUT_MD = RES / "REVIEWER_HARDENING_ANALYSES.md"


TRACE_PATHS = {
    "deepseek-v4-flash": RES / "debate_traces_openrouter_deepseek-v4-flash.jsonl",
    "gemma-4-31b-it-awq": RES / "debate_traces_vllm_gemma-4-31b-it.jsonl",
    "qwen3.5-4b": RES / "debate_traces_vllm_qwen3.5-4b.jsonl",
    "qwen3.5-9b": RES / "debate_traces_vllm_qwen3.5-9b.jsonl",
    "qwen3.6-27b-fp8": RES / "debate_traces_vllm_qwen3.6-27b-fp8.jsonl",
    "qwen3.6-35b-a3b-fp8": RES / "debate_traces_vllm_qwen3.6-35b-a3b-fp8.jsonl",
}

PREDICTOR_META = {
    "alpha_tot": {
        "label": r"8-probe $\alpha_{\mathrm{tot}}$",
        "role": "pre-debate fingerprint",
        "note": "predictor of record; measured before debate compute",
    },
    "init_acc": {
        "label": "initial-majority accuracy",
        "role": "capability proxy",
        "note": "pre-debate capability/difficulty proxy; negative rho means weaker initial panels collapse more",
    },
    "final_acc": {
        "label": "final debate accuracy",
        "role": "post-debate descriptive control",
        "note": "not deployable as a predictor; included to measure capability confounding pressure",
    },
    "correction_pct": {
        "label": "conditional correction rate",
        "role": "post-debate transition control",
        "note": "not deployable pre-debate; captures the opposing productive-revision transition",
    },
    "debate_revision_proxy": {
        "label": "raw debate revision proxy",
        "role": "revision-quantity baseline",
        "note": "debate FR for sealed rows and majority-flip rate for post-A6 trace rows",
    },
    "raw_collapse_pct": {
        "label": "raw collapse rate",
        "role": "outcome-derived upper bound",
        "note": "shares the collapse numerator with C_cond; sanity check only, not a usable predictor",
    },
}

TRACE_DIVERSITY_META = {
    "initial_disagreement_rate": {
        "label": "initial disagreement rate",
        "note": "fraction of debates where the three initial answers are not unanimous",
    },
    "initial_unanimity_rate": {
        "label": "initial unanimity rate",
        "note": "fraction of debates where all parsed initial answers agree",
    },
    "initial_entropy_norm": {
        "label": "normalized initial-answer entropy",
        "note": "mean Shannon entropy of initial answers, normalized by log2(3)",
    },
    "at_risk_initial_disagreement_rate": {
        "label": "at-risk initial disagreement rate",
        "note": "initial disagreement restricted to debates whose initial majority is correct",
    },
}

CONDITIONAL_TABLE_NAME_MAP = {
    "Claude Sonnet 4.5": "sonnet-4.5",
    "GPT-4o-mini": "gpt-4o-mini",
    "GPT-5.4-mini": "gpt-5.4-mini",
    "Gemini 3-flash": "gemini-3-flash",
    "Phi-4-mini": "phi-4-mini",
    "Qwen3-4B": "qwen3-4b",
    "Llama-3.1-8B": "llama-3.1-8b",
    "Qwen3-8B": "qwen3-8b",
}


def majority(xs: Iterable[str | None]) -> str:
    vals = [x for x in xs if x]
    if not vals:
        return ""
    counts = Counter(vals)
    m = max(counts.values())
    return sorted(k for k, v in counts.items() if v == m)[0]


def answer_profile(xs: Iterable[str | None]) -> dict:
    vals = [x for x in xs if x]
    if not vals:
        return {
            "n_parsed": 0,
            "n_unique": 0,
            "is_disagreement": False,
            "is_unanimous": False,
            "entropy_norm": math.nan,
            "majority_margin": math.nan,
        }
    counts = Counter(vals)
    total = sum(counts.values())
    entropy = -sum((v / total) * math.log(v / total, 2) for v in counts.values())
    return {
        "n_parsed": total,
        "n_unique": len(counts),
        "is_disagreement": len(counts) > 1,
        "is_unanimous": len(counts) == 1 and total == 3,
        "entropy_norm": entropy / math.log(3, 2),
        "majority_margin": max(counts.values()) / total,
    }


def trace_summary(path: Path) -> dict:
    n = init_correct = final_correct = collapses = corrections = majority_flips = 0
    n_init_wrong = 0
    initial_disagreements = initial_unanimous = at_risk_disagreements = 0
    entropy_sum = majority_margin_sum = 0.0
    rel_path = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if not path.exists():
        return {"available": False, "path": rel_path}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            correct = row.get("correct_label")
            profile = answer_profile(row.get("initial_answers") or [])
            initial_disagreements += int(profile["is_disagreement"])
            initial_unanimous += int(profile["is_unanimous"])
            if math.isfinite(profile["entropy_norm"]):
                entropy_sum += profile["entropy_norm"]
            if math.isfinite(profile["majority_margin"]):
                majority_margin_sum += profile["majority_margin"]
            init = majority(row.get("initial_answers") or [])
            final = row.get("majority_answer") or majority(row.get("final_answers") or [])
            ic = bool(init and correct and init == correct)
            fc = bool(final and correct and final == correct)
            init_correct += int(ic)
            at_risk_disagreements += int(ic and profile["is_disagreement"])
            final_correct += int(fc)
            n_init_wrong += int(not ic)
            collapses += int(ic and not fc)
            corrections += int((not ic) and fc)
            majority_flips += int(bool(init and final and init != final))
    return {
        "available": True,
        "path": rel_path,
        "n_debates": n,
        "n_init_correct": init_correct,
        "n_init_wrong": n_init_wrong,
        "n_collapses": collapses,
        "n_corrections": corrections,
        "init_acc": init_correct / n if n else math.nan,
        "final_acc": final_correct / n if n else math.nan,
        "c_cond_pct": 100 * collapses / init_correct if init_correct else math.nan,
        "correction_pct": 100 * corrections / n_init_wrong if n_init_wrong else math.nan,
        "majority_flip_rate": majority_flips / n if n else math.nan,
        "raw_collapse_pct": 100 * collapses / n if n else math.nan,
        "initial_disagreement_rate": initial_disagreements / n if n else math.nan,
        "initial_unanimity_rate": initial_unanimous / n if n else math.nan,
        "initial_entropy_norm": entropy_sum / n if n else math.nan,
        "mean_initial_majority_margin": majority_margin_sum / n if n else math.nan,
        "at_risk_initial_disagreement_rate": at_risk_disagreements / init_correct if init_correct else math.nan,
    }


def load_rows() -> list[dict]:
    headline = json.load(open(RES / "family_level_headline_aggregation.json"))["per_model_rows"]
    rows = []
    for r in headline:
        rows.append({
            "model": r["model"],
            "family": r["family"],
            "alpha_tot": float(r["alpha_tot"]),
            "c_cond_pct": float(r["c_cond_pct"]),
            "wilson_hw": float(r["wilson_hw"]),
        })

    # Older sealed-lane table has init/final/correction/raw-FR diagnostics.
    cct = json.load(open(RES / "conditional_collapse_table.json"))["table"]
    by_model = {r["model"]: r for r in rows}
    for old in cct:
        key = CONDITIONAL_TABLE_NAME_MAP.get(old["model"])
        if key in by_model:
            by_model[key].update({
                "n_debates": int(old["n_deb"]),
                "n_init_correct": int(old["n_init_correct"]),
                "n_init_wrong": int(old["n_init_wrong"]),
                "init_acc": float(old["init_acc"]),
                "final_acc": (old["n_init_correct"] - old["n_cond_coll"] + old["n_recover"]) / old["n_deb"],
                "n_collapses": int(old["n_cond_coll"]),
                "n_corrections": int(old["n_recover"]),
                "correction_pct": float(old["recover_pct"]),
                "raw_collapse_pct": float(old["raw_collapse_pct"]),
                "debate_revision_proxy": float(old["FR"]),
                "diagnostic_source": "conditional_collapse_table",
            })

    for model, path in TRACE_PATHS.items():
        if model in by_model:
            s = trace_summary(path)
            if s.get("available"):
                by_model[model].update({k: v for k, v in s.items() if k not in {"available", "path"}})
                by_model[model]["trace_path"] = s["path"]
                by_model[model]["debate_revision_proxy"] = s["majority_flip_rate"]
                by_model[model]["diagnostic_source"] = "debate_trace_summary"

    return rows


def finite_pairs(rows: list[dict], x_key: str, y_key: str) -> list[tuple[float, float]]:
    out = []
    for r in rows:
        x, y = r.get(x_key), r.get(y_key)
        if x is None or y is None:
            continue
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            continue
        out.append((float(x), float(y)))
    return out


def spearman(rows: list[dict], x_key: str, y_key: str = "c_cond_pct") -> dict:
    pairs = finite_pairs(rows, x_key, y_key)
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
        return {"n": len(pairs), "rho": None, "p_two_sided": None, "note": "insufficient variation"}
    rho, p = stats.spearmanr([x for x, _ in pairs], [y for _, y in pairs])
    return {"n": len(pairs), "rho": float(rho), "p_two_sided": float(p)}


def spearman_exact_permutation(rows: list[dict], x_key: str, y_key: str = "c_cond_pct") -> dict:
    pairs = finite_pairs(rows, x_key, y_key)
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
        return {"n": len(pairs), "rho": None, "p_two_sided_exact": None, "note": "insufficient variation"}
    x = [x for x, _ in pairs]
    y = [y for _, y in pairs]
    obs = float(stats.spearmanr(x, y).statistic)
    null = [float(stats.spearmanr(x, list(p)).statistic) for p in set(itertools.permutations(y))]
    if obs >= 0:
        one_sided = sum(v >= obs - 1e-12 for v in null)
        direction = "positive"
    else:
        one_sided = sum(v <= obs + 1e-12 for v in null)
        direction = "negative"
    two_sided = sum(abs(v) >= abs(obs) - 1e-12 for v in null)
    return {
        "n": len(pairs),
        "rho": obs,
        "n_permutations": len(null),
        "one_sided_in_observed_direction_count": one_sided,
        "p_one_sided_in_observed_direction_exact": one_sided / len(null),
        "observed_direction": direction,
        "two_sided_abs_count": two_sided,
        "p_two_sided_exact": two_sided / len(null),
    }


def partial_spearman(rows: list[dict], x_key: str, y_key: str, control_keys: list[str]) -> dict:
    data = []
    for r in rows:
        vals = [r.get(x_key), r.get(y_key)] + [r.get(k) for k in control_keys]
        if any(v is None for v in vals):
            continue
        vals = [float(v) for v in vals]
        if all(math.isfinite(v) for v in vals):
            data.append(vals)
    n = len(data)
    k = len(control_keys)
    if n <= k + 3:
        return {"n": n, "rho": None, "p_two_sided": None, "controls": control_keys, "note": "insufficient df"}
    arr = np.asarray(data, dtype=float)
    ranked = np.apply_along_axis(stats.rankdata, 0, arr)
    x = ranked[:, 0]
    y = ranked[:, 1]
    z = ranked[:, 2:]
    design = np.column_stack([np.ones(n), z])
    bx = np.linalg.lstsq(design, x, rcond=None)[0]
    by = np.linalg.lstsq(design, y, rcond=None)[0]
    rx = x - design @ bx
    ry = y - design @ by
    rho = float(np.corrcoef(rx, ry)[0, 1])
    df = n - k - 2
    t = rho * math.sqrt(df / max(1e-12, 1 - rho * rho))
    p = 2 * stats.t.sf(abs(t), df)
    return {"n": n, "rho": rho, "p_two_sided": float(p), "controls": control_keys, "df": df}


def family_aggregate(rows: list[dict], reducer: str = "mean") -> list[dict]:
    fams: dict[str, list[dict]] = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    out = []
    for fam, rs in sorted(fams.items()):
        op = np.mean if reducer == "mean" else np.median
        entry = {
            "family": fam,
            "n_models": len(rs),
        }
        for key in ["c_cond_pct", *PREDICTOR_META.keys()]:
            vals = [r.get(key) for r in rs]
            vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
            if vals:
                entry[key] = float(op(vals))
        out.append(entry)
    return out


def bakeoff(rows: list[dict], predictor_keys: list[str]) -> dict:
    fam_mean = family_aggregate(rows, "mean")
    out = {}
    for key in predictor_keys:
        out[key] = {
            **PREDICTOR_META[key],
            "model_row_spearman": spearman(rows, key),
            "family_mean_exact_spearman": spearman_exact_permutation(fam_mean, key),
        }
    return out


def weighted_group(rows: list[dict]) -> dict:
    n_deb = sum(int(r.get("n_debates", 0)) for r in rows)
    init = sum(int(r.get("n_init_correct", 0)) for r in rows)
    wrong = sum(int(r.get("n_init_wrong", 0)) for r in rows)
    coll = sum(int(r.get("n_collapses", 0)) for r in rows)
    corr = sum(int(r.get("n_corrections", 0)) for r in rows)
    final_correct = sum(round(float(r.get("final_acc", 0)) * int(r.get("n_debates", 0))) for r in rows)
    return {
        "n_models": len(rows),
        "models": [r["model"] for r in rows],
        "mean_alpha": float(np.mean([r["alpha_tot"] for r in rows])) if rows else math.nan,
        "n_debates": n_deb,
        "conditional_collapse_pct": 100 * coll / init if init else math.nan,
        "correction_pct": 100 * corr / wrong if wrong else math.nan,
        "final_acc": final_correct / n_deb if n_deb else math.nan,
    }


def selection_utility(rows: list[dict]) -> dict:
    usable = [r for r in rows if "n_debates" in r and r.get("n_init_correct", 0)]
    med = float(np.median([r["alpha_tot"] for r in usable]))
    low = [r for r in usable if r["alpha_tot"] <= med]
    high = [r for r in usable if r["alpha_tot"] > med]

    pairs = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            if abs(float(a["init_acc"]) - float(b["init_acc"])) <= 0.05:
                lo, hi = (a, b) if a["alpha_tot"] <= b["alpha_tot"] else (b, a)
                pairs.append({
                    "low_alpha_model": lo["model"],
                    "high_alpha_model": hi["model"],
                    "init_acc_gap": abs(float(a["init_acc"]) - float(b["init_acc"])),
                    "delta_ccond_low_minus_high": lo["c_cond_pct"] - hi["c_cond_pct"],
                    "delta_final_acc_low_minus_high": float(lo["final_acc"]) - float(hi["final_acc"]),
                })
    wins = [p for p in pairs if p["delta_ccond_low_minus_high"] < 0]
    return {
        "median_alpha_split": med,
        "low_alpha_group": weighted_group(low),
        "high_alpha_group": weighted_group(high),
        "accuracy_matched_pair_window_init_acc": 0.05,
        "n_accuracy_matched_pairs": len(pairs),
        "low_alpha_lower_collapse_pairs": len(wins),
        "mean_delta_ccond_low_minus_high": float(np.mean([p["delta_ccond_low_minus_high"] for p in pairs])) if pairs else math.nan,
        "pairs": pairs,
    }


def trace_diversity_diagnostics(rows: list[dict]) -> dict:
    trace_rows = [r for r in rows if r.get("diagnostic_source") == "debate_trace_summary"]
    out = {
        "description": (
            "Trace-available diagnostic only: six post-A6 rows with raw initial answers "
            "(DeepSeek, Gemma, and four Qwen variants). These metrics are measured after "
            "the three initial answers are sampled, so they are not substitutes for the "
            "single-model pre-debate alpha fingerprint."
        ),
        "n_models": len(trace_rows),
        "models": [r["model"] for r in trace_rows],
        "spearman_vs_ccond": {},
        "partial_spearman": {},
        "per_model": [],
    }
    for key, meta in TRACE_DIVERSITY_META.items():
        out["spearman_vs_ccond"][key] = {**meta, **spearman(trace_rows, key)}
    out["partial_spearman"]["alpha_controlling_initial_disagreement_rate"] = partial_spearman(
        trace_rows, "alpha_tot", "c_cond_pct", ["initial_disagreement_rate"]
    )
    for r in trace_rows:
        out["per_model"].append(
            {
                "model": r["model"],
                "n_debates": r.get("n_debates"),
                "alpha_tot": r.get("alpha_tot"),
                "c_cond_pct": r.get("c_cond_pct"),
                "initial_disagreement_rate": r.get("initial_disagreement_rate"),
                "initial_unanimity_rate": r.get("initial_unanimity_rate"),
                "initial_entropy_norm": r.get("initial_entropy_norm"),
                "at_risk_initial_disagreement_rate": r.get("at_risk_initial_disagreement_rate"),
            }
        )
    return out


def main() -> None:
    rows = load_rows()
    predictor_keys = list(PREDICTOR_META)
    predictor_bakeoff = bakeoff(rows, predictor_keys)
    partials = {
        "alpha_controlling_init_acc": partial_spearman(rows, "alpha_tot", "c_cond_pct", ["init_acc"]),
        "alpha_controlling_init_and_final_acc": partial_spearman(rows, "alpha_tot", "c_cond_pct", ["init_acc", "final_acc"]),
        "alpha_controlling_revision_proxy": partial_spearman(rows, "alpha_tot", "c_cond_pct", ["debate_revision_proxy"]),
        "alpha_controlling_init_acc_and_revision_proxy": partial_spearman(
            rows, "alpha_tot", "c_cond_pct", ["init_acc", "debate_revision_proxy"]
        ),
        "alpha_controlling_init_final_and_revision_proxy": partial_spearman(
            rows, "alpha_tot", "c_cond_pct", ["init_acc", "final_acc", "debate_revision_proxy"]
        ),
    }
    fam_mean = family_aggregate(rows, "mean")
    fam_median = family_aggregate(rows, "median")
    out = {
        "description": "Zero-cost reviewer-hardening analyses on the realized N=14 headline cohort.",
        "n_models": len(rows),
        "per_model": rows,
        "predictor_bakeoff_spearman_vs_ccond": {k: v["model_row_spearman"] for k, v in predictor_bakeoff.items()},
        "predictor_bakeoff": predictor_bakeoff,
        "partial_spearman": partials,
        "family_mean_spearman": spearman_exact_permutation(fam_mean, "alpha_tot"),
        "family_median_spearman": spearman_exact_permutation(fam_median, "alpha_tot"),
        "trace_diversity_diagnostics": trace_diversity_diagnostics(rows),
        "selection_utility": selection_utility(rows),
        "notes": [
            "debate_revision_proxy is debate FR for sealed conditional-collapse rows and majority-flip rate for post-A6 trace rows; use only as a coarse anti-raw-revision diagnostic.",
            "initial disagreement/diversity metrics are available only for six post-A6 raw-trace rows and are reported as runtime diagnostics, not as the primary selection-time fingerprint.",
            "final_acc, correction_pct, and raw_collapse_pct are post-debate or outcome-derived controls; they are not deployable pre-debate predictors.",
            "selection utility is descriptive and zero-cost; it is not a learned deployment policy.",
        ],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    write_markdown(out)


def fmt_stat(d: dict) -> str:
    if d.get("rho") is None:
        return "NA"
    return f"{d['rho']:+.3f} (n={d['n']}, p={d['p_two_sided']:.3g})"


def fmt_exact_stat(d: dict) -> str:
    if d.get("rho") is None:
        return "NA"
    return (
        f"{d['rho']:+.3f} "
        f"(G={d['n']}, exact p2={d['p_two_sided_exact']:.3g}, "
        f"p1-{d['observed_direction'][:3]}={d['p_one_sided_in_observed_direction_exact']:.3g})"
    )


def write_markdown(out: dict) -> None:
    su = out["selection_utility"]
    td = out["trace_diversity_diagnostics"]
    lines = [
        "# Reviewer-Hardening Analyses",
        "",
        "Zero-cost diagnostics on the realized N=14 headline cohort. These are intended to block common reviewer confounds, not to create a new primary claim.",
        "",
        "## Predictor Bakeoff vs Conditional Collapse",
        "",
        "Only the fingerprint and initial-majority accuracy are pre-debate quantities. Final accuracy, correction rate, and raw collapse are included as descriptive controls rather than deployable predictors.",
        "",
        "| Predictor | Role | Model-row Spearman | Family-mean exact Spearman | Note |",
        "|---|---|---:|---:|---|",
    ]
    for k, d in out["predictor_bakeoff"].items():
        lines.append(
            f"| {d['label']} | {d['role']} | {fmt_stat(d['model_row_spearman'])} | "
            f"{fmt_exact_stat(d['family_mean_exact_spearman'])} | {d['note']} |"
        )
    lines += [
        "",
        "## Partial Spearman Checks",
        "",
        "Rank residualization tests whether the fingerprint remains associated with conditional collapse after removing simple capability or revision-quantity explanations. These are observational sensitivity checks on a small cohort, not new confirmatory tests.",
        "",
        "| Check | Result | Controls |",
        "|---|---:|---|",
    ]
    for k, d in out["partial_spearman"].items():
        lines.append(f"| `{k}` | {fmt_stat(d)} | {', '.join(d.get('controls', []))} |")
    lines += [
        "",
        "## Family Aggregation",
        "",
        f"- Family mean aggregation: {fmt_exact_stat(out['family_mean_spearman'])}",
        f"- Family median aggregation: {fmt_exact_stat(out['family_median_spearman'])}",
        "",
        "## Trace-Available Initial Diversity Diagnostics",
        "",
        td["description"],
        "",
        "| Metric | Spearman vs C^cond | Note |",
        "|---|---:|---|",
    ]
    for key, d in td["spearman_vs_ccond"].items():
        lines.append(f"| {d['label']} | {fmt_stat(d)} | {d['note']} |")
    partial = td["partial_spearman"]["alpha_controlling_initial_disagreement_rate"]
    lines += [
        "",
        f"Partial alpha check within the same trace-only slice, controlling initial disagreement: {fmt_stat(partial)}. This is underpowered (n=6) and is included only to make the disagreement confound explicit.",
        "",
        "| Model | debates | alpha | C^cond | init disagree | init unanimous | init entropy | at-risk disagree |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in td["per_model"]:
        lines.append(
            f"| {r['model']} | {r['n_debates']} | {r['alpha_tot']:.4f} | {r['c_cond_pct']:.2f}% | "
            f"{r['initial_disagreement_rate']:.2%} | {r['initial_unanimity_rate']:.2%} | "
            f"{r['initial_entropy_norm']:.3f} | {r['at_risk_initial_disagreement_rate']:.2%} |"
        )
    lines += [
        "",
        "## Selection Utility Diagnostic",
        "",
        f"Median alpha split: `{su['median_alpha_split']:.4f}`.",
        "",
        "| Group | Models | C^cond | Correction | Final acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["low_alpha_group", "high_alpha_group"]:
        g = su[name]
        lines.append(
            f"| {name.replace('_', ' ')} | {g['n_models']} | "
            f"{g['conditional_collapse_pct']:.2f}% | {g['correction_pct']:.2f}% | {g['final_acc']:.2%} |"
        )
    lines += [
        "",
        f"Accuracy-matched pair window on initial accuracy: +/-{su['accuracy_matched_pair_window_init_acc']:.2f}.",
        f"Low-alpha model has lower conditional collapse in {su['low_alpha_lower_collapse_pairs']}/{su['n_accuracy_matched_pairs']} matched pairs; mean low-minus-high C^cond delta = {su['mean_delta_ccond_low_minus_high']:+.2f}pp.",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in out["notes"])
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
