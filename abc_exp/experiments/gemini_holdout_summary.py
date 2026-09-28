"""Summarize the Gemini 3.1 Flash-Lite holdout runs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RES = Path("abc_exp/results")
ALPHA = RES / "sa_causal_gemini_3_1_flash_lite_n200.jsonl"
DEBATE = RES / "debate_traces_gemini_3_1_flash_lite_n200.jsonl"
CROSS = RES / "cross_benchmark_gemini_3_1_flash_lite_n50.json"
OUT_JSON = RES / "gemini_3_1_flash_lite_holdout_summary.json"
OUT_MD = RES / "GEMINI_3_1_FLASH_LITE_HOLDOUT_SUMMARY.md"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def mode(values: list[str | None]) -> str | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    counts = Counter(clean)
    max_count = max(counts.values())
    return sorted(v for v, c in counts.items() if c == max_count)[0]


def pct(num: int | float, den: int | float) -> float | None:
    return 100.0 * num / den if den else None


def summarize_alpha(rows: list[dict]) -> dict:
    by_condition: dict[str, list[float]] = defaultdict(list)
    by_question: dict[str, list[float]] = defaultdict(list)
    total_cost = 0.0
    for row in rows:
        fr = float(row["flip_rate"])
        by_condition[row["condition"]].append(fr)
        by_question[row["question_id"]].append(fr)
        total_cost += float(row.get("total_cost", 0.0))
    return {
        "n_rows": len(rows),
        "n_questions": len(by_question),
        "condition_counts": dict(Counter(r["condition"] for r in rows)),
        "alpha_tot": float(np.mean([float(r["flip_rate"]) for r in rows])),
        "alpha_by_condition": {
            k: float(np.mean(v)) for k, v in sorted(by_condition.items())
        },
        "question_alpha_mean": float(np.mean([np.mean(v) for v in by_question.values()])),
        "total_cost": total_cost,
    }


def summarize_debate(rows: list[dict]) -> dict:
    init_maj_correct = 0
    final_maj_correct = 0
    majority_collapses = 0
    majority_corrections = 0
    majority_flips = 0
    agent_collapses = 0
    agent_corrections = 0
    total_cost = 0.0
    for row in rows:
        correct = row["correct_label"]
        init_maj = mode(row.get("initial_answers") or [])
        final_maj = row.get("majority_answer") or mode(row.get("final_answers") or [])
        if init_maj is not None and final_maj is not None and final_maj != init_maj:
            majority_flips += 1
        init_ok = init_maj == correct
        final_ok = final_maj == correct
        init_maj_correct += int(init_ok)
        final_maj_correct += int(final_ok)
        majority_collapses += int(init_ok and not final_ok)
        majority_corrections += int((not init_ok) and final_ok)
        agent_collapses += int(bool(row.get("outcome", {}).get("collapse")))
        agent_corrections += int(bool(row.get("outcome", {}).get("correction")))
        total_cost += float(row.get("total_cost", 0.0))
    n = len(rows)
    return {
        "n_debates": n,
        "init_majority_correct": init_maj_correct,
        "final_majority_correct": final_maj_correct,
        "initial_accuracy_pct": pct(init_maj_correct, n),
        "final_accuracy_pct": pct(final_maj_correct, n),
        "majority_collapses": majority_collapses,
        "majority_corrections": majority_corrections,
        "conditional_collapse_pct": pct(majority_collapses, init_maj_correct),
        "conditional_correction_pct": pct(majority_corrections, n - init_maj_correct),
        "majority_flip_rate_pct": pct(majority_flips, n),
        "agent_level_collapse_events": agent_collapses,
        "agent_level_correction_events": agent_corrections,
        "total_cost": total_cost,
    }


def fmt_pct(x: float | None) -> str:
    if x is None or math.isnan(x):
        return "NA"
    return f"{x:.2f}%"


def main() -> None:
    alpha_rows = read_jsonl(ALPHA)
    debate_rows = read_jsonl(DEBATE)
    cross = json.loads(CROSS.read_text()) if CROSS.exists() else None
    alpha = summarize_alpha(alpha_rows)
    debate = summarize_debate(debate_rows)
    total_cost = alpha["total_cost"] + debate["total_cost"]
    if cross is not None:
        total_cost += float(cross.get("summary", {}).get("overall", {}).get("total_cost", 0.0) or 0.0)

    payload = {
        "model": "models/gemini-3.1-flash-lite-preview",
        "alpha": alpha,
        "debate": debate,
        "cross_benchmark_summary": cross["summary"] if cross else None,
        "actual_costs_usd": {
            "alpha": alpha["total_cost"],
            "debate": debate["total_cost"],
            "cross_benchmark_n50_each": 2.6323 if cross else None,
            "alpha_plus_debate": alpha["total_cost"] + debate["total_cost"],
        },
        "interpretation": (
            "This is a latest-Gemini Google-family holdout. It is strong as a "
            "boundary-condition / floor-collapse result, not as a new-family "
            "independent replication of the cross-model alpha-Ccond trend."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    cb = payload["cross_benchmark_summary"] or {}
    overall = cb.get("overall", {})
    lines = [
        "# Gemini 3.1 Flash-Lite Holdout Summary",
        "",
        "Model: `models/gemini-3.1-flash-lite-preview`.",
        "",
        "## MMLU-Pro High-FR Alpha",
        "",
        f"- Rows: {alpha['n_rows']} over {alpha['n_questions']} questions.",
        f"- Alpha total: {alpha['alpha_tot']:.4f}.",
        "- Alpha by condition: " + ", ".join(
            f"{k}={v:.4f}" for k, v in alpha["alpha_by_condition"].items()
        ) + ".",
        f"- Cost: ${alpha['total_cost']:.2f}.",
        "",
        "## MMLU-Pro High-FR Debate",
        "",
        f"- Debates: {debate['n_debates']}.",
        f"- Initial/final majority accuracy: {fmt_pct(debate['initial_accuracy_pct'])} -> {fmt_pct(debate['final_accuracy_pct'])}.",
        f"- Majority collapses: {debate['majority_collapses']} / {debate['init_majority_correct']} at-risk ({fmt_pct(debate['conditional_collapse_pct'])}).",
        f"- Majority corrections: {debate['majority_corrections']} / {debate['n_debates'] - debate['init_majority_correct']} initially-wrong ({fmt_pct(debate['conditional_correction_pct'])}).",
        f"- Majority flip rate: {fmt_pct(debate['majority_flip_rate_pct'])}.",
        f"- Agent-level collapse/correction events: {debate['agent_level_collapse_events']} / {debate['agent_level_correction_events']}.",
        f"- Cost: ${debate['total_cost']:.2f}.",
        "",
        "## Cross-Benchmark Sanity",
        "",
        f"- GPQA + TruthfulQA N: {overall.get('n', 'NA')}.",
        f"- Initial/final accuracy: {overall.get('initial_accuracy', 0):.1%} -> {overall.get('final_accuracy', 0):.1%}.",
        f"- Collapses: {overall.get('n_collapses', 'NA')} (Ccond={fmt_pct(overall.get('conditional_collapse_pct'))}).",
        f"- Mean alpha: {overall.get('mean_alpha', float('nan')):.4f}.",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "Practical paper use: include as a held-out latest-Gemini boundary condition; do not oversell it as strengthening the main cross-family correlation because the collapse rate is at the floor.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
