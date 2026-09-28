"""Summarize a post-hoc model alpha/debate holdout pair.

This is intentionally generic: it reads one S/A alpha JSONL and one debate-trace
JSONL, then writes compact JSON/Markdown artifacts for reviewer-facing triage.
It does not modify sealed headline tables.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def majority(values: list[str | None]) -> str | None:
    clean = [v for v in values if v]
    if not clean:
        return None
    counts = Counter(clean)
    m = max(counts.values())
    return sorted(v for v, c in counts.items() if c == m)[0]


def pct(num: float, den: float) -> float | None:
    return 100.0 * num / den if den else None


def fmt_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.2f}%"


def row_cost(row: dict[str, Any]) -> float:
    if row.get("total_cost"):
        return float(row["total_cost"])
    total = 0.0
    for state in row.get("initial_states", []):
        total += float((state.get("usage") or {}).get("cost") or 0.0)
    for round_states in row.get("round_traces", []):
        for state in round_states:
            total += float((state.get("usage") or {}).get("cost") or 0.0)
    return total


def summarize_alpha(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[float]] = defaultdict(list)
    by_question: dict[str, list[float]] = defaultdict(list)
    by_provider = Counter()
    triple_counts = Counter()
    for row in rows:
        alpha = float(row.get("alpha_total", row.get("flip_rate", 0.0)) or 0.0)
        by_condition[row.get("condition", "unknown")].append(alpha)
        by_question[row.get("question_id", "unknown")].append(alpha)
        by_provider.update(row.get("provider_distribution") or {})
        triple_counts[(
            row.get("question_id", "unknown"),
            row.get("condition", "unknown"),
            int(row.get("agent_idx", -1) if row.get("agent_idx") is not None else -1),
        )] += 1
    values = [v for vals in by_condition.values() for v in vals]
    conditions = sorted({row.get("condition", "unknown") for row in rows})
    agent_indices = sorted({int(row.get("agent_idx", -1) if row.get("agent_idx") is not None else -1) for row in rows})
    expected_rows = len(by_question) * len(conditions) * len(agent_indices) if by_question else 0
    duplicate_triples = sum(count - 1 for count in triple_counts.values() if count > 1)
    return {
        "n_rows": len(rows),
        "n_questions": len(by_question),
        "n_expected_rows_observed_grid": expected_rows,
        "missing_rows_observed_grid": max(expected_rows - len(triple_counts), 0),
        "duplicate_triples": duplicate_triples,
        "conditions": conditions,
        "agent_indices": agent_indices,
        "condition_counts": dict(Counter(row.get("condition", "unknown") for row in rows)),
        "alpha_tot": sum(values) / len(values) if values else math.nan,
        "alpha_by_condition": {
            key: sum(vals) / len(vals) for key, vals in sorted(by_condition.items()) if vals
        },
        "question_alpha_mean": (
            sum(sum(vals) / len(vals) for vals in by_question.values()) / len(by_question)
            if by_question else math.nan
        ),
        "total_cost": sum(float(row.get("total_cost") or 0.0) for row in rows),
        "provider_distribution": dict(by_provider),
    }


def summarize_debate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    init_correct = 0
    final_correct = 0
    collapses = 0
    corrections = 0
    majority_flips = 0
    agent_collapses = 0
    agent_corrections = 0
    by_provider = Counter()
    question_counts = Counter()
    for row in rows:
        question_counts[row.get("question_id", "unknown")] += 1
        correct = (row.get("correct_label") or "").upper()
        init = majority(row.get("initial_answers") or [])
        final = row.get("majority_answer") or majority(row.get("final_answers") or [])
        init_ok = bool(init and init.upper() == correct)
        final_ok = bool(final and final.upper() == correct)
        init_correct += int(init_ok)
        final_correct += int(final_ok)
        collapses += int(init_ok and not final_ok)
        corrections += int((not init_ok) and final_ok)
        majority_flips += int(bool(init and final and init.upper() != final.upper()))
        agent_collapses += int(bool((row.get("outcome") or {}).get("collapse")))
        agent_corrections += int(bool((row.get("outcome") or {}).get("correction")))
        by_provider.update(row.get("provider_distribution") or {})
    n = len(rows)
    return {
        "n_debates": n,
        "n_questions": len(question_counts),
        "duplicate_questions": sum(count - 1 for count in question_counts.values() if count > 1),
        "init_majority_correct": init_correct,
        "final_majority_correct": final_correct,
        "initial_accuracy_pct": pct(init_correct, n),
        "final_accuracy_pct": pct(final_correct, n),
        "majority_collapses": collapses,
        "majority_corrections": corrections,
        "conditional_collapse_pct": pct(collapses, init_correct),
        "conditional_correction_pct": pct(corrections, n - init_correct),
        "majority_flip_rate_pct": pct(majority_flips, n),
        "agent_level_collapse_events": agent_collapses,
        "agent_level_correction_events": agent_corrections,
        "total_cost": sum(row_cost(row) for row in rows),
        "provider_distribution": dict(by_provider),
    }


def summarize_pair_qc(alpha_rows: list[dict[str, Any]], debate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    alpha_questions = {row.get("question_id", "unknown") for row in alpha_rows}
    debate_questions = {row.get("question_id", "unknown") for row in debate_rows}
    overlap = alpha_questions & debate_questions
    return {
        "alpha_questions": len(alpha_questions),
        "debate_questions": len(debate_questions),
        "overlap_questions": len(overlap),
        "alpha_only_questions": len(alpha_questions - debate_questions),
        "debate_only_questions": len(debate_questions - alpha_questions),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    alpha = payload["alpha"]
    debate = payload["debate"]
    pair_qc = payload["pair_qc"]
    lines = [
        f"# {payload['model_label']} Holdout Summary",
        "",
        "This is a post-hoc holdout/expansion artifact and is not folded into the sealed headline table unless explicitly relabeled.",
        "",
        "## Alpha",
        "",
        f"- Rows: {alpha['n_rows']} over {alpha['n_questions']} questions.",
        f"- Grid QC: {alpha['missing_rows_observed_grid']} missing rows and {alpha['duplicate_triples']} duplicate triples on the observed condition-agent grid.",
        f"- Alpha total: {alpha['alpha_tot']:.4f}.",
        "- Alpha by condition: " + ", ".join(
            f"{key}={value:.4f}" for key, value in alpha["alpha_by_condition"].items()
        ) + ".",
        f"- Cost: ${alpha['total_cost']:.4f}.",
        "",
        "## Debate",
        "",
        f"- Debates: {debate['n_debates']}.",
        f"- Debate QC: {debate['duplicate_questions']} duplicate question rows.",
        f"- Initial/final majority accuracy: {fmt_pct(debate['initial_accuracy_pct'])} -> {fmt_pct(debate['final_accuracy_pct'])}.",
        f"- Majority collapses: {debate['majority_collapses']} / {debate['init_majority_correct']} at-risk ({fmt_pct(debate['conditional_collapse_pct'])}).",
        f"- Majority corrections: {debate['majority_corrections']} / {debate['n_debates'] - debate['init_majority_correct']} initially-wrong ({fmt_pct(debate['conditional_correction_pct'])}).",
        f"- Majority flip rate: {fmt_pct(debate['majority_flip_rate_pct'])}.",
        f"- Agent-level collapse/correction events: {debate['agent_level_collapse_events']} / {debate['agent_level_correction_events']}.",
        f"- Cost: ${debate['total_cost']:.4f}.",
        "",
        "## Provider Provenance",
        "",
        f"- Alpha providers: `{alpha['provider_distribution']}`.",
        f"- Debate providers: `{debate['provider_distribution']}`.",
        "",
        "## Pair QC",
        "",
        f"- Alpha/debate question overlap: {pair_qc['overlap_questions']} shared questions; {pair_qc['alpha_only_questions']} alpha-only; {pair_qc['debate_only_questions']} debate-only.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize alpha/debate holdout pair")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--alpha", required=True, type=Path)
    parser.add_argument("--debate", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    alpha_rows = read_jsonl(args.alpha)
    debate_rows = read_jsonl(args.debate)
    payload = {
        "model_label": args.model_label,
        "alpha_path": str(args.alpha),
        "debate_path": str(args.debate),
        "alpha": summarize_alpha(alpha_rows),
        "debate": summarize_debate(debate_rows),
        "pair_qc": summarize_pair_qc(alpha_rows, debate_rows),
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(args.out_md, payload)


if __name__ == "__main__":
    main()
