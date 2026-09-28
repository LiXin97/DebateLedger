"""Post-A6 R1/cascade replication from raw debate traces.

Reads existing ``abc_exp/results/debate_traces_*.jsonl`` files and computes
majority-level collapse/correction metrics. The parser accepts both trace
schemas seen in this project:

* ``round_traces`` as ``list[list[agent_dict]]`` with per-agent answers.
* ``round_traces`` as ``list[round_dict]`` with round summaries.

Outputs:
  - abc_exp/results/post_a6_r1_replication.json
  - abc_exp/results/POST_A6_R1_REPLICATION.md

Run:
  python -m abc_exp.experiments.post_a6_r1_replication
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

POST_A6_MODEL_MARKERS = (
    "deepseek-v4-flash",
    "gemma-4-31b-it",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.6-27b-fp8",
    "qwen3.6-35b-a3b-fp8",
)


@dataclass(frozen=True)
class DebateRecord:
    source_file: str
    model: str
    debate_id: str
    correct_label: str | None
    initial_answers: tuple[str, ...]
    final_answers: tuple[str, ...]
    round_majorities: tuple[str | None, ...]
    initial_majority: str | None
    final_majority: str | None
    initial_majority_correct: bool | None
    final_majority_correct: bool | None
    collapse_onset_round: int | None
    cohort: str

    @property
    def collapsed(self) -> bool:
        return self.initial_majority_correct is True and self.final_majority_correct is False

    @property
    def corrected(self) -> bool:
        return self.initial_majority_correct is False and self.final_majority_correct is True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def clean_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.upper() if text else None


def answer_list(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    out = [clean_answer(v) for v in values]
    return tuple(v for v in out if v is not None)


def majority_vote(answers: tuple[str, ...] | list[str]) -> tuple[str | None, int, int, bool]:
    cleaned = [clean_answer(a) for a in answers]
    cleaned = [a for a in cleaned if a is not None]
    if not cleaned:
        return None, 0, 0, False
    counts = Counter(cleaned)
    top_count = max(counts.values())
    winners = sorted(a for a, n in counts.items() if n == top_count)
    if len(winners) != 1:
        return None, top_count, len(cleaned), True
    return winners[0], top_count, len(cleaned), False


def bool_from_any(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        current: Any = row
        found = True
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found:
            return current
    return None


def model_name(row: dict[str, Any], path: Path) -> str:
    for key in ("model_name", "model", "openrouter_model_id", "model_id"):
        value = row.get(key)
        if value:
            return str(value)
    return path.stem.removeprefix("debate_traces_")


def normalize_round_traces(row: dict[str, Any]) -> list[dict[str, Any]]:
    traces = row.get("round_traces") or row.get("rounds") or []
    if not isinstance(traces, list):
        return []

    rounds: list[dict[str, Any]] = []
    for idx, trace in enumerate(traces, start=1):
        round_num = idx
        answers: tuple[str, ...] = ()
        explicit_majority: str | None = None

        if isinstance(trace, list):
            agents = [a for a in trace if isinstance(a, dict)]
            if agents:
                round_num = int(first_present(agents[0], ("round_num", "round")) or idx)
            answers = tuple(a for a in (clean_answer(a.get("answer")) for a in agents) if a is not None)
        elif isinstance(trace, dict):
            round_num = int(trace.get("round") or trace.get("round_num") or idx)
            explicit_majority = clean_answer(
                trace.get("majority")
                or trace.get("majority_answer")
                or trace.get("final_majority")
            )
            raw_answers = (
                trace.get("answers")
                or trace.get("agent_answers")
                or trace.get("final_answers")
                or []
            )
            answers = answer_list(raw_answers)

        majority, _, _, _ = majority_vote(answers)
        rounds.append({
            "round": round_num,
            "answers": answers,
            "majority": majority or explicit_majority,
        })

    rounds.sort(key=lambda r: r["round"])
    return rounds


def infer_correctness(
    row: dict[str, Any],
    majority: str | None,
    correct_label: str | None,
    fallback_keys: tuple[str, ...],
) -> bool | None:
    if majority is not None and correct_label is not None:
        return majority == correct_label

    explicit = bool_from_any(first_present(row, fallback_keys))
    if explicit is not None:
        return explicit

    count_key = "outcome.initial_n_correct" if "initial" in fallback_keys[0] else "outcome.final_n_correct"
    n_correct = first_present(row, (count_key,))
    if isinstance(n_correct, int):
        n_agents = len(row.get("initial_answers") or row.get("final_answers") or []) or 3
        return n_correct >= (n_agents // 2 + 1)
    return None


def cohort_for(path: Path, model: str, n_rows_in_file: int) -> str:
    haystack = f"{path.name} {model}".lower()
    if n_rows_in_file == 200 and any(marker in haystack for marker in POST_A6_MODEL_MARKERS):
        return "post_a6_200row"
    return "legacy_or_partial"


def normalize_record(row: dict[str, Any], path: Path, n_rows_in_file: int) -> DebateRecord:
    correct_label = clean_answer(row.get("correct_label") or row.get("answer") or row.get("gold"))
    initial_answers = answer_list(row.get("initial_answers"))
    final_answers = answer_list(row.get("final_answers"))

    rounds = normalize_round_traces(row)
    round_majorities = tuple(r["majority"] for r in rounds)

    initial_majority, _, _, _ = majority_vote(initial_answers)
    final_majority, _, _, _ = majority_vote(final_answers)
    if final_majority is None and round_majorities:
        final_majority = round_majorities[-1]

    initial_correct = infer_correctness(
        row,
        initial_majority,
        correct_label,
        ("initial_correct", "outcome.initial_correct", "majority_initial_correct"),
    )
    final_correct = infer_correctness(
        row,
        final_majority,
        correct_label,
        ("final_correct", "outcome.final_correct", "majority_correct"),
    )

    onset: int | None = None
    if initial_correct is True and correct_label is not None:
        for round_info in rounds:
            majority = round_info["majority"]
            if majority is not None and majority != correct_label:
                onset = int(round_info["round"])
                break
        if onset is None and final_correct is False:
            # Collapse is known from final answers, but not localizable from traces.
            onset = None

    model = model_name(row, path)
    return DebateRecord(
        source_file=path.name,
        model=model,
        debate_id=str(row.get("debate_id") or row.get("question_id") or ""),
        correct_label=correct_label,
        initial_answers=initial_answers,
        final_answers=final_answers,
        round_majorities=round_majorities,
        initial_majority=initial_majority,
        final_majority=final_majority,
        initial_majority_correct=initial_correct,
        final_majority_correct=final_correct,
        collapse_onset_round=onset,
        cohort=cohort_for(path, model, n_rows_in_file),
    )


def pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return 100.0 * num / den


def summarize(records: list[DebateRecord]) -> dict[str, Any]:
    n = len(records)
    init_correct = sum(r.initial_majority_correct is True for r in records)
    final_correct = sum(r.final_majority_correct is True for r in records)
    collapses = sum(r.collapsed for r in records)
    corrections = sum(r.corrected for r in records)
    onset_counts = Counter(str(r.collapse_onset_round) for r in records if r.collapsed and r.collapse_onset_round is not None)
    unlocalized = sum(r.collapsed and r.collapse_onset_round is None for r in records)

    return {
        "n_debates": n,
        "init_majority_correct_count": init_correct,
        "init_majority_accuracy_pct": pct(init_correct, n),
        "collapses": collapses,
        "conditional_collapse_pct": pct(collapses, init_correct),
        "corrections": corrections,
        "correction_pct_of_initial_wrong": pct(corrections, n - init_correct),
        "final_correct_count": final_correct,
        "final_accuracy_pct": pct(final_correct, n),
        "collapse_onset_round_counts": dict(sorted(onset_counts.items(), key=lambda kv: int(kv[0]))),
        "collapse_onset_round_pct_of_collapses": {
            k: pct(v, collapses) for k, v in sorted(onset_counts.items(), key=lambda kv: int(kv[0]))
        },
        "collapse_onset_unlocalized": unlocalized,
    }


def grouped(records: list[DebateRecord], attr: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[DebateRecord]] = defaultdict(list)
    for record in records:
        groups[str(getattr(record, attr))].append(record)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.1f}%"


def fmt_onsets(summary: dict[str, Any]) -> str:
    counts = summary["collapse_onset_round_counts"]
    if not counts:
        return "-"
    parts = [f"R{round_num}: {count}" for round_num, count in counts.items()]
    if summary["collapse_onset_unlocalized"]:
        parts.append(f"unlocalized: {summary['collapse_onset_unlocalized']}")
    return ", ".join(parts)


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Post-A6 R1 Replication",
        "",
        "Majority-level metrics from `debate_traces_*.jsonl`. Collapse onset uses natural debate-round numbering: first debate round is R1.",
        "",
        "## Cohort Comparison",
        "",
        "| Cohort | N | Init Maj Correct | Collapses | Cond Collapse | Corrections | Final Acc | Collapse Onset |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    cohort_labels = {
        "post_a6_200row": "Post-A6 200-row traces",
        "legacy_or_partial": "Legacy/partial traces",
    }
    for cohort, summary in payload["by_cohort"].items():
        lines.append(
            f"| {cohort_labels.get(cohort, cohort)} | {summary['n_debates']} | "
            f"{summary['init_majority_correct_count']} | {summary['collapses']} | "
            f"{fmt_pct(summary['conditional_collapse_pct'])} | {summary['corrections']} | "
            f"{fmt_pct(summary['final_accuracy_pct'])} | {fmt_onsets(summary)} |"
        )

    pooled = payload["pooled"]
    lines += [
        "",
        "## Pooled All Traces",
        "",
        f"- N debates: {pooled['n_debates']}",
        f"- Initial majority correct: {pooled['init_majority_correct_count']} ({fmt_pct(pooled['init_majority_accuracy_pct'])})",
        f"- Collapses: {pooled['collapses']} ({fmt_pct(pooled['conditional_collapse_pct'])} conditional on initial majority correct)",
        f"- Corrections: {pooled['corrections']} ({fmt_pct(pooled['correction_pct_of_initial_wrong'])} of initial-majority-wrong debates)",
        f"- Final accuracy: {pooled['final_correct_count']} / {pooled['n_debates']} ({fmt_pct(pooled['final_accuracy_pct'])})",
        f"- Collapse onset: {fmt_onsets(pooled)}",
        "",
        "## Per Model",
        "",
        "| Model | Cohort | N | Init Maj Correct | Collapses | Cond Collapse | Corrections | Final Acc | Collapse Onset |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    model_cohorts = payload["model_cohorts"]
    for model, summary in payload["by_model"].items():
        lines.append(
            f"| {model} | {model_cohorts.get(model, '')} | {summary['n_debates']} | "
            f"{summary['init_majority_correct_count']} | {summary['collapses']} | "
            f"{fmt_pct(summary['conditional_collapse_pct'])} | {summary['corrections']} | "
            f"{fmt_pct(summary['final_accuracy_pct'])} | {fmt_onsets(summary)} |"
        )

    lines += [
        "",
        "## Sources",
        "",
        "| File | Rows | Cohort |",
        "|---|---:|---|",
    ]
    for item in payload["sources"]:
        lines.append(f"| `{item['file']}` | {item['rows']} | {item['cohort']} |")

    return "\n".join(lines) + "\n"


def build_payload(results_dir: Path) -> dict[str, Any]:
    records: list[DebateRecord] = []
    sources: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("debate_traces_*.jsonl")):
        rows = load_jsonl(path)
        file_records = [normalize_record(row, path, len(rows)) for row in rows]
        records.extend(file_records)
        cohort = file_records[0].cohort if file_records else cohort_for(path, path.stem, len(rows))
        sources.append({"file": path.name, "rows": len(rows), "cohort": cohort})

    model_cohorts: dict[str, str] = {}
    for record in records:
        existing = model_cohorts.get(record.model)
        if existing is None:
            model_cohorts[record.model] = record.cohort
        elif existing != record.cohort:
            model_cohorts[record.model] = "mixed"

    return {
        "metric_definitions": {
            "collapse": "initial majority correct and final majority incorrect",
            "correction": "initial majority incorrect and final majority correct",
            "collapse_onset_round": "first debate round whose majority is not the correct label; natural numbering starts at R1",
            "post_a6_200row": "200-row deepseek/gemma/qwen3.5/qwen3.6 trace files",
        },
        "sources": sources,
        "pooled": summarize(records),
        "by_cohort": grouped(records, "cohort"),
        "by_model": grouped(records, "model"),
        "model_cohorts": model_cohorts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--json-out", type=Path, default=RESULTS / "post_a6_r1_replication.json")
    parser.add_argument("--md-out", type=Path, default=RESULTS / "POST_A6_R1_REPLICATION.md")
    args = parser.parse_args()

    payload = build_payload(args.results_dir)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(markdown_report(payload))
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
