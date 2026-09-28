"""Audit transition-cell stability under a stricter MCQ parser.

This zero-API audit re-extracts answers from checked-in debate traces using a
stricter parser that removes the primary parser's permissive last-valid-letter
fallback. It then recomputes initial/final majorities and the four transition
cells. The goal is not to replace the primary parser, but to bound how often the
reported collapse/correction cell is an artifact of fallback extraction.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "abc_exp" / "results"
OUT_JSON = RES / "parser_cell_stability_audit.json"
OUT_MD = RES / "PARSER_CELL_STABILITY_AUDIT.md"

TRACE_FILES = {
    "deepseek-v4-flash": "debate_traces_openrouter_deepseek-v4-flash.jsonl",
    "gemma-4-31b-it-awq": "debate_traces_vllm_gemma-4-31b-it.jsonl",
    "qwen3.5-4b": "debate_traces_vllm_qwen3.5-4b.jsonl",
    "qwen3.5-9b": "debate_traces_vllm_qwen3.5-9b.jsonl",
    "qwen3.6-27b-fp8": "debate_traces_vllm_qwen3.6-27b-fp8.jsonl",
    "qwen3.6-35b-a3b-fp8": "debate_traces_vllm_qwen3.6-35b-a3b-fp8.jsonl",
    "qwen3-32b-a7-n200": "debate_traces_vllm_qwen3_32b_a7_n200.jsonl",
    "gemini-3.1-flash-lite-n200": "debate_traces_gemini_3_1_flash_lite_n200.jsonl",
    "mistral-small-4-n20": "debate_traces_openrouter_mistral_small_4_n20.jsonl",
}

FINAL_TAG_RE = re.compile(r"Final\s+Answer\s*:\s*([A-J])\b", re.IGNORECASE)
FINAL_LINE_RE = re.compile(r"^[\(\*\s]*([A-J])[\)\.\*\s]*$", re.IGNORECASE)
ANSWER_PATTERNS = [
    re.compile(r"(?:the\s+)?answer\s+is\s*:?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"(?:I\s+)?(?:choose|select|pick)\s+(?:option\s+)?([A-J])\b", re.IGNORECASE),
    re.compile(r"(?:correct\s+)?answer\s*:\s*([A-J])\b", re.IGNORECASE),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def no_fallback_extract(text: str | None, valid_labels: list[str] | None = None) -> str | None:
    if not text:
        return None
    valid = set(valid_labels or list("ABCDEFGHIJ"))
    tags = [m.group(1).upper() for m in FINAL_TAG_RE.finditer(text)]
    for tag in reversed(tags):
        if tag in valid:
            return tag
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match and match.group(1).upper() in valid:
            return match.group(1).upper()
    for line in reversed([ln.strip() for ln in text.strip().splitlines() if ln.strip()]):
        m = FINAL_LINE_RE.match(line)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
    return None


def final_tag_extract(text: str | None, valid_labels: list[str] | None = None) -> str | None:
    if not text:
        return None
    valid = set(valid_labels or list("ABCDEFGHIJ"))
    tags = [m.group(1).upper() for m in FINAL_TAG_RE.finditer(text)]
    for tag in reversed(tags):
        if tag in valid:
            return tag
    for line in reversed([ln.strip() for ln in text.strip().splitlines() if ln.strip()]):
        m = FINAL_LINE_RE.match(line)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
        break
    return None


def majority(values: list[str | None]) -> str | None:
    clean = [v for v in values if v]
    if not clean:
        return None
    counts = Counter(clean)
    max_count = max(counts.values())
    return sorted(v for v, c in counts.items() if c == max_count)[0]


def majority_tie(values: list[str | None]) -> bool:
    clean = [v for v in values if v]
    if not clean:
        return False
    counts = Counter(clean)
    max_count = max(counts.values())
    return sum(1 for c in counts.values() if c == max_count) > 1


def cell(initial_majority: str | None, final_majority: str | None, correct: str) -> str:
    if not initial_majority or not final_majority:
        return "unlabeled"
    init_correct = initial_majority == correct
    final_correct = final_majority == correct
    if init_correct and final_correct:
        return "kept_correct"
    if init_correct and not final_correct:
        return "collapse"
    if not init_correct and final_correct:
        return "correction"
    return "remained_wrong"


def state_answers(states: list[dict[str, Any]], mode: str) -> list[str | None]:
    if mode == "no_fallback":
        return [no_fallback_extract(s.get("reasoning")) for s in states]
    if mode == "final_tag_only":
        return [final_tag_extract(s.get("reasoning")) for s in states]
    return [s.get("answer") for s in states]


def final_states(row: dict[str, Any]) -> list[dict[str, Any]]:
    rounds = row.get("round_traces") or row.get("round_states") or []
    if rounds:
        return rounds[-1]
    return row.get("final_states") or []


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [r for r in records if r["strict_cell"] != "unlabeled"]
    primary_collapses = [r for r in records if r["primary_cell"] == "collapse"]
    primary_corrections = [r for r in records if r["primary_cell"] == "correction"]
    changed_labeled = [r for r in labeled if r["primary_cell"] != r["strict_cell"]]
    strict_unlabeled_from_primary_collapses = [
        r for r in primary_collapses if r["strict_cell"] == "unlabeled"
    ]
    strict_unlabeled_from_primary_corrections = [
        r for r in primary_corrections if r["strict_cell"] == "unlabeled"
    ]
    collapse_changed = [r for r in labeled if (r["primary_cell"] == "collapse") != (r["strict_cell"] == "collapse")]
    correction_changed = [r for r in labeled if (r["primary_cell"] == "correction") != (r["strict_cell"] == "correction")]
    primary_initial_ties = [r for r in records if r.get("primary_initial_tie")]
    primary_final_ties = [r for r in records if r.get("primary_final_tie")]
    primary_any_ties = [r for r in records if r.get("primary_initial_tie") or r.get("primary_final_tie")]
    primary_tie_state_reductions = len(primary_initial_ties) + len(primary_final_ties)
    primary_state_reductions = 2 * len(records)
    return {
        "n_rows": len(records),
        "n_strict_labeled": len(labeled),
        "strict_unlabeled": len(records) - len(labeled),
        "strict_unlabeled_rate": (len(records) - len(labeled)) / len(records) if records else None,
        "primary_cell_counts": dict(Counter(r["primary_cell"] for r in records)),
        "strict_cell_counts": dict(Counter(r["strict_cell"] for r in records)),
        "labeled_cell_changed": len(changed_labeled),
        "labeled_cell_changed_rate": len(changed_labeled) / len(labeled) if labeled else None,
        "collapse_label_changed_labeled": len(collapse_changed),
        "correction_label_changed_labeled": len(correction_changed),
        "primary_collapses": len(primary_collapses),
        "primary_corrections": len(primary_corrections),
        "primary_collapses_unlabeled_by_strict": len(strict_unlabeled_from_primary_collapses),
        "primary_corrections_unlabeled_by_strict": len(strict_unlabeled_from_primary_corrections),
        "primary_initial_ties": len(primary_initial_ties),
        "primary_final_ties": len(primary_final_ties),
        "primary_any_tie_rows": len(primary_any_ties),
        "primary_tie_state_reductions": primary_tie_state_reductions,
        "primary_state_reductions": primary_state_reductions,
        "primary_tie_state_rate": primary_tie_state_reductions / primary_state_reductions if primary_state_reductions else None,
        "primary_any_tie_row_rate": len(primary_any_ties) / len(records) if records else None,
    }


def audit_file(label: str, fname: str, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    for row in read_jsonl(RES / fname):
        correct = str(row.get("correct_label") or "").upper()
        init_states = row.get("initial_states") or []
        fin_states = final_states(row)
        if not correct or not init_states or not fin_states:
            continue
        primary_init = majority(state_answers(init_states, mode="primary"))
        primary_final = majority(state_answers(fin_states, mode="primary"))
        primary_initial_tie = majority_tie(state_answers(init_states, mode="primary"))
        primary_final_tie = majority_tie(state_answers(fin_states, mode="primary"))
        strict_init = majority(state_answers(init_states, mode=mode))
        strict_final = majority(state_answers(fin_states, mode=mode))
        rec = {
            "source": label,
            "file": fname,
            "question_id": row.get("question_id"),
            "correct_label": correct,
            "primary_initial_majority": primary_init,
            "primary_final_majority": primary_final,
            "primary_initial_tie": primary_initial_tie,
            "primary_final_tie": primary_final_tie,
            "strict_initial_majority": strict_init,
            "strict_final_majority": strict_final,
            "primary_cell": cell(primary_init, primary_final, correct),
            "strict_cell": cell(strict_init, strict_final, correct),
            "strict_initial_missing_agents": sum(v is None for v in state_answers(init_states, mode=mode)),
            "strict_final_missing_agents": sum(v is None for v in state_answers(fin_states, mode=mode)),
        }
        records.append(rec)
    summary = summarize_records(records)
    summary.update({"source": label, "file": fname})
    return summary, records


def write_markdown(payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    lines = [
        "# Parser Cell-Stability Audit",
        "",
        "Zero-API audit over checked-in debate traces. The reported strict parser keeps explicit answer patterns but removes the primary parser's permissive last-valid-letter fallback. We also compute a harsher final-tag-only diagnostic in JSON.",
        "",
        "## Overall",
        "",
        f"- Rows audited: **{overall['n_rows']}**.",
        f"- Strict parser labeled both initial and final majorities for **{overall['n_strict_labeled']}** rows; **{overall['strict_unlabeled']}** rows were unlabeled ({100 * overall['strict_unlabeled_rate']:.2f}%).",
        f"- Among strict-labeled rows, transition-cell membership changed for **{overall['labeled_cell_changed']}** rows ({100 * overall['labeled_cell_changed_rate']:.2f}%).",
        f"- Collapse label changed among strict-labeled rows: **{overall['collapse_label_changed_labeled']}**; correction label changed: **{overall['correction_label_changed_labeled']}**.",
        f"- Primary majority reductions used alphabetic tie-breaking for **{overall['primary_tie_state_reductions']} / {overall['primary_state_reductions']}** initial/final state reductions ({100 * overall['primary_tie_state_rate']:.2f}%); **{overall['primary_any_tie_rows']} / {overall['n_rows']}** rows had at least one primary initial/final tie ({100 * overall['primary_any_tie_row_rate']:.2f}%).",
        f"- Primary collapses/corrections in this open trace subset: **{overall['primary_collapses']}** / **{overall['primary_corrections']}**.",
        f"- Primary collapses/corrections made unlabeled by strict parsing: **{overall['primary_collapses_unlabeled_by_strict']}** / **{overall['primary_corrections_unlabeled_by_strict']}**.",
        "",
        "## Per Trace",
        "",
        "| Source | Rows | Primary tie states | Any tie rows | Strict unlabeled | Labeled cell changes | Collapse changes | Correction changes | Primary C/R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_file"]:
        lines.append(
            f"| `{row['source']}` | {row['n_rows']} | {row['primary_tie_state_reductions']} | "
            f"{row['primary_any_tie_rows']} | {row['strict_unlabeled']} | {row['labeled_cell_changed']} | "
            f"{row['collapse_label_changed_labeled']} | {row['correction_label_changed_labeled']} | "
            f"{row['primary_collapses']}/{row['primary_corrections']} |"
        )
    lines.extend([
        "",
        "Interpretation: this is a stability check for trace files present in the open checkout. It does not reconstruct gated full transcripts or sealed lanes that are only available as aggregate transition tables.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    audits = {}
    for mode in ["no_fallback", "final_tag_only"]:
        per_file = []
        all_records = []
        for label, fname in TRACE_FILES.items():
            summary, records = audit_file(label, fname, mode=mode)
            if summary["n_rows"]:
                per_file.append(summary)
                all_records.extend(records)
        audits[mode] = {
            "overall": summarize_records(all_records),
            "per_file": per_file,
            "changed_examples": [
                r for r in all_records if r["strict_cell"] != "unlabeled" and r["primary_cell"] != r["strict_cell"]
            ][:20],
        }
    payload = {
        "generated_by": "abc_exp/experiments/parser_cell_stability_audit.py",
        "reported_mode": "no_fallback",
        "strict_parser": "Final Answer, answer-is/choose/correct-answer, or standalone option-line patterns; no last-valid-letter fallback",
        "audits": audits,
        "overall": audits["no_fallback"]["overall"],
        "per_file": audits["no_fallback"]["per_file"],
        "changed_examples": audits["no_fallback"]["changed_examples"],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    write_markdown(payload)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
