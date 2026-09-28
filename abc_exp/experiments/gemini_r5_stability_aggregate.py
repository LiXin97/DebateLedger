"""Aggregate Gemini R5 same-prompt alpha stability panels.

The per-day runner writes one JSONL record per question-repeat pair. This
script reads one or more panels and reports within-panel stability plus
cross-panel drift on the shared question IDs. It is zero-API and intentionally
keeps R5 separate from the sealed headline artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "abc_exp/results"


def panel_name(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl"):
        return name[:-6]
    if name.endswith(".json"):
        return name[:-5]
    return path.stem


def read_panel(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize_panel(name: str, records: list[dict]) -> dict:
    by_q: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_q[rec["question_id"]].append(rec)

    per_question = []
    for qid, rows in sorted(by_q.items()):
        alphas = [float(r["alpha"]) for r in rows if r.get("alpha") is not None]
        answers = [r.get("initial_answer") for r in rows]
        answer_counts = Counter(answers)
        per_question.append(
            {
                "question_id": qid,
                "n_repeats": len(rows),
                "initial_answers": answers,
                "majority_initial_answer": answer_counts.most_common(1)[0][0] if answer_counts else None,
                "n_unique_initial_answers": len(set(answers)),
                "alpha_values": alphas,
                "alpha_mean": mean(alphas) if alphas else math.nan,
                "alpha_range": max(alphas) - min(alphas) if len(alphas) >= 2 else 0.0,
                "post_answer_none": sum(r.get("post_answer_none") or 0 for r in rows),
            }
        )

    alpha_ranges = [r["alpha_range"] for r in per_question if math.isfinite(r["alpha_range"])]
    total_probe_trials = sum((r.get("n_probes") or 0) for r in records)
    total_post_none = sum((r.get("post_answer_none") or 0) for r in records)
    return {
        "panel": name,
        "model": records[0].get("model") if records else None,
        "n_records": len(records),
        "n_questions": len(per_question),
        "repeats_per_question_min": min((r["n_repeats"] for r in per_question), default=0),
        "repeats_per_question_max": max((r["n_repeats"] for r in per_question), default=0),
        "questions_with_initial_answer_instability": sum(
            r["n_unique_initial_answers"] > 1 for r in per_question
        ),
        "mean_alpha_range_within_question": mean(alpha_ranges) if alpha_ranges else math.nan,
        "max_alpha_range_within_question": max(alpha_ranges) if alpha_ranges else math.nan,
        "probe_post_answer_none_rate": total_post_none / total_probe_trials if total_probe_trials else math.nan,
        "total_cost": sum(float(r.get("initial_cost") or 0) + float(r.get("probe_cost") or 0) for r in records),
        "per_question": per_question,
    }


def compare_panels(a: dict, b: dict) -> dict:
    qa = {r["question_id"]: r for r in a["per_question"]}
    qb = {r["question_id"]: r for r in b["per_question"]}
    shared = sorted(set(qa) & set(qb))
    alpha_pairs = [
        (qa[q]["alpha_mean"], qb[q]["alpha_mean"])
        for q in shared
        if math.isfinite(qa[q]["alpha_mean"]) and math.isfinite(qb[q]["alpha_mean"])
    ]
    if len(alpha_pairs) >= 3 and len({x for x, _ in alpha_pairs}) >= 2 and len({y for _, y in alpha_pairs}) >= 2:
        rho, p = stats.spearmanr([x for x, _ in alpha_pairs], [y for _, y in alpha_pairs])
        rho = float(rho)
        p = float(p)
    else:
        rho = None
        p = None
    deltas = [abs(x - y) for x, y in alpha_pairs]
    answer_changes = sum(
        qa[q]["majority_initial_answer"] != qb[q]["majority_initial_answer"] for q in shared
    )
    return {
        "panel_a": a["panel"],
        "panel_b": b["panel"],
        "shared_questions": len(shared),
        "alpha_pair_count": len(alpha_pairs),
        "alpha_mean_spearman": rho,
        "alpha_mean_spearman_p_two_sided": p,
        "mean_abs_alpha_mean_delta": mean(deltas) if deltas else math.nan,
        "max_abs_alpha_mean_delta": max(deltas) if deltas else math.nan,
        "initial_answer_majority_changes": answer_changes,
    }


def fmt_float(x, digits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def write_markdown(path: Path, out: dict) -> None:
    lines = [
        "# Gemini R5 Stability Aggregate",
        "",
        "Zero-API aggregation of same-prompt Gemini alpha panels. A true R5 non-determinism check requires repeated panels on separate calendar days; same-day comparisons are reported as smoke/stability diagnostics only.",
        "",
        "## Panels",
        "",
        "| Panel | records | questions | repeats | init-answer unstable | mean alpha range | max alpha range | post-answer None | cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in out["panels"]:
        repeats = f"{s['repeats_per_question_min']}--{s['repeats_per_question_max']}"
        lines.append(
            f"| `{s['panel']}` | {s['n_records']} | {s['n_questions']} | {repeats} | "
            f"{s['questions_with_initial_answer_instability']} | "
            f"{fmt_float(s['mean_alpha_range_within_question'])} | "
            f"{fmt_float(s['max_alpha_range_within_question'])} | "
            f"{s['probe_post_answer_none_rate']:.2%} | ${s['total_cost']:.4f} |"
        )
    lines += ["", "## Pairwise Shared-Question Drift", ""]
    if out["pairwise"]:
        lines += [
            "| Panel A | Panel B | shared q | alpha pairs | Spearman | mean abs delta | max abs delta | init-answer changes |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for c in out["pairwise"]:
            lines.append(
                f"| `{c['panel_a']}` | `{c['panel_b']}` | {c['shared_questions']} | {c['alpha_pair_count']} | "
                f"{fmt_float(c['alpha_mean_spearman'])} | {fmt_float(c['mean_abs_alpha_mean_delta'])} | "
                f"{fmt_float(c['max_abs_alpha_mean_delta'])} | {c['initial_answer_majority_changes']} |"
            )
    else:
        lines.append("Only one panel was supplied; cross-panel drift is not estimable yet.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", nargs="+", required=True, help="R5 JSONL panel files")
    parser.add_argument("--output-stem", default="r5_gemini_alpha_panel_aggregate")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    out_json = result_dir / f"{args.output_stem}.json"
    out_md = result_dir / f"{args.output_stem.upper()}.md"
    if not args.overwrite and (out_json.exists() or out_md.exists()):
        raise SystemExit(f"Output exists; pass --overwrite: {out_json} / {out_md}")

    summaries = []
    for raw in args.panels:
        path = Path(raw)
        summaries.append(summarize_panel(panel_name(path), read_panel(path)))

    pairwise = []
    for i, a in enumerate(summaries):
        for b in summaries[i + 1 :]:
            pairwise.append(compare_panels(a, b))

    out = {"panels": summaries, "pairwise": pairwise}
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    write_markdown(out_md, out)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
