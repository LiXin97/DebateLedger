"""Gemini same-prompt alpha stability panel for R5 nondeterminism checks.

This script intentionally writes to a new output stem and does not touch the
sealed headline artifacts. It repeats the same initial-answer + 8-probe alpha
measurement on fixed MMLU-Pro question IDs so repeated same-day or cross-day
runs can quantify closed-API drift in answer extraction and alpha ranks.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from abc_exp.config.prompts import INITIAL_ANSWER_PROMPT  # noqa: E402
from abc_exp.src.alpha.estimation import estimate_alpha_fliprate  # noqa: E402
from abc_exp.src.alpha.probes import run_probes  # noqa: E402
from abc_exp.src.alpha.revision import extract_answer  # noqa: E402
from abc_exp.src.api.client import GeminiClient  # noqa: E402
from abc_exp.src.data.loader import load_benchmark  # noqa: E402


DEFAULT_MODEL = "models/gemini-3.1-flash-lite-preview"


def stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode()).hexdigest()
    return int(digest[:8], 16)


def output_paths(results_dir: Path, output_stem: str) -> tuple[Path, Path, Path]:
    return (
        results_dir / f"{output_stem}.jsonl",
        results_dir / f"{output_stem}.json",
        results_dir / f"{output_stem.upper()}.md",
    )


def load_done(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((rec.get("question_id", ""), int(rec.get("repeat", -1))))
    return done


async def run_one(client: GeminiClient, question, repeat: int, seed: int) -> dict:
    prompt = INITIAL_ANSWER_PROMPT.format(question=question.format_for_prompt())
    initial = await client.generate(
        prompt,
        temperature=0.0,
        max_output_tokens=512,
        metadata={"phase": "r5_initial", "question_id": question.id, "repeat": repeat},
    )
    initial_answer = extract_answer(initial.text, question.option_labels)
    rec = {
        "question_id": question.id,
        "benchmark": question.benchmark,
        "category": question.metadata.get("category", "unknown"),
        "correct_label": question.correct_label,
        "repeat": repeat,
        "model": client.model,
        "timestamp": initial.timestamp,
        "initial_answer": initial_answer,
        "initial_correct": bool(initial_answer == question.correct_label) if initial_answer else False,
        "initial_parse_none": initial_answer is None,
        "initial_tokens": {
            "input": initial.input_tokens,
            "output": initial.output_tokens,
            "thinking": initial.thinking_tokens,
        },
        "initial_cost": initial.cost,
        "alpha": None,
        "n_revised": 0,
        "n_probes": 0,
        "post_answer_none": None,
        "probe_cost": 0.0,
        "probe_results": [],
    }
    if not initial_answer:
        return rec

    # Reset the RNG per question so every repeat uses the same wrong-option target.
    rng = np.random.default_rng(stable_seed(seed, question.id))
    probes = await run_probes(client, question, initial_answer, f"repeat_{repeat}", rng)
    est = estimate_alpha_fliprate(probes)
    rec.update(
        {
            "alpha": est.alpha,
            "n_revised": est.n_revised,
            "n_probes": est.n_probes,
            "post_answer_none": sum(p.probe_answer is None for p in probes),
            "probe_cost": sum(p.cost for p in probes),
            "probe_results": [p.to_dict() for p in probes],
        }
    )
    return rec


def summarize(records: list[dict]) -> dict:
    by_q: dict[str, list[dict]] = {}
    for rec in records:
        by_q.setdefault(rec["question_id"], []).append(rec)

    per_question = []
    for qid, rows in sorted(by_q.items()):
        alphas = [r["alpha"] for r in rows if r.get("alpha") is not None]
        answers = [r.get("initial_answer") for r in rows]
        per_question.append(
            {
                "question_id": qid,
                "n_repeats": len(rows),
                "initial_answers": answers,
                "n_unique_initial_answers": len(set(answers)),
                "alpha_values": alphas,
                "alpha_mean": mean(alphas) if alphas else math.nan,
                "alpha_range": (max(alphas) - min(alphas)) if len(alphas) >= 2 else 0.0,
                "post_answer_none": sum(r.get("post_answer_none") or 0 for r in rows),
            }
        )

    alpha_ranges = [r["alpha_range"] for r in per_question if math.isfinite(r["alpha_range"])]
    answer_unstable = [r for r in per_question if r["n_unique_initial_answers"] > 1]
    total_probe_trials = sum((r.get("n_probes") or 0) for r in records)
    total_post_none = sum((r.get("post_answer_none") or 0) for r in records)
    return {
        "n_records": len(records),
        "n_questions": len(per_question),
        "questions_with_initial_answer_instability": len(answer_unstable),
        "mean_alpha_range_within_question": mean(alpha_ranges) if alpha_ranges else math.nan,
        "max_alpha_range_within_question": max(alpha_ranges) if alpha_ranges else math.nan,
        "probe_post_answer_none_rate": total_post_none / total_probe_trials if total_probe_trials else math.nan,
        "total_cost": sum(float(r.get("initial_cost") or 0) + float(r.get("probe_cost") or 0) for r in records),
        "per_question": per_question,
    }


def write_markdown(path: Path, args, summary: dict) -> None:
    lines = [
        "# Gemini R5 Nondeterminism Alpha Panel",
        "",
        "This is a same-prompt alpha stability panel for closed-API drift auditing. It is a pilot unless repeated across multiple calendar days with the same question IDs and seed.",
        "",
        f"Model: `{args.model}`; questions: {args.n_questions}; repeats: {args.repeats}; seed: {args.seed}.",
        "",
        "## Summary",
        "",
        f"- Records: {summary['n_records']} over {summary['n_questions']} questions.",
        f"- Questions with initial-answer instability: {summary['questions_with_initial_answer_instability']}.",
        f"- Mean within-question alpha range: {summary['mean_alpha_range_within_question']:.4f}.",
        f"- Max within-question alpha range: {summary['max_alpha_range_within_question']:.4f}.",
        f"- Probe post-answer None rate: {summary['probe_post_answer_none_rate']:.4%}.",
        f"- Total estimated cost: ${summary['total_cost']:.4f}.",
        "",
        "| Question | repeats | unique init answers | alpha values | alpha range | post-answer None |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_question"]:
        alphas = ", ".join(f"{v:.3f}" for v in row["alpha_values"])
        lines.append(
            f"| `{row['question_id']}` | {row['n_repeats']} | {row['n_unique_initial_answers']} | "
            f"{alphas} | {row['alpha_range']:.3f} | {row['post_answer_none']} |"
        )
    path.write_text("\n".join(lines) + "\n")


async def main(args) -> int:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, summary_path, md_path = output_paths(results_dir, args.output_stem)
    if args.overwrite:
        for path in (jsonl_path, summary_path, md_path):
            if path.exists():
                path.unlink()

    qids = json.loads(Path(args.questions).read_text())[: args.n_questions]
    all_questions = load_benchmark("mmlu_pro", n_samples=12000, seed=0)
    by_id = {q.id: q for q in all_questions}
    questions = [by_id[qid] for qid in qids if qid in by_id]
    missing = [qid for qid in qids if qid not in by_id]
    if missing:
        raise SystemExit(f"Missing MMLU-Pro question IDs: {missing[:5]}")

    done = load_done(jsonl_path) if args.resume else set()
    jobs = [(q, rep) for q in questions for rep in range(args.repeats) if (q.id, rep) not in done]
    print(f"[r5] {len(questions)} questions, {len(jobs)} remaining question-repeat jobs")
    if not jobs:
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        summary = summarize(records)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        write_markdown(md_path, args, summary)
        return 0

    client = GeminiClient(
        model=args.model,
        budget_cap=args.budget,
        max_concurrent=args.max_concurrent,
        log_dir=results_dir / "logs",
    )
    sem = asyncio.Semaphore(max(1, args.job_concurrency))
    lock = asyncio.Lock()
    started = time.time()
    n_done = 0

    async def worker(question, repeat):
        nonlocal n_done
        async with sem:
            rec = await run_one(client, question, repeat, args.seed)
            async with lock:
                with jsonl_path.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1
                elapsed = time.time() - started
                print(
                    f"[r5] {n_done}/{len(jobs)} jobs done; "
                    f"cost=${client.cost_tracker.total_cost:.4f}; "
                    f"rate={60 * n_done / max(elapsed, 1e-9):.1f} jobs/min",
                    flush=True,
                )

    try:
        await asyncio.gather(*(worker(q, rep) for q, rep in jobs))
    finally:
        print(client.cost_tracker.summary())
        await client.aclose()

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    summary = summarize(records)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(md_path, args, summary)
    print(f"[r5] wrote {summary_path}")
    print(f"[r5] wrote {md_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--questions", default=str(PROJECT_ROOT / "abc_exp/results/high_fr_question_ids.json"))
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--job-concurrency", type=int, default=2)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "abc_exp/results"))
    parser.add_argument("--output-stem", default="r5_gemini_alpha_panel_day1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
