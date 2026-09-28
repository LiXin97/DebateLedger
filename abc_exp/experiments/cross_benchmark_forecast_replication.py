"""Small cross-benchmark forecast replication for the compliance fingerprint.

This harness is deliberately separate from block3_multi_benchmark.py so a
deadline sanity run does not append to the canonical block3 JSONL. It runs a
minimal subset of the original protocol:

  1. sample questions from non-MMLU benchmarks,
  2. estimate per-question probe flip rate from three agents,
  3. run a standard multi-agent debate,
  4. report whether higher probe alpha is associated with collapse/correction.

The output is intended as a scoped replication/sanity artifact, not as a new
headline unless the run is expanded and pre-registered.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from abc_exp.src.api.client import GeminiClient
from abc_exp.src.alpha.estimation import estimate_alpha_fliprate
from abc_exp.src.alpha.probes import run_probes
from abc_exp.src.alpha.revision import extract_answer
from abc_exp.src.data.loader import load_benchmark
from abc_exp.src.debate.standard import run_debate
from abc_exp.config.prompts import INITIAL_ANSWER_PROMPT


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    model: str = "models/gemini-2.5-flash"
    benchmarks: list[str] = field(default_factory=lambda: ["gpqa", "truthfulqa"])
    n_questions: int = 5
    seed: int = 45
    n_agents: int = 3
    n_rounds: int = 3
    temperatures: tuple[float, ...] = (0.5, 0.7, 1.0)
    budget_cap: float = 1.0
    max_concurrent: int = 5
    question_concurrency: int = 1
    results_dir: str = "abc_exp/results"
    output_stem: str = "cross_benchmark_forecast_replication"
    overwrite: bool = False
    question_ids_file: str | None = None


def output_paths(config: Config) -> tuple[Path, Path, Path]:
    root = Path(config.results_dir)
    return (
        root / f"{config.output_stem}.jsonl",
        root / f"{config.output_stem}.json",
        root / f"{config.output_stem.upper()}.md",
    )


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(rec.get("question_id", ""))
    return done


def load_questions_for_config(config: Config):
    """Load benchmark questions, optionally restricted to an explicit ID list."""
    wanted_ids: list[str] | None = None
    if config.question_ids_file:
        wanted_ids = json.loads(Path(config.question_ids_file).read_text())
        if not isinstance(wanted_ids, list):
            raise ValueError(f"question_ids_file must contain a JSON list: {config.question_ids_file}")

    questions = []
    for bm in config.benchmarks:
        if wanted_ids is None:
            questions.extend(load_benchmark(bm, n_samples=config.n_questions, seed=config.seed))
            continue

        by_id = {q.id: q for q in load_benchmark(bm, n_samples=None, seed=config.seed)}
        selected = [by_id[qid] for qid in wanted_ids if qid in by_id]
        if config.n_questions:
            selected = selected[: config.n_questions]
        questions.extend(selected)
    return questions


async def initial_answers_and_alpha(client: GeminiClient, question, config: Config, rng) -> tuple[list[dict], float]:
    prompt = INITIAL_ANSWER_PROMPT.format(question=question.format_for_prompt())
    temps = [config.temperatures[i % len(config.temperatures)] for i in range(config.n_agents)]
    initial = await asyncio.gather(*[
        client.generate(
            prompt,
            temperature=temps[i],
            max_output_tokens=1024,
            metadata={"phase": "crossbench_alpha_initial", "question_id": question.id, "agent": i},
        )
        for i in range(config.n_agents)
    ])
    total_cost = sum(r.cost for r in initial)
    agents: list[dict] = []
    probe_tasks = []
    for i, result in enumerate(initial):
        answer = extract_answer(result.text, question.option_labels)
        agent = {
            "agent_idx": i,
            "initial_answer": answer,
            "initial_correct": bool(answer == question.correct_label) if answer else False,
            "initial_cost": result.cost,
            "alpha": None,
            "n_revised": 0,
            "n_probes": 0,
        }
        agents.append(agent)
        if answer:
            agent_rng = np.random.default_rng(int(rng.integers(0, 2**32)) + i)
            probe_tasks.append((i, run_probes(client, question, answer, f"agent_{i}", agent_rng)))

    for i, task in probe_tasks:
        probes = await task
        total_cost += sum(p.cost for p in probes)
        est = estimate_alpha_fliprate(probes)
        agents[i]["alpha"] = est.alpha
        agents[i]["n_revised"] = est.n_revised
        agents[i]["n_probes"] = est.n_probes

    return agents, total_cost


async def process_question(client: GeminiClient, question, config: Config, rng) -> dict:
    agents, alpha_cost = await initial_answers_and_alpha(client, question, config, rng)
    debate = await run_debate(
        client,
        question,
        n_agents=config.n_agents,
        n_rounds=config.n_rounds,
        temperatures=config.temperatures,
    )
    alpha_values = [a["alpha"] for a in agents if a["alpha"] is not None]
    mean_alpha = float(np.mean(alpha_values)) if alpha_values else math.nan
    max_alpha = float(np.max(alpha_values)) if alpha_values else math.nan
    return {
        "question_id": question.id,
        "benchmark": question.benchmark,
        "model": config.model,
        "correct_label": question.correct_label,
        "category": question.metadata.get("category", "unknown"),
        "alpha": {
            "mean": mean_alpha,
            "max": max_alpha,
            "agents": agents,
            "cost": alpha_cost,
        },
        "standard_debate": {
            "initial_majority": debate.initial_majority,
            "initial_correct": debate.initial_correct,
            "final_answer": debate.final_answer,
            "final_correct": debate.final_correct,
            "collapsed": debate.collapsed,
            "corrected": debate.corrected,
            "cost": debate.total_cost,
        },
        "total_cost": alpha_cost + debate.total_cost,
    }


def _safe_spearman(x: list[float], y: list[float]) -> dict:
    pairs = [(a, b) for a, b in zip(x, y) if not math.isnan(a)]
    if len(pairs) < 3 or len({b for _, b in pairs}) < 2 or len({a for a, _ in pairs}) < 2:
        return {"n": len(pairs), "rho": None, "p_two_sided": None, "note": "insufficient variation"}
    rho, p = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return {"n": len(pairs), "rho": float(rho), "p_two_sided": float(p)}


def summarize(records: list[dict]) -> dict:
    by_bench: dict[str, list[dict]] = {}
    for r in records:
        by_bench.setdefault(r["benchmark"], []).append(r)

    def one(rows: list[dict]) -> dict:
        n = len(rows)
        init_correct = [r["standard_debate"]["initial_correct"] for r in rows]
        final_correct = [r["standard_debate"]["final_correct"] for r in rows]
        collapsed = [r["standard_debate"]["collapsed"] for r in rows]
        corrected = [r["standard_debate"]["corrected"] for r in rows]
        mean_alpha = [r["alpha"]["mean"] for r in rows]
        init_at_risk = sum(init_correct)
        return {
            "n": n,
            "initial_accuracy": sum(init_correct) / n if n else 0.0,
            "final_accuracy": sum(final_correct) / n if n else 0.0,
            "n_init_majority_correct": init_at_risk,
            "n_collapses": sum(collapsed),
            "conditional_collapse_pct": (100 * sum(collapsed) / init_at_risk) if init_at_risk else None,
            "n_corrections": sum(corrected),
            "mean_alpha": float(np.nanmean(mean_alpha)) if mean_alpha else math.nan,
            "spearman_alpha_vs_collapse": _safe_spearman(mean_alpha, [int(v) for v in collapsed]),
            "spearman_alpha_vs_final_wrong": _safe_spearman(mean_alpha, [int(not v) for v in final_correct]),
        }

    return {
        "overall": one(records),
        "per_benchmark": {k: one(v) for k, v in sorted(by_bench.items())},
    }


def write_markdown(path: Path, config: Config, summary: dict) -> None:
    lines = [
        "# Cross-Benchmark Forecast Replication",
        "",
        "This is a small, independent Gemini sanity run. It is not a pre-registered headline result.",
        "",
        f"Model: `{config.model}`; benchmarks: {', '.join(config.benchmarks)}; requested n/benchmark: {config.n_questions}.",
        "",
        "## Summary",
        "",
        "| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [("overall", summary["overall"])] + list(summary["per_benchmark"].items())
    for name, s in rows:
        rho = s["spearman_alpha_vs_collapse"].get("rho")
        lines.append(
            f"| {name} | {s['n']} | {s['initial_accuracy']:.1%} | {s['final_accuracy']:.1%} | "
            f"{s['n_collapses']} | "
            f"{s['conditional_collapse_pct']:.1f}%" if s["conditional_collapse_pct"] is not None else "NA"
        )
        # Replace the just-appended partial row with a complete row; keep simple to avoid nested f-string clutter.
        lines[-1] = (
            f"| {name} | {s['n']} | {s['initial_accuracy']:.1%} | {s['final_accuracy']:.1%} | "
            f"{s['n_collapses']} | "
            f"{s['conditional_collapse_pct']:.1f}% | " if s["conditional_collapse_pct"] is not None else
            f"| {name} | {s['n']} | {s['initial_accuracy']:.1%} | {s['final_accuracy']:.1%} | {s['n_collapses']} | NA | "
        ) + f"{s['n_corrections']} | {s['mean_alpha']:.3f} | {rho if rho is not None else 'NA'} |"
    path.write_text("\n".join(lines) + "\n")


async def main(config: Config) -> None:
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, summary_path, md_path = output_paths(config)
    if config.overwrite:
        for p in (jsonl_path, summary_path, md_path):
            if p.exists():
                p.unlink()
    completed = load_completed(jsonl_path)

    questions = load_questions_for_config(config)
    remaining = [q for q in questions if q.id not in completed]
    logger.info("Loaded %d questions, %d remaining", len(questions), len(remaining))

    client = GeminiClient(
        model=config.model,
        budget_cap=config.budget_cap,
        max_concurrent=config.max_concurrent,
        log_dir=results_dir / "logs",
    )
    rng = np.random.default_rng(config.seed)
    all_q_seeds = {q.id: int(rng.integers(0, 2**32)) for q in questions}
    q_seeds = {q.id: all_q_seeds[q.id] for q in remaining}
    question_sem = asyncio.Semaphore(max(1, config.question_concurrency))
    write_lock = asyncio.Lock()

    async def _run_one(idx: int, q) -> dict | None:
        async with question_sem:
            logger.info("[%d/%d] %s", idx, len(remaining), q.id)
            q_rng = np.random.default_rng(q_seeds[q.id])
            try:
                rec = await process_question(client, q, config, q_rng)
            except Exception:
                logger.exception("Question failed: %s", q.id)
                return None
            async with write_lock:
                with jsonl_path.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(
                "  alpha=%.3f init=%s final=%s collapse=%s cost=$%.4f",
                rec["alpha"]["mean"],
                rec["standard_debate"]["initial_correct"],
                rec["standard_debate"]["final_correct"],
                rec["standard_debate"]["collapsed"],
                rec["total_cost"],
            )
            return rec

    try:
        tasks = [_run_one(i, q) for i, q in enumerate(remaining, start=1)]
        await asyncio.gather(*tasks)
    finally:
        await client.aclose()

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    summary = {
        "config": {
            "model": config.model,
            "benchmarks": config.benchmarks,
            "n_questions": config.n_questions,
            "seed": config.seed,
            "n_agents": config.n_agents,
            "n_rounds": config.n_rounds,
            "question_concurrency": config.question_concurrency,
            "question_ids_file": config.question_ids_file,
        },
        "summary": summarize(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    write_markdown(md_path, config, summary["summary"])
    print(json.dumps(summary["summary"], indent=2))
    print(client.cost_tracker.summary())
    print(f"Wrote {jsonl_path}, {summary_path}, {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=["gpqa", "truthfulqa"])
    parser.add_argument("--n_questions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--model", default="models/gemini-2.5-flash")
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--n_rounds", type=int, default=3)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--max_concurrent", type=int, default=5)
    parser.add_argument("--question_concurrency", type=int, default=1)
    parser.add_argument("--output_stem", default="cross_benchmark_forecast_replication")
    parser.add_argument("--question_ids_file", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = Config(
        model=args.model,
        benchmarks=args.benchmarks,
        n_questions=args.n_questions,
        seed=args.seed,
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
        budget_cap=args.budget,
        max_concurrent=args.max_concurrent,
        question_concurrency=args.question_concurrency,
        output_stem=args.output_stem,
        question_ids_file=args.question_ids_file,
        overwrite=args.overwrite,
    )
    asyncio.run(main(cfg))
