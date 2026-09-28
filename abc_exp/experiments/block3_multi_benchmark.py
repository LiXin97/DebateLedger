"""Block 3: Multi-Benchmark Generalization Experiment.

Tests ABC weighted debate across all 5 benchmarks to evaluate
whether revisability-based weighting generalizes beyond MMLU-Pro.

Benchmarks: mmlu_pro, gpqa, arc (ARC-Challenge), truthfulqa, gsm8k

For each benchmark:
  1. Sample 100 questions (stratified)
  2. For each question:
     a. Get initial answers from 3 agents
     b. Run alpha estimation (8 probes per agent) via flip-rate (ICC=0.76)
     c. Run standard debate (3 rounds, 3 agents)
     d. Run ABC weighted debate (using measured revisability)
  3. Record per-question accuracy for both methods
  4. Compute McNemar's test (ABC vs standard) per benchmark
  5. Print per-benchmark accuracy comparison table

Usage:
    python -m abc_exp.experiments.block3_multi_benchmark \\
        [--benchmarks mmlu_pro gpqa arc truthfulqa gsm8k] \\
        [--n_questions 100] [--budget 80.0]

Resume support: results are saved as JSONL with benchmark name in each
record; completed question IDs are skipped on restart.
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from abc_exp.src.api.client import GeminiClient
from abc_exp.src.data.loader import load_benchmark
from abc_exp.src.debate.standard import run_debate
from abc_exp.src.debate.abc_protocol import run_abc_debate
from abc_exp.src.alpha.probes import run_probes
from abc_exp.src.alpha.estimation import estimate_alpha_fliprate
from abc_exp.src.alpha.revision import extract_answer
from abc_exp.config.prompts import INITIAL_ANSWER_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


ALL_BENCHMARKS = ["mmlu_pro", "gpqa", "arc", "truthfulqa", "gsm8k"]


@dataclass
class Config:
    model: str = "models/gemini-2.5-flash"
    benchmarks: list[str] = field(default_factory=lambda: list(ALL_BENCHMARKS))
    n_questions: int = 100
    n_agents: int = 3
    agent_temperatures: tuple[float, ...] = (0.5, 0.7, 1.0)
    n_debate_rounds: int = 3
    abc_epsilon: float = 0.05
    budget_cap: float = 80.0  # USD total across all benchmarks
    max_concurrent: int = 20
    seed: int = 42
    results_dir: str = "abc_exp/results"
    log_dir: str = "abc_exp/results/logs"


def get_results_path(config: Config) -> Path:
    return Path(config.results_dir) / "block3_multi_benchmark.jsonl"


def get_summary_path(config: Config) -> Path:
    return Path(config.results_dir) / "block3_summary.json"


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_ids(results_path: Path) -> set[str]:
    """Load question IDs already completed (for resume support).

    Returns a set of question_id strings. Each question_id is globally
    unique across benchmarks (e.g. "mmlu_pro_42", "gpqa_7").
    """
    completed = set()
    if results_path.exists():
        with open(results_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    completed.add(data["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


# ---------------------------------------------------------------------------
# Alpha estimation: initial answers + probes for a single question
# ---------------------------------------------------------------------------

async def estimate_revisabilities(
    client: GeminiClient,
    question,
    config: Config,
    rng: np.random.Generator,
) -> tuple[dict[str, float], list[dict], float]:
    """Get initial answers and run flip-rate alpha estimation for all agents.

    Returns
    -------
    revisabilities : dict[str, float]
        Mapping of agent_id (debate convention: "1", "2", "3") to flip rate.
    agent_infos : list[dict]
        Per-agent info dicts with initial answer, alpha, etc.
    cost : float
        Total API cost for initial answers + probes.
    """
    question_text = question.format_for_prompt()
    prompt = INITIAL_ANSWER_PROMPT.format(question=question_text)
    total_cost = 0.0

    # Phase 1: Get initial answers from all agents concurrently
    initial_results = await asyncio.gather(*[
        client.generate(
            contents=prompt,
            temperature=config.agent_temperatures[i],
            max_output_tokens=1024,
            metadata={
                "question_id": question.id,
                "agent_id": f"agent_{i}",
                "phase": "alpha_initial",
            },
        )
        for i in range(config.n_agents)
    ])

    agents = []
    for i, result in enumerate(initial_results):
        answer = extract_answer(result.text, question.option_labels)
        agents.append({
            "agent_id": f"agent_{i}",
            "debate_id": str(i + 1),
            "initial_answer": answer,
            "initial_correct": answer == question.correct_label if answer else False,
            "cost": result.cost,
            "temperature": config.agent_temperatures[i],
        })
        total_cost += result.cost

    # Phase 2: Run 8 probes per agent concurrently (agents with valid answers)
    probe_tasks = []
    for i, agent in enumerate(agents):
        if agent["initial_answer"] is None:
            logger.warning(
                "Question %s agent_%d: no initial answer, skipping probes",
                question.id, i,
            )
            continue
        agent_rng = np.random.default_rng(rng.integers(0, 2**32) + i)
        probe_tasks.append((
            i,
            run_probes(
                client, question,
                agent["initial_answer"], f"agent_{i}", agent_rng,
            ),
        ))

    # Await all probe tasks and estimate flip rates
    revisabilities: dict[str, float] = {}
    for i, task in probe_tasks:
        probe_results = await task
        total_cost += sum(p.cost for p in probe_results)

        estimate = estimate_alpha_fliprate(probe_results)
        debate_id = str(i + 1)
        revisabilities[debate_id] = estimate.alpha

        agents[i]["alpha"] = estimate.alpha
        agents[i]["n_revised"] = estimate.n_revised
        agents[i]["n_probes"] = estimate.n_probes

    # Fill default for agents without probes (failed initial answer)
    for i in range(config.n_agents):
        debate_id = str(i + 1)
        if debate_id not in revisabilities:
            revisabilities[debate_id] = 0.5  # uninformative default

    return revisabilities, agents, total_cost


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------

async def process_question(
    client: GeminiClient,
    question,
    config: Config,
    rng: np.random.Generator,
) -> dict:
    """Run alpha estimation + both debate protocols on a single question.

    Steps (alpha estimation runs first, then both debates concurrently):
    1. Estimate revisabilities via flip-rate probes
    2. Run standard debate and ABC debate concurrently
    3. Return combined result record
    """
    results: dict = {
        "question_id": question.id,
        "benchmark": question.benchmark,
        "correct_label": question.correct_label,
        "category": question.metadata.get("category", "unknown"),
    }

    # Step 1: Alpha estimation (must finish before ABC debate)
    revisabilities, agent_infos, alpha_cost = await estimate_revisabilities(
        client, question, config, rng,
    )
    results["alpha_estimation"] = {
        "agents": agent_infos,
        "revisabilities": revisabilities,
        "cost": alpha_cost,
    }

    total_cost = alpha_cost

    # Step 2: Run standard debate and ABC debate concurrently
    std_task = _run_standard_debate(client, question, config)
    abc_task = _run_abc_debate(client, question, revisabilities, config)

    std_result_data, abc_result_data = await asyncio.gather(
        std_task, abc_task, return_exceptions=True,
    )

    # Process standard debate result
    if isinstance(std_result_data, Exception):
        logger.error("Standard debate failed for %s: %s", question.id, std_result_data)
        results["standard_debate"] = {"error": str(std_result_data)}
    else:
        results["standard_debate"] = std_result_data
        total_cost += std_result_data.get("cost", 0.0)

    # Process ABC debate result
    if isinstance(abc_result_data, Exception):
        logger.error("ABC debate failed for %s: %s", question.id, abc_result_data)
        results["abc_debate"] = {"error": str(abc_result_data)}
    else:
        results["abc_debate"] = abc_result_data
        total_cost += abc_result_data.get("cost", 0.0)

    results["total_cost"] = total_cost
    return results


async def _run_standard_debate(
    client: GeminiClient,
    question,
    config: Config,
) -> dict:
    """Run standard debate and return a serializable result dict."""
    std_result = await run_debate(
        client=client,
        question=question,
        n_agents=config.n_agents,
        n_rounds=config.n_debate_rounds,
        temperatures=config.agent_temperatures,
    )
    return {
        "final_answer": std_result.final_answer,
        "final_correct": std_result.final_correct,
        "initial_majority": std_result.initial_majority,
        "initial_correct": std_result.initial_correct,
        "collapsed": std_result.collapsed,
        "corrected": std_result.corrected,
        "cost": std_result.total_cost,
    }


async def _run_abc_debate(
    client: GeminiClient,
    question,
    revisabilities: dict[str, float],
    config: Config,
) -> dict:
    """Run ABC weighted debate and return a serializable result dict."""
    abc_result = await run_abc_debate(
        client=client,
        question=question,
        agent_revisabilities=revisabilities,
        n_agents=config.n_agents,
        n_rounds=config.n_debate_rounds,
        temperatures=config.agent_temperatures,
        epsilon=config.abc_epsilon,
    )
    return {
        "final_answer": abc_result.final_answer,
        "final_correct": abc_result.final_correct,
        "final_unweighted_answer": abc_result.final_unweighted_answer,
        "final_unweighted_correct": abc_result.final_unweighted_correct,
        "initial_weighted_majority": abc_result.initial_weighted_majority,
        "initial_weighted_correct": abc_result.initial_weighted_correct,
        "collapsed": abc_result.collapsed,
        "corrected": abc_result.corrected,
        "agent_revisabilities": abc_result.agent_revisabilities,
        "cost": abc_result.total_cost,
        "round_summaries": [
            {
                "round_num": rs.round_num,
                "weighted_majority": rs.weighted_majority,
                "unweighted_majority": rs.unweighted_majority,
                "consensus_strength": rs.consensus_strength,
            }
            for rs in abc_result.round_summaries
        ],
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_aggregate_stats(results_path: Path) -> dict:
    """Compute per-benchmark aggregate statistics from all results.

    Returns accuracy per method per benchmark, McNemar's test per benchmark,
    and overall (pooled) statistics.
    """
    records = []
    with open(results_path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if not records:
        return {"error": "no results found"}

    # Group records by benchmark
    by_benchmark: dict[str, list[dict]] = {}
    for rec in records:
        bm = rec.get("benchmark", "unknown")
        by_benchmark.setdefault(bm, []).append(rec)

    per_benchmark = {}
    all_std_correct: list[bool] = []
    all_abc_correct: list[bool] = []

    for bm, bm_records in sorted(by_benchmark.items()):
        std_correct = []
        abc_correct = []
        std_costs = []
        abc_costs = []
        std_collapsed = 0
        std_corrected = 0
        abc_collapsed = 0
        abc_corrected = 0

        for rec in bm_records:
            std_data = rec.get("standard_debate", {})
            abc_data = rec.get("abc_debate", {})

            if "error" not in std_data:
                std_correct.append(std_data.get("final_correct", False))
                std_costs.append(std_data.get("cost", 0.0))
                if std_data.get("collapsed", False):
                    std_collapsed += 1
                if std_data.get("corrected", False):
                    std_corrected += 1

            if "error" not in abc_data:
                abc_correct.append(abc_data.get("final_correct", False))
                abc_costs.append(abc_data.get("cost", 0.0))
                if abc_data.get("collapsed", False):
                    abc_collapsed += 1
                if abc_data.get("corrected", False):
                    abc_corrected += 1

        # McNemar's test for this benchmark
        mcnemar = _mcnemars_test(std_correct, abc_correct)

        per_benchmark[bm] = {
            "n_questions": len(bm_records),
            "standard_debate": {
                "accuracy": sum(std_correct) / len(std_correct) if std_correct else 0.0,
                "n": len(std_correct),
                "n_correct": sum(std_correct),
                "mean_cost": sum(std_costs) / len(std_costs) if std_costs else 0.0,
                "total_cost": sum(std_costs),
                "collapsed": std_collapsed,
                "corrected": std_corrected,
            },
            "abc_debate": {
                "accuracy": sum(abc_correct) / len(abc_correct) if abc_correct else 0.0,
                "n": len(abc_correct),
                "n_correct": sum(abc_correct),
                "mean_cost": sum(abc_costs) / len(abc_costs) if abc_costs else 0.0,
                "total_cost": sum(abc_costs),
                "collapsed": abc_collapsed,
                "corrected": abc_corrected,
            },
            "mcnemar_abc_vs_standard": mcnemar,
        }

        all_std_correct.extend(std_correct)
        all_abc_correct.extend(abc_correct)

    # Overall pooled McNemar's test
    overall_mcnemar = _mcnemars_test(all_std_correct, all_abc_correct)

    return {
        "n_total_questions": len(records),
        "n_benchmarks": len(by_benchmark),
        "per_benchmark": per_benchmark,
        "overall": {
            "standard_debate_accuracy": (
                sum(all_std_correct) / len(all_std_correct) if all_std_correct else 0.0
            ),
            "abc_debate_accuracy": (
                sum(all_abc_correct) / len(all_abc_correct) if all_abc_correct else 0.0
            ),
            "n_standard": len(all_std_correct),
            "n_abc": len(all_abc_correct),
            "mcnemar_abc_vs_standard": overall_mcnemar,
        },
    }


def _mcnemars_test(
    correct_a: list[bool],
    correct_b: list[bool],
) -> dict:
    """McNemar's test for paired binary outcomes.

    Tests whether the two methods have significantly different error rates.

    Returns
    -------
    dict with keys: n_discordant, b (A right B wrong), c (A wrong B right),
    chi2, p_value
    """
    n = min(len(correct_a), len(correct_b))
    if n == 0:
        return {"error": "insufficient data"}

    b = sum(1 for i in range(n) if correct_a[i] and not correct_b[i])
    c = sum(1 for i in range(n) if not correct_a[i] and correct_b[i])

    n_discordant = b + c

    if n_discordant == 0:
        return {
            "n": n,
            "n_discordant": 0,
            "b_std_right_abc_wrong": b,
            "c_std_wrong_abc_right": c,
            "chi2": 0.0,
            "p_value": 1.0,
            "note": "no discordant pairs",
        }

    # McNemar's chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi2 distribution with df=1
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(1 - chi2_dist.cdf(chi2, df=1))
    except ImportError:
        # Fallback: approximate using normal distribution
        z = math.sqrt(chi2)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

    return {
        "n": n,
        "n_discordant": n_discordant,
        "b_std_right_abc_wrong": b,
        "c_std_wrong_abc_right": c,
        "chi2": chi2,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary(stats: dict, total_completed: int):
    """Print a formatted per-benchmark accuracy comparison table."""
    print("\n" + "=" * 80)
    print("BLOCK 3: MULTI-BENCHMARK GENERALIZATION RESULTS")
    print("=" * 80)
    print(f"Total questions completed: {total_completed}")
    print()

    per_bm = stats.get("per_benchmark", {})
    if not per_bm:
        print("No per-benchmark results available.")
        return

    # Header
    print(
        f"{'Benchmark':<15} {'N':>5} "
        f"{'Std Acc':>9} {'ABC Acc':>9} {'Delta':>8} "
        f"{'Std Col':>8} {'ABC Col':>8} "
        f"{'Std Cor':>8} {'ABC Cor':>8} "
        f"{'p-value':>9}"
    )
    print("-" * 100)

    for bm, bm_data in sorted(per_bm.items()):
        std = bm_data["standard_debate"]
        abc = bm_data["abc_debate"]
        mcn = bm_data.get("mcnemar_abc_vs_standard", {})

        std_acc = std["accuracy"]
        abc_acc = abc["accuracy"]
        delta = abc_acc - std_acc

        p_val = mcn.get("p_value", float("nan"))
        p_str = f"{p_val:.4f}" if not math.isnan(p_val) else "N/A"
        sig = " *" if p_val < 0.05 else ""

        print(
            f"{bm:<15} {bm_data['n_questions']:>5} "
            f"{std_acc:>8.1%} {abc_acc:>8.1%} {delta:>+7.1%} "
            f"{std['collapsed']:>8} {abc['collapsed']:>8} "
            f"{std['corrected']:>8} {abc['corrected']:>8} "
            f"{p_str:>9}{sig}"
        )

    # Overall
    print("-" * 100)
    overall = stats.get("overall", {})
    overall_std = overall.get("standard_debate_accuracy", 0.0)
    overall_abc = overall.get("abc_debate_accuracy", 0.0)
    overall_delta = overall_abc - overall_std
    overall_mcn = overall.get("mcnemar_abc_vs_standard", {})
    overall_p = overall_mcn.get("p_value", float("nan"))
    overall_p_str = f"{overall_p:.4f}" if not math.isnan(overall_p) else "N/A"
    overall_sig = " *" if overall_p < 0.05 else ""

    print(
        f"{'OVERALL':<15} {stats.get('n_total_questions', 0):>5} "
        f"{overall_std:>8.1%} {overall_abc:>8.1%} {overall_delta:>+7.1%} "
        f"{'':>8} {'':>8} "
        f"{'':>8} {'':>8} "
        f"{overall_p_str:>9}{overall_sig}"
    )

    print()
    print("Col = collapse (correct -> wrong), Cor = correction (wrong -> correct)")
    print("* = significant at p < 0.05 (McNemar's test with continuity correction)")

    # Per-benchmark McNemar detail
    print()
    print("McNemar's Test Detail (ABC vs Standard Debate):")
    for bm, bm_data in sorted(per_bm.items()):
        mcn = bm_data.get("mcnemar_abc_vs_standard", {})
        if "error" in mcn:
            print(f"  {bm}: {mcn['error']}")
            continue
        print(
            f"  {bm}: discordant={mcn.get('n_discordant', 0)}, "
            f"std_right_abc_wrong={mcn.get('b_std_right_abc_wrong', 0)}, "
            f"std_wrong_abc_right={mcn.get('c_std_wrong_abc_right', 0)}, "
            f"chi2={mcn.get('chi2', 0):.3f}, p={mcn.get('p_value', 1):.4f}"
        )

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(config: Config):
    """Run the Block 3 multi-benchmark experiment."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = get_results_path(config)
    summary_path = get_summary_path(config)
    completed = load_completed_ids(results_path)

    if completed:
        logger.info("Resuming: %d questions already completed", len(completed))

    # Validate requested benchmarks
    for bm in config.benchmarks:
        if bm not in ALL_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark '{bm}'. Available: {ALL_BENCHMARKS}"
            )

    # Load all requested benchmarks
    all_questions = []
    for bm in config.benchmarks:
        logger.info("Loading benchmark: %s ...", bm)
        questions = load_benchmark(bm, n_samples=config.n_questions, seed=config.seed)
        logger.info("  Loaded %d questions from %s", len(questions), bm)
        all_questions.extend(questions)

    logger.info(
        "Total: %d questions across %d benchmarks",
        len(all_questions), len(config.benchmarks),
    )

    # Filter out completed
    remaining = [q for q in all_questions if q.id not in completed]
    logger.info("Remaining: %d questions to process", len(remaining))

    if not remaining:
        logger.info("All questions already completed!")
        stats = compute_aggregate_stats(results_path)
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=2)
        _print_summary(stats, len(completed))
        return

    # Initialize client
    client = GeminiClient(
        model=config.model,
        budget_cap=config.budget_cap,
        max_concurrent=config.max_concurrent,
        log_dir=config.log_dir,
    )

    rng = np.random.default_rng(config.seed)
    total_cost = 0.0
    n_processed = 0

    try:
        for i, question in enumerate(remaining):
            # Per-question RNG for reproducibility
            q_rng = np.random.default_rng(rng.integers(0, 2**32))

            logger.info(
                "[%d/%d] Processing %s (%s, cat: %s) | Cost: $%.4f",
                i + 1, len(remaining), question.id,
                question.benchmark,
                question.metadata.get("category", "?"),
                total_cost,
            )

            record = await process_question(client, question, config, q_rng)

            # Append to JSONL
            with open(results_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_cost += record["total_cost"]
            n_processed += 1

            # Log per-question outcome
            rev = record.get("alpha_estimation", {}).get("revisabilities", {})
            std_data = record.get("standard_debate", {})
            abc_data = record.get("abc_debate", {})

            std_correct_str = (
                "Y" if std_data.get("final_correct") else "N"
            ) if "error" not in std_data else "ERR"
            abc_correct_str = (
                "Y" if abc_data.get("final_correct") else "N"
            ) if "error" not in abc_data else "ERR"

            logger.info(
                "  Revisabilities: %s | Standard: %s (%s) | ABC: %s (%s)",
                {k: f"{v:.3f}" for k, v in rev.items()},
                std_data.get("final_answer", "?"), std_correct_str,
                abc_data.get("final_answer", "?"), abc_correct_str,
            )

    except Exception as e:
        logger.error("Experiment stopped: %s", e)
        raise
    finally:
        # Compute and save aggregate statistics
        stats = compute_aggregate_stats(results_path)
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=2)

        total_completed = len(completed) + n_processed
        _print_summary(stats, total_completed)

        print(f"\nThis run cost: ${total_cost:.4f}")
        print(client.cost_tracker.summary())
        print(f"\nResults: {results_path}")
        print(f"Summary: {summary_path}")

        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Block 3: Multi-Benchmark Generalization Experiment"
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=ALL_BENCHMARKS,
        choices=ALL_BENCHMARKS,
        help="Benchmarks to evaluate (default: all 5)",
    )
    parser.add_argument("--n_questions", type=int, default=100,
                        help="Questions per benchmark (stratified sample)")
    parser.add_argument("--budget", type=float, default=80.0,
                        help="Total budget cap in USD")
    parser.add_argument("--max_concurrent", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="models/gemini-2.5-flash")
    parser.add_argument("--n_rounds", type=int, default=3,
                        help="Number of debate rounds")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="ABC weighting epsilon")
    args = parser.parse_args()

    config = Config(
        model=args.model,
        benchmarks=args.benchmarks,
        n_questions=args.n_questions,
        n_debate_rounds=args.n_rounds,
        abc_epsilon=args.epsilon,
        budget_cap=args.budget,
        max_concurrent=args.max_concurrent,
        seed=args.seed,
    )

    asyncio.run(main(config))
