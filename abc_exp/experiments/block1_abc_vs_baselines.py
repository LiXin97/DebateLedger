"""Block 1: ABC vs Baselines Comparison Experiment.

Compares ABC weighted debate against multiple baselines:
  a. Standard debate (data from Block 0.2, or re-run)
  b. ABC weighted debate (using Block 0.1 revisability scores)
  c. Confidence-weighted baseline
  d. Self-consistency (5 samples)
  e. Best-of-5

For each of 200 MMLU-Pro questions, runs all methods and records
per-question results. Computes aggregate accuracy and McNemar's
test of ABC vs standard debate.

Usage:
    python -m abc_exp.experiments.block1_abc_vs_baselines [--n_questions 200] [--budget 50.0]

Resume support: results are saved as JSONL, completed question IDs
are skipped on restart.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from abc_exp.src.api.client import GeminiClient
from abc_exp.src.data.loader import load_benchmark
from abc_exp.src.debate.standard import run_debate
from abc_exp.src.debate.abc_protocol import run_abc_debate
from abc_exp.src.debate.baselines import (
    majority_vote_baseline,
    confidence_weighted_baseline,
    self_consistency_baseline,
    best_of_n_baseline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    model: str = "models/gemini-2.5-flash"
    n_questions: int = 200
    n_agents: int = 3
    agent_temperatures: tuple[float, ...] = (0.5, 0.7, 1.0)
    n_debate_rounds: int = 3
    abc_epsilon: float = 0.05
    self_consistency_samples: int = 5
    best_of_n: int = 5
    budget_cap: float = 50.0  # USD for this experiment
    max_concurrent: int = 20
    seed: int = 42
    results_dir: str = "abc_exp/results"
    log_dir: str = "abc_exp/results/logs"
    block01_results: str = "abc_exp/results/block0_alpha.jsonl"
    skip_standard_debate: bool = False  # set True if Block 0.2 data exists


def get_results_path(config: Config) -> Path:
    return Path(config.results_dir) / "block1_abc_vs_baselines.jsonl"


def get_summary_path(config: Config) -> Path:
    return Path(config.results_dir) / "block1_summary.json"


# ---------------------------------------------------------------------------
# Load pre-computed revisability scores from Block 0.1
# ---------------------------------------------------------------------------

def load_revisability_scores(path: str | Path) -> dict[str, dict[str, float]]:
    """Load per-question, per-agent revisability scores from Block 0.1 JSONL.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of question_id -> {agent_id -> flip_rate}.
        Agent IDs are mapped from "agent_0" -> "1", "agent_1" -> "2", etc.
        to match the debate agent naming convention.
    """
    path = Path(path)
    scores: dict[str, dict[str, float]] = {}

    if not path.exists():
        logger.warning("Block 0.1 results not found at %s", path)
        return scores

    with open(path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            qid = data["question_id"]
            alpha_estimates = data.get("alpha_estimates", {})

            agent_scores: dict[str, float] = {}
            for agent_key, alpha_data in alpha_estimates.items():
                # Map "agent_0" -> "1", "agent_1" -> "2", "agent_2" -> "3"
                if agent_key.startswith("agent_"):
                    debate_id = str(int(agent_key.split("_")[1]) + 1)
                else:
                    debate_id = agent_key

                # Use mean_flip_rate if available (from MLE), otherwise use alpha
                # For flip-rate estimates, alpha IS the flip rate.
                flip_rate = alpha_data.get("mean_flip_rate", alpha_data.get("alpha", 0.5))
                agent_scores[debate_id] = flip_rate

            scores[qid] = agent_scores

    logger.info(
        "Loaded revisability scores for %d questions from %s",
        len(scores), path,
    )
    return scores


def get_default_revisabilities(n_agents: int = 3) -> dict[str, float]:
    """Return default revisability scores when Block 0.1 data is unavailable.

    Uses 0.5 (uninformative) for all agents, so weighting is uniform.
    """
    return {str(i + 1): 0.5 for i in range(n_agents)}


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_ids(results_path: Path) -> set[str]:
    """Load question IDs already completed (for resume support)."""
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
# Per-question processing
# ---------------------------------------------------------------------------

async def process_question(
    client: GeminiClient,
    question,
    revisabilities: dict[str, float],
    config: Config,
) -> dict:
    """Run all methods on a single question and return a combined result record."""
    results: dict = {
        "question_id": question.id,
        "benchmark": question.benchmark,
        "correct_label": question.correct_label,
        "category": question.metadata.get("category", "unknown"),
    }

    total_cost = 0.0

    # --- Method A: Standard debate ---
    if not config.skip_standard_debate:
        try:
            std_result = await run_debate(
                client=client,
                question=question,
                n_agents=config.n_agents,
                n_rounds=config.n_debate_rounds,
                temperatures=config.agent_temperatures,
            )
            results["standard_debate"] = {
                "final_answer": std_result.final_answer,
                "final_correct": std_result.final_correct,
                "initial_majority": std_result.initial_majority,
                "initial_correct": std_result.initial_correct,
                "collapsed": std_result.collapsed,
                "corrected": std_result.corrected,
                "cost": std_result.total_cost,
            }
            total_cost += std_result.total_cost

            # Extract initial answers for majority vote baseline
            initial_answers = std_result.initial_answers
        except Exception as e:
            logger.error("Standard debate failed for %s: %s", question.id, e)
            results["standard_debate"] = {"error": str(e)}
            initial_answers = None
    else:
        initial_answers = None

    # --- Method B: ABC weighted debate ---
    try:
        abc_result = await run_abc_debate(
            client=client,
            question=question,
            agent_revisabilities=revisabilities,
            n_agents=config.n_agents,
            n_rounds=config.n_debate_rounds,
            temperatures=config.agent_temperatures,
            epsilon=config.abc_epsilon,
        )
        results["abc_debate"] = {
            "final_answer": abc_result.final_answer,
            "final_correct": abc_result.final_correct,
            "final_unweighted_answer": abc_result.final_unweighted_answer,
            "final_unweighted_correct": abc_result.final_unweighted_correct,
            "initial_weighted_majority": abc_result.initial_weighted_majority,
            "initial_weighted_correct": abc_result.initial_weighted_correct,
            "collapsed": abc_result.collapsed,
            "corrected": abc_result.corrected,
            "collapsed_unweighted": abc_result.collapsed_unweighted,
            "corrected_unweighted": abc_result.corrected_unweighted,
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
        total_cost += abc_result.total_cost

        # Use ABC initial answers for majority vote baseline if needed
        if initial_answers is None:
            initial_answers = abc_result.initial_answers
    except Exception as e:
        logger.error("ABC debate failed for %s: %s", question.id, e)
        results["abc_debate"] = {"error": str(e)}

    # --- Method 0 (free): Majority vote baseline ---
    if initial_answers is not None:
        mv_result = majority_vote_baseline(initial_answers, question)
        results["majority_vote"] = mv_result.to_dict()
    else:
        results["majority_vote"] = {"error": "no initial answers available"}

    # --- Method C: Confidence-weighted ---
    try:
        cw_result = await confidence_weighted_baseline(
            client=client,
            question=question,
            n_agents=config.n_agents,
            temperatures=config.agent_temperatures,
        )
        results["confidence_weighted"] = cw_result.to_dict()
        total_cost += cw_result.total_cost
    except Exception as e:
        logger.error("Confidence-weighted failed for %s: %s", question.id, e)
        results["confidence_weighted"] = {"error": str(e)}

    # --- Method D: Self-consistency ---
    try:
        sc_result = await self_consistency_baseline(
            client=client,
            question=question,
            n_samples=config.self_consistency_samples,
            temperature=0.7,
        )
        results["self_consistency"] = sc_result.to_dict()
        total_cost += sc_result.total_cost
    except Exception as e:
        logger.error("Self-consistency failed for %s: %s", question.id, e)
        results["self_consistency"] = {"error": str(e)}

    # --- Method E: Best-of-N ---
    try:
        bon_result = await best_of_n_baseline(
            client=client,
            question=question,
            n=config.best_of_n,
            temperature=0.0,
        )
        results["best_of_n"] = bon_result.to_dict()
        total_cost += bon_result.total_cost
    except Exception as e:
        logger.error("Best-of-N failed for %s: %s", question.id, e)
        results["best_of_n"] = {"error": str(e)}

    results["total_cost"] = total_cost
    return results


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_aggregate_stats(results_path: Path) -> dict:
    """Compute aggregate statistics from all per-question results.

    Returns accuracy per method and McNemar's test p-values.
    """
    records = []
    with open(results_path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    n = len(records)
    if n == 0:
        return {"error": "no results found"}

    # Collect per-method correctness
    methods = [
        "standard_debate", "abc_debate", "majority_vote",
        "confidence_weighted", "self_consistency", "best_of_n",
    ]

    correct: dict[str, list[bool]] = {m: [] for m in methods}
    costs: dict[str, list[float]] = {m: [] for m in methods}

    # ABC-specific tracking
    abc_collapsed = 0
    abc_corrected = 0
    std_collapsed = 0
    std_corrected = 0

    for rec in records:
        for method in methods:
            mdata = rec.get(method, {})
            if "error" in mdata:
                continue
            if method == "abc_debate":
                is_correct = mdata.get("final_correct", False)
            else:
                is_correct = mdata.get("final_correct", False)
            correct[method].append(is_correct)
            costs[method].append(mdata.get("cost", mdata.get("total_cost", 0.0)))

        # Collapse/correction tracking
        abc_data = rec.get("abc_debate", {})
        if "error" not in abc_data:
            if abc_data.get("collapsed", False):
                abc_collapsed += 1
            if abc_data.get("corrected", False):
                abc_corrected += 1

        std_data = rec.get("standard_debate", {})
        if "error" not in std_data:
            if std_data.get("collapsed", False):
                std_collapsed += 1
            if std_data.get("corrected", False):
                std_corrected += 1

    # Accuracy per method
    accuracy = {}
    for method in methods:
        vals = correct[method]
        if vals:
            accuracy[method] = {
                "accuracy": sum(vals) / len(vals),
                "n": len(vals),
                "n_correct": sum(vals),
                "mean_cost": sum(costs[method]) / len(costs[method]),
                "total_cost": sum(costs[method]),
            }

    # McNemar's test: ABC vs standard debate
    mcnemar_result = _mcnemars_test(
        correct.get("standard_debate", []),
        correct.get("abc_debate", []),
    )

    stats = {
        "n_questions": n,
        "accuracy": accuracy,
        "mcnemar_abc_vs_standard": mcnemar_result,
        "collapse_correction": {
            "standard_debate": {
                "collapsed": std_collapsed,
                "corrected": std_corrected,
            },
            "abc_debate": {
                "collapsed": abc_collapsed,
                "corrected": abc_corrected,
            },
        },
    }

    return stats


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
    # Align lengths (use minimum)
    n = min(len(correct_a), len(correct_b))
    if n == 0:
        return {"error": "insufficient data"}

    # 2x2 contingency table
    # b = A correct, B incorrect
    # c = A incorrect, B correct
    b = sum(1 for i in range(n) if correct_a[i] and not correct_b[i])
    c = sum(1 for i in range(n) if not correct_a[i] and correct_b[i])

    n_discordant = b + c

    if n_discordant == 0:
        return {
            "n": n,
            "n_discordant": 0,
            "b_a_right_b_wrong": b,
            "c_a_wrong_b_right": c,
            "chi2": 0.0,
            "p_value": 1.0,
            "note": "no discordant pairs",
        }

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi2 distribution with df=1
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(1 - chi2_dist.cdf(chi2, df=1))
    except ImportError:
        # Fallback: approximate using normal distribution
        import math
        z = math.sqrt(chi2)
        # One-sided to two-sided
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

    return {
        "n": n,
        "n_discordant": n_discordant,
        "b_a_right_b_wrong": b,
        "c_a_wrong_b_right": c,
        "chi2": chi2,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(config: Config):
    """Run the Block 1 comparison experiment."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = get_results_path(config)
    summary_path = get_summary_path(config)
    completed = load_completed_ids(results_path)

    if completed:
        logger.info("Resuming: %d questions already completed", len(completed))

    # Load revisability scores from Block 0.1
    all_revisabilities = load_revisability_scores(config.block01_results)

    # Load questions (same seed/sample as Block 0.1 for alignment)
    logger.info("Loading MMLU-Pro dataset...")
    questions = load_benchmark("mmlu_pro", n_samples=config.n_questions, seed=config.seed)
    logger.info("Loaded %d questions", len(questions))

    # Filter out completed
    remaining = [q for q in questions if q.id not in completed]
    logger.info("Remaining: %d questions to process", len(remaining))

    if not remaining:
        logger.info("All questions already completed!")
        # Still compute aggregate stats
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

    total_cost = 0.0
    n_processed = 0

    try:
        for i, question in enumerate(remaining):
            # Get revisability scores for this question
            revisabilities = all_revisabilities.get(
                question.id,
                get_default_revisabilities(config.n_agents),
            )

            logger.info(
                "[%d/%d] Processing %s (cat: %s) | Cost: $%.4f | "
                "Revisabilities: %s",
                i + 1, len(remaining), question.id,
                question.metadata.get("category", "?"),
                total_cost,
                {k: f"{v:.3f}" for k, v in revisabilities.items()},
            )

            record = await process_question(
                client, question, revisabilities, config,
            )

            # Append to JSONL
            with open(results_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_cost += record["total_cost"]
            n_processed += 1

            # Print per-question summary
            for method_name in ["standard_debate", "abc_debate",
                                "confidence_weighted", "self_consistency",
                                "best_of_n"]:
                mdata = record.get(method_name, {})
                if "error" not in mdata:
                    correct_str = "Y" if mdata.get("final_correct") else "N"
                    ans = mdata.get("final_answer", "?")
                    logger.info(
                        "  %-22s  answer=%s  correct=%s",
                        method_name, ans, correct_str,
                    )

    except Exception as e:
        logger.error("Experiment stopped: %s", e)
        raise
    finally:
        # Compute aggregate statistics
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


def _print_summary(stats: dict, total_completed: int):
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("BLOCK 1: ABC vs BASELINES SUMMARY")
    print("=" * 70)
    print(f"Questions completed: {total_completed}")
    print()

    accuracy = stats.get("accuracy", {})
    if accuracy:
        print(f"{'Method':<25} {'Accuracy':>10} {'N':>6} {'Avg Cost':>10}")
        print("-" * 55)
        for method, data in sorted(accuracy.items()):
            print(
                f"{method:<25} {data['accuracy']:>9.1%} "
                f"{data['n']:>6} "
                f"${data['mean_cost']:>8.4f}"
            )

    print()

    # Collapse/correction
    cc = stats.get("collapse_correction", {})
    for method_name, cc_data in cc.items():
        print(
            f"{method_name}: collapsed={cc_data['collapsed']}, "
            f"corrected={cc_data['corrected']}"
        )

    # McNemar's test
    mcnemar = stats.get("mcnemar_abc_vs_standard", {})
    if "error" not in mcnemar:
        print(f"\nMcNemar's test (ABC vs Standard Debate):")
        print(f"  Discordant pairs: {mcnemar.get('n_discordant', 0)}")
        print(f"  Standard right, ABC wrong: {mcnemar.get('b_a_right_b_wrong', 0)}")
        print(f"  Standard wrong, ABC right: {mcnemar.get('c_a_wrong_b_right', 0)}")
        print(f"  chi2 = {mcnemar.get('chi2', 0):.3f}, p = {mcnemar.get('p_value', 1):.4f}")
        p = mcnemar.get('p_value', 1.0)
        if p < 0.05:
            print(f"  ==> Significant difference (p < 0.05)")
        else:
            print(f"  ==> No significant difference (p >= 0.05)")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Block 1: ABC vs Baselines Comparison"
    )
    parser.add_argument("--n_questions", type=int, default=200)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--max_concurrent", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="models/gemini-2.5-flash")
    parser.add_argument("--n_rounds", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sc_samples", type=int, default=5,
                        help="Self-consistency samples")
    parser.add_argument("--bon_n", type=int, default=5,
                        help="Best-of-N count")
    parser.add_argument("--block01_results", type=str,
                        default="abc_exp/results/block0_alpha.jsonl",
                        help="Path to Block 0.1 revisability results")
    parser.add_argument("--skip_standard", action="store_true",
                        help="Skip standard debate (use if Block 0.2 data exists)")
    args = parser.parse_args()

    config = Config(
        model=args.model,
        n_questions=args.n_questions,
        n_debate_rounds=args.n_rounds,
        abc_epsilon=args.epsilon,
        self_consistency_samples=args.sc_samples,
        best_of_n=args.bon_n,
        budget_cap=args.budget,
        max_concurrent=args.max_concurrent,
        seed=args.seed,
        block01_results=args.block01_results,
        skip_standard_debate=args.skip_standard,
    )

    asyncio.run(main(config))
