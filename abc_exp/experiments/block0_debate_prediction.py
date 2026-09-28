"""Block 0.2: Alpha Predicts Debate Collapse.

Loads alpha estimates from Block 0.1.
Runs 1000 standard 3-agent debates (5 per question for 200 MMLU-Pro questions).
Records collapse events (correct->wrong flips).
Saves debate results for analysis.

Usage:
    python -m abc_exp.experiments.block0_debate_prediction [--n_debates_per_q 5] [--budget 15.0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from abc_exp.src.api.client import GeminiClient
from abc_exp.src.data.loader import load_benchmark
from abc_exp.src.debate.standard import run_debate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 5 temperature profiles for debate diversity
TEMPERATURE_PROFILES: list[tuple[float, ...]] = [
    (0.3, 0.5, 0.7),
    (0.5, 0.7, 0.9),
    (0.7, 0.9, 1.1),
    (0.3, 0.7, 1.1),
    (0.5, 0.9, 1.0),
]


@dataclass
class Config:
    model: str = "models/gemini-2.5-flash"
    n_questions: int = 200
    n_agents: int = 3
    n_debates_per_q: int = 5
    n_rounds: int = 3  # debate rounds
    budget_cap: float = 15.0
    max_concurrent: int = 20
    seed: int = 42
    results_dir: str = "abc_exp/results"
    log_dir: str = "abc_exp/results/logs"
    alpha_file: str = "abc_exp/results/block0_alpha.jsonl"


def load_alpha_estimates(alpha_path: Path) -> dict[str, dict]:
    """Load alpha estimates from Block 0.1 results.

    Returns a mapping: question_id -> {"agent_0": {...}, "agent_1": {...}, ...}
    """
    alpha_map: dict[str, dict] = {}
    if not alpha_path.exists():
        logger.warning("Alpha file not found: %s", alpha_path)
        return alpha_map

    with open(alpha_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                qid = data["question_id"]
                alpha_map[qid] = data.get("alpha_estimates", {})
            except (json.JSONDecodeError, KeyError):
                continue

    logger.info("Loaded alpha estimates for %d questions", len(alpha_map))
    return alpha_map


def load_completed_debates(results_path: Path) -> dict[str, set[int]]:
    """Load completed debate IDs per question for resume support.

    Returns a mapping: question_id -> set of completed debate indices.
    """
    completed: dict[str, set[int]] = {}
    if not results_path.exists():
        return completed

    with open(results_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                qid = data["question_id"]
                did = data["debate_id"]
                completed.setdefault(qid, set()).add(did)
            except (json.JSONDecodeError, KeyError):
                continue

    return completed


def classify_debate_outcome(
    initial_answers: list[str | None],
    final_answers: list[str | None],
    correct_label: str,
) -> dict:
    """Classify a debate outcome: collapse, correction, or neutral.

    Returns a dict with:
    - "collapse": bool -- at least one agent flipped correct -> wrong
    - "correction": bool -- at least one agent flipped wrong -> correct
    - "initial_n_correct": int
    - "final_n_correct": int
    - "accuracy_delta": int (final - initial)
    - "any_flip": bool -- any agent changed answer
    - "agent_outcomes": list of per-agent outcome strings
    """
    n = len(initial_answers)
    collapse = False
    correction = False
    any_flip = False
    agent_outcomes = []

    initial_correct = 0
    final_correct = 0

    for i in range(n):
        init = initial_answers[i]
        final = final_answers[i]

        init_is_correct = (init is not None and init.upper() == correct_label.upper())
        final_is_correct = (final is not None and final.upper() == correct_label.upper())

        if init_is_correct:
            initial_correct += 1
        if final_is_correct:
            final_correct += 1

        flipped = (init is not None and final is not None
                   and init.upper() != final.upper())
        if flipped:
            any_flip = True

        if init_is_correct and not final_is_correct:
            collapse = True
            agent_outcomes.append("collapse")
        elif not init_is_correct and final_is_correct:
            correction = True
            agent_outcomes.append("correction")
        elif flipped:
            agent_outcomes.append("flip_wrong_to_wrong")
        else:
            agent_outcomes.append("held")

    return {
        "collapse": collapse,
        "correction": correction,
        "initial_n_correct": initial_correct,
        "final_n_correct": final_correct,
        "accuracy_delta": final_correct - initial_correct,
        "any_flip": any_flip,
        "agent_outcomes": agent_outcomes,
    }


async def run_single_debate(
    client: GeminiClient,
    question,
    debate_id: int,
    temperatures: tuple[float, ...],
    config: Config,
) -> dict:
    """Run a single 3-agent debate and return the result record.

    Calls the standard debate runner, then classifies the outcome.
    """
    debate_result = await run_debate(
        client=client,
        question=question,
        n_agents=config.n_agents,
        n_rounds=config.n_rounds,
        temperatures=temperatures,
    )

    # debate_result is expected to have:
    #   .initial_answers: list[str | None]
    #   .final_answers: list[str | None]
    #   .rounds: list of round data
    #   .total_cost: float
    initial_answers = debate_result.initial_answers
    final_answers = debate_result.final_answers

    outcome = classify_debate_outcome(
        initial_answers, final_answers, question.correct_label,
    )

    # Majority vote on final answers (excluding None)
    from collections import Counter
    valid_final = [a for a in final_answers if a is not None]
    if valid_final:
        majority_answer = Counter(valid_final).most_common(1)[0][0]
        majority_correct = majority_answer.upper() == question.correct_label.upper()
    else:
        majority_answer = None
        majority_correct = False

    record = {
        "question_id": question.id,
        "debate_id": debate_id,
        "benchmark": question.benchmark,
        "category": question.metadata.get("category", "unknown"),
        "correct_label": question.correct_label,
        "temperatures": list(temperatures),
        "n_rounds": config.n_rounds,
        "initial_answers": initial_answers,
        "final_answers": final_answers,
        "majority_answer": majority_answer,
        "majority_correct": majority_correct,
        "outcome": outcome,
        "total_cost": debate_result.total_cost,
    }

    return record


async def process_question(
    client: GeminiClient,
    question,
    alpha_estimates: dict,
    completed_debates: set[int],
    config: Config,
) -> list[dict]:
    """Run all remaining debates for a single question.

    Skips debate indices that are already completed (resume support).
    Attaches alpha estimates from Block 0.1 to each record.
    """
    records = []

    for debate_id in range(config.n_debates_per_q):
        if debate_id in completed_debates:
            continue

        temperatures = TEMPERATURE_PROFILES[debate_id % len(TEMPERATURE_PROFILES)]

        record = await run_single_debate(
            client=client,
            question=question,
            debate_id=debate_id,
            temperatures=temperatures,
            config=config,
        )

        # Attach alpha estimates (measured independently in Block 0.1)
        record["alpha_estimates"] = alpha_estimates

        records.append(record)

    return records


async def main(config: Config):
    """Run the Block 0.2 debate prediction experiment."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "block0_debates.jsonl"
    alpha_path = Path(config.alpha_file)

    # Load alpha estimates from Block 0.1
    alpha_map = load_alpha_estimates(alpha_path)
    if not alpha_map:
        logger.error(
            "No alpha estimates found at %s. Run Block 0.1 first.", alpha_path,
        )
        sys.exit(1)

    # Load resume state
    completed = load_completed_debates(results_path)
    n_already = sum(len(v) for v in completed.values())
    if n_already > 0:
        logger.info("Resuming: %d debates already completed", n_already)

    # Load questions (same seed as Block 0.1 for consistent sampling)
    logger.info("Loading MMLU-Pro dataset...")
    questions = load_benchmark("mmlu_pro", n_samples=config.n_questions, seed=config.seed)
    logger.info("Loaded %d questions", len(questions))

    # Filter to questions that have alpha estimates
    questions_with_alpha = [q for q in questions if q.id in alpha_map]
    logger.info(
        "%d/%d questions have alpha estimates",
        len(questions_with_alpha), len(questions),
    )

    # Determine which questions still need debates
    remaining = []
    for q in questions_with_alpha:
        done = completed.get(q.id, set())
        if len(done) < config.n_debates_per_q:
            remaining.append((q, done))

    logger.info("Remaining: %d questions with incomplete debates", len(remaining))

    if not remaining:
        logger.info("All debates already completed!")
        return

    # Initialize client
    client = GeminiClient(
        model=config.model,
        budget_cap=config.budget_cap,
        max_concurrent=config.max_concurrent,
        log_dir=config.log_dir,
    )

    total_cost = 0.0
    n_debates_run = 0

    try:
        for i, (question, done_debates) in enumerate(remaining):
            n_remaining_for_q = config.n_debates_per_q - len(done_debates)
            logger.info(
                "[%d/%d] Question %s: running %d/%d debates | Cost: $%.4f",
                i + 1, len(remaining), question.id,
                n_remaining_for_q, config.n_debates_per_q, total_cost,
            )

            alpha_est = alpha_map.get(question.id, {})

            records = await process_question(
                client=client,
                question=question,
                alpha_estimates=alpha_est,
                completed_debates=done_debates,
                config=config,
            )

            # Append each record to JSONL (incremental save)
            with open(results_path, "a") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            for record in records:
                total_cost += record["total_cost"]
                n_debates_run += 1

                outcome = record["outcome"]
                status = []
                if outcome["collapse"]:
                    status.append("COLLAPSE")
                if outcome["correction"]:
                    status.append("CORRECTION")
                if not status:
                    status.append("no_change" if not outcome["any_flip"] else "flip")

                logger.info(
                    "  debate_%d: %s | init=%d final=%d correct | majority=%s (%s)",
                    record["debate_id"],
                    "+".join(status),
                    outcome["initial_n_correct"],
                    outcome["final_n_correct"],
                    record["majority_answer"],
                    "correct" if record["majority_correct"] else "wrong",
                )

    except Exception as e:
        logger.error("Experiment stopped: %s", e)
        raise
    finally:
        # Print summary
        print("\n" + "=" * 60)
        print("BLOCK 0.2 SUMMARY")
        print("=" * 60)
        print(f"Debates run this session: {n_debates_run}")
        print(f"Total debates completed: {n_already + n_debates_run}")
        print(f"Target total: {config.n_questions * config.n_debates_per_q}")
        print(f"Total cost this run: ${total_cost:.4f}")
        print(f"Results saved to: {results_path}")
        print(client.cost_tracker.summary())
        print("=" * 60)

        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Block 0.2: Alpha Predicts Debate Collapse",
    )
    parser.add_argument(
        "--n_questions", type=int, default=200,
        help="Number of MMLU-Pro questions (must match Block 0.1)",
    )
    parser.add_argument(
        "--n_debates_per_q", type=int, default=5,
        help="Number of debates per question (default: 5)",
    )
    parser.add_argument(
        "--n_rounds", type=int, default=3,
        help="Number of rounds per debate (default: 3)",
    )
    parser.add_argument(
        "--budget", type=float, default=15.0,
        help="Budget cap in USD (default: 15.0)",
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=20,
        help="Max concurrent API requests (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (must match Block 0.1)",
    )
    parser.add_argument(
        "--model", type=str, default="models/gemini-2.5-flash",
        help="Model identifier (default: models/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--alpha_file", type=str, default="abc_exp/results/block0_alpha.jsonl",
        help="Path to Block 0.1 alpha estimates",
    )
    args = parser.parse_args()

    config = Config(
        model=args.model,
        n_questions=args.n_questions,
        n_debates_per_q=args.n_debates_per_q,
        n_rounds=args.n_rounds,
        budget_cap=args.budget,
        max_concurrent=args.max_concurrent,
        seed=args.seed,
        alpha_file=args.alpha_file,
    )

    asyncio.run(main(config))
