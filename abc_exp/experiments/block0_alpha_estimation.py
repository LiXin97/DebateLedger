"""Block 0.1: Alpha Estimation Experiment.

Estimates belief revisability (alpha) for agents on MMLU-Pro questions.
- 200 questions (stratified sample)
- 3 agents per question (temperature 0.7 for diversity)
- 8 adversarial probes per agent (4 strength x 2 social)
- MLE alpha estimation under logistic model

Usage:
    python -m abc_exp.experiments.block0_alpha_estimation [--n_questions 200] [--budget 10.0] [--retest]
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
from abc_exp.src.alpha.estimation import estimate_alpha
from abc_exp.src.alpha.probes import run_probes
from abc_exp.src.alpha.revision import extract_answer
from abc_exp.src.data.loader import load_benchmark
from abc_exp.config.prompts import INITIAL_ANSWER_PROMPT

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
    budget_cap: float = 10.0  # USD for this experiment
    max_concurrent: int = 20
    seed: int = 42
    results_dir: str = "abc_exp/results"
    log_dir: str = "abc_exp/results/logs"
    retest: bool = False  # if True, writes to separate file for ICC computation


def get_results_path(config: Config) -> Path:
    suffix = "_retest" if config.retest else ""
    return Path(config.results_dir) / f"block0_alpha{suffix}.jsonl"


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


async def process_question(
    client: GeminiClient,
    question,
    config: Config,
    rng: np.random.Generator,
) -> dict:
    """Process a single question: get initial answers + run probes + estimate alpha."""
    question_text = question.format_for_prompt()
    prompt = INITIAL_ANSWER_PROMPT.format(question=question_text)

    # Phase 1: Get initial answers from all agents (concurrent)
    from abc_exp.src.api.types import BatchItem

    initial_items = [
        BatchItem(
            contents=prompt,
            metadata={"question_id": question.id, "agent_id": f"agent_{i}"},
            temperature=config.agent_temperatures[i],
        )
        for i in range(config.n_agents)
    ]

    initial_results = await asyncio.gather(*[
        client.generate(
            contents=item.contents,
            temperature=item.temperature,
            max_output_tokens=1024,
            metadata=item.metadata,
        )
        for item in initial_items
    ])

    # Extract initial answers
    agents = []
    for i, result in enumerate(initial_results):
        answer = extract_answer(result.text, question.option_labels)
        agents.append({
            "agent_id": f"agent_{i}",
            "initial_answer": answer,
            "initial_correct": answer == question.correct_label if answer else False,
            "initial_cost": result.cost,
            "temperature": config.agent_temperatures[i],
        })

    # Phase 2: Run probes for each agent (concurrent across agents)
    all_probe_results = {}
    all_alpha_estimates = {}

    probe_tasks = []
    for i, agent in enumerate(agents):
        if agent["initial_answer"] is None:
            logger.warning(
                "Question %s agent_%d: failed to extract initial answer, skipping probes",
                question.id, i,
            )
            continue
        # Each agent gets its own RNG to ensure different wrong options
        agent_rng = np.random.default_rng(rng.integers(0, 2**32) + i)
        probe_tasks.append((
            i,
            run_probes(client, question, agent["initial_answer"], f"agent_{i}", agent_rng),
        ))

    for i, task in probe_tasks:
        probe_results = await task
        all_probe_results[f"agent_{i}"] = [p.to_dict() for p in probe_results]

        # Estimate alpha
        alpha_est = estimate_alpha(probe_results)
        all_alpha_estimates[f"agent_{i}"] = alpha_est.to_dict()

    # Build result record
    total_cost = sum(a["initial_cost"] for a in agents)
    for probes in all_probe_results.values():
        total_cost += sum(p["cost"] for p in probes)

    record = {
        "question_id": question.id,
        "benchmark": question.benchmark,
        "category": question.metadata.get("category", "unknown"),
        "correct_label": question.correct_label,
        "agents": agents,
        "alpha_estimates": all_alpha_estimates,
        "probe_results": all_probe_results,
        "total_cost": total_cost,
    }

    return record


async def main(config: Config):
    """Run the Block 0.1 experiment."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = get_results_path(config)
    completed = load_completed_ids(results_path)

    if completed:
        logger.info("Resuming: %d questions already completed", len(completed))

    # Load and sample questions
    logger.info("Loading MMLU-Pro dataset...")
    questions = load_benchmark("mmlu_pro", n_samples=config.n_questions, seed=config.seed)
    logger.info("Loaded %d questions", len(questions))

    # Filter out already completed
    remaining = [q for q in questions if q.id not in completed]
    logger.info("Remaining: %d questions to process", len(remaining))

    if not remaining:
        logger.info("All questions already completed!")
        return

    # Initialize client
    client = GeminiClient(
        model=config.model,
        budget_cap=config.budget_cap,
        max_concurrent=config.max_concurrent,
        log_dir=config.log_dir,
    )

    rng = np.random.default_rng(config.seed)

    # Process questions one at a time (each question has concurrent agents/probes)
    total_cost = 0.0
    n_processed = 0

    try:
        for i, question in enumerate(remaining):
            logger.info(
                "[%d/%d] Processing question %s (category: %s) | Cost so far: $%.4f",
                i + 1, len(remaining), question.id,
                question.metadata.get("category", "?"), total_cost,
            )

            # Per-question RNG for reproducibility
            q_rng = np.random.default_rng(rng.integers(0, 2**32))

            record = await process_question(client, question, config, q_rng)

            # Append to JSONL (incremental save)
            with open(results_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_cost += record["total_cost"]
            n_processed += 1

            # Print alpha summary for this question
            for agent_id, alpha in record["alpha_estimates"].items():
                logger.info(
                    "  %s: alpha=%.3f (CI: %.3f-%.3f), flipped %d/%d probes",
                    agent_id, alpha["alpha"], alpha["ci_lower"], alpha["ci_upper"],
                    alpha["n_revised"], alpha["n_probes"],
                )

    except Exception as e:
        logger.error("Experiment stopped: %s", e)
        raise
    finally:
        # Print summary
        print("\n" + "=" * 60)
        print("BLOCK 0.1 SUMMARY")
        print("=" * 60)
        print(f"Questions processed: {n_processed}")
        print(f"Total completed: {len(completed) + n_processed}/{config.n_questions}")
        print(f"Total cost this run: ${total_cost:.4f}")
        print(f"Results saved to: {results_path}")
        print(client.cost_tracker.summary())
        print("=" * 60)

        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block 0.1: Alpha Estimation")
    parser.add_argument("--n_questions", type=int, default=200)
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--max_concurrent", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retest", action="store_true",
                        help="Run as retest (saves to separate file for ICC)")
    parser.add_argument("--model", type=str, default="models/gemini-2.5-flash")
    args = parser.parse_args()

    config = Config(
        model=args.model,
        n_questions=args.n_questions,
        n_agents=args.n_agents,
        budget_cap=args.budget,
        max_concurrent=args.max_concurrent,
        seed=args.seed,
        retest=args.retest,
    )

    asyncio.run(main(config))
