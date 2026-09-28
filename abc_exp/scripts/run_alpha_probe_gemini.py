"""Run S/A alpha probes against Gemini models on the canonical high-FR pool.

This is the Gemini sibling of ``run_alpha_probe_openrouter.py``.  It preserves
the one-row-per ``(question, condition, agent_idx)`` schema used by the current
post-A6 artifacts, while using ``GeminiClient`` so model-aware thinking budgets
and project cost accounting stay centralized.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from abc_exp.config.prompts import PROBE_TEMPLATES, SOCIAL_LEVELS, SOCIAL_SUFFIX  # noqa: E402
from abc_exp.src.alpha.revision import extract_answer  # noqa: E402
from abc_exp.src.api.client import GeminiClient  # noqa: E402
from abc_exp.src.api.cost import BudgetExhaustedError  # noqa: E402
from abc_exp.src.data.loader import load_benchmark  # noqa: E402
from abc_exp.src.data.question import Question  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HIGH_FR_PATH = REPO_ROOT / "abc_exp/results/high_fr_question_ids.json"

DEFAULT_SYSTEM = "You are an expert."
ANTI_SOCIAL_SYSTEM = (
    "You are an expert. "
    "When evaluating counterarguments, focus EXCLUSIVELY on the logical quality "
    "of the argument itself. Whether others agree or disagree with you is completely "
    "irrelevant -- only the strength of the reasoning matters. A weak argument from "
    "a unanimous group deserves less weight than a strong argument from a single voice."
)
ANTI_ARGUMENT_SYSTEM = (
    "You are an expert. "
    "When evaluating counterarguments, pay close attention to the consensus view. "
    "If multiple experts agree on something, that is strong evidence even if you "
    "cannot immediately follow every detail of their reasoning. Group wisdom is "
    "often more reliable than individual analysis, especially for difficult questions."
)

CONDITION_SYSTEMS = {
    "default": DEFAULT_SYSTEM,
    "anti_social": ANTI_SOCIAL_SYSTEM,
    "anti_argument": ANTI_ARGUMENT_SYSTEM,
}

INITIAL_PARSE_RETRIES = 3
INITIAL_FORMAT_FOLLOWUPS = 2


def mean(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def load_high_fr_questions(n_questions: int | None) -> list[Question]:
    qids = json.loads(HIGH_FR_PATH.read_text())
    if n_questions is not None:
        qids = qids[:n_questions]
    all_questions = load_benchmark("mmlu_pro", n_samples=12000, seed=0)
    by_id = {q.id: q for q in all_questions}
    missing = [qid for qid in qids if qid not in by_id]
    if missing:
        raise SystemExit(f"Missing MMLU-Pro question IDs: {missing[:5]}")
    return [by_id[qid] for qid in qids]


def load_completed(path: Path, resume: bool) -> set[tuple[str, str, int]]:
    if not resume or not path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((rec["question_id"], rec["condition"], int(rec["agent_idx"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def build_triples(questions: list[Question], completed: set[tuple[str, str, int]]) -> list[dict[str, Any]]:
    temperatures = (0.5, 0.7, 1.0)
    triples: list[dict[str, Any]] = []
    for question in questions:
        for condition in CONDITION_SYSTEMS:
            for agent_idx, temperature in enumerate(temperatures):
                key = (question.id, condition, agent_idx)
                if key in completed:
                    continue
                triples.append(
                    {
                        "question": question,
                        "condition": condition,
                        "agent_idx": agent_idx,
                        "temperature": temperature,
                    }
                )
    return triples


def construct_probe(alt_answer: str, alt_text: str, strength: str, social: bool) -> tuple[str, float]:
    # We do not have option text at this point; the prompt below includes the
    # full question, so the label is enough to identify the target option.
    template = PROBE_TEMPLATES[strength]
    probe_text = template["template"].format(wrong_label=alt_answer, wrong_text=alt_text)
    if social:
        social_info = SOCIAL_LEVELS[True]
        probe_text += SOCIAL_SUFFIX.format(
            n_dissenters=social_info["n_dissenters"],
            n_total=social_info["n_total"],
            wrong_label=alt_answer,
        )
    return probe_text, float(template["strength"])


def usage_dict(result) -> dict[str, Any]:
    return {
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "thinking_tokens": result.thinking_tokens,
        "cost": result.cost,
        "model": result.model,
        "timestamp": result.timestamp,
    }


async def run_agent_probes(
    *,
    client: GeminiClient,
    triple: dict[str, Any],
    max_tokens_initial: int,
    max_tokens_probe: int,
    seed: int,
) -> dict[str, Any] | None:
    question: Question = triple["question"]
    condition = triple["condition"]
    agent_idx = int(triple["agent_idx"])
    temperature = float(triple["temperature"])
    valid_labels = question.option_labels
    formatted = question.format_for_prompt()
    system = CONDITION_SYSTEMS[condition]
    total_cost = 0.0
    initial_usages: list[dict[str, Any]] = []

    prompt = (
        "Answer the following multiple-choice question.\n"
        "Think step by step, then give your final answer.\n\n"
        f"{formatted}\n\n"
        'You MUST end your response with exactly: "Final Answer: X" '
        "where X is the letter of your chosen option."
    )

    async def repair_answer_format(previous_text: str, attempt: int) -> str | None:
        nonlocal total_cost
        repair_prompt = (
            "Your previous response did not end with a valid option letter.\n\n"
            f"Question:\n{formatted}\n\n"
            f"Previous response:\n{previous_text}\n\n"
            "Do not add reasoning. Based on your previous analysis, reply with exactly "
            "one capital-letter option from the question (for example: A)."
        )
        result = await client.generate(
            repair_prompt,
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=8,
            metadata={
                "phase": "initial_format_repair",
                "attempt": attempt,
                "question_id": question.id,
                "condition": condition,
                "agent_idx": agent_idx,
            },
        )
        total_cost += result.cost
        initial_usages.append({"phase": "initial_format_repair", "attempt": attempt, **usage_dict(result)})
        return extract_answer(result.text, valid_labels)

    initial_answer = None
    initial_text = ""
    for attempt in range(1, INITIAL_PARSE_RETRIES + 1):
        result = await client.generate(
            prompt,
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens_initial,
            metadata={
                "phase": "initial",
                "attempt": attempt,
                "question_id": question.id,
                "condition": condition,
                "agent_idx": agent_idx,
            },
        )
        total_cost += result.cost
        initial_text = result.text
        initial_usages.append({"phase": "initial", "attempt": attempt, **usage_dict(result)})
        initial_answer = extract_answer(result.text, valid_labels)
        if not initial_answer:
            for repair_attempt in range(1, INITIAL_FORMAT_FOLLOWUPS + 1):
                initial_answer = await repair_answer_format(result.text, repair_attempt)
                if initial_answer:
                    break
        if initial_answer:
            break
        logger.warning(
            "Failed to parse initial answer for %s/%s/agent%d on attempt %d",
            question.id,
            condition,
            agent_idx,
            attempt,
        )

    if not initial_answer:
        return None

    rng = random.Random(f"{seed}:{question.id}:{condition}:{agent_idx}")
    alt_labels = [label for label in valid_labels if label != initial_answer]
    alt_answer = rng.choice(alt_labels) if alt_labels else valid_labels[-1]
    alt_text_by_label = dict(zip(question.option_labels, question.options))
    alt_text = alt_text_by_label.get(alt_answer, f"option {alt_answer}")

    async def run_probe(strength: str, social: bool) -> dict[str, Any]:
        nonlocal total_cost
        probe_text, strength_value = construct_probe(alt_answer, alt_text, strength, social)
        probe_prompt = (
            "You previously answered the following question:\n\n"
            f"{formatted}\n\n"
            f"Your answer was: {initial_answer}\n\n"
            "Another expert responds:\n\n"
            f"{probe_text}\n\n"
            "After considering this response, what is your final answer? "
            'You MUST end with exactly: "Final Answer: X"'
        )
        result = await client.generate(
            probe_prompt,
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=max_tokens_probe,
            metadata={
                "phase": "probe",
                "question_id": question.id,
                "condition": condition,
                "agent_idx": agent_idx,
                "strength": strength,
                "social": social,
            },
        )
        total_cost += result.cost
        post_answer = extract_answer(result.text, valid_labels)
        revised = post_answer != initial_answer if post_answer else False
        return {
            "strength": strength,
            "strength_value": strength_value,
            "social": social,
            "alt_answer": alt_answer,
            "post_answer": post_answer,
            "revised": revised,
            "cost": float(result.cost),
            "usage": usage_dict(result),
        }

    probe_tasks = [
        run_probe(strength, social)
        for strength in PROBE_TEMPLATES
        for social in (False, True)
    ]
    probe_results = list(await asyncio.gather(*probe_tasks))

    flips = [int(p["revised"]) for p in probe_results]
    solo_flips = [int(p["revised"]) for p in probe_results if not p["social"]]
    social_flips = [int(p["revised"]) for p in probe_results if p["social"]]
    alpha_total = mean(flips)
    alpha_solo = mean(solo_flips)
    alpha_social = mean(social_flips)
    social_sensitivity = alpha_social - alpha_solo

    strength_flips: dict[str, list[int]] = defaultdict(list)
    strength_solo: dict[str, list[int]] = defaultdict(list)
    strength_social: dict[str, list[int]] = defaultdict(list)
    for probe in probe_results:
        strength = probe["strength"]
        flip = int(probe["revised"])
        strength_flips[strength].append(flip)
        if probe["social"]:
            strength_social[strength].append(flip)
        else:
            strength_solo[strength].append(flip)

    sorted_strengths = sorted(strength_flips, key=lambda s: PROBE_TEMPLATES[s]["strength"])
    if len(sorted_strengths) >= 2:
        argument_sensitivity = mean(strength_flips[sorted_strengths[-1]]) - mean(strength_flips[sorted_strengths[0]])
    else:
        argument_sensitivity = 0.0

    by_strength = {
        strength: {
            "alpha_total": mean(strength_flips[strength]),
            "alpha_solo": mean(strength_solo[strength]),
            "alpha_social": mean(strength_social[strength]),
            "n": len(strength_flips[strength]),
        }
        for strength in sorted_strengths
    }

    sa_ratio = float(social_sensitivity / argument_sensitivity) if argument_sensitivity > 0 else None
    timestamp = time.time()
    return {
        "question_id": question.id,
        "benchmark": question.benchmark,
        "category": question.metadata.get("category"),
        "correct_label": question.correct_label,
        "condition": condition,
        "agent_idx": agent_idx,
        "temperature": temperature,
        "initial_answer": initial_answer,
        "initial_correct": initial_answer == question.correct_label,
        "initial_parse_repaired": any(u["phase"] == "initial_format_repair" for u in initial_usages),
        "alpha_total": float(alpha_total),
        "alpha_solo": float(alpha_solo),
        "alpha_social": float(alpha_social),
        "by_strength": by_strength,
        "flip_rate": float(alpha_total),
        "social_sensitivity": float(social_sensitivity),
        "argument_sensitivity": float(argument_sensitivity),
        "sa_ratio": sa_ratio,
        "probe_results": probe_results,
        "total_cost": float(total_cost),
        "timestamp": timestamp,
        "model_name": client.model,
        "backend": "gemini_api",
        "thinking_budget": client.thinking_budget,
        "initial_usage": initial_usages,
        "initial_response_prefix": initial_text[:500],
    }


async def main(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    questions = load_high_fr_questions(args.n_questions)
    completed = load_completed(output_path, args.resume)
    triples = build_triples(questions, completed)
    logger.info(
        "pool=%d high-FR questions; resume_completed=%d; to_run=%d triples",
        len(questions),
        len(completed),
        len(triples),
    )
    if not triples:
        return 0

    client = GeminiClient(
        model=args.model,
        budget_cap=args.budget,
        max_concurrent=args.max_concurrent,
        log_dir=output_path.parent / "logs",
        thinking_budget=args.thinking_budget,
    )
    outer_sem = asyncio.Semaphore(max(1, args.outer_concurrency))
    write_lock = asyncio.Lock()
    open_mode = "a" if args.resume and output_path.exists() else "w"
    n_done = 0
    n_failed = 0
    t0 = time.time()

    out_f = output_path.open(open_mode)
    try:
        async def worker(triple: dict[str, Any]) -> None:
            nonlocal n_done, n_failed
            async with outer_sem:
                try:
                    row = await run_agent_probes(
                        client=client,
                        triple=triple,
                        max_tokens_initial=args.max_tokens_initial,
                        max_tokens_probe=args.max_tokens_probe,
                        seed=args.seed,
                    )
                except BudgetExhaustedError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    n_failed += 1
                    logger.error(
                        "error on %s/%s/agent%d: %s",
                        triple["question"].id,
                        triple["condition"],
                        triple["agent_idx"],
                        exc,
                    )
                    return
                if row is None:
                    n_failed += 1
                    return
                async with write_lock:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_done += 1
                    if n_done % args.report_every == 0 or n_done == len(triples):
                        elapsed = max(time.time() - t0, 1e-9)
                        logger.info(
                            "%d/%d rows done (failed=%d), cost=$%.4f, %.1f rows/min",
                            n_done,
                            len(triples),
                            n_failed,
                            client.cost_tracker.total_cost,
                            n_done / elapsed * 60,
                        )

        tasks = [asyncio.create_task(worker(triple)) for triple in triples]
        try:
            await asyncio.gather(*tasks)
        except BudgetExhaustedError as exc:
            logger.warning("%s; cancelling remaining work", exc)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        out_f.close()
        print(client.cost_tracker.summary())
        await client.aclose()

    elapsed = max(time.time() - t0, 1e-9)
    logger.info(
        "complete: wrote %d rows (failed=%d) to %s in %.0fs (%.1f rows/min)",
        n_done,
        n_failed,
        output_path,
        elapsed,
        n_done / elapsed * 60,
    )
    if math.isclose(client.cost_tracker.total_cost, 0.0):
        logger.warning("Gemini cost tracker remained at $0.0000")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini S/A alpha probes")
    parser.add_argument("--model", required=True, help="Gemini model id, e.g. models/gemini-3.1-pro-preview")
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--budget", type=float, default=100.0, help="Gemini cost cap in USD")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Maximum concurrent Gemini HTTP calls")
    parser.add_argument("--outer-concurrency", type=int, default=2, help="Maximum concurrent alpha triples")
    parser.add_argument("--n-questions", type=int, default=None, help="Optional cap on canonical high-FR questions")
    parser.add_argument("--max-tokens-initial", type=int, default=1024)
    parser.add_argument("--max-tokens-probe", type=int, default=512)
    parser.add_argument("--thinking-budget", type=int, default=None, help="Override Gemini thinking budget; default is model-aware")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip existing (question_id, condition, agent_idx) rows")
    parser.add_argument("--report-every", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
