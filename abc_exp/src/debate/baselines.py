"""Baseline aggregation methods for comparison against ABC debate.

Baselines:
1. majority_vote_baseline      — Simple majority vote (no debate)
2. confidence_weighted_baseline — Ask each agent for self-rated confidence, weight by it
3. self_consistency_baseline    — Sample N answers per agent, majority across all samples
4. best_of_n_baseline          — Generate N independent answers, majority vote

All baselines return a BaselineResult with accuracy info and cost.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from abc_exp.src.api.client import GeminiClient
from abc_exp.src.data.question import Question
from abc_exp.config.prompts import INITIAL_ANSWER_PROMPT
from abc_exp.src.alpha.revision import extract_answer

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from a baseline method on a single question."""
    question_id: str
    benchmark: str
    correct_label: str
    method: str                # "majority_vote", "confidence_weighted", etc.
    final_answer: str
    final_correct: bool
    all_answers: list[str | None]  # all individual answers considered
    total_cost: float
    metadata: dict = field(default_factory=dict)  # method-specific data

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "benchmark": self.benchmark,
            "correct_label": self.correct_label,
            "method": self.method,
            "final_answer": self.final_answer,
            "final_correct": self.final_correct,
            "all_answers": self.all_answers,
            "total_cost": self.total_cost,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Helper: majority vote (re-exports from standard for convenience)
# ---------------------------------------------------------------------------

def _majority_vote(answers: list[str | None]) -> str:
    """Return majority answer. Break ties alphabetically. Filter None."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return ""
    counts = Counter(valid)
    max_count = max(counts.values())
    tied = sorted(a for a, c in counts.items() if c == max_count)
    return tied[0]


# ---------------------------------------------------------------------------
# Baseline 1: Simple majority vote (no debate)
# ---------------------------------------------------------------------------

def majority_vote_baseline(
    answers: list[str | None],
    question: Question,
) -> BaselineResult:
    """Simple majority vote over pre-existing answers.

    This is the cheapest baseline: no API calls, just aggregates the
    initial answers from the agents.

    Parameters
    ----------
    answers : list[str | None]
        Pre-computed answers from agents (e.g. initial debate answers).
    question : Question
        The question being answered.

    Returns
    -------
    BaselineResult
    """
    final = _majority_vote(answers)
    return BaselineResult(
        question_id=question.id,
        benchmark=question.benchmark,
        correct_label=question.correct_label,
        method="majority_vote",
        final_answer=final,
        final_correct=(final == question.correct_label),
        all_answers=list(answers),
        total_cost=0.0,
    )


# ---------------------------------------------------------------------------
# Baseline 2: Confidence-weighted aggregation
# ---------------------------------------------------------------------------

CONFIDENCE_PROMPT = """\
You are an expert. Answer the following multiple-choice question.
Think step by step, then give your final answer AND a confidence score.

{question}

You MUST end your response with EXACTLY this format (on separate lines):
Final Answer: X
Confidence: N

Where X is the letter of your chosen option and N is your confidence as an \
integer from 0 to 100 (0 = pure guess, 100 = absolutely certain)."""


def _extract_confidence(text: str) -> int:
    """Extract confidence score from model output. Default 50 if not found."""
    # Try "Confidence: N" pattern
    m = re.search(r"Confidence\s*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        return max(0, min(100, val))
    return 50  # default


async def confidence_weighted_baseline(
    client: GeminiClient,
    question: Question,
    n_agents: int = 3,
    temperatures: tuple[float, ...] = (0.5, 0.7, 1.0),
) -> BaselineResult:
    """Ask each agent for answer + confidence, weight by confidence.

    Parameters
    ----------
    client : GeminiClient
        API client.
    question : Question
        The question.
    n_agents : int
        Number of agents.
    temperatures : tuple[float, ...]
        Per-agent temperatures.

    Returns
    -------
    BaselineResult
    """
    formatted_question = question.format_for_prompt()
    prompt = CONFIDENCE_PROMPT.format(question=formatted_question)
    valid_labels = question.option_labels

    agent_temps = tuple(
        temperatures[i % len(temperatures)] for i in range(n_agents)
    )

    async def _get_answer(agent_idx: int):
        result = await client.generate(
            contents=prompt,
            temperature=agent_temps[agent_idx],
            metadata={
                "agent_id": str(agent_idx + 1),
                "phase": "confidence_baseline",
                "question_id": question.id,
            },
        )
        answer = extract_answer(result.text, valid_labels)
        confidence = _extract_confidence(result.text)
        return answer, confidence, result.cost

    results = await asyncio.gather(*[_get_answer(i) for i in range(n_agents)])

    answers = [r[0] for r in results]
    confidences = [r[1] for r in results]
    costs = [r[2] for r in results]
    total_cost = sum(costs)

    # Weighted aggregation: sum confidence per answer, pick highest
    answer_weights: dict[str, float] = {}
    for ans, conf in zip(answers, confidences):
        if ans is not None:
            answer_weights[ans] = answer_weights.get(ans, 0.0) + conf

    if answer_weights:
        max_weight = max(answer_weights.values())
        tied = sorted(a for a, w in answer_weights.items() if abs(w - max_weight) < 1e-10)
        final = tied[0]
    else:
        final = ""

    return BaselineResult(
        question_id=question.id,
        benchmark=question.benchmark,
        correct_label=question.correct_label,
        method="confidence_weighted",
        final_answer=final,
        final_correct=(final == question.correct_label),
        all_answers=answers,
        total_cost=total_cost,
        metadata={
            "confidences": confidences,
            "answer_weights": answer_weights,
        },
    )


# ---------------------------------------------------------------------------
# Baseline 3: Self-consistency (sample N answers per agent)
# ---------------------------------------------------------------------------

async def self_consistency_baseline(
    client: GeminiClient,
    question: Question,
    n_samples: int = 5,
    temperature: float = 0.7,
) -> BaselineResult:
    """Self-consistency: sample N answers, take majority across all samples.

    This implements the Wang et al. (2022) self-consistency idea.
    We generate n_samples independent answers at a moderate temperature
    and take the majority vote across all samples.

    Parameters
    ----------
    client : GeminiClient
        API client.
    question : Question
        The question.
    n_samples : int
        Number of independent samples.
    temperature : float
        Sampling temperature (should be > 0 for diversity).

    Returns
    -------
    BaselineResult
    """
    formatted_question = question.format_for_prompt()
    prompt = INITIAL_ANSWER_PROMPT.format(question=formatted_question)
    valid_labels = question.option_labels

    async def _sample(idx: int):
        result = await client.generate(
            contents=prompt,
            temperature=temperature,
            metadata={
                "sample_id": str(idx),
                "phase": "self_consistency",
                "question_id": question.id,
            },
        )
        answer = extract_answer(result.text, valid_labels)
        return answer, result.cost

    results = await asyncio.gather(*[_sample(i) for i in range(n_samples)])

    answers = [r[0] for r in results]
    costs = [r[1] for r in results]
    total_cost = sum(costs)

    final = _majority_vote(answers)

    return BaselineResult(
        question_id=question.id,
        benchmark=question.benchmark,
        correct_label=question.correct_label,
        method="self_consistency",
        final_answer=final,
        final_correct=(final == question.correct_label),
        all_answers=answers,
        total_cost=total_cost,
        metadata={
            "n_samples": n_samples,
            "temperature": temperature,
        },
    )


# ---------------------------------------------------------------------------
# Baseline 4: Best-of-N (generate N answers, majority vote)
# ---------------------------------------------------------------------------

async def best_of_n_baseline(
    client: GeminiClient,
    question: Question,
    n: int = 5,
    temperature: float = 0.0,
) -> BaselineResult:
    """Best-of-N: generate N independent answers at low temperature, majority vote.

    Unlike self-consistency (which uses higher temperature for diversity),
    this uses temperature=0 for each sample. If all samples agree, we have
    high confidence. If they disagree, majority vote resolves it.

    This is the "scaling inference compute" baseline: throw more compute
    at each question without any debate structure.

    Parameters
    ----------
    client : GeminiClient
        API client.
    question : Question
        The question.
    n : int
        Number of independent answers.
    temperature : float
        Sampling temperature (default 0 for greedy).

    Returns
    -------
    BaselineResult
    """
    formatted_question = question.format_for_prompt()
    prompt = INITIAL_ANSWER_PROMPT.format(question=formatted_question)
    valid_labels = question.option_labels

    async def _generate(idx: int):
        # Use slightly varied temperatures for best-of-N to get diversity
        # even at low base temp. Shift by small amounts.
        temp = max(0.0, temperature + idx * 0.1)
        result = await client.generate(
            contents=prompt,
            temperature=temp,
            metadata={
                "sample_id": str(idx),
                "phase": "best_of_n",
                "question_id": question.id,
            },
        )
        answer = extract_answer(result.text, valid_labels)
        return answer, result.cost

    results = await asyncio.gather(*[_generate(i) for i in range(n)])

    answers = [r[0] for r in results]
    costs = [r[1] for r in results]
    total_cost = sum(costs)

    final = _majority_vote(answers)

    return BaselineResult(
        question_id=question.id,
        benchmark=question.benchmark,
        correct_label=question.correct_label,
        method="best_of_n",
        final_answer=final,
        final_correct=(final == question.correct_label),
        all_answers=answers,
        total_cost=total_cost,
        metadata={
            "n": n,
            "base_temperature": temperature,
        },
    )
