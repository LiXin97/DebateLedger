"""Adversarial probe construction and execution.

Implements the 4x2 factorial probe design:
- 4 argument strength levels (weak, moderate, strong, very_strong)
- 2 social pressure levels (no pressure, 2/3 majority disagree)
Total: 8 probes per agent per question.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...config.prompts import PROBE_TEMPLATES, SOCIAL_LEVELS, SOCIAL_SUFFIX
from .revision import detect_revision, extract_answer

if TYPE_CHECKING:
    from ..api.client import GeminiClient
    from ..data.question import Question


@dataclass
class ProbeResult:
    """Result from a single adversarial probe."""
    question_id: str
    agent_id: str
    probe_type: str        # "weak", "moderate", "strong", "very_strong"
    social: bool           # whether social pressure was applied
    strength: float        # s value (0.25, 0.50, 0.75, 1.00)
    original_answer: str   # agent's initial answer
    probe_answer: str | None  # answer after probe (None if extraction failed)
    revised: bool          # True if agent changed answer
    cost: float            # API cost for this probe call
    raw_response: str      # full model response text

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "agent_id": self.agent_id,
            "probe_type": self.probe_type,
            "social": self.social,
            "strength": self.strength,
            "original_answer": self.original_answer,
            "probe_answer": self.probe_answer,
            "revised": self.revised,
            "cost": self.cost,
        }


def construct_probes(
    question: Question,
    agent_answer: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Construct 8 probe prompts for a given agent's answer.

    Returns a list of dicts, each with:
    - "prompt": the full probe text
    - "probe_type": strength name
    - "social": bool
    - "strength": float (s value)
    """
    probes = []
    # Pick a wrong option to use across probes (consistent per agent-question)
    wrong_label, wrong_text = question.random_wrong_option(rng)

    for probe_name, probe_info in PROBE_TEMPLATES.items():
        template = probe_info["template"]
        strength = probe_info["strength"]

        for social in [False, True]:
            # Format the probe text
            probe_text = template.format(
                wrong_label=wrong_label,
                wrong_text=wrong_text,
            )

            # Add social pressure suffix if needed
            if social:
                social_info = SOCIAL_LEVELS[True]
                probe_text += SOCIAL_SUFFIX.format(
                    n_dissenters=social_info["n_dissenters"],
                    n_total=social_info["n_total"],
                    wrong_label=wrong_label,
                )

            probes.append({
                "prompt": probe_text,
                "probe_type": probe_name,
                "social": social,
                "strength": strength,
            })

    return probes


async def run_probes(
    client: GeminiClient,
    question: Question,
    agent_answer: str,
    agent_id: str,
    rng: np.random.Generator,
) -> list[ProbeResult]:
    """Run all 8 probes for one agent on one question.

    Probes are fired concurrently (they are independent).
    Returns list of 8 ProbeResult objects.
    """
    probes = construct_probes(question, agent_answer, rng)
    valid_labels = question.option_labels

    # Build conversation context for each probe:
    # The agent first answered the question, then we present the probe.
    question_text = question.format_for_prompt()

    async def _run_single_probe(probe: dict) -> ProbeResult:
        # Build a multi-turn prompt: first the question + answer, then the probe
        contents = [
            {"role": "user", "parts": [{"text": f"Answer this question:\n\n{question_text}"}]},
            {"role": "model", "parts": [{"text": f"Final Answer: {agent_answer}"}]},
            {"role": "user", "parts": [{"text": probe["prompt"]}]},
        ]

        result = await client.generate(
            contents=contents,
            temperature=0.0,  # deterministic for measurement
            max_output_tokens=512,
            metadata={
                "question_id": question.id,
                "agent_id": agent_id,
                "probe_type": probe["probe_type"],
                "social": probe["social"],
            },
        )

        probe_answer = extract_answer(result.text, valid_labels)
        revised = detect_revision(agent_answer, probe_answer)

        return ProbeResult(
            question_id=question.id,
            agent_id=agent_id,
            probe_type=probe["probe_type"],
            social=probe["social"],
            strength=probe["strength"],
            original_answer=agent_answer,
            probe_answer=probe_answer,
            revised=revised,
            cost=result.cost,
            raw_response=result.text,
        )

    # Run all 8 probes concurrently
    tasks = [_run_single_probe(p) for p in probes]
    results = await asyncio.gather(*tasks)
    return list(results)
