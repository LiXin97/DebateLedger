"""Standard multi-agent debate protocol.

Implements Du et al. (2023): 3 agents, T rounds, majority vote.
Each agent sees all others' previous responses and updates.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field

from abc_exp.src.api.client import GeminiClient
from abc_exp.src.data.question import Question
from abc_exp.config.prompts import INITIAL_ANSWER_PROMPT, DEBATE_ROUND_PROMPT, DEBATE_SYSTEM_PROMPT
from abc_exp.src.alpha.revision import extract_answer

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    agent_id: str
    answer: str | None    # extracted answer letter
    reasoning: str        # full response text
    round_num: int
    cost: float


@dataclass
class DebateResult:
    question_id: str
    benchmark: str
    correct_label: str
    initial_states: list[AgentState]        # round 0 (before debate)
    round_states: list[list[AgentState]]    # per-round states
    final_answer: str                       # majority vote
    final_correct: bool
    initial_majority: str                   # majority before debate
    initial_correct: bool                   # was initial majority correct
    collapsed: bool                         # initially-correct majority flipped to wrong
    corrected: bool                         # initially-wrong majority flipped to correct
    n_rounds: int
    total_cost: float

    @property
    def initial_answers(self) -> list[str | None]:
        return [s.answer for s in self.initial_states]

    @property
    def final_answers(self) -> list[str | None]:
        if self.round_states:
            return [s.answer for s in self.round_states[-1]]
        return self.initial_answers

    def to_dict(self) -> dict:
        def _state_to_dict(s: AgentState) -> dict:
            return {
                "agent_id": s.agent_id,
                "answer": s.answer,
                "reasoning": s.reasoning,
                "round_num": s.round_num,
                "cost": s.cost,
            }

        return {
            "question_id": self.question_id,
            "benchmark": self.benchmark,
            "correct_label": self.correct_label,
            "initial_states": [_state_to_dict(s) for s in self.initial_states],
            "round_states": [
                [_state_to_dict(s) for s in round_list]
                for round_list in self.round_states
            ],
            "final_answer": self.final_answer,
            "final_correct": self.final_correct,
            "initial_majority": self.initial_majority,
            "initial_correct": self.initial_correct,
            "collapsed": self.collapsed,
            "corrected": self.corrected,
            "n_rounds": self.n_rounds,
            "total_cost": self.total_cost,
        }


def majority_vote(answers: list[str]) -> str:
    """Return majority answer. Break ties alphabetically (first in sorted order).

    Filters out None answers before counting. If all answers are None, returns "".
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return ""
    counts = Counter(valid)
    max_count = max(counts.values())
    # All answers with the maximum count, sorted alphabetically for tie-breaking.
    tied = sorted(a for a, c in counts.items() if c == max_count)
    return tied[0]


def _format_agent_responses(states: list[AgentState]) -> str:
    """Format all agents' responses for the debate prompt."""
    parts = []
    for s in states:
        answer_str = s.answer if s.answer else "unclear"
        parts.append(
            f"Expert {s.agent_id}:\n"
            f"Answer: {answer_str}\n"
            f"Reasoning: {s.reasoning}"
        )
    return "\n\n---\n\n".join(parts)


async def run_debate(
    client: GeminiClient,
    question: Question,
    n_agents: int = 3,
    n_rounds: int = 3,
    temperatures: tuple[float, ...] = (0.5, 0.7, 1.0),
) -> DebateResult:
    """Run a standard multi-agent debate.

    Phase 1: Each agent answers independently (different temperatures for diversity)
    Phase 2: For each round, each agent sees all others' responses and updates
    Phase 3: Majority vote on final answers

    Track collapse: if initial correct majority -> final wrong majority
    Track correction: if initial wrong majority -> final correct majority
    """
    valid_labels = question.option_labels
    formatted_question = question.format_for_prompt()
    total_cost = 0.0

    # Ensure we have enough temperatures; cycle if needed.
    agent_temps = tuple(
        temperatures[i % len(temperatures)] for i in range(n_agents)
    )

    # ------------------------------------------------------------------
    # Phase 1: Independent initial answers
    # ------------------------------------------------------------------
    async def _get_initial_answer(agent_idx: int) -> AgentState:
        agent_id = str(agent_idx + 1)
        prompt = INITIAL_ANSWER_PROMPT.format(question=formatted_question)
        result = await client.generate(
            contents=prompt,
            temperature=agent_temps[agent_idx],
            metadata={"agent_id": agent_id, "phase": "initial", "question_id": question.id},
        )
        answer = extract_answer(result.text, valid_labels)
        return AgentState(
            agent_id=agent_id,
            answer=answer,
            reasoning=result.text,
            round_num=0,
            cost=result.cost,
        )

    initial_states = list(await asyncio.gather(
        *[_get_initial_answer(i) for i in range(n_agents)]
    ))
    total_cost += sum(s.cost for s in initial_states)

    initial_answers = [s.answer for s in initial_states]
    initial_majority = majority_vote(initial_answers)
    initial_correct = (initial_majority == question.correct_label)

    # ------------------------------------------------------------------
    # Phase 2: Debate rounds
    # ------------------------------------------------------------------
    round_states: list[list[AgentState]] = []
    current_states = initial_states  # most recent state per agent

    for round_num in range(1, n_rounds + 1):
        agent_responses_text = _format_agent_responses(current_states)

        async def _debate_round_agent(agent_idx: int) -> AgentState:
            agent_id = str(agent_idx + 1)
            prompt = DEBATE_ROUND_PROMPT.format(
                agent_responses=agent_responses_text,
                agent_num=agent_id,
                question=formatted_question,
            )
            result = await client.generate(
                contents=prompt,
                temperature=agent_temps[agent_idx],
                system_instruction=DEBATE_SYSTEM_PROMPT,
                metadata={
                    "agent_id": agent_id,
                    "phase": "debate",
                    "round": round_num,
                    "question_id": question.id,
                },
            )
            answer = extract_answer(result.text, valid_labels)
            return AgentState(
                agent_id=agent_id,
                answer=answer,
                reasoning=result.text,
                round_num=round_num,
                cost=result.cost,
            )

        new_states = list(await asyncio.gather(
            *[_debate_round_agent(i) for i in range(n_agents)]
        ))
        total_cost += sum(s.cost for s in new_states)
        round_states.append(new_states)
        current_states = new_states

    # ------------------------------------------------------------------
    # Phase 3: Final majority vote
    # ------------------------------------------------------------------
    final_answers = [s.answer for s in current_states]
    final_answer = majority_vote(final_answers)
    final_correct = (final_answer == question.correct_label)

    collapsed = initial_correct and not final_correct
    corrected = not initial_correct and final_correct

    return DebateResult(
        question_id=question.id,
        benchmark=question.benchmark,
        correct_label=question.correct_label,
        initial_states=initial_states,
        round_states=round_states,
        final_answer=final_answer,
        final_correct=final_correct,
        initial_majority=initial_majority,
        initial_correct=initial_correct,
        collapsed=collapsed,
        corrected=corrected,
        n_rounds=n_rounds,
        total_cost=total_cost,
    )
