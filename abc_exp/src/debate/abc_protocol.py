"""ABC (Adversarial Belief Calibration) weighted debate protocol.

Key insight from Block 0 analysis:
- Raw flip rate has ICC=0.760 (reliable measure of revisability)
- In collapse scenarios, CORRECT agents are MORE stubborn (lower flip rate)
- LOW revisability = more trustworthy
- Weight: w_i = 1 / (revisability_i + epsilon) -- stubborn agents get higher weight

The ABC protocol modifies standard debate by:
1. Taking pre-computed revisability scores per agent (from Block 0.1 data)
2. Applying inverse-revisability weighting at each round
3. Using weighted majority as the "consensus signal" shown to agents
4. Including a CONFIDENCE_SHIELD prompt to discourage conformity
5. Tracking per-round weighted votes, collapse events, correction events
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from abc_exp.src.api.client import GeminiClient
from abc_exp.src.data.question import Question
from abc_exp.config.prompts import (
    INITIAL_ANSWER_PROMPT,
    DEBATE_ROUND_PROMPT,
    DEBATE_SYSTEM_PROMPT,
    CONFIDENCE_SHIELD,
)
from abc_exp.src.alpha.revision import extract_answer
from abc_exp.src.debate.standard import AgentState, majority_vote

logger = logging.getLogger(__name__)

# Default epsilon to avoid division by zero in 1/(revisability + epsilon).
DEFAULT_EPSILON = 0.05


@dataclass
class WeightedVote:
    """A single agent's weighted vote at one round."""
    agent_id: str
    answer: str | None
    raw_weight: float        # 1 / (revisability + epsilon)
    normalized_weight: float  # raw_weight / sum(raw_weights)
    revisability: float       # agent's pre-computed revisability score


@dataclass
class RoundSummary:
    """Summary of weighted voting for one debate round."""
    round_num: int
    weighted_votes: list[WeightedVote]
    weighted_majority: str           # answer with highest total weight
    unweighted_majority: str         # plain majority for comparison
    answer_weights: dict[str, float]  # answer -> total normalized weight
    consensus_strength: float         # weight of winning answer (0-1)


@dataclass
class ABCDebateResult:
    """Full result from an ABC weighted debate."""
    question_id: str
    benchmark: str
    correct_label: str

    # Agent states per phase
    initial_states: list[AgentState]
    round_states: list[list[AgentState]]

    # Per-round weighted voting summaries
    round_summaries: list[RoundSummary]

    # Final outcome
    final_answer: str              # weighted majority at last round
    final_correct: bool
    final_unweighted_answer: str   # unweighted majority for comparison
    final_unweighted_correct: bool

    # Initial outcome
    initial_majority: str
    initial_correct: bool
    initial_weighted_majority: str
    initial_weighted_correct: bool

    # Event tracking
    collapsed: bool                # initial correct -> final wrong (weighted)
    corrected: bool                # initial wrong -> final correct (weighted)
    collapsed_unweighted: bool     # same but for unweighted
    corrected_unweighted: bool

    # Per-agent revisability used
    agent_revisabilities: dict[str, float]

    # Cost and config
    n_rounds: int
    total_cost: float
    epsilon: float

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

        def _vote_to_dict(v: WeightedVote) -> dict:
            return {
                "agent_id": v.agent_id,
                "answer": v.answer,
                "raw_weight": v.raw_weight,
                "normalized_weight": v.normalized_weight,
                "revisability": v.revisability,
            }

        def _round_to_dict(r: RoundSummary) -> dict:
            return {
                "round_num": r.round_num,
                "weighted_votes": [_vote_to_dict(v) for v in r.weighted_votes],
                "weighted_majority": r.weighted_majority,
                "unweighted_majority": r.unweighted_majority,
                "answer_weights": r.answer_weights,
                "consensus_strength": r.consensus_strength,
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
            "round_summaries": [_round_to_dict(r) for r in self.round_summaries],
            "final_answer": self.final_answer,
            "final_correct": self.final_correct,
            "final_unweighted_answer": self.final_unweighted_answer,
            "final_unweighted_correct": self.final_unweighted_correct,
            "initial_majority": self.initial_majority,
            "initial_correct": self.initial_correct,
            "initial_weighted_majority": self.initial_weighted_majority,
            "initial_weighted_correct": self.initial_weighted_correct,
            "collapsed": self.collapsed,
            "corrected": self.corrected,
            "collapsed_unweighted": self.collapsed_unweighted,
            "corrected_unweighted": self.corrected_unweighted,
            "agent_revisabilities": self.agent_revisabilities,
            "n_rounds": self.n_rounds,
            "total_cost": self.total_cost,
            "epsilon": self.epsilon,
        }


def _compute_weighted_majority(
    states: list[AgentState],
    revisabilities: dict[str, float],
    epsilon: float,
) -> RoundSummary:
    """Compute weighted majority vote for a set of agent states.

    Weight: w_i = 1 / (revisability_i + epsilon)
    Stubborn agents (low revisability) get higher weight.
    """
    # Compute raw weights
    raw_weights: dict[str, float] = {}
    for s in states:
        rev = revisabilities.get(s.agent_id, 0.5)  # default to 0.5 if unknown
        raw_weights[s.agent_id] = 1.0 / (rev + epsilon)

    # Normalize
    total_raw = sum(raw_weights.values())
    norm_weights = {aid: w / total_raw for aid, w in raw_weights.items()}

    # Build weighted votes
    weighted_votes = []
    for s in states:
        rev = revisabilities.get(s.agent_id, 0.5)
        weighted_votes.append(WeightedVote(
            agent_id=s.agent_id,
            answer=s.answer,
            raw_weight=raw_weights[s.agent_id],
            normalized_weight=norm_weights[s.agent_id],
            revisability=rev,
        ))

    # Aggregate weights per answer
    answer_weights: dict[str, float] = defaultdict(float)
    for v in weighted_votes:
        if v.answer is not None:
            answer_weights[v.answer] += v.normalized_weight

    # Weighted majority: answer with highest total weight, break ties alphabetically
    if answer_weights:
        max_weight = max(answer_weights.values())
        tied = sorted(a for a, w in answer_weights.items() if abs(w - max_weight) < 1e-10)
        weighted_majority = tied[0]
        consensus_strength = max_weight
    else:
        weighted_majority = ""
        consensus_strength = 0.0

    # Unweighted majority for comparison
    unweighted_majority = majority_vote([s.answer for s in states])

    return RoundSummary(
        round_num=states[0].round_num if states else 0,
        weighted_votes=weighted_votes,
        weighted_majority=weighted_majority,
        unweighted_majority=unweighted_majority,
        answer_weights=dict(answer_weights),
        consensus_strength=consensus_strength,
    )


def _format_agent_responses_with_weights(
    states: list[AgentState],
    round_summary: RoundSummary,
) -> str:
    """Format agent responses for the ABC debate prompt.

    Includes the weighted consensus signal so agents know the
    "calibration-weighted" group opinion, not just raw majority.
    """
    parts = []
    for s in states:
        answer_str = s.answer if s.answer else "unclear"
        parts.append(
            f"Expert {s.agent_id}:\n"
            f"Answer: {answer_str}\n"
            f"Reasoning: {s.reasoning}"
        )

    responses_text = "\n\n---\n\n".join(parts)

    # Add weighted consensus signal
    consensus_note = (
        f"\n\n=== Calibration-Weighted Consensus ===\n"
        f"Based on each expert's track record of belief calibration, the "
        f"weighted group assessment favors: {round_summary.weighted_majority} "
        f"(confidence: {round_summary.consensus_strength:.1%})"
    )

    return responses_text + consensus_note


# The ABC debate prompt augments the standard debate prompt with
# the confidence shield to discourage conformity.
ABC_DEBATE_SYSTEM_PROMPT = DEBATE_SYSTEM_PROMPT + "\n\n" + CONFIDENCE_SHIELD

ABC_DEBATE_ROUND_PROMPT = """\
The following experts have provided their answers and reasoning:

{agent_responses}

You are Expert {agent_num}. Review the other experts' reasoning carefully.

{confidence_shield}

If you find a genuine logical error in your own reasoning, you may change your answer.
Otherwise, defend your position with additional arguments.

Original question:
{question}

You MUST end your response with exactly: "Final Answer: X" where X is the letter of your chosen option."""


async def run_abc_debate(
    client: GeminiClient,
    question: Question,
    agent_revisabilities: dict[str, float],
    n_agents: int = 3,
    n_rounds: int = 3,
    temperatures: tuple[float, ...] = (0.5, 0.7, 1.0),
    epsilon: float = DEFAULT_EPSILON,
) -> ABCDebateResult:
    """Run an ABC weighted debate.

    Parameters
    ----------
    client : GeminiClient
        API client.
    question : Question
        The question to debate.
    agent_revisabilities : dict[str, float]
        Pre-computed revisability scores keyed by agent_id (e.g. "1", "2", "3").
        These come from Block 0.1 flip-rate estimation.
        Values should be in [0, 1] where lower = more stubborn = more weight.
    n_agents : int
        Number of agents.
    n_rounds : int
        Number of debate rounds.
    temperatures : tuple[float, ...]
        Per-agent sampling temperatures.
    epsilon : float
        Smoothing constant for 1/(revisability + epsilon) weighting.

    Returns
    -------
    ABCDebateResult
        Full result with per-round weighted voting data.
    """
    valid_labels = question.option_labels
    formatted_question = question.format_for_prompt()
    total_cost = 0.0

    # Ensure we have enough temperatures; cycle if needed.
    agent_temps = tuple(
        temperatures[i % len(temperatures)] for i in range(n_agents)
    )

    # ------------------------------------------------------------------
    # Phase 1: Independent initial answers (same as standard debate)
    # ------------------------------------------------------------------
    async def _get_initial_answer(agent_idx: int) -> AgentState:
        agent_id = str(agent_idx + 1)
        prompt = INITIAL_ANSWER_PROMPT.format(question=formatted_question)
        result = await client.generate(
            contents=prompt,
            temperature=agent_temps[agent_idx],
            metadata={
                "agent_id": agent_id,
                "phase": "initial",
                "question_id": question.id,
                "protocol": "abc",
            },
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

    # Compute initial weighted majority
    initial_round_summary = _compute_weighted_majority(
        initial_states, agent_revisabilities, epsilon
    )

    initial_majority = majority_vote([s.answer for s in initial_states])
    initial_correct = (initial_majority == question.correct_label)
    initial_weighted_majority = initial_round_summary.weighted_majority
    initial_weighted_correct = (initial_weighted_majority == question.correct_label)

    # ------------------------------------------------------------------
    # Phase 2: ABC debate rounds (with weighted consensus + shield)
    # ------------------------------------------------------------------
    round_states: list[list[AgentState]] = []
    round_summaries: list[RoundSummary] = [initial_round_summary]
    current_states = initial_states

    for round_num in range(1, n_rounds + 1):
        # Get the weighted summary from previous round
        prev_summary = round_summaries[-1]

        # Format responses with weighted consensus signal
        agent_responses_text = _format_agent_responses_with_weights(
            current_states, prev_summary
        )

        async def _abc_debate_round_agent(agent_idx: int) -> AgentState:
            agent_id = str(agent_idx + 1)
            prompt = ABC_DEBATE_ROUND_PROMPT.format(
                agent_responses=agent_responses_text,
                agent_num=agent_id,
                question=formatted_question,
                confidence_shield=CONFIDENCE_SHIELD,
            )
            result = await client.generate(
                contents=prompt,
                temperature=agent_temps[agent_idx],
                system_instruction=ABC_DEBATE_SYSTEM_PROMPT,
                metadata={
                    "agent_id": agent_id,
                    "phase": "debate",
                    "round": round_num,
                    "question_id": question.id,
                    "protocol": "abc",
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
            *[_abc_debate_round_agent(i) for i in range(n_agents)]
        ))
        total_cost += sum(s.cost for s in new_states)
        round_states.append(new_states)

        # Compute weighted majority for this round
        round_summary = _compute_weighted_majority(
            new_states, agent_revisabilities, epsilon
        )
        round_summary.round_num = round_num
        round_summaries.append(round_summary)

        current_states = new_states

    # ------------------------------------------------------------------
    # Phase 3: Final outcome
    # ------------------------------------------------------------------
    final_summary = round_summaries[-1]
    final_answer = final_summary.weighted_majority
    final_correct = (final_answer == question.correct_label)
    final_unweighted_answer = final_summary.unweighted_majority
    final_unweighted_correct = (final_unweighted_answer == question.correct_label)

    # Collapse/correction tracking (using weighted majority)
    collapsed = initial_weighted_correct and not final_correct
    corrected = not initial_weighted_correct and final_correct

    # Same for unweighted
    collapsed_unweighted = initial_correct and not final_unweighted_correct
    corrected_unweighted = not initial_correct and final_unweighted_correct

    return ABCDebateResult(
        question_id=question.id,
        benchmark=question.benchmark,
        correct_label=question.correct_label,
        initial_states=initial_states,
        round_states=round_states,
        round_summaries=round_summaries,
        final_answer=final_answer,
        final_correct=final_correct,
        final_unweighted_answer=final_unweighted_answer,
        final_unweighted_correct=final_unweighted_correct,
        initial_majority=initial_majority,
        initial_correct=initial_correct,
        initial_weighted_majority=initial_weighted_majority,
        initial_weighted_correct=initial_weighted_correct,
        collapsed=collapsed,
        corrected=corrected,
        collapsed_unweighted=collapsed_unweighted,
        corrected_unweighted=corrected_unweighted,
        agent_revisabilities=dict(agent_revisabilities),
        n_rounds=n_rounds,
        total_cost=total_cost,
        epsilon=epsilon,
    )
