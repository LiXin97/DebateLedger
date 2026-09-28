"""Block 4: Ablation Studies.

Tests four dimensions of the ABC protocol to understand which
design choices matter most:

  Ablation 1 -- Alpha estimation method (flip_rate, strength_weighted,
                bayesian, composite, mle)
  Ablation 2 -- Number of probes (2, 4, 6, 8 -- probe subset selection
                by informativeness)
  Ablation 3 -- Weighting function (inverse, exponential, quadratic,
                rank-based, uniform)
  Ablation 4 -- Social pressure impact (all 8 probes vs 4 non-social
                probes only)

Runs on 100 MMLU-Pro questions (seed=42 subset).
For efficiency, alpha is computed once with ALL 8 probes, then
subsampled for ablation 2.

Usage:
    python -m abc_exp.experiments.block4_ablations [--n_questions 100] [--budget 40.0]

Resume support: results are saved as JSONL; completed question IDs
are skipped on restart.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from abc_exp.src.api.client import GeminiClient
from abc_exp.src.data.loader import load_benchmark
from abc_exp.src.debate.abc_protocol import (
    ABCDebateResult,
    RoundSummary,
    WeightedVote,
    _compute_weighted_majority,
    run_abc_debate,
)
from abc_exp.src.debate.standard import AgentState, majority_vote
from abc_exp.src.alpha.estimation import (
    estimate_alpha_fliprate,
    estimate_alpha_strength_weighted,
    estimate_alpha_bayesian,
    estimate_alpha_mle,
    compute_revisability_score,
)
from abc_exp.src.alpha.probes import ProbeResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class Config:
    model: str = "models/gemini-2.5-flash"
    n_questions: int = 100
    n_agents: int = 3
    agent_temperatures: tuple[float, ...] = (0.5, 0.7, 1.0)
    n_debate_rounds: int = 3
    abc_epsilon: float = 0.05
    budget_cap: float = 40.0  # USD for this experiment
    max_concurrent: int = 20
    seed: int = 42
    results_dir: str = "abc_exp/results"
    log_dir: str = "abc_exp/results/logs"
    block0_results: str = "abc_exp/results/block0_alpha.jsonl"


# ===================================================================
# Result paths
# ===================================================================

def get_results_path(config: Config) -> Path:
    return Path(config.results_dir) / "block4_ablations.jsonl"


def get_summary_path(config: Config) -> Path:
    return Path(config.results_dir) / "block4_summary.json"


# ===================================================================
# Resume support
# ===================================================================

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


# ===================================================================
# Load Block 0 probe-level data
# ===================================================================

def load_block0_probe_data(
    path: str | Path,
) -> dict[str, dict[str, list[ProbeResult]]]:
    """Load raw probe results from Block 0.1 JSONL.

    Returns
    -------
    dict[str, dict[str, list[ProbeResult]]]
        question_id -> agent_id -> list of ProbeResult
    """
    path = Path(path)
    data: dict[str, dict[str, list[ProbeResult]]] = {}

    if not path.exists():
        logger.warning("Block 0 results not found at %s", path)
        return data

    with open(path, "r") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            qid = record["question_id"]
            probe_results_by_agent: dict[str, list[ProbeResult]] = {}

            for agent_key, probes_raw in record.get("probe_results", {}).items():
                probe_list: list[ProbeResult] = []
                for p in probes_raw:
                    probe_list.append(ProbeResult(
                        question_id=qid,
                        agent_id=agent_key,
                        probe_type=p["probe_type"],
                        social=p["social"],
                        strength=p["strength"],
                        original_answer=p["original_answer"],
                        probe_answer=p.get("probe_answer"),
                        revised=p["revised"],
                        cost=p.get("cost", 0.0),
                        raw_response="",  # not stored in JSONL
                    ))
                probe_results_by_agent[agent_key] = probe_list

            data[qid] = probe_results_by_agent

    logger.info("Loaded probe data for %d questions from %s", len(data), path)
    return data


def load_revisability_scores(path: str | Path) -> dict[str, dict[str, float]]:
    """Load per-question per-agent revisability scores from Block 0.1.

    Returns question_id -> {debate_agent_id -> flip_rate}.
    Agent IDs are mapped from 'agent_0' -> '1', etc.
    """
    path = Path(path)
    scores: dict[str, dict[str, float]] = {}

    if not path.exists():
        logger.warning("Block 0 results not found at %s", path)
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
                if agent_key.startswith("agent_"):
                    debate_id = str(int(agent_key.split("_")[1]) + 1)
                else:
                    debate_id = agent_key

                flip_rate = alpha_data.get(
                    "mean_flip_rate", alpha_data.get("alpha", 0.5)
                )
                agent_scores[debate_id] = flip_rate

            scores[qid] = agent_scores

    logger.info("Loaded revisability scores for %d questions from %s", len(scores), path)
    return scores


# ===================================================================
# Alpha estimation helpers (Ablation 1)
# ===================================================================

ESTIMATION_METHODS = ["flip_rate", "strength_weighted", "bayesian", "composite", "mle"]


def estimate_alpha_by_method(
    method: str,
    probe_results: list[ProbeResult],
    question_id: str,
    agent_id: str,
) -> float:
    """Run one of five estimation methods and return the alpha value."""
    if method == "flip_rate":
        est = estimate_alpha_fliprate(probe_results, question_id, agent_id)
        return est.alpha
    elif method == "strength_weighted":
        est = estimate_alpha_strength_weighted(probe_results, question_id, agent_id)
        return est.alpha
    elif method == "bayesian":
        est = estimate_alpha_bayesian(probe_results, question_id, agent_id)
        return est.alpha
    elif method == "composite":
        est = compute_revisability_score(probe_results, question_id, agent_id)
        return est.score
    elif method == "mle":
        est = estimate_alpha_mle(probe_results, question_id, agent_id)
        return est.mean_flip_rate  # use mean_flip_rate for comparability
    else:
        raise ValueError(f"Unknown estimation method: {method}")


# ===================================================================
# Probe subset selection (Ablation 2)
# ===================================================================

# Probe types in order of strength value
PROBE_TYPES = ["weak", "moderate", "strong", "very_strong"]
PROBE_STRENGTHS = {"weak": 0.25, "moderate": 0.50, "strong": 0.75, "very_strong": 1.00}

# Full set: 8 probes = 4 types x 2 social levels
# Each count selects the most informative probes by agreement rate
# (disagreement = informativeness for distinguishing agents).
PROBE_COUNTS = [2, 4, 6, 8]


def compute_probe_informativeness(
    all_probe_data: dict[str, dict[str, list[ProbeResult]]],
) -> list[tuple[str, bool]]:
    """Rank probes by informativeness (variance in flip rate across agents).

    Returns a list of (probe_type, social) tuples sorted from most to
    least informative.
    """
    # For each (probe_type, social) pair, compute flip rate across all
    # agents and questions, then measure variance.
    probe_flip_rates: dict[tuple[str, bool], list[float]] = defaultdict(list)

    for qid, agents in all_probe_data.items():
        for agent_id, probes in agents.items():
            for p in probes:
                key = (p.probe_type, p.social)
                probe_flip_rates[key].append(float(p.revised))

    # Informativeness = variance in flip rate (higher variance = more discriminating)
    probe_scores: list[tuple[tuple[str, bool], float]] = []
    for key, rates in probe_flip_rates.items():
        if len(rates) > 1:
            variance = float(np.var(rates))
        else:
            variance = 0.0
        probe_scores.append((key, variance))

    # Sort by variance descending (most informative first)
    probe_scores.sort(key=lambda x: x[1], reverse=True)

    return [key for key, _ in probe_scores]


def select_probe_subset(
    probes: list[ProbeResult],
    n_probes: int,
    ranked_probes: list[tuple[str, bool]],
) -> list[ProbeResult]:
    """Select the top n_probes most informative probes from a list."""
    if n_probes >= len(probes):
        return probes

    # Build a set of the top-n probe signatures
    top_signatures = set(ranked_probes[:n_probes])

    selected = [p for p in probes if (p.probe_type, p.social) in top_signatures]

    # If selection yielded fewer than expected (e.g., missing probes),
    # fall back to taking the first n_probes
    if len(selected) < n_probes:
        selected = probes[:n_probes]

    return selected


# ===================================================================
# Weighting functions (Ablation 3)
# ===================================================================

WeightingFn = Callable[[dict[str, float], float], dict[str, float]]

WEIGHTING_FUNCTIONS: dict[str, str] = {
    "inverse": "1/(r + epsilon)",
    "exponential": "exp(-r * beta), beta=2",
    "quadratic": "(1-r)^2",
    "rank_based": "rank-based weights",
    "uniform": "equal weight (= standard debate)",
}


def _weight_inverse(
    revisabilities: dict[str, float],
    epsilon: float,
) -> dict[str, float]:
    """Current default: w_i = 1 / (r_i + epsilon)."""
    raw = {aid: 1.0 / (r + epsilon) for aid, r in revisabilities.items()}
    total = sum(raw.values())
    return {aid: w / total for aid, w in raw.items()}


def _weight_exponential(
    revisabilities: dict[str, float],
    epsilon: float,
    beta: float = 2.0,
) -> dict[str, float]:
    """Exponential decay: w_i = exp(-r_i * beta)."""
    raw = {aid: float(np.exp(-r * beta)) for aid, r in revisabilities.items()}
    total = sum(raw.values())
    return {aid: w / total for aid, w in raw.items()}


def _weight_quadratic(
    revisabilities: dict[str, float],
    epsilon: float,
) -> dict[str, float]:
    """Quadratic: w_i = (1 - r_i)^2."""
    raw = {aid: (1.0 - r) ** 2 for aid, r in revisabilities.items()}
    total = sum(raw.values())
    if total == 0:
        n = len(revisabilities)
        return {aid: 1.0 / n for aid in revisabilities}
    return {aid: w / total for aid, w in raw.items()}


def _weight_rank_based(
    revisabilities: dict[str, float],
    epsilon: float,
) -> dict[str, float]:
    """Rank-based: sort agents by revisability, assign weights by rank.

    Lowest revisability gets highest rank weight.
    For 3 agents sorted by revisability: weights = [3, 2, 1] (normalized).
    """
    sorted_agents = sorted(revisabilities.items(), key=lambda x: x[1])
    n = len(sorted_agents)
    raw: dict[str, float] = {}
    for rank_idx, (aid, _) in enumerate(sorted_agents):
        raw[aid] = float(n - rank_idx)  # lowest revisability -> highest rank
    total = sum(raw.values())
    return {aid: w / total for aid, w in raw.items()}


def _weight_uniform(
    revisabilities: dict[str, float],
    epsilon: float,
) -> dict[str, float]:
    """Uniform: all agents get equal weight (= standard debate)."""
    n = len(revisabilities)
    return {aid: 1.0 / n for aid in revisabilities}


WEIGHTING_FN_MAP: dict[str, WeightingFn] = {
    "inverse": _weight_inverse,
    "exponential": _weight_exponential,
    "quadratic": _weight_quadratic,
    "rank_based": _weight_rank_based,
    "uniform": _weight_uniform,
}


def apply_custom_weighting(
    states: list[AgentState],
    revisabilities: dict[str, float],
    epsilon: float,
    weighting_fn: WeightingFn,
) -> RoundSummary:
    """Compute weighted majority using a custom weighting function.

    Mirrors _compute_weighted_majority but lets us plug in different
    revisability-to-weight mappings.
    """
    norm_weights = weighting_fn(revisabilities, epsilon)

    # Build weighted votes
    weighted_votes = []
    for s in states:
        rev = revisabilities.get(s.agent_id, 0.5)
        weighted_votes.append(WeightedVote(
            agent_id=s.agent_id,
            answer=s.answer,
            raw_weight=norm_weights.get(s.agent_id, 1.0 / len(states)),
            normalized_weight=norm_weights.get(s.agent_id, 1.0 / len(states)),
            revisability=rev,
        ))

    # Aggregate weights per answer
    answer_weights: dict[str, float] = defaultdict(float)
    for v in weighted_votes:
        if v.answer is not None:
            answer_weights[v.answer] += v.normalized_weight

    # Weighted majority
    if answer_weights:
        max_weight = max(answer_weights.values())
        tied = sorted(
            a for a, w in answer_weights.items() if abs(w - max_weight) < 1e-10
        )
        weighted_majority = tied[0]
        consensus_strength = max_weight
    else:
        weighted_majority = ""
        consensus_strength = 0.0

    # Unweighted majority
    unweighted_majority = majority_vote([s.answer for s in states])

    return RoundSummary(
        round_num=states[0].round_num if states else 0,
        weighted_votes=weighted_votes,
        weighted_majority=weighted_majority,
        unweighted_majority=unweighted_majority,
        answer_weights=dict(answer_weights),
        consensus_strength=consensus_strength,
    )


# ===================================================================
# Per-question ablation processing
# ===================================================================

async def process_question_ablations(
    client: GeminiClient,
    question,
    probe_data: dict[str, list[ProbeResult]] | None,
    default_revisabilities: dict[str, float],
    ranked_probes: list[tuple[str, bool]],
    config: Config,
) -> dict:
    """Run all four ablation dimensions for a single question.

    The ABC debate is run once per ablation variant that requires it.
    For ablation 3 (weighting) and ablation 1 (estimation method), we
    can reuse debate transcripts and just re-weight, saving API cost.

    Strategy:
    - Run ONE ABC debate using default (flip_rate + inverse) config.
    - Reuse the debate transcript (agent states per round) to evaluate
      alternative estimation methods (ablation 1), alternative weightings
      (ablation 3), and social pressure impact (ablation 4).
    - For ablation 2 (probe count), re-estimate alpha from probe subsets
      and re-weight the same debate transcript.
    """
    qid = question.id
    record: dict = {
        "question_id": qid,
        "benchmark": question.benchmark,
        "correct_label": question.correct_label,
        "category": question.metadata.get("category", "unknown"),
    }

    total_cost = 0.0

    # -----------------------------------------------------------------
    # Step 0: Compute revisabilities under ALL estimation methods
    # -----------------------------------------------------------------
    all_revisabilities: dict[str, dict[str, float]] = {}

    if probe_data:
        for method in ESTIMATION_METHODS:
            agent_revs: dict[str, float] = {}
            for agent_key, probes in probe_data.items():
                # Map agent_0 -> 1 for debate agent IDs
                if agent_key.startswith("agent_"):
                    debate_id = str(int(agent_key.split("_")[1]) + 1)
                else:
                    debate_id = agent_key
                agent_revs[debate_id] = estimate_alpha_by_method(
                    method, probes, qid, agent_key,
                )
            all_revisabilities[method] = agent_revs
    else:
        for method in ESTIMATION_METHODS:
            all_revisabilities[method] = dict(default_revisabilities)

    record["all_revisabilities"] = all_revisabilities

    # -----------------------------------------------------------------
    # Step 1: Run a single ABC debate (using flip_rate + inverse weight)
    # -----------------------------------------------------------------
    base_revisabilities = all_revisabilities.get("flip_rate", default_revisabilities)

    try:
        abc_result = await run_abc_debate(
            client=client,
            question=question,
            agent_revisabilities=base_revisabilities,
            n_agents=config.n_agents,
            n_rounds=config.n_debate_rounds,
            temperatures=config.agent_temperatures,
            epsilon=config.abc_epsilon,
        )
        total_cost += abc_result.total_cost

        # Extract agent states per round for re-weighting
        all_round_states = [abc_result.initial_states] + abc_result.round_states

    except Exception as e:
        logger.error("ABC debate failed for %s: %s", qid, e)
        record["error"] = str(e)
        record["total_cost"] = 0.0
        return record

    # -----------------------------------------------------------------
    # Ablation 1: Alpha estimation method
    # -----------------------------------------------------------------
    ablation1_results: dict[str, dict] = {}

    for method in ESTIMATION_METHODS:
        method_revs = all_revisabilities[method]

        # Re-weight the debate transcript with this method's revisabilities
        final_states = all_round_states[-1]
        summary = _compute_weighted_majority(
            final_states, method_revs, config.abc_epsilon,
        )
        final_answer = summary.weighted_majority
        final_correct = final_answer == question.correct_label

        # Initial weighted majority
        initial_summary = _compute_weighted_majority(
            all_round_states[0], method_revs, config.abc_epsilon,
        )

        ablation1_results[method] = {
            "revisabilities": method_revs,
            "final_answer": final_answer,
            "final_correct": final_correct,
            "initial_weighted_answer": initial_summary.weighted_majority,
            "initial_weighted_correct": (
                initial_summary.weighted_majority == question.correct_label
            ),
            "consensus_strength": summary.consensus_strength,
        }

    record["ablation1_estimation_method"] = ablation1_results

    # -----------------------------------------------------------------
    # Ablation 2: Number of probes
    # -----------------------------------------------------------------
    ablation2_results: dict[str, dict] = {}

    for n_probes in PROBE_COUNTS:
        label = f"{n_probes}_probes"

        if probe_data:
            # Subsample probes for each agent
            agent_revs_subset: dict[str, float] = {}
            for agent_key, probes in probe_data.items():
                if agent_key.startswith("agent_"):
                    debate_id = str(int(agent_key.split("_")[1]) + 1)
                else:
                    debate_id = agent_key

                subset = select_probe_subset(probes, n_probes, ranked_probes)
                est = estimate_alpha_fliprate(subset, qid, agent_key)
                agent_revs_subset[debate_id] = est.alpha

            # Re-weight debate transcript
            final_states = all_round_states[-1]
            summary = _compute_weighted_majority(
                final_states, agent_revs_subset, config.abc_epsilon,
            )
            final_answer = summary.weighted_majority
            final_correct = final_answer == question.correct_label

            ablation2_results[label] = {
                "n_probes": n_probes,
                "revisabilities": agent_revs_subset,
                "final_answer": final_answer,
                "final_correct": final_correct,
                "consensus_strength": summary.consensus_strength,
            }
        else:
            # No probe data: all probe counts give same result
            ablation2_results[label] = {
                "n_probes": n_probes,
                "revisabilities": dict(default_revisabilities),
                "final_answer": abc_result.final_answer,
                "final_correct": abc_result.final_correct,
                "consensus_strength": 0.0,
                "note": "no probe data available",
            }

    record["ablation2_probe_count"] = ablation2_results

    # -----------------------------------------------------------------
    # Ablation 3: Weighting function
    # -----------------------------------------------------------------
    ablation3_results: dict[str, dict] = {}

    for wf_name, wf_fn in WEIGHTING_FN_MAP.items():
        final_states = all_round_states[-1]
        summary = apply_custom_weighting(
            final_states, base_revisabilities, config.abc_epsilon, wf_fn,
        )
        final_answer = summary.weighted_majority
        final_correct = final_answer == question.correct_label

        # Per-round tracking for deeper analysis
        round_answers = []
        for round_states in all_round_states:
            rs = apply_custom_weighting(
                round_states, base_revisabilities, config.abc_epsilon, wf_fn,
            )
            round_answers.append(rs.weighted_majority)

        ablation3_results[wf_name] = {
            "weighting": WEIGHTING_FUNCTIONS[wf_name],
            "final_answer": final_answer,
            "final_correct": final_correct,
            "consensus_strength": summary.consensus_strength,
            "weights": {
                v.agent_id: v.normalized_weight
                for v in summary.weighted_votes
            },
            "round_answers": round_answers,
        }

    record["ablation3_weighting_function"] = ablation3_results

    # -----------------------------------------------------------------
    # Ablation 4: Social pressure impact
    # -----------------------------------------------------------------
    ablation4_results: dict = {}

    if probe_data:
        # Compute flip rates WITH and WITHOUT social probes per agent
        social_stats: dict[str, dict] = {}

        for agent_key, probes in probe_data.items():
            if agent_key.startswith("agent_"):
                debate_id = str(int(agent_key.split("_")[1]) + 1)
            else:
                debate_id = agent_key

            social_probes = [p for p in probes if p.social]
            nonsocial_probes = [p for p in probes if not p.social]

            flip_all = sum(1 for p in probes if p.revised) / len(probes) if probes else 0.0
            flip_social = (
                sum(1 for p in social_probes if p.revised) / len(social_probes)
                if social_probes else 0.0
            )
            flip_nonsocial = (
                sum(1 for p in nonsocial_probes if p.revised) / len(nonsocial_probes)
                if nonsocial_probes else 0.0
            )
            social_sensitivity = flip_social - flip_nonsocial

            social_stats[debate_id] = {
                "flip_rate_all": flip_all,
                "flip_rate_social": flip_social,
                "flip_rate_nonsocial": flip_nonsocial,
                "social_sensitivity": social_sensitivity,
                "n_social": len(social_probes),
                "n_nonsocial": len(nonsocial_probes),
            }

        ablation4_results["per_agent_stats"] = social_stats

        # Variant A: revisability from ALL 8 probes
        revs_all: dict[str, float] = {}
        for agent_key, probes in probe_data.items():
            if agent_key.startswith("agent_"):
                debate_id = str(int(agent_key.split("_")[1]) + 1)
            else:
                debate_id = agent_key
            est = estimate_alpha_fliprate(probes, qid, agent_key)
            revs_all[debate_id] = est.alpha

        # Variant B: revisability from 4 non-social probes only
        revs_nonsocial: dict[str, float] = {}
        for agent_key, probes in probe_data.items():
            if agent_key.startswith("agent_"):
                debate_id = str(int(agent_key.split("_")[1]) + 1)
            else:
                debate_id = agent_key
            nonsocial_probes = [p for p in probes if not p.social]
            if nonsocial_probes:
                est = estimate_alpha_fliprate(nonsocial_probes, qid, agent_key)
                revs_nonsocial[debate_id] = est.alpha
            else:
                revs_nonsocial[debate_id] = 0.5

        # Re-weight debate transcript with each variant
        final_states = all_round_states[-1]

        summary_all = _compute_weighted_majority(
            final_states, revs_all, config.abc_epsilon,
        )
        summary_nonsocial = _compute_weighted_majority(
            final_states, revs_nonsocial, config.abc_epsilon,
        )

        ablation4_results["all_probes"] = {
            "revisabilities": revs_all,
            "final_answer": summary_all.weighted_majority,
            "final_correct": summary_all.weighted_majority == question.correct_label,
            "consensus_strength": summary_all.consensus_strength,
        }
        ablation4_results["nonsocial_only"] = {
            "revisabilities": revs_nonsocial,
            "final_answer": summary_nonsocial.weighted_majority,
            "final_correct": (
                summary_nonsocial.weighted_majority == question.correct_label
            ),
            "consensus_strength": summary_nonsocial.consensus_strength,
        }
    else:
        ablation4_results["note"] = "no probe data available"

    record["ablation4_social_pressure"] = ablation4_results

    record["total_cost"] = total_cost
    return record


# ===================================================================
# Aggregate statistics
# ===================================================================

def compute_aggregate_stats(results_path: Path) -> dict:
    """Compute per-ablation aggregate statistics from all results."""
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

    stats: dict = {"n_questions": n}

    # --- Ablation 1: estimation method ---
    abl1: dict[str, dict] = {}
    for method in ESTIMATION_METHODS:
        correct_list = []
        for rec in records:
            a1 = rec.get("ablation1_estimation_method", {})
            mdata = a1.get(method, {})
            if "final_correct" in mdata:
                correct_list.append(mdata["final_correct"])
        if correct_list:
            abl1[method] = {
                "accuracy": sum(correct_list) / len(correct_list),
                "n": len(correct_list),
                "n_correct": sum(correct_list),
            }
    stats["ablation1_estimation_method"] = abl1

    # --- Ablation 2: probe count ---
    abl2: dict[str, dict] = {}
    for n_probes in PROBE_COUNTS:
        label = f"{n_probes}_probes"
        correct_list = []
        for rec in records:
            a2 = rec.get("ablation2_probe_count", {})
            mdata = a2.get(label, {})
            if "final_correct" in mdata:
                correct_list.append(mdata["final_correct"])
        if correct_list:
            abl2[label] = {
                "accuracy": sum(correct_list) / len(correct_list),
                "n": len(correct_list),
                "n_correct": sum(correct_list),
            }
    stats["ablation2_probe_count"] = abl2

    # --- Ablation 3: weighting function ---
    abl3: dict[str, dict] = {}
    for wf_name in WEIGHTING_FN_MAP:
        correct_list = []
        for rec in records:
            a3 = rec.get("ablation3_weighting_function", {})
            mdata = a3.get(wf_name, {})
            if "final_correct" in mdata:
                correct_list.append(mdata["final_correct"])
        if correct_list:
            abl3[wf_name] = {
                "accuracy": sum(correct_list) / len(correct_list),
                "n": len(correct_list),
                "n_correct": sum(correct_list),
            }
    stats["ablation3_weighting_function"] = abl3

    # --- Ablation 4: social pressure ---
    abl4: dict = {}
    correct_all = []
    correct_nonsocial = []
    sensitivity_vals = []

    for rec in records:
        a4 = rec.get("ablation4_social_pressure", {})
        if "note" in a4:
            continue

        ap_data = a4.get("all_probes", {})
        ns_data = a4.get("nonsocial_only", {})

        if "final_correct" in ap_data:
            correct_all.append(ap_data["final_correct"])
        if "final_correct" in ns_data:
            correct_nonsocial.append(ns_data["final_correct"])

        # Collect per-agent social sensitivity
        per_agent = a4.get("per_agent_stats", {})
        for agent_id, astats in per_agent.items():
            sensitivity_vals.append(astats.get("social_sensitivity", 0.0))

    if correct_all:
        abl4["all_probes"] = {
            "accuracy": sum(correct_all) / len(correct_all),
            "n": len(correct_all),
            "n_correct": sum(correct_all),
        }
    if correct_nonsocial:
        abl4["nonsocial_only"] = {
            "accuracy": sum(correct_nonsocial) / len(correct_nonsocial),
            "n": len(correct_nonsocial),
            "n_correct": sum(correct_nonsocial),
        }
    if sensitivity_vals:
        abl4["social_sensitivity"] = {
            "mean": float(np.mean(sensitivity_vals)),
            "std": float(np.std(sensitivity_vals)),
            "min": float(np.min(sensitivity_vals)),
            "max": float(np.max(sensitivity_vals)),
            "n_agents": len(sensitivity_vals),
        }

    # Agreement between all-probes and nonsocial-only
    if correct_all and correct_nonsocial:
        n_agree = sum(
            1 for a, b in zip(correct_all, correct_nonsocial) if a == b
        )
        abl4["agreement_rate"] = n_agree / len(correct_all)

    stats["ablation4_social_pressure"] = abl4

    return stats


# ===================================================================
# Summary printing
# ===================================================================

def _print_summary(stats: dict, total_completed: int):
    """Print formatted comparison tables per ablation."""
    print("\n" + "=" * 70)
    print("BLOCK 4: ABLATION STUDIES SUMMARY")
    print("=" * 70)
    print(f"Questions completed: {total_completed}")

    # --- Ablation 1 ---
    print("\n--- Ablation 1: Alpha Estimation Method ---")
    abl1 = stats.get("ablation1_estimation_method", {})
    if abl1:
        print(f"{'Method':<22} {'Accuracy':>10} {'N':>6} {'Correct':>8}")
        print("-" * 50)
        for method in ESTIMATION_METHODS:
            data = abl1.get(method, {})
            if data:
                print(
                    f"{method:<22} {data['accuracy']:>9.1%} "
                    f"{data['n']:>6} {data['n_correct']:>8}"
                )
        # Highlight best
        if abl1:
            best = max(abl1.items(), key=lambda x: x[1].get("accuracy", 0))
            print(f"  --> Best: {best[0]} ({best[1]['accuracy']:.1%})")
    else:
        print("  (no data)")

    # --- Ablation 2 ---
    print("\n--- Ablation 2: Number of Probes ---")
    abl2 = stats.get("ablation2_probe_count", {})
    if abl2:
        print(f"{'Probes':<22} {'Accuracy':>10} {'N':>6} {'Correct':>8}")
        print("-" * 50)
        for n_probes in PROBE_COUNTS:
            label = f"{n_probes}_probes"
            data = abl2.get(label, {})
            if data:
                print(
                    f"{label:<22} {data['accuracy']:>9.1%} "
                    f"{data['n']:>6} {data['n_correct']:>8}"
                )
        # Check if fewer probes match full performance
        if "8_probes" in abl2 and "4_probes" in abl2:
            full_acc = abl2["8_probes"]["accuracy"]
            half_acc = abl2["4_probes"]["accuracy"]
            diff = full_acc - half_acc
            print(f"  --> 4 vs 8 probes accuracy gap: {diff:+.1%}")
    else:
        print("  (no data)")

    # --- Ablation 3 ---
    print("\n--- Ablation 3: Weighting Function ---")
    abl3 = stats.get("ablation3_weighting_function", {})
    if abl3:
        print(f"{'Weighting':<22} {'Accuracy':>10} {'N':>6} {'Correct':>8}")
        print("-" * 50)
        for wf_name in WEIGHTING_FN_MAP:
            data = abl3.get(wf_name, {})
            if data:
                print(
                    f"{wf_name:<22} {data['accuracy']:>9.1%} "
                    f"{data['n']:>6} {data['n_correct']:>8}"
                )
        if abl3:
            best = max(abl3.items(), key=lambda x: x[1].get("accuracy", 0))
            print(f"  --> Best: {best[0]} ({best[1]['accuracy']:.1%})")
        # Compare best vs uniform
        if "uniform" in abl3 and abl3:
            uni_acc = abl3["uniform"]["accuracy"]
            best_acc = best[1]["accuracy"]
            print(f"  --> Gain over uniform: {best_acc - uni_acc:+.1%}")
    else:
        print("  (no data)")

    # --- Ablation 4 ---
    print("\n--- Ablation 4: Social Pressure Impact ---")
    abl4 = stats.get("ablation4_social_pressure", {})
    if abl4:
        ap = abl4.get("all_probes", {})
        ns = abl4.get("nonsocial_only", {})
        if ap and ns:
            print(f"{'Probe Set':<22} {'Accuracy':>10} {'N':>6}")
            print("-" * 42)
            print(f"{'All 8 probes':<22} {ap['accuracy']:>9.1%} {ap['n']:>6}")
            print(f"{'4 non-social only':<22} {ns['accuracy']:>9.1%} {ns['n']:>6}")
            diff = ap["accuracy"] - ns["accuracy"]
            print(f"  --> Accuracy delta (all vs non-social): {diff:+.1%}")

        agree = abl4.get("agreement_rate")
        if agree is not None:
            print(f"  --> Agreement rate: {agree:.1%}")

        ss = abl4.get("social_sensitivity", {})
        if ss:
            print(
                f"  --> Social sensitivity: "
                f"mean={ss['mean']:.3f}, std={ss['std']:.3f}, "
                f"range=[{ss['min']:.3f}, {ss['max']:.3f}]"
            )
    else:
        print("  (no data)")

    print("\n" + "=" * 70)


# ===================================================================
# Main
# ===================================================================

async def main(config: Config):
    """Run the Block 4 ablation experiments."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = get_results_path(config)
    summary_path = get_summary_path(config)
    completed = load_completed_ids(results_path)

    if completed:
        logger.info("Resuming: %d questions already completed", len(completed))

    # Load Block 0 data
    logger.info("Loading Block 0 probe data...")
    all_probe_data = load_block0_probe_data(config.block0_results)
    all_revisabilities = load_revisability_scores(config.block0_results)

    # Compute probe informativeness ranking (for ablation 2)
    if all_probe_data:
        ranked_probes = compute_probe_informativeness(all_probe_data)
        logger.info(
            "Probe informativeness ranking: %s",
            [(pt, s) for pt, s in ranked_probes],
        )
    else:
        # Default ordering: strong probes first, non-social preferred
        ranked_probes = [
            ("very_strong", False), ("very_strong", True),
            ("strong", False), ("strong", True),
            ("moderate", False), ("moderate", True),
            ("weak", False), ("weak", True),
        ]

    default_revisabilities = {str(i + 1): 0.5 for i in range(config.n_agents)}

    # Load questions (100 from same seed=42 distribution)
    logger.info("Loading MMLU-Pro dataset...")
    questions = load_benchmark(
        "mmlu_pro", n_samples=config.n_questions, seed=config.seed,
    )
    logger.info("Loaded %d questions", len(questions))

    # Filter out completed
    remaining = [q for q in questions if q.id not in completed]
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

    total_cost = 0.0
    n_processed = 0

    try:
        for i, question in enumerate(remaining):
            # Get probe data for this question
            probe_data = all_probe_data.get(question.id)
            revs = all_revisabilities.get(question.id, default_revisabilities)

            logger.info(
                "[%d/%d] Processing %s (cat: %s) | Cost: $%.4f",
                i + 1, len(remaining), question.id,
                question.metadata.get("category", "?"),
                total_cost,
            )

            record = await process_question_ablations(
                client=client,
                question=question,
                probe_data=probe_data,
                default_revisabilities=default_revisabilities,
                ranked_probes=ranked_probes,
                config=config,
            )

            # Append to JSONL
            with open(results_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_cost += record.get("total_cost", 0.0)
            n_processed += 1

            # Print per-question mini-summary
            a1 = record.get("ablation1_estimation_method", {})
            a1_summary = {m: d.get("final_correct", "?") for m, d in a1.items()}
            logger.info("  Abl1 (est. method): %s", a1_summary)

            a3 = record.get("ablation3_weighting_function", {})
            a3_summary = {w: d.get("final_correct", "?") for w, d in a3.items()}
            logger.info("  Abl3 (weighting):   %s", a3_summary)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Block 4: Ablation Studies"
    )
    parser.add_argument("--n_questions", type=int, default=100)
    parser.add_argument("--budget", type=float, default=40.0)
    parser.add_argument("--max_concurrent", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="models/gemini-2.5-flash")
    parser.add_argument("--n_rounds", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--block0_results", type=str,
                        default="abc_exp/results/block0_alpha.jsonl",
                        help="Path to Block 0.1 results (with probe data)")
    args = parser.parse_args()

    config = Config(
        model=args.model,
        n_questions=args.n_questions,
        n_debate_rounds=args.n_rounds,
        abc_epsilon=args.epsilon,
        budget_cap=args.budget,
        max_concurrent=args.max_concurrent,
        seed=args.seed,
        block0_results=args.block0_results,
    )

    asyncio.run(main(config))
