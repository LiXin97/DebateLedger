"""Alpha estimation: multiple methods for measuring belief revisability.

Methods:
    1. estimate_alpha_mle        — MLE logistic model (original, ICC=0.485)
    2. estimate_alpha_fliprate   — Raw flip rate (ICC=0.760, reliable baseline)
    3. estimate_alpha_strength_weighted — Weighted flip rate (weight by 1-strength)
    4. estimate_alpha_bayesian   — Beta-Binomial posterior mean
    5. compute_revisability_score — Composite score (flip_rate + strength_sensitivity + social_sensitivity)

Legacy wrapper:
    estimate_alpha              — Calls estimate_alpha_mle for backward compat
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .probes import ProbeResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AlphaEstimate:
    """Result of alpha estimation for one agent on one question."""
    question_id: str
    agent_id: str
    alpha: float              # MLE estimate
    kappa: float              # MLE confidence offset
    ci_lower: float           # 95% CI lower bound
    ci_upper: float           # 95% CI upper bound
    mean_flip_rate: float     # simple mean of binary revision indicators
    n_probes: int             # number of probes used
    n_revised: int            # number of probes where agent revised
    converged: bool           # whether MLE converged

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "agent_id": self.agent_id,
            "alpha": self.alpha,
            "kappa": self.kappa,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "mean_flip_rate": self.mean_flip_rate,
            "n_probes": self.n_probes,
            "n_revised": self.n_revised,
            "converged": self.converged,
        }


@dataclass
class FlipRateEstimate:
    """Result of flip-rate alpha estimation."""
    question_id: str
    agent_id: str
    alpha: float              # raw flip rate in [0, 1]
    n_probes: int
    n_revised: int
    method: str = "fliprate"

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "agent_id": self.agent_id,
            "alpha": self.alpha,
            "n_probes": self.n_probes,
            "n_revised": self.n_revised,
            "method": self.method,
        }


@dataclass
class StrengthWeightedEstimate:
    """Result of strength-weighted flip-rate estimation."""
    question_id: str
    agent_id: str
    alpha: float              # weighted flip rate
    raw_flip_rate: float      # unweighted for comparison
    n_probes: int
    n_revised: int
    method: str = "strength_weighted"

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "agent_id": self.agent_id,
            "alpha": self.alpha,
            "raw_flip_rate": self.raw_flip_rate,
            "n_probes": self.n_probes,
            "n_revised": self.n_revised,
            "method": self.method,
        }


@dataclass
class BayesianEstimate:
    """Result of Bayesian (Beta-Binomial) alpha estimation."""
    question_id: str
    agent_id: str
    alpha: float              # posterior mean = (flips + 1) / (total + 2)
    posterior_a: float         # Beta posterior parameter a = flips + 1
    posterior_b: float         # Beta posterior parameter b = (total - flips) + 1
    ci_lower: float           # 95% credible interval lower (Beta quantile)
    ci_upper: float           # 95% credible interval upper (Beta quantile)
    n_probes: int
    n_revised: int
    method: str = "bayesian"

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "agent_id": self.agent_id,
            "alpha": self.alpha,
            "posterior_a": self.posterior_a,
            "posterior_b": self.posterior_b,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_probes": self.n_probes,
            "n_revised": self.n_revised,
            "method": self.method,
        }


@dataclass
class RevisabilityScore:
    """Composite revisability score combining three dimensions."""
    question_id: str
    agent_id: str
    score: float              # composite score in [0, 1]
    flip_rate: float          # component 1: raw flip rate
    strength_sensitivity: float  # component 2: corr(strength, flip)
    social_sensitivity: float    # component 3: flip_rate_social - flip_rate_nosocial
    n_probes: int
    n_social: int             # probes with social pressure
    n_nonsocial: int          # probes without social pressure
    method: str = "composite"

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "agent_id": self.agent_id,
            "score": self.score,
            "flip_rate": self.flip_rate,
            "strength_sensitivity": self.strength_sensitivity,
            "social_sensitivity": self.social_sensitivity,
            "n_probes": self.n_probes,
            "n_social": self.n_social,
            "n_nonsocial": self.n_nonsocial,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_ids(probe_results: list[ProbeResult],
                 question_id: str | None = None,
                 agent_id: str | None = None) -> tuple[str, str]:
    """Extract question_id and agent_id from probe results or overrides."""
    qid = question_id or (probe_results[0].question_id if probe_results else "unknown")
    aid = agent_id or (probe_results[0].agent_id if probe_results else "unknown")
    return qid, aid


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _neg_log_likelihood(
    params: np.ndarray,
    strengths: np.ndarray,
    revisions: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Negative log-likelihood of the logistic revision model.

    params: [alpha, kappa]
    strengths: array of probe strength values
    revisions: array of binary revision indicators (0/1)
    """
    alpha, kappa = params
    z = alpha * strengths - kappa
    p = _sigmoid(z)
    p = np.clip(p, eps, 1.0 - eps)
    ll = np.sum(revisions * np.log(p) + (1 - revisions) * np.log(1 - p))
    return -ll


def _bootstrap_ci(
    strengths: np.ndarray,
    revisions: np.ndarray,
    n_bootstrap: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for MLE alpha."""
    rng = np.random.default_rng(seed)
    n = len(strengths)
    alphas = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s_boot = strengths[idx]
        r_boot = revisions[idx]

        # Skip degenerate samples
        if r_boot.sum() == 0 or r_boot.sum() == n:
            continue

        result = minimize(
            _neg_log_likelihood,
            x0=np.array([1.0, 0.0]),
            args=(s_boot, r_boot),
            method="L-BFGS-B",
            bounds=[(0.01, 10.0), (-5.0, 5.0)],
        )
        if result.success:
            alphas.append(result.x[0])

    if len(alphas) < 10:
        # Not enough successful bootstraps
        return 0.01, 10.0

    lower = float(np.percentile(alphas, (1 - ci) / 2 * 100))
    upper = float(np.percentile(alphas, (1 + ci) / 2 * 100))
    return lower, upper


# ---------------------------------------------------------------------------
# Method 1: MLE logistic (original, kept for comparison)
# ---------------------------------------------------------------------------

def estimate_alpha_mle(
    probe_results: list[ProbeResult],
    question_id: str | None = None,
    agent_id: str | None = None,
) -> AlphaEstimate:
    """Estimate alpha via MLE under the logistic revision model.

    Model: P(revise | alpha, s, kappa) = sigmoid(alpha * s - kappa)

    NOTE: ICC = 0.485 in Block 0 -- unreliable. The logistic model conflates
    "truly stubborn" and "randomly noisy" agents (both get alpha near 0).
    Kept for comparison; prefer estimate_alpha_fliprate or estimate_alpha_bayesian.
    """
    strengths = np.array([p.strength for p in probe_results])
    revisions = np.array([float(p.revised) for p in probe_results])

    n_probes = len(probe_results)
    n_revised = int(revisions.sum())
    mean_flip = float(revisions.mean()) if n_probes > 0 else 0.0

    qid, aid = _extract_ids(probe_results, question_id, agent_id)

    # Edge cases: all revised or none revised
    if n_revised == 0:
        return AlphaEstimate(
            question_id=qid, agent_id=aid,
            alpha=0.01, kappa=5.0,
            ci_lower=0.01, ci_upper=0.5,
            mean_flip_rate=0.0, n_probes=n_probes,
            n_revised=0, converged=True,
        )
    if n_revised == n_probes:
        return AlphaEstimate(
            question_id=qid, agent_id=aid,
            alpha=10.0, kappa=-5.0,
            ci_lower=5.0, ci_upper=10.0,
            mean_flip_rate=1.0, n_probes=n_probes,
            n_revised=n_probes, converged=True,
        )

    # MLE via L-BFGS-B
    result = minimize(
        _neg_log_likelihood,
        x0=np.array([1.0, 0.0]),
        args=(strengths, revisions),
        method="L-BFGS-B",
        bounds=[(0.01, 10.0), (-5.0, 5.0)],
    )

    converged = result.success
    if converged:
        alpha_hat, kappa_hat = result.x
    else:
        # Fallback: use mean flip rate as alpha proxy
        alpha_hat = mean_flip * 4.0  # rough scaling
        kappa_hat = 0.0

    # Bootstrap CI
    ci_lower, ci_upper = _bootstrap_ci(strengths, revisions, n_bootstrap=200)

    return AlphaEstimate(
        question_id=qid, agent_id=aid,
        alpha=float(alpha_hat), kappa=float(kappa_hat),
        ci_lower=float(ci_lower), ci_upper=float(ci_upper),
        mean_flip_rate=mean_flip, n_probes=n_probes,
        n_revised=n_revised, converged=converged,
    )


# ---------------------------------------------------------------------------
# Method 2: Raw flip rate (ICC = 0.760, reliable baseline)
# ---------------------------------------------------------------------------

def estimate_alpha_fliprate(
    probe_results: list[ProbeResult],
    question_id: str | None = None,
    agent_id: str | None = None,
) -> FlipRateEstimate:
    """Estimate alpha as raw flip rate: n_revised / n_probes.

    ICC = 0.760 in Block 0 -- the most reliable single measure.
    Simple, interpretable, and does not conflate stubborn with noisy.
    """
    qid, aid = _extract_ids(probe_results, question_id, agent_id)

    n_probes = len(probe_results)
    n_revised = sum(1 for p in probe_results if p.revised)
    alpha = n_revised / n_probes if n_probes > 0 else 0.0

    return FlipRateEstimate(
        question_id=qid,
        agent_id=aid,
        alpha=alpha,
        n_probes=n_probes,
        n_revised=n_revised,
    )


# ---------------------------------------------------------------------------
# Method 3: Strength-weighted flip rate
# ---------------------------------------------------------------------------

def estimate_alpha_strength_weighted(
    probe_results: list[ProbeResult],
    question_id: str | None = None,
    agent_id: str | None = None,
) -> StrengthWeightedEstimate:
    """Weighted flip rate: weight each flip by (1 - probe_strength).

    Flipping on a WEAK probe (strength=0.25) gets weight 0.75.
    Flipping on a STRONG probe (strength=1.0) gets weight 0.0.

    This discriminates between agents that flip on anything (high alpha)
    vs agents that only flip when the probe is genuinely strong (lower alpha).

    An agent that only revises on strong probes is more calibrated, so it
    should have a lower revisability score than one that revises on weak probes.
    """
    qid, aid = _extract_ids(probe_results, question_id, agent_id)

    n_probes = len(probe_results)
    n_revised = sum(1 for p in probe_results if p.revised)
    raw_flip_rate = n_revised / n_probes if n_probes > 0 else 0.0

    if n_probes == 0:
        return StrengthWeightedEstimate(
            question_id=qid, agent_id=aid,
            alpha=0.0, raw_flip_rate=0.0,
            n_probes=0, n_revised=0,
        )

    # Weight each probe: flip contributes (1 - strength), non-flip contributes 0
    weights = np.array([1.0 - p.strength for p in probe_results])
    flips = np.array([float(p.revised) for p in probe_results])

    # Normalize: weighted sum of flips / sum of all possible weights
    # This keeps alpha in [0, 1] range
    total_weight = weights.sum()
    if total_weight > 0:
        alpha = float((flips * weights).sum() / total_weight)
    else:
        # All probes have strength=1.0, so weights are all 0
        # Fall back to raw flip rate
        alpha = raw_flip_rate

    return StrengthWeightedEstimate(
        question_id=qid, agent_id=aid,
        alpha=alpha, raw_flip_rate=raw_flip_rate,
        n_probes=n_probes, n_revised=n_revised,
    )


# ---------------------------------------------------------------------------
# Method 4: Bayesian Beta-Binomial
# ---------------------------------------------------------------------------

def estimate_alpha_bayesian(
    probe_results: list[ProbeResult],
    question_id: str | None = None,
    agent_id: str | None = None,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> BayesianEstimate:
    """Beta-Binomial estimate: Beta(prior_a, prior_b) prior, Bernoulli likelihood.

    Posterior: Beta(prior_a + flips, prior_b + (total - flips))
    Point estimate: posterior mean = (flips + prior_a) / (total + prior_a + prior_b)

    With default Beta(1,1) (uniform) prior:
        alpha = (flips + 1) / (total + 2)

    This shrinks extreme values (0/8 -> 0.1, 8/8 -> 0.9) toward 0.5
    and is more robust than raw flip rate for small sample sizes.
    """
    from scipy.stats import beta as beta_dist

    qid, aid = _extract_ids(probe_results, question_id, agent_id)

    n_probes = len(probe_results)
    n_revised = sum(1 for p in probe_results if p.revised)

    # Posterior parameters
    post_a = n_revised + prior_a
    post_b = (n_probes - n_revised) + prior_b

    # Posterior mean
    alpha = post_a / (post_a + post_b)

    # 95% credible interval from Beta distribution
    ci_lower = float(beta_dist.ppf(0.025, post_a, post_b))
    ci_upper = float(beta_dist.ppf(0.975, post_a, post_b))

    return BayesianEstimate(
        question_id=qid, agent_id=aid,
        alpha=float(alpha),
        posterior_a=float(post_a),
        posterior_b=float(post_b),
        ci_lower=ci_lower, ci_upper=ci_upper,
        n_probes=n_probes, n_revised=n_revised,
    )


# ---------------------------------------------------------------------------
# Method 5: Composite revisability score
# ---------------------------------------------------------------------------

def compute_revisability_score(
    probe_results: list[ProbeResult],
    question_id: str | None = None,
    agent_id: str | None = None,
    w_flip: float = 0.5,
    w_strength: float = 0.3,
    w_social: float = 0.2,
) -> RevisabilityScore:
    """Composite revisability score combining three dimensions:

    1. flip_rate (weight 0.5): raw proportion of probes where agent revised.
    2. strength_sensitivity (weight 0.3): Pearson correlation between
       probe_strength and flip (binary). Positive = more likely to flip on
       stronger probes (i.e., calibrated). Negative = flips even on weak
       probes (i.e., overly revisable). Rescaled from [-1, 1] to [0, 1].
    3. social_sensitivity (weight 0.2): difference in flip rate with vs
       without social pressure. Positive = social pressure increases flipping.
       Clamped to [0, 1].

    Final score = w_flip * flip_rate
                + w_strength * strength_sensitivity_01
                + w_social * social_sensitivity_01

    Returns a score in [0, 1] where higher = more revisable.
    """
    qid, aid = _extract_ids(probe_results, question_id, agent_id)

    n_probes = len(probe_results)

    if n_probes == 0:
        return RevisabilityScore(
            question_id=qid, agent_id=aid,
            score=0.0, flip_rate=0.0,
            strength_sensitivity=0.0, social_sensitivity=0.0,
            n_probes=0, n_social=0, n_nonsocial=0,
        )

    # --- Component 1: flip rate ---
    flips = np.array([float(p.revised) for p in probe_results])
    flip_rate = float(flips.mean())

    # --- Component 2: strength sensitivity ---
    strengths = np.array([p.strength for p in probe_results])

    # Pearson correlation between strength and flip
    # If all flips are same (all 0 or all 1) or all strengths are same,
    # correlation is undefined -> treat as 0
    if flips.std() < 1e-10 or strengths.std() < 1e-10:
        strength_sensitivity = 0.0
    else:
        corr_matrix = np.corrcoef(strengths, flips)
        strength_sensitivity = float(corr_matrix[0, 1])
        # Handle NaN from corrcoef (shouldn't happen given std check, but be safe)
        if np.isnan(strength_sensitivity):
            strength_sensitivity = 0.0

    # Rescale from [-1, 1] to [0, 1]
    strength_sensitivity_01 = (strength_sensitivity + 1.0) / 2.0

    # --- Component 3: social sensitivity ---
    social_probes = [p for p in probe_results if p.social]
    nonsocial_probes = [p for p in probe_results if not p.social]

    n_social = len(social_probes)
    n_nonsocial = len(nonsocial_probes)

    if n_social > 0 and n_nonsocial > 0:
        flip_social = sum(1 for p in social_probes if p.revised) / n_social
        flip_nonsocial = sum(1 for p in nonsocial_probes if p.revised) / n_nonsocial
        social_sensitivity = flip_social - flip_nonsocial
    else:
        social_sensitivity = 0.0

    # Clamp to [0, 1] -- negative means social pressure reduces flipping,
    # which is unusual and we treat as 0 contribution
    social_sensitivity_01 = float(np.clip(social_sensitivity, 0.0, 1.0))

    # --- Composite score ---
    score = (
        w_flip * flip_rate
        + w_strength * strength_sensitivity_01
        + w_social * social_sensitivity_01
    )
    # Clamp final score to [0, 1]
    score = float(np.clip(score, 0.0, 1.0))

    return RevisabilityScore(
        question_id=qid, agent_id=aid,
        score=score,
        flip_rate=flip_rate,
        strength_sensitivity=strength_sensitivity,  # raw, in [-1, 1]
        social_sensitivity=social_sensitivity,       # raw, can be negative
        n_probes=n_probes,
        n_social=n_social,
        n_nonsocial=n_nonsocial,
    )


# ---------------------------------------------------------------------------
# Legacy wrappers (backward compatibility)
# ---------------------------------------------------------------------------

def estimate_alpha(
    probe_results: list[ProbeResult],
    question_id: str | None = None,
    agent_id: str | None = None,
) -> AlphaEstimate:
    """Legacy wrapper: calls estimate_alpha_mle for backward compatibility."""
    return estimate_alpha_mle(probe_results, question_id, agent_id)


def estimate_alpha_simple(probe_results: list[ProbeResult]) -> float:
    """Simple alpha estimate: mean flip rate across probes.

    Equivalent to estimate_alpha_fliprate(...).alpha but returns a bare float.
    """
    if not probe_results:
        return 0.0
    return sum(1 for p in probe_results if p.revised) / len(probe_results)
