"""Bayesian hierarchical errors-in-variables model for the α–C^cond rank association.

Pre-registered as T1.3 robustness check (PRE_REGISTRATION_N18.md §6.2).

Motivation
----------
The headline test is a one-sided Spearman ρ between *point estimates* of α (8-probe
revisability) and C^cond (conditional debate-collapse). Both quantities are noisy:
α uses a finite probe set per model; C^cond is a binomial proportion with O(50–500)
trials. A naive Spearman ignores measurement error and may be biased toward 0
(attenuation) or amplified by per-model variance.

This script fits a hierarchical errors-in-variables model:

  - latent α_m and θ_m (true conditional collapse rate) for each model m
  - observed alpha_hat_m ~ Normal(α_m, σ_α_m)         # σ from probe SE
  - observed k_m ~ Binomial(n_m, θ_m)                 # exact debate-trace likelihood
  - rank correlation between (α_m, θ_m) inferred via
    Gaussian copula on the latent ranks.

Output: posterior median + 95% credible interval on the rank correlation r_latent.

This run uses the current N=10 matched cohort. When the N=18 expansion is complete
(see PRE_REGISTRATION_N18.md), rerun this script unchanged on the larger dataset.
The headline Spearman test in the main paper is unaffected; this is a disclosed
secondary diagnostic.

Usage:
  python experiments/hierarchical_association.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# ---------------------------------------------------------------------------
# Load matched per-model dataset
# ---------------------------------------------------------------------------
cond = {t["model"]: t for t in json.load(open(RES / "conditional_collapse_table.json"))["table"]}
alpha = json.load(open(RES / "alpha_split.json"))["per_model"]

NAME_MAP = {
    "Claude Sonnet 4.5": "Sonnet 4.5",
    "Claude Haiku 4.5": "Haiku 4.5",  # may be missing in alpha
    "GPT-4o-mini": "GPT-4o-mini",
    "GPT-5.4-mini": "GPT-5.4-mini",
    "Gemini 3-flash": "Gemini 3-flash",
    "Phi-4-mini": "Phi-4-mini",
    "Qwen3-4B": "Qwen3-4B",
    "Llama-3.1-8B": "Llama-3.1-8B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-32B": "Qwen3-32B",
    "Gemini 2.5 Flash": "Gemini 2.5 Flash",
}

rows = []
for cname, ckey in NAME_MAP.items():
    if cname not in cond or ckey not in alpha:
        continue
    c = cond[cname]
    a = alpha[ckey]
    # Use total alpha = mean of social / nonsocial alpha_total (matches Table 1 headline).
    a_total = 0.5 * (a["alpha_total_nonsocial"] + a["alpha_total_social"])
    # Approx probe SE: sqrt(p(1-p)/n_probe), n_probe = 600 questions x 8 probes / 8 ≈ 600 per arm.
    # Conservative: use 600 effective trials per arm, so combined SE on average is
    # sqrt(p(1-p)/1200).
    sd_a = float(np.sqrt(max(a_total * (1 - a_total), 1e-4) / 1200.0))
    n_init_correct = c["n_init_correct"]
    n_collapse = c["n_cond_coll"]
    rows.append(
        dict(
            model=cname,
            alpha_hat=float(a_total),
            sd_alpha=sd_a,
            n=int(n_init_correct),
            k=int(n_collapse),
            cond_pct=float(c["cond_collapse_pct"]),
        )
    )

print(f"N matched models: {len(rows)}")
for r in rows:
    print(
        f"  {r['model']:<24} α={r['alpha_hat']:.3f}±{r['sd_alpha']:.3f} "
        f"k/n={r['k']}/{r['n']}  ({r['cond_pct']:.2f}%)"
    )

# ---------------------------------------------------------------------------
# Try numpyro first; fall back to a simple bootstrap-with-noise estimator.
# ---------------------------------------------------------------------------
M = len(rows)
alpha_hat = np.array([r["alpha_hat"] for r in rows])
sd_a = np.array([r["sd_alpha"] for r in rows])
ks = np.array([r["k"] for r in rows], dtype=int)
ns = np.array([r["n"] for r in rows], dtype=int)


def posterior_summary(samples):
    s = np.asarray(samples)
    return dict(
        mean=float(s.mean()),
        median=float(np.median(s)),
        ci_lo=float(np.quantile(s, 0.025)),
        ci_hi=float(np.quantile(s, 0.975)),
        p_gt_0=float((s > 0).mean()),
    )


def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def fit_numpyro():
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import jax
    import jax.numpy as jnp

    def model(alpha_hat, sd_a, k, n):
        M = alpha_hat.shape[0]
        # Latent α_m on the unit interval via logit transform
        mu_a = numpyro.sample("mu_a", dist.Normal(0.0, 2.0))
        tau_a = numpyro.sample("tau_a", dist.HalfNormal(2.0))
        with numpyro.plate("models", M):
            z_a = numpyro.sample("z_a", dist.Normal(0.0, 1.0))
            alpha_lat = numpyro.deterministic("alpha_lat", jax.nn.sigmoid(mu_a + tau_a * z_a))
            numpyro.sample("alpha_obs", dist.Normal(alpha_lat, sd_a), obs=alpha_hat)
            # latent θ on logit scale, partially correlated with z_a via Gaussian copula
            rho_lat = numpyro.sample("rho_lat", dist.Uniform(-1.0, 1.0)) if False else None
            # We treat rho_lat as a top-level param (sampled outside plate).
            pass
        # Top-level rank correlation parameter
        rho = numpyro.sample("rho", dist.Uniform(-1.0, 1.0))
        with numpyro.plate("models2", M):
            z_t = numpyro.sample(
                "z_t",
                dist.Normal(rho * z_a, jnp.sqrt(jnp.maximum(1 - rho ** 2, 1e-6))),
            )
            mu_t = numpyro.sample("mu_t", dist.Normal(-2.0, 2.0).expand([1])) if False else None
        # Hyperprior for theta (logit scale)
        mu_t_global = numpyro.sample("mu_t_global", dist.Normal(-2.5, 2.0))
        tau_t = numpyro.sample("tau_t", dist.HalfNormal(2.0))
        theta_lat = numpyro.deterministic("theta_lat", jax.nn.sigmoid(mu_t_global + tau_t * z_t))
        numpyro.sample("k_obs", dist.Binomial(n, theta_lat), obs=k)

    rng = jax.random.PRNGKey(2026_04_19)
    kernel = NUTS(model, target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=2000, num_samples=4000, num_chains=2, progress_bar=False)
    mcmc.run(rng, jnp.asarray(alpha_hat), jnp.asarray(sd_a), jnp.asarray(ks), jnp.asarray(ns))
    samples = mcmc.get_samples()
    rho_samples = np.asarray(samples["rho"])
    alpha_lat = np.asarray(samples["alpha_lat"])
    theta_lat = np.asarray(samples["theta_lat"])
    # latent rank correlation per draw
    rank_corrs = np.array([_spearman(alpha_lat[i], theta_lat[i]) for i in range(alpha_lat.shape[0])])
    return rho_samples, rank_corrs, "numpyro"


def fit_bootstrap_with_noise(n_iter: int = 5000, seed: int = 2026_04_19):
    """Fallback: parametric bootstrap that perturbs each α and θ by its measurement
    error, then computes Spearman ρ. This is a frequentist analog of the
    Bayesian posterior on rank correlation under the same measurement model.
    """
    rng = np.random.default_rng(seed)
    rs = np.zeros(n_iter)
    for i in range(n_iter):
        a_draw = alpha_hat + rng.normal(0.0, sd_a)
        # Beta-Binomial: sample θ from Beta(k+1, n-k+1) Jeffreys posterior
        theta_draw = rng.beta(ks + 0.5, ns - ks + 0.5)
        rs[i] = _spearman(a_draw, theta_draw)
    return rs, rs, "bootstrap_with_noise"


try:
    rho_samples, rank_corrs, method = fit_numpyro()
except Exception as e:  # pragma: no cover
    print(f"[numpyro unavailable: {e}] falling back to parametric bootstrap")
    rho_samples, rank_corrs, method = fit_bootstrap_with_noise()

# ---------------------------------------------------------------------------
# Headline (point) Spearman for reference
# ---------------------------------------------------------------------------
theta_hat = ks / np.maximum(ns, 1)
headline_rho = _spearman(alpha_hat, theta_hat)
print(f"\nPoint Spearman ρ (no measurement error): {headline_rho:+.3f}  N={M}")

print(f"\nMethod: {method}")
print("Posterior on latent rank correlation r:")
ps = posterior_summary(rank_corrs)
for k, v in ps.items():
    print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")

print("\nPosterior on copula correlation ρ (latent z space):")
ps_rho = posterior_summary(rho_samples)
for k, v in ps_rho.items():
    print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
out = dict(
    method=method,
    n_models=M,
    headline_point_spearman=headline_rho,
    rho_copula=ps_rho,
    rank_correlation=ps,
    rows=rows,
)
out_path = RES / "hierarchical_association.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nWrote {out_path}")

# Markdown summary for the appendix
md = [
    "# Hierarchical Errors-in-Variables Association",
    "",
    f"Method: `{method}`. N = {M} matched models.",
    "",
    f"Point Spearman ρ (no measurement error): **{headline_rho:+.3f}**.",
    "",
    "## Posterior on latent rank correlation",
    "",
    f"- median = **{ps['median']:+.3f}**",
    f"- 95% CI = [{ps['ci_lo']:+.3f}, {ps['ci_hi']:+.3f}]",
    f"- Pr(r > 0) = {ps['p_gt_0']:.3f}",
    "",
    "## Posterior on copula correlation ρ",
    "",
    f"- median = **{ps_rho['median']:+.3f}**",
    f"- 95% CI = [{ps_rho['ci_lo']:+.3f}, {ps_rho['ci_hi']:+.3f}]",
    f"- Pr(ρ > 0) = {ps_rho['p_gt_0']:.3f}",
    "",
    "Pre-registered (PRE_REGISTRATION_N18.md §6.2). Rerun unchanged on N=18 cohort.",
]
md_path = RES / "HIERARCHICAL_ASSOCIATION.md"
md_path.write_text("\n".join(md) + "\n")
print(f"Wrote {md_path}")
