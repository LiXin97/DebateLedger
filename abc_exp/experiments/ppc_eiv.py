"""Posterior predictive check for the EIV (bootstrap_with_noise) model.

For each draw from the posterior, simulate replicate (alpha_obs, k_obs)
datasets and compute Spearman rho on each. Compare the resulting
posterior-predictive distribution of rho to the observed point Spearman.

Output:
  abc_exp/results/ppc_eiv.json
  abc_exp/results/figures/ppc_eiv.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ABC_ROOT = Path(__file__).resolve().parent.parent
RES = ABC_ROOT / "results"
FIG_DIR = RES / "figures"
SEED = 20260424


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    eiv = json.loads((RES / "hierarchical_association.json").read_text())
    rows = eiv["rows"]
    alpha_hat = np.asarray([r["alpha_hat"] for r in rows])
    sd_alpha = np.asarray([r["sd_alpha"] for r in rows])
    ks = np.asarray([r["k"] for r in rows])
    ns = np.asarray([r["n"] for r in rows])
    n_models = len(rows)

    rho_obs = spearman(alpha_hat, ks / np.maximum(ns, 1))

    rng = np.random.default_rng(SEED)
    n_draws = 5000

    # Posterior on latent (alpha, theta)
    alpha_post = alpha_hat + rng.normal(0.0, sd_alpha, size=(n_draws, n_models))
    alpha_post = np.clip(alpha_post, 1e-6, 1 - 1e-6)
    theta_post = rng.beta(ks + 0.5, ns - ks + 0.5, size=(n_draws, n_models))

    # Posterior-predictive datasets: alpha_rep ~ Normal(alpha_lat, sd),
    # k_rep ~ Binomial(n, theta_lat). Recompute rho on each replicate.
    alpha_rep = alpha_post + rng.normal(0.0, sd_alpha, size=(n_draws, n_models))
    k_rep = rng.binomial(np.broadcast_to(ns, (n_draws, n_models)), theta_post)
    theta_rep = k_rep / np.maximum(ns, 1)

    rho_rep = np.array([spearman(alpha_rep[i], theta_rep[i]) for i in range(n_draws)])
    rho_lat = np.array([spearman(alpha_post[i], theta_post[i]) for i in range(n_draws)])

    bayes_p = float((rho_rep >= rho_obs).mean())
    out = dict(
        n_draws=n_draws,
        n_models=n_models,
        seed=SEED,
        rho_observed=rho_obs,
        rho_replicate=dict(
            mean=float(rho_rep.mean()),
            median=float(np.median(rho_rep)),
            ci_lo=float(np.quantile(rho_rep, 0.025)),
            ci_hi=float(np.quantile(rho_rep, 0.975)),
        ),
        rho_latent=dict(
            mean=float(rho_lat.mean()),
            median=float(np.median(rho_lat)),
            ci_lo=float(np.quantile(rho_lat, 0.025)),
            ci_hi=float(np.quantile(rho_lat, 0.975)),
        ),
        posterior_predictive_p_one_sided=bayes_p,
        interpretation=(
            "p in [0.05, 0.95] -> observed rho lies inside the bulk of the "
            "posterior predictive distribution (model is consistent with data)."
        ),
    )
    out_path = RES / "ppc_eiv.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Observed rho = {rho_obs:+.4f}")
    print(f"Replicate rho: median={np.median(rho_rep):+.4f}, "
          f"95% CrI=[{np.quantile(rho_rep, 0.025):+.3f}, {np.quantile(rho_rep, 0.975):+.3f}]")
    print(f"Posterior-predictive one-sided p = {bayes_p:.3f}")
    print(f"Wrote {out_path}")

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(rho_rep, bins=60, density=True, color="#999999", alpha=0.55,
            label="posterior predictive")
    ax.hist(rho_lat, bins=60, density=True, color="#762a83", alpha=0.45,
            label="latent rank corr (no obs noise)", histtype="step", lw=1.4)
    ax.axvline(rho_obs, color="#1b7837", lw=2.0, label=f"observed rho={rho_obs:+.2f}")
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("density")
    ax.set_title(
        f"PPC: EIV bootstrap-with-noise, N={n_models}, ppc-p={bayes_p:.2f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    pdf = FIG_DIR / "ppc_eiv.pdf"
    png = FIG_DIR / "ppc_eiv.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"Wrote {pdf} and {png}")


if __name__ == "__main__":
    main()
