"""Exact-style permutation null for Spearman rho(alpha_total, C^cond) at N=9.

Backstop for the asymptotic Spearman p=0.013 reported as the pre-registered
primary test in PRE_REGISTRATION_N18.md §1. With N=9 model-level pairs,
9! = 362,880 unique permutations exist; we both enumerate the exact null
and run a 10,000-replicate randomized null with a fixed seed.

Output: abc_exp/results/perm_null_rho.json
"""
from __future__ import annotations
import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np

ABC_ROOT = Path(__file__).resolve().parent.parent
RES = ABC_ROOT / "results"
SEED = 20260424


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    headline = json.loads((RES / "alpha_decomp_headline_n9.json").read_text())
    alpha = np.asarray(headline["alpha_total"], dtype=float)
    c_cond = np.asarray(headline["C_cond"], dtype=float)
    n = len(alpha)
    assert n == 9, f"expected N=9 cohort, got {n}"

    rho_obs = spearman(alpha, c_cond)

    # Exact enumeration: N=9 -> 362,880 perms, easily tractable.
    null_exact = np.empty(math.factorial(n), dtype=np.float64)
    for i, perm in enumerate(permutations(range(n))):
        null_exact[i] = spearman(alpha, c_cond[list(perm)])

    p_one_sided_exact = float((null_exact >= rho_obs).mean())
    p_two_sided_exact = float((np.abs(null_exact) >= abs(rho_obs)).mean())

    rng = np.random.default_rng(SEED)
    n_rand = 10_000
    null_rand = np.empty(n_rand, dtype=np.float64)
    for i in range(n_rand):
        null_rand[i] = spearman(alpha, rng.permutation(c_cond))

    p_one_sided_rand = float((null_rand >= rho_obs).mean())
    p_two_sided_rand = float((np.abs(null_rand) >= abs(rho_obs)).mean())

    out = dict(
        N=n,
        seed=SEED,
        rho_observed=rho_obs,
        spearman_p_asymptotic_two_sided=headline["rho_alpha_total_vs_Ccond"]["p"],
        exact_enumeration=dict(
            n_permutations=int(math.factorial(n)),
            p_one_sided=p_one_sided_exact,
            p_two_sided=p_two_sided_exact,
            null_mean=float(null_exact.mean()),
            null_sd=float(null_exact.std(ddof=1)),
            null_q025=float(np.quantile(null_exact, 0.025)),
            null_q975=float(np.quantile(null_exact, 0.975)),
        ),
        randomized=dict(
            n_replicates=n_rand,
            p_one_sided=p_one_sided_rand,
            p_two_sided=p_two_sided_rand,
        ),
    )
    out_path = RES / "perm_null_rho.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Observed rho={rho_obs:+.4f}")
    print(f"Exact one-sided p (n!={math.factorial(n)}): {p_one_sided_exact:.6f}")
    print(f"Exact two-sided p:                 {p_two_sided_exact:.6f}")
    print(f"Randomized one-sided p (n=10000):  {p_one_sided_rand:.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
