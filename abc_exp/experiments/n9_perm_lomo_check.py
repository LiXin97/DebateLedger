#!/usr/bin/env python3
"""Permutation null and leave-one-model-out R^2 behind the N=9 sensitivity check in the paper's appendix (zero API, zero GPU).

Reads alpha_decomp_headline_n9.json only. Permutation: 10,000 shuffles of C^cond (seed as in perm_null_rho.py), plus the
exact null over all 9! orderings. LOMO: ranks over all 9 models, OLS of rank C^cond on rank alpha on 8 models, predict the
ninth, R^2 = 1 - sum (y - yhat)^2 / sum (y - ybar)^2 with ybar over all 9.

Run: python abc_exp/experiments/n9_perm_lomo_check.py [--results-dir abc_exp/results] [--out-dir abc_exp/results]
"""
import argparse
import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

RES = Path(__file__).resolve().parents[1] / "results"
SEED = 20260424


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def perm_test(x, y, n_rand=10_000):
    rx, ry = rankdata(x), rankdata(y)
    obs = corr(rx, ry)
    rng = np.random.default_rng(SEED)
    rand = np.array([corr(rx, rng.permutation(ry)) for _ in range(n_rand)])
    exact = np.array([corr(rx, ry[list(p)]) for p in permutations(range(len(x)))])
    assert len(exact) == math.factorial(len(x))
    return {"rho": obs, "p_two_sided_random": float((np.abs(rand) >= abs(obs) - 1e-12).mean()),
            "p_two_sided_exact": float((np.abs(exact) >= abs(obs) - 1e-12).mean())}


def lomo_r2(x, y, loo_mean=False):
    n = len(x)
    press = tss = 0.0
    for i in range(n):
        keep = np.arange(n) != i
        b, a = np.polyfit(x[keep], y[keep], 1)
        press += (y[i] - (a + b * x[i])) ** 2
        tss += (y[i] - (y[keep].mean() if loo_mean else y.mean())) ** 2
    return float(1 - press / tss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RES)
    ap.add_argument("--out-dir", type=Path, default=RES)
    a = ap.parse_args()
    h = json.load(open(a.results_dir / "alpha_decomp_headline_n9.json"))
    c = np.asarray(h["C_cond"], float)
    out = {"n_models": len(c)}
    for k, key in (("alpha_total", "rho_alpha_total_vs_Ccond"), ("alpha_adv", "rho_alpha_adv_vs_Ccond")):
        x = np.asarray(h[k], float)
        rx, ry = rankdata(x), rankdata(c)
        out[k] = {**perm_test(x, c), "p_asymptotic": h[key]["p"], "lomo_r2_rank": lomo_r2(rx, ry),
                  "lomo_r2_rank_loo_mean": lomo_r2(rx, ry, True), "lomo_r2_raw": lomo_r2(x, c)}
    (a.out_dir / "n9_perm_r2_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
