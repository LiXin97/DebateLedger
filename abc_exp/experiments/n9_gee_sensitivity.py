#!/usr/bin/env python3
"""GEE and GLM logit tables behind the N=9 sensitivity check in the paper's appendix (zero API, zero GPU).

Each trial is a debate whose initial majority is correct; the outcome is collapse (final majority wrong). Predictors are
model-constant: alpha_adv and initial accuracy. Standard errors are clustered by family, 6 clusters over the 9 models.
Because the predictors do not vary within a model, the per-model counts in conditional_collapse_table.json fix the
likelihood. Fisher 90% intervals are for the Spearman correlations with C^cond.

Run: python abc_exp/experiments/n9_gee_sensitivity.py [--results-dir abc_exp/results]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, spearmanr
from statsmodels.genmod.cov_struct import Exchangeable, Independence

RES = Path(__file__).resolve().parents[1] / "results"
ROW = {"Sonnet 4.5": "Claude Sonnet 4.5"}


def fisher(rho, n, level=0.90):
    z, h = math.atanh(rho), norm.ppf(0.5 + level / 2) / math.sqrt(n - 3)
    return [math.tanh(z - h), math.tanh(z + h)]


def coefs(fit):
    return {k: {"coef": float(fit.params[i]), "se": float(fit.bse[i]), "z": float(fit.tvalues[i]), "p": float(fit.pvalues[i])}
            for i, k in enumerate(["const", "alpha_adv", "init_acc"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RES)
    a = ap.parse_args()
    head = json.load(open(a.results_dir / "alpha_decomp_headline_n9.json"))
    table = {r["model"]: r for r in json.load(open(a.results_dir / "conditional_collapse_table.json"))["table"]}
    rows = []
    for m, adv, tot, cc in zip(head["models"], head["alpha_adv"], head["alpha_total"], head["C_cond"]):
        t = table[ROW.get(m, m)]
        assert abs(100 * t["n_cond_coll"] / t["n_init_correct"] - cc) < 1e-9, m
        rows.append({"model": m, "family": t["family"], "alpha_adv": adv, "alpha_total": tot, "init_acc": t["init_acc"], "C_cond_pct": cc,
                     "n_init_correct": t["n_init_correct"], "n_collapse": t["n_cond_coll"]})
    fams = sorted({r["family"] for r in rows})
    y, X, g = [], [], []
    for r in rows:
        for i in range(r["n_init_correct"]):
            y.append(int(i < r["n_collapse"]))
            X.append([1.0, r["alpha_adv"], r["init_acc"]])
            g.append(fams.index(r["family"]))
    y, X, g = np.array(y), np.array(X), np.array(g)
    fam = sm.families.Binomial()
    out = {"n_models": len(rows), "n_families": len(fams), "n_trials": int(len(y)), "n_collapses": int(y.sum()), "rows": rows,
           "gee_independence": coefs(sm.GEE(y, X, groups=g, family=fam, cov_struct=Independence()).fit()),
           "gee_exchangeable": coefs(sm.GEE(y, X, groups=g, family=fam, cov_struct=Exchangeable()).fit()),
           "glm_family_cluster": coefs(sm.GLM(y, X, family=fam).fit(cov_type="cluster", cov_kwds={"groups": g}))}
    col = lambda k: [r[k] for r in rows]
    n = len(rows)
    out["spearman_alpha_adv_vs_init_acc"] = float(spearmanr(col("alpha_adv"), col("init_acc")).statistic)
    out["fisher_90"] = {}
    for k in ("alpha_total", "alpha_adv", "init_acc"):
        rho = float(spearmanr(col(k), col("C_cond_pct")).statistic)
        out["fisher_90"][k] = {"rho": rho, "ci": fisher(rho, n)}
    (a.results_dir / "n9_gee_sensitivity.json").write_text(json.dumps(out, indent=1) + "\n")
    md = ["# N=9 GEE and GLM tables", "",
          f"{out['n_trials']} debates with a correct initial majority, {out['n_collapses']} collapses, {n} models, {len(fams)} families.", "",
          "| Fit | Term | Coef | SE | z | p |", "|---|---|---:|---:|---:|---:|"]
    for f, lab in (("gee_independence", "GEE logit, independence"), ("gee_exchangeable", "GEE logit, exchangeable"), ("glm_family_cluster", "GLM logit, family-clustered SE")):
        for k in ("alpha_adv", "init_acc"):
            c = out[f][k]
            md.append(f"| {lab} | {k} | {c['coef']:+.3f} | {c['se']:.3f} | {c['z']:+.2f} | {c['p']:.3f} |")
    md += ["", f"Spearman between alpha_adv and initial accuracy: {out['spearman_alpha_adv_vs_init_acc']:+.2f}.", "",
           "| Predictor of C^cond | Spearman rho | Fisher 90% CI |", "|---|---:|---:|"]
    for k, v in out["fisher_90"].items():
        md.append(f"| {k} | {v['rho']:+.2f} | [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] |")
    (a.results_dir / "N9_GEE_SENSITIVITY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
