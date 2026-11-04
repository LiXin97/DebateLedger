#!/usr/bin/env python3
"""Original-cohort N=9 partial rank associations and the seven-test Holm family (zero API).

Rebuilds tab:partial and the p_Holm column of tab:decomp_n9_sealed from released aggregates:
  * alpha_decomp_headline_n9.json: the nine models, their 8-probe alpha_total, alpha_adv, alpha_cor and C_cond;
  * conditional_collapse_table.json: initial accuracy and debate FR of the same models.
Partial association of a probe channel with C_cond given initial accuracy: rank both variables, regress each rank vector
on the rank of initial accuracy (least squares), and correlate the two residual vectors (Pearson); p from a t test with
n - 3 degrees of freedom. The Holm family has m = 7 tests: the four secondary Spearman rows of tab:decomp_n9_sealed
(alpha_adv, alpha_cor, debate FR, initial accuracy, each against C_cond) and the three partial rows; standard Holm
step-down adjustment (statsmodels multipletests, method="holm").

usage: n9_partial_holm.py [--results-dir DIR] [--out FILE.json]
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

RES = Path(__file__).resolve().parents[1] / "results"
ROW = {"Sonnet 4.5": "Claude Sonnet 4.5"}


def rank_residual(y, x):
    ry, rx = stats.rankdata(y), stats.rankdata(x)
    b = np.polyfit(rx, ry, 1)
    return ry - np.polyval(b, rx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RES)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    head = json.load(open(a.results_dir / "alpha_decomp_headline_n9.json"))
    table = {r["model"]: r for r in json.load(open(a.results_dir / "conditional_collapse_table.json"))["table"]}
    models = head["models"]
    rows = [table[ROW.get(m, m)] for m in models]
    y = np.array(head["C_cond"], float)
    init_acc = np.array([r["init_acc"] for r in rows], float)
    debate_fr = np.array([r["FR"] for r in rows], float)
    n = len(models)

    secondary = {}
    for key, x in (("alpha_adv", head["alpha_adv"]), ("alpha_cor", head["alpha_cor"]), ("debate_FR", debate_fr), ("init_acc", init_acc)):
        rho, p = stats.spearmanr(x, y)
        secondary[key] = dict(rho=float(rho), p_raw=float(p))

    partial = {}
    for key in ("alpha_total", "alpha_adv", "alpha_cor"):
        ra, rb = rank_residual(np.array(head[key], float), init_acc), rank_residual(y, init_acc)
        r = float(np.corrcoef(ra, rb)[0, 1])
        t = r * np.sqrt((n - 3) / (1 - r * r))
        partial[key] = dict(partial_rho=r, p_raw=float(2 * stats.t.sf(abs(t), n - 3)))

    family = [("secondary", k) for k in secondary] + [("partial", k) for k in partial]
    raw = [(secondary if g == "secondary" else partial)[k]["p_raw"] for g, k in family]
    for (g, k), adj in zip(family, multipletests(raw, method="holm")[1]):
        (secondary if g == "secondary" else partial)[k]["p_holm"] = float(adj)

    out = dict(n_models=n, models=models, holm_family_size=len(family), secondary_spearman=secondary, partial_given_init_acc=partial)
    path = a.out or a.results_dir / "n9_partial_holm.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
