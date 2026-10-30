"""Family-taxonomy sweep of the G=7 family-mean Spearman (zero API; 2026-10-30).

Regenerates the grouping-rule table of the camera-ready appendix (A_appendix.tex, "Grouping rule" table: canonical
mean/median, worst leave-one-family-out, drop Qwen, drop Meta and Qwen, Qwen split by generation, Phi + Gemma
small-dense merge, both alternatives) from the per-model rows in family_level_headline_aggregation.json. Before this
script the last three rows had no generator. The Qwen3-32B rows come from round2_reviewer_hardening.json and are not
recomputed here.

    python3 abc_exp/experiments/taxonomy_sweep.py [--rows <json with per_model_rows>] [--out-stem taxonomy_sweep_r3]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scipy.stats import spearmanr

RES = Path(__file__).resolve().parents[1] / "results"
QWEN_GEN = {"qwen3-4b": "Qwen-3", "qwen3-8b": "Qwen-3", "qwen3.5-4b": "Qwen-3.5", "qwen3.5-9b": "Qwen-3.5",
            "qwen3.6-27b-fp8": "Qwen-3.6", "qwen3.6-35b-a3b-fp8": "Qwen-3.6"}


def grouped(rows, key, agg=statistics.mean):
    groups = sorted({key(r) for r in rows})
    a = [agg([r["alpha_tot"] for r in rows if key(r) == g]) for g in groups]
    c = [agg([r["c_cond_pct"] for r in rows if key(r) == g]) for g in groups]
    return groups, a, c


def rho(rows, key, agg=statistics.mean):
    groups, a, c = grouped(rows, key, agg)
    return len(groups), float(spearmanr(a, c).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(RES / "family_level_headline_aggregation.json"))
    ap.add_argument("--out-stem", default="taxonomy_sweep_r3")
    a = ap.parse_args()
    rows = json.load(open(a.rows))["per_model_rows"]
    fam = lambda r: r["family"]
    qsplit = lambda r: QWEN_GEN.get(r["model"], r["family"])
    merge = lambda r: "Phi+Gemma" if r["family"] == "Phi" or r["model"].startswith("gemma") else r["family"]
    both = lambda r: QWEN_GEN.get(r["model"], merge(r))
    out = []
    g, r = rho(rows, fam); out.append(("Canonical family mean", g, r))
    g, r = rho(rows, fam, statistics.median); out.append(("Canonical family median", g, r))
    lofo = []
    for f in sorted({fam(x) for x in rows}):
        sub = [x for x in rows if fam(x) != f]
        lofo.append((rho(sub, fam)[1], f))
    worst = min(lofo)[0]
    out.append((f"Worst leave-one-family-out (drop {'/'.join(f for v, f in lofo if abs(v - worst) < 1e-12)})", 6, worst))
    out.append(("Drop Qwen family", *rho([x for x in rows if fam(x) != "Qwen"], fam)))
    out.append(("Drop Meta and Qwen", *rho([x for x in rows if fam(x) not in ("Qwen", "Meta")], fam)))
    out.append(("Qwen split by generation", *rho(rows, qsplit)))
    out.append(("Phi + Gemma small-dense merge", *rho(rows, merge)))
    out.append(("Both alternative taxonomy choices", *rho(rows, both)))
    res = [{"rule": n, "groups": g, "rho": r} for n, g, r in out]
    (RES / f"{a.out_stem}.json").write_text(json.dumps({"source": Path(a.rows).name, "rows": res}, indent=1))
    md = ["| Grouping rule | Groups | Spearman rho |", "|---|---:|---:|"] + [f"| {x['rule']} | {x['groups']} | {x['rho']:+.3f} |" for x in res]
    (RES / f"{a.out_stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
