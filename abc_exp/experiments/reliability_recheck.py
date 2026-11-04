#!/usr/bin/env python3
"""Probe reliability of Section 4 (odd/even split-half, test-retest and inter-agent ICC) on the public block0 pilot, zero API.

usage: reliability_recheck.py [<results-dir>] [<out.json>]
Unit: one (question, agent) record with 8 probes; score = fraction of probes revised (8-probe flip rate).
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
A = {r["question_id"]: r for r in map(json.loads, open(R / "block0_alpha.jsonl"))}
B = {r["question_id"]: r for r in map(json.loads, open(R / "block0_alpha_retest.jsonl"))}
AG = ("agent_0", "agent_1", "agent_2")


def icc(M):
    """Shrout-Fleiss ICC(1,1), (2,1), (3,1) for an n x k matrix."""
    n, k = M.shape; g = M.mean()
    ssr = k * ((M.mean(1) - g) ** 2).sum(); ssc = n * ((M.mean(0) - g) ** 2).sum(); sst = ((M - g) ** 2).sum()
    msr = ssr / (n - 1); msc = ssc / (k - 1); mse = (sst - ssr - ssc) / ((n - 1) * (k - 1)); msw = (sst - ssr) / (n * (k - 1))
    return {"icc11": (msr - msw) / (msr + (k - 1) * msw), "icc21": (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), "icc31": (msr - mse) / (msr + (k - 1) * mse)}


def fr(rec, a):
    e = rec["alpha_estimates"].get(a)
    return None if e is None else e["mean_flip_rate"]


def probes(rec, a):
    """Per-probe revised flags for one agent, in file order; None if the file has no probe-level rows."""
    for key in ("probe_results", "probes"):
        p = rec.get(key)
        if isinstance(p, dict) and a in p: return [bool(x["revised"]) for x in p[a]]
        if isinstance(p, list) and p and isinstance(p[0], dict) and "agent_id" in p[0]:
            return [bool(x["revised"]) for x in p if x["agent_id"] == a]
    for ag in rec.get("agents", []):
        if ag.get("agent_id") == a and ag.get("probe_results"): return [bool(x["revised"]) for x in ag["probe_results"]]
    return None


out = {}
for name, D in (("test", A), ("retest", B)):
    M = np.array([[fr(D[q], a) for a in AG] for q in D if all(fr(D[q], a) is not None for a in AG)], dtype=float)
    out[f"inter_agent_{name}"] = dict(n_questions=len(M), **{k: round(float(v), 4) for k, v in icc(M).items()})
pairs = np.array([[fr(A[q], a), fr(B[q], a)] for q in A if q in B for a in AG if fr(A[q], a) is not None and fr(B[q], a) is not None])
out["test_retest"] = dict(n_pairs=len(pairs), pearson=round(float(np.corrcoef(pairs.T)[0, 1]), 4), **{k: round(float(v), 4) for k, v in icc(pairs).items()})
# Split-half: odd vs even probe positions, per (question, agent) and per question (mean over agents); Spearman-Brown corrected.
sh = {}
for name, D in (("test", A), ("retest", B)):
    rows, qrows = [], []
    for q, rec in D.items():
        per = [probes(rec, a) for a in AG]
        if any(p is None or len(p) != 8 for p in per): continue
        h = [(np.mean(p[0::2]), np.mean(p[1::2])) for p in per]
        rows += h; qrows.append((np.mean([x[0] for x in h]), np.mean([x[1] for x in h])))
    for unit, X in (("agent", rows), ("question", qrows)):
        if len(X) < 3: sh[f"{name}_{unit}"] = None; continue
        X = np.array(X); r = float(np.corrcoef(X.T)[0, 1])
        sh[f"{name}_{unit}"] = dict(n=len(X), r=round(r, 4), spearman_brown=round(2 * r / (1 + r), 4), icc21=round(float(icc(X)["icc21"]), 4))
out["split_half_odd_even"] = sh
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else R / "reliability_recheck.json", "w"), indent=1)
print(json.dumps(out, indent=1))
