"""Thinking-on versus thinking-off table for the four Qwen3.5/3.6 rows, at Round 3 with alphabetic ties (zero API).

Run through abc_exp/r3_unified/rebuild_thinkoff.sh. With --droptrunc-replay it adds two sensitivities: "trunc dropped"
removes triples and debates with a truncated call; "strict alpha" uses the last "Final Answer" of complete generations.
Both need the model text in the gated tier.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

from scipy.stats import rankdata, spearmanr

from thinkoff_truncation import c_cond as c_cond_rows, load, probe_alpha

RES = Path(__file__).resolve().parents[1] / "results"
POST_A6 = {  # model key in family_level_headline_aggregation.json -> (debate trace, alpha probe)
    "deepseek-v4-flash": ("debate_traces_openrouter_deepseek-v4-flash.jsonl", "sa_causal_router_deepseek-v4-flash.jsonl"),
    "gemma-4-31b-it-awq": ("debate_traces_vllm_gemma-4-31b-it.jsonl", "sa_causal_vllm_gemma-4-31b-it-awq.jsonl"),
    "qwen3.5-4b": ("debate_traces_vllm_qwen3.5-4b.jsonl", "sa_causal_vllm_qwen3.5-4b.jsonl"),
    "qwen3.5-9b": ("debate_traces_vllm_qwen3.5-9b.jsonl", "sa_causal_vllm_qwen3.5-9b.jsonl"),
    "qwen3.6-27b-fp8": ("debate_traces_vllm_qwen3.6-27b-fp8.jsonl", "sa_causal_vllm_qwen3.6-27b-fp8.jsonl"),
    "qwen3.6-35b-a3b-fp8": ("debate_traces_vllm_qwen3.6-35b-a3b-fp8.jsonl", "sa_causal_vllm_qwen3.6-35b-a3b-fp8.jsonl"),
}
RERUN = ["qwen3.5-4b", "qwen3.5-9b", "qwen3.6-27b-fp8", "qwen3.6-35b-a3b-fp8"]
ENDPOINTS = ("R3_alpha",)
REPLAY_KEY = {"R3_alpha": "endpoint=R3_alpha, R0 ties=alpha"}
GATES = [("r1_majority_changed", "Round 1 majority change"), ("learned_r1_stump", "Learned Round 1 stump"),
         ("always_freeze", "Always freeze"), ("disagreement_gate", "DisagreementGate")]
PROBE_GATED = {"prevented": 29, "lost": 108, "net": -79, "acc_delta_pp": -1.21}


def c_cond(path: Path, drop_trunc: bool = False):
    return {e: {f: v[f] for f in ("k", "n", "pct")} for e, v in c_cond_rows(load(path), drop_trunc).items() if e in ENDPOINTS}


def mean_flip_rate(path: Path, mode: str = "main") -> float:
    if mode == "main":
        return statistics.mean(float(json.loads(l)["flip_rate"]) for l in open(path) if l.strip())
    return probe_alpha(load(path))[{"droptrunc": "alpha_drop_truncated_triples", "strict": "alpha_strict"}[mode]]


def exact_spearman(x, y):
    rho = spearmanr(x, y).statistic
    rx, ry = rankdata(x), rankdata(y)
    ge = sum(1 for p in itertools.permutations(ry) if abs(spearmanr(rx, p).statistic) >= abs(rho) - 1e-12)
    return float(rho), ge / math.factorial(len(x))


def family_stats(rows):
    fams = sorted({r["family"] for r in rows})
    fa = [statistics.mean(r["alpha_tot"] for r in rows if r["family"] == f) for f in fams]
    fc = [statistics.mean(r["c_cond_pct"] for r in rows if r["family"] == f) for f in fams]
    rho, p = exact_spearman(fa, fc)
    q = fams.index("Qwen")
    return {"family_rho": rho, "family_p_exact": p, "qwen_alpha": fa[q], "qwen_c_cond_pct": fc[q],
            "model_rho_n14": float(spearmanr([r["alpha_tot"] for r in rows], [r["c_cond_pct"] for r in rows]).statistic)}


def vintage(trace_dir: Path, replay_json: Path, base_rows, use_new_alpha: bool, mode: str = "main", fixed_dir: Path = RES):
    """trace_dir holds the four rerun rows of this vintage; the two other added rows are read from fixed_dir."""
    cc = {m: c_cond((trace_dir if m in RERUN else fixed_dir) / POST_A6[m][0], mode == "droptrunc" and m in RERUN) for m in POST_A6}
    alpha = {m: (mean_flip_rate(trace_dir / POST_A6[m][1], mode) if use_new_alpha and m in RERUN else None) for m in POST_A6}
    out = {"c_cond": cc, "alpha_tot": {}, "family": {}, "replay": {}}
    for e in ENDPOINTS:
        rows = []
        for r in base_rows:
            r = dict(r)
            if r["model"] in POST_A6:
                r["c_cond_pct"] = cc[r["model"]][e]["pct"]
                if alpha[r["model"]] is not None:
                    r["alpha_tot"] = alpha[r["model"]]
            rows.append(r)
        out["family"][e] = family_stats(rows)
    out["alpha_tot"] = {r["model"]: (alpha.get(r["model"]) or r["alpha_tot"]) for r in base_rows if r["model"] in POST_A6}
    sens = json.load(open(replay_json))["sensitivity"]
    for e in ENDPOINTS:
        key = next(k for k in sens if k.startswith(REPLAY_KEY[e]))
        s = sens[key]
        out["replay"][e] = {"n": s["n"], "collapses": s["collapses"], "corrections": s["corrections"],
                            **{g: {f: s[g][f] for f in ("prevented", "lost", "net", "acc_delta_pp")} for g, _ in GATES}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", type=Path, required=True)
    ap.add_argument("--new-dir", type=Path, required=True)
    ap.add_argument("--old-replay", type=Path, required=True)
    ap.add_argument("--new-replay", type=Path, required=True)
    ap.add_argument("--droptrunc-replay", type=Path, default=None,
                    help="replay JSON on the new traces with truncated debates removed (adds the two option-A columns)")
    ap.add_argument("--family-json", type=Path, default=RES / "family_level_headline_aggregation.json",
                    help="printed per-model rows (sealed rows and the printed old alpha_tot)")
    ap.add_argument("--tag", default="20261030")
    ap.add_argument("--out-dir", type=Path, default=RES)
    a = ap.parse_args()
    base = json.load(open(a.family_json))["per_model_rows"]
    V = {"old": vintage(a.old_dir, a.old_replay, base, False, fixed_dir=a.new_dir), "new": vintage(a.new_dir, a.new_replay, base, True)}
    cols = [(v, e) for v in ("old", "new") for e in ENDPOINTS]
    if a.droptrunc_replay:
        V["new_droptrunc"] = vintage(a.new_dir, a.droptrunc_replay, base, True, "droptrunc")
        V["new_strict"] = vintage(a.new_dir, a.new_replay, base, True, "strict")
        cols += [("new_droptrunc", "R3_alpha"), ("new_strict", "R3_alpha")]
    lab = {"old": "old", "new": "new", "new_droptrunc": "new, trunc dropped", "new_strict": "new, strict alpha"}
    hdr = "| | " + " | ".join(f"{lab[v]} {e}" for v, e in cols) + " |\n|---|" + "---:|" * len(cols)
    md = ["# Thinking-off rerun: thinking-on (old) vs thinking-off (new)", "",
          "R3_alpha = Round 3, alphabetic ties, the endpoint of the paper.", "",
          "## 1. C^cond (collapses / initial-majority-correct debates)", "", hdr]
    for m in POST_A6:
        md.append(f"| {m} | " + " | ".join(f"{V[v]['c_cond'][m][e]['k']}/{V[v]['c_cond'][m][e]['n']} ({V[v]['c_cond'][m][e]['pct']:.2f}%)" for v, e in cols) + " |")
    for f, lab, fmt in [("qwen_c_cond_pct", "Qwen family C^cond (%)", "{:.2f}"), ("qwen_alpha", "Qwen family alpha_tot", "{:.3f}"),
                        ("family_rho", "family rho (G=7)", "{:+.4f}"), ("family_p_exact", "exact p", "{:.4f}"), ("model_rho_n14", "model-row rho (N=14)", "{:+.4f}")]:
        md.append(f"| {lab} | " + " | ".join(fmt.format(V[v]["family"][e][f]) for v, e in cols) + " |")
    md += ["", "alpha_tot of the rerun rows: " + ", ".join(f"{m} {V['old']['alpha_tot'][m]:.4f} -> {V['new']['alpha_tot'][m]:.4f}"
                                                       + "".join(f" / {V[k]['alpha_tot'][m]:.4f}" for k in ("new_droptrunc", "new_strict") if k in V)
                                                       for m in RERUN)
           + (" (main / trunc dropped / strict)" if "new_droptrunc" in V else ""), "",
           "## 2. Replay cohort: collapses / corrections", "", hdr]
    for f in ("n", "collapses", "corrections"):
        md.append(f"| {f} | " + " | ".join(str(V[v]["replay"][e][f]) for v, e in cols) + " |")
    md += ["", "## 3. Net per gate (prevented / lost / net, accuracy change)", "", hdr,
           "| Probe-gated freeze (6,525 sealed, aggregate only) | " + " | ".join("29 / 108 / -79 (-1.21pp)" for _ in cols) + " |"]
    for g, lab in GATES:
        md.append(f"| {lab} | " + " | ".join(
            "{prevented} / {lost} / {net:+d} ({acc_delta_pp:+.2f}pp)".format(**V[v]["replay"][e][g]) for v, e in cols) + " |")
    (a.out_dir / f"thinkoff_compare_{a.tag}.json").write_text(json.dumps(V, indent=1))
    (a.out_dir / f"thinkoff_compare_{a.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
