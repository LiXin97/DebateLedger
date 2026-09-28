"""B5 — R1-aware decision-rule intervention (parameter-free).

Reviewer W4/Q4 asked for a constructive R1-aware alternative to the negative
probe-gated freeze (LOMO -1.21pp). The Bayesian MLM coefficients
(b_R1_majchange = +0.904, b_R1_unanimous = -0.983) directly suggest:

    Rule: freeze (revert to R0 majority) iff R1_majchange = 1 AND R1_unanimous = 0

Rationale: initial split (not unanimous) + R1 majority swing = MLM-flagged
cascade signature. The rule has NO learned parameters, so leave-one-model-out
training is degenerate (no classifier or threshold to fit). We still report
per-model and question-clustered pooled CIs in the same format as
pilot_gated_lomo.json so the comparison is direct.

Cohort: 4 OSS models (Phi-4-mini, Qwen3-4B, Llama-3.1-8B, Qwen3-8B).

Output: abc_exp/results/r1_aware_decision_rule.json
"""
from __future__ import annotations
import json
import os
import pathlib
from collections import defaultdict
from statistics import mean

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = pathlib.Path(os.environ.get('ABC_RESULTS', ROOT / 'results'))
SEED = 20260424

TRACE_FILES = {
    "llama-3.1-8b": RES / "debate_traces_vllm_llama-3.1-8b.jsonl",
    "phi-4-mini":   RES / "debate_traces_vllm_phi-4-mini.jsonl",
    "qwen3-4b":     RES / "debate_traces_vllm_qwen3-4b.jsonl",
    "qwen3-8b":     RES / "debate_traces_vllm_qwen3-8b.jsonl",
}


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def r1_majchange(rec):
    return 1 if rec["round_traces"][0]["majority_changed"] else 0


def r1_unanimous(rec):
    ans = rec["round_traces"][0]["answers"]
    if not ans:
        return 0
    return 1 if all(a == ans[0] for a in ans) else 0


def apply_rule(rec):
    """Returns 'freeze' or 'continue'.

    R1-aware rule: freeze iff r1_majchange=1 AND r1_unanimous=0.
    """
    if r1_majchange(rec) == 1 and r1_unanimous(rec) == 0:
        return "freeze"
    return "continue"


def evaluate(records):
    """Per-model evaluation under the R1-aware rule."""
    n = len(records)
    base_correct = sum(int(r["final_correct"]) for r in records)
    base_collapse = sum(int(r["initial_correct"] and not r["final_correct"]) for r in records)
    base_correction = sum(int(not r["initial_correct"] and r["final_correct"]) for r in records)

    n_gated = 0
    prevented = 0
    lost = 0
    gated_correct = 0
    n_collapse_init_majwrong = 0  # cases the rule never touches but collapse anyway

    for rec in records:
        was_collapse = rec["initial_correct"] and not rec["final_correct"]
        was_correction = (not rec["initial_correct"]) and rec["final_correct"]
        if apply_rule(rec) == "freeze":
            n_gated += 1
            if was_collapse:
                prevented += 1
            if was_correction:
                lost += 1
            gated_correct += int(rec["initial_correct"])
        else:
            gated_correct += int(rec["final_correct"])

    return {
        "n": n,
        "n_gated": n_gated,
        "gate_rate": n_gated / n if n else 0.0,
        "baseline_acc": base_correct / n if n else 0.0,
        "gated_acc": gated_correct / n if n else 0.0,
        "acc_delta": (gated_correct - base_correct) / n if n else 0.0,
        "baseline_collapses": base_collapse,
        "baseline_corrections": base_correction,
        "prevented_collapses": prevented,
        "lost_corrections": lost,
        "net_flips": prevented - lost,
    }


def evaluate_with_records(records):
    """Same as evaluate, but also returns per-record (was_correct_base, was_correct_gated)
    for bootstrap CI."""
    summary = evaluate(records)
    pairs = []
    for rec in records:
        base_c = int(rec["final_correct"])
        if apply_rule(rec) == "freeze":
            gated_c = int(rec["initial_correct"])
        else:
            gated_c = int(rec["final_correct"])
        pairs.append((rec.get("question_id"), base_c, gated_c))
    return summary, pairs


def cluster_bootstrap_acc_delta(pairs_by_model, B=2000, seed=SEED):
    """Question-clustered bootstrap on pooled (gated_acc - baseline_acc).

    Cluster key = question_id (string). Within each bootstrap iteration we
    resample question_ids with replacement and aggregate ALL records from those
    questions across all models. This respects the cross-model dependence in
    questions (same question is shown to all 4 models)."""
    # Build q -> list of (base, gated) across models
    by_q = defaultdict(list)
    for model, pairs in pairs_by_model.items():
        for qid, b, g in pairs:
            by_q[qid].append((b, g))
    qids = list(by_q.keys())
    n_q = len(qids)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(B):
        sampled = rng.integers(0, n_q, n_q)
        n_total = 0
        s_b = 0
        s_g = 0
        for qi in sampled:
            for b, g in by_q[qids[qi]]:
                s_b += b
                s_g += g
                n_total += 1
        if n_total > 0:
            deltas.append((s_g - s_b) / n_total)
    return {
        "B": B,
        "B_used": len(deltas),
        "n_clusters": n_q,
        "ci_2.5": float(np.percentile(deltas, 2.5)),
        "ci_97.5": float(np.percentile(deltas, 97.5)),
        "median": float(np.percentile(deltas, 50)),
    }


def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main():
    data = {m: load(p) for m, p in TRACE_FILES.items()}
    print("Loaded:")
    for m, recs in data.items():
        nc = sum(1 for r in recs if r["initial_correct"] and not r["final_correct"])
        print(f"  {m}: {len(recs)} debates, {nc} collapses")

    rule_text = "freeze (revert to R0 majority) iff r1_majority_changed=1 AND r1_unanimous=0"

    per_model = {}
    pairs_by_model = {}
    pooled_records = []
    for m, recs in data.items():
        summ, pairs = evaluate_with_records(recs)
        prev_ci = wilson_ci(summ["prevented_collapses"], summ["baseline_collapses"])
        lost_ci = wilson_ci(summ["lost_corrections"], summ["baseline_corrections"])
        per_model[m] = {
            **summ,
            "prevented_ci95": prev_ci,
            "lost_ci95": lost_ci,
        }
        pairs_by_model[m] = pairs
        pooled_records.extend(recs)
        print(f"\n[{m}] gate={summ['n_gated']}/{summ['n']} ({summ['gate_rate']*100:.1f}%)  "
              f"prevented={summ['prevented_collapses']}/{summ['baseline_collapses']}  "
              f"lost={summ['lost_corrections']}/{summ['baseline_corrections']}  "
              f"net={summ['net_flips']:+d}  Δacc={summ['acc_delta']*100:+.2f}pp")

    pooled = evaluate(pooled_records)
    boot = cluster_bootstrap_acc_delta(pairs_by_model, B=2000, seed=SEED)
    print("\n==== POOLED (4-OSS, parameter-free R1-aware rule) ====")
    print(f"  gate={pooled['n_gated']}/{pooled['n']} ({pooled['gate_rate']*100:.1f}%)  "
          f"prevented={pooled['prevented_collapses']}/{pooled['baseline_collapses']}  "
          f"lost={pooled['lost_corrections']}/{pooled['baseline_corrections']}  "
          f"net={pooled['net_flips']:+d}  Δacc={pooled['acc_delta']*100:+.2f}pp")
    print(f"  Cluster bootstrap (question-level, B=2000) 95% CI on Δacc: "
          f"[{boot['ci_2.5']*100:+.2f}, {boot['ci_97.5']*100:+.2f}]pp")

    # Comparison vs probe-gated LOMO baseline
    pgl_path = RES / "pilot_gated_lomo.json"
    pgl_pooled = None
    if pgl_path.exists():
        pgl = json.load(open(pgl_path))
        if "pooled" in pgl:
            pgl_pooled = pgl["pooled"]
            print(f"\n  Probe-gated LOMO baseline pooled Δacc: "
                  f"{pgl_pooled.get('acc_delta', float('nan'))*100:+.2f}pp")

    out = {
        "rule": rule_text,
        "lomo_note": (
            "Rule has NO learned parameters (no tau, no classifier weights), so "
            "leave-one-model-out training is degenerate. Per-model + pooled metrics "
            "with question-clustered bootstrap are reported instead, in the same "
            "format as pilot_gated_lomo.json for direct comparison."
        ),
        "cohort": list(TRACE_FILES.keys()),
        "per_model": per_model,
        "pooled": {
            **pooled,
            "cluster_bootstrap_acc_delta_ci95": boot,
        },
        "comparison_probe_gated_lomo": pgl_pooled,
    }

    out_path = RES / "r1_aware_decision_rule.json"
    json.dump(out, open(out_path, "w"), indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
