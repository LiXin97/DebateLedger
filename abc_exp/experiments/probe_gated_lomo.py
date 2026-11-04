#!/usr/bin/env python3
"""Probe-gated freeze with leave-one-model-out selection on the archived local-model debates (zero API).
The scripts and probe inputs of the original pilot were not retained. This re-implements the pilot rule on newly
collected probe scores (abc_exp/results/gate_rerun/); it is a post-hoc reconstruction, not a reproduction of the pilot.

Rules of the original pilot: tau in {0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7}; mode in {freeze_initial, freeze_if_stable}; for each held-out
model, (tau*, mode*) maximizes net = prevented - lost on the other models; then frozen and applied to the held-out model.
Choices the original pilot does not record (made here):
  * classifier: logistic regression of `collapsed` on per-debate probe features (mean, max, min of the three agents'
    default-condition 8-probe flip rate on that question), fitted on the training models; the gate fires when p >= tau.
  * freeze_initial: when the gate fires, the debate returns its initial majority.
  * freeze_if_stable: when the gate fires, an agent whose Round-1 answer equals its initial answer keeps its initial answer;
    the other agents keep their final answers; the majority is recomputed (unparsed answers dropped, ties alphabetical).
Outputs per held-out model: tau*, mode*, gate%, prevented, lost, net, delta-acc (pp) with a debate-bootstrap 95% CI, and the
oracle delta-acc (best tau/mode chosen on the held-out model itself); pooled totals; weight ladder (r * prevented - lost) / n.

usage: probe_gated_lomo.py --debates DIR --probes DIR --out FILE.json [--selftest]
  --selftest replaces the probe features with seeded random numbers to check the pipeline on the public debates only.
"""
import argparse, glob, json, os
from collections import Counter
import numpy as np
import statsmodels.api as sm

MODELS = ["llama-3.1-8b", "phi-4-mini", "qwen3-4b", "qwen3-8b"]
TAUS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MODES = ["freeze_initial", "freeze_if_stable"]
WEIGHTS = [0.5, 1, 2, 4, 8]


def majority(answers):
    c = Counter(a for a in answers if a)
    if not c: return None
    top = max(c.values())
    return sorted(a for a, n in c.items() if n == top)[0]


def load_debates(d, model):
    out = []
    for r in map(json.loads, open(os.path.join(d, f"debate_traces_vllm_{model}.jsonl"))):
        r1 = r["round_traces"][0]["answers"] if r.get("round_traces") else r["final_answers"]
        stable_final = [ini if (ini and ini == a1) else fin for ini, a1, fin in zip(r["initial_answers"], r1, r["final_answers"])]
        out.append(dict(q=r["question_id"], y=int(bool(r["collapsed"])), coll=int(bool(r["collapsed"])), corr=int(bool(r["corrected"])),
                        orig=int(bool(r["final_correct"])), init=int(majority(r["initial_answers"]) == r["correct_label"]),
                        stable=int(majority(stable_final) == r["correct_label"])))
    return out


def row_flip_rate(r, none_mode):
    """Per-row 8-probe flip rate. paper: the runner's flip_rate (an unparsed reply counts as no flip, as in the paper).
    flip: an unparsed reply counts as a flip (upper bound). drop: unparsed replies are left out (None if all are).
    The paper reports the gate under all three."""
    if none_mode == "paper":
        return r.get("flip_rate")
    pr = r.get("probe_results") or []
    if none_mode == "flip":
        v = [1 if (p.get("post_answer") is None or p.get("revised")) else 0 for p in pr]
    else:
        v = [int(bool(p.get("revised"))) for p in pr if p.get("post_answer") is not None]
    return float(np.mean(v)) if v else None


def load_probe_features(d, model, none_mode="paper"):
    fr = {}
    for f in glob.glob(os.path.join(d, f"gate_default_vllm_{model}_s*.jsonl")) + glob.glob(os.path.join(d, f"sa_causal_vllm_{model}_rerun.jsonl")):
        for r in map(json.loads, open(f)):
            v = row_flip_rate(r, none_mode) if r.get("condition") == "default" else None
            if v is not None:
                fr.setdefault(r["question_id"], {})[int(r["agent_idx"])] = v
    return {q: [float(np.mean(list(v.values()))), max(v.values()), min(v.values())] for q, v in fr.items()}


def outcome(rows, fire, mode):
    """New correctness per debate under the policy. prevented = original collapses now correct; lost = original
    corrections now wrong (classified by the original transition). Other changes are returned separately."""
    new = np.array([(r["init"] if mode == "freeze_initial" else r["stable"]) if f else r["orig"] for r, f in zip(rows, fire)])
    orig = np.array([r["orig"] for r in rows]); coll = np.array([r["coll"] for r in rows]); corr = np.array([r["corr"] for r in rows])
    pv = int(((new == 1) & (coll == 1)).sum()); lo = int(((new == 0) & (corr == 1)).sum())
    other = int(((new == 1) & (orig == 0) & (coll == 0)).sum()) - int(((new == 0) & (orig == 1) & (corr == 0)).sum())
    return new, pv, lo, other


def fit(rows):
    X = sm.add_constant(np.array([r["x"] for r in rows])); y = np.array([r["y"] for r in rows])
    return sm.GLM(y, X, family=sm.families.Binomial()).fit()


def score(model_fit, rows):
    return model_fit.predict(sm.add_constant(np.array([r["x"] for r in rows]), has_constant="add"))


def best(rows, p, modes=MODES, noop=False):
    res = []
    for mode in modes:
        for tau in TAUS:
            new, pv, lo, ot = outcome(rows, p >= tau, mode); res.append((pv - lo, -float((p >= tau).mean()), tau, mode))   # selection target: net = prevented - lost
    if noop: res.insert(0, (0, 0.0, None, "no_op"))
    net, _, tau, mode = max(res, key=lambda x: (x[0], x[1]))   # highest net, then lowest firing rate; ties keep the earlier entry
    return tau, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debates", required=True); ap.add_argument("--probes"); ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--none-mode", choices=["paper", "flip", "drop"], default="paper", help="unparsed probe replies: paper (no flip), flip (count as flip), drop (leave out)")
    a = ap.parse_args(); rng = np.random.default_rng(20261031)
    data = {}
    for m in MODELS:
        rows = load_debates(a.debates, m)
        feats = {} if a.selftest else load_probe_features(a.probes, m, a.none_mode)
        kept = []
        for r in rows:
            r["x"] = list(rng.random(3)) if a.selftest else feats.get(r["q"])
            if r["x"] is not None: kept.append(r)
        data[m] = dict(rows=kept, n_all=len(rows), n_missing_probe=len(rows) - len(kept))
    out = dict(selftest=a.selftest, notes=__doc__.split("usage:")[0].strip(), variants={})

    def cluster_ci(D, Q):
        """95% interval of 100 * mean(D), resampling question_ids (clusters shared across models)."""
        qs = sorted(set(Q)); idx = {q: [] for q in qs}
        for i, q in enumerate(Q): idx[q].append(i)
        groups = [np.array(idx[q]) for q in qs]; bs = []
        for _ in range(a.boot):
            take = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]); bs.append(100 * D[take].mean())
        return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

    def select(train, p_train, modes, noop, nested):
        if not nested: return best(train, p_train, modes, noop)
        # choose (tau, mode) by inner leave-one-model-out predictions on the training models
        inner = [r for r in train]; pin = np.empty(len(inner)); ms = sorted({r["m"] for r in inner})
        for v in ms:
            tr = [r for r in inner if r["m"] != v]; f2 = fit(tr); sel = [i for i, r in enumerate(inner) if r["m"] == v]
            pin[sel] = score(f2, [inner[i] for i in sel])
        return best(inner, pin, modes, noop)

    def run(name, modes=MODES, noop=False, nested=False, gate=None):
        res, D, Q, tot = {}, [], [], Counter()
        for t in MODELS:
            train = [r for m in MODELS if m != t for r in data[m]["rows"]]; test = data[t]["rows"]; n = len(test)
            if gate is None:
                f = fit(train); tau, mode = select(train, score(f, train), modes, noop, nested); p = score(f, test)
                fire = np.zeros(n, bool) if mode == "no_op" else p >= tau
            else:
                tau, mode, fire = gate(train, test)
            new, pv, lo, ot = outcome(test, fire, "freeze_initial" if mode == "no_op" else mode)
            orig = np.array([r["orig"] for r in test]); d = new - orig
            entry = dict(n=n, n_missing_probe=data[t]["n_missing_probe"], tau=tau, mode=mode, gate_pct=round(100 * float(fire.mean()), 2),
                         baseline_acc=round(float(orig.mean()), 4), prevented=pv, lost=lo, other_net=ot, net=int(d.sum()),
                         dacc_pp=round(100 * float(d.mean()), 3), ci95=cluster_ci(d, [r["q"] for r in test]))
            if gate is None:
                o_tau, o_mode = best(test, p, modes); o_new, *_ = outcome(test, p >= o_tau, o_mode)
                entry["oracle"] = dict(tau=o_tau, mode=o_mode, dacc_pp=round(100 * float((o_new - orig).mean()), 3))
            res[t] = entry; D += d.tolist(); Q += [r["q"] for r in test]; tot.update(dict(n=n, prevented=pv, lost=lo, net=int(d.sum())))
        D = np.array(D)
        pooled = dict(n=tot["n"], prevented=tot["prevented"], lost=tot["lost"], net=tot["net"], dacc_pp=round(100 * tot["net"] / tot["n"], 3),
                      ci95=cluster_ci(D, Q), weight_ladder_pp={str(w): round(100 * (w * tot["prevented"] - tot["lost"]) / tot["n"], 3) for w in WEIGHTS},
                      break_even_r=round(tot["lost"] / tot["prevented"], 3) if tot["prevented"] else None)
        out["variants"][name] = dict(per_model=res, pooled=pooled)
        print(name, "pooled", {k: pooled[k] for k in ("prevented", "lost", "net", "dacc_pp", "ci95", "break_even_r")})
        for m, v in res.items(): print("   ", m, {k: v[k] for k in ("tau", "mode", "gate_pct", "prevented", "lost", "net", "dacc_pp", "ci95")})

    for m in MODELS:
        for r in data[m]["rows"]: r["m"] = m
    run("e5_rule")                                              # primary: the pilot rule (both modes, selection on the training models)
    run("freeze_initial_only", modes=["freeze_initial"])       # sensitivity: the initial-majority mode alone
    run("nested_lomo", nested=True)                             # sensitivity: inner LOMO selection
    run("with_noop", noop=True)                                 # sensitivity: allow the selector to choose no intervention
    run("always_freeze", gate=lambda tr, te: (None, "freeze_initial", np.ones(len(te), bool)))
    def raw_fr(tr, te):                                         # raw mean-FR threshold chosen on the training models, freeze_initial
        tau, _ = best(tr, np.array([r["x"][0] for r in tr]), ["freeze_initial"]); return tau, "freeze_initial", np.array([r["x"][0] for r in te]) >= tau
    run("raw_fr_gate", gate=raw_fr)
    def random_rate(tr, te):                                    # random gate at the e5_rule firing rate of that held-out model
        rate = out["variants"]["e5_rule"]["per_model"][te[0]["m"]]["gate_pct"] / 100; return None, "freeze_initial", rng.random(len(te)) < rate
    run("random_same_rate", gate=random_rate)
    out["none_mode"] = a.none_mode; out["probes_dir"] = a.probes
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
