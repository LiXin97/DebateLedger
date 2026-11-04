#!/usr/bin/env python3
"""tab:construct on the public 200-question runs (zero API). The table is defined on public data; it does not reproduce
an earlier version of the table, whose logs were not retained.

Unit: one debate (model x question) whose initial majority is correct; outcome = collapse at Round 3 (parsed Round-3
majority wrong; a debate with no parsed Round-3 answer is not counted as a collapse).
Per-question features from the same model's public probe file (same 200 questions), default condition:
  mean probe FR = mean share of revised probes over the three agents; S/A = mean over agents of the paper's S/A formula
  recomputed from probe_results (rec_scores);
  difficulty = 1 - mean probe initial_correct over all rows of the question (all conditions and agents);
  initial disagreement = 1 - (largest initial-answer count / number of parsed initial answers) in the debate.
Single predictors: AUC of the raw score (not re-oriented, as in the printed table). Combined: logistic regression,
out-of-fold predictions with 5 question folds (pooled: same folds across models, plus model indicators).

usage: construct_auc_public_n200.py <results-dir> <out.json>
"""
import json, os, sys
from collections import Counter
import numpy as np
import statsmodels.api as sm
from scipy.stats import rankdata

R, OUT = sys.argv[1], sys.argv[2]
PAIRS = {"Gemini 3-flash": ("gemini_3_flash_n200", "gemini_3_flash_n200"), "Gemini 3.1-flash-lite": ("gemini_3_1_flash_lite_n200", "gemini_3_1_flash_lite_n200"),
         "Gemini 3.1-pro": ("gemini_3_1_pro_n200", "gemini_3_1_pro_n200"), "GLM-4.6 (no reasoning)": ("openrouter_glm_4_6_noreason_n200", "openrouter_glm_4_6_noreason_n200"),
         "Grok 4.1 Fast": ("openrouter_grok_4_1_fast_n200", "openrouter_grok_4_1_fast_n200"), "HY3 Preview": ("openrouter_hy3_preview_free_n200", "openrouter_hy3_preview_free_n200"),
         "Llama 3.3 70B": ("openrouter_llama_3_3_70b_instruct_n200", "openrouter_llama_3_3_70b_instruct_n200"), "Llama 4 Maverick": ("openrouter_llama_4_maverick_n200", "openrouter_llama_4_maverick_n200"),
         "Llama 4 Scout": ("openrouter_llama_4_scout_n200", "openrouter_llama_4_scout_n200"), "Mistral Small 4": ("openrouter_mistral_small_4_n200", "openrouter_mistral_small_4_n200")}


def maj(ans):
    c = Counter(a for a in ans if a)
    if not c: return None, 0.0
    top = max(c.values()); return sorted(a for a, n in c.items() if n == top)[0], top / sum(c.values())


def final_answers(r):
    """Answers at Round 3, the paper's default endpoint (round_traces[2]; the last logged round is Round 4)."""
    rt = r.get("round_traces") or []
    if len(rt) < 3: raise SystemExit(f"{r['question_id']}: fewer than 3 debate rounds; Round-3 endpoint undefined")
    last = rt[2]
    if isinstance(last, dict): return last.get("answers") or []
    return [x.get("answer") for x in last] if last and isinstance(last[0], dict) else last


def rec_scores(r, keep=lambda p: True):
    """(FR, S/A) of one probe record from its probe_results, S/A by the paper's formula (3_method.tex):
    s_soc = r_social - r_solo, s_arg = r_{strong(+v.strong)} - r_{weak+moderate}, S/A = s_soc / (|s_soc| + |s_arg| + 1e-6)."""
    ps = [p for p in r.get("probe_results") or [] if keep(p) and p.get("revised") is not None]
    if not ps: return None, None
    fr = float(np.mean([p["revised"] for p in ps]))
    soc = [p["revised"] for p in ps if p["social"]]; solo = [p["revised"] for p in ps if not p["social"]]
    hi = [p["revised"] for p in ps if p["strength"] in ("strong", "very_strong")]; lo = [p["revised"] for p in ps if p["strength"] in ("weak", "moderate")]
    if not (soc and solo and hi and lo): return fr, None
    s_soc = np.mean(soc) - np.mean(solo); s_arg = np.mean(hi) - np.mean(lo)
    return fr, float(s_soc / (abs(s_soc) + abs(s_arg) + 1e-6))


def auc(score, y):
    y = np.asarray(y); s = np.asarray(score, float); pos = y == 1; n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0: return None
    r = rankdata(s); return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oof(X, y, folds):
    p = np.empty(len(y))
    for k in sorted(set(folds)):
        tr = folds != k
        m = sm.GLM(y[tr], sm.add_constant(X[tr], has_constant="add"), family=sm.families.Binomial()).fit()
        p[~tr] = m.predict(sm.add_constant(X[~tr], has_constant="add"))
    return p


rows = []
for name, (dfile, pfile) in PAIRS.items():
    dp, pp = os.path.join(R, f"debate_traces_{dfile}.jsonl"), os.path.join(R, f"sa_causal_{pfile}.jsonl")
    if not (os.path.exists(dp) and os.path.exists(pp)): continue
    probe = {}
    for r in map(json.loads, open(pp)):
        probe.setdefault(r["question_id"], []).append(r)
    for r in map(json.loads, open(dp)):
        q = r["question_id"]; pr = probe.get(q)
        if not pr: continue
        im, share = maj(r["initial_answers"]); fm, _ = maj(final_answers(r))
        if im is None or im != r["correct_label"]: continue          # at-risk set: initial majority correct
        d = [x for x in pr if x["condition"] == "default"]
        sc = [rec_scores(x) for x in d]; fr = [f for f, _ in sc if f is not None]; sa = [s for _, s in sc if s is not None]
        ic = [bool(x["initial_correct"]) for x in pr if x.get("initial_correct") is not None]
        if not fr or not ic: continue
        rows.append(dict(model=name, q=q, y=int(fm is not None and fm != r["correct_label"]), fr=float(np.mean(fr)), sa=float(np.mean(sa)) if sa else 0.0,
                         diff=1 - float(np.mean(ic)), dis=1 - share))
qs = sorted({r["q"] for r in rows}); fold_of = {q: i % 5 for i, q in enumerate(np.random.default_rng(20261031).permutation(qs))}
out = {"definition": __doc__.split("usage:")[0].strip(), "per_model": {}}
models = list(dict.fromkeys(r["model"] for r in rows))
for name in models + ["Pooled"]:
    rs = rows if name == "Pooled" else [r for r in rows if r["model"] == name]
    y = np.array([r["y"] for r in rs]); folds = np.array([fold_of[r["q"]] for r in rs])
    base = np.array([[r["fr"], r["diff"], r["dis"]] for r in rs]); full = np.column_stack([base, [r["sa"] for r in rs]])
    if name == "Pooled":
        dummies = np.array([[r["model"] == m for m in models[1:]] for r in rs], float); base = np.column_stack([base, dummies]); full = np.column_stack([full, dummies])
    res = dict(n=len(rs), collapses=int(y.sum()), fr=auc([r["fr"] for r in rs], y), sa=auc([r["sa"] for r in rs], y), dis=auc([r["dis"] for r in rs], y))
    if y.sum() >= 5:
        res["base"] = auc(oof(base, y, folds), y); res["full"] = auc(oof(full, y, folds), y); res["delta"] = res["full"] - res["base"]
    out["per_model"][name] = res
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
json.dump(out, open(OUT, "w"), indent=1)
