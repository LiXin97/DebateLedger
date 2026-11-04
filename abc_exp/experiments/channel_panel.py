#!/usr/bin/env python3
"""22-model channel panel of the paper (tab:full12, tab:sa_ci, tab:shared_pool_sa, Figure 2), zero API.

The panel is
  * the 18 released probe runs with all three system-prompt conditions (at least 500 AS/AA pairs), with the four
    Qwen3.5 / 3.6 rows taken from the thinking-off reruns (abc_exp/results/thinking_off_rerun/), and
  * the four local models rerun for the gate analysis (Llama-3.1-8B, Phi-4-mini, Qwen3-4B, Qwen3-8B;
    abc_exp/results/gate_rerun/sa_causal_*_rerun.jsonl).
Per model (all records with a flip rate; an unparsed probe reply counts as no flip, as in the paper):
  N = agent-condition records; FR(D/AS/AA) = mean record flip rate; order: check = AA > D > AS, sim = AA > AS only;
  d = paired Cohen's d of AA - AS over (question, agent) pairs; p = paired t-test; Holm over the panel.
  S/A (default) = mean over questions of the question-level S/A (mean over agents of the paper's per-record S/A,
  3_method.tex: s_soc / (|s_soc| + |s_arg| + 1e-6)); 95% CI from 10,000 bootstrap resamples of questions (seed 0).
  Shared pool = questions present in every panel run (all conditions and agents); alpha = pooled record flip rate,
  default condition only, question-weighted: S_sens / A_sens = mean s_soc / s_arg, S/A = mean per-question S/A, InitAcc = initial_correct rate.
  Spearman rho between the per-model-pool and shared-pool S/A columns.

usage: channel_panel.py [--rerun DIR] --out FILE.json [--tex FILE.tex]
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "abc_exp/results"
THINKOFF = REL / "thinking_off_rerun"
COND = ("default", "anti_social", "anti_argument")
# display name, family, file (released unless marked); order = table order within family
API = "API models"; LOC = "Local open-weight models (vLLM)"
PANEL = [  # (display name, family, file, group); group sets the table blocks and the figure bands
    ("Gemini 3-flash", "Google", "rel:sa_causal_gemini_3_flash_n200.jsonl", API),
    ("Gemini 3.1-flash-lite", "Google", "rel:sa_causal_gemini_3_1_flash_lite_n200.jsonl", API),
    ("Gemini 3.1-pro", "Google", "rel:sa_causal_gemini_3_1_pro_n200.jsonl", API),
    ("DeepSeek V4 Flash", "DeepSeek", "rel:sa_causal_router_deepseek-v4-flash.jsonl", API),
    ("DeepSeek V4 Pro", "DeepSeek", "rel:sa_causal_router_deepseek-v4-pro.jsonl", API),
    ("GLM-4.6 (no reasoning)", "Zhipu", "rel:sa_causal_openrouter_glm_4_6_noreason_n200.jsonl", API),
    ("Grok 4.1 Fast", "xAI", "rel:sa_causal_openrouter_grok_4_1_fast_n200.jsonl", API),
    ("HY3 Preview", "Tencent", "rel:sa_causal_openrouter_hy3_preview_free_n200.jsonl", API),
    ("Mistral Small 4", "Mistral", "rel:sa_causal_openrouter_mistral_small_4_n200.jsonl", API),
    ("Llama 3.3 70B", "Meta", "rel:sa_causal_openrouter_llama_3_3_70b_instruct_n200.jsonl", API),
    ("Llama 4 Scout", "Meta", "rel:sa_causal_openrouter_llama_4_scout_n200.jsonl", API),
    ("Llama 4 Maverick", "Meta", "rel:sa_causal_openrouter_llama_4_maverick_n200.jsonl", API),
    ("Gemma 4 31B", "Google", "rel:sa_causal_vllm_gemma-4-31b-it-awq.jsonl", LOC),
    ("Llama-3.1-8B", "Meta", "rerun:sa_causal_vllm_llama-3.1-8b_rerun.jsonl", LOC),
    ("Phi-4-mini", "Microsoft", "rerun:sa_causal_vllm_phi-4-mini_rerun.jsonl", LOC),
    ("Qwen3-4B", "Qwen", "rerun:sa_causal_vllm_qwen3-4b_rerun.jsonl", LOC),
    ("Qwen3-8B", "Qwen", "rerun:sa_causal_vllm_qwen3-8b_rerun.jsonl", LOC),
    ("Qwen3-32B", "Qwen", "rel:sa_causal_vllm_qwen3_32b_a7_n200.jsonl", LOC),
    ("Qwen3.5-4B", "Qwen", "off:sa_causal_vllm_qwen3.5-4b_thinkoff.jsonl", LOC),
    ("Qwen3.5-9B", "Qwen", "off:sa_causal_vllm_qwen3.5-9b_thinkoff.jsonl", LOC),
    ("Qwen3.6-27B", "Qwen", "off:sa_causal_vllm_qwen3.6-27b-fp8_thinkoff.jsonl", LOC),
    ("Qwen3.6-35B-A3B", "Qwen", "off:sa_causal_vllm_qwen3.6-35b-a3b-fp8_thinkoff.jsonl", LOC),
]


def rec_sa(r):
    """Paper S/A of one record (same as construct_auc_public_n200.rec_scores); also the two contrasts."""
    ps = [p for p in r.get("probe_results") or [] if p.get("revised") is not None]
    soc = [p["revised"] for p in ps if p["social"]]; solo = [p["revised"] for p in ps if not p["social"]]
    hi = [p["revised"] for p in ps if p["strength"] in ("strong", "very_strong")]; lo = [p["revised"] for p in ps if p["strength"] in ("weak", "moderate")]
    if not (soc and solo and hi and lo): return None, None, None
    s_soc = float(np.mean(soc) - np.mean(solo)); s_arg = float(np.mean(hi) - np.mean(lo))
    return s_soc / (abs(s_soc) + abs(s_arg) + 1e-6), s_soc, s_arg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rerun", default=str(REL / "gate_rerun")); ap.add_argument("--out", required=True); ap.add_argument("--tex")
    a = ap.parse_args()
    base = {"rel": REL, "off": THINKOFF, "rerun": Path(a.rerun)}
    data = {}
    for name, fam, spec, grp in PANEL:
        k, fn = spec.split(":", 1)
        rows = [json.loads(l) for l in open(base[k] / fn) if l.strip()]
        data[name] = [r for r in rows if r.get("flip_rate") is not None and r.get("condition") in COND]
    per = {}
    for name, fam, spec, grp in PANEL:
        rs = data[name]
        by = {c: {(r["question_id"], int(r["agent_idx"])): r["flip_rate"] for r in rs if r["condition"] == c} for c in COND}
        keys = sorted(set(by["anti_social"]) & set(by["anti_argument"]))
        aa = np.array([by["anti_argument"][x] for x in keys]); as_ = np.array([by["anti_social"][x] for x in keys]); dif = aa - as_
        fr = {c: float(np.mean(list(by[c].values()))) for c in COND}
        q = defaultdict(list)
        for r in rs:
            if r["condition"] == "default":
                s = rec_sa(r)[0]
                if s is not None: q[r["question_id"]].append(s)
        qsa = np.array([np.mean(v) for v in q.values()])
        boot = np.random.default_rng(0).choice(qsa, size=(10000, len(qsa)), replace=True).mean(1)
        per[name] = dict(family=fam, group=grp, file=spec, N=len(rs), pairs=len(keys), FR_D=fr["default"], FR_AS=fr["anti_social"], FR_AA=fr["anti_argument"],
                         order="check" if fr["anti_argument"] > fr["default"] > fr["anti_social"] else ("sim" if fr["anti_argument"] > fr["anti_social"] else "none"),
                         d=float(dif.mean() / dif.std(ddof=1)), p=float(stats.ttest_rel(aa, as_).pvalue),
                         sa_default=float(qsa.mean()), sa_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], sa_questions=len(qsa))
    holm = multipletests([per[n]["p"] for n, *_ in PANEL], method="holm")[1]
    for (n, *_), h in zip(PANEL, holm): per[n]["p_holm"] = float(h); per[n]["holm_sig"] = bool(h < 0.05 and per[n]["FR_AA"] > per[n]["FR_AS"])
    shared = set.intersection(*[{r["question_id"] for r in data[n]} for n, *_ in PANEL])
    for n, *_ in PANEL:
        # same basis as the per-model S/A: default condition only, each question weighted equally
        rs = [r for r in data[n] if r["question_id"] in shared and r["condition"] == "default"]; byq = defaultdict(list)
        for r in rs:
            x = rec_sa(r)
            if x[0] is not None: byq[r["question_id"]].append(x)
        qm = lambda i: float(np.mean([np.mean([x[i] for x in v]) for v in byq.values()]))
        qa = defaultdict(list); qi = defaultdict(list)
        for r in rs: qa[r["question_id"]].append(r["flip_rate"]); qi[r["question_id"]].append(bool(r.get("initial_correct")))
        per[n]["shared"] = dict(records=len(rs), questions=len(byq), alpha=float(np.mean([np.mean(v) for v in qa.values()])), S_sens=qm(1), A_sens=qm(2), SA=qm(0),
                                InitAcc=float(np.mean([np.mean(v) for v in qi.values()])))
    names = [n for n, *_ in PANEL]
    rho = stats.spearmanr([per[n]["sa_default"] for n in names], [per[n]["shared"]["SA"] for n in names])
    fams = defaultdict(list)
    for n in names: fams[per[n]["family"]].append(per[n]["d"])
    summary = dict(models=len(names), AA_gt_AS=sum(per[n]["FR_AA"] > per[n]["FR_AS"] for n in names), holm_sig=sum(per[n]["holm_sig"] for n in names),
                   strict_order=sum(per[n]["order"] == "check" for n in names), mean_d=float(np.mean([per[n]["d"] for n in names])),
                   family_mean_d={f: float(np.mean(v)) for f, v in fams.items()},
                   group_mean_d={g: float(np.mean([per[n]['d'] for n in names if per[n]['group'] == g])) for g in (API, LOC)},
                   group_counts={g: sum(per[n]['group'] == g for n in names) for g in (API, LOC)}, shared_questions=len(shared),
                   sa_positive_per_model_pool=sum(per[n]["sa_default"] > 0 for n in names), sa_positive_shared=sum(per[n]["shared"]["SA"] > 0 for n in names),
                   sa_ci_excludes_zero_pos=sum(per[n]["sa_ci"][0] > 0 for n in names), spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue),
                   sign_flips=[n for n in names if (per[n]["sa_default"] > 0) != (per[n]["shared"]["SA"] > 0)], rerun_dir=str(a.rerun))
    json.dump(dict(summary=summary, per_model=per), open(a.out, "w"), indent=1)
    if a.tex:
        f3 = lambda x: f"{x:.3f}".replace("0.", ".", 1) if x >= 0 else f"{x:.3f}"
        sg = lambda x: f"${x:+.3f}$"
        def pfmt(p):
            if p < 1e-3:
                m, e = f"{p:.1e}".split("e"); return f"{m}e{int(e)}"
            return f"{p:.3f}".lstrip("0")
        L = ["% tab:full12 rows"]
        for fam in (API, LOC):
            ms = [n for n in names if per[n]["group"] == fam]
            L.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{fam} ({len(ms)} model{'s' if len(ms) > 1 else ''}, mean $d = {np.mean([per[n]['d'] for n in ms]):.3f}$)}}}} \\\\")
            for n in ms:
                v = per[n]; o = {"check": "$\\checkmark$", "sim": "$\\sim$", "none": "--"}[v["order"]]; star = "$^\\star$" if v["holm_sig"] else ""
                L.append(f"{n} & {v['N']:,} & {f3(v['FR_D'])} & {f3(v['FR_AS'])} & {f3(v['FR_AA'])} & {o} & {v['d']:.3f} & {pfmt(v['p'])}{star} \\\\")
            L.append("\\midrule")
        L += ["", "% tab:sa_ci rows"] + [f"{n} & {sg(per[n]['sa_default'])} & [{sg(per[n]['sa_ci'][0])}, {sg(per[n]['sa_ci'][1])}] \\\\" for n in names]
        L += ["", "% tab:shared_pool_sa rows"] + [f"{n} & {per[n]['shared']['alpha']:.3f} & {sg(per[n]['shared']['S_sens'])} & {sg(per[n]['shared']['A_sens'])} & {sg(per[n]['shared']['SA'])} & {100 * per[n]['shared']['InitAcc']:.1f}\\% \\\\" for n in names]
        open(a.tex, "w").write("\n".join(L) + "\n")
    print(json.dumps(summary, indent=1))
    for n in names:
        v = per[n]; print(f"{n:24s} N={v['N']:5d} D={v['FR_D']:.3f} AS={v['FR_AS']:.3f} AA={v['FR_AA']:.3f} {v['order']:5s} d={v['d']:.3f} holm={v['p_holm']:.1e} SA={v['sa_default']:+.3f} shSA={v['shared']['SA']:+.3f}")


if __name__ == "__main__":
    main()
