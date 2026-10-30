"""Freeze the response-period extension cohort at the paper's R3 endpoint.

Rebuilds every row from checked-in artifacts, recodes new MMLU-Pro rows to the
paper's Round-3 endpoint, and computes the family-level screen association with
midrank (tie-corrected) Spearman plus a full-enumeration exact permutation test.

This is a POST-HOC response-period extension: it is reported beside, never
merged into, the submitted-paper primary G=7 analysis.  The April
pre-registration targeted an N=18 model-row extension; G=7 is the realized-
cohort non-independence correction, not the preregistered sample size.

Usage (repo root):  python abc_exp/scripts/freeze_extension_R3.py
Writes: abc_exp/results/rebuttal_extension_R3_frozen.json
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import stats as st

REPO_ROOT = Path(__file__).resolve().parents[2]
RES = REPO_ROOT / "abc_exp/results"
R3_INDEX = 2  # round_traces[2] == R3

# New rows recoded from raw traces: model -> (family, alpha_tot, trace file)
TRACE_ROWS = {
    "qwen3-32b":        ("Qwen",    0.6767, "debate_traces_vllm_qwen3_32b_a7_n200.jsonl"),
    "llama-3.3-70b":    ("Meta",    0.4308, "debate_traces_openrouter_llama_3_3_70b_instruct_n200.jsonl"),
    "llama-4-maverick": ("Meta",    0.6881, "debate_traces_openrouter_llama_4_maverick_n200.jsonl"),
    "llama-4-scout":    ("Meta",    0.5311, "debate_traces_openrouter_llama_4_scout_n200.jsonl"),
    "hy3-preview":      ("Tencent", 0.2683, "debate_traces_openrouter_hy3_preview_noreason_n200.jsonl"),
    "glm-4.6":          ("Z.ai",    0.5274, "debate_traces_openrouter_glm_4_6_noreason_n200.jsonl"),
    "qwq-32b":          ("Qwen",    0.5849, "debate_traces_vllm_qwq-32b_n200.jsonl"),
}
# Holdout rows: recomputed from raw traces under the SAME convention as every
# other row (stored c_cond_r3 fields drop debates whose final turns all fail to
# parse; we count them, because an undefined final majority is not correct).
HOLDOUT_ROWS = {
    "gemini-3.1-flash-lite": ("Google",  "debate_traces_gemini_3_1_flash_lite_n200.jsonl"),
    "gemini-3.1-pro":        ("Google",  "debate_traces_gemini_3_1_pro_n200.jsonl"),
    "mistral-small-4":       ("Mistral", "debate_traces_openrouter_mistral_small_4_n200.jsonl"),
    "grok-4.1-fast":         ("xAI",     "debate_traces_openrouter_grok_4_1_fast_n200.jsonl"),
}


def majority(answers):
    valid = sorted(a for a in answers if a is not None)
    if not valid:
        return None
    counts = Counter(valid)
    top = max(counts.values())
    return sorted(a for a, c in counts.items() if c == top)[0]


def r3_row(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    ic = coll = 0
    none_ans = tot_ans = 0
    for r in rows:
        correct = r["correct_label"].upper()
        im = majority(r["initial_answers"])
        fin = [s["answer"] for s in r["round_traces"][R3_INDEX]]
        fm = majority(fin)
        i_ok = im is not None and im.upper() == correct
        f_ok = fm is not None and fm.upper() == correct
        ic += i_ok
        if i_ok and not f_ok:
            coll += 1
        answers = r["initial_answers"] + [s["answer"] for rd in r["round_traces"] for s in rd]
        none_ans += sum(a is None for a in answers)
        tot_ans += len(answers)
    return {
        "n_debates": n, "n_init_correct": ic, "collapses": coll,
        "c_cond_pct": round(100 * coll / max(ic, 1), 4),
        "init_acc": round(ic / n, 4),
        "parse_none_rate": round(none_ans / max(tot_ans, 1), 4),
    }


def partial_midrank(a, c, i) -> float:
    ra, rc, ri = st.rankdata(a), st.rankdata(c), st.rankdata(i)
    r_ac = np.corrcoef(ra, rc)[0, 1]
    r_ai = np.corrcoef(ra, ri)[0, 1]
    r_ci = np.corrcoef(rc, ri)[0, 1]
    return float((r_ac - r_ai * r_ci) / np.sqrt((1 - r_ai ** 2) * (1 - r_ci ** 2)))


def exact_permutation_ps(A, C, I, rho_m, rho_p) -> tuple[float, float, int]:
    """Full enumeration over all n! relabelings of C (statistic uses midranks,
    so tied C values simply yield duplicate statistic values)."""
    ra, rc, ri = st.rankdata(A), st.rankdata(C), st.rankdata(I)
    n = len(ra)
    ra_c = ra - ra.mean(); ri_c = ri - ri.mean(); rc_c = rc - rc.mean()
    sa = np.sqrt((ra_c ** 2).sum()); si = np.sqrt((ri_c ** 2).sum()); sc = np.sqrt((rc_c ** 2).sum())
    r_ai = float(ra_c @ ri_c / (sa * si))
    cnt_m = cnt_p = tot = 0
    chunk = []
    CHUNK = 400_000
    for perm in permutations(range(n)):
        chunk.append(perm)
        if len(chunk) == CHUNK:
            cm, cp, t = _score_chunk(np.asarray(chunk, dtype=np.int8), rc_c, ra_c, ri_c,
                                     sa, si, sc, r_ai, rho_m, rho_p)
            cnt_m += cm; cnt_p += cp; tot += t
            chunk = []
    if chunk:
        cm, cp, t = _score_chunk(np.asarray(chunk, dtype=np.int8), rc_c, ra_c, ri_c,
                                 sa, si, sc, r_ai, rho_m, rho_p)
        cnt_m += cm; cnt_p += cp; tot += t
    return cnt_m / tot, cnt_p / tot, tot


def _score_chunk(P, rc_c, ra_c, ri_c, sa, si, sc, r_ai, rho_m, rho_p):
    RC = rc_c[P]                        # (B, n) permuted centred C ranks
    r_ac = RC @ ra_c / (sa * sc)
    r_ci = RC @ ri_c / (si * sc)
    part = (r_ac - r_ai * r_ci) / np.sqrt((1 - r_ai ** 2) * (1 - r_ci ** 2))
    return (int((np.abs(r_ac) >= abs(rho_m) - 1e-12).sum()),
            int((np.abs(part) >= abs(rho_p) - 1e-12).sum()),
            P.shape[0])


def main() -> int:
    rows = []
    # (a) pre-registered cohort rows, exactly as published (paper endpoint)
    for v in json.loads((RES / "reviewer_hardening_analyses.json").read_text())["per_model"]:
        rows.append(dict(model=v["model"], family=v["family"], alpha_tot=v["alpha_tot"],
                         c_cond_pct=v["c_cond_pct"], init_acc=v["init_acc"],
                         coding="paper-cohort (as published)"))
    # (b) post-hoc holdouts: alpha from the stored summary, C^cond recomputed at R3
    ph = json.loads((RES / "posthoc_matched_holdout_summary.json").read_text())
    keyed = []

    def walk(o):
        if isinstance(o, dict):
            if "key" in o:
                keyed.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(ph)
    alpha_by_key = {r["key"]: r["alpha_total"] for r in keyed if "alpha_total" in r}
    for k, (fam, fn) in HOLDOUT_ROWS.items():
        s = r3_row(RES / fn)
        rows.append(dict(model=k, family=fam, alpha_tot=alpha_by_key[k],
                         c_cond_pct=s["c_cond_pct"], init_acc=s["init_acc"],
                         coding="R3 recode from round_traces[2]",
                         parse_none_rate=s["parse_none_rate"],
                         collapses=s["collapses"], n_init_correct=s["n_init_correct"]))
    # (c) new rows recoded from raw traces at R3
    for m, (fam, alpha, fn) in TRACE_ROWS.items():
        s = r3_row(RES / fn)
        rows.append(dict(model=m, family=fam, alpha_tot=alpha, c_cond_pct=s["c_cond_pct"],
                         init_acc=s["init_acc"], coding="R3 recode from round_traces[2]",
                         parse_none_rate=s["parse_none_rate"],
                         collapses=s["collapses"], n_init_correct=s["n_init_correct"]))

    fams: dict[str, list] = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    names = sorted(fams)
    A = np.array([np.mean([x["alpha_tot"] for x in fams[f]]) for f in names])
    C = np.array([np.mean([x["c_cond_pct"] for x in fams[f]]) for f in names])
    I = np.array([np.mean([x["init_acc"] for x in fams[f]]) for f in names])

    rho_m = float(st.spearmanr(A, C)[0])
    rho_p = partial_midrank(A, C, I)
    partial_df = len(names) - 3  # n - controls(1) - 2
    partial_t = rho_p * np.sqrt(partial_df / (1 - rho_p ** 2))
    partial_t_p = float(2 * st.t.sf(abs(partial_t), df=partial_df))
    row_rho = float(st.spearmanr([r["alpha_tot"] for r in rows], [r["c_cond_pct"] for r in rows])[0])
    p_m, p_p, n_perm = exact_permutation_ps(A, C, I, rho_m, rho_p)

    # Design-mismatch sensitivity: the Tencent/Hy3 screen and debate rows came
    # from different provider tiers/reasoning modes.  Keep the inclusive cohort
    # as the extension result, but report the transparent family-drop recoding.
    keep_names = [name for name in names if name != "Tencent"]
    A_drop = np.array([np.mean([x["alpha_tot"] for x in fams[f]]) for f in keep_names])
    C_drop = np.array([np.mean([x["c_cond_pct"] for x in fams[f]]) for f in keep_names])
    I_drop = np.array([np.mean([x["init_acc"] for x in fams[f]]) for f in keep_names])
    rho_m_drop = float(st.spearmanr(A_drop, C_drop)[0])
    rho_p_drop = partial_midrank(A_drop, C_drop, I_drop)
    p_m_drop, p_p_drop, n_perm_drop = exact_permutation_ps(
        A_drop, C_drop, I_drop, rho_m_drop, rho_p_drop
    )
    partial_df_drop = len(keep_names) - 3
    partial_t_drop = rho_p_drop * np.sqrt(
        partial_df_drop / (1 - rho_p_drop ** 2)
    )
    partial_t_p_drop = float(
        2 * st.t.sf(abs(partial_t_drop), df=partial_df_drop)
    )

    tied = [n for n, v in zip(names, C) if list(C).count(v) > 1]
    out = dict(
        description=("FROZEN post-hoc response-period extension cohort at the paper's R3 endpoint. "
                     "Reported beside, never merged into, the submitted-paper primary G=7 analysis."),
        generator="abc_exp/scripts/freeze_extension_R3.py",
        endpoint="R3 (round_traces[2]); majority = plurality over parsed answers, alphabetic tie-break",
        ranking=("midrank (tie-corrected) Spearman; exact family-label permutation for the "
                 "unadjusted statistic; partial p-values are sensitivity calculations"),
        n_rows=len(rows), g_families=len(names),
        row_marginal_spearman=round(row_rho, 4),
        family_marginal=dict(rho=round(rho_m, 4), exact_p_two_sided=round(p_m, 4)),
        family_partial_after_init_acc=dict(
            rho=round(rho_p, 4),
            t_approx_p_two_sided=round(partial_t_p, 4),
            label_permutation_p_two_sided=round(p_p, 4),
            note=("The full-enumeration label permutation is not an exact conditional-null "
                  "test because it does not preserve the collapse/capability association."),
        ),
        documented_mismatch_sensitivity=dict(
            excluded_family="Tencent",
            reason=("Hy3 screen used a free-tier endpoint; debate used a paid endpoint "
                    "with provider reasoning disabled."),
            status=("post-hoc family-drop sensitivity chosen from a documented measurement "
                    "mismatch; not a replacement confirmatory analysis"),
            n_rows=sum(len(fams[f]) for f in keep_names),
            g_families=len(keep_names),
            family_marginal=dict(
                rho=round(rho_m_drop, 4),
                exact_p_two_sided=round(p_m_drop, 6),
            ),
            family_partial_after_init_acc=dict(
                rho=round(rho_p_drop, 4),
                t_approx_p_two_sided=round(partial_t_p_drop, 6),
                label_permutation_p_two_sided=round(p_p_drop, 6),
                note=("The label-permutation value remains a sensitivity, not an exact "
                      "conditional-null test."),
            ),
            n_permutations=n_perm_drop,
        ),
        n_permutations=n_perm,
        families_tied_at_same_c_cond=tied,
        rows=rows,
        exclusions=dict(
            r1_distill="QC-excluded: 25.7% parse failure at an 8192-token budget (diagnostic only)",
            step_minimax="structurally broken probe lanes (93.7% / 36.3% parse failure)"),
        caveats=[
            "hy3 screen row from the free-tier endpoint, debate row from the paid endpoint with reasoning disabled (endpoint mismatch disclosed)",
            "glm-4.6 screen and debate rows both collected with provider reasoning disabled",
            "capability-adjusted family partial NOT confirmed — supports the paper's triage-only boundary",
        ],
    )
    (RES / "rebuttal_extension_R3_frozen.json").write_text(json.dumps(out, indent=1))
    print(f"N={len(rows)} rows, G={len(names)} families; tied C^cond families: {tied}")
    print(f"row marginal rho={row_rho:.4f}")
    print(f"family marginal rho={rho_m:.4f}  exact two-sided p={p_m:.4f}  ({n_perm} permutations)")
    print(f"family partial   rho={rho_p:.4f}  t-approx p2={partial_t_p:.4f}; "
          f"label-permutation sensitivity p2={p_p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
