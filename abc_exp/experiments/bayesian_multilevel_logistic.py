"""Bayesian multilevel logistic regression for collapse | round-1 trajectory.

Outcome: per-debate binary collapse, conditional on initial-majority correct.
Fixed effects: round-1 trajectory features (R1 majority change, R1 agent-flip count,
                R1 unanimity) + standardized model-level alpha_total.
Random effects: crossed model + question (intercepts only).

Priors (per D3-v2 § 3 and pre-reg amendment 1 from PRE_REGISTRATION_DEVIATIONS.md):
- fixed effects: Normal(0, 1) on standardized predictors
- intercept: Normal(0, 2)
- random-effect SDs (sigma_model, sigma_question): half-Normal(0, 2)

Identifiability constraint (per team-lead E6 spec): only 2 grouping levels
(model, question). No per-family or per-domain grouping at this N.

Status: pre-registered as exploratory primary analysis introduced post-pivot
(see PRE_REGISTRATION_DEVIATIONS.md amendments). Frequentist Spearman primary
test from PRE_REGISTRATION_N18.md §1 is unaffected.

Output: abc_exp/results/bayesian_multilevel_logistic.json
        abc_exp/results/figures/bayes_mlm_posterior.{pdf,png}
"""
from __future__ import annotations
import json, os
from pathlib import Path

import numpy as np
import pymc as pm
import arviz as az

ABC_ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("ABC_RESULTS", ABC_ROOT / "results"))
FIG_DIR = RES / "figures"
SEED = 20260424

# Pre-extracted feature file (canonical from E5: per_debate_r1_features.jsonl).
# Falls back to the legacy r1_trajectory_features.jsonl name, then to local
# extraction from debate_traces_*.jsonl, so the model is never blocked.
FEATURE_PATH_CANONICAL = RES / "per_debate_r1_features.jsonl"
FEATURE_PATH_LEGACY = RES / "r1_trajectory_features.jsonl"

TRACE_FILES = {
    "Gemini 3-flash":  "debate_traces_gemini_3-flash.jsonl",
    "GPT-5.4-mini":    "debate_traces_openai_gpt-5.4-mini.jsonl",
    "Llama-3.1-8B":    "debate_traces_vllm_llama-3.1-8b.jsonl",
    "Phi-4-mini":      "debate_traces_vllm_phi-4-mini.jsonl",
    "Qwen3-4B":        "debate_traces_vllm_qwen3-4b.jsonl",
    "Qwen3-8B":        "debate_traces_vllm_qwen3-8b.jsonl",
}


def load_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = [json.loads(l) for l in path.open() if l.strip()]
    return rows

# Model-level alpha_total from the locked N=9 cohort
ALPHA_TOTAL = {
    "Sonnet 4.5": 0.45040838801450006,
    "GPT-4o-mini": 0.3424399376361884,
    "GPT-5.4-mini": 0.3499487474438962,
    "Gemini 3-flash": 0.2644407483102608,
    "Phi-4-mini": 0.4685593413292949,
    "Qwen3-4B": 0.6067957553618638,
    "Llama-3.1-8B": 0.7737292774360385,
    "Qwen3-8B": 0.6111478230671383,
    "Qwen3-32B": 0.671420581655481,
}


def build_features_from_traces() -> list[dict]:
    rows = []
    for model_name, fname in TRACE_FILES.items():
        path = RES / fname
        if not path.exists():
            print(f"  skipping {model_name}: {fname} missing")
            continue
        with open(path) as h:
            for line in h:
                d = json.loads(line)
                if not d.get("initial_correct"):
                    continue
                init_ans = d["initial_answers"]
                rounds = d.get("round_traces", [])
                if not rounds or rounds[0]["round"] != 1:
                    continue
                r1 = rounds[0]["answers"]
                r1_n_flips_from_initial = sum(int(a != b) for a, b in zip(r1, init_ans))
                r1_majority_changed = int(rounds[0]["majority"] != d["initial_majority"])
                r1_unanimous = int(len(set(r1)) == 1)
                rows.append(dict(
                    model=model_name,
                    question_id=d["question_id"],
                    collapsed=int(bool(d["collapsed"])),
                    r1_majority_changed=r1_majority_changed,
                    r1_n_flips_from_initial=r1_n_flips_from_initial,
                    r1_unanimous=r1_unanimous,
                ))
    return rows


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    rows = load_jsonl_rows(FEATURE_PATH_CANONICAL)
    source = None
    if rows:
        source = FEATURE_PATH_CANONICAL
    else:
        if FEATURE_PATH_CANONICAL.exists():
            print(f"Canonical feature file exists but is empty: {FEATURE_PATH_CANONICAL}; treating it as missing")
        rows = load_jsonl_rows(FEATURE_PATH_LEGACY)
        if rows:
            source = FEATURE_PATH_LEGACY
    if rows:
        print(f"Loaded {len(rows)} feature rows from {source}")
    else:
        rows = build_features_from_traces()
        print(f"Built {len(rows)} feature rows locally (E5 file not found yet)")
    assert rows, (
        "no feature rows found; provide the gated archived per_debate_r1_features.jsonl "
        "used for the reported MLM, or provide raw debate_traces files with per-round traces"
    )

    # Normalize column names + condition on init-majority-correct.
    # Canonical E5 schema (per_debate_r1_features.jsonl): model, question_id,
    #   collapsed, init_majority_correct, init_unanimous,
    #   r1_majority_changed, r1_n_flipped_from_init, r1_agree_frac.
    # Legacy/local schema: model, collapsed, r1_majority_changed,
    #   r1_n_flips_from_initial, r1_unanimous, initial_correct.
    # Model-name canonicalization for ALPHA_TOTAL lookup.
    MODEL_CANON = {
        "vllm/llama-3.1-8b":     "Llama-3.1-8B",
        "vllm/phi-4-mini":       "Phi-4-mini",
        "vllm/qwen3-4b":         "Qwen3-4B",
        "vllm/qwen3-8b":         "Qwen3-8B",
        "vllm/qwen3-32b":        "Qwen3-32B",
        "openai/gpt-5.4-mini":   "GPT-5.4-mini",
        "gemini/3-flash":        "Gemini 3-flash",
    }
    norm_rows = []
    for r in rows:
        m = r.get("model", r.get("model_id"))
        m = MODEL_CANON.get(m, m)
        # Condition on init_majority_correct (canonical) or initial_correct (legacy).
        init_ok = r.get("init_majority_correct",
                        r.get("initial_correct", r.get("init_correct", 0)))
        if not init_ok:
            continue
        # r1_unanimous: prefer explicit field, else derive from agree_frac.
        r1_unan = r.get("r1_unanimous")
        if r1_unan is None:
            agf = r.get("r1_agree_frac")
            r1_unan = int(agf is not None and agf >= 1.0)
        norm_rows.append(dict(
            model=m,
            question_id=r.get("question_id"),
            collapsed=int(r.get("collapsed", r.get("collapse_y", 0))),
            r1_majority_changed=float(r.get("r1_majority_changed",
                                            r.get("R1_majority_shift", 0))),
            r1_n_flips_from_initial=float(r.get("r1_n_flipped_from_init",
                                                r.get("r1_n_flips_from_initial", 0))),
            r1_unanimous=float(r1_unan),
        ))
    rows = norm_rows
    print(f"After init-majority-correct conditioning: {len(rows)} debates")

    # Drop n_pos<=1 models from the model-level random effect: with 0 or 1
    # collapses they cannot inform sigma_model and they distort the random-
    # intercept posterior. Keeps closed-model debates out of the model-level
    # variance component but does not affect the question-level structure.
    from collections import Counter
    pos_per_model = Counter(r["model"] for r in rows if r["collapsed"])
    keep_models = {m for m in {r["model"] for r in rows}
                   if pos_per_model[m] >= 2}
    n_dropped = sum(1 for r in rows if r["model"] not in keep_models)
    rows = [r for r in rows if r["model"] in keep_models]
    print(f"Dropped {n_dropped} debates from {len(pos_per_model) - len(keep_models)} models with n_pos<=1; kept models: {sorted(keep_models)}")

    # Prepare design matrix
    models = sorted({r["model"] for r in rows})
    questions = sorted({r["question_id"] for r in rows})
    m_idx = {m: i for i, m in enumerate(models)}
    q_idx = {q: i for i, q in enumerate(questions)}

    y = np.array([r["collapsed"] for r in rows], dtype=int)
    x_maj = np.array([r["r1_majority_changed"] for r in rows], dtype=float)
    x_flips = np.array([r["r1_n_flips_from_initial"] for r in rows], dtype=float)
    x_unan = np.array([r["r1_unanimous"] for r in rows], dtype=float)
    alpha_vec = np.array([ALPHA_TOTAL.get(r["model"], np.nan) for r in rows])
    m_vec = np.array([m_idx[r["model"]] for r in rows])
    q_vec = np.array([q_idx[r["question_id"]] for r in rows])

    # Standardize predictors (z-score each fixed effect)
    def z(x):
        s = x.std(ddof=0)
        return (x - x.mean()) / s if s > 0 else x - x.mean()

    x_maj_z = z(x_maj)
    x_flips_z = z(x_flips)
    x_unan_z = z(x_unan)
    alpha_z = z(alpha_vec)

    print(f"\nN debates (init-correct subset): {len(y)}")
    print(f"N collapsed: {y.sum()}  ({100*y.mean():.2f}%)")
    print(f"N models: {len(models)}; N questions: {len(questions)}")

    coords = {"model": models, "question": questions, "obs": np.arange(len(y))}
    with pm.Model(coords=coords) as mlm:
        # Fixed effects
        b0 = pm.Normal("intercept", 0.0, 2.0)
        b_maj = pm.Normal("b_R1_majchange", 0.0, 1.0)
        b_flips = pm.Normal("b_R1_flips", 0.0, 1.0)
        b_unan = pm.Normal("b_R1_unanimous", 0.0, 1.0)
        b_alpha = pm.Normal("b_alpha", 0.0, 1.0)

        # Crossed random intercepts (non-centered parameterization for stability)
        sigma_model = pm.HalfNormal("sigma_model", 2.0)
        sigma_q = pm.HalfNormal("sigma_question", 2.0)
        z_model = pm.Normal("z_model", 0.0, 1.0, dims="model")
        z_q = pm.Normal("z_question", 0.0, 1.0, dims="question")
        u_model = pm.Deterministic("u_model", sigma_model * z_model, dims="model")
        u_q = pm.Deterministic("u_question", sigma_q * z_q, dims="question")

        eta = (
            b0
            + b_maj * x_maj_z
            + b_flips * x_flips_z
            + b_unan * x_unan_z
            + b_alpha * alpha_z
            + u_model[m_vec]
            + u_q[q_vec]
        )
        pm.Bernoulli("y_obs", logit_p=eta, observed=y, dims="obs")

        idata = pm.sample(
            draws=2000, tune=3000, chains=4,
            target_accept=0.99, random_seed=SEED, progressbar=False,
        )
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=SEED, progressbar=False))

    summary = az.summary(
        idata,
        var_names=["intercept", "b_R1_majchange", "b_R1_flips", "b_R1_unanimous",
                   "b_alpha", "sigma_model", "sigma_question"],
        hdi_prob=0.95,
    )
    print("\nPosterior summary:")
    print(summary)

    # Per-model random intercept (posterior mean of u_model)
    u_post = idata.posterior["u_model"].mean(("chain", "draw")).to_numpy()
    per_model_icc_proxy = {
        m: float(u) for m, u in zip(models, u_post)
    }

    # ICC = sigma_model^2 / (sigma_model^2 + sigma_question^2 + pi^2 / 3)
    sm = idata.posterior["sigma_model"].to_numpy().reshape(-1)
    sq = idata.posterior["sigma_question"].to_numpy().reshape(-1)
    icc_model = (sm ** 2) / (sm ** 2 + sq ** 2 + (np.pi ** 2) / 3.0)
    icc_q = (sq ** 2) / (sm ** 2 + sq ** 2 + (np.pi ** 2) / 3.0)

    # PPC summary
    ppc = idata.posterior_predictive["y_obs"].to_numpy().reshape(-1, len(y))
    ppc_collapse_rate = ppc.mean(axis=1)
    obs_rate = float(y.mean())
    ppc_p = float((ppc_collapse_rate >= obs_rate).mean())

    out = dict(
        n_debates=int(len(y)),
        n_collapsed=int(y.sum()),
        n_models=len(models),
        n_questions=len(questions),
        seed=SEED,
        priors=dict(
            fixed_effects="Normal(0, 1)",
            intercept="Normal(0, 2)",
            sd_components="HalfNormal(0, 2)",
        ),
        coefficients={
            row: dict(
                mean=float(summary.loc[row, "mean"]),
                hdi_2_5=float(summary.loc[row, "hdi_2.5%"]),
                hdi_97_5=float(summary.loc[row, "hdi_97.5%"]),
                rhat=float(summary.loc[row, "r_hat"]),
                ess_bulk=float(summary.loc[row, "ess_bulk"]),
            )
            for row in [
                "intercept", "b_R1_majchange", "b_R1_flips", "b_R1_unanimous",
                "b_alpha", "sigma_model", "sigma_question",
            ]
        },
        icc=dict(
            model_median=float(np.median(icc_model)),
            model_hdi_2_5=float(np.quantile(icc_model, 0.025)),
            model_hdi_97_5=float(np.quantile(icc_model, 0.975)),
            question_median=float(np.median(icc_q)),
            question_hdi_2_5=float(np.quantile(icc_q, 0.025)),
            question_hdi_97_5=float(np.quantile(icc_q, 0.975)),
        ),
        per_model_random_intercept_post_mean=per_model_icc_proxy,
        ppc=dict(
            observed_collapse_rate=obs_rate,
            replicate_mean=float(ppc_collapse_rate.mean()),
            replicate_q025=float(np.quantile(ppc_collapse_rate, 0.025)),
            replicate_q975=float(np.quantile(ppc_collapse_rate, 0.975)),
            ppc_p_one_sided=ppc_p,
        ),
        models=models,
    )
    out_path = RES / "bayesian_multilevel_logistic.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    # Posterior trace + forest plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    var_names = ["b_R1_majchange", "b_R1_flips", "b_R1_unanimous", "b_alpha",
                 "sigma_model", "sigma_question"]
    az.plot_forest(idata, var_names=var_names, hdi_prob=0.95, combined=True, ax=ax)
    ax.set_title(
        f"Bayesian MLM coefficients (N={len(y)} debates, {len(models)} models, {len(questions)} questions)\n"
        "fixed effects: Normal(0,1) on standardized predictors; ICC_model="
        f"{np.median(icc_model):.3f}, ICC_q={np.median(icc_q):.3f}",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "bayes_mlm_posterior.pdf")
    fig.savefig(FIG_DIR / "bayes_mlm_posterior.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'bayes_mlm_posterior.pdf'}")


if __name__ == "__main__":
    main()
