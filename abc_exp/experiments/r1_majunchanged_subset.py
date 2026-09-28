"""A2 — R1-majunchanged subset reanalysis (anti-tautology check).

Reviewer W3 concern: predicting collapse from R1 features is partly tautological
because r1_majority_changed IS already a cascade-onset signature.

Test: filter to debates with r1_majority_changed=0, then refit MLM and recompute
pooled-OOF AUCs. If b_alpha direction preserved AND AUC still lifts, the
non-tautological R1 trajectory information adds value beyond the trivial signal.

Cohort: same 4-OSS as canonical MLM (Phi-4-mini, Qwen3-4B, Llama-3.1-8B,
Qwen3-8B) with init_majority_correct=1.

Output: r1_majunchanged_subset_analysis.json
"""
from __future__ import annotations
import json, os, random
from collections import Counter
from pathlib import Path

import numpy as np

RES = Path(os.environ.get('ABC_RESULTS', Path(__file__).resolve().parent.parent / 'results'))
SEED = 20260424

# 4-OSS cohort + alpha values (matches bayesian_multilevel_logistic.py)
ALPHA_TOTAL = {
    'Phi-4-mini':    0.4685593413292949,
    'Qwen3-4B':      0.6067957553618638,
    'Llama-3.1-8B':  0.7737292774360385,
    'Qwen3-8B':      0.6111478230671383,
}
MODEL_CANON = {
    'vllm/llama-3.1-8b': 'Llama-3.1-8B',
    'vllm/phi-4-mini':   'Phi-4-mini',
    'vllm/qwen3-4b':     'Qwen3-4B',
    'vllm/qwen3-8b':     'Qwen3-8B',
}


def load_subset():
    rows = [json.loads(l) for l in open(RES / 'per_debate_r1_features.jsonl')]
    out = []
    for r in rows:
        m = MODEL_CANON.get(r.get('model'), r.get('model'))
        if m not in ALPHA_TOTAL:
            continue
        if not r.get('init_majority_correct'):
            continue
        if int(r.get('r1_majority_changed', 0)) != 0:
            continue  # filter to majunchanged subset
        out.append({
            'model': m,
            'question_id': r['question_id'],
            'collapsed': int(r.get('collapsed', 0)),
            'r1_n_flipped_from_init': float(r.get('r1_n_flipped_from_init', 0)),
            'r1_agree_frac': float(r.get('r1_agree_frac', 0.0)),
            'init_unanimous': int(r.get('init_unanimous', 0)),
            'init_agree_frac': float(r.get('init_agree_frac', 0.0)),
            'r2_majority_changed': r.get('r2_majority_changed'),
            'r3_majority_changed': r.get('r3_majority_changed'),
        })
    return out


def auc_roc(pos, neg):
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return float('nan')
    combined = [(s, 1) for s in pos] + [(s, 0) for s in neg]
    combined.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if combined[k][1] == 1:
                rank_sum += avg
        i = j + 1
    u = rank_sum - n_p * (n_p + 1) / 2
    return u / (n_p * n_n)


def fit_logistic(X, y, lr=0.1, n_epochs=300, l2=1e-3, seed=0):
    keys = sorted({k for x in X for k in x})
    w = {k: 0.0 for k in keys}
    b = 0.0
    n = len(X)
    rng = random.Random(seed)
    idx = list(range(n))
    import math
    for ep in range(n_epochs):
        rng.shuffle(idx)
        gw = {k: 0.0 for k in keys}
        gb = 0.0
        for i in idx:
            s = b
            for k in keys:
                s += w[k] * X[i].get(k, 0.0)
            p = 1.0 / (1.0 + math.exp(-s))
            err = p - y[i]
            for k in keys:
                gw[k] += err * X[i].get(k, 0.0)
            gb += err
        for k in keys:
            w[k] = w[k] - lr * (gw[k] / n + l2 * w[k])
        b = b - lr * (gb / n)
    return w, b


def logistic_score(x, w, b):
    import math
    s = b
    for k, ww in w.items():
        s += ww * x.get(k, 0.0)
    return 1.0 / (1.0 + math.exp(-s))


def oof_probs(X, y, n_folds=5, seed=42):
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    folds = [idx[i::n_folds] for i in range(n_folds)]
    probs = [None] * len(X)
    for fi in range(n_folds):
        test_i = set(folds[fi])
        train_i = [i for i in idx if i not in test_i]
        if not train_i or not test_i:
            continue
        Xtr = [X[i] for i in train_i]
        ytr = [y[i] for i in train_i]
        if sum(ytr) == 0 or sum(ytr) == len(ytr):
            for i in test_i:
                probs[i] = 0.5
            continue
        w, b = fit_logistic(Xtr, ytr)
        for i in test_i:
            probs[i] = logistic_score(X[i], w, b)
    fb = sum(p for p in probs if p is not None) / max(sum(1 for p in probs if p is not None), 1)
    return [fb if p is None else p for p in probs]


def auc_from_probs(probs, y, idx_subset=None):
    if idx_subset is None:
        idx_subset = list(range(len(y)))
    pos = [probs[i] for i in idx_subset if y[i] == 1]
    neg = [probs[i] for i in idx_subset if y[i] == 0]
    return auc_roc(pos, neg)


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fit_mlm(rows):
    """Refit Bayesian MLM on the subset."""
    import pymc as pm
    import arviz as az

    models = sorted({r['model'] for r in rows})
    questions = sorted({r['question_id'] for r in rows})
    m_idx = {m: i for i, m in enumerate(models)}
    q_idx = {q: i for i, q in enumerate(questions)}

    y = np.array([r['collapsed'] for r in rows], dtype=int)
    x_flips = np.array([r['r1_n_flipped_from_init'] for r in rows], dtype=float)
    x_unan = np.array([1.0 if r['r1_agree_frac'] >= 1.0 else 0.0 for r in rows])
    alpha_vec = np.array([ALPHA_TOTAL[r['model']] for r in rows])
    m_vec = np.array([m_idx[r['model']] for r in rows])
    q_vec = np.array([q_idx[r['question_id']] for r in rows])

    def z(x):
        s = x.std(ddof=0)
        return (x - x.mean()) / s if s > 0 else x - x.mean()

    x_flips_z = z(x_flips)
    x_unan_z = z(x_unan)
    alpha_z = z(alpha_vec)

    coords = {'model': models, 'question': questions, 'obs': np.arange(len(y))}
    with pm.Model(coords=coords) as mlm:
        b0 = pm.Normal('intercept', 0.0, 2.0)
        b_flips = pm.Normal('b_R1_flips', 0.0, 1.0)
        b_unan = pm.Normal('b_R1_unanimous', 0.0, 1.0)
        b_alpha = pm.Normal('b_alpha', 0.0, 1.0)
        sigma_model = pm.HalfNormal('sigma_model', 2.0)
        sigma_q = pm.HalfNormal('sigma_question', 2.0)
        z_model = pm.Normal('z_model', 0.0, 1.0, dims='model')
        z_q = pm.Normal('z_question', 0.0, 1.0, dims='question')
        u_model = pm.Deterministic('u_model', sigma_model * z_model, dims='model')
        u_q = pm.Deterministic('u_question', sigma_q * z_q, dims='question')
        eta = (b0 + b_flips * x_flips_z + b_unan * x_unan_z + b_alpha * alpha_z
               + u_model[m_vec] + u_q[q_vec])
        pm.Bernoulli('y_obs', logit_p=eta, observed=y, dims='obs')
        idata = pm.sample(draws=2000, tune=3000, chains=4,
                          target_accept=0.99, random_seed=SEED, progressbar=False)

    summ = az.summary(idata,
                      var_names=['intercept', 'b_R1_flips', 'b_R1_unanimous',
                                 'b_alpha', 'sigma_model', 'sigma_question'],
                      hdi_prob=0.95)
    coefs = {row: dict(mean=float(summ.loc[row, 'mean']),
                       hdi_2_5=float(summ.loc[row, 'hdi_2.5%']),
                       hdi_97_5=float(summ.loc[row, 'hdi_97.5%']),
                       rhat=float(summ.loc[row, 'r_hat']),
                       ess_bulk=float(summ.loc[row, 'ess_bulk']))
             for row in ['intercept', 'b_R1_flips', 'b_R1_unanimous',
                         'b_alpha', 'sigma_model', 'sigma_question']}
    return coefs, models, questions


def main():
    rows = load_subset()
    n = len(rows)
    n_pos = sum(r['collapsed'] for r in rows)
    n_q = len({r['question_id'] for r in rows})
    n_m = len({r['model'] for r in rows})
    print(f"Subset: n_debates={n}, n_collapsed={n_pos}, n_questions={n_q}, n_models={n_m}")
    print(f"  per-model collapses: {Counter((r['model'], r['collapsed']) for r in rows)}")

    # AUCs (Φ_0 baseline; Φ_1 R1-trajectory features that survive filtering;
    # Φ_full adds R2/R3 majority-change indicators)
    Xb, X1, Xf, y = [], [], [], []
    for r in rows:
        b = {'init_unanimous': r['init_unanimous'],
             'init_agree_frac': r['init_agree_frac']}
        x1 = dict(b)
        x1['r1_n_flipped_from_init'] = r['r1_n_flipped_from_init']
        x1['r1_agree_frac'] = r['r1_agree_frac']
        xf = dict(x1)
        xf['r2_majority_changed'] = float(r['r2_majority_changed'] or 0)
        xf['r3_majority_changed'] = float(r['r3_majority_changed'] or 0)
        Xb.append(b); X1.append(x1); Xf.append(xf); y.append(r['collapsed'])

    pb = oof_probs(Xb, y); p1 = oof_probs(X1, y); pf = oof_probs(Xf, y)
    auc_b = auc_from_probs(pb, y)
    auc_1 = auc_from_probs(p1, y)
    auc_f = auc_from_probs(pf, y)
    print(f"\nPooled-OOF AUC: baseline={auc_b:.4f}  R1-traj={auc_1:.4f}  full={auc_f:.4f}")
    print(f"  delta (R1-base) = {auc_1 - auc_b:+.4f}")
    print(f"  delta (full-R1) = {auc_f - auc_1:+.4f}")

    # Bootstrap CIs on deltas (B=1000)
    rng = random.Random(SEED)
    deltas_1b, deltas_f1 = [], []
    skipped = 0
    for _ in range(1000):
        s = [rng.randint(0, n - 1) for _ in range(n)]
        ys = [y[i] for i in s]
        if sum(ys) == 0 or sum(ys) == n:
            skipped += 1
            continue
        ab = auc_from_probs(pb, y, s); a1 = auc_from_probs(p1, y, s); af = auc_from_probs(pf, y, s)
        deltas_1b.append(a1 - ab); deltas_f1.append(af - a1)

    boot = {
        'B': 1000, 'B_used': len(deltas_1b), 'B_skipped': skipped, 'seed': SEED,
        'delta_r1_minus_baseline': {
            'obs': auc_1 - auc_b,
            'ci_2.5': percentile(deltas_1b, 2.5),
            'ci_97.5': percentile(deltas_1b, 97.5),
        },
        'delta_full_minus_r1': {
            'obs': auc_f - auc_1,
            'ci_2.5': percentile(deltas_f1, 2.5),
            'ci_97.5': percentile(deltas_f1, 97.5),
        },
    }

    # MLM refit on subset
    print("\nRefitting Bayesian MLM on subset...")
    mlm_coefs, mlm_models, mlm_q = fit_mlm(rows)
    print(f"\n  b_alpha mean = {mlm_coefs['b_alpha']['mean']:+.3f}  "
          f"95% CrI=[{mlm_coefs['b_alpha']['hdi_2_5']:+.3f}, "
          f"{mlm_coefs['b_alpha']['hdi_97_5']:+.3f}]")

    # Compare to canonical (full) MLM
    canon = json.load(open(RES / 'bayesian_multilevel_logistic.json'))
    canon_b_alpha = canon['coefficients']['b_alpha']

    same_dir = (mlm_coefs['b_alpha']['mean'] < 0) == (canon_b_alpha['mean'] < 0)
    auc_lifts = (auc_1 - auc_b) > 0 and boot['delta_r1_minus_baseline']['ci_2.5'] > 0

    verdict = (
        'CONFIRMS_NON_TAUTOLOGY' if (same_dir and auc_lifts)
        else 'TAUTOLOGY_CONCERN_PARTIAL'
    )

    out = {
        'subset_definition': 'r1_majority_changed = 0 AND init_majority_correct = 1 AND model in 4-OSS cohort',
        'n_debates': n, 'n_collapsed': n_pos, 'n_questions': n_q, 'n_models': n_m,
        'collapse_rate': n_pos / max(n, 1),
        'pooled_oof_auc': {'baseline': auc_b, 'r1_trajectory': auc_1, 'full_trajectory': auc_f},
        'bootstrap_delta_auc': boot,
        'mlm_subset': {
            'coefficients': mlm_coefs,
            'n_models': len(mlm_models),
            'n_questions': len(mlm_q),
        },
        'mlm_canonical_b_alpha_for_comparison': {
            'mean': canon_b_alpha['mean'],
            'hdi_2_5': canon_b_alpha['hdi_2_5'],
            'hdi_97_5': canon_b_alpha['hdi_97_5'],
        },
        'verdict': verdict,
        'verdict_explanation': {
            'b_alpha_same_sign_as_canonical': same_dir,
            'auc_lifts_with_ci_excluding_zero': auc_lifts,
        },
    }

    out_path = RES / 'r1_majunchanged_subset_analysis.json'
    json.dump(out, open(out_path, 'w'), indent=2, default=float)
    print(f"\n=== Verdict: {verdict} ===")
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
