"""E5 — Verify cascade round distribution + per-model R1-trajectory AUC.

Step 1: Re-count R1 vs R2+ collapses from raw debate_traces_*.jsonl files
(no reliance on cached multiround_probe_powered.json).

Step 2: For each model with per-round trace data, compute:
  (a) baseline AUC: pre-debate features only (initial agreement among agents,
      initial-correct flag, init majority unanimity).
  (b) R1-trajectory AUC: baseline + R1 dynamics (R1 majority shift,
      R1 disagreement, # agents who flipped from initial in R1, R1 majority
      strength, R1-vs-init divergence).
  (c) full-trajectory AUC: baseline + all three rounds.

Pooled OSS AUC reported alongside per-model. ROC computed via Mann-Whitney U.

Output: abc_exp/results/cascade_r1_auc.json plus E5 markdown report.
"""
from __future__ import annotations
import json, math, glob, os, random
from pathlib import Path
from collections import Counter, defaultdict

RES = Path(os.environ.get('ABC_RESULTS', Path(__file__).resolve().parent.parent / 'results'))


def round_of_collapse(row):
    """Return 1, 2, or 3 (debate round in which majority first became wrong),
    or None if init was wrong (cannot collapse) or never wrong (not collapsed).
    Uses NATURAL numbering: round 1 = first debate round, NOT initial state."""
    if not row.get('initial_correct'):
        return None
    if not row.get('collapsed'):
        return None
    correct = row['correct_label']
    for tr in row.get('round_traces') or []:
        if tr['majority'] != correct:
            return tr['round']
    return None  # collapsed but not localizable (final disagreement)


def featurize(row):
    """Return dict of features for AUC analysis. None if missing data."""
    rt = row.get('round_traces') or []
    if len(rt) < 1:
        return None
    init_answers = row.get('initial_answers') or []
    init_maj = row.get('initial_majority')
    init_correct = bool(row.get('initial_correct'))

    # Baseline (pre-debate)
    init_unanimous = int(len(set(a for a in init_answers if a)) <= 1)
    init_agree_count = max(Counter(init_answers).values()) if init_answers else 0
    init_agree_frac = init_agree_count / max(len(init_answers), 1)
    base = {
        'init_correct': int(init_correct),
        'init_unanimous': init_unanimous,
        'init_agree_frac': init_agree_frac,
    }

    # R1 features (the FIRST debate round)
    r1 = rt[0]
    r1_answers = r1['answers']
    r1_maj_changed = int(r1['majority'] != init_maj)
    n_flipped_in_r1 = sum(1 for a, b in zip(init_answers, r1_answers) if a and b and a != b)
    r1_agree_count = max(Counter(r1_answers).values()) if r1_answers else 0
    r1_agree_frac = r1_agree_count / max(len(r1_answers), 1)
    r1_features = {
        **base,
        'r1_majority_changed': r1_maj_changed,
        'r1_n_flipped': n_flipped_in_r1,
        'r1_agree_frac': r1_agree_frac,
    }

    # Full trajectory features
    full = dict(r1_features)
    if len(rt) >= 2:
        r2 = rt[1]
        full['r2_majority_changed'] = int(r2['majority'] != r1['majority'])
        full['r2_agree_frac'] = (max(Counter(r2['answers']).values()) / max(len(r2['answers']), 1))
    if len(rt) >= 3:
        r3 = rt[2]
        full['r3_majority_changed'] = int(r3['majority'] != rt[1]['majority'])
        full['r3_agree_frac'] = (max(Counter(r3['answers']).values()) / max(len(r3['answers']), 1))

    return base, r1_features, full


def auc_roc(scores_pos, scores_neg):
    n_p, n_n = len(scores_pos), len(scores_neg)
    if n_p == 0 or n_n == 0:
        return float('nan')
    combined = [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg]
    combined.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if combined[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j + 1
    u = rank_sum_pos - n_p * (n_p + 1) / 2
    return u / (n_p * n_n)


def logistic_score(features, weights, bias):
    s = bias
    for k, w in weights.items():
        s += w * features.get(k, 0.0)
    return 1.0 / (1.0 + math.exp(-s))


def fit_logistic(X, y, lr=0.1, n_epochs=300, l2=1e-3, seed=0):
    """Tiny batch logistic regression with L2."""
    keys = sorted({k for x in X for k in x})
    w = {k: 0.0 for k in keys}
    b = 0.0
    n = len(X)
    rng = random.Random(seed)
    idx = list(range(n))
    for ep in range(n_epochs):
        rng.shuffle(idx)
        gw = {k: 0.0 for k in keys}
        gb = 0.0
        for i in idx:
            p = logistic_score(X[i], w, b)
            err = p - y[i]
            for k in keys:
                gw[k] += err * X[i].get(k, 0.0)
            gb += err
        for k in keys:
            w[k] = w[k] - lr * (gw[k] / n + l2 * w[k])
        b = b - lr * (gb / n)
    return w, b


def cv_auc(X, y, n_folds=5, seed=42):
    """K-fold CV AUC with tiny logistic regression."""
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    folds = [idx[i::n_folds] for i in range(n_folds)]
    aucs = []
    for fi in range(n_folds):
        test_i = set(folds[fi])
        train_i = [i for i in idx if i not in test_i]
        if not train_i or not test_i:
            continue
        X_tr = [X[i] for i in train_i]
        y_tr = [y[i] for i in train_i]
        # Skip degenerate folds
        if sum(y_tr) == 0 or sum(y_tr) == len(y_tr):
            continue
        w, b = fit_logistic(X_tr, y_tr)
        scores_pos, scores_neg = [], []
        for i in test_i:
            p = logistic_score(X[i], w, b)
            (scores_pos if y[i] == 1 else scores_neg).append(p)
        if scores_pos and scores_neg:
            aucs.append(auc_roc(scores_pos, scores_neg))
    return {
        'mean': sum(aucs) / len(aucs) if aucs else float('nan'),
        'std': (sum((a - sum(aucs)/len(aucs))**2 for a in aucs)/len(aucs))**0.5 if aucs else float('nan'),
        'folds': aucs,
        'n': len(X),
        'n_pos': sum(y),
    }


def per_model(rows):
    base_X, r1_X, full_X, ys = [], [], [], []
    for r in rows:
        feats = featurize(r)
        if feats is None:
            continue
        b, r1, full = feats
        base_X.append(b)
        r1_X.append(r1)
        full_X.append(full)
        ys.append(int(bool(r.get('collapsed', False))))
    if sum(ys) == 0 or sum(ys) == len(ys):
        return {'note': 'degenerate label distribution', 'n': len(ys), 'n_pos': sum(ys)}
    return {
        'baseline': cv_auc(base_X, ys),
        'r1_trajectory': cv_auc(r1_X, ys),
        'full_trajectory': cv_auc(full_X, ys),
    }


def main():
    files = sorted(glob.glob(str(RES / 'debate_traces_*.jsonl')))
    summary = {'cascade_round_distribution': {}, 'per_model_auc': {}}
    pooled_rows = []
    pooled_round_dist = Counter()
    pooled_unlocalized = 0
    for f in files:
        name = Path(f).stem.replace('debate_traces_', '')
        rows = [json.loads(l) for l in open(f) if l.strip()]
        round_dist = Counter()
        unloc = 0
        n_coll = 0
        for r in rows:
            if r.get('collapsed'):
                n_coll += 1
                cr = round_of_collapse(r)
                if cr is None:
                    unloc += 1
                else:
                    round_dist[cr] += 1
        summary['cascade_round_distribution'][name] = {
            'n_debates': len(rows),
            'n_collapses': n_coll,
            'round_1': round_dist[1],
            'round_2': round_dist[2],
            'round_3': round_dist[3],
            'unlocalized': unloc,
        }
        pooled_rows.extend(rows)
        pooled_round_dist.update(round_dist)
        pooled_unlocalized += unloc
        # Per-model AUC
        summary['per_model_auc'][name] = per_model(rows)
    summary['cascade_round_distribution']['_pooled'] = {
        'n_debates': len(pooled_rows),
        'n_collapses': sum(1 for r in pooled_rows if r.get('collapsed')),
        'round_1': pooled_round_dist[1],
        'round_2': pooled_round_dist[2],
        'round_3': pooled_round_dist[3],
        'unlocalized': pooled_unlocalized,
    }
    summary['pooled_auc'] = per_model(pooled_rows)

    out_path = RES / 'cascade_r1_auc.json'
    json.dump(summary, open(out_path, 'w'), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    print('wrote', out_path)


if __name__ == '__main__':
    main()
