"""E5 follow-up — bootstrap CIs on AUC deltas (R1-baseline, full-R1).

Strategy:
  1. Build pooled (X_baseline, X_r1, X_full, y) over all OSS+closed debates.
  2. Generate 5-fold out-of-fold predicted probs for each feature set, using the
     SAME fold assignment + seed as cascade_r1_auc.cv_auc(seed=42).
  3. Paired bootstrap (B=1000, seed=20260424): resample debate indices with
     replacement, compute AUC for each feature set on the resampled OOF probs,
     record (auc_r1 - auc_baseline) and (auc_full - auc_r1).
  4. Append bootstrap_delta_auc block to cascade_r1_auc.json.

Reviewer-hardening extension:
  5. If the row-level inputs for the exact reported cascade_r1_auc.json cohort
     are present, append question-clustered, model-clustered, and two-way-like
     clustered bootstrap CIs under bootstrap_delta_auc_clustered.
  6. If those row-level inputs are absent, preserve the old debate-index block
     and append an explicit infeasibility note instead of recomputing against a
     different currently-present trace set.

Output: in-place update of cascade_r1_auc.json with new key:
  "bootstrap_delta_auc": {
    "seed": 20260424, "B": 1000,
    "delta_r1_minus_baseline": {"obs": ..., "ci_2.5": ..., "ci_97.5": ..., "median": ...},
    "delta_full_minus_r1":     {"obs": ..., "ci_2.5": ..., "ci_97.5": ..., "median": ...}
  }
"""
from __future__ import annotations
from collections import Counter, defaultdict
import glob, json, os, random
from pathlib import Path

try:
    from abc_exp.experiments.cascade_r1_auc import (
        featurize, fit_logistic, logistic_score, auc_roc,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from abc_exp.experiments.cascade_r1_auc import (
        featurize, fit_logistic, logistic_score, auc_roc,
    )

RES = Path(os.environ.get('ABC_RESULTS', Path(__file__).resolve().parent.parent / 'results'))
OUT_PATH = RES / 'cascade_r1_auc.json'
B = 1000
BOOT_SEED = 20260424
OOF_SEED = 42


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def delta_summary(obs, draws):
    return {
        'obs': obs,
        'median': pct(draws, 50),
        'ci_2.5': pct(draws, 2.5),
        'ci_97.5': pct(draws, 97.5),
    }


def build_pooled_xy():
    files = sorted(glob.glob(str(RES / 'debate_traces_*.jsonl')))
    Xb, Xr1, Xfull, y = [], [], [], []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            feats = featurize(row)
            if feats is None:
                continue
            b, r1, full = feats
            Xb.append(b); Xr1.append(r1); Xfull.append(full)
            y.append(int(bool(row.get('collapsed', False))))
    return Xb, Xr1, Xfull, y


def _features_from_trace_row(row):
    feats = featurize(row)
    if feats is None:
        return None
    b, r1, full = feats
    return {
        'model': str(row.get('_bootstrap_model_label') or row.get('model') or row.get('model_name') or ''),
        'question_id': str(row.get('question_id') or ''),
        'Xb': b,
        'Xr1': r1,
        'Xfull': full,
        'y': int(bool(row.get('collapsed', False))),
    }


def _features_from_cached_row(row):
    def as_float_or_zero(value):
        return 0.0 if value is None else float(value)

    b = {
        'init_correct': int(bool(row.get('init_correct', row.get('initial_correct', 0)))),
        'init_unanimous': int(bool(row.get('init_unanimous', 0))),
        'init_agree_frac': as_float_or_zero(row.get('init_agree_frac', 0.0)),
    }
    r1 = {
        **b,
        'r1_majority_changed': int(bool(row.get('r1_majority_changed', 0))),
        'r1_n_flipped': as_float_or_zero(row.get('r1_n_flipped_from_init', row.get('r1_n_flips_from_initial', 0))),
        'r1_agree_frac': as_float_or_zero(row.get('r1_agree_frac', 0.0)),
    }
    full = dict(r1)
    if 'r2_majority_changed' in row:
        full['r2_majority_changed'] = int(bool(row.get('r2_majority_changed') or 0))
        full['r2_agree_frac'] = as_float_or_zero(row.get('r2_agree_frac', 0.0))
    if 'r3_majority_changed' in row:
        full['r3_majority_changed'] = int(bool(row.get('r3_majority_changed') or 0))
        full['r3_agree_frac'] = as_float_or_zero(row.get('r3_agree_frac', 0.0))
    return {
        'model': str(row.get('model') or row.get('model_id') or ''),
        'question_id': str(row.get('question_id') or ''),
        'Xb': b,
        'Xr1': r1,
        'Xfull': full,
        'y': int(bool(row.get('collapsed', row.get('collapse_y', 0)))),
    }


def _summary_model_counts(summary):
    return {
        model: int(stats['n_debates'])
        for model, stats in summary.get('cascade_round_distribution', {}).items()
        if model != '_pooled'
    }


def _validate_records_against_summary(records, summary, source_label):
    expected = _summary_model_counts(summary)
    expected_total = int(summary.get('cascade_round_distribution', {}).get('_pooled', {}).get('n_debates', -1))
    expected_pos = int(summary.get('cascade_round_distribution', {}).get('_pooled', {}).get('n_collapses', -1))
    observed_counts = Counter(r['model'] for r in records)
    observed_total = len(records)
    observed_pos = sum(r['y'] for r in records)
    problems = []
    if expected_total >= 0 and observed_total != expected_total:
        problems.append(f'{source_label}: expected {expected_total} rows from cascade_r1_auc.json, found {observed_total}')
    if expected_pos >= 0 and observed_pos != expected_pos:
        problems.append(f'{source_label}: expected {expected_pos} collapsed rows, found {observed_pos}')
    if expected:
        missing_models = sorted(set(expected) - set(observed_counts))
        extra_models = sorted(set(observed_counts) - set(expected))
        if missing_models:
            problems.append(f'{source_label}: missing model rows for {missing_models}')
        if extra_models:
            problems.append(f'{source_label}: extra model rows for {extra_models}')
        for model, n_expected in expected.items():
            n_observed = observed_counts.get(model)
            if n_observed is not None and n_observed != n_expected:
                problems.append(f'{source_label}: model {model} expected {n_expected} rows, found {n_observed}')
    return problems


def load_reported_cohort_records(summary):
    """Load row-level records for the exact cohort already summarized in JSON.

    The working tree currently contains newer trace files. This loader only
    accepts inputs that match the model labels/counts in cascade_r1_auc.json;
    otherwise it returns no records and explains why.
    """
    notes = []
    expected = _summary_model_counts(summary)
    records = []
    missing = []
    for model_label in sorted(expected):
        path = RES / f'debate_traces_{model_label}.jsonl'
        if not path.exists():
            missing.append(str(path.relative_to(RES.parent)))
            continue
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                row['_bootstrap_model_label'] = model_label
                rec = _features_from_trace_row(row)
                if rec is not None:
                    records.append(rec)
    if missing:
        notes.append('Missing reported-cohort raw trace files: ' + ', '.join(missing))
    trace_problems = _validate_records_against_summary(records, summary, 'reported raw traces') if records else []
    if records and not missing and not trace_problems:
        return records, {'source': 'reported_raw_trace_files', 'notes': notes}
    notes.extend(trace_problems)

    cache = RES / 'per_debate_r1_features.jsonl'
    if not cache.exists():
        notes.append(f'{cache.relative_to(RES.parent)} is absent')
        return [], {'source': None, 'notes': notes}
    if cache.stat().st_size == 0:
        notes.append(f'{cache.relative_to(RES.parent)} exists but is empty')
        return [], {'source': None, 'notes': notes}

    cached_records = []
    with cache.open() as fh:
        for line in fh:
            if line.strip():
                cached_records.append(_features_from_cached_row(json.loads(line)))
    cache_problems = _validate_records_against_summary(cached_records, summary, 'per_debate_r1_features cache')
    if cache_problems:
        notes.extend(cache_problems)
        return [], {'source': None, 'notes': notes}
    return cached_records, {'source': 'per_debate_r1_features.jsonl', 'notes': notes}


def oof_probs(X, y, n_folds=5, seed=42):
    """Return out-of-fold probabilities aligned to the original index order."""
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
        X_tr = [X[i] for i in train_i]
        y_tr = [y[i] for i in train_i]
        if sum(y_tr) == 0 or sum(y_tr) == len(y_tr):
            for i in test_i:
                probs[i] = 0.5
            continue
        w, b = fit_logistic(X_tr, y_tr)
        for i in test_i:
            probs[i] = logistic_score(X[i], w, b)
    # Any leftover (degenerate folds) get mean of pooled probs
    fallback = sum(p for p in probs if p is not None) / max(sum(1 for p in probs if p is not None), 1)
    probs = [fallback if p is None else p for p in probs]
    return probs


def auc_from_probs(probs, y, idx_subset=None):
    if idx_subset is None:
        pos = [probs[i] for i in range(len(y)) if y[i] == 1]
        neg = [probs[i] for i in range(len(y)) if y[i] == 0]
    else:
        pos = [probs[i] for i in idx_subset if y[i] == 1]
        neg = [probs[i] for i in idx_subset if y[i] == 0]
    return auc_roc(pos, neg)


def pooled_oof_from_records(records):
    Xb = [r['Xb'] for r in records]
    Xr1 = [r['Xr1'] for r in records]
    Xfull = [r['Xfull'] for r in records]
    y = [r['y'] for r in records]
    p_b = oof_probs(Xb, y, seed=OOF_SEED)
    p_r1 = oof_probs(Xr1, y, seed=OOF_SEED)
    p_full = oof_probs(Xfull, y, seed=OOF_SEED)
    return y, p_b, p_r1, p_full


def _draw_delta(sample, y, p_b, p_r1, p_full):
    ys = [y[i] for i in sample]
    if not sample or sum(ys) == 0 or sum(ys) == len(sample):
        return None
    a_b = auc_from_probs(p_b, y, sample)
    a_r1 = auc_from_probs(p_r1, y, sample)
    a_full = auc_from_probs(p_full, y, sample)
    return a_r1 - a_b, a_full - a_r1


def clustered_bootstrap(records, y, p_b, p_r1, p_full, key, seed):
    clusters = defaultdict(list)
    for i, row in enumerate(records):
        cluster_value = row.get(key)
        if cluster_value in (None, ''):
            continue
        clusters[cluster_value].append(i)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        return {
            'status': 'infeasible_missing_cluster_ids',
            'cluster_key': key,
            'n_clusters': 0,
            'B': B,
            'B_used': 0,
            'delta_r1_minus_baseline': delta_summary(auc_from_probs(p_r1, y) - auc_from_probs(p_b, y), []),
            'delta_full_minus_r1': delta_summary(auc_from_probs(p_full, y) - auc_from_probs(p_r1, y), []),
        }
    rng = random.Random(seed)
    deltas_r1_b, deltas_full_r1 = [], []
    skipped = 0
    for _ in range(B):
        sample = []
        for cluster_id in (rng.choice(cluster_ids) for _ in range(len(cluster_ids))):
            sample.extend(clusters[cluster_id])
        delta = _draw_delta(sample, y, p_b, p_r1, p_full)
        if delta is None:
            skipped += 1
            continue
        d_r1_b, d_full_r1 = delta
        deltas_r1_b.append(d_r1_b)
        deltas_full_r1.append(d_full_r1)
    return {
        'status': 'computed',
        'cluster_key': key,
        'n_clusters': len(cluster_ids),
        'B': B,
        'B_used': len(deltas_r1_b),
        'B_skipped_degenerate': skipped,
        'delta_r1_minus_baseline': delta_summary(auc_from_probs(p_r1, y) - auc_from_probs(p_b, y), deltas_r1_b),
        'delta_full_minus_r1': delta_summary(auc_from_probs(p_full, y) - auc_from_probs(p_r1, y), deltas_full_r1),
    }


def two_way_like_bootstrap(records, y, p_b, p_r1, p_full, seed):
    questions = sorted({r['question_id'] for r in records if r.get('question_id')})
    models = sorted({r['model'] for r in records if r.get('model')})
    if not questions or not models:
        return {
            'status': 'infeasible_missing_cluster_ids',
            'cluster_keys': ['question_id', 'model'],
            'n_question_clusters': len(questions),
            'n_model_clusters': len(models),
        }
    rng = random.Random(seed)
    deltas_r1_b, deltas_full_r1 = [], []
    skipped = 0
    for _ in range(B):
        q_counts = Counter(rng.choice(questions) for _ in range(len(questions)))
        m_counts = Counter(rng.choice(models) for _ in range(len(models)))
        sample = []
        for i, row in enumerate(records):
            multiplicity = q_counts[row['question_id']] * m_counts[row['model']]
            if multiplicity:
                sample.extend([i] * multiplicity)
        delta = _draw_delta(sample, y, p_b, p_r1, p_full)
        if delta is None:
            skipped += 1
            continue
        d_r1_b, d_full_r1 = delta
        deltas_r1_b.append(d_r1_b)
        deltas_full_r1.append(d_full_r1)
    return {
        'status': 'computed',
        'method': 'crossed_cluster_resample_with_question_model_multiplicity_product',
        'cluster_keys': ['question_id', 'model'],
        'n_question_clusters': len(questions),
        'n_model_clusters': len(models),
        'B': B,
        'B_used': len(deltas_r1_b),
        'B_skipped_degenerate': skipped,
        'delta_r1_minus_baseline': delta_summary(auc_from_probs(p_r1, y) - auc_from_probs(p_b, y), deltas_r1_b),
        'delta_full_minus_r1': delta_summary(auc_from_probs(p_full, y) - auc_from_probs(p_r1, y), deltas_full_r1),
        'caveat': 'Two-way-like percentile bootstrap resamples question and model clusters independently using fixed out-of-fold predictions; it is a design-aware sensitivity, not a full refit exact two-way clustered bootstrap.',
    }


def infeasible_cluster_block(summary, notes):
    old = summary.get('bootstrap_delta_auc', {})
    obs_r1 = old.get('delta_r1_minus_baseline', {}).get('obs')
    obs_full = old.get('delta_full_minus_r1', {}).get('obs')

    def empty_delta(obs):
        return {'obs': obs, 'median': None, 'ci_2.5': None, 'ci_97.5': None}

    subblock = {
        'status': 'infeasible_missing_reported_cohort_row_level_data',
        'B': B,
        'B_used': 0,
        'delta_r1_minus_baseline': empty_delta(obs_r1),
        'delta_full_minus_r1': empty_delta(obs_full),
        'caveat': 'The checked-out cascade_r1_auc.json has aggregate AUCs and the old debate-index bootstrap, but the exact reported-cohort row-level traces/cache with question_id and model identifiers are absent or empty. Clustered CIs cannot be recovered from aggregate AUC summaries without changing the estimand.',
    }
    return {
        'status': 'infeasible_missing_reported_cohort_row_level_data',
        'seed': BOOT_SEED,
        'B': B,
        'oof_seed': OOF_SEED,
        'observed_delta_source': 'cascade_r1_auc.json:bootstrap_delta_auc',
        'input_audit': notes,
        'question_clustered': {**subblock, 'cluster_key': 'question_id', 'n_clusters': None},
        'model_clustered': {**subblock, 'cluster_key': 'model', 'n_clusters': None},
        'two_way_like_clustered': {
            **subblock,
            'cluster_keys': ['question_id', 'model'],
            'n_question_clusters': None,
            'n_model_clusters': None,
        },
    }


def append_clustered_bootstrap(summary, records=None, audit=None):
    if records is None or audit is None:
        records, audit = load_reported_cohort_records(summary)
    if not records:
        return infeasible_cluster_block(summary, audit['notes'])

    y, p_b, p_r1, p_full = pooled_oof_from_records(records)
    auc_b_obs = auc_from_probs(p_b, y)
    auc_r1_obs = auc_from_probs(p_r1, y)
    auc_full_obs = auc_from_probs(p_full, y)
    return {
        'status': 'computed',
        'seed': BOOT_SEED,
        'B': B,
        'oof_seed': OOF_SEED,
        'source': audit['source'],
        'source_notes': audit['notes'],
        'n_debates': len(records),
        'n_collapsed': sum(y),
        'pooled_oof_auc': {
            'baseline': auc_b_obs,
            'r1_trajectory': auc_r1_obs,
            'full_trajectory': auc_full_obs,
        },
        'question_clustered': clustered_bootstrap(records, y, p_b, p_r1, p_full, 'question_id', BOOT_SEED + 101),
        'model_clustered': clustered_bootstrap(records, y, p_b, p_r1, p_full, 'model', BOOT_SEED + 202),
        'two_way_like_clustered': two_way_like_bootstrap(records, y, p_b, p_r1, p_full, BOOT_SEED + 303),
    }


def debate_index_bootstrap_from_records(records):
    y, p_b, p_r1, p_full = pooled_oof_from_records(records)
    n = len(y)
    auc_b_obs = auc_from_probs(p_b, y)
    auc_r1_obs = auc_from_probs(p_r1, y)
    auc_full_obs = auc_from_probs(p_full, y)
    delta_r1_b_obs = auc_r1_obs - auc_b_obs
    delta_full_r1_obs = auc_full_obs - auc_r1_obs

    print(f'  N = {n}, n_pos = {sum(y)}')
    print(f'  pooled OOF AUC: baseline={auc_b_obs:.4f}  r1={auc_r1_obs:.4f}  full={auc_full_obs:.4f}')
    print(f'  observed deltas: R1-baseline={delta_r1_b_obs:+.4f}  full-R1={delta_full_r1_obs:+.4f}')

    rng = random.Random(BOOT_SEED)
    deltas_r1_b, deltas_full_r1 = [], []
    skipped = 0
    for bi in range(B):
        sample = [rng.randint(0, n - 1) for _ in range(n)]
        delta = _draw_delta(sample, y, p_b, p_r1, p_full)
        if delta is None:
            skipped += 1
            continue
        d_r1_b, d_full_r1 = delta
        deltas_r1_b.append(d_r1_b)
        deltas_full_r1.append(d_full_r1)
        if (bi + 1) % 200 == 0:
            print(f'  ...{bi + 1}/{B}')

    return {
        'seed': BOOT_SEED,
        'B': B,
        'B_used': len(deltas_r1_b),
        'B_skipped_degenerate': skipped,
        'delta_r1_minus_baseline': delta_summary(delta_r1_b_obs, deltas_r1_b),
        'delta_full_minus_r1': delta_summary(delta_full_r1_obs, deltas_full_r1),
        'pooled_oof_auc': {
            'baseline': auc_b_obs,
            'r1_trajectory': auc_r1_obs,
            'full_trajectory': auc_full_obs,
        },
    }


def main():
    summary = json.load(open(OUT_PATH))
    records, audit = load_reported_cohort_records(summary)
    if records:
        print('Loaded exact reported-cohort row-level data; recomputing debate-index and clustered bootstraps.')
        print(f'Computing 5-fold OOF probs (seed={OOF_SEED}, matching cv_auc)...')
        block = debate_index_bootstrap_from_records(records)
        summary['bootstrap_delta_auc'] = block
    else:
        print('Exact reported-cohort row-level data is unavailable; preserving existing bootstrap_delta_auc block.')
        block = summary.get('bootstrap_delta_auc', {})
        for note in audit['notes']:
            print(f'  - {note}')
    summary['bootstrap_delta_auc_clustered'] = append_clustered_bootstrap(summary, records, audit)
    json.dump(summary, open(OUT_PATH, 'w'), indent=2, default=float)
    print('\n=== bootstrap_delta_auc ===')
    print(json.dumps(block, indent=2, default=float))
    print('\n=== bootstrap_delta_auc_clustered ===')
    print(json.dumps(summary['bootstrap_delta_auc_clustered'], indent=2, default=float))
    print(f'\nUpdated {OUT_PATH}')


if __name__ == '__main__':
    main()
