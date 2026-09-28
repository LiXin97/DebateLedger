"""Build the per-debate R1-feature matrix for prof-statistician's E6 Bayesian MLM.

Reads all `debate_traces_*.jsonl`, emits one row per debate with:
  - identifiers: model, question_id, correct_label
  - outcome: collapsed (0/1), corrected (0/1), final_correct (0/1)
  - pre-debate features: init_correct, init_unanimous, init_agree_frac,
    init_majority_correct
  - R1 features: r1_majority_changed, r1_n_flipped_from_init, r1_agree_frac,
    r1_majority_correct, r1_disagree_with_correct (n_agents whose R1 != correct)
  - R2/R3 features (where round_traces extends): r2_*, r3_*
  - cascade_round: 1/2/3 (NATURAL numbering: R1 = first debate round) or null
    if not collapsed or unlocalizable

Output: abc_exp/results/per_debate_r1_features.jsonl + .csv (for R/Stan/PyMC).
"""
from __future__ import annotations
import json, glob, csv, os
from pathlib import Path
from collections import Counter

RES = Path(os.environ.get('ABC_RESULTS', Path(__file__).resolve().parent.parent / 'results'))


def round_majority_correct(answers, correct):
    if not answers:
        return 0
    cnt = Counter(answers)
    top = max(cnt.values())
    winners = sorted([a for a, c in cnt.items() if c == top])
    return int(winners[0] == correct)


def featurize(row):
    init_answers = row.get('initial_answers') or []
    correct = row.get('correct_label')
    init_maj = row.get('initial_majority')
    init_correct = int(bool(row.get('initial_correct')))

    rt = row.get('round_traces') or []
    if not rt:
        return None

    init_unanimous = int(len(set(a for a in init_answers if a)) <= 1)
    init_agree_frac = (max(Counter(init_answers).values()) / max(len(init_answers), 1)) if init_answers else 0.0

    feats = {
        'model': row.get('model', ''),
        'question_id': row.get('question_id', ''),
        'correct_label': correct,
        'collapsed': int(bool(row.get('collapsed', False))),
        'corrected': int(bool(row.get('corrected', False))),
        'final_correct': int(bool(row.get('final_correct', False))),
        'init_correct': init_correct,
        'init_unanimous': init_unanimous,
        'init_agree_frac': init_agree_frac,
        'init_majority_correct': int(init_maj == correct),
    }

    r1 = rt[0]
    r1_answers = r1['answers']
    feats['r1_majority_changed'] = int(r1['majority'] != init_maj)
    feats['r1_n_flipped_from_init'] = sum(
        1 for a, b in zip(init_answers, r1_answers) if a and b and a != b
    )
    feats['r1_agree_frac'] = (max(Counter(r1_answers).values()) / max(len(r1_answers), 1)) if r1_answers else 0.0
    feats['r1_majority_correct'] = int(r1['majority'] == correct)
    feats['r1_n_disagree_with_correct'] = sum(1 for a in r1_answers if a != correct)

    if len(rt) >= 2:
        r2 = rt[1]
        feats['r2_majority_changed'] = int(r2['majority'] != r1['majority'])
        feats['r2_agree_frac'] = (max(Counter(r2['answers']).values()) / max(len(r2['answers']), 1))
        feats['r2_majority_correct'] = int(r2['majority'] == correct)
    else:
        feats['r2_majority_changed'] = None
        feats['r2_agree_frac'] = None
        feats['r2_majority_correct'] = None

    if len(rt) >= 3:
        r3 = rt[2]
        feats['r3_majority_changed'] = int(r3['majority'] != rt[1]['majority'])
        feats['r3_agree_frac'] = (max(Counter(r3['answers']).values()) / max(len(r3['answers']), 1))
        feats['r3_majority_correct'] = int(r3['majority'] == correct)
    else:
        feats['r3_majority_changed'] = None
        feats['r3_agree_frac'] = None
        feats['r3_majority_correct'] = None

    cr = None
    if feats['collapsed'] and feats['init_correct']:
        for tr in rt:
            if tr['majority'] != correct:
                cr = tr['round']
                break
    feats['cascade_round'] = cr

    return feats


def main():
    files = sorted(glob.glob(str(RES / 'debate_traces_*.jsonl')))
    out_jsonl = RES / 'per_debate_r1_features.jsonl'
    out_csv = RES / 'per_debate_r1_features.csv'
    rows = []
    n_files = 0
    for f in files:
        n_files += 1
        model_tag = Path(f).stem.replace('debate_traces_', '')
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r.setdefault('model', model_tag)
            feats = featurize(r)
            if feats is not None:
                rows.append(feats)
    with open(out_jsonl, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    keys = list(rows[0].keys()) if rows else []
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    by_model = Counter(r['model'] for r in rows)
    n_collapsed = sum(r['collapsed'] for r in rows)
    n_corrected = sum(r['corrected'] for r in rows)
    print(f'Wrote {len(rows)} rows from {n_files} files')
    print(f'  collapsed={n_collapsed}  corrected={n_corrected}')
    print(f'  per-model: {dict(by_model)}')
    print(f'  jsonl: {out_jsonl}')
    print(f'  csv:   {out_csv}')


if __name__ == '__main__':
    main()
