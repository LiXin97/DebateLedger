"""E3.2 — Post-hoc hybrid policy replay on OSS debate traces.

Two policy variants are simulated:

1. **freeze_at_r2_if_majority_changed** (D2-v2 framing A's "freeze-at-R2" idea):
   if Round-1 majority differs from initial majority, freeze each agent to its
   initial answer. Pure debate-trace replay (uses round_traces[0]).

2. **shield_high_revisability** (question-level shielding by probe mean_fr):
   for each question, compute mean_fr from sa_causal default condition. If
   mean_fr > tau, freeze to initial majority (skip debate). Threshold tau swept
   from 0.0 to 1.0 in 0.05 steps; report tau* maximizing test ACC and tau* by
   best collapse-suppression-with-acc-non-decreasing rule.

Note: per-agent shielded replay (the paper's `deployable_hybrid` rule) is NOT
possible because OSS sa_causal initial_answers were drawn independently of the
debate run (different temperature samples). We document this and use the two
question-level / round-level proxies above instead — both faithful to the
"prevent harmful late-cascade revisions" intuition.

Output: abc_exp/results/oss_hybrid_replay.json.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / 'results'
OSS = ['llama-3.1-8b', 'phi-4-mini', 'qwen3-4b', 'qwen3-8b']  # qwen3-32b: no debate trace yet


def majority(answers):
    valid = [a for a in answers if a]
    if not valid:
        return ''
    cnt = Counter(valid)
    m = max(cnt.values())
    return sorted(a for a, c in cnt.items() if c == m)[0]


def standard_outcome(row):
    return {'final': row['final_answer'], 'correct': row['final_correct'],
            'collapsed': row.get('collapsed', False),
            'corrected': row.get('corrected', False)}


def freeze_at_r2_if_majority_changed(row):
    """Freeze if R1 majority differs from initial majority — i.e., suspect cascade."""
    init_maj = row['initial_majority']
    if not row['round_traces']:
        return standard_outcome(row)
    r1 = row['round_traces'][0]
    if r1['majority'] != init_maj:
        # freeze: take initial majority as final
        final = init_maj
        correct = (final == row['correct_label'])
        # collapsed = init was right, frozen final is wrong (impossible if frozen to init when init was right)
        # corrected = init wrong, final right
        init_correct = row['initial_correct']
        collapsed = init_correct and not correct  # cannot happen with freeze-to-init
        corrected = (not init_correct) and correct  # also cannot happen
        return {'final': final, 'correct': correct, 'collapsed': collapsed, 'corrected': corrected,
                'fired': True}
    else:
        out = standard_outcome(row)
        out['fired'] = False
        return out


def shield_by_probe_fr(row, mean_fr, tau):
    """Shield (revert to initial majority) if question's probe-derived mean_fr > tau."""
    if mean_fr is not None and mean_fr > tau:
        final = row['initial_majority']
        correct = (final == row['correct_label'])
        return {'final': final, 'correct': correct,
                'collapsed': False, 'corrected': False, 'fired': True}
    out = standard_outcome(row)
    out['fired'] = False
    return out


def load_per_q_meanfr(model):
    rows = [json.loads(l) for l in open(RES / f'sa_causal_vllm_{model}.jsonl')]
    by_q = defaultdict(list)
    for r in rows:
        if r['condition'] == 'default':
            for ag in [r]:  # already row-per-agent
                by_q[r['question_id']].append(ag['flip_rate'])
    return {q: sum(v) / len(v) for q, v in by_q.items()}


def policy_summary(rows, name):
    n = len(rows)
    if n == 0:
        return {'policy': name, 'n': 0}
    acc = sum(r['correct'] for r in rows) / n
    coll = sum(r.get('collapsed', False) for r in rows) / n
    corr = sum(r.get('corrected', False) for r in rows) / n
    fired = sum(r.get('fired', False) for r in rows) / n if 'fired' in rows[0] else None
    return {'policy': name, 'n': n, 'acc': round(acc, 4), 'collapse_rate': round(coll, 4),
            'correction_rate': round(corr, 4), 'fired_rate': None if fired is None else round(fired, 4)}


def run_for_model(model):
    deb = [json.loads(l) for l in open(RES / f'debate_traces_vllm_{model}.jsonl')]
    mfr = load_per_q_meanfr(model)
    out = {'model': model, 'n_debates': len(deb)}
    # Standard
    std = [standard_outcome(r) for r in deb]
    out['standard'] = policy_summary(std, 'standard')
    # Freeze-at-R2
    fr2 = [freeze_at_r2_if_majority_changed(r) for r in deb]
    out['freeze_at_r2'] = policy_summary(fr2, 'freeze_at_r2_if_majority_changed')
    # Shield sweep
    sweep = []
    for tau in [round(x*0.05, 2) for x in range(0, 21)]:
        sh = [shield_by_probe_fr(r, mfr.get(r['question_id']), tau) for r in deb]
        s = policy_summary(sh, f'shield_tau={tau}')
        s['tau'] = tau
        sweep.append(s)
    # report best by acc and by collapse-suppression-with-acc>=std
    best_acc = max(sweep, key=lambda s: s['acc'])
    std_acc = out['standard']['acc']
    feasible = [s for s in sweep if s['acc'] >= std_acc - 0.005]
    best_safe = min(feasible, key=lambda s: s['collapse_rate']) if feasible else None
    out['shield_sweep'] = sweep
    out['shield_best_acc'] = best_acc
    out['shield_best_safe'] = best_safe
    return out


def main():
    results = {'models': {}}
    for m in OSS:
        try:
            results['models'][m] = run_for_model(m)
        except FileNotFoundError as e:
            results['models'][m] = {'error': f'missing {e.filename}'}
    # Pooled across OSS
    pooled = {'standard': {'tp': 0, 'coll': 0, 'corr': 0, 'n': 0},
              'freeze_at_r2': {'tp': 0, 'coll': 0, 'corr': 0, 'n': 0, 'fired': 0}}
    for m, d in results['models'].items():
        if 'error' in d: continue
        deb = [json.loads(l) for l in open(RES / f'debate_traces_vllm_{m}.jsonl')]
        std = [standard_outcome(r) for r in deb]
        fr2 = [freeze_at_r2_if_majority_changed(r) for r in deb]
        for label, rows in [('standard', std), ('freeze_at_r2', fr2)]:
            pooled[label]['n'] += len(rows)
            pooled[label]['tp'] += sum(r['correct'] for r in rows)
            pooled[label]['coll'] += sum(r.get('collapsed', False) for r in rows)
            pooled[label]['corr'] += sum(r.get('corrected', False) for r in rows)
            if label == 'freeze_at_r2':
                pooled[label]['fired'] += sum(r.get('fired', False) for r in rows)
    for label, p in pooled.items():
        n = p['n'] or 1
        p['acc'] = round(p['tp']/n, 4)
        p['collapse_rate'] = round(p['coll']/n, 4)
        p['correction_rate'] = round(p['corr']/n, 4)
        if 'fired' in p:
            p['fired_rate'] = round(p['fired']/n, 4)
    results['pooled'] = pooled
    out_path = RES / 'oss_hybrid_replay.json'
    json.dump(results, open(out_path, 'w'), indent=2)
    # Print compact summary
    print('=== Per-model summary ===')
    for m, d in results['models'].items():
        if 'error' in d:
            print(m, d); continue
        std = d['standard']; fr = d['freeze_at_r2']; sb = d.get('shield_best_safe')
        print(f"{m:14s} N={d['n_debates']:4d} | std acc={std['acc']:.3f} coll={std['collapse_rate']:.3f} | "
              f"freeze@R2 acc={fr['acc']:.3f} coll={fr['collapse_rate']:.3f} fired={fr['fired_rate']:.3f}")
        if sb:
            print(f"                  shield_best_safe tau={sb['tau']} acc={sb['acc']:.3f} coll={sb['collapse_rate']:.3f} fired={sb['fired_rate']:.3f}")
    print()
    print('=== Pooled OSS ===')
    print(json.dumps(pooled, indent=2))
    print('wrote', out_path)


if __name__ == '__main__':
    main()
