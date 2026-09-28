"""E3.1 — OSS per-question S/A AUC for collapse prediction.

For each of the 5 OSS vLLM models, computes per-question (s_soc, s_arg, sa_ratio,
mean_fr) under the *default* condition from sa_causal_vllm_<m>.jsonl, then ROC-AUC
for predicting per-question collapse from debate_traces_vllm_<m>.jsonl.

Pooled OSS bakeoff parallels PAPER_RESULTS_SUMMARY.md sec 3.9.1 (closed models).
Output: abc_exp/results/oss_sa_auc.json + concise stdout.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict

RES = Path(__file__).resolve().parent.parent / 'results'

OSS = ['llama-3.1-8b', 'phi-4-mini', 'qwen3-4b', 'qwen3-8b', 'qwen3-32b']


def load_default_per_q(model: str) -> dict[str, dict]:
    rows = [json.loads(l) for l in open(RES / f'sa_causal_vllm_{model}.jsonl')]
    by_q = defaultdict(list)
    for r in rows:
        if r['condition'] == 'default':
            by_q[r['question_id']].append(r)
    out = {}
    for q, agents in by_q.items():
        soc = [pr['revised'] for a in agents for pr in a['probe_results'] if pr['social']]
        arg = [pr['revised'] for a in agents for pr in a['probe_results'] if not pr['social']]
        s = sum(soc) / len(soc) if soc else 0.0
        a = sum(arg) / len(arg) if arg else 0.0
        sa = (s - a) / (s + a) if (s + a) > 0 else 0.0
        all_pr = [pr['revised'] for ag in agents for pr in ag['probe_results']]
        mean_fr = sum(all_pr) / len(all_pr) if all_pr else 0.0
        max_fr = max((sum(pr['revised'] for pr in ag['probe_results']) / len(ag['probe_results'])) for ag in agents)
        out[q] = {'s_soc': s, 's_arg': a, 'sa_ratio': sa, 'mean_fr': mean_fr, 'max_fr': max_fr,
                  'n_agents': len(agents)}
    return out


def load_debate_per_q(model: str) -> dict[str, dict]:
    rows = [json.loads(l) for l in open(RES / f'debate_traces_vllm_{model}.jsonl')]
    out = {}
    for r in rows:
        out[r['question_id']] = {
            'collapsed': bool(r.get('collapsed', False)),
            'corrected': bool(r.get('corrected', False)),
            'init_correct': bool(r.get('initial_correct', False)),
            'final_correct': bool(r.get('final_correct', False)),
        }
    return out


def auc_roc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Mann-Whitney U based AUC."""
    n_p, n_n = len(scores_pos), len(scores_neg)
    if n_p == 0 or n_n == 0:
        return float('nan')
    combined = [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg]
    combined.sort(key=lambda x: x[0])
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k, (_, lbl) in enumerate(combined) if lbl == 1)
    u = rank_sum_pos - n_p * (n_p + 1) / 2
    return u / (n_p * n_n)


def per_model_auc(model: str) -> dict:
    sa = load_default_per_q(model)
    deb = load_debate_per_q(model)
    shared = sorted(set(sa) & set(deb))
    rows = [(sa[q], deb[q]) for q in shared]
    n_collapse = sum(1 for _, d in rows if d['collapsed'])
    out = {'model': model, 'n_questions': len(shared), 'n_collapse': n_collapse}
    if n_collapse == 0 or n_collapse == len(rows):
        out['note'] = 'degenerate label distribution; AUC undefined'
        return out
    for feat in ['s_soc', 's_arg', 'sa_ratio', 'mean_fr', 'max_fr']:
        pos = [s[feat] for s, d in rows if d['collapsed']]
        neg = [s[feat] for s, d in rows if not d['collapsed']]
        out[f'auc_{feat}'] = round(auc_roc(pos, neg), 4)
    # also pooled S/A as |sa_ratio| (since direction varies across models)
    return out


def pooled_oss(models: list[str]) -> dict:
    rows = []
    for m in models:
        sa = load_default_per_q(m)
        deb = load_debate_per_q(m)
        for q in (set(sa) & set(deb)):
            rows.append((m, sa[q], deb[q]))
    n_collapse = sum(1 for _, _, d in rows if d['collapsed'])
    out = {'pooled_oss': True, 'n_questions': len(rows), 'n_collapse': n_collapse}
    for feat in ['s_soc', 's_arg', 'sa_ratio', 'mean_fr', 'max_fr']:
        pos = [s[feat] for _, s, d in rows if d['collapsed']]
        neg = [s[feat] for _, s, d in rows if not d['collapsed']]
        out[f'auc_{feat}'] = round(auc_roc(pos, neg), 4)
    return out


def main():
    per = []
    for m in OSS:
        try:
            r = per_model_auc(m)
        except FileNotFoundError as e:
            r = {'model': m, 'note': f'missing file: {e.filename}'}
        per.append(r)
        print(json.dumps(r))
    # only pool models that have debate trace files
    avail = [m for m in OSS if (RES / f'debate_traces_vllm_{m}.jsonl').exists()]
    p = pooled_oss(avail) if avail else {}
    print('--- POOLED (avail models:', avail, ') ---')
    print(json.dumps(p))
    out_path = RES / 'oss_sa_auc.json'
    json.dump({'per_model': per, 'pooled': p, 'available_models': avail}, open(out_path, 'w'), indent=2)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
