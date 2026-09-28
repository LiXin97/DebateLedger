"""E3.3 — Continuous scaling-law fit across all available models with sa_causal data.

For each of the 16 models, compute (from sa_causal_*.jsonl default condition):
  - mean_fr_default
  - sa_ratio (mean across questions/agents)
  - selectivity = mean(FR_AA) / mean(FR_AS)  (channel-substitution proxy)

Then fit log(mean_fr) ~ log(params) and S/A ~ log(params), per-family slopes,
plus log(mean_fr) ~ initial_accuracy. Closed-model parameter counts are not
publicly available — we use censored intervals and report family-level results
without imputing point estimates for closed models.

Output: abc_exp/results/scaling_fit_16model.json.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict

RES = Path(__file__).resolve().parent.parent / 'results'

# (sa_causal_filename, display_name, family, params_billions [None=closed/unknown])
# Closed models are intentionally None — we will NOT make up param counts.
MODELS = [
    ('sa_causal_anthropic_claude-haiku.jsonl-MISSING',  'Haiku 4.5',           'Anthropic', None),  # use sa_causal_manipulation
    ('sa_causal_anthropic_claude-sonnet-4.5.jsonl',     'Sonnet 4.5',          'Anthropic', None),
    ('sa_causal_anthropic_claude-sonnet-4.6.jsonl',     'Sonnet 4.6',          'Anthropic', None),
    ('sa_causal_anthropic_claude-opus-4.5.jsonl',       'Opus 4.5',            'Anthropic', None),
    ('sa_causal_anthropic_claude-opus-4.6.jsonl',       'Opus 4.6',            'Anthropic', None),
    ('sa_causal_openai_gpt-4o-mini.jsonl',              'GPT-4o-mini',         'OpenAI',    None),
    ('sa_causal_openai_gpt-5.4-nano.jsonl',             'GPT-5.4-nano',        'OpenAI',    None),
    ('sa_causal_openai_gpt-5.4-mini.jsonl',             'GPT-5.4-mini',        'OpenAI',    None),
    ('sa_causal_openai_gpt-5.4.jsonl',                  'GPT-5.4',             'OpenAI',    None),
    ('sa_causal_gemini_3.1-flash-lite.jsonl',           'Gemini 3.1-flash-lite','Google',   None),
    ('sa_causal_gemini_3-flash.jsonl',                  'Gemini 3-flash',      'Google',    None),
    ('sa_causal_gemini_3.1-pro.jsonl',                  'Gemini 3.1-pro',      'Google',    None),
    ('sa_causal_vllm_phi-4-mini.jsonl',                 'Phi-4-mini',          'Microsoft', 3.8),
    ('sa_causal_vllm_qwen3-4b.jsonl',                   'Qwen3-4B',            'Qwen',      4.0),
    ('sa_causal_vllm_qwen3-8b.jsonl',                   'Qwen3-8B',            'Qwen',      8.0),
    ('sa_causal_vllm_llama-3.1-8b.jsonl',               'Llama-3.1-8B',        'Meta',      8.0),
    ('sa_causal_vllm_qwen3-32b.jsonl',                  'Qwen3-32B',           'Qwen',      32.0),
]


def fr_summary(rows):
    """Per row: aggregate flip_rate across agents, separate by condition."""
    by_cond = defaultdict(list)
    init_correct = []
    for r in rows:
        by_cond[r['condition']].append(r['flip_rate'])
        if r['condition'] == 'default':
            init_correct.append(int(bool(r['initial_correct'])))
    out = {}
    for c, vs in by_cond.items():
        out[c] = sum(vs) / len(vs) if vs else None
    out['init_acc'] = sum(init_correct) / len(init_correct) if init_correct else None
    out['n_rows'] = len(rows)
    # social vs argument
    soc, arg = [], []
    for r in rows:
        if r['condition'] != 'default':
            continue
        for pr in r['probe_results']:
            (soc if pr['social'] else arg).append(int(pr['revised']))
    s = sum(soc)/len(soc) if soc else 0.0
    a = sum(arg)/len(arg) if arg else 0.0
    out['s_soc'] = s
    out['s_arg'] = a
    out['sa_ratio'] = (s - a) / (s + a) if (s + a) > 0 else 0.0
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx*dy > 0 else None


def spearman(xs, ys):
    def ranks(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        i = 0
        while i < len(s):
            j = i
            while j+1 < len(s) and v[s[j+1]] == v[s[i]]:
                j += 1
            avg = (i+j)/2 + 1
            for k in range(i, j+1):
                r[s[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def linfit(xs, ys):
    n = len(xs)
    if n < 3: return None
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = sum((x-mx)**2 for x in xs)
    if den == 0: return None
    slope = num/den
    intercept = my - slope*mx
    yhat = [slope*x + intercept for x in xs]
    ss_res = sum((y-yh)**2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y-my)**2 for y in ys)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else None
    return {'slope': round(slope, 4), 'intercept': round(intercept, 4), 'r2': round(r2, 4) if r2 is not None else None, 'n': n}


def main():
    per = []
    for fn, name, family, params in MODELS:
        path = RES / fn
        if not path.exists():
            per.append({'name': name, 'family': family, 'params_b': params, 'note': 'sa_causal file missing'})
            continue
        rows = [json.loads(l) for l in open(path)]
        s = fr_summary(rows)
        s.update({'name': name, 'family': family, 'params_b': params})
        per.append(s)

    # OSS-only scaling fit (params known)
    oss = [m for m in per if m.get('params_b') is not None and 'default' in m]
    print('=== OSS scaling (N=%d) ===' % len(oss))
    log_p = [math.log10(m['params_b']) for m in oss]
    fr = [m['default'] for m in oss]
    sa = [m['sa_ratio'] for m in oss]
    init = [m['init_acc'] for m in oss]
    fits = {
        'log_FR_vs_log_params': linfit(log_p, [math.log10(max(f, 1e-3)) for f in fr]),
        'SA_vs_log_params': linfit(log_p, sa),
        'init_acc_vs_log_params': linfit(log_p, init),
        'spearman_FR_vs_params': spearman(log_p, fr),
        'spearman_SA_vs_params': spearman(log_p, sa),
        'spearman_initacc_vs_params': spearman(log_p, init),
    }
    print(json.dumps(fits, indent=2))

    # All-model fits using initial accuracy as the capability proxy
    valid = [m for m in per if 'default' in m and m.get('init_acc') is not None]
    print(f'\n=== All-model with init_acc as capability proxy (N={len(valid)}) ===')
    init_all = [m['init_acc'] for m in valid]
    fr_all = [m['default'] for m in valid]
    sa_all = [m['sa_ratio'] for m in valid]
    capfits = {
        'log_FR_vs_init_acc': linfit(init_all, [math.log10(max(f, 1e-3)) for f in fr_all]),
        'SA_vs_init_acc': linfit(init_all, sa_all),
        'spearman_FR_vs_initacc': spearman(init_all, fr_all),
        'spearman_SA_vs_initacc': spearman(init_all, sa_all),
    }
    print(json.dumps(capfits, indent=2))

    # Per-family means + slopes (SA vs init_acc within family)
    by_fam = defaultdict(list)
    for m in valid:
        by_fam[m['family']].append(m)
    family_summary = {}
    for fam, ms in by_fam.items():
        if len(ms) >= 2:
            xs = [m['init_acc'] for m in ms]
            ys = [m['sa_ratio'] for m in ms]
            family_summary[fam] = {
                'n': len(ms),
                'mean_FR': round(sum(m['default'] for m in ms)/len(ms), 4),
                'mean_SA': round(sum(m['sa_ratio'] for m in ms)/len(ms), 4),
                'mean_init_acc': round(sum(xs)/len(xs), 4),
                'fit_SA_vs_initacc': linfit(xs, ys),
            }
        else:
            family_summary[fam] = {'n': len(ms), 'note': 'too few for fit',
                                   'mean_FR': round(ms[0]['default'], 4),
                                   'mean_SA': round(ms[0]['sa_ratio'], 4),
                                   'mean_init_acc': round(ms[0]['init_acc'], 4)}
    print('\n=== Per-family summary ===')
    print(json.dumps(family_summary, indent=2))

    out = {'per_model': per, 'oss_fits': fits, 'allmodel_fits': capfits, 'family_summary': family_summary}
    out_path = RES / 'scaling_fit_16model.json'
    json.dump(out, open(out_path, 'w'), indent=2)
    print('\nwrote', out_path)
    # Compact table
    print('\n=== Compact table ===')
    print(f"{'model':22s} {'fam':10s} {'p_b':>6s} {'init':>6s} {'FR_d':>6s} {'S/A':>7s}")
    for m in per:
        if 'default' not in m: continue
        p = '?' if m['params_b'] is None else f"{m['params_b']:.1f}"
        print(f"{m['name']:22s} {m['family']:10s} {p:>6s} {m['init_acc']:.3f} {m['default']:.3f} {m['sa_ratio']:>+7.3f}")


if __name__ == '__main__':
    main()
