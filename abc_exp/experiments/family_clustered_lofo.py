"""A1 — Family-clustered bootstrap + LOFO for headline ρ(α, C^cond).

Reviewer W1 concern: N=9 has only 4 model families; effective sample size <9.

Family map (CORRECTED from team-lead's brief):
  Anthropic: Sonnet 4.5  (singleton; Haiku 4.5 is NOT in the N=9 cohort)
  OpenAI:    GPT-4o-mini, GPT-5.4-mini
  Google:    Gemini 3-flash  (singleton)
  OSS:       Phi-4-mini, Qwen3-4B, Llama-3.1-8B, Qwen3-8B, Qwen3-32B
  (team-lead's brief omitted Qwen3-32B; verified against alpha_decomp_headline_n9.json)

Outputs:
  family_clustered_lofo.json — cluster bootstrap medians + 95% percentile CIs
                                + LOFO ρ values (4 leave-one-family fits).
"""
from __future__ import annotations
import json, os, random
from pathlib import Path

RES = Path(os.environ.get('ABC_RESULTS', Path(__file__).resolve().parent.parent / 'results'))
SEED = 20260424
B = 10000

FAMILY = {
    "Sonnet 4.5":      "Anthropic",
    "GPT-4o-mini":     "OpenAI",
    "GPT-5.4-mini":    "OpenAI",
    "Gemini 3-flash":  "Google",
    "Phi-4-mini":      "OSS",
    "Qwen3-4B":        "OSS",
    "Llama-3.1-8B":    "OSS",
    "Qwen3-8B":        "OSS",
    "Qwen3-32B":       "OSS",
}


def spearman_rho(x, y):
    """Tie-aware Spearman ρ via average ranks."""
    n = len(x)
    if n < 2:
        return float('nan')
    def avg_ranks(a):
        idx = sorted(range(n), key=lambda i: a[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[idx[j + 1]] == a[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
            i = j + 1
        return ranks
    rx, ry = avg_ranks(x), avg_ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx2 = sum((rx[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ry[i] - my) ** 2 for i in range(n))
    den = (dx2 * dy2) ** 0.5
    return num / den if den > 0 else float('nan')


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    headline = json.load(open(RES / 'alpha_decomp_headline_n9.json'))
    models = headline['models']
    alpha_total = headline['alpha_total']
    alpha_adv = headline['alpha_adv']
    C_cond = headline['C_cond']
    families = [FAMILY[m] for m in models]
    family_set = sorted(set(families))

    # Verify partition matches
    assert len(models) == 9, f"expected 9 models, got {len(models)}"
    assert len(family_set) == 4, f"expected 4 families, got {family_set}"

    # by-family index map
    fam_to_idx = {f: [i for i, ff in enumerate(families) if ff == f] for f in family_set}

    # Observed ρ on the full N=9 set
    rho_tot_obs = spearman_rho(alpha_total, C_cond)
    rho_adv_obs = spearman_rho(alpha_adv, C_cond)

    # 1. Family-clustered bootstrap (B=10,000, resample at family level)
    rng = random.Random(SEED)
    rhos_tot, rhos_adv = [], []
    skipped = 0
    for b in range(B):
        sampled_families = [rng.choice(family_set) for _ in family_set]
        idx = []
        for f in sampled_families:
            idx.extend(fam_to_idx[f])
        # Need >=2 unique values in both x and y for ρ to be defined
        xs_t = [alpha_total[i] for i in idx]
        xs_a = [alpha_adv[i] for i in idx]
        ys = [C_cond[i] for i in idx]
        if len(set(ys)) < 2 or len(set(xs_t)) < 2:
            skipped += 1
            continue
        rhos_tot.append(spearman_rho(xs_t, ys))
        rhos_adv.append(spearman_rho(xs_a, ys))

    boot = {
        'B': B, 'B_used': len(rhos_tot), 'B_skipped_degenerate': skipped, 'seed': SEED,
        'rho_alpha_total': {
            'obs_n9': rho_tot_obs,
            'median': percentile(rhos_tot, 50),
            'ci_2.5': percentile(rhos_tot, 2.5),
            'ci_97.5': percentile(rhos_tot, 97.5),
        },
        'rho_alpha_adv': {
            'obs_n9': rho_adv_obs,
            'median': percentile(rhos_adv, 50),
            'ci_2.5': percentile(rhos_adv, 2.5),
            'ci_97.5': percentile(rhos_adv, 97.5),
        },
    }

    # 2. LOFO — drop one family at a time
    lofo = []
    for drop_family in family_set:
        keep_idx = [i for i, f in enumerate(families) if f != drop_family]
        kept_models = [models[i] for i in keep_idx]
        a_t = [alpha_total[i] for i in keep_idx]
        a_a = [alpha_adv[i] for i in keep_idx]
        cc = [C_cond[i] for i in keep_idx]
        lofo.append({
            'dropped_family': drop_family,
            'dropped_models': [models[i] for i in range(9) if families[i] == drop_family],
            'n_remaining': len(keep_idx),
            'kept_models': kept_models,
            'rho_alpha_total': spearman_rho(a_t, cc),
            'rho_alpha_adv': spearman_rho(a_a, cc),
        })

    out = {
        'family_map': FAMILY,
        'family_counts': {f: len(fam_to_idx[f]) for f in family_set},
        'family_map_correction': (
            "team-lead brief listed 'Anthropic: Sonnet 4.5, Haiku 4.5' but Haiku 4.5 "
            "is NOT in the N=9 cohort; team-lead also omitted Qwen3-32B which IS in N=9. "
            "Corrected here against alpha_decomp_headline_n9.json."
        ),
        'cluster_bootstrap': boot,
        'lofo': lofo,
        'verdict_rule': "robust if all 4 LOFO ρ_alpha_total >= +0.6",
    }

    out_path = RES / 'family_clustered_lofo.json'
    json.dump(out, open(out_path, 'w'), indent=2, default=float)

    print('=== Family-clustered bootstrap ===')
    print(f"rho(αtot)  obs={rho_tot_obs:.3f}  boot median={boot['rho_alpha_total']['median']:.3f}  "
          f"95% CI=[{boot['rho_alpha_total']['ci_2.5']:.3f}, {boot['rho_alpha_total']['ci_97.5']:.3f}]")
    print(f"rho(αadv)  obs={rho_adv_obs:.3f}  boot median={boot['rho_alpha_adv']['median']:.3f}  "
          f"95% CI=[{boot['rho_alpha_adv']['ci_2.5']:.3f}, {boot['rho_alpha_adv']['ci_97.5']:.3f}]")
    print()
    print('=== LOFO (drop-one-family) ===')
    for r in lofo:
        print(f"drop {r['dropped_family']:10s} (n={r['n_remaining']})  "
              f"ρ(αtot)={r['rho_alpha_total']:+.3f}  ρ(αadv)={r['rho_alpha_adv']:+.3f}")
    min_lofo_tot = min(r['rho_alpha_total'] for r in lofo)
    print()
    print(f"LOFO min ρ(αtot) = {min_lofo_tot:+.3f}  → "
          f"{'ROBUST (≥ +0.6)' if min_lofo_tot >= 0.6 else 'FRAGILE (< +0.6)'}")
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
