"""T2.1 robustness check: drop the very-strong probe (which contains authority cues:
'authoritative references and expert consensus') and recompute α / α_adv / α_cor and
the headline Spearman ρ. This responds to the reviewer concern that very-strong's
'authoritative references' phrasing leaks an auxiliary social/authority cue beyond
pure argument strength, potentially inflating α_adv on models that defer to authority.

Output: results/T2_1_drop_very_strong.json + appendix-ready table.

Pre-registered (PRE_REGISTRATION_N18.md §6.1).
"""
from __future__ import annotations
import json
from pathlib import Path
import glob
import numpy as np
from scipy.stats import spearmanr, norm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

NAME_MAP_FILES = {
    "Sonnet 4.5": "sa_causal_anthropic_claude-sonnet-4.5.jsonl",
    "GPT-4o-mini": "sa_causal_openai_gpt-4o-mini.jsonl",
    "GPT-5.4-mini": "sa_causal_openai_gpt-5.4-mini.jsonl",
    "Gemini 3-flash": "sa_causal_gemini_3-flash.jsonl",
    "Phi-4-mini": "sa_causal_vllm_phi-4-mini.jsonl",
    "Qwen3-4B": "sa_causal_vllm_qwen3-4b.jsonl",
    "Llama-3.1-8B": "sa_causal_vllm_llama-3.1-8b.jsonl",
    "Qwen3-8B": "sa_causal_vllm_qwen3-8b.jsonl",
    "Qwen3-32B": "sa_causal_vllm_qwen3-32b.jsonl",
}

DROP_LEVELS = {"very_strong"}


def alpha_decomp(rows, drop_levels=frozenset()):
    """Compute total α, α_adv, α_cor across the (default) condition.
    Restrict to non-social probes for the headline alpha definition;
    return same triple but excluding any strength in drop_levels.
    """
    n_total = n_adv_den = n_adv_num = n_cor_den = n_cor_num = 0
    for r in rows:
        if r.get("condition") != "default":
            continue
        is_correct = bool(r["initial_correct"])
        for p in r["probe_results"]:
            if p["strength"] in drop_levels:
                continue
            if p["social"]:
                continue  # non-social arm only, matches headline α
            n_total += 1
            revised = bool(p["revised"])
            if is_correct:
                n_adv_den += 1
                if revised:
                    n_adv_num += 1
            else:
                n_cor_den += 1
                if revised:
                    n_cor_num += 1
    flips_total = n_adv_num + n_cor_num
    return dict(
        alpha_total=flips_total / max(n_total, 1),
        alpha_adv=n_adv_num / max(n_adv_den, 1) if n_adv_den else float("nan"),
        alpha_cor=n_cor_num / max(n_cor_den, 1) if n_cor_den else float("nan"),
        n_probe=n_total,
        n_adv=n_adv_den,
        n_cor=n_cor_den,
    )


cond = {t["model"]: t for t in json.load(open(RES / "conditional_collapse_table.json"))["table"]}
NAME_MAP_COND = {
    "Sonnet 4.5": "Claude Sonnet 4.5",
    "GPT-4o-mini": "GPT-4o-mini",
    "GPT-5.4-mini": "GPT-5.4-mini",
    "Gemini 3-flash": "Gemini 3-flash",
    "Phi-4-mini": "Phi-4-mini",
    "Qwen3-4B": "Qwen3-4B",
    "Llama-3.1-8B": "Llama-3.1-8B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-32B": "Qwen3-32B",
}

per_model = {}
for model, fname in NAME_MAP_FILES.items():
    fp = RES / fname
    if not fp.exists():
        print(f"missing {fname}, skip {model}")
        continue
    rows = [json.loads(l) for l in open(fp)]
    orig = alpha_decomp(rows)
    drop = alpha_decomp(rows, drop_levels=DROP_LEVELS)
    per_model[model] = dict(orig=orig, drop_vs=drop)


def fisherz_ci(rho, n, ci=0.95):
    if abs(rho) >= 1 or n < 4:
        return (float("nan"), float("nan"))
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    z_lo, z_hi = z - norm.ppf((1 + ci) / 2) * se, z + norm.ppf((1 + ci) / 2) * se
    return float(np.tanh(z_lo)), float(np.tanh(z_hi))


def report_rho(label, alpha_dict_key):
    pairs = []
    for model, d in per_model.items():
        cname = NAME_MAP_COND[model]
        if cname not in cond:
            continue
        a = d[alpha_dict_key]["alpha_total"]
        c = cond[cname]["cond_collapse_pct"]
        pairs.append((model, a, c))
    pairs.sort(key=lambda x: x[1])
    a_vals = np.array([p[1] for p in pairs])
    c_vals = np.array([p[2] for p in pairs])
    rho, p = spearmanr(a_vals, c_vals)
    lo, hi = fisherz_ci(rho, len(pairs))
    print(f"\n=== {label}: N={len(pairs)} ===")
    for m, a, c in pairs:
        print(f"  {m:<20} α={a:.3f}  C^cond={c:.2f}%")
    print(f"  Spearman ρ = {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")
    return dict(rho=float(rho), ci_lo=lo, ci_hi=hi, p=float(p), n=len(pairs))


orig_summary = report_rho("Original (8 probes, very_strong INCLUDED)", "orig")
drop_summary = report_rho("Robustness (6 probes, very_strong DROPPED)", "drop_vs")

# Per-model α_adv comparison
print("\nPer-model α / α_adv comparison:")
print(f"{'Model':<20} α_orig  α_drop  Δα     α_adv_orig  α_adv_drop")
for model, d in per_model.items():
    print(
        f"{model:<20} {d['orig']['alpha_total']:.3f}  {d['drop_vs']['alpha_total']:.3f}  "
        f"{d['drop_vs']['alpha_total']-d['orig']['alpha_total']:+.3f}  "
        f"{d['orig']['alpha_adv']:.3f}      {d['drop_vs']['alpha_adv']:.3f}"
    )

out = dict(
    description="T2.1 sensitivity: rerun headline ρ after dropping very_strong probe.",
    drop_levels=list(DROP_LEVELS),
    per_model={m: {k: v for k, v in d.items()} for m, d in per_model.items()},
    headline_original=orig_summary,
    headline_dropped=drop_summary,
)
out_path = RES / "T2_1_drop_very_strong.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nWrote {out_path}")

# Markdown for appendix
md = [
    "# T2.1 Robustness: Drop very-strong probe",
    "",
    "Pre-registered (PRE_REGISTRATION_N18.md §6.1). The very-strong probe contains",
    "the phrase 'According to authoritative references and expert consensus', which",
    "may leak an authority/social cue beyond pure argument strength. We rerun the",
    "headline α / C^cond Spearman after restricting to the 6-probe set",
    "(weak / moderate / strong x social/non-social), dropping the 2 very-strong probes.",
    "",
    "## Headline ρ comparison",
    "",
    f"- **Original (8 probes, very_strong included):** ρ = {orig_summary['rho']:+.3f},",
    f"  95% CI [{orig_summary['ci_lo']:+.3f}, {orig_summary['ci_hi']:+.3f}], p = {orig_summary['p']:.3f}, N = {orig_summary['n']}.",
    f"- **Robustness (6 probes, very_strong dropped):** ρ = {drop_summary['rho']:+.3f},",
    f"  95% CI [{drop_summary['ci_lo']:+.3f}, {drop_summary['ci_hi']:+.3f}], p = {drop_summary['p']:.3f}, N = {drop_summary['n']}.",
    "",
    "## Per-model α_total under the two probe sets",
    "",
    "| Model | α (8-probe) | α (6-probe, no very_strong) | Δα |",
    "|---|---|---|---|",
]
for model, d in per_model.items():
    a0 = d["orig"]["alpha_total"]
    a1 = d["drop_vs"]["alpha_total"]
    md.append(f"| {model} | {a0:.3f} | {a1:.3f} | {a1-a0:+.3f} |")

md_path = RES / "T2_1_DROP_VERY_STRONG.md"
md_path.write_text("\n".join(md) + "\n")
print(f"Wrote {md_path}")
