"""T2.4 robustness: log-ratio variant of S/A normalization.

Pre-registered (PRE_REGISTRATION_N18.md §6.4): replace
  S/A_orig = s_soc / (|s_soc| + |s_arg| + eps)
with
  S/A_log = log((s_soc + eps) / (s_arg + eps))
and verify the per-model rank order is preserved.

Output: results/T2_4_log_ratio.json + appendix-ready table.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

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

EPS = 1e-3


def per_model_sa(rows):
    """Average per-question S and A across the agents in a model file."""
    s_vals, a_vals = [], []
    for r in rows:
        if r.get("condition") != "default":
            continue
        s = r.get("social_sensitivity")
        a = r.get("argument_sensitivity")
        if s is None or a is None:
            continue
        s_vals.append(float(s))
        a_vals.append(float(a))
    s_mean = float(np.mean(s_vals))
    a_mean = float(np.mean(a_vals))
    sa_orig = s_mean / (abs(s_mean) + abs(a_mean) + EPS)
    sa_log = float(np.log((max(s_mean, 0) + EPS) / (max(a_mean, 0) + EPS)))
    return dict(s=s_mean, a=a_mean, sa_orig=sa_orig, sa_log=sa_log)


per_model = {}
for model, fname in NAME_MAP_FILES.items():
    fp = RES / fname
    if not fp.exists():
        continue
    rows = [json.loads(l) for l in open(fp)]
    per_model[model] = per_model_sa(rows)

print(f"{'Model':<20} {'S':>8} {'A':>8} {'SA_orig':>10} {'SA_log':>10}")
for m, d in per_model.items():
    print(f"{m:<20} {d['s']:>+8.3f} {d['a']:>+8.3f} {d['sa_orig']:>+10.3f} {d['sa_log']:>+10.3f}")

# Rank correlation between SA_orig and SA_log across the matched cohort
sa_orig_vals = np.array([d["sa_orig"] for d in per_model.values()])
sa_log_vals = np.array([d["sa_log"] for d in per_model.values()])
rho, p = spearmanr(sa_orig_vals, sa_log_vals)
print(f"\nSpearman rank consistency between SA_orig and SA_log: ρ = {rho:+.3f} (p = {p:.3f})")

out = dict(
    description="T2.4 sensitivity: log-ratio variant of S/A normalization.",
    per_model=per_model,
    sa_norm_rank_consistency=dict(rho=float(rho), p=float(p), n=len(per_model)),
)
out_path = RES / "T2_4_log_ratio.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"Wrote {out_path}")

md = [
    "# T2.4 Robustness: Log-ratio S/A normalization",
    "",
    "Pre-registered (PRE_REGISTRATION_N18.md §6.4). The headline S/A normalization",
    "uses `s_soc / (|s_soc| + |s_arg| + eps)`; we report a log-ratio variant",
    "`log((s_soc + eps) / (s_arg + eps))` for completeness.",
    "",
    "| Model | S | A | S/A (orig) | S/A (log) |",
    "|---|---|---|---|---|",
]
for m, d in per_model.items():
    md.append(
        f"| {m} | {d['s']:+.3f} | {d['a']:+.3f} | {d['sa_orig']:+.3f} | {d['sa_log']:+.3f} |"
    )
md.append("")
md.append(
    f"Spearman rank consistency across the {len(per_model)} models: ρ = {rho:+.3f} (p = {p:.3f})."
)
md_path = RES / "T2_4_LOG_RATIO.md"
md_path.write_text("\n".join(md) + "\n")
print(f"Wrote {md_path}")
