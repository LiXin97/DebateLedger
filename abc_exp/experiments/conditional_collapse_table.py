"""Compute conditional collapse rate P(final wrong | initial majority correct)
for all debated models, with Wilson 95% CIs and Spearman tests against revision quantity.

Outputs:
  - results/conditional_collapse_table.json  (machine-readable)
  - results/CONDITIONAL_COLLAPSE_TABLE.md    (paper-ready table)

Run: python -m abc_exp.experiments.conditional_collapse_table
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from scipy.stats import binomtest, spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    ci = binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="wilson")
    return p, ci.low, ci.high


# Each entry: (display name, list of (path, extractor_fn))
# extractor_fn returns (initial_correct: bool, final_correct: bool, collapsed: bool)
def std_extractor(d):
    return bool(d.get("initial_correct")), bool(d.get("final_correct")), bool(d.get("collapsed"))


def adaptive_extractor(d):
    """For adaptive_debate_*.jsonl use 'standard_*' fields if available, else 'shielded_*'."""
    ic = bool(d.get("initial_correct"))
    if "standard_correct" in d:
        fc = bool(d.get("standard_correct"))
        coll = ic and not fc
    else:
        fc = bool(d.get("shielded_correct"))
        coll = ic and not fc
    return ic, fc, coll


def block1_extractor(d):
    sd = d.get("standard_debate", {})
    ic = bool(sd.get("initial_correct"))
    fc = bool(sd.get("final_correct"))
    return ic, fc, ic and not fc


def multimodel_filter_extractor(model_match):
    def fn(d):
        if d.get("model") != model_match:
            return None
        return std_extractor(d)
    return fn


# Source map: (display_name, family, debate_FR_from_table, S/A_from_table, source_file, extractor_fn)
SOURCES = [
    ("Claude Sonnet 4.5", "Anthropic", 0.705, 0.15,
     "adaptive_debate_sonnet.jsonl", adaptive_extractor),
    ("Claude Haiku 4.5", "Anthropic", 0.512, -0.03,
     "multimodel_debates_mmlu_pro.jsonl", multimodel_filter_extractor("anthropic/claude-haiku")),
    ("GPT-4o-mini", "OpenAI", 0.458, 0.05,
     "multimodel_debates_mmlu_pro.jsonl", multimodel_filter_extractor("openai/gpt-4o-mini")),
    ("Gemini 2.5 Flash", "Google", 0.648, 0.14,
     "block1_abc_vs_baselines.jsonl", block1_extractor),
    ("GPT-5.4-mini", "OpenAI", 0.350, 0.05,
     "debate_traces_openai_gpt-5.4-mini.jsonl", std_extractor),
    ("Gemini 3-flash", "Google", 0.249, -0.08,
     "debate_traces_gemini_3-flash.jsonl", std_extractor),
    ("Phi-4-mini", "Microsoft", 0.451, 0.23,
     "debate_traces_vllm_phi-4-mini.jsonl", std_extractor),
    ("Qwen3-4B", "Qwen", 0.596, 0.08,
     "debate_traces_vllm_qwen3-4b.jsonl", std_extractor),
    ("Llama-3.1-8B", "Meta", 0.726, 0.06,
     "debate_traces_vllm_llama-3.1-8b.jsonl", std_extractor),
    ("Qwen3-8B", "Qwen", 0.596, -0.05,
     "debate_traces_vllm_qwen3-8b.jsonl", std_extractor),
    ("Qwen3-32B", "Qwen", 0.293, 0.24,
     "multimodel_debates_mmlu_pro.jsonl", multimodel_filter_extractor("vllm/qwen3-32b")),
]


def main():
    table = []
    for name, fam, fr, sa, fn, extractor in SOURCES:
        path = RES / fn
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        rows = load_jsonl(path)
        # Apply extractor; skip None for filtered files
        triples = []
        for d in rows:
            res = extractor(d)
            if res is None:
                continue
            triples.append(res)
        n = len(triples)
        n_init_c = sum(1 for ic, _, _ in triples if ic)
        n_init_w = n - n_init_c
        n_cond_coll = sum(1 for ic, fc, _ in triples if ic and not fc)
        n_recover = sum(1 for ic, fc, _ in triples if (not ic) and fc)
        n_raw_coll = sum(1 for _, _, c in triples if c)

        raw_p, raw_lo, raw_hi = wilson_ci(n_raw_coll, n)
        cond_p, cond_lo, cond_hi = wilson_ci(n_cond_coll, n_init_c)
        rec_p, rec_lo, rec_hi = wilson_ci(n_recover, n_init_w)
        # G/B = corrections / collapses (use cond definition: corrections of init-wrong, collapses of init-right)
        gb = n_recover / max(n_cond_coll, 1)

        table.append({
            "model": name, "family": fam, "FR": fr, "SA": sa,
            "n_deb": n, "n_init_correct": n_init_c, "n_init_wrong": n_init_w,
            "raw_collapse_pct": raw_p * 100, "raw_lo": raw_lo * 100, "raw_hi": raw_hi * 100,
            "cond_collapse_pct": cond_p * 100, "cond_lo": cond_lo * 100, "cond_hi": cond_hi * 100,
            "init_acc": n_init_c / n,
            "recover_pct": rec_p * 100, "recover_lo": rec_lo * 100, "recover_hi": rec_hi * 100,
            "n_cond_coll": n_cond_coll, "n_recover": n_recover,
            "GB_ratio": gb,
        })

    # Cross-model rank tests
    fr_v = [t["FR"] for t in table]
    sa_v = [t["SA"] for t in table]
    raw_v = [t["raw_collapse_pct"] for t in table]
    cond_v = [t["cond_collapse_pct"] for t in table]
    rec_v = [t["recover_pct"] for t in table]
    init_v = [t["init_acc"] for t in table]

    stats = {
        "N_models": len(table),
        "spearman_FR_raw": spearmanr(fr_v, raw_v)._asdict(),
        "spearman_FR_cond": spearmanr(fr_v, cond_v)._asdict(),
        "spearman_FR_recover": spearmanr(fr_v, rec_v)._asdict(),
        "spearman_SA_cond": spearmanr(sa_v, cond_v)._asdict(),
        "kendall_FR_cond": kendalltau(fr_v, cond_v)._asdict(),
        "spearman_init_acc_cond": spearmanr(init_v, cond_v)._asdict(),
        "spearman_init_acc_raw": spearmanr(init_v, raw_v)._asdict(),
    }

    # Convert SignificanceResult to dict
    def fix(x):
        if hasattr(x, "_asdict"):
            d = x._asdict()
            return {k: float(v) if hasattr(v, "item") else v for k, v in d.items()}
        return x

    # Already done via _asdict()
    out = {"table": table, "stats": {k: {kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in stats.items()}}
    out_path = RES / "conditional_collapse_table.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Markdown table
    md_lines = ["# Conditional Collapse Table", "",
                f"N models = {len(table)}", "",
                "| Model | Family | N_deb | InitAcc | FR | S/A | Raw Coll % [95% CI] | Cond Coll % [95% CI] | Recover % | G/B |",
                "|---|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        md_lines.append(
            f"| {t['model']} | {t['family']} | {t['n_deb']} | {t['init_acc']*100:.1f}% | {t['FR']:.3f} | {t['SA']:+.2f} | "
            f"{t['raw_collapse_pct']:.2f} [{t['raw_lo']:.2f}, {t['raw_hi']:.2f}] | "
            f"{t['cond_collapse_pct']:.2f} [{t['cond_lo']:.2f}, {t['cond_hi']:.2f}] | "
            f"{t['recover_pct']:.2f} | {t['GB_ratio']:.2f} |"
        )

    md_lines += ["", "## Cross-model rank tests"]
    for k, v in stats.items():
        if isinstance(v, dict):
            stat = v.get("statistic", v.get("correlation", "?"))
            p = v.get("pvalue", "?")
            md_lines.append(f"- **{k}**: stat={float(stat):.3f}, p={float(p):.4f}")
        else:
            md_lines.append(f"- **{k}**: {v}")

    md_path = RES / "CONDITIONAL_COLLAPSE_TABLE.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print("\n".join(md_lines))
    print(f"\nWrote {out_path} and {md_path}")


if __name__ == "__main__":
    main()
