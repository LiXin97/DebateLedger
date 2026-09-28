"""Zero-API summary for Round-5 API cross-benchmark hardening runs.

The API-generating harnesses write row-level JSONL plus coarse JSON summaries.
This script turns those paid runs into an auditable, reproducible appendix
artifact: conditional denominators, Wilson/Jeffreys intervals, signed utility,
alpha-valid correlations, parser/alpha missingness, provider counts, and costs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "abc_exp" / "results"


@dataclass(frozen=True)
class RunSpec:
    stem: str
    label: str
    role: str
    backend: str
    include_in_paper_table: bool = True


RUNS = [
    RunSpec(
        stem="cross_benchmark_openrouter_mistral_small_4_gpqa_full_20260430",
        label="Mistral Small 4 GPQA full",
        role="dense non-MMLU stress test",
        backend="openrouter",
    ),
    RunSpec(
        stem="cross_benchmark_gemini_3_1_flash_lite_gpqa_full_20260430",
        label="Gemini 3.1 Flash-Lite GPQA full",
        role="Google-family full-benchmark boundary check",
        backend="gemini",
    ),
    RunSpec(
        stem="cross_benchmark_openrouter_deepseek_v4_flash_gpqa_full_20260502",
        label="DeepSeek V4 Flash GPQA full",
        role="OpenRouter full-benchmark stress/boundary replication",
        backend="openrouter",
    ),
    RunSpec(
        stem="cross_benchmark_openrouter_llama_3_3_70b_gpqa_full_20260502",
        label="Llama 3.3 70B GPQA full",
        role="OpenRouter full-benchmark stress/boundary replication",
        backend="openrouter",
    ),
    RunSpec(
        stem="cross_benchmark_openrouter_grok_4_1_fast_gpqa_full_20260502",
        label="Grok 4.1 Fast GPQA full",
        role="OpenRouter full-benchmark stress/boundary replication",
        backend="openrouter",
    ),
    RunSpec(
        stem="cross_benchmark_gemini_3_1_flash_lite_n50",
        label="Gemini 3.1 Flash-Lite GPQA+TruthfulQA N50",
        role="earlier sparse boundary check",
        backend="gemini",
        include_in_paper_table=False,
    ),
    RunSpec(
        stem="cross_benchmark_gemini_2_5_flash_arc_gpqa_tqa_n20_smoke_20260430",
        label="Gemini 2.5 Flash ARC/GPQA/TQA N20 smoke",
        role="low-collapse smoke/boundary check",
        backend="gemini",
        include_in_paper_table=False,
    ),
    RunSpec(
        stem="cross_benchmark_openrouter_mistral_small_4_gpqa_tqa_n20_20260430",
        label="Mistral Small 4 GPQA+TruthfulQA N20 smoke",
        role="model/benchmark scout before full GPQA",
        backend="openrouter",
        include_in_paper_table=False,
    ),
    RunSpec(
        stem="cross_benchmark_openrouter_mistral_small_4_gpqa_n1_smoke_20260430",
        label="Mistral Small 4 GPQA N1 smoke",
        role="OpenRouter schema/cost smoke",
        backend="openrouter",
        include_in_paper_table=False,
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def pct(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "NA"
    return f"{100 * x:.1f}%"


def money(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "NA"
    return f"${x:.4f}"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float | None]:
    if n <= 0:
        return [None, None]
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def jeffreys(k: int, n: int) -> list[float | None]:
    if n <= 0:
        return [None, None]
    return [float(beta.ppf(0.025, k + 0.5, n - k + 0.5)), float(beta.ppf(0.975, k + 0.5, n - k + 0.5))]


def safe_spearman(rows: list[dict[str, Any]], y_key: str) -> dict[str, Any]:
    pairs: list[tuple[float, int]] = []
    for row in rows:
        alpha = row.get("alpha", {}).get("mean")
        if not finite(alpha):
            continue
        debate = row.get("standard_debate", {})
        if y_key == "collapse":
            y = int(bool(debate.get("collapsed")))
        elif y_key == "correction":
            y = int(bool(debate.get("corrected")))
        elif y_key == "final_wrong":
            y = int(not bool(debate.get("final_correct")))
        else:
            raise ValueError(f"Unknown y_key: {y_key}")
        pairs.append((float(alpha), y))
    if len(pairs) < 3 or len({a for a, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
        return {"n": len(pairs), "rho": None, "p_two_sided": None, "note": "insufficient variation"}
    rho, p = stats.spearmanr([a for a, _ in pairs], [y for _, y in pairs])
    return {"n": len(pairs), "rho": float(rho), "p_two_sided": float(p)}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    init_correct_rows = [r for r in rows if bool(r.get("standard_debate", {}).get("initial_correct"))]
    init_wrong_rows = [r for r in rows if not bool(r.get("standard_debate", {}).get("initial_correct"))]
    final_correct_rows = [r for r in rows if bool(r.get("standard_debate", {}).get("final_correct"))]
    collapses = [r for r in rows if bool(r.get("standard_debate", {}).get("collapsed"))]
    corrections = [r for r in rows if bool(r.get("standard_debate", {}).get("corrected"))]

    n_init = len(init_correct_rows)
    n_wrong = len(init_wrong_rows)
    n_collapse = len(collapses)
    n_correction = len(corrections)
    alpha_values = [float(r["alpha"]["mean"]) for r in rows if finite(r.get("alpha", {}).get("mean"))]
    agent_rows = [a for r in rows for a in r.get("alpha", {}).get("agents", [])]
    total_cost = float(sum(float(r.get("total_cost") or 0.0) for r in rows))
    alpha_cost = float(sum(float(r.get("alpha", {}).get("cost") or 0.0) for r in rows))
    debate_cost = float(sum(float(r.get("standard_debate", {}).get("cost") or 0.0) for r in rows))

    by_bench: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_bench.setdefault(str(r.get("benchmark", "unknown")), []).append(r)

    signed_net = n_correction - n_collapse
    return {
        "n": n,
        "benchmarks": sorted(by_bench),
        "initial_correct": n_init,
        "initial_wrong": n_wrong,
        "initial_accuracy": (n_init / n) if n else None,
        "final_accuracy": (len(final_correct_rows) / n) if n else None,
        "accuracy_delta_pp": (100 * (len(final_correct_rows) - n_init) / n) if n else None,
        "collapses": n_collapse,
        "conditional_collapse": {
            "k": n_collapse,
            "n": n_init,
            "rate": (n_collapse / n_init) if n_init else None,
            "wilson95": wilson(n_collapse, n_init),
            "jeffreys95": jeffreys(n_collapse, n_init),
        },
        "corrections": n_correction,
        "conditional_correction": {
            "k": n_correction,
            "n": n_wrong,
            "rate": (n_correction / n_wrong) if n_wrong else None,
            "wilson95": wilson(n_correction, n_wrong),
            "jeffreys95": jeffreys(n_correction, n_wrong),
        },
        "signed_utility_standard_vs_freeze_initial": {
            "corrections_minus_collapses": signed_net,
            "delta_accuracy_pp": (100 * signed_net / n) if n else None,
            "equal_weights": True,
        },
        "alpha": {
            "valid_rows": len(alpha_values),
            "missing_rows": n - len(alpha_values),
            "mean": float(np.mean(alpha_values)) if alpha_values else None,
            "min": float(np.min(alpha_values)) if alpha_values else None,
            "max": float(np.max(alpha_values)) if alpha_values else None,
            "agent_initial_answer_missing": sum(1 for a in agent_rows if not a.get("initial_answer")),
            "agent_alpha_missing": sum(1 for a in agent_rows if a.get("alpha") is None),
            "agent_total": len(agent_rows),
            "probe_total": sum(int(a.get("n_probes") or 0) for a in agent_rows),
        },
        "parser_missing": {
            "debate_initial_majority_missing": sum(1 for r in rows if not r.get("standard_debate", {}).get("initial_majority")),
            "debate_final_answer_missing": sum(1 for r in rows if not r.get("standard_debate", {}).get("final_answer")),
        },
        "spearman": {
            "alpha_vs_collapse_all_rows": safe_spearman(rows, "collapse"),
            "alpha_vs_collapse_initial_correct_only": safe_spearman(init_correct_rows, "collapse"),
            "alpha_vs_correction_initial_wrong_only": safe_spearman(init_wrong_rows, "correction"),
            "alpha_vs_final_wrong_all_rows": safe_spearman(rows, "final_wrong"),
        },
        "cost": {"total": total_cost, "alpha": alpha_cost, "debate": debate_cost},
        "per_benchmark": {name: summarize_rows(sub) for name, sub in sorted(by_bench.items())} if len(by_bench) > 1 else {},
    }


def summarize_run(spec: RunSpec) -> dict[str, Any] | None:
    summary_path = RES / f"{spec.stem}.json"
    jsonl_path = RES / f"{spec.stem}.jsonl"
    if not summary_path.exists() or not jsonl_path.exists():
        return None
    summary_json = read_json(summary_path)
    rows = read_jsonl(jsonl_path)
    config = summary_json.get("config", {})
    openrouter_usage = summary_json.get("openrouter_usage")
    row_summary = summarize_rows(rows)
    usage_scope = None
    if openrouter_usage:
        # Resumed runs rewrite the coarse JSON summary from the last invocation,
        # while row-level JSONL remains cumulative. Guard the ledger against
        # presenting a resume-only provider count as full-run routing evidence.
        n_calls = openrouter_usage.get("n_calls")
        min_expected_calls = row_summary["alpha"]["agent_total"]
        usage_scope = "full_or_single_invocation"
        if isinstance(n_calls, int) and n_calls < min_expected_calls:
            usage_scope = "latest_resume_invocation_only"
    return {
        "stem": spec.stem,
        "label": spec.label,
        "role": spec.role,
        "backend": spec.backend,
        "include_in_paper_table": spec.include_in_paper_table,
        "paths": {"summary_json": str(summary_path.relative_to(ROOT)), "rows_jsonl": str(jsonl_path.relative_to(ROOT))},
        "config": config,
        "stored_summary": summary_json.get("summary"),
        "openrouter_usage": openrouter_usage,
        "openrouter_usage_scope": usage_scope,
        "row_summary": row_summary,
    }


def interval_text(block: dict[str, Any]) -> str:
    rate = block["rate"]
    lo, hi = block["wilson95"]
    return f"{block['k']}/{block['n']} ({pct(rate)} [{pct(lo)}, {pct(hi)}])"


def rho_text(block: dict[str, Any]) -> str:
    rho = block.get("rho")
    p = block.get("p_two_sided")
    if rho is None:
        return f"NA (n={block.get('n')})"
    return f"rho={rho:+.3f}, p={p:.4g}, n={block.get('n')}"


def write_markdown(out: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# Round-5 API Cross-Benchmark Hardening",
        "",
        "Post-hoc reviewer-hardening runs. These are excluded from the primary MMLU-Pro family-level association and are used only as cross-benchmark stress/boundary evidence.",
        "",
        "## Run Ledger",
        "",
        "| Run | Role | Backend/model | Benchmarks | N | Row cost | Provider notes |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for run in out["runs"]:
        rs = run["row_summary"]
        cfg = run["config"]
        usage = run.get("openrouter_usage") or {}
        model = cfg.get("model_id") or cfg.get("model") or run["backend"]
        providers = usage.get("providers")
        provider_text = json.dumps(providers, sort_keys=True) if providers else ""
        if provider_text and run.get("openrouter_usage_scope") == "latest_resume_invocation_only":
            provider_text = f"latest resume only: {provider_text}; full routing mixed, see logs"
        lines.append(
            f"| {run['label']} | {run['role']} | {run['backend']} / `{model}` | "
            f"{', '.join(rs['benchmarks'])} | {rs['n']} | {money(rs['cost']['total'])} | {provider_text} |"
        )

    lines += [
        "",
        "## Paper-Facing Rows",
        "",
        "| Run | init acc | final acc | C^cond (Wilson 95%) | correction (Wilson 95%) | signed utility | alpha | alpha->collapse |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in [r for r in out["runs"] if r["include_in_paper_table"]]:
        rs = run["row_summary"]
        signed = rs["signed_utility_standard_vs_freeze_initial"]
        alpha = rs["alpha"]
        corr = rs["spearman"]["alpha_vs_collapse_initial_correct_only"]
        lines.append(
            f"| {run['label']} | {pct(rs['initial_accuracy'])} | {pct(rs['final_accuracy'])} | "
            f"{interval_text(rs['conditional_collapse'])} | {interval_text(rs['conditional_correction'])} | "
            f"{signed['corrections_minus_collapses']:+d} ({signed['delta_accuracy_pp']:+.2f}pp) | "
            f"{alpha['mean']:.3f} ({alpha['valid_rows']}/{rs['n']} rows) | {rho_text(corr)} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Mistral Small 4 on full GPQA Diamond is the dense non-MMLU stress test: nontrivial conditional collapse and correction both appear, and standard debate is only +1 net debate under equal collapse/correction weights.",
        "- Llama 3.3 70B and DeepSeek V4 Flash add full-GPQA stress rows where collapses are nonzero but corrections dominate under equal weights. These rows strengthen the signed-utility claim, not benchmark-general alpha transfer.",
        "- Gemini GPQA/ARC/TruthfulQA runs remain boundary checks: the full Gemini GPQA row has low collapse despite nonzero alpha, so low event counts should be reported as limits, not transfer success.",
        "- OpenRouter rows must retain provider-routing notes. Full runs may be served by multiple providers and resumed summaries can contain only latest-invocation provider counts, so these are not snapshot-pinned vendor replications.",
        "",
        "## Missingness",
        "",
        "| Run | alpha-valid rows | missing agent initials | missing debate initials | missing final answers |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in out["runs"]:
        rs = run["row_summary"]
        alpha = rs["alpha"]
        parser = rs["parser_missing"]
        lines.append(
            f"| {run['label']} | {alpha['valid_rows']}/{rs['n']} | "
            f"{alpha['agent_initial_answer_missing']}/{alpha['agent_total']} | "
            f"{parser['debate_initial_majority_missing']} | {parser['debate_final_answer_missing']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    runs = [r for spec in RUNS if (r := summarize_run(spec)) is not None]
    totals_by_backend: dict[str, float] = {}
    for run in runs:
        totals_by_backend.setdefault(run["backend"], 0.0)
        totals_by_backend[run["backend"]] += run["row_summary"]["cost"]["total"]
    out = {
        "generated_by": str(Path(__file__).relative_to(ROOT)),
        "note": "Zero-API recomputation over checked-in row-level API artifacts; no provider calls.",
        "runs": runs,
        "cost_totals_by_backend": totals_by_backend,
    }
    json_path = RES / "round5_api_crossbench_hardening.json"
    md_path = RES / "ROUND5_API_CROSSBENCH_HARDENING.md"
    json_path.write_text(json.dumps(out, indent=2, allow_nan=False))
    write_markdown(out, md_path)
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
