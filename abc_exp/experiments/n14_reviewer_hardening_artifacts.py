"""Generate zero-API reviewer-hardening artifacts for the NeurIPS draft.

This script only reads existing result files and writes new derived artifacts:

  - abc_exp/results/n14_reviewer_hardening_artifacts.json
  - abc_exp/results/N14_REVIEWER_HARDENING_ARTIFACTS.md

It intentionally does not overwrite sealed-vintage artifacts.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import beta, binomtest


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "abc_exp" / "results"
OUT_JSON = RES / "n14_reviewer_hardening_artifacts.json"
OUT_MD = RES / "N14_REVIEWER_HARDENING_ARTIFACTS.md"

SEED = 20260427
N_EIV_DRAWS = 20_000

NEW_ALPHA_FILES = {
    "deepseek-v4-flash": "sa_causal_router_deepseek-v4-flash.jsonl",
    "gemma-4-31b-it-awq": "sa_causal_vllm_gemma-4-31b-it-awq.jsonl",
    "qwen3.5-4b": "sa_causal_vllm_qwen3.5-4b.jsonl",
    "qwen3.5-9b": "sa_causal_vllm_qwen3.5-9b.jsonl",
    "qwen3.6-27b-fp8": "sa_causal_vllm_qwen3.6-27b-fp8.jsonl",
    "qwen3.6-35b-a3b-fp8": "sa_causal_vllm_qwen3.6-35b-a3b-fp8.jsonl",
}

TRACE_FILES = {
    "deepseek-v4-flash": "debate_traces_openrouter_deepseek-v4-flash.jsonl",
    "gemma-4-31b-it-awq": "debate_traces_vllm_gemma-4-31b-it.jsonl",
    "qwen3.5-4b": "debate_traces_vllm_qwen3.5-4b.jsonl",
    "qwen3.5-9b": "debate_traces_vllm_qwen3.5-9b.jsonl",
    "qwen3.6-27b-fp8": "debate_traces_vllm_qwen3.6-27b-fp8.jsonl",
    "qwen3.6-35b-a3b-fp8": "debate_traces_vllm_qwen3.6-35b-a3b-fp8.jsonl",
    "qwen3_32b_a7_n200": "debate_traces_vllm_qwen3_32b_a7_n200.jsonl",
    "gemini_3_1_flash_lite_n200": "debate_traces_gemini_3_1_flash_lite_n200.jsonl",
    "mistral_small_4_n20": "debate_traces_openrouter_mistral_small_4_n20.jsonl",
}

OLD_ALPHA_EFFECTIVE_N = 1200


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def majority(values: list[str | None]) -> str | None:
    clean = [v for v in values if v]
    if not clean:
        return None
    counts = Counter(clean)
    max_count = max(counts.values())
    return sorted(k for k, v in counts.items() if v == max_count)[0]


def spearman(x: list[float], y: list[float]) -> float:
    return float(stats.spearmanr(x, y).statistic)


def load_rows() -> list[dict[str, Any]]:
    return json.loads((RES / "reviewer_hardening_analyses.json").read_text())["per_model"]


def family_exact_permutation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    family_rows = []
    for family, vals in sorted(by_family.items()):
        family_rows.append(
            {
                "family": family,
                "n_models": len(vals),
                "alpha_tot": float(np.mean([v["alpha_tot"] for v in vals])),
                "c_cond_pct": float(np.mean([v["c_cond_pct"] for v in vals])),
            }
        )

    x = [r["alpha_tot"] for r in family_rows]
    y = [r["c_cond_pct"] for r in family_rows]
    observed = spearman(x, y)
    null = [spearman(x, list(p)) for p in itertools.permutations(y)]
    ge = sum(v >= observed - 1e-12 for v in null)
    two = sum(abs(v) >= abs(observed) - 1e-12 for v in null)
    return {
        "family_rows": family_rows,
        "observed_rho": observed,
        "n_permutations": len(null),
        "one_sided_ge_count": ge,
        "p_one_sided_positive_exact": ge / len(null),
        "two_sided_abs_count": two,
        "p_two_sided_exact": two / len(null),
    }


def intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        k = int(row["n_collapses"])
        n = int(row["n_init_correct"])
        p = k / n if n else math.nan
        wilson = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        jeff = (
            float(beta.ppf(0.025, k + 0.5, n - k + 0.5)),
            float(beta.ppf(0.975, k + 0.5, n - k + 0.5)),
        )
        out.append(
            {
                "model": row["model"],
                "family": row["family"],
                "k_collapses": k,
                "n_init_correct": n,
                "c_cond": p,
                "c_cond_pct": 100 * p,
                "wilson_95": [float(wilson.low), float(wilson.high)],
                "wilson_95_pct": [100 * float(wilson.low), 100 * float(wilson.high)],
                "jeffreys_95": list(jeff),
                "jeffreys_95_pct": [100 * jeff[0], 100 * jeff[1]],
            }
        )
    return out


def alpha_missing_bounds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, float]] = {}
    for model, fname in NEW_ALPHA_FILES.items():
        path = RES / fname
        current_vals = []
        upper_vals = []
        lower_vals = []
        n_probe = 0
        n_none = 0
        for row in read_jsonl(path):
            probes = row.get("probe_results") or []
            if not probes:
                continue
            revised = sum(1 for p in probes if p.get("revised"))
            missing = sum(1 for p in probes if p.get("post_answer") is None)
            denom = len(probes)
            current_vals.append(revised / denom)
            lower_vals.append((revised - missing) / denom)
            upper_vals.append((revised + missing) / denom)
            n_probe += denom
            n_none += missing
        if current_vals:
            by_model[model] = {
                "current_alpha": float(np.mean(current_vals)),
                "drop_or_none_as_no_flip_alpha": float(np.mean(lower_vals)),
                "none_as_flip_alpha": float(np.mean(upper_vals)),
                "n_probe_trials": n_probe,
                "n_post_answer_none": n_none,
                "none_rate": n_none / n_probe if n_probe else math.nan,
            }

    scenarios = {}
    for scenario, key in [
        ("current", "alpha_tot"),
        ("new_lanes_none_as_flip", "none_as_flip_alpha"),
    ]:
        x = []
        y = []
        for row in rows:
            if scenario != "current" and row["model"] in by_model:
                x.append(by_model[row["model"]][key])
            else:
                x.append(float(row["alpha_tot"]))
            y.append(float(row["c_cond_pct"]))
        rho, p_two = stats.spearmanr(x, y)
        scenarios[scenario] = {"rho": float(rho), "p_two_sided_asymptotic": float(p_two)}
    return {"new_lane_bounds": by_model, "headline_scenarios": scenarios}


def parser_counts() -> dict[str, Any]:
    final_answer_pat = re.compile(r"Final\s+Answer\s*:\s*([A-J])\b", re.I)
    traces = []
    for label, fname in TRACE_FILES.items():
        path = RES / fname
        rows = read_jsonl(path)
        if not rows:
            continue
        n_debates = len(rows)
        init_none_agents = final_none_agents = any_init_none = any_final_none = maj_none = 0
        states = none_states = final_tags = answer_final_tag_mismatch = 0
        for row in rows:
            init = row.get("initial_answers") or []
            final = row.get("final_answers") or []
            init_none_agents += sum(v is None for v in init)
            final_none_agents += sum(v is None for v in final)
            any_init_none += int(any(v is None for v in init))
            any_final_none += int(any(v is None for v in final))
            maj_none += int(row.get("majority_answer") is None)
            all_states = (row.get("initial_states") or []) + [
                state for round_states in (row.get("round_traces") or []) for state in round_states
            ]
            for state in all_states:
                states += 1
                answer = state.get("answer")
                none_states += int(answer is None)
                tags = final_answer_pat.findall(state.get("reasoning") or "")
                if tags:
                    final_tags += 1
                    answer_final_tag_mismatch += int(bool(answer) and answer.upper() != tags[-1].upper())
        traces.append(
            {
                "label": label,
                "file": fname,
                "n_debates": n_debates,
                "initial_answer_none_agents": init_none_agents,
                "final_answer_none_agents": final_none_agents,
                "debates_with_any_initial_none": any_init_none,
                "debates_with_any_final_none": any_final_none,
                "majority_answer_none": maj_none,
                "state_count": states,
                "state_answer_none": none_states,
                "state_answer_none_rate": none_states / states if states else math.nan,
                "states_with_final_answer_tag": final_tags,
                "answer_vs_last_final_answer_tag_mismatch": answer_final_tag_mismatch,
            }
        )

    alphas = []
    for path in sorted(RES.glob("sa_causal_*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        n_rows = init_none = post_none = n_probe = rows_any_none = 0
        for row in rows:
            n_rows += 1
            init_none += int(row.get("initial_answer") is None)
            any_none = False
            for probe in row.get("probe_results") or []:
                n_probe += 1
                if probe.get("post_answer") is None:
                    post_none += 1
                    any_none = True
            rows_any_none += int(any_none)
        alphas.append(
            {
                "file": path.name,
                "n_rows": n_rows,
                "initial_answer_none_rows": init_none,
                "probe_trials": n_probe,
                "post_answer_none": post_none,
                "post_answer_none_rate": post_none / n_probe if n_probe else math.nan,
                "rows_with_any_post_answer_none": rows_any_none,
            }
        )
    return {"debate_traces": traces, "alpha_traces": alphas}


def eiv_n14(rows: list[dict[str, Any]], seed: int = SEED, n_draws: int = N_EIV_DRAWS) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    alpha = np.array([float(r["alpha_tot"]) for r in rows])
    k = np.array([int(r["n_collapses"]) for r in rows])
    n = np.array([int(r["n_init_correct"]) for r in rows])
    sd = []
    for row in rows:
        model = row["model"]
        # Use actual probe count where available; old sealed lanes use the same
        # conservative effective N as the existing N=9 EIV script.
        n_eff = OLD_ALPHA_EFFECTIVE_N
        if model in NEW_ALPHA_FILES:
            path = RES / NEW_ALPHA_FILES[model]
            probe_trials = sum(len(r.get("probe_results") or []) for r in read_jsonl(path))
            if probe_trials:
                n_eff = probe_trials
        p = float(row["alpha_tot"])
        sd.append(math.sqrt(max(p * (1 - p), 1e-6) / n_eff))
    sd_alpha = np.array(sd)

    draws = np.zeros(n_draws)
    for i in range(n_draws):
        a_draw = np.clip(alpha + rng.normal(0.0, sd_alpha), 0.0, 1.0)
        theta_draw = rng.beta(k + 0.5, n - k + 0.5)
        draws[i] = spearman(list(a_draw), list(theta_draw))

    point = spearman(list(alpha), list(k / n))
    return {
        "method": "parametric_bootstrap_alpha_normal_jeffreys_theta",
        "seed": seed,
        "n_draws": n_draws,
        "n_models": len(rows),
        "point_spearman": point,
        "median": float(np.median(draws)),
        "mean": float(np.mean(draws)),
        "ci_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "p_gt_0": float(np.mean(draws > 0)),
        "rows": [
            {
                "model": row["model"],
                "alpha_tot": float(row["alpha_tot"]),
                "sd_alpha": float(sd_alpha[i]),
                "k_collapses": int(k[i]),
                "n_init_correct": int(n[i]),
            }
            for i, row in enumerate(rows)
        ],
    }


def provenance() -> list[dict[str, Any]]:
    files = []
    for label, fname in {**TRACE_FILES, **NEW_ALPHA_FILES}.items():
        path = RES / fname
        if not path.exists():
            continue
        rows = read_jsonl(path)
        timestamps = [r.get("timestamp") for r in rows if r.get("timestamp")]
        backends = sorted({str(r.get("backend")) for r in rows if r.get("backend")})
        model_names = sorted({str(r.get("model_name")) for r in rows if r.get("model_name")})
        provider_counts = Counter()
        for row in rows:
            provider_counts.update(row.get("provider_distribution") or {})
            if row.get("provider_resolved"):
                provider_counts[str(row["provider_resolved"])] += 1
        files.append(
            {
                "label": label,
                "file": fname,
                "sha256": sha256(path),
                "rows": len(rows),
                "timestamp_min": min(timestamps) if timestamps else None,
                "timestamp_max": max(timestamps) if timestamps else None,
                "backends": backends,
                "model_names": model_names,
                "provider_distribution": dict(provider_counts),
            }
        )
    return files


def write_markdown(payload: dict[str, Any]) -> None:
    intervals_rows = payload["per_model_intervals"]
    parser = payload["parser_counts"]
    family = payload["family_exact_permutation"]
    eiv = payload["eiv_n14"]
    missing = payload["alpha_missing_sensitivity"]
    partial = payload["partial_spearman"]
    predictor_bakeoff = payload.get("predictor_bakeoff") or {}

    def fmt_stat(d: dict[str, Any]) -> str:
        if d.get("rho") is None:
            return "NA"
        return f"{d['rho']:+.3f} (n={d['n']}, p={d['p_two_sided']:.3g})"

    def fmt_exact(d: dict[str, Any]) -> str:
        if d.get("rho") is None:
            return "NA"
        return (
            f"{d['rho']:+.3f} "
            f"(G={d['n']}, p2={d['p_two_sided_exact']:.3g}, "
            f"p1-{d['observed_direction'][:3]}={d['p_one_sided_in_observed_direction_exact']:.3g})"
        )

    lines = [
        "# N14 Reviewer-Hardening Artifacts",
        "",
        "Zero-API derived artifact for the NeurIPS 2026 submission. This file does not overwrite sealed-vintage results.",
        "",
        "## Family-Level Primary Inference",
        "",
        f"- Family-aggregated Spearman rho: **{family['observed_rho']:+.4f}** over G=7 families.",
        f"- Exact permutation one-sided p: **{family['one_sided_ge_count']}/{family['n_permutations']} = {family['p_one_sided_positive_exact']:.4f}**.",
        f"- Exact permutation two-sided p: **{family['p_two_sided_exact']:.4f}**.",
        "",
        "## N14 Errors-in-Variables",
        "",
        f"- Method: `{eiv['method']}`, seed={eiv['seed']}, draws={eiv['n_draws']}.",
        f"- Point Spearman: **{eiv['point_spearman']:+.4f}**.",
        f"- Latent rank-correlation median: **{eiv['median']:+.4f}**, 95% interval [{eiv['ci_95'][0]:+.4f}, {eiv['ci_95'][1]:+.4f}], Pr(r>0)={eiv['p_gt_0']:.3f}.",
        "",
        "## Capability/Revision Confound Checks",
        "",
    ]
    for name, vals in partial.items():
        lines.append(f"- `{name}`: rho={vals['rho']:+.4f}, p={vals['p_two_sided']:.4f}, controls={','.join(vals['controls'])}.")
    if predictor_bakeoff:
        lines.extend([
            "",
            "## Baseline/Confound Bakeoff",
            "",
            "Only the fingerprint and initial-majority accuracy are pre-debate quantities. Post-debate and outcome-derived controls are included to quantify confounding pressure, not as deployable predictors.",
            "",
            "| Predictor | Role | Model-row Spearman | Family-mean exact Spearman |",
            "|---|---|---:|---:|",
        ])
        for vals in predictor_bakeoff.values():
            lines.append(
                f"| {vals['label']} | {vals['role']} | "
                f"{fmt_stat(vals['model_row_spearman'])} | {fmt_exact(vals['family_mean_exact_spearman'])} |"
            )
    lines.extend([
        "",
        "## Per-Model Conditional-Collapse Intervals",
        "",
        "| Model | k/n | C_cond | Wilson 95% | Jeffreys 95% |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in intervals_rows:
        lines.append(
            f"| `{row['model']}` | {row['k_collapses']}/{row['n_init_correct']} | "
            f"{row['c_cond_pct']:.2f}% | [{row['wilson_95_pct'][0]:.2f}, {row['wilson_95_pct'][1]:.2f}] | "
            f"[{row['jeffreys_95_pct'][0]:.2f}, {row['jeffreys_95_pct'][1]:.2f}] |"
        )
    lines.extend([
        "",
        "## Parser Failure Summary",
        "",
        "| Debate trace | states | answer None | debates with init None | debates with final None | answer/final-tag mismatch |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in parser["debate_traces"]:
        lines.append(
            f"| `{row['file']}` | {row['state_count']} | {row['state_answer_none']} | "
            f"{row['debates_with_any_initial_none']} | {row['debates_with_any_final_none']} | "
            f"{row['answer_vs_last_final_answer_tag_mismatch']} |"
        )
    lines.extend([
        "",
        "| Alpha trace | rows | probe trials | post-answer None | post-answer None rate |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in parser["alpha_traces"]:
        if row["post_answer_none"] == 0 and row["n_rows"] < 1000:
            continue
        lines.append(
            f"| `{row['file']}` | {row['n_rows']} | {row['probe_trials']} | "
            f"{row['post_answer_none']} | {100 * row['post_answer_none_rate']:.2f}% |"
        )
    lines.extend([
        "",
        "## Parse-Missing Sensitivity",
        "",
    ])
    for name, vals in missing["headline_scenarios"].items():
        lines.append(f"- `{name}`: rho={vals['rho']:+.4f}, two-sided asymptotic p={vals['p_two_sided_asymptotic']:.4g}.")
    lines.extend([
        "",
        "## Artifact Provenance",
        "",
        "| File | Rows | Timestamp range | Backend/model metadata present | SHA256 prefix |",
        "|---|---:|---|---|---:|",
    ])
    for row in payload["provenance"]:
        ts = "--" if not row["timestamp_min"] else f"{row['timestamp_min']} to {row['timestamp_max']}"
        meta = ", ".join(row["backends"] + row["model_names"]) or "partial/missing"
        lines.append(f"| `{row['file']}` | {row['rows']} | {ts} | {meta} | `{row['sha256'][:12]}` |")
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = load_rows()
    hardening = json.loads((RES / "reviewer_hardening_analyses.json").read_text())
    payload = {
        "description": "Zero-API N=14 reviewer-hardening artifact generated from existing results.",
        "source_rows": "abc_exp/results/reviewer_hardening_analyses.json:per_model",
        "family_exact_permutation": family_exact_permutation(rows),
        "per_model_intervals": intervals(rows),
        "partial_spearman": hardening["partial_spearman"],
        "predictor_bakeoff_spearman_vs_ccond": hardening["predictor_bakeoff_spearman_vs_ccond"],
        "predictor_bakeoff": hardening.get("predictor_bakeoff", {}),
        "alpha_missing_sensitivity": alpha_missing_bounds(rows),
        "parser_counts": parser_counts(),
        "eiv_n14": eiv_n14(rows),
        "provenance": provenance(),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    write_markdown(payload)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
