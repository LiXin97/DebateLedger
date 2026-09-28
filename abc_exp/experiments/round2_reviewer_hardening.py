"""Round-2 reviewer hardening analyses for the ED-track draft.

This script is intentionally zero-cost: it reads checked-in artifacts only and
does not run model calls.  It addresses the round-2 review gaps that can be
answered from the current checkout:

  * held-out-family LoFO prediction intervals for C^cond;
  * numerical comparator/proxy bakeoff against capability, social-pressure,
    revision-quantity, and runtime trajectory scores;
  * Qwen3-32B post-hoc sensitivity as N=15 / stress-family rows;
  * artifact audit for the requested 6,525-debate LOMO same-pool gate replay.

Outputs:
  - abc_exp/results/round2_reviewer_hardening.json
  - abc_exp/results/ROUND2_REVIEWER_HARDENING.md
"""

from __future__ import annotations

import json
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abc_exp.experiments import correction_preserving_policy_replay as replay


RES = ROOT / "abc_exp" / "results"
OUT_JSON = RES / "round2_reviewer_hardening.json"
OUT_MD = RES / "ROUND2_REVIEWER_HARDENING.md"


ALPHA_SPLIT_NAME = {
    "sonnet-4.5": "Sonnet 4.5",
    "gemini-3-flash": "Gemini 3-flash",
    "llama-3.1-8b": "Llama-3.1-8B",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "phi-4-mini": "Phi-4-mini",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
}

SA_FILE_BY_MODEL = {
    "deepseek-v4-flash": RES / "sa_causal_router_deepseek-v4-flash.jsonl",
    "gemini-3-flash": RES / "sa_causal_gemini_3_flash_n200.jsonl",
    "gemma-4-31b-it-awq": RES / "sa_causal_vllm_gemma-4-31b-it-awq.jsonl",
    "qwen3.5-4b": RES / "sa_causal_vllm_qwen3.5-4b.jsonl",
    "qwen3.5-9b": RES / "sa_causal_vllm_qwen3.5-9b.jsonl",
    "qwen3.6-27b-fp8": RES / "sa_causal_vllm_qwen3.6-27b-fp8.jsonl",
    "qwen3.6-35b-a3b-fp8": RES / "sa_causal_vllm_qwen3.6-35b-a3b-fp8.jsonl",
}

RAW_ALPHA_FILE_BY_MODEL = {
    "deepseek-v4-flash": RES / "sa_causal_router_deepseek-v4-flash.jsonl",
    "gemma-4-31b-it-awq": RES / "sa_causal_vllm_gemma-4-31b-it-awq.jsonl",
    "qwen3.5-4b": RES / "sa_causal_vllm_qwen3.5-4b.jsonl",
    "qwen3.5-9b": RES / "sa_causal_vllm_qwen3.5-9b.jsonl",
    "qwen3.6-27b-fp8": RES / "sa_causal_vllm_qwen3.6-27b-fp8.jsonl",
    "qwen3.6-35b-a3b-fp8": RES / "sa_causal_vllm_qwen3.6-35b-a3b-fp8.jsonl",
}

TRACE_SOURCE_TO_MODEL = {
    "openrouter_deepseek-v4-flash": "deepseek-v4-flash",
    "vllm_gemma-4-31b-it": "gemma-4-31b-it-awq",
    "vllm_qwen3.5-4b": "qwen3.5-4b",
    "vllm_qwen3.5-9b": "qwen3.5-9b",
    "vllm_qwen3.6-27b-fp8": "qwen3.6-27b-fp8",
    "vllm_qwen3.6-35b-a3b-fp8": "qwen3.6-35b-a3b-fp8",
}


def read_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def finite_pairs(rows: Iterable[dict[str, Any]], x_key: str, y_key: str = "c_cond_pct") -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        x = float(x)
        y = float(y)
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    return pairs


def spearman(rows: Iterable[dict[str, Any]], x_key: str, y_key: str = "c_cond_pct") -> dict[str, Any]:
    pairs = finite_pairs(rows, x_key, y_key)
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
        return {"n": len(pairs), "rho": None, "p_two_sided": None, "note": "insufficient variation"}
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    rho, p = stats.spearmanr(x, y)
    return {"n": len(pairs), "rho": float(rho), "p_two_sided": float(p), "p_value_method": "scipy_spearman"}


def exact_spearman(rows: Iterable[dict[str, Any]], x_key: str, y_key: str = "c_cond_pct") -> dict[str, Any]:
    pairs = finite_pairs(rows, x_key, y_key)
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
        return {"n": len(pairs), "rho": None, "p_two_sided": None, "note": "insufficient variation"}
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    obs = float(stats.spearmanr(x, y).statistic)
    null = [float(stats.spearmanr(x, list(p)).statistic) for p in set(itertools.permutations(y))]
    if obs >= 0:
        one_sided = sum(v >= obs - 1e-12 for v in null)
        direction = "positive"
    else:
        one_sided = sum(v <= obs + 1e-12 for v in null)
        direction = "negative"
    two_sided = sum(abs(v) >= abs(obs) - 1e-12 for v in null)
    scipy = spearman(rows, x_key, y_key)
    return {
        "n": len(pairs),
        "rho": obs,
        "p_two_sided": two_sided / len(null),
        "p_value_method": "exact_permutation",
        "p_two_sided_exact": two_sided / len(null),
        "p_one_sided_in_observed_direction_exact": one_sided / len(null),
        "observed_direction": direction,
        "n_permutations": len(null),
        "p_two_sided_scipy": scipy.get("p_two_sided"),
    }


def partial_spearman(rows: list[dict[str, Any]], x_key: str, y_key: str, control_keys: list[str]) -> dict[str, Any]:
    data = []
    for row in rows:
        vals = [row.get(x_key), row.get(y_key)] + [row.get(k) for k in control_keys]
        if any(v is None for v in vals):
            continue
        vals = [float(v) for v in vals]
        if all(math.isfinite(v) for v in vals):
            data.append(vals)
    n = len(data)
    k = len(control_keys)
    if n <= k + 3:
        return {"n": n, "rho": None, "p_two_sided": None, "controls": control_keys, "note": "insufficient df"}
    arr = np.asarray(data, dtype=float)
    ranked = np.apply_along_axis(stats.rankdata, 0, arr)
    x = ranked[:, 0]
    y = ranked[:, 1]
    z = ranked[:, 2:]
    design = np.column_stack([np.ones(n), z])
    bx = np.linalg.lstsq(design, x, rcond=None)[0]
    by = np.linalg.lstsq(design, y, rcond=None)[0]
    rx = x - design @ bx
    ry = y - design @ by
    rho = float(np.corrcoef(rx, ry)[0, 1])
    df = n - k - 2
    t = rho * math.sqrt(df / max(1e-12, 1 - rho * rho))
    p = 2 * stats.t.sf(abs(t), df)
    return {"n": n, "rho": rho, "p_two_sided": float(p), "df": df, "controls": control_keys}


def aggregate_by_family(rows: Iterable[dict[str, Any]], key_names: Iterable[str]) -> list[dict[str, Any]]:
    fams: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fams[str(row["family"])].append(row)
    out: list[dict[str, Any]] = []
    for family, rs in sorted(fams.items()):
        entry: dict[str, Any] = {"family": family, "n_models": len(rs), "models": [r["model"] for r in rs]}
        for key in key_names:
            vals = [r.get(key) for r in rs]
            vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
            if vals:
                entry[key] = float(np.mean(vals))
        out.append(entry)
    return out


def lofo_family_prediction(family_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for held in family_rows:
        train = [r for r in family_rows if r["family"] != held["family"]]
        x = np.asarray([float(r["alpha_tot"]) for r in train])
        y = np.asarray([float(r["c_cond_pct"]) for r in train])
        design = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        pred = float(np.asarray([1.0, held["alpha_tot"]]) @ beta)
        resid = y - design @ beta
        df = len(x) - 2
        sigma = float(np.sqrt(float(resid @ resid) / df))
        xbar = float(np.mean(x))
        sxx = float(np.sum((x - xbar) ** 2))
        pred_se = sigma * math.sqrt(1.0 + 1.0 / len(x) + (float(held["alpha_tot"]) - xbar) ** 2 / sxx)
        t80 = float(stats.t.ppf(0.90, df))
        t95 = float(stats.t.ppf(0.975, df))
        lo80, hi80 = pred - t80 * pred_se, pred + t80 * pred_se
        lo95, hi95 = pred - t95 * pred_se, pred + t95 * pred_se
        obs = float(held["c_cond_pct"])
        rows.append(
            {
                "heldout_family": held["family"],
                "alpha_tot": float(held["alpha_tot"]),
                "observed_c_cond_pct": obs,
                "predicted_c_cond_pct": pred,
                "prediction_se": pred_se,
                "pi80": [lo80, hi80],
                "pi95": [lo95, hi95],
                "covered_80": lo80 <= obs <= hi80,
                "covered_95": lo95 <= obs <= hi95,
                "train_df": df,
            }
        )

    obs = [r["observed_c_cond_pct"] for r in rows]
    pred = [r["predicted_c_cond_pct"] for r in rows]
    rho, p = stats.spearmanr(pred, obs)
    pred_rank = {r["heldout_family"]: i + 1 for i, r in enumerate(sorted(rows, key=lambda z: z["predicted_c_cond_pct"]))}
    obs_rank = {r["heldout_family"]: i + 1 for i, r in enumerate(sorted(rows, key=lambda z: z["observed_c_cond_pct"]))}
    for row in rows:
        fam = row["heldout_family"]
        row["predicted_rank"] = pred_rank[fam]
        row["observed_rank"] = obs_rank[fam]
        row["absolute_rank_error"] = abs(pred_rank[fam] - obs_rank[fam])
    return {
        "description": "Post-hoc leave-one-family-out linear predictive check on family means; intervals are unbounded Gaussian prediction intervals fit on the other six families.",
        "n_families": len(rows),
        "rows": rows,
        "predicted_vs_observed_spearman": {"rho": float(rho), "p_two_sided": float(p), "n": len(rows)},
        "coverage_80": {"covered": int(sum(r["covered_80"] for r in rows)), "total": len(rows)},
        "coverage_95": {"covered": int(sum(r["covered_95"] for r in rows)), "total": len(rows)},
        "mean_absolute_rank_error": float(np.mean([r["absolute_rank_error"] for r in rows])),
        "largest_absolute_errors": sorted(rows, key=lambda r: abs(r["observed_c_cond_pct"] - r["predicted_c_cond_pct"]), reverse=True)[:2],
    }


def summarize_sa_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition", "default") != "default":
                continue
            rows.append(row)
    if not rows:
        return None
    social_vals: list[float] = []
    solo_vals: list[float] = []
    social_lifts: list[float] = []
    sensitivities: list[float] = []
    for row in rows:
        social = row.get("alpha_social")
        solo = row.get("alpha_solo")
        if isinstance(social, (int, float)) and isinstance(solo, (int, float)):
            social_f = float(social)
            solo_f = float(solo)
        else:
            probe_results = row.get("probe_results") or []
            social_revised = [float(bool(p.get("revised"))) for p in probe_results if bool(p.get("social"))]
            solo_revised = [float(bool(p.get("revised"))) for p in probe_results if not bool(p.get("social"))]
            if not social_revised or not solo_revised:
                continue
            social_f = float(np.mean(social_revised))
            solo_f = float(np.mean(solo_revised))
        social_vals.append(social_f)
        solo_vals.append(solo_f)
        social_lifts.append(social_f - solo_f)
        if isinstance(row.get("social_sensitivity"), (int, float)):
            sensitivities.append(float(row["social_sensitivity"]))
        else:
            sensitivities.append(social_f - solo_f)
    if not social_vals:
        return None
    return {
        "n_rows": len(rows),
        "n_rows_with_social_rates": len(social_vals),
        "alpha_social": float(np.mean(social_vals)),
        "alpha_solo": float(np.mean(solo_vals)),
        "social_lift": float(np.mean(social_lifts)),
        "mean_social_sensitivity": float(np.mean(sensitivities)),
        "source": str(path.relative_to(ROOT)),
    }


def attach_social_proxy(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    alpha_split = read_json(RES / "alpha_split.json")["per_model"]
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        enriched = dict(row)
        model = row["model"]
        summary = summarize_sa_file(SA_FILE_BY_MODEL[model]) if model in SA_FILE_BY_MODEL else None
        if summary is not None:
            enriched.update(
                {
                    "social_proxy_alpha_social": summary["alpha_social"],
                    "social_proxy_lift": summary["social_lift"],
                    "social_proxy_mean_social_sensitivity": summary["mean_social_sensitivity"],
                    "social_proxy_source": summary["source"],
                }
            )
        elif model in ALPHA_SPLIT_NAME and ALPHA_SPLIT_NAME[model] in alpha_split:
            values = alpha_split[ALPHA_SPLIT_NAME[model]]
            social = float(values["alpha_total_social"])
            nonsocial = float(values["alpha_total_nonsocial"])
            enriched.update(
                {
                    "social_proxy_alpha_social": social,
                    "social_proxy_lift": social - nonsocial,
                    "social_proxy_mean_social_sensitivity": social - nonsocial,
                    "social_proxy_source": "alpha_split.json/per_model",
                }
            )
        else:
            missing.append(model)
        out.append(enriched)
    return out, missing


def trace_runtime_rows(headline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model = {r["model"]: r for r in headline_rows}
    records, sources = replay.load_debate_records()
    grouped: dict[str, list[Any]] = defaultdict(list)
    for rec in records:
        model = TRACE_SOURCE_TO_MODEL.get(rec.source)
        if model in by_model and not sources[rec.source].get("excluded_from_primary"):
            grouped[model].append(rec)
    rows: list[dict[str, Any]] = []
    for model, recs in sorted(grouped.items()):
        n = len(recs)
        rows.append(
            {
                "model": model,
                "family": by_model[model]["family"],
                "n_debates": n,
                "alpha_tot": by_model[model]["alpha_tot"],
                "c_cond_pct": by_model[model]["c_cond_pct"],
                "initial_disagreement_rate": float(np.mean([1.0 - float(r.features.get("init_unanimous") or 0.0) for r in recs])),
                "r1_majority_changed_rate": float(np.mean([float(r.features.get("r1_majority_changed") or 0.0) for r in recs])),
                "any_r1_flip_rate": float(np.mean([float(r.features.get("r1_n_flipped") or 0.0) > 0.0 for r in recs])),
                "r1_flip_frac": float(np.mean([float(r.features.get("r1_flip_frac") or 0.0) for r in recs])),
            }
        )
    return rows


def comparator_bakeoff(headline_rows: list[dict[str, Any]], hardening_rows: list[dict[str, Any]]) -> dict[str, Any]:
    social_rows, social_missing = attach_social_proxy(headline_rows)
    runtime_rows = trace_runtime_rows(headline_rows)
    rows_by_model = {r["model"]: dict(r) for r in headline_rows}
    for row in hardening_rows:
        if row["model"] in rows_by_model:
            rows_by_model[row["model"]].update(row)
    merged_rows = list(rows_by_model.values())
    for row in merged_rows:
        if row.get("init_acc") is not None:
            row["engels_capability_pressure_proxy"] = 1.0 - float(row["init_acc"])

    social_family = aggregate_by_family(
        [r for r in social_rows if r.get("social_proxy_alpha_social") is not None],
        ["c_cond_pct", "social_proxy_alpha_social", "social_proxy_lift", "social_proxy_mean_social_sensitivity"],
    )
    merged_family = aggregate_by_family(
        [r for r in merged_rows if r.get("engels_capability_pressure_proxy") is not None],
        ["c_cond_pct", "alpha_tot", "engels_capability_pressure_proxy", "debate_revision_proxy"],
    )

    entries = [
        {
            "name": "alpha_tot",
            "role": "selection-time fingerprint",
            "model_row": spearman(headline_rows, "alpha_tot"),
            "family_mean": exact_spearman(aggregate_by_family(headline_rows, ["c_cond_pct", "alpha_tot"]), "alpha_tot"),
            "note": "headline selection-time measurement; measured before downstream debate sweep",
        },
        {
            "name": "engels_capability_pressure_proxy",
            "role": "Engels-style capability proxy (not exact Engels gap)",
            "model_row": spearman(merged_rows, "engels_capability_pressure_proxy"),
            "family_mean": exact_spearman(merged_family, "engels_capability_pressure_proxy"),
            "note": "defined as 1 - initial-majority accuracy; exact heterogeneous capability-gap/overseer score is not in local artifacts",
        },
        {
            "name": "debate_revision_proxy",
            "role": "raw revision-quantity baseline",
            "model_row": spearman(merged_rows, "debate_revision_proxy"),
            "family_mean": exact_spearman(merged_family, "debate_revision_proxy"),
            "note": "debate flip-rate / majority-flip proxy; measured from debate traces or older conditional-collapse table",
        },
        {
            "name": "social_proxy_alpha_social",
            "role": "Sharma/Perez-style social-pressure proxy",
            "model_row": spearman(social_rows, "social_proxy_alpha_social"),
            "family_mean": exact_spearman(social_family, "social_proxy_alpha_social"),
            "note": "mean flip rate on social-attributed counterarguments, using sa_causal default rows where available and alpha_split fallback otherwise",
        },
        {
            "name": "social_proxy_lift",
            "role": "social-over-solo lift proxy",
            "model_row": spearman(social_rows, "social_proxy_lift"),
            "family_mean": exact_spearman(social_family, "social_proxy_lift"),
            "note": "difference between social and solo probe flip rates; narrower sycophancy/conformity proxy than alpha_tot",
        },
        {
            "name": "initial_disagreement_rate",
            "role": "Tang-style runtime disagreement proxy",
            "model_row": spearman(runtime_rows, "initial_disagreement_rate"),
            "family_mean": exact_spearman(aggregate_by_family(runtime_rows, ["c_cond_pct", "initial_disagreement_rate"]), "initial_disagreement_rate"),
            "note": "available only for six saved post-A6 raw-trace rows; measured after sampling three initial answers",
        },
        {
            "name": "r1_majority_changed_rate",
            "role": "Tang-style R1 trajectory proxy",
            "model_row": spearman(runtime_rows, "r1_majority_changed_rate"),
            "family_mean": exact_spearman(aggregate_by_family(runtime_rows, ["c_cond_pct", "r1_majority_changed_rate"]), "r1_majority_changed_rate"),
            "note": "available only for six saved post-A6 raw-trace rows; runtime diagnostic, not selection-time measurement",
        },
    ]
    return {
        "description": "Numerical comparator/proxy bakeoff from local artifacts. Pandey-circuit and exact Engels-gap scores are not present, so proxies are labeled explicitly.",
        "entries": entries,
        "partial_alpha_after_capability_proxy": partial_spearman(merged_rows, "alpha_tot", "c_cond_pct", ["engels_capability_pressure_proxy"]),
        "partial_alpha_after_capability_and_revision": partial_spearman(
            merged_rows, "alpha_tot", "c_cond_pct", ["engels_capability_pressure_proxy", "debate_revision_proxy"]
        ),
        "social_proxy_rows": social_rows,
        "social_proxy_missing_models": social_missing,
        "runtime_proxy_rows": runtime_rows,
        "unavailable_exact_scores": [
            "Pandey-2026 mechanistic sycophancy/circuit activations: no local *pandey* artifact or activation-cache score found.",
            "Exact Engels-2025 capability-gap forecast: local traces are homogeneous three-agent debates, not overseer/worker capability-gap panels.",
        ],
    }


def summarize_raw_alpha_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    full: list[float] = []
    default_all: list[float] = []
    default_agent0: list[float] = []
    default_agent_counts: dict[str, int] = defaultdict(int)
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            probes = row.get("probe_results") or []
            if not probes:
                continue
            fr = float(np.mean([1.0 if p.get("revised") else 0.0 for p in probes]))
            full.append(fr)
            if row.get("condition", "default") == "default":
                default_all.append(fr)
                agent = str(row.get("agent_idx"))
                default_agent_counts[agent] += 1
                if agent == "0":
                    default_agent0.append(fr)
    if not full:
        return None
    return {
        "full_all_conditions_all_agents": float(np.mean(full)),
        "default_condition_all_agents": float(np.mean(default_all)) if default_all else None,
        "strict_alpha_lite_agent0_default": float(np.mean(default_agent0)) if default_agent0 else None,
        "n_rows_full": len(full),
        "n_rows_default_all_agents": len(default_all),
        "n_rows_default_agent0": len(default_agent0),
        "default_agent_counts": dict(sorted(default_agent_counts.items())),
        "source": str(path.relative_to(ROOT)),
    }


def alpha_lite_retrospective(headline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Retrospective alpha-lite cost check from checked-in alpha artifacts.

    The strict one-agent alpha-lite estimate is available only for raw post-A6
    lanes. Older sealed rows retain aggregate default-condition alpha but not
    per-agent raw rows, so the N=14 comparison uses a default-condition
    all-agent reduction as the closest zero-API proxy.
    """
    alpha_split = read_json(RES / "alpha_split.json")["per_model"]
    rows: list[dict[str, Any]] = []
    for row in headline_rows:
        model = row["model"]
        entry = {
            "model": model,
            "family": row["family"],
            "headline_alpha_tot": float(row["alpha_tot"]),
            "c_cond_pct": float(row["c_cond_pct"]),
        }
        if model in ALPHA_SPLIT_NAME and ALPHA_SPLIT_NAME[model] in alpha_split:
            vals = alpha_split[ALPHA_SPLIT_NAME[model]]
            entry.update(
                {
                    "default_condition_all_agents": float(
                        (vals["alpha_total_nonsocial"] + vals["alpha_total_social"]) / 2.0
                    ),
                    "strict_alpha_lite_agent0_default": None,
                    "source": "alpha_split.json/per_model aggregate; per-agent raw rows unavailable in open checkout",
                }
            )
        else:
            summary = summarize_raw_alpha_file(RAW_ALPHA_FILE_BY_MODEL.get(model, Path("/nonexistent")))
            if summary is not None:
                entry.update(summary)
            else:
                entry.update(
                    {
                        "default_condition_all_agents": None,
                        "strict_alpha_lite_agent0_default": None,
                        "source": "missing raw alpha artifact",
                    }
                )
        rows.append(entry)

    default_rows = [r for r in rows if r.get("default_condition_all_agents") is not None]
    strict_rows = [r for r in rows if r.get("strict_alpha_lite_agent0_default") is not None]
    default_family = aggregate_by_family(default_rows, ["c_cond_pct", "headline_alpha_tot", "default_condition_all_agents"])
    strict_family = aggregate_by_family(strict_rows, ["c_cond_pct", "headline_alpha_tot", "strict_alpha_lite_agent0_default"])
    return {
        "description": "Zero-API retrospective alpha-lite check. N=14 uses default-condition all-agent alpha because older sealed rows lack per-agent raw rows; strict one-agent alpha-lite is available only on six post-A6 raw lanes.",
        "n14_default_condition_all_agents": {
            "model_row": spearman(default_rows, "default_condition_all_agents"),
            "family_mean": exact_spearman(default_family, "default_condition_all_agents"),
            "family_rows": default_family,
        },
        "post_a6_strict_one_agent_default": {
            "model_row": spearman(strict_rows, "strict_alpha_lite_agent0_default"),
            "family_mean": exact_spearman(strict_family, "strict_alpha_lite_agent0_default"),
            "family_rows": strict_family,
            "n_models": len(strict_rows),
        },
        "rows": rows,
        "interpretation": "The available N=14 default-condition reduction preserves the headline family rank correlation, but a strict N=14 one-agent alpha-lite replay cannot be claimed without recovering per-agent raw rows for the sealed lanes.",
    }


def qwen32_sensitivity(headline_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]]) -> dict[str, Any]:
    q = read_json(RES / "qwen3_32b_a7_n200_holdout_summary.json")
    qrow = {
        "model": "qwen3-32b",
        "family": "Qwen",
        "alpha_tot": float(q["alpha"]["alpha_tot"]),
        "c_cond_pct": float(q["debate"]["conditional_collapse_pct"]),
        "correction_pct": float(q["debate"]["conditional_correction_pct"]),
        "n_debates": int(q["debate"]["n_debates"]),
    }
    rows15 = headline_rows + [qrow]
    same_family = aggregate_by_family(rows15, ["alpha_tot", "c_cond_pct"])
    separate_rows = headline_rows + [{**qrow, "family": "Qwen3-32B-stress"}]
    separate_family = aggregate_by_family(separate_rows, ["alpha_tot", "c_cond_pct"])
    return {
        "qwen32_row": qrow,
        "model_row_n15": spearman(rows15, "alpha_tot"),
        "family_same_qwen_g7": exact_spearman(same_family, "alpha_tot"),
        "family_treat_as_stress_g8": exact_spearman(separate_family, "alpha_tot"),
        "same_qwen_family_rows": same_family,
        "stress_family_rows": separate_family,
        "note": "Post-hoc corrected-pool holdout; not folded into the primary pre-registered N=14 test.",
    }


def local_lomo_artifact_audit() -> dict[str, Any]:
    candidates = sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("**/*pilot*gated*lomo*.json"))
    candidates += sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("**/*lomo*.json"))
    candidates = sorted(set(candidates))
    has_pilot_gated_lomo = (RES / "pilot_gated_lomo.json").exists()
    records, sources = replay.load_debate_records()
    existing_replay = read_json(RES / "correction_preserving_policy_replay.json")
    existing_primary_sources = set(existing_replay["cohorts"]["eligible_all_primary"]["models"])
    existing_source_limits = {
        source: int(meta.get("usable_rows", 0))
        for source, meta in existing_replay.get("sources", {}).items()
        if source in existing_primary_sources
    }
    eligible_sources = existing_primary_sources or {
        source for source, meta in sources.items() if not meta.get("excluded_from_primary")
    }
    if existing_source_limits:
        counts: dict[str, int] = defaultdict(int)
        eligible = []
        for rec in records:
            if rec.source not in existing_source_limits:
                continue
            if counts[rec.source] >= existing_source_limits[rec.source]:
                continue
            eligible.append(rec)
            counts[rec.source] += 1
    else:
        eligible = [rec for rec in records if rec.source in eligible_sources]
    dg_summary, dg_rows = replay.evaluate_gate(eligible, lambda rec: not bool(rec.features.get("init_unanimous")))
    dg_summary["validation"] = "fixed_rule_on_available_saved_traces_not_6525_lomo"
    dg_summary["question_bootstrap_ci95"] = replay.bootstrap_eval(dg_rows)
    return {
        "requested_pool": "6,525-debate OSS LOMO pool from Appendix P / Table 17",
        "pilot_gated_lomo_json_present": has_pilot_gated_lomo,
        "candidate_lomo_json_files": candidates,
        "available_saved_trace_primary": {
            "n_records": len(eligible),
            "n_sources": len(eligible_sources),
            "sources": sorted(eligible_sources),
        },
        "disagreement_gate_available_trace_replay": dg_summary,
        "conclusion": (
            "The exact 6,525-row pilot-gated LOMO records are not present in this checkout; only the aggregate Table 17 numbers "
            "and a separate 1,255-record saved-trace replay cohort are available. Therefore a strict matched-tau LOMO DG/DRS comparison "
            "cannot be honestly claimed from local files without recovering the missing row-level pool."
        ),
    }


def fmt_stat(stat: dict[str, Any]) -> str:
    if stat.get("rho") is None:
        return "NA"
    if stat.get("p_value_method") == "exact_permutation":
        return (
            f"{stat['rho']:+.3f} (n={stat['n']}, exact p2={stat['p_two_sided_exact']:.3g}, "
            f"p1-{stat['observed_direction'][:3]}={stat['p_one_sided_in_observed_direction_exact']:.3g})"
        )
    return f"{stat['rho']:+.3f} (n={stat['n']}, p={stat['p_two_sided']:.3g})"


def fmt_interval(xs: list[float]) -> str:
    return f"[{xs[0]:+.2f}, {xs[1]:+.2f}]"


def write_markdown(result: dict[str, Any]) -> None:
    lofo = result["lofo_family_prediction"]
    comp = result["comparator_bakeoff"]
    lite = result["alpha_lite_retrospective"]
    qwen = result["qwen32_sensitivity"]
    audit = result["lomo_artifact_audit"]
    dg = audit["disagreement_gate_available_trace_replay"]
    lines = [
        "# Round-2 Reviewer Hardening",
        "",
        "Zero-cost analyses computed from checked-in artifacts only. These are rebuttal/camera-ready diagnostics, not new confirmatory tests.",
        "",
        "## Held-Out-Family LoFO Predictive Check",
        "",
        lofo["description"],
        "",
        f"Predicted-vs-observed Spearman: {fmt_stat(lofo['predicted_vs_observed_spearman'])}. 80% PI coverage: {lofo['coverage_80']['covered']}/{lofo['coverage_80']['total']}; 95% PI coverage: {lofo['coverage_95']['covered']}/{lofo['coverage_95']['total']}. Mean absolute rank error: {lofo['mean_absolute_rank_error']:.3f}.",
        "",
        "| Held-out family | alpha | Observed C^cond | Predicted C^cond | 80% PI | 95% PI | Rank err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in lofo["rows"]:
        lines.append(
            f"| {row['heldout_family']} | {row['alpha_tot']:.4f} | {row['observed_c_cond_pct']:.2f}% | "
            f"{row['predicted_c_cond_pct']:.2f}% | {fmt_interval(row['pi80'])} | {fmt_interval(row['pi95'])} | "
            f"{row['absolute_rank_error']} |"
        )

    lines.extend(
        [
            "",
            "## Comparator / Proxy Bakeoff",
            "",
            comp["description"],
            "",
            "| Score | Role | Model-row Spearman vs C^cond | Family Spearman vs C^cond | Note |",
            "|---|---|---:|---:|---|",
        ]
    )
    for entry in comp["entries"]:
        lines.append(
            f"| `{entry['name']}` | {entry['role']} | {fmt_stat(entry['model_row'])} | {fmt_stat(entry['family_mean'])} | {entry['note']} |"
        )
    lines.extend(
        [
            "",
            f"Partial alpha after capability-pressure proxy: {fmt_stat(comp['partial_alpha_after_capability_proxy'])}.",
            f"Partial alpha after capability-pressure + raw-revision proxy: {fmt_stat(comp['partial_alpha_after_capability_and_revision'])}.",
            "",
            "Unavailable exact scores:",
        ]
    )
    lines.extend(f"- {note}" for note in comp["unavailable_exact_scores"])

    lines.extend(
        [
            "",
            "## Alpha-Lite Retrospective Cost Check",
            "",
            lite["description"],
            "",
            "| Check | Spearman vs C^cond | Scope |",
            "|---|---:|---|",
            f"| N=14 default-condition reduction | {fmt_stat(lite['n14_default_condition_all_agents']['family_mean'])} | all headline rows; sealed rows are aggregate, not per-agent |",
            f"| Post-A6 strict one-agent alpha-lite | {fmt_stat(lite['post_a6_strict_one_agent_default']['family_mean'])} | {lite['post_a6_strict_one_agent_default']['n_models']} raw rows across three families |",
            "",
            lite["interpretation"],
        ]
    )

    lines.extend(
        [
            "",
            "## Qwen3-32B Sensitivity",
            "",
            f"Qwen3-32B corrected-pool holdout: alpha={qwen['qwen32_row']['alpha_tot']:.4f}, C^cond={qwen['qwen32_row']['c_cond_pct']:.2f}%, correction={qwen['qwen32_row']['correction_pct']:.2f}% over {qwen['qwen32_row']['n_debates']} debates.",
            "",
            "| Sensitivity | Spearman | Note |",
            "|---|---:|---|",
            f"| Model row N=15 | {fmt_stat(qwen['model_row_n15'])} | Qwen3-32B appended as a model row |",
            f"| Family G=7, Qwen updated | {fmt_stat(qwen['family_same_qwen_g7'])} | Qwen3-32B averaged into Qwen family |",
            f"| Stress-family G=8 | {fmt_stat(qwen['family_treat_as_stress_g8'])} | Qwen3-32B shown as separate post-hoc stress row |",
            "",
            "## Same-Pool LOMO Gate Audit",
            "",
            audit["conclusion"],
            "",
            f"Available saved-trace replay cohort: {audit['available_saved_trace_primary']['n_records']} records across {audit['available_saved_trace_primary']['n_sources']} sources.",
            f"DisagreementGate on that available cohort: delta={dg['acc_delta'] * 100:+.2f}pp, prevented={dg['collapses_prevented']}, lost={dg['corrections_lost']}, net={dg['net']}, gate={dg['gate_rate'] * 100:.1f}%.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    family_artifact = read_json(RES / "family_level_headline_aggregation.json")
    hardening = read_json(RES / "reviewer_hardening_analyses.json")
    headline_rows = [dict(r) for r in family_artifact["per_model_rows"]]
    family_rows = [dict(r) for r in family_artifact["aggregations"]["mean"]["families"]]
    result = {
        "description": "Round-2 zero-cost reviewer hardening analyses.",
        "inputs": {
            "family_level": "abc_exp/results/family_level_headline_aggregation.json",
            "reviewer_hardening": "abc_exp/results/reviewer_hardening_analyses.json",
            "qwen32": "abc_exp/results/qwen3_32b_a7_n200_holdout_summary.json",
            "correction_replay": "abc_exp/results/correction_preserving_policy_replay.json",
            "alpha_split": "abc_exp/results/alpha_split.json",
            "raw_post_a6_alpha_files": [str(path.relative_to(ROOT)) for path in RAW_ALPHA_FILE_BY_MODEL.values()],
        },
        "lofo_family_prediction": lofo_family_prediction(family_rows),
        "comparator_bakeoff": comparator_bakeoff(headline_rows, hardening["per_model"]),
        "alpha_lite_retrospective": alpha_lite_retrospective(headline_rows),
        "qwen32_sensitivity": qwen32_sensitivity(headline_rows, family_rows),
        "lomo_artifact_audit": local_lomo_artifact_audit(),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_markdown(result)
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
