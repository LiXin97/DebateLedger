"""Zero-cost correction-preserving policy replay on saved debate traces.

The replay question is deliberately narrow: if a policy fires after observing
only R1 dynamics, replace the standard final majority answer with the initial
majority answer. This can prevent majority-level collapses, but it can also
discard genuine corrections. We therefore report both sides of the ledger.

Inputs:
  - abc_exp/results/debate_traces_*.jsonl
  - abc_exp/results/sa_causal_*.jsonl, when a per-question probe mean flip-rate
    can be matched to the debate trace source.

Policies:
  - standard: saved final majority.
  - naive_always_freeze: always revert to initial majority.
  - naive_r1_majority_changed: revert if R1 majority differs from initial.
  - naive_any_r1_flip: revert if any agent changes answer in R1.
  - learned_r1_stump: leave-one-model-out percentile/stump rule over R1-only
    features. The training objective is net accuracy benefit with a small gate
    penalty, and the no-gate rule is always an allowed training outcome.
  - learned_r1_probe_stump: same as above, with optional per-question probe
    mean flip-rate features when available.

Outputs:
  - abc_exp/results/correction_preserving_policy_replay.json
  - abc_exp/results/CORRECTION_PRESERVING_POLICY_REPLAY.md

This script does not run any models and does not edit the paper.
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RES = Path(os.environ.get("ABC_RESULTS", ROOT / "results"))
SEED = 20260426
BOOTSTRAPS = 2000
MIN_PRIMARY_ROWS = 50
MIN_LOMO_MODELS = 4
GATE_PENALTY = 0.02


def majority(answers: list[str | None]) -> str:
    valid = [str(a).strip() for a in answers if a is not None and str(a).strip()]
    if not valid:
        return ""
    counts = Counter(valid)
    top = max(counts.values())
    return sorted(a for a, c in counts.items() if c == top)[0]


def clean_answer(answer: Any) -> str:
    if answer is None:
        return ""
    return str(answer).strip()


def source_from_trace_path(path: Path) -> str:
    return path.stem.replace("debate_traces_", "", 1)


def source_from_probe_path(path: Path) -> str:
    return path.stem.replace("sa_causal_", "", 1)


def normalized_source(source: str) -> str:
    s = source.lower()
    for prefix in ("debate_traces_", "sa_causal_", "openrouter_", "router_", "vllm_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in ("_smoke", "-smoke", "-awq"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def model_display_name(row: dict[str, Any], source: str) -> str:
    name = row.get("model_name") or row.get("model") or source
    return str(name).replace("Qwen/", "").replace("google/", "")


def round_answers(row: dict[str, Any], round_index: int) -> list[str]:
    traces = row.get("round_traces") or []
    if round_index >= len(traces):
        return []
    round_trace = traces[round_index]
    if isinstance(round_trace, dict):
        return [clean_answer(a) for a in round_trace.get("answers") or []]
    if isinstance(round_trace, list):
        def sort_key(item: dict[str, Any]) -> tuple[int, str]:
            raw = str(item.get("agent_id", ""))
            return (int(raw), raw) if raw.isdigit() else (10_000, raw)

        return [clean_answer(item.get("answer")) for item in sorted(round_trace, key=sort_key)]
    return []


def round_majority(row: dict[str, Any], round_index: int) -> str:
    traces = row.get("round_traces") or []
    if round_index < len(traces) and isinstance(traces[round_index], dict):
        existing = traces[round_index].get("majority")
        if existing:
            return clean_answer(existing)
    return majority(round_answers(row, round_index))


def safe_frac(num: float, den: float) -> float:
    return num / den if den else 0.0


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


@dataclass
class DebateRecord:
    source: str
    model: str
    backend: str
    question_id: str
    correct_label: str
    initial_majority: str
    standard_majority: str
    initial_correct: int
    standard_correct: int
    collapse: int
    correction: int
    features: dict[str, float | None]

    @property
    def freeze_delta(self) -> int:
        return self.initial_correct - self.standard_correct


def featurize_trace(row: dict[str, Any], source: str, probe_mean_fr: float | None) -> DebateRecord | None:
    initial_answers = [clean_answer(a) for a in row.get("initial_answers") or []]
    final_answers = [clean_answer(a) for a in row.get("final_answers") or []]
    if not initial_answers or not final_answers:
        return None

    correct = clean_answer(row.get("correct_label"))
    if not correct:
        return None

    init_majority = majority(initial_answers)
    standard_majority = clean_answer(row.get("majority_answer")) or majority(final_answers)
    if not init_majority or not standard_majority:
        return None

    r1_answers = round_answers(row, 0)
    r1_majority = round_majority(row, 0) if r1_answers else ""
    n_agents = max(len(initial_answers), 1)
    init_counts = Counter(a for a in initial_answers if a)
    r1_counts = Counter(a for a in r1_answers if a)
    init_agree_frac = safe_frac(max(init_counts.values()) if init_counts else 0, n_agents)
    r1_agree_frac = safe_frac(max(r1_counts.values()) if r1_counts else 0, max(len(r1_answers), 1))
    r1_n_flipped = sum(
        1 for before, after in zip(initial_answers, r1_answers) if before and after and before != after
    )
    r1_majority_changed = int(bool(r1_majority) and r1_majority != init_majority)
    r1_flip_frac = safe_frac(r1_n_flipped, n_agents)
    init_unanimous = int(len(init_counts) <= 1)
    r1_unanimous = int(len(r1_counts) <= 1) if r1_answers else 0
    probe = probe_mean_fr

    features: dict[str, float | None] = {
        "init_agree_frac": init_agree_frac,
        "init_unanimous": float(init_unanimous),
        "r1_agree_frac": r1_agree_frac,
        "r1_unanimous": float(r1_unanimous),
        "r1_majority_changed": float(r1_majority_changed),
        "r1_n_flipped": float(r1_n_flipped),
        "r1_flip_frac": r1_flip_frac,
        "r1_consensus_gain": r1_agree_frac - init_agree_frac,
        "r1_changed_x_agree": float(r1_majority_changed) * r1_agree_frac,
        "r1_changed_x_init_split": float(r1_majority_changed) * (1.0 - init_agree_frac),
        "r1_changed_x_nonunanimous": float(r1_majority_changed) * (1.0 - float(r1_unanimous)),
        "probe_mean_fr": probe,
        "probe_x_r1_changed": None if probe is None else probe * float(r1_majority_changed),
        "probe_x_r1_flip_frac": None if probe is None else probe * r1_flip_frac,
        "probe_x_init_split": None if probe is None else probe * (1.0 - init_agree_frac),
    }

    initial_correct = int(init_majority == correct)
    standard_correct = int(standard_majority == correct)
    return DebateRecord(
        source=source,
        model=model_display_name(row, source),
        backend=str(row.get("backend") or source.split("_", 1)[0]),
        question_id=str(row.get("question_id") or ""),
        correct_label=correct,
        initial_majority=init_majority,
        standard_majority=standard_majority,
        initial_correct=initial_correct,
        standard_correct=standard_correct,
        collapse=int(initial_correct == 1 and standard_correct == 0),
        correction=int(initial_correct == 0 and standard_correct == 1),
        features=features,
    )


def load_probe_means() -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    by_source: dict[str, dict[str, float]] = {}
    norm_to_source: dict[str, str] = {}
    for path in sorted(RES.glob("sa_causal_*.jsonl")):
        source = source_from_probe_path(path)
        values: dict[str, list[float]] = defaultdict(list)
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("condition") != "default":
                    continue
                qid = str(row.get("question_id") or "")
                fr = row.get("flip_rate")
                if qid and isinstance(fr, (int, float)):
                    values[qid].append(float(fr))
        if values:
            by_source[source] = {qid: sum(v) / len(v) for qid, v in values.items()}
            norm_to_source.setdefault(normalized_source(source), source)
    return by_source, norm_to_source


def matched_probe_source(trace_source: str, probe_norms: dict[str, str]) -> str | None:
    norm = normalized_source(trace_source)
    if norm in probe_norms:
        return probe_norms[norm]
    candidates = [probe_source for probe_norm, probe_source in probe_norms.items() if norm in probe_norm or probe_norm in norm]
    if not candidates:
        return None
    return sorted(candidates, key=len)[0]


def load_debate_records() -> tuple[list[DebateRecord], dict[str, Any]]:
    probe_means, probe_norms = load_probe_means()
    records: list[DebateRecord] = []
    sources: dict[str, Any] = {}
    for path in sorted(RES.glob("debate_traces_*.jsonl")):
        source = source_from_trace_path(path)
        probe_source = matched_probe_source(source, probe_norms)
        probe_by_q = probe_means.get(probe_source or "", {})
        loaded = 0
        usable = 0
        probe_hits = 0
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                loaded += 1
                row = json.loads(line)
                qid = str(row.get("question_id") or "")
                rec = featurize_trace(row, source, probe_by_q.get(qid))
                if rec is not None:
                    records.append(rec)
                    usable += 1
                    probe_hits += int(qid in probe_by_q)
        sources[source] = {
            "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            "loaded_rows": loaded,
            "usable_rows": usable,
            "probe_source": probe_source,
            "probe_questions": len(probe_by_q),
            "probe_matched_rows": probe_hits,
            "probe_matched_row_rate": probe_hits / usable if usable else 0.0,
            "excluded_from_primary": source.startswith("smoke_") or loaded < MIN_PRIMARY_ROWS,
        }
    return records, sources


@dataclass
class EvalRow:
    source: str
    model: str
    question_id: str
    standard_correct: int
    policy_correct: int
    collapse: int
    correction: int
    gated: int
    prevented: int
    lost: int


GateFn = Callable[[DebateRecord], bool]


def evaluate_gate(records: list[DebateRecord], gate_fn: GateFn) -> tuple[dict[str, Any], list[EvalRow]]:
    rows: list[EvalRow] = []
    for rec in records:
        gated = int(gate_fn(rec))
        policy_correct = rec.initial_correct if gated else rec.standard_correct
        prevented = int(gated and rec.collapse)
        lost = int(gated and rec.correction)
        rows.append(
            EvalRow(
                source=rec.source,
                model=rec.model,
                question_id=rec.question_id,
                standard_correct=rec.standard_correct,
                policy_correct=policy_correct,
                collapse=rec.collapse,
                correction=rec.correction,
                gated=gated,
                prevented=prevented,
                lost=lost,
            )
        )
    return summarize_eval(rows), rows


def summarize_eval(rows: list[EvalRow]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {
            "n": 0,
            "standard_acc": None,
            "policy_acc": None,
            "acc_delta": None,
            "gate_rate": None,
            "collapses_prevented": 0,
            "corrections_lost": 0,
            "net": 0,
        }
    standard_correct = sum(r.standard_correct for r in rows)
    policy_correct = sum(r.policy_correct for r in rows)
    prevented = sum(r.prevented for r in rows)
    lost = sum(r.lost for r in rows)
    gated = sum(r.gated for r in rows)
    return {
        "n": n,
        "standard_acc": standard_correct / n,
        "policy_acc": policy_correct / n,
        "acc_delta": (policy_correct - standard_correct) / n,
        "gate_count": gated,
        "gate_rate": gated / n,
        "collapses_prevented": prevented,
        "corrections_lost": lost,
        "net": prevented - lost,
        "standard_collapses": sum(r.collapse for r in rows),
        "standard_corrections": sum(r.correction for r in rows),
    }


def summarize_standard(records: list[DebateRecord]) -> dict[str, Any]:
    n = len(records)
    standard_correct = sum(r.standard_correct for r in records)
    return {
        "n": n,
        "standard_acc": standard_correct / n if n else None,
        "policy_acc": standard_correct / n if n else None,
        "acc_delta": 0.0 if n else None,
        "gate_count": 0,
        "gate_rate": 0.0 if n else None,
        "collapses_prevented": 0,
        "corrections_lost": 0,
        "net": 0,
        "standard_collapses": sum(r.collapse for r in records),
        "standard_corrections": sum(r.correction for r in records),
    }


def bootstrap_eval(rows: list[EvalRow], b: int = BOOTSTRAPS, seed: int = SEED) -> dict[str, Any] | None:
    by_q: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_q[row.question_id].append(row)
    qids = sorted(q for q in by_q if q)
    if len(qids) < 10:
        return None
    rng = random.Random(seed)
    acc_deltas: list[float] = []
    gate_rates: list[float] = []
    net_rates: list[float] = []
    prevented_rates: list[float] = []
    lost_rates: list[float] = []
    for _ in range(b):
        sampled_rows: list[EvalRow] = []
        for _ in qids:
            sampled_rows.extend(by_q[rng.choice(qids)])
        if not sampled_rows:
            continue
        n = len(sampled_rows)
        standard = sum(r.standard_correct for r in sampled_rows)
        policy = sum(r.policy_correct for r in sampled_rows)
        gated = sum(r.gated for r in sampled_rows)
        prevented = sum(r.prevented for r in sampled_rows)
        lost = sum(r.lost for r in sampled_rows)
        acc_deltas.append((policy - standard) / n)
        gate_rates.append(gated / n)
        net_rates.append((prevented - lost) / n)
        prevented_rates.append(prevented / n)
        lost_rates.append(lost / n)

    def ci(values: list[float]) -> dict[str, float]:
        return {
            "median": percentile(values, 50),
            "ci_2.5": percentile(values, 2.5),
            "ci_97.5": percentile(values, 97.5),
        }

    return {
        "B": b,
        "B_used": len(acc_deltas),
        "n_question_clusters": len(qids),
        "acc_delta": ci(acc_deltas),
        "gate_rate": ci(gate_rates),
        "net_rate": ci(net_rates),
        "prevented_rate": ci(prevented_rates),
        "lost_rate": ci(lost_rates),
    }


R1_FEATURES = [
    "init_agree_frac",
    "init_unanimous",
    "r1_agree_frac",
    "r1_unanimous",
    "r1_majority_changed",
    "r1_n_flipped",
    "r1_flip_frac",
    "r1_consensus_gain",
    "r1_changed_x_agree",
    "r1_changed_x_init_split",
    "r1_changed_x_nonunanimous",
]

PROBE_FEATURES = [
    "probe_mean_fr",
    "probe_x_r1_changed",
    "probe_x_r1_flip_frac",
    "probe_x_init_split",
]


def feature_values(records: list[DebateRecord], feature: str, fill: float | None = None) -> list[float]:
    vals: list[float] = []
    observed = [r.features.get(feature) for r in records if r.features.get(feature) is not None]
    replacement = fill if fill is not None else (sum(float(v) for v in observed) / len(observed) if observed else 0.0)
    for rec in records:
        value = rec.features.get(feature)
        vals.append(float(replacement if value is None else value))
    return vals


def candidate_thresholds(values: list[float]) -> list[tuple[float, str]]:
    thresholds: list[tuple[float, str]] = []
    for pct in range(0, 101, 5):
        thresholds.append((percentile(values, pct), f"p{pct}"))
    unique = sorted(set(values))
    if len(unique) <= 25:
        thresholds.extend((v, "unique") for v in unique)
    dedup: dict[float, str] = {}
    for value, label in thresholds:
        if not math.isnan(value):
            dedup.setdefault(round(value, 12), label)
    return [(value, label) for value, label in dedup.items()]


def gate_mask_from_rule(values: list[float], direction: str, threshold: float) -> list[int]:
    if direction == "high":
        return [int(v >= threshold) for v in values]
    return [int(v <= threshold) for v in values]


def score_rule(mask: list[int], records: list[DebateRecord], gate_penalty: float) -> dict[str, Any]:
    n = len(records)
    prevented = sum(int(m and r.collapse) for m, r in zip(mask, records))
    lost = sum(int(m and r.correction) for m, r in zip(mask, records))
    gated = sum(mask)
    net = prevented - lost
    net_rate = net / n if n else 0.0
    gate_rate = gated / n if n else 0.0
    return {
        "objective": net_rate - gate_penalty * gate_rate,
        "net_rate": net_rate,
        "gate_rate": gate_rate,
        "prevented": prevented,
        "lost": lost,
        "net": net,
        "gated": gated,
    }


def train_percentile_stump(
    records: list[DebateRecord],
    features: list[str],
    gate_penalty: float = GATE_PENALTY,
) -> dict[str, Any]:
    best: dict[str, Any] = {
        "kind": "no_gate",
        "objective": 0.0,
        "net_rate": 0.0,
        "gate_rate": 0.0,
        "prevented": 0,
        "lost": 0,
        "net": 0,
        "gated": 0,
        "feature": None,
        "direction": None,
        "threshold": None,
        "threshold_label": None,
        "fill_value": None,
    }

    for feature in features:
        observed = [r.features.get(feature) for r in records if r.features.get(feature) is not None]
        if not observed:
            continue
        fill = sum(float(v) for v in observed) / len(observed)
        values = feature_values(records, feature, fill=fill)
        for threshold, threshold_label in candidate_thresholds(values):
            for direction in ("high", "low"):
                mask = gate_mask_from_rule(values, direction, threshold)
                scored = score_rule(mask, records, gate_penalty)
                candidate = {
                    **scored,
                    "kind": "stump",
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "threshold_label": threshold_label,
                    "fill_value": fill,
                }
                candidate_key = (
                    candidate["objective"],
                    candidate["net_rate"],
                    -candidate["lost"],
                    -candidate["gate_rate"],
                )
                best_key = (
                    best["objective"],
                    best["net_rate"],
                    -best["lost"],
                    -best["gate_rate"],
                )
                if candidate_key > best_key:
                    best = candidate
    best["gate_penalty"] = gate_penalty
    best["n_train"] = len(records)
    return best


def apply_trained_rule(rule: dict[str, Any], rec: DebateRecord) -> bool:
    if rule.get("kind") == "no_gate" or not rule.get("feature"):
        return False
    value = rec.features.get(str(rule["feature"]))
    if value is None:
        value = rule.get("fill_value", 0.0)
    threshold = float(rule["threshold"])
    if rule.get("direction") == "high":
        return float(value) >= threshold
    return float(value) <= threshold


def evaluate_lomo_stump(
    records: list[DebateRecord],
    model_key: Callable[[DebateRecord], str],
    features: list[str],
    mode: str,
) -> tuple[dict[str, Any], list[EvalRow], list[dict[str, Any]]]:
    models = sorted({model_key(r) for r in records})
    all_eval_rows: list[EvalRow] = []
    folds: list[dict[str, Any]] = []
    if len(models) >= MIN_LOMO_MODELS:
        for heldout in models:
            train = [r for r in records if model_key(r) != heldout]
            test = [r for r in records if model_key(r) == heldout]
            rule = train_percentile_stump(train, features)
            summary, eval_rows = evaluate_gate(test, lambda rec, trained=rule: apply_trained_rule(trained, rec))
            all_eval_rows.extend(eval_rows)
            folds.append({"heldout_model": heldout, "train_models": [m for m in models if m != heldout], "rule": rule, "test_summary": summary})
        summary = summarize_eval(all_eval_rows)
        summary["validation"] = "strict_leave_one_model_out"
    else:
        rule = train_percentile_stump(records, features)
        summary, all_eval_rows = evaluate_gate(records, lambda rec, trained=rule: apply_trained_rule(trained, rec))
        summary["validation"] = "exploratory_in_sample_not_lomo"
        folds.append({"heldout_model": None, "train_models": models, "rule": rule, "test_summary": summary})
    summary["mode"] = mode
    summary["n_models"] = len(models)
    return summary, all_eval_rows, folds


def policy_table_entry(name: str, summary: dict[str, Any], eval_rows: list[EvalRow] | None = None) -> dict[str, Any]:
    entry = {"policy": name, **summary}
    if eval_rows:
        entry["question_bootstrap_ci95"] = bootstrap_eval(eval_rows)
    return entry


def cohort_summary(name: str, records: list[DebateRecord]) -> dict[str, Any]:
    models = sorted({r.source for r in records})
    standard = summarize_standard(records)
    policies: list[dict[str, Any]] = [policy_table_entry("standard", standard)]

    naive_specs: list[tuple[str, GateFn]] = [
        ("naive_always_freeze", lambda r: True),
        ("naive_r1_majority_changed", lambda r: bool(r.features.get("r1_majority_changed"))),
        ("naive_any_r1_flip", lambda r: float(r.features.get("r1_n_flipped") or 0.0) > 0.0),
    ]
    for policy_name, gate_fn in naive_specs:
        summary, rows = evaluate_gate(records, gate_fn)
        summary["validation"] = "fixed_rule_no_training"
        policies.append(policy_table_entry(policy_name, summary, rows))

    learned_r1, rows_r1, folds_r1 = evaluate_lomo_stump(records, lambda r: r.source, R1_FEATURES, "r1_only")
    policies.append(policy_table_entry("learned_r1_percentile_stump", learned_r1, rows_r1))

    learned_probe, rows_probe, folds_probe = evaluate_lomo_stump(
        records, lambda r: r.source, R1_FEATURES + PROBE_FEATURES, "r1_plus_optional_probe_mean_fr"
    )
    policies.append(policy_table_entry("learned_r1_probe_percentile_stump", learned_probe, rows_probe))

    best_lomo = [
        p for p in policies
        if p.get("validation") == "strict_leave_one_model_out" and p.get("policy") != "standard"
    ]
    paper_worthy = False
    paper_worthy_reason = "No strict LOMO policy has a positive question-bootstrap lower CI on accuracy delta."
    if best_lomo:
        best = max(best_lomo, key=lambda p: (p.get("acc_delta") or 0.0, p.get("net") or 0))
        ci = (best.get("question_bootstrap_ci95") or {}).get("acc_delta") if best.get("question_bootstrap_ci95") else None
        if ci and ci.get("ci_2.5", -1.0) > 0 and (best.get("net") or 0) > 0:
            paper_worthy = True
            paper_worthy_reason = f"{best['policy']} has positive strict-LOMO delta and bootstrap CI excludes zero."

    return {
        "cohort": name,
        "n_records": len(records),
        "n_models": len(models),
        "models": models,
        "standard_collapses": sum(r.collapse for r in records),
        "standard_corrections": sum(r.correction for r in records),
        "probe_feature_coverage": sum(1 for r in records if r.features.get("probe_mean_fr") is not None) / len(records) if records else 0.0,
        "policies": policies,
        "learned_folds": {
            "learned_r1_percentile_stump": folds_r1,
            "learned_r1_probe_percentile_stump": folds_probe,
        },
        "paper_worthy": paper_worthy,
        "paper_worthy_reason": paper_worthy_reason,
    }


def build_cohorts(records: list[DebateRecord], sources: dict[str, Any]) -> dict[str, list[DebateRecord]]:
    eligible_sources = {
        source for source, meta in sources.items()
        if not meta.get("excluded_from_primary")
    }
    eligible = [r for r in records if r.source in eligible_sources]
    cohorts: dict[str, list[DebateRecord]] = {"eligible_all_primary": eligible}
    by_backend: dict[str, list[DebateRecord]] = defaultdict(list)
    for rec in eligible:
        by_backend[rec.backend].append(rec)
    for backend, backend_records in sorted(by_backend.items()):
        suffix = "lomo" if len({r.source for r in backend_records}) >= MIN_LOMO_MODELS else "exploratory"
        cohorts[f"backend_{backend}_{suffix}"] = backend_records
    exploratory_sources = {source for source, meta in sources.items() if meta.get("excluded_from_primary")}
    exploratory = [r for r in records if r.source in exploratory_sources]
    if exploratory:
        cohorts["excluded_tiny_or_smoke_exploratory"] = exploratory
    return cohorts


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_pp(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:+.{digits}f} pp"


def markdown_table_for_policies(cohort: dict[str, Any]) -> list[str]:
    lines = ["| Policy | Validation | Gate | Delta vs standard | CI95 delta | Prevented | Lost | Net |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in cohort["policies"]:
        ci = policy.get("question_bootstrap_ci95")
        ci_text = "NA"
        if ci and ci.get("acc_delta"):
            acc_ci = ci["acc_delta"]
            ci_text = f"[{fmt_pp(acc_ci['ci_2.5'])}, {fmt_pp(acc_ci['ci_97.5'])}]"
        lines.append(
            "| {policy} | {validation} | {gate} | {delta} | {ci} | {prevented} | {lost} | {net} |".format(
                policy=policy["policy"],
                validation=policy.get("validation", "fixed"),
                gate=fmt_pct(policy.get("gate_rate")),
                delta=fmt_pp(policy.get("acc_delta")),
                ci=ci_text,
                prevented=policy.get("collapses_prevented", 0),
                lost=policy.get("corrections_lost", 0),
                net=policy.get("net", 0),
            )
        )
    return lines


def write_markdown(result: dict[str, Any], out_path: Path) -> None:
    primary = result["cohorts"].get("eligible_all_primary")
    lines: list[str] = [
        "# Correction-Preserving Policy Replay",
        "",
        "Zero-cost replay over saved `debate_traces_*.jsonl`: a gate observes only within-trace R1 features, and when it fires the final answer is replayed as the initial majority. This prevents majority-level collapses only when the initial majority was correct, and loses corrections when the initial majority was wrong but standard debate ended correct.",
        "",
        f"Gate penalty used for learned stumps: `{GATE_PENALTY}` per gated debate in the training objective. Learned policies include a no-gate option, and use strict leave-one-model-out when the cohort has at least {MIN_LOMO_MODELS} trace sources.",
        "",
        "## Inputs",
        "",
        "| Source | Rows | Usable | Probe | Probe overlap | Primary? |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for source, meta in sorted(result["sources"].items()):
        primary_flag = "no" if meta["excluded_from_primary"] else "yes"
        lines.append(
            f"| {source} | {meta['loaded_rows']} | {meta['usable_rows']} | {meta.get('probe_source') or 'none'} | {meta['probe_matched_rows']} ({fmt_pct(meta['probe_matched_row_rate'])}) | {primary_flag} |"
        )
    if primary:
        lines.extend([
            "",
            "## Primary Cohort",
            "",
            f"Models: {primary['n_models']}; debates: {primary['n_records']}; standard collapses: {primary['standard_collapses']}; standard corrections: {primary['standard_corrections']}; probe coverage: {fmt_pct(primary['probe_feature_coverage'])}.",
            "",
        ])
        lines.extend(markdown_table_for_policies(primary))
        lines.extend(["", f"Verdict: {'paper-worthy' if primary['paper_worthy'] else 'not paper-worthy as a main claim'}; {primary['paper_worthy_reason']}"])

    for cohort_name, cohort in result["cohorts"].items():
        if cohort_name == "eligible_all_primary":
            continue
        lines.extend([
            "",
            f"## {cohort_name}",
            "",
            f"Models: {cohort['n_models']}; debates: {cohort['n_records']}; standard collapses: {cohort['standard_collapses']}; standard corrections: {cohort['standard_corrections']}; probe coverage: {fmt_pct(cohort['probe_feature_coverage'])}.",
            "",
        ])
        lines.extend(markdown_table_for_policies(cohort))
        has_lomo = any(p.get("validation") == "strict_leave_one_model_out" for p in cohort["policies"])
        verdict_label = "paper-worthy" if cohort["paper_worthy"] else ("not paper-worthy" if has_lomo else "exploratory/not paper-worthy")
        lines.extend(["", f"Verdict: {verdict_label}; {cohort['paper_worthy_reason']}"])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "A positive net means the replay prevented more standard-debate collapses than it discarded corrections. The paper-worthy bar used here is stricter: a strict-LOMO learned policy must have positive net and a question-bootstrap 95% CI for accuracy delta entirely above zero.",
    ])
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    records, sources = load_debate_records()
    cohorts = build_cohorts(records, sources)
    cohort_results = {name: cohort_summary(name, recs) for name, recs in cohorts.items() if recs}
    result = {
        "analysis": "correction_preserving_policy_replay",
        "seed": SEED,
        "bootstrap_B": BOOTSTRAPS,
        "min_lomo_models": MIN_LOMO_MODELS,
        "min_primary_rows_per_source": MIN_PRIMARY_ROWS,
        "gate_penalty": GATE_PENALTY,
        "method_note": (
            "Replay is post-hoc and zero-cost: a fired gate substitutes the initial majority for "
            "the saved standard final majority. Metrics are majority-level, not per-agent."
        ),
        "sources": sources,
        "cohorts": cohort_results,
    }
    out_json = RES / "correction_preserving_policy_replay.json"
    out_md = RES / "CORRECTION_PRESERVING_POLICY_REPLAY.md"
    out_json.write_text(json.dumps(result, indent=2, default=float) + "\n")
    write_markdown(result, out_md)

    primary = cohort_results.get("eligible_all_primary", {})
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if primary:
        print("\nPrimary cohort:")
        for policy in primary["policies"]:
            print(
                f"  {policy['policy']:<36} gate={fmt_pct(policy.get('gate_rate')):>7} "
                f"delta={fmt_pp(policy.get('acc_delta')):>10} "
                f"prevented={policy.get('collapses_prevented', 0):>3} "
                f"lost={policy.get('corrections_lost', 0):>3} net={policy.get('net', 0):>+4}"
            )
        print(f"Verdict: {'paper-worthy' if primary.get('paper_worthy') else 'not paper-worthy'}")
        print(primary.get("paper_worthy_reason"))


if __name__ == "__main__":
    main()
