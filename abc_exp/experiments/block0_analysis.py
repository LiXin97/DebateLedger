"""Block 0 Analysis: GO/NO-GO Decision.

Loads Block 0.1 (alpha estimation) and Block 0.2 (debate) results.
Computes:
1. Alpha distribution statistics
2. Test-retest reliability (ICC) -- requires retest data
3. Cronbach's alpha across probe types
4. Debate collapse rate
5. Logistic regression: P(collapse | alpha profile) -> AUC-ROC
6. GO/NO-GO recommendation

Usage:
    python -m abc_exp.experiments.block0_analysis [--results_dir abc_exp/results]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from abc_exp.src.eval.stats import (
    icc_2_1,
    cronbach_alpha,
    compute_auc,
    logistic_regression,
)


# ======================================================================
# Data loading
# ======================================================================

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    if not path.exists():
        return records
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_alpha_records(results_dir: Path) -> list[dict]:
    """Load Block 0.1 alpha estimation results."""
    path = results_dir / "block0_alpha.jsonl"
    records = load_jsonl(path)
    print(f"Loaded {len(records)} alpha records from {path}")
    return records


def load_alpha_retest_records(results_dir: Path) -> list[dict]:
    """Load Block 0.1 retest alpha results (if available)."""
    path = results_dir / "block0_alpha_retest.jsonl"
    records = load_jsonl(path)
    if records:
        print(f"Loaded {len(records)} retest alpha records from {path}")
    else:
        print(f"No retest data found at {path}")
    return records


def load_debate_records(results_dir: Path) -> list[dict]:
    """Load Block 0.2 debate results."""
    path = results_dir / "block0_debates.jsonl"
    records = load_jsonl(path)
    print(f"Loaded {len(records)} debate records from {path}")
    return records


# ======================================================================
# 1. Alpha distribution analysis
# ======================================================================

def analyze_alpha_distribution(alpha_records: list[dict]) -> dict:
    """Compute alpha distribution statistics across all agents and questions."""
    all_alphas = []
    per_agent: dict[str, list[float]] = defaultdict(list)

    for record in alpha_records:
        estimates = record.get("alpha_estimates", {})
        for agent_id, est in estimates.items():
            alpha_val = est.get("alpha", None)
            if alpha_val is not None:
                all_alphas.append(alpha_val)
                per_agent[agent_id].append(alpha_val)

    if not all_alphas:
        return {"error": "No alpha values found"}

    arr = np.array(all_alphas)
    stats = {
        "n_values": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "per_agent": {},
    }

    for agent_id in sorted(per_agent.keys()):
        a = np.array(per_agent[agent_id])
        stats["per_agent"][agent_id] = {
            "n": len(a),
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "median": float(np.median(a)),
        }

    return stats


def print_histogram(values: list[float], n_bins: int = 10, width: int = 40) -> None:
    """Print a simple ASCII histogram."""
    if not values:
        print("  (no data)")
        return

    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=n_bins)
    max_count = max(counts) if max(counts) > 0 else 1

    for i in range(len(counts)):
        bar_len = int(counts[i] / max_count * width)
        bar = "#" * bar_len
        print(f"  [{edges[i]:5.2f}, {edges[i+1]:5.2f}) | {bar} ({counts[i]})")


# ======================================================================
# 2. Test-retest reliability (ICC)
# ======================================================================

def compute_icc(
    alpha_records: list[dict],
    retest_records: list[dict],
) -> dict:
    """Compute ICC(2,1) between test and retest alpha measurements.

    Pairs records by (question_id, agent_id).
    """
    # Build lookup: (question_id, agent_id) -> alpha
    test_lookup: dict[tuple[str, str], float] = {}
    for record in alpha_records:
        qid = record["question_id"]
        for agent_id, est in record.get("alpha_estimates", {}).items():
            alpha_val = est.get("alpha")
            if alpha_val is not None:
                test_lookup[(qid, agent_id)] = alpha_val

    retest_lookup: dict[tuple[str, str], float] = {}
    for record in retest_records:
        qid = record["question_id"]
        for agent_id, est in record.get("alpha_estimates", {}).items():
            alpha_val = est.get("alpha")
            if alpha_val is not None:
                retest_lookup[(qid, agent_id)] = alpha_val

    # Find paired keys
    common_keys = sorted(set(test_lookup.keys()) & set(retest_lookup.keys()))
    if len(common_keys) < 5:
        return {
            "icc": None,
            "n_pairs": len(common_keys),
            "error": f"Too few paired measurements ({len(common_keys)}). Need >= 5.",
        }

    test_values = np.array([test_lookup[k] for k in common_keys])
    retest_values = np.array([retest_lookup[k] for k in common_keys])

    icc_val = icc_2_1(test_values, retest_values)

    # Bootstrap CI for ICC
    rng = np.random.RandomState(42)
    n_boot = 1000
    boot_iccs = []
    n = len(test_values)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_icc = icc_2_1(test_values[idx], retest_values[idx])
        boot_iccs.append(boot_icc)
    ci_lower = float(np.percentile(boot_iccs, 2.5))
    ci_upper = float(np.percentile(boot_iccs, 97.5))

    # Pearson correlation for comparison
    corr = float(np.corrcoef(test_values, retest_values)[0, 1])

    return {
        "icc": float(icc_val),
        "icc_ci_lower": float(ci_lower),
        "icc_ci_upper": float(ci_upper),
        "pearson_r": corr,
        "n_pairs": len(common_keys),
    }


# ======================================================================
# 3. Cronbach's alpha across probe types
# ======================================================================

def compute_cronbach(alpha_records: list[dict]) -> dict:
    """Compute Cronbach's alpha across probe types for internal consistency.

    Each probe type (8 types: 4 strengths x 2 social) is treated as an "item."
    Each agent-question is a "subject."
    The score is binary: 1 if the agent revised, 0 otherwise.
    """
    # Collect probe results per (question_id, agent_id): probe_type -> revised
    probe_types = [
        ("weak", False), ("weak", True),
        ("moderate", False), ("moderate", True),
        ("strong", False), ("strong", True),
        ("very_strong", False), ("very_strong", True),
    ]
    probe_labels = [
        f"{pt}{'_social' if soc else ''}" for pt, soc in probe_types
    ]

    # Build item matrix: rows = subjects, cols = probe types
    subject_data: dict[tuple[str, str], dict[str, int]] = {}

    for record in alpha_records:
        qid = record["question_id"]
        for agent_id, probes in record.get("probe_results", {}).items():
            key = (qid, agent_id)
            row = {}
            for probe in probes:
                pt = probe.get("probe_type", "")
                soc = probe.get("social", False)
                label = f"{pt}{'_social' if soc else ''}"
                if label in probe_labels:
                    row[label] = 1 if probe.get("revised", False) else 0
            if len(row) == len(probe_labels):
                subject_data[key] = row

    if len(subject_data) < 5:
        return {
            "cronbach_alpha": None,
            "n_subjects": len(subject_data),
            "error": f"Too few complete subjects ({len(subject_data)}). Need >= 5.",
        }

    # Build matrix
    matrix = np.array([
        [subject_data[key][label] for label in probe_labels]
        for key in sorted(subject_data.keys())
    ], dtype=float)

    ca = cronbach_alpha(matrix)

    return {
        "cronbach_alpha": float(ca),
        "n_subjects": len(subject_data),
        "n_items": len(probe_labels),
        "item_labels": probe_labels,
    }


# ======================================================================
# 4. Debate outcome analysis
# ======================================================================

def analyze_debates(debate_records: list[dict]) -> dict:
    """Compute debate outcome statistics."""
    if not debate_records:
        return {"error": "No debate records found"}

    n_total = len(debate_records)
    n_collapse = sum(1 for r in debate_records if r.get("outcome", {}).get("collapse", False))
    n_correction = sum(1 for r in debate_records if r.get("outcome", {}).get("correction", False))
    n_any_flip = sum(1 for r in debate_records if r.get("outcome", {}).get("any_flip", False))
    n_majority_correct = sum(1 for r in debate_records if r.get("majority_correct", False))

    # Accuracy deltas
    deltas = [r.get("outcome", {}).get("accuracy_delta", 0) for r in debate_records]
    init_correct = [r.get("outcome", {}).get("initial_n_correct", 0) for r in debate_records]
    final_correct = [r.get("outcome", {}).get("final_n_correct", 0) for r in debate_records]

    # Per-temperature-profile breakdown
    per_profile: dict[str, dict] = defaultdict(lambda: {"total": 0, "collapse": 0, "correction": 0})
    for r in debate_records:
        temps = tuple(r.get("temperatures", []))
        key = str(temps)
        per_profile[key]["total"] += 1
        if r.get("outcome", {}).get("collapse", False):
            per_profile[key]["collapse"] += 1
        if r.get("outcome", {}).get("correction", False):
            per_profile[key]["correction"] += 1

    return {
        "n_total": n_total,
        "n_collapse": n_collapse,
        "collapse_rate": n_collapse / n_total,
        "n_correction": n_correction,
        "correction_rate": n_correction / n_total,
        "n_any_flip": n_any_flip,
        "flip_rate": n_any_flip / n_total,
        "n_majority_correct": n_majority_correct,
        "majority_accuracy": n_majority_correct / n_total,
        "mean_accuracy_delta": float(np.mean(deltas)),
        "mean_initial_correct": float(np.mean(init_correct)),
        "mean_final_correct": float(np.mean(final_correct)),
        "per_profile": dict(per_profile),
    }


# ======================================================================
# 5. Predictive power: logistic regression of collapse on alpha
# ======================================================================

def build_prediction_dataset(
    debate_records: list[dict],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build feature matrix X and label vector y for prediction.

    Features per debate:
    - mean_alpha: mean alpha across agents
    - max_alpha: max alpha
    - min_alpha: min alpha
    - alpha_spread: max - min alpha
    - alpha_std: std of alphas

    Label: 1 if collapse, 0 otherwise.
    """
    X_rows = []
    y_list = []

    for record in debate_records:
        alpha_est = record.get("alpha_estimates", {})
        if not alpha_est:
            continue

        alphas = []
        for agent_id in sorted(alpha_est.keys()):
            a = alpha_est[agent_id].get("alpha")
            if a is not None:
                alphas.append(a)

        if len(alphas) < 2:
            continue

        arr = np.array(alphas)
        features = [
            float(np.mean(arr)),      # mean_alpha
            float(np.max(arr)),        # max_alpha
            float(np.min(arr)),        # min_alpha
            float(np.max(arr) - np.min(arr)),  # alpha_spread
            float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,  # alpha_std
        ]
        X_rows.append(features)

        collapse = record.get("outcome", {}).get("collapse", False)
        y_list.append(1 if collapse else 0)

    if len(X_rows) < 20:
        return None

    X = np.array(X_rows, dtype=float)
    y = np.array(y_list, dtype=float)

    return X, y


def run_prediction_analysis(debate_records: list[dict]) -> dict:
    """Run logistic regression to predict collapse from alpha features."""
    dataset = build_prediction_dataset(debate_records)
    if dataset is None:
        return {
            "error": "Insufficient data for prediction analysis (need >= 20 debates with alpha)",
            "auc": None,
        }

    X, y = dataset

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    if n_pos < 5 or n_neg < 5:
        return {
            "error": f"Class imbalance too extreme: {n_pos} collapse, {n_neg} no-collapse. "
                     f"Need >= 5 of each.",
            "auc": None,
            "n_collapse": n_pos,
            "n_no_collapse": n_neg,
        }

    # Fit logistic regression and compute AUC
    lr_result = logistic_regression(X, y)
    coefs = lr_result["coefficients"]
    intercept = lr_result["intercept"]
    y_scores = lr_result["predictions"]
    auc = lr_result["auc"]

    # Bootstrap CI for AUC
    rng = np.random.RandomState(42)
    n_boot = 1000
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), size=len(y))
        boot_y = y[idx]
        boot_scores = y_scores[idx]
        if boot_y.sum() > 0 and boot_y.sum() < len(boot_y):
            boot_aucs.append(compute_auc(boot_y.astype(int), boot_scores))
    if boot_aucs:
        auc_ci_lower = float(np.percentile(boot_aucs, 2.5))
        auc_ci_upper = float(np.percentile(boot_aucs, 97.5))
    else:
        auc_ci_lower = auc
        auc_ci_upper = auc

    feature_names = ["mean_alpha", "max_alpha", "min_alpha", "alpha_spread", "alpha_std"]
    feature_importance = {
        name: float(coef) for name, coef in zip(feature_names, coefs)
    }

    return {
        "auc": float(auc),
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper,
        "n_total": len(y),
        "n_collapse": n_pos,
        "n_no_collapse": n_neg,
        "collapse_rate": n_pos / len(y),
        "intercept": float(intercept),
        "feature_importance": feature_importance,
    }


# ======================================================================
# 6. GO/NO-GO decision
# ======================================================================

def make_decision(
    icc_result: dict,
    prediction_result: dict,
    cronbach_result: dict,
) -> dict:
    """Make GO/NO-GO decision based on thresholds.

    GREEN: ICC > 0.70 AND AUC > 0.70
    YELLOW: ICC > 0.50 AND AUC > 0.60
    RED: otherwise

    If ICC is unavailable (no retest data), decision is based on AUC alone
    with a note that retest is needed for full validation.
    """
    icc_val = icc_result.get("icc")
    auc_val = prediction_result.get("auc")
    ca_val = cronbach_result.get("cronbach_alpha")

    # Track which criteria are met
    criteria = {
        "icc_available": icc_val is not None,
        "icc_value": icc_val,
        "auc_available": auc_val is not None,
        "auc_value": auc_val,
        "cronbach_available": ca_val is not None,
        "cronbach_value": ca_val,
    }

    decision = "RED"
    reasons = []

    if icc_val is not None and auc_val is not None:
        if icc_val > 0.70 and auc_val > 0.70:
            decision = "GREEN"
            reasons.append(f"ICC={icc_val:.3f} > 0.70")
            reasons.append(f"AUC={auc_val:.3f} > 0.70")
        elif icc_val > 0.50 and auc_val > 0.60:
            decision = "YELLOW"
            reasons.append(f"ICC={icc_val:.3f} > 0.50 (but <= 0.70)")
            reasons.append(f"AUC={auc_val:.3f} > 0.60 (but <= 0.70)")
        else:
            reasons.append(f"ICC={icc_val:.3f} {'> 0.50' if icc_val > 0.50 else '<= 0.50'}")
            reasons.append(f"AUC={auc_val:.3f} {'> 0.60' if auc_val > 0.60 else '<= 0.60'}")
    elif auc_val is not None:
        # No ICC available -- judge on AUC alone with caveat
        reasons.append("ICC: NOT AVAILABLE (run with --retest to compute)")
        if auc_val > 0.70:
            decision = "YELLOW"  # can't be GREEN without ICC
            reasons.append(f"AUC={auc_val:.3f} > 0.70 (strong, but need ICC for GREEN)")
        elif auc_val > 0.60:
            decision = "YELLOW"
            reasons.append(f"AUC={auc_val:.3f} > 0.60")
        else:
            reasons.append(f"AUC={auc_val:.3f} <= 0.60")
    else:
        reasons.append("AUC: NOT AVAILABLE (need debate data)")
        if icc_val is not None:
            reasons.append(f"ICC={icc_val:.3f}")
        else:
            reasons.append("ICC: NOT AVAILABLE")

    # Add Cronbach info
    if ca_val is not None:
        ca_status = "good" if ca_val > 0.70 else "acceptable" if ca_val > 0.60 else "poor"
        reasons.append(f"Cronbach alpha={ca_val:.3f} ({ca_status})")

    return {
        "decision": decision,
        "reasons": reasons,
        "criteria": criteria,
    }


# ======================================================================
# Report printing
# ======================================================================

def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_report(
    alpha_stats: dict,
    all_alphas: list[float],
    icc_result: dict,
    cronbach_result: dict,
    debate_stats: dict,
    prediction_result: dict,
    decision_result: dict,
) -> None:
    """Print the comprehensive Block 0 analysis report."""

    print()
    print("#" * 60)
    print("#" + " BLOCK 0 ANALYSIS REPORT ".center(58) + "#")
    print("#" * 60)

    # --- 1. Alpha Distribution ---
    print_section("1. ALPHA DISTRIBUTION")

    if "error" in alpha_stats:
        print(f"  ERROR: {alpha_stats['error']}")
    else:
        print(f"  Total alpha measurements: {alpha_stats['n_values']}")
        print(f"  Mean:   {alpha_stats['mean']:.4f}")
        print(f"  Std:    {alpha_stats['std']:.4f}")
        print(f"  Min:    {alpha_stats['min']:.4f}")
        print(f"  Q25:    {alpha_stats['q25']:.4f}")
        print(f"  Median: {alpha_stats['median']:.4f}")
        print(f"  Q75:    {alpha_stats['q75']:.4f}")
        print(f"  Max:    {alpha_stats['max']:.4f}")

        print()
        print("  Per-agent breakdown:")
        for agent_id, agent_stats in alpha_stats.get("per_agent", {}).items():
            print(
                f"    {agent_id}: mean={agent_stats['mean']:.4f}, "
                f"std={agent_stats['std']:.4f}, "
                f"median={agent_stats['median']:.4f} (n={agent_stats['n']})"
            )

        print()
        print("  Histogram:")
        print_histogram(all_alphas)

    # --- 2. Reliability ---
    print_section("2. RELIABILITY")

    print("  Test-retest ICC(2,1):")
    if icc_result.get("icc") is not None:
        print(f"    ICC:     {icc_result['icc']:.4f}")
        print(f"    95% CI:  [{icc_result.get('icc_ci_lower', 'N/A'):.4f}, "
              f"{icc_result.get('icc_ci_upper', 'N/A'):.4f}]")
        print(f"    Pearson: {icc_result.get('pearson_r', 'N/A'):.4f}")
        print(f"    N pairs: {icc_result['n_pairs']}")
    elif "error" in icc_result:
        print(f"    {icc_result['error']}")
    else:
        print("    Not available (run Block 0.1 with --retest)")

    print()
    print("  Cronbach's alpha (internal consistency):")
    if cronbach_result.get("cronbach_alpha") is not None:
        print(f"    Alpha:    {cronbach_result['cronbach_alpha']:.4f}")
        print(f"    N items:  {cronbach_result['n_items']}")
        print(f"    N subjects: {cronbach_result['n_subjects']}")
    elif "error" in cronbach_result:
        print(f"    {cronbach_result['error']}")
    else:
        print("    Not available")

    # --- 3. Debate Statistics ---
    print_section("3. DEBATE OUTCOMES")

    if "error" in debate_stats:
        print(f"  ERROR: {debate_stats['error']}")
    else:
        print(f"  Total debates:     {debate_stats['n_total']}")
        print(f"  Collapse events:   {debate_stats['n_collapse']} "
              f"({debate_stats['collapse_rate']:.1%})")
        print(f"  Correction events: {debate_stats['n_correction']} "
              f"({debate_stats['correction_rate']:.1%})")
        print(f"  Any flip:          {debate_stats['n_any_flip']} "
              f"({debate_stats['flip_rate']:.1%})")
        print(f"  Majority correct:  {debate_stats['n_majority_correct']} "
              f"({debate_stats['majority_accuracy']:.1%})")
        print(f"  Mean acc delta:    {debate_stats['mean_accuracy_delta']:+.3f}")
        print(f"  Mean initial correct: {debate_stats['mean_initial_correct']:.2f}/3")
        print(f"  Mean final correct:   {debate_stats['mean_final_correct']:.2f}/3")

        per_profile = debate_stats.get("per_profile", {})
        if per_profile:
            print()
            print("  Per temperature-profile breakdown:")
            for profile, stats in sorted(per_profile.items()):
                collapse_rate = stats["collapse"] / stats["total"] if stats["total"] > 0 else 0
                correction_rate = stats["correction"] / stats["total"] if stats["total"] > 0 else 0
                print(
                    f"    {profile}: {stats['total']} debates, "
                    f"collapse={collapse_rate:.1%}, "
                    f"correction={correction_rate:.1%}"
                )

    # --- 4. Predictive Power ---
    print_section("4. PREDICTIVE POWER: P(collapse | alpha)")

    if "error" in prediction_result:
        print(f"  {prediction_result['error']}")
    elif prediction_result.get("auc") is not None:
        print(f"  AUC-ROC:     {prediction_result['auc']:.4f}")
        print(f"  95% CI:      [{prediction_result.get('auc_ci_lower', 'N/A'):.4f}, "
              f"{prediction_result.get('auc_ci_upper', 'N/A'):.4f}]")
        print(f"  N debates:   {prediction_result['n_total']}")
        print(f"  N collapse:  {prediction_result['n_collapse']} "
              f"({prediction_result['collapse_rate']:.1%})")
        print(f"  N no-collapse: {prediction_result['n_no_collapse']}")
        print()
        print("  Feature importance (logistic regression coefficients):")
        for name, coef in prediction_result.get("feature_importance", {}).items():
            direction = "+" if coef > 0 else "-" if coef < 0 else " "
            print(f"    {name:15s}: {direction}{abs(coef):.4f}")
        print(f"    {'intercept':15s}: {prediction_result.get('intercept', 0):.4f}")
    else:
        print("  Not available (need debate data with alpha estimates)")

    # --- 5. GO/NO-GO ---
    print_section("5. GO/NO-GO DECISION")

    decision = decision_result["decision"]
    color_map = {"GREEN": "\033[92m", "YELLOW": "\033[93m", "RED": "\033[91m"}
    reset = "\033[0m"
    color = color_map.get(decision, "")

    print(f"  Decision: {color}{decision}{reset}")
    print()
    print("  Rationale:")
    for reason in decision_result["reasons"]:
        print(f"    - {reason}")

    print()
    if decision == "GREEN":
        print("  Recommendation: PROCEED to Block 1.")
        print("  Alpha is reliable and predictive of debate collapse.")
    elif decision == "YELLOW":
        print("  Recommendation: PROCEED with caution.")
        print("  Consider collecting more data or running retest before Block 1.")
        if not decision_result["criteria"]["icc_available"]:
            print("  ACTION: Run Block 0.1 with --retest to obtain ICC.")
    else:
        print("  Recommendation: DO NOT PROCEED.")
        print("  Alpha measurement is unreliable or not predictive.")
        print("  Consider revising the probe design or alpha estimation method.")

    print()
    print("#" * 60)


# ======================================================================
# Main
# ======================================================================

def main(results_dir: str = "abc_exp/results") -> None:
    """Run the Block 0 GO/NO-GO analysis."""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"ERROR: Results directory not found: {results_path}")
        sys.exit(1)

    # Load data
    print("Loading data...")
    alpha_records = load_alpha_records(results_path)
    retest_records = load_alpha_retest_records(results_path)
    debate_records = load_debate_records(results_path)

    if not alpha_records:
        print("ERROR: No alpha records found. Run Block 0.1 first.")
        sys.exit(1)

    # 1. Alpha distribution
    alpha_stats = analyze_alpha_distribution(alpha_records)

    # Collect all alpha values for histogram
    all_alphas = []
    for record in alpha_records:
        for est in record.get("alpha_estimates", {}).values():
            a = est.get("alpha")
            if a is not None:
                all_alphas.append(a)

    # 2. ICC (test-retest reliability)
    if retest_records:
        icc_result = compute_icc(alpha_records, retest_records)
    else:
        icc_result = {"icc": None, "n_pairs": 0}

    # 3. Cronbach's alpha
    cronbach_result = compute_cronbach(alpha_records)

    # 4. Debate analysis
    debate_stats = analyze_debates(debate_records)

    # 5. Predictive power
    prediction_result = run_prediction_analysis(debate_records)

    # 6. GO/NO-GO decision
    decision_result = make_decision(icc_result, prediction_result, cronbach_result)

    # Print full report
    print_report(
        alpha_stats=alpha_stats,
        all_alphas=all_alphas,
        icc_result=icc_result,
        cronbach_result=cronbach_result,
        debate_stats=debate_stats,
        prediction_result=prediction_result,
        decision_result=decision_result,
    )

    # Save machine-readable summary
    summary = {
        "alpha_distribution": alpha_stats,
        "icc": icc_result,
        "cronbach": cronbach_result,
        "debate_stats": debate_stats,
        "prediction": prediction_result,
        "decision": decision_result,
    }
    summary_path = results_path / "block0_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nMachine-readable summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Block 0 Analysis: GO/NO-GO Decision",
    )
    parser.add_argument(
        "--results_dir", type=str, default="abc_exp/results",
        help="Directory containing Block 0 results (default: abc_exp/results)",
    )
    args = parser.parse_args()

    main(results_dir=args.results_dir)
