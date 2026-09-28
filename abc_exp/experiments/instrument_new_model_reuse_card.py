#!/usr/bin/env python3
"""Populate a debate-evaluation reuse card for a new model-scaffold pair.

This is the minimal adoption walkthrough referenced by the paper. It does not
run model APIs. It takes metrics produced by an alpha/debate audit and writes a
single JSON card matching the fields documented in
``abc_exp/results/debate_eval_reuse_card.schema.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = [
    "identity",
    "question_pool",
    "selection_time_fingerprint",
    "debate_outcome_accounting",
    "signed_utility",
    "release",
]


def example_card() -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "artifact_type": "debate_evaluation_reuse_card_instance",
        "identity": {
            "model_name": "example-model",
            "model_family": "example-family",
            "backend": "local-or-api-backend",
            "model_version_or_snapshot": "snapshot-id-or-date",
            "serving_parameters": {"temperature": 0.0},
        },
        "question_pool": {
            "benchmark": "MMLU-Pro",
            "question_pool_id": "target-task-high-fr-n200-v1",
            "pool_selection_rule": "outcome-blind high flip-rate screen",
            "n_questions": 200,
            "outcome_blind_selection": True,
        },
        "selection_time_fingerprint": {
            "alpha_tot": 0.0,
            "alpha_adv": 0.0,
            "alpha_cor": 0.0,
            "social_argument_ratio": 0.0,
            "n_probe_rows": 0,
            "n_probe_completions": 0,
            "parser_none_rate": 0.0,
        },
        "debate_outcome_accounting": {
            "n_debates": 0,
            "initial_majority_accuracy": 0.0,
            "final_majority_accuracy": 0.0,
            "conditional_collapse_rate": 0.0,
            "conditional_correction_rate": 0.0,
            "collapse_onset_distribution": {"r1": 0, "r2": 0, "r3": 0},
        },
        "signed_utility": {
            "w_coll": 1.0,
            "w_corr": 1.0,
            "gate_name": "none-or-policy-name",
            "validation_cohort_id": "heldout-cohort-id",
            "collapses_prevented": 0,
            "corrections_lost": 0,
            "net_utility": 0.0,
            "accuracy_delta": 0.0,
        },
        "release": {
            "aggregate_fields_open": True,
            "probe_templates_open_or_gated": "open-low-risk-plus-gated-convince-wrong",
            "full_transcripts_open_or_gated": "gated",
            "r1_feature_matrix_open_or_gated": "gated-or-omitted",
            "known_missing_artifacts": [],
        },
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def validate_card(card: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in card]
    if missing:
        raise ValueError(f"missing required top-level fields: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, help="JSON object with populated reuse-card fields")
    parser.add_argument("--schema", type=Path, default=Path("abc_exp/results/debate_eval_reuse_card.schema.json"))
    parser.add_argument("--output", type=Path, default=Path("abc_exp/results/example_reuse_card.json"))
    parser.add_argument("--example", action="store_true", help="write an example card instead of reading metrics")
    args = parser.parse_args()

    if args.example:
        card = example_card()
    elif args.metrics_json:
        card = load_json(args.metrics_json)
    else:
        raise SystemExit("pass --example or --metrics-json")

    validate_card(card)
    schema_summary = load_json(args.schema)
    card.setdefault("schema_version", schema_summary.get("schema_version", "0.1"))
    card.setdefault("artifact_type", "debate_evaluation_reuse_card_instance")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(card, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
