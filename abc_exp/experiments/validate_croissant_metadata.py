"""Generate and validate Croissant/RAI metadata for the review artifact.

The previous metadata file was a project-specific JSON block. This script emits
a Croissant JSON-LD record through the `mlcroissant` package and immediately
loads it back with `mlcroissant.Dataset`, which exercises the package validator.
It is zero-API and only hashes local documentation/result artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
RES = ARTIFACT_ROOT / "results"
OUT_JSON = RES / "debate_eval_croissant_rai_metadata.json"
OUT_TOPLEVEL_JSON = ROOT / "croissant.json"
OUT_ARTIFACT_COPY_JSON = ARTIFACT_ROOT / "croissant.json"
OUT_RECEIPT = RES / "debate_eval_croissant_validation_receipt.json"
OUT_MD = RES / "DEBATE_EVAL_CROISSANT_VALIDATION.md"
DEFAULT_REVIEW_URL = "https://huggingface.co/datasets/XINLI1997/DebateLedger"
PLACEHOLDER_TOKEN = "REPLACE" + "_ME"

FILES = [
    "README_REPRO.md",
    "requirements-artifact.txt",
    "abc_exp/results/DEBATE_EVAL_ARTIFACT_DATASHEET.md",
    "abc_exp/results/debate_eval_reuse_card.schema.json",
    "abc_exp/results/cohort_n14_headline.json",
    "abc_exp/results/family_level_headline_aggregation.json",
    "abc_exp/results/conditional_collapse_table.json",
    "abc_exp/results/reviewer_hardening_analyses.json",
    "abc_exp/results/N14_REVIEWER_HARDENING_ARTIFACTS.md",
    "abc_exp/results/n14_reviewer_hardening_artifacts.json",
    "abc_exp/results/ROUND4_CAMERA_READY_HARDENING.md",
    "abc_exp/results/ROUND5_API_CROSSBENCH_HARDENING.md",
    "abc_exp/results/round5_api_crossbench_hardening.json",
    "abc_exp/results/PARSER_CELL_STABILITY_AUDIT.md",
    "abc_exp/results/parser_cell_stability_audit.json",
    "abc_exp/results/correction_preserving_policy_replay.json",
    "abc_exp/results/CORRECTION_PRESERVING_POLICY_REPLAY.md",
    "abc_exp/results/block0_alpha.jsonl",
    "abc_exp/results/block0_debates.jsonl",
    "abc_exp/results/block1_abc_vs_baselines.jsonl",
    "abc_exp/results/block3_multi_benchmark.jsonl",
]

FILE_ROLES = {
    "README_REPRO.md": "zero-api rebuild card",
    "requirements-artifact.txt": "Python dependency manifest",
    "abc_exp/results/DEBATE_EVAL_ARTIFACT_DATASHEET.md": "artifact datasheet",
    "abc_exp/results/debate_eval_reuse_card.schema.json": "new-row reuse schema",
    "abc_exp/results/cohort_n14_headline.json": "N14 headline rows",
    "abc_exp/results/family_level_headline_aggregation.json": "family-level primary aggregation",
    "abc_exp/results/conditional_collapse_table.json": "collapse/correction transition table",
    "abc_exp/results/reviewer_hardening_analyses.json": "review audit diagnostics",
    "abc_exp/results/N14_REVIEWER_HARDENING_ARTIFACTS.md": "N14 diagnostic summary",
    "abc_exp/results/n14_reviewer_hardening_artifacts.json": "N14 diagnostic data",
    "abc_exp/results/ROUND4_CAMERA_READY_HARDENING.md": "revision audit summary",
    "abc_exp/results/ROUND5_API_CROSSBENCH_HARDENING.md": "cross-benchmark stress summary",
    "abc_exp/results/round5_api_crossbench_hardening.json": "cross-benchmark stress data",
    "abc_exp/results/PARSER_CELL_STABILITY_AUDIT.md": "parser stability summary",
    "abc_exp/results/parser_cell_stability_audit.json": "parser stability data",
    "abc_exp/results/correction_preserving_policy_replay.json": "signed replay ledger",
    "abc_exp/results/CORRECTION_PRESERVING_POLICY_REPLAY.md": "signed replay summary",
    "abc_exp/results/block0_alpha.jsonl": "probe trace sample",
    "abc_exp/results/block0_debates.jsonl": "debate trace sample",
    "abc_exp/results/block1_abc_vs_baselines.jsonl": "baseline comparison trace rows",
    "abc_exp/results/block3_multi_benchmark.jsonl": "boundary benchmark trace rows",
}


def resolve_artifact_path(rel_path: str) -> Path:
    artifact_path = ARTIFACT_ROOT / rel_path
    if artifact_path.exists():
        return artifact_path
    return ROOT / rel_path


def sha256(rel_path: str) -> str:
    path = resolve_artifact_path(rel_path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_id(rel_path: str) -> str:
    return "file_" + "".join(ch if ch.isalnum() else "_" for ch in rel_path).strip("_")


def file_object(mlc: Any, rel_path: str) -> Any:
    if rel_path.endswith(".jsonl"):
        media_type = "application/jsonlines"
    elif rel_path.endswith(".json"):
        media_type = "application/json"
    elif rel_path.endswith(".txt"):
        media_type = "text/plain"
    else:
        media_type = "text/markdown"
    return mlc.FileObject(
        id=file_id(rel_path),
        name=Path(rel_path).name,
        content_url=rel_path,
        encoding_formats=[media_type],
        sha256=sha256(rel_path),
    )


def review_url() -> str:
    return os.environ.get("DEBATELEDGER_DATASET_URL", DEFAULT_REVIEW_URL)


def build_file_inventory_record_set(mlc: Any) -> Any:
    return mlc.RecordSet(
        id="artifact_file_inventory",
        name="artifact_file_inventory",
        description=(
            "Core files in the public DebateLedger release that define the reusable "
            "collapse/correction evaluation, zero-API rebuild, and signed-replay ledgers."
        ),
        key=["artifact_path"],
        fields=[
            mlc.Field(
                id="artifact_path",
                name="artifact_path",
                data_types=[mlc.DataType.TEXT],
                description="Repository-relative path for a core artifact file.",
            ),
            mlc.Field(
                id="artifact_role",
                name="artifact_role",
                data_types=[mlc.DataType.TEXT],
                description="Role of the file in reproducing or reusing the evaluation protocol.",
            ),
            mlc.Field(
                id="encoding_format",
                name="encoding_format",
                data_types=[mlc.DataType.TEXT],
                description="Media type recorded in the Croissant distribution entry.",
            ),
            mlc.Field(
                id="sha256",
                name="sha256",
                data_types=[mlc.DataType.TEXT],
                description="SHA-256 digest of the checked-in file.",
            ),
            mlc.Field(
                id="release_tier",
                name="release_tier",
                data_types=[mlc.DataType.TEXT],
                description="Release tier of the file in the public release.",
            ),
        ],
        data=[
            {
                "artifact_path": rel_path,
                "artifact_role": FILE_ROLES[rel_path],
                "encoding_format": (
                    "application/jsonlines"
                    if rel_path.endswith(".jsonl")
                    else "application/json"
                    if rel_path.endswith(".json")
                    else "text/plain"
                    if rel_path.endswith(".txt")
                    else "text/markdown"
                ),
                "sha256": sha256(rel_path),
                "release_tier": "open; full transcripts and convince-wrong templates are in the gated dataset",
            }
            for rel_path in FILES
        ],
    )


# Trace files described record by record (added 2026-10-30, release 1.0.1). Each FileSet
# groups files that share a schema; each RecordSet lists only the keys present in every
# record of that group, so the records load with mlcroissant across all files.
TRACE_SETS = [
    {
        "id": "debate_traces",
        "includes": "abc_exp/results/debate_traces_*.jsonl",
        "description": "Debate traces, one record per three-agent debate from Round 0 to the final round. "
        "Model output text is null in the open tier; full text is in the gated dataset.",
        "fields": [
            ("question_id", "TEXT", False, "Benchmark question identifier."),
            ("correct_label", "TEXT", False, "Answer-key option letter."),
            ("n_rounds", "INTEGER", False, "Number of rounds, Round 0 included."),
            ("initial_answers", "TEXT", True, "Parsed Round-0 answers of the three agents."),
            ("final_answers", "TEXT", True, "Parsed final-round answers of the three agents."),
        ],
    },
    {
        "id": "probe_traces",
        "includes": "abc_exp/results/sa_causal_*.jsonl",
        "description": "8-probe screen traces, one record per agent and question: the initial answer and "
        "the revision outcome under each probe condition.",
        "fields": [
            ("question_id", "TEXT", False, "Benchmark question identifier."),
            ("agent_idx", "INTEGER", False, "Agent index within the panel."),
            ("condition", "TEXT", False, "System-prompt condition of the probe run."),
            ("initial_answer", "TEXT", False, "Parsed answer before any probe."),
            ("initial_correct", "BOOL", False, "Whether the initial answer matches the key."),
            ("flip_rate", "FLOAT", False, "Fraction of the eight probes that changed the answer."),
        ],
    },
    {
        "id": "cross_benchmark_traces",
        "includes": "abc_exp/results/cross_benchmark_*.jsonl",
        "description": "Cross-benchmark probe and debate rows (ARC, GPQA, TruthfulQA and others).",
        "fields": [
            ("question_id", "TEXT", False, "Benchmark question identifier."),
            ("benchmark", "TEXT", False, "Benchmark name."),
            ("model", "TEXT", False, "Model identifier."),
            ("correct_label", "TEXT", False, "Answer-key option letter."),
        ],
    },
    {
        "id": "probe_panel_traces",
        "includes": "abc_exp/results/r5_*.jsonl",
        "description": "Repeated probe-panel rows used for the multi-day aggregation check.",
        "fields": [
            ("question_id", "TEXT", False, "Benchmark question identifier."),
            ("model", "TEXT", False, "Model identifier."),
            ("initial_answer", "TEXT", False, "Parsed answer before any probe."),
            ("n_probes", "INTEGER", False, "Number of probes administered."),
            ("n_revised", "INTEGER", False, "Number of probes that changed the answer."),
        ],
    },
    {
        "id": "round1_feature_matrix",
        "includes": "abc_exp/results/per_debate_r1_features.jsonl",
        "description": "Transcript-free per-debate matrix: transition labels and per-round agreement "
        "features for the 6,925 primary MMLU-Pro debates.",
        "fields": [
            ("model", "TEXT", False, "Model identifier."),
            ("question_id", "TEXT", False, "Benchmark question identifier."),
            ("collapsed", "BOOL", False, "Correct initial majority, wrong final majority."),
            ("corrected", "BOOL", False, "Wrong initial majority, correct final majority."),
            ("init_majority_correct", "BOOL", False, "Whether the Round-0 majority is correct."),
            ("r1_majority_changed", "BOOL", False, "Whether the majority changed in Round 1."),
            ("r1_agree_frac", "FLOAT", False, "Fraction of agents agreeing after Round 1."),
        ],
    },
]
TRACE_SETS += [
    dict(TRACE_SETS[0], id="thinking_off_debate_traces", includes="abc_exp/results/thinking_off_rerun/debate_traces_*.jsonl",
         description="Debate traces of the thinking-off rerun of the four Qwen3.5/3.6 rows. Model output text is null in the open tier."),
    dict(TRACE_SETS[1], id="thinking_off_probe_traces", includes="abc_exp/results/thinking_off_rerun/sa_causal_*.jsonl",
         description="8-probe screen traces of the thinking-off rerun of the four Qwen3.5/3.6 rows. Model output text is null in the open tier."),
]


def build_trace_sets(mlc: Any) -> tuple[list[Any], list[Any]]:
    file_sets, record_sets = [], []
    for spec in TRACE_SETS:
        fs_id = f"fileset_{spec['id']}"
        file_sets.append(
            mlc.FileSet(
                id=fs_id,
                name=fs_id,
                description=spec["description"],
                encoding_formats=["application/jsonlines"],
                includes=[spec["includes"]],
            )
        )
        fields = []
        for name, dtype, repeated, desc in spec["fields"]:
            fields.append(
                mlc.Field(
                    id=f"{spec['id']}/{name}",
                    name=name,
                    description=desc,
                    data_types=[getattr(mlc.DataType, dtype)],
                    repeated=repeated,
                    source=mlc.Source(file_set=fs_id, extract=mlc.Extract(column=name)),
                )
            )
        record_sets.append(
            mlc.RecordSet(id=spec["id"], name=spec["id"], description=spec["description"], fields=fields)
        )
    return file_sets, record_sets


def build_metadata(mlc: Any) -> dict[str, Any]:
    trace_file_sets, trace_record_sets = build_trace_sets(mlc)
    metadata = mlc.Metadata(
        name="Debate Collapse/Correction Evaluation Artifact",
        description=(
            "Validated Croissant/RAI metadata for the collapse/correction "
            "evaluation protocol bundle used in a homogeneous-panel LLM debate audit."
        ),
        version="1.0.3",
        cite_as="Xin Li, Mengbing Liu, and Chau Yuen. Measuring Collapse and Correction in Homogeneous-Panel LLM Debate. NeurIPS 2026 Evaluations and Datasets Track. DebateLedger artifact version 1.0.3.",
        date_created=datetime(2026, 5, 2),
        date_published=datetime(2026, 9, 28),
        url=review_url(),
        license=["https://creativecommons.org/licenses/by/4.0/"],
        keywords=[
            "multi-agent debate",
            "evaluation",
            "collapse",
            "correction",
            "MMLU-Pro",
            "responsible AI",
        ],
        in_language=["en"],
        data_collection=(
            "Public benchmark MCQ questions were processed by LLM agents under fixed probe "
            "and debate prompts. No human-subjects or personal data were collected."
        ),
        extra_properties={
            "rai:hasSyntheticData": True,
        },
        data_collection_type=["machine-generated traces from public benchmark questions"],
        data_collection_missing_data=(
            "Closed-API snapshot metadata are partial for older sealed lanes; full transcripts "
            "and convince-wrong templates are gated for dual-use reasons."
        ),
        data_collection_raw_data=(
            "Upstream public MCQ benchmark questions are not owned by this artifact; derived "
            "aggregate tables and trace summaries are released subject to upstream benchmark terms."
        ),
        data_preprocessing_protocol=[
            "Rule-based MCQ answer extraction; majority voting after filtering unparsed answers; "
            "alphabetic tie-breaking; deterministic collapse/correction/onset labels."
        ],
        data_annotation_protocol=[
            "No human labeling. Collapse/correction labels are deterministic functions of parsed "
            "MCQ answers and answer keys."
        ],
        machine_annotation_tools=[
            "Rule-based MCQ answer parser included in the artifact; no LLM judge step is used."
        ],
        data_biases=[
            "Primary evidence is MMLU-Pro; homogeneous same-model three-agent scaffold; "
            "realized G=7 family cohort with Qwen-heavy model rows."
        ],
        data_use_cases=[
            "Zero-API recomputation of aggregate tables; applying the transition-table and "
            "signed-utility protocol to new model-scaffold pairs; documenting gated release boundaries; "
            "public artifact hosting on Hugging Face with code on GitHub."
        ],
        data_limitations=[
            "Not a benchmark-general law; not a calibrated per-question oracle; not a deployable "
            "controller; gated tier needed for convince-wrong templates and full transcripts."
        ],
        data_social_impact=(
            "Helps audit debate interventions that may reduce collapses while discarding corrections; "
            "gated release mitigates misuse as a misleading-answer corpus."
        ),
        personal_sensitive_information=["none"],
        data_release_maintenance_plan=(
            "Versioned public release (alpha-tot-v1.0.3): open data on Hugging Face, code on GitHub, "
            "and a gated Hugging Face dataset for full transcripts and convince-wrong templates. "
            "Corrections and provider or parser changes are released as new minor versions."
        ),
        distribution=[file_object(mlc, rel_path) for rel_path in FILES] + trace_file_sets,
        record_sets=[build_file_inventory_record_set(mlc)] + trace_record_sets,
    )
    return metadata.to_json()


def write_markdown(receipt: dict[str, Any]) -> None:
    lines = [
        "# Croissant/RAI Validation Receipt",
        "",
        f"- Status: **{receipt['status']}**.",
        f"- Validator: `{receipt['validator']}`.",
        f"- Metadata file: `{receipt['metadata_file']}`.",
        f"- Top-level Croissant file: `{receipt['top_level_metadata_file']}`.",
        f"- Dataset URL in metadata: `{receipt['dataset_url']}`.",
        f"- Files hashed: **{len(receipt['files'])}**.",
        "",
        "## RAI Fields Populated",
        "",
    ]
    if receipt.get("dataset_url_is_placeholder"):
        lines[9:9] = [
            "The dataset URL is a placeholder; rerun the validator with `DEBATELEDGER_DATASET_URL` set to the public dataset URL.",
            "",
        ]
    else:
        lines[9:9] = [
            "The metadata URL is set to the public Hugging Face dataset. If the URL changes, rerun the validator with `DEBATELEDGER_DATASET_URL` set to the new URL.",
            "",
        ]
    for field in receipt["rai_fields_populated"]:
        lines.append(f"- `{field}`")
    lines.extend(["", "## Hashed Distribution Files", "", "| File | SHA256 prefix |", "|---|---:|"])
    for row in receipt["files"]:
        lines.append(f"| `{row['path']}` | `{row['sha256'][:12]}` |")
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    try:
        import mlcroissant as mlc
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "mlcroissant is required for validation. Install with `python -m pip install mlcroissant`."
        ) from exc

    metadata = build_metadata(mlc)
    OUT_JSON.write_text(json.dumps(metadata, indent=2) + "\n")
    OUT_TOPLEVEL_JSON.write_text(json.dumps(metadata, indent=2) + "\n")
    OUT_ARTIFACT_COPY_JSON.write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        mlc.Dataset(str(OUT_JSON))
        mlc.Dataset(str(OUT_TOPLEVEL_JSON))
        mlc.Dataset(str(OUT_ARTIFACT_COPY_JSON))
        status = "PASS"
        error = None
    except Exception as exc:  # pragma: no cover - receipt records validator failure.
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
    receipt = {
        "status": status,
        "validator": "mlcroissant.Dataset load validation",
        "metadata_file": str(OUT_JSON.relative_to(ROOT)),
        "top_level_metadata_file": str(OUT_TOPLEVEL_JSON.relative_to(ROOT)),
        "dataset_url": review_url(),
        "dataset_url_is_placeholder": PLACEHOLDER_TOKEN in review_url(),
        "error": error,
        "files": [{"path": rel_path, "sha256": sha256(rel_path)} for rel_path in FILES],
        "rai_fields_populated": [
            "rai:dataCollection",
            "rai:hasSyntheticData",
            "rai:dataCollectionType",
            "rai:dataCollectionMissingData",
            "rai:dataCollectionRawData",
            "rai:dataPreprocessingProtocol",
            "rai:dataAnnotationProtocol",
            "rai:machineAnnotationTools",
            "rai:dataBiases",
            "rai:dataUseCases",
            "rai:dataLimitations",
            "rai:dataSocialImpact",
            "rai:personalSensitiveInformation",
            "rai:dataReleaseMaintenancePlan",
        ],
    }
    OUT_RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    write_markdown(receipt)
    if status != "PASS":
        raise SystemExit(error)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TOPLEVEL_JSON}")
    print(f"Wrote {OUT_ARTIFACT_COPY_JSON}")
    print(f"Wrote {OUT_RECEIPT}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
