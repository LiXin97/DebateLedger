# Reproducibility Card

This repository contains the code and data for *Measuring Collapse and Correction in Homogeneous-Panel LLM Debate* (NeurIPS 2026, Evaluations and Datasets Track). The analyses below are zero-API: they read existing traces and write derived tables only.

## Zero-API Rebuild

Run from the root of a clone of https://github.com/LiXin97/DebateLedger. The traces live in the Hugging Face dataset; download them first, into the same root:

```bash
pip install -U huggingface_hub
hf download XINLI1997/DebateLedger --repo-type dataset --local-dir .
```

Then run:

The Croissant validation receipt requires the `mlcroissant` Python package. If it is not already installed, run `python -m pip install mlcroissant` before the validation step.

```bash
python abc_exp/experiments/reviewer_hardening_analyses.py
python abc_exp/experiments/n14_reviewer_hardening_artifacts.py
python abc_exp/experiments/round4_camera_ready_hardening.py
python abc_exp/experiments/leave_one_qwen_row_sensitivity.py
python abc_exp/experiments/round5_api_crossbench_hardening.py
python abc_exp/experiments/parser_cell_stability_audit.py   # needs the gated tier (re-reads model text)
python abc_exp/experiments/validate_croissant_metadata.py
python abc_exp/experiments/gemini_r5_stability_aggregate.py \
  --panels abc_exp/results/r5_gemini_alpha_panel_20260427_day1.jsonl \
           abc_exp/results/r5_gemini_alpha_panel_20260428_day2.jsonl \
           abc_exp/results/r5_gemini_alpha_panel_20260430_day3.jsonl \
  --output-stem r5_gemini_alpha_panel_multiday --overwrite
```

Expected derived outputs:

- `abc_exp/results/REVIEWER_HARDENING_ANALYSES.md`
- `abc_exp/results/reviewer_hardening_analyses.json`
- `abc_exp/results/N14_REVIEWER_HARDENING_ARTIFACTS.md`
- `abc_exp/results/n14_reviewer_hardening_artifacts.json`
- `abc_exp/results/ROUND4_CAMERA_READY_HARDENING.md`
- `abc_exp/results/round4_camera_ready_hardening.json`
- `abc_exp/results/leave_one_qwen_row_sensitivity.json`
- `abc_exp/results/ROUND5_API_CROSSBENCH_HARDENING.md`
- `abc_exp/results/round5_api_crossbench_hardening.json`
- `abc_exp/results/PARSER_CELL_STABILITY_AUDIT.md`
- `abc_exp/results/parser_cell_stability_audit.json`
- `abc_exp/results/DEBATE_EVAL_CROISSANT_VALIDATION.md`
- `abc_exp/results/debate_eval_croissant_validation_receipt.json`
- `abc_exp/results/debate_eval_croissant_rai_metadata.json`
- `croissant.json`
- `abc_exp/results/R5_GEMINI_ALPHA_PANEL_MULTIDAY.md`
- `abc_exp/results/r5_gemini_alpha_panel_multiday.json`

These scripts recompute the family-level exact permutation test, N14 errors-in-variables bootstrap, Wilson/Jeffreys conditional-collapse intervals, parser-failure counts, parse-missing sensitivity, parser cell-stability under a no-fallback extractor, provenance summaries, baseline/confound bakeoff, alpha-lite retrospective checks, trace-available initial-diversity diagnostics, Jeffreys-propagated held-out-family prediction intervals, Meta/Qwen leverage sensitivity, leave-one-Qwen-row sensitivity, API cross-benchmark summaries over checked-in paid-run JSONL files, Croissant/RAI validation, and zero-API R5 panel aggregation. They do not call external APIs and do not overwrite sealed-vintage artifacts. On the open tier they reproduce the checked-in outputs with three exceptions: the parser cell-stability audit and the final-answer-tag counts in `n14_reviewer_hardening_artifacts.py` re-read model text and need the gated tier; `n14_reviewer_hardening_artifacts.py` inventories every probe trace in `abc_exp/results/`, so traces added after the submission make a rerun list more files than the checked-in version reported in the paper; and `round4_camera_ready_hardening.py` records the size of `per_debate_r1_features.jsonl`, which was an empty placeholder at submission and is filled in this release.

## Main Trace Inputs

- `abc_exp/results/family_level_headline_aggregation.json`: realized N14 model/family headline rows.
- `abc_exp/results/reviewer_hardening_analyses.json`: consolidated N14 diagnostics used by the paper.
- `abc_exp/results/block0_debates.jsonl`, `block1_*.jsonl`, `block3_*.jsonl`: outcome-level debate rows from the early pilot blocks, without round-by-round traces. Round-by-round traces for the primary six-model debate cohort (6,925 debates, 253 collapses) are the six `debate_traces_{gemini_3-flash,openai_gpt-5.4-mini,vllm_llama-3.1-8b,vllm_phi-4-mini,vllm_qwen3-4b,vllm_qwen3-8b}.jsonl` files. That cohort was recorded with per-round answers only; no model text was saved.
- `abc_exp/results/debate_traces_*.jsonl`: round-by-round debate traces used for parser and provenance audits on the realized expansion rows and the response-period checks. In the open tier the `reasoning` fields are `null`.
- `abc_exp/results/sa_causal_*.jsonl`: probe traces used for alpha/parser audits. In the open tier the `initial_response_prefix` field is `null`.
- `abc_exp/results/conditional_collapse_table.json`: sealed-vintage debate diagnostics for older rows.
- `abc_exp/results/per_debate_r1_features.jsonl`: row-level R1 feature matrix for the reported Bayesian MLM. It was rebuilt on 2026-09-28 from the six primary-cohort trace files with `abc_exp/experiments/build_per_debate_features.py` and reproduces the MLM input exactly (3,145 initially correct debates from Llama-3.1-8B, Phi-4-mini, Qwen3-4B and Qwen3-8B, 251 collapses, 1,509 questions). The fitted MLM summary is checked in as `abc_exp/results/bayesian_multilevel_logistic.json`. This matrix is not required for the headline transition tables, parser audits, family-rank analyses, or signed-replay ledgers; `abc_exp/experiments/bayesian_multilevel_logistic.py` reads it directly. To rebuild it, run `build_per_debate_features.py` with `ABC_RESULTS` pointing to a folder that holds only the six primary-cohort files, because the script reads every `debate_traces_*.jsonl` in that folder.

## Protocol Pointers

- Probe and debate prompt card: the paper's appendix, "Probe Templates".
- Canonical prompt implementation: `abc_exp/config/prompts.py`. In the public code the very-strong probe and the social-pressure suffix are replaced by a placeholder; the gated tier restores the full file and the two probe runners below.
- Local-vLLM alpha runner: `abc_exp/scripts/run_alpha_probe_vllm.py`.
- OpenRouter alpha runner: `abc_exp/scripts/run_alpha_probe_openrouter.py`.
- Debate runner and majority policy: `abc_exp/src/debate/standard.py`.
- MCQ answer parser: `abc_exp/src/alpha/revision.py`.
- Artifact datasheet: `abc_exp/results/DEBATE_EVAL_ARTIFACT_DATASHEET.md`.
- Reuse-card schema, R1 derived-matrix schema, validated Croissant/RAI metadata block, validation receipt, and minimal walkthrough: `abc_exp/results/debate_eval_reuse_card.schema.json`, `abc_exp/results/per_debate_r1_features.schema.json`, top-level `croissant.json`, `abc_exp/results/debate_eval_croissant_rai_metadata.json`, `abc_exp/results/DEBATE_EVAL_CROISSANT_VALIDATION.md`, and `abc_exp/experiments/instrument_new_model_reuse_card.py`.

The top-level `croissant.json` points to the Hugging Face dataset. It hashes files from both the code repository and the open data tier, so regenerate it from a clone with the open tier downloaded:

```bash
DEBATELEDGER_DATASET_URL="https://huggingface.co/datasets/XINLI1997/DebateLedger" \
  python abc_exp/experiments/validate_croissant_metadata.py
```

Parser policy: prefer `Final Answer: X`, then answer/choose variants, then standalone final-line option letters, then the last valid option letter. No LLM judge step is used for final-answer extraction, equivalence judging, collapse/correction labels, or onset. Unparsed probe post-answers are treated as no revision in the primary analysis; debate majority vote filters `None` before alphabetic tie-breaking. Parser-failure counts and a `None`-as-flip sensitivity are in `N14_REVIEWER_HARDENING_ARTIFACTS.md`; a no-fallback parser cell-stability audit is in `PARSER_CELL_STABILITY_AUDIT.md`.

## Minimal Adoption Walkthrough

The short walkthrough below shows the artifact boundary for a new model-scaffold pair without calling any paid APIs. First run an alpha/debate audit with the local or API runner, summarize it into the fields listed in `debate_eval_reuse_card.schema.json`, then emit a populated card:

```bash
python abc_exp/experiments/instrument_new_model_reuse_card.py \
  --example \
  --output abc_exp/results/example_reuse_card.json

python abc_exp/experiments/instrument_new_model_reuse_card.py \
  --metrics-json path/to/new_model_metrics.json \
  --output abc_exp/results/new_model_reuse_card.json
```

The script validates that the card contains identity, question-pool, selection-time screen, debate outcome accounting, signed-utility, and release-boundary fields. It is a packaging step, not a substitute for the actual alpha/debate run.

## Closed-API Limits

The zero-API artifacts are exactly reproducible from checked-in traces. Closed-API runs are not bitwise reproducible because providers can route aliases to changing snapshots. Newer traces include timestamp/backend/model metadata where available; older sealed lanes have partial metadata and are marked as such in the provenance tables.

The headline family test, N14 sensitivity checks, R5 aggregation, and baseline/confound bakeoffs are rebuildable from the open tier; the parser audits also need the gated tier. The Bayesian R1 MLM reads the `per_debate_r1_features.jsonl` row matrix, which is included; the script treats an empty file as missing and fails with an explicit message rather than silently fitting on zero rows. This MLM is an anatomy check, not the primary family-level inference; its fitted summary, convergence diagnostics, and posterior predictive check are checked in. A transcript-free derived matrix for this diagnostic should include only derived fields such as anonymized row id, model family or release-safe model id, question id or hashed question id, initial/final majority correctness, collapse/correction labels, R1 majority-change and flip-count features, agreement fractions, split/fold identifiers where used, and release provenance.

The R5 three-day API nondeterminism panel is a closed-API drift audit for future rerun stability, not an input to the headline association. The repository includes a same-day Gemini Flash-Lite pilot and three fixed-question panels from 2026-04-27, 2026-04-28, and 2026-04-30. In the checked-in aggregation, all day pairs agree exactly on the 20 shared question-level mean-alpha values (Spearman 1.000, mean/max absolute delta 0.0000, zero initial-answer majority changes). This bounds same-snapshot repeatability for one Gemini lane only; it does not bound future provider-side snapshot changes.

A same-day Gemini Flash-Lite pilot can be reproduced with:

```bash
python abc_exp/experiments/gemini_r5_nondeterminism_panel.py \
  --n-questions 5 --repeats 2 --budget 1 \
  --max-concurrent 4 --job-concurrency 1 \
  --output-stem r5_gemini_alpha_panel_20260427_pilot --overwrite
```

This pilot is not the full R5 criterion; it only verifies that the fixed-prompt alpha harness records answer stability, parser failures, and cost accounting in a form that can be repeated across calendar days.

The day-level panel command can be reproduced with a Gemini API key; change the output stem for each calendar-day rerun:

```bash
python abc_exp/experiments/gemini_r5_nondeterminism_panel.py \
  --n-questions 20 --repeats 3 --budget 5 \
  --max-concurrent 8 --job-concurrency 2 \
  --output-stem r5_gemini_alpha_panel_20260427_day1 --overwrite
```

Observed per-day summary for day1/day2/day3: 60 records over 20 questions, zero initial-answer instabilities, mean/max within-question alpha range 0.0000/0.0000, probe post-answer `None` rate 4.38%, estimated cost $0.3520. The pilot and day-1 panels agree exactly on their five shared questions; the three day-level panels agree exactly on all 20 shared questions.

The post-hoc API cross-benchmark stress runs are paid, drift-limited inputs whose row-level outputs are checked in and summarized by the zero-API Round-5 script above. The full OpenRouter Mistral Small 4 GPQA run used `cross_benchmark_openrouter_replication.py` with an $8 run cap and observed $3.3500 cost over 198 debates; the smaller OpenRouter smokes observed $0.4856 total. The full Gemini 3.1 Flash-Lite GPQA run used a $15 run cap and observed $7.1227 cost over 198 debates; the smaller Gemini cross-benchmark runs observed $5.7895 total from checked-in row costs. These figures are accounting notes for the stored artifacts, not a request to rerun provider calls during the zero-API rebuild.

To reproduce or extend the R5 audit, rerun the same panel with fresh stems, then aggregate the day-level JSONL files:

```bash
python abc_exp/experiments/gemini_r5_nondeterminism_panel.py \
  --n-questions 20 --repeats 3 --budget 5 \
  --max-concurrent 8 --job-concurrency 2 \
  --output-stem r5_gemini_alpha_panel_20260428_day2 --overwrite

python abc_exp/experiments/gemini_r5_nondeterminism_panel.py \
  --n-questions 20 --repeats 3 --budget 5 \
  --max-concurrent 8 --job-concurrency 2 \
  --output-stem r5_gemini_alpha_panel_20260430_day3 --overwrite

python abc_exp/experiments/gemini_r5_stability_aggregate.py \
  --panels abc_exp/results/r5_gemini_alpha_panel_20260427_day1.jsonl \
           abc_exp/results/r5_gemini_alpha_panel_20260428_day2.jsonl \
           abc_exp/results/r5_gemini_alpha_panel_20260430_day3.jsonl \
  --output-stem r5_gemini_alpha_panel_multiday --overwrite
```

## Artifact Access

The release has three parts:

- Code repository (https://github.com/LiXin97/DebateLedger): evaluation code, parser, zero-API scripts, aggregate tables, figures, schemas, the datasheet and the Croissant/RAI metadata. Code is MIT-licensed; tables, figures and schemas are CC BY 4.0, subject to upstream benchmark terms.
- Open data tier (https://huggingface.co/datasets/XINLI1997/DebateLedger), CC BY 4.0: the aggregate files plus every probe and debate trace with the model-generated text fields set to `null`. Parsed answers, correctness labels, round structure, token counts and costs are kept. The non-social weak, moderate and strong probe templates are open.
- Gated data tier (https://huggingface.co/datasets/XINLI1997/DebateLedger-gated), research-use license with no redistribution and no fine-tuning of models that target real users without independent safety review: the full-text versions of the traces that contain model text, and the very-strong (convince-wrong) probe and social-pressure suffix. These can be reused as a misleading-answer corpus, so access is reviewed on request.

The first public release is versioned `alpha-tot-v1.0`. Closed-API snapshot reruns and corrections are released as new minor versions rather than in-place replacements.
