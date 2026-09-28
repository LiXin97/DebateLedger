# DebateLedger

**Measuring Collapse and Correction in Homogeneous-Panel LLM Debate** · NeurIPS 2026, Evaluations and Datasets Track

Xin Li\*, Mengbing Liu\*, Chau Yuen · Nanyang Technological University · \*Equal contribution

[Project page](https://lixin.ai/DebateLedger/) · [OpenReview](https://openreview.net/forum?id=E8FfL8c7XE) · [Dataset (Hugging Face)](https://huggingface.co/datasets/XINLI1997/DebateLedger) · [Gated dataset (Hugging Face)](https://huggingface.co/datasets/XINLI1997/DebateLedger-gated)

DebateLedger evaluates multi-agent LLM debate by the transitions it causes rather
than by final accuracy alone. Three copies of one model answer a multiple-choice
question and debate for three rounds. Each run is recorded as a transition
ledger that crosses initial-majority with final-majority correctness
(preserved, collapse, correction, unrepaired), together with the onset round at
which a correct majority first breaks. An intervention is scored on the same
saved debates by the collapses it prevents and the corrections it loses. On
6,925 MMLU-Pro debates the ledger records 253 collapses. A leave-one-model-out
probe-gated freeze prevents 29 of 251 collapses but loses 108 of 804
corrections on 6,525 held-out debates, a net of −1.21 accuracy points under
equal weights. A pre-debate 8-probe screen ranks model families for trace
logging (G = 7, Spearman ρ = 0.893); initial-majority accuracy is a close
comparator (ρ = 0.821), so the screen is a triage signal, not a calibrated
predictor.

## Data release (alpha-tot-v1.0)

The traces are distributed on Hugging Face in two tiers. Download the open tier
into the root of this repository; the files land in `abc_exp/results/`, where
the scripts expect them.

```bash
pip install -U huggingface_hub
hf download XINLI1997/DebateLedger --repo-type dataset --local-dir .
```

### Open tier ([XINLI1997/DebateLedger](https://huggingface.co/datasets/XINLI1997/DebateLedger), CC BY 4.0)

Every trace with the model-generated text fields set to `null`. Parsed answers,
correctness labels, round structure, token counts and costs are kept.

| Files (`abc_exp/results/`) | Content | Records |
|---|---|---:|
| `debate_traces_{gemini_3-flash, openai_gpt-5.4-mini, vllm_llama-3.1-8b, vllm_phi-4-mini, vllm_qwen3-4b, vllm_qwen3-8b}.jsonl` | Primary MMLU-Pro debate cohort (six models), per-round answers | 6,925 debates |
| `per_debate_r1_features.jsonl` | Round-1 feature matrix derived from the primary cohort | 6,925 debates |
| other `debate_traces_*.jsonl` (50 files) | Extension and response-period debates: further models, reasoning-mode and private-revision checks, mixed-model panels, GSM8K | 7,636 debates |
| `sa_causal_*.jsonl` (44 files) | 8-probe screen: one record per model, question and agent with the initial answer and the eight probe replies | 35,664 records |
| `cross_benchmark_*.jsonl` (16 files) | GPQA, TruthfulQA and ARC-Challenge stress checks | 1,033 debates |
| `block0_*`, `block1_*`, `block3_*` | Early pilot blocks | 1,705 records |
| `r5_gemini_alpha_panel_*.jsonl` | Three-day closed-API repeatability panel | 190 records |

Aggregate tables and audits (`*.json`, `*.md`, `*.csv`) are also in this
repository. `croissant.json` carries Croissant and Responsible AI metadata with
SHA-256 digests of the core files; `abc_exp/results/DEBATE_EVAL_ARTIFACT_DATASHEET.md`
is the datasheet.

### Gated tier ([XINLI1997/DebateLedger-gated](https://huggingface.co/datasets/XINLI1997/DebateLedger-gated), research-use license)

The full-text versions of the 57 trace files that contain model text (agent
responses in debate rounds and probe response prefixes), and the very-strong
(convince-wrong) probe template and social-pressure suffix. These can be reused
as a misleading-answer corpus, so access is reviewed on request. In this
repository the two templates are replaced by a placeholder in
`abc_exp/config/prompts.py`, `abc_exp/scripts/run_alpha_probe_openrouter.py` and
`abc_exp/scripts/run_alpha_probe_vllm.py`; downloading the gated tier into the
repository root restores the full files.

No model weights are released.

### Reproducibility boundary

- The paper's tables are rebuilt from the open tier without API calls (commands
  below). A rerun differs from the checked-in outputs in three documented
  places: the final-answer-tag counts in `n14_reviewer_hardening_artifacts.py`
  and `parser_cell_stability_audit.py` re-read model text and need the gated
  tier; `n14_reviewer_hardening_artifacts.py` inventories every probe trace, and
  traces added after the submission extend that inventory; and
  `round4_camera_ready_hardening.py` records the size of
  `per_debate_r1_features.jsonl`, which was an empty placeholder at submission.
- The primary debate cohort was recorded with per-round answers only. No model
  text exists for it in either tier.
- Closed-API rows cannot be regenerated bit for bit, because providers can
  route model aliases to changing snapshots. The traces keep the provider,
  backend and timestamp metadata that was available at collection time.

## Reproducing the paper's tables

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-artifact.txt
hf download XINLI1997/DebateLedger --repo-type dataset --local-dir .

python abc_exp/experiments/reviewer_hardening_analyses.py
python abc_exp/experiments/n14_reviewer_hardening_artifacts.py
python abc_exp/experiments/round4_camera_ready_hardening.py
python abc_exp/experiments/leave_one_qwen_row_sensitivity.py
python abc_exp/experiments/round5_api_crossbench_hardening.py
python abc_exp/experiments/gemini_r5_stability_aggregate.py \
  --panels abc_exp/results/r5_gemini_alpha_panel_20260427_day1.jsonl \
           abc_exp/results/r5_gemini_alpha_panel_20260428_day2.jsonl \
           abc_exp/results/r5_gemini_alpha_panel_20260430_day3.jsonl \
  --output-stem r5_gemini_alpha_panel_multiday --overwrite
```

`README_REPRO.md` is the detailed reproducibility card: rebuild commands, trace
inputs, protocol pointers, and the closed-API panels that can be rerun with
provider keys.

## Code

| Path | Purpose |
|---|---|
| `abc_exp/src/debate/standard.py` | Homogeneous three-agent debate runner and majority rule |
| `abc_exp/src/alpha/revision.py` | Rule-based multiple-choice answer parser (no LLM judge) |
| `abc_exp/config/prompts.py` | Debate prompts and the 8-probe templates |
| `abc_exp/scripts/run_alpha_probe_{vllm,openrouter,gemini}.py` | 8-probe screen runners for local vLLM, OpenRouter and Gemini |
| `abc_exp/experiments/build_per_debate_features.py` | Round-1 feature matrix from round-by-round debate traces |
| `abc_exp/experiments/cascade_r1_auc.py`, `abc_exp/experiments/bayesian_multilevel_logistic.py` | Collapse onset and Round-1 trajectory analyses |
| `abc_exp/experiments/instrument_new_model_reuse_card.py` | Reuse card for adding a new model–scaffold row under the same denominators |
| `abc_exp/experiments/validate_croissant_metadata.py` | Croissant and RAI metadata generation and validation |

## Citation

```bibtex
@inproceedings{
li2026debateledger,
title={Measuring Collapse and Correction in Homogeneous-Panel {LLM} Debate},
author={Xin Li and Mengbing Liu and Chau Yuen},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
year={2026}
}
```

## License

Code is MIT (`LICENSE`). Result tables, figures, schemas and the open data tier
are CC BY 4.0. The gated tier is released under the DebateLedger Gated Tier
Research Use License. Benchmark questions remain subject to the licenses of
MMLU-Pro, GPQA, GSM8K, ARC-Challenge and TruthfulQA.
