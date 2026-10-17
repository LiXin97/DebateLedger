# DebateLedger data release (alpha-tot-v1.0)

The traces are distributed on Hugging Face in two tiers. Download the open tier
into the root of this repository; the files land in `abc_exp/results/`, where
the scripts expect them.

```bash
pip install -U huggingface_hub
hf download XINLI1997/DebateLedger --repo-type dataset --local-dir .
```

## Open tier ([XINLI1997/DebateLedger](https://huggingface.co/datasets/XINLI1997/DebateLedger), CC BY 4.0)

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

## Gated tier ([XINLI1997/DebateLedger-gated](https://huggingface.co/datasets/XINLI1997/DebateLedger-gated), research-use license)

The full-text versions of the 57 trace files that contain model text (agent
responses in debate rounds and probe response prefixes), and the very-strong
(convince-wrong) probe template and social-pressure suffix. These can be reused
as a misleading-answer corpus, so access is reviewed on request. In this
repository the two templates are replaced by a placeholder in
`abc_exp/config/prompts.py`, `abc_exp/scripts/run_alpha_probe_openrouter.py` and
`abc_exp/scripts/run_alpha_probe_vllm.py`; downloading the gated tier into the
repository root restores the full files.

No model weights are released.

## Reproducibility boundary

- The paper's tables are rebuilt from the open tier without API calls (commands in the README's
  [Quick start](../README.md#quick-start)). A rerun differs from the checked-in outputs in three documented
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
