<div align="center">

# DebateLedger

### Measuring Collapse and Correction in Homogeneous-Panel LLM Debate

*NeurIPS 2026, Evaluations and Datasets Track*

Xin Li\*, Mengbing Liu\*, Chau Yuen<br>
Nanyang Technological University · \*Equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2609.35279-b31b1b?logo=arxiv)](https://arxiv.org/abs/2609.35279)
[![OpenReview](https://img.shields.io/badge/OpenReview-NeurIPS%202026-8c1b13)](https://openreview.net/forum?id=E8FfL8c7XE)
[![Project page](https://img.shields.io/badge/Project-page-059669)](https://lixin.ai/DebateLedger/)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data-DebateLedger-ffcc4d)](https://huggingface.co/datasets/XINLI1997/DebateLedger)
[![Code license](https://img.shields.io/badge/code-MIT-blue)](LICENSE)
[![Data license](https://img.shields.io/badge/data-per%20tier-lightgrey)](#license)

**DebateLedger evaluates multi-agent LLM debate by the transitions it causes rather than by final accuracy alone.**

</div>

## News

- **2026-09-28** The paper is on arXiv: [2609.35279](https://arxiv.org/abs/2609.35279). The data release alpha-tot-v1.0
  and the code are public.

## The idea in 30 seconds

- **Score transitions, not only final accuracy.** Three copies of one model answer a multiple-choice question and
  debate for three rounds. Each run is recorded as a transition ledger that crosses initial-majority with
  final-majority correctness (preserved, collapse, correction, unrepaired), together with the onset round at which a
  correct majority first breaks.
- **Score interventions on the same debates.** An intervention is scored on the same saved debates by the collapses it
  prevents and the corrections it loses.
- **Result.** On 6,925 MMLU-Pro debates the ledger records 253 collapses. A leave-one-model-out probe-gated freeze
  prevents 29 of 251 collapses but loses 108 of 804 corrections on 6,525 held-out debates, a net of −1.21 accuracy
  points under equal weights.
- **A pre-debate screen.** A pre-debate 8-probe screen ranks model families for trace logging (G = 7, Spearman ρ =
  0.893); initial-majority accuracy is a close comparator (ρ = 0.821), so the screen is a triage signal, not a
  calibrated predictor.

## Quick start

The paper's tables are rebuilt from the open tier without API calls:

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

## Data

The traces are distributed on Hugging Face in two tiers. Download the open tier into the root of this repository; the
files land in `abc_exp/results/`, where the scripts expect them.

- [🤗 XINLI1997/DebateLedger](https://huggingface.co/datasets/XINLI1997/DebateLedger) (open tier, CC BY 4.0): every
  trace with the model-generated text fields set to `null`.
- [🤗 XINLI1997/DebateLedger-gated](https://huggingface.co/datasets/XINLI1997/DebateLedger-gated) (gated tier,
  research-use license): the full-text versions of the trace files that contain model text; access is reviewed on
  request.

No model weights are released. The file list with record counts, the gated tier and the reproducibility boundary are in
[docs/DATA.md](docs/DATA.md).

## Repository layout

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
| `docs/DATA.md` | Data release: files, tiers and reproducibility boundary |

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
