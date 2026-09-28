# Correction-Preserving Policy Replay

Zero-cost replay over saved `debate_traces_*.jsonl`: a gate observes only within-trace R1 features, and when it fires the final answer is replayed as the initial majority. This prevents majority-level collapses only when the initial majority was correct, and loses corrections when the initial majority was wrong but standard debate ended correct.

Gate penalty used for learned stumps: `0.02` per gated debate in the training objective. Learned policies include a no-gate option, and use strict leave-one-model-out when the cohort has at least 4 trace sources.

## Inputs

| Source | Rows | Usable | Probe | Probe overlap | Primary? |
|---|---:|---:|---|---:|---:|
| gemini_3_1_flash_lite_n200 | 200 | 200 | gemini_3_1_flash_lite_n200 | 200 (100.0%) | yes |
| gemini_3_1_flash_lite_smoke | 2 | 2 | gemini_3_1_flash_lite_smoke | 0 (0.0%) | no |
| gemini_3_1_pro_n200 | 200 | 180 | gemini_3_1_pro_n200 | 180 (100.0%) | yes |
| gemini_3_1_pro_smoke | 2 | 2 | gemini_3_1_pro_n200 | 2 (100.0%) | no |
| gemini_3_flash_n200 | 200 | 200 | gemini_3_flash_n200 | 200 (100.0%) | yes |
| openrouter_deepseek-v4-flash | 200 | 180 | router_deepseek-v4-flash | 1 (0.6%) | yes |
| openrouter_deepseek-v4-pro | 81 | 79 | router_deepseek-v4-pro | 4 (5.1%) | yes |
| openrouter_glm_4_6_n20_partial | 7 | 7 | none | 0 (0.0%) | no |
| openrouter_grok_4_1_fast_n200 | 200 | 200 | openrouter_grok_4_1_fast_n200 | 200 (100.0%) | yes |
| openrouter_grok_4_1_fast_smoke | 1 | 1 | openrouter_grok_4_1_fast_smoke | 1 (100.0%) | no |
| openrouter_hy3_preview_free_smoke | 1 | 0 | openrouter_hy3_preview_free_smoke | 0 (0.0%) | no |
| openrouter_kimi-k2.6 | 3 | 2 | none | 0 (0.0%) | no |
| openrouter_llama_4_maverick_n20_partial | 4 | 4 | none | 0 (0.0%) | no |
| openrouter_mistral_small_4_n20 | 20 | 20 | openrouter_mistral_small_4_n20 | 3 (15.0%) | no |
| openrouter_mistral_small_4_n200 | 200 | 200 | openrouter_mistral_small_4_n200 | 200 (100.0%) | yes |
| smoke_qwen3-8b | 1 | 1 | none | 0 (0.0%) | no |
| vllm_gemma-4-31b-it | 200 | 200 | vllm_gemma-4-31b-it-awq | 3 (1.5%) | yes |
| vllm_qwen3.5-4b | 200 | 200 | vllm_qwen3.5-4b | 3 (1.5%) | yes |
| vllm_qwen3.5-9b | 200 | 199 | vllm_qwen3.5-9b | 3 (1.5%) | yes |
| vllm_qwen3.6-27b-fp8 | 200 | 200 | vllm_qwen3.6-27b-fp8 | 3 (1.5%) | yes |
| vllm_qwen3.6-35b-a3b-fp8 | 200 | 200 | vllm_qwen3.6-35b-a3b-fp8 | 3 (1.5%) | yes |
| vllm_qwen3_32b_a7_batch4_bench | 4 | 3 | none | 0 (0.0%) | no |
| vllm_qwen3_32b_a7_batch8_bench | 8 | 8 | none | 0 (0.0%) | no |
| vllm_qwen3_32b_a7_n200 | 200 | 200 | vllm_qwen3_32b_a7_n200 | 200 (100.0%) | yes |

## Primary Cohort

Models: 13; debates: 2438; standard collapses: 111; standard corrections: 194; probe coverage: 49.2%.

| Policy | Validation | Gate | Delta vs standard | CI95 delta | Prevented | Lost | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard | fixed | 0.0% | +0.00 pp | NA | 0 | 0 | 0 |
| naive_always_freeze | fixed_rule_no_training | 100.0% | -3.40 pp | [-4.89 pp, -1.98 pp] | 111 | 194 | -83 |
| naive_r1_majority_changed | fixed_rule_no_training | 16.2% | -2.17 pp | [-3.26 pp, -1.16 pp] | 67 | 120 | -53 |
| naive_any_r1_flip | fixed_rule_no_training | 35.1% | -3.24 pp | [-4.55 pp, -2.00 pp] | 90 | 169 | -79 |
| learned_r1_percentile_stump | strict_leave_one_model_out | 0.0% | +0.00 pp | [+0.00 pp, +0.00 pp] | 0 | 0 | 0 |
| learned_r1_probe_percentile_stump | strict_leave_one_model_out | 0.5% | -0.12 pp | [-0.33 pp, +0.04 pp] | 1 | 4 | -3 |

Verdict: not paper-worthy as a main claim; No strict LOMO policy has a positive question-bootstrap lower CI on accuracy delta.

## backend_gemini_api_exploratory

Models: 3; debates: 580; standard collapses: 1; standard corrections: 15; probe coverage: 100.0%.

| Policy | Validation | Gate | Delta vs standard | CI95 delta | Prevented | Lost | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard | fixed | 0.0% | +0.00 pp | NA | 0 | 0 | 0 |
| naive_always_freeze | fixed_rule_no_training | 100.0% | -2.41 pp | [-3.83 pp, -1.04 pp] | 1 | 15 | -14 |
| naive_r1_majority_changed | fixed_rule_no_training | 4.0% | -1.90 pp | [-3.25 pp, -0.69 pp] | 1 | 12 | -11 |
| naive_any_r1_flip | fixed_rule_no_training | 9.3% | -2.07 pp | [-3.45 pp, -0.86 pp] | 1 | 13 | -12 |
| learned_r1_percentile_stump | exploratory_in_sample_not_lomo | 0.0% | +0.00 pp | [+0.00 pp, +0.00 pp] | 0 | 0 | 0 |
| learned_r1_probe_percentile_stump | exploratory_in_sample_not_lomo | 0.0% | +0.00 pp | [+0.00 pp, +0.00 pp] | 0 | 0 | 0 |

Verdict: exploratory/not paper-worthy; No strict LOMO policy has a positive question-bootstrap lower CI on accuracy delta.

## backend_local_vllm_lomo

Models: 6; debates: 1199; standard collapses: 101; standard corrections: 159; probe coverage: 17.9%.

| Policy | Validation | Gate | Delta vs standard | CI95 delta | Prevented | Lost | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard | fixed | 0.0% | +0.00 pp | NA | 0 | 0 | 0 |
| naive_always_freeze | fixed_rule_no_training | 100.0% | -4.84 pp | [-7.66 pp, -2.07 pp] | 101 | 159 | -58 |
| naive_r1_majority_changed | fixed_rule_no_training | 28.1% | -2.50 pp | [-4.59 pp, -0.56 pp] | 62 | 92 | -30 |
| naive_any_r1_flip | fixed_rule_no_training | 58.5% | -4.75 pp | [-7.22 pp, -2.34 pp] | 83 | 140 | -57 |
| learned_r1_percentile_stump | strict_leave_one_model_out | 8.8% | -1.67 pp | [-2.45 pp, -1.01 pp] | 0 | 20 | -20 |
| learned_r1_probe_percentile_stump | strict_leave_one_model_out | 8.8% | -1.67 pp | [-2.45 pp, -1.01 pp] | 0 | 20 | -20 |

Verdict: not paper-worthy; No strict LOMO policy has a positive question-bootstrap lower CI on accuracy delta.

## backend_openrouter_lomo

Models: 4; debates: 659; standard collapses: 9; standard corrections: 20; probe coverage: 61.5%.

| Policy | Validation | Gate | Delta vs standard | CI95 delta | Prevented | Lost | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard | fixed | 0.0% | +0.00 pp | NA | 0 | 0 | 0 |
| naive_always_freeze | fixed_rule_no_training | 100.0% | -1.67 pp | [-3.20 pp, -0.15 pp] | 9 | 20 | -11 |
| naive_r1_majority_changed | fixed_rule_no_training | 5.5% | -1.82 pp | [-3.02 pp, -0.61 pp] | 4 | 16 | -12 |
| naive_any_r1_flip | fixed_rule_no_training | 15.2% | -1.52 pp | [-2.77 pp, -0.30 pp] | 6 | 16 | -10 |
| learned_r1_percentile_stump | strict_leave_one_model_out | 0.0% | +0.00 pp | [+0.00 pp, +0.00 pp] | 0 | 0 | 0 |
| learned_r1_probe_percentile_stump | strict_leave_one_model_out | 0.0% | +0.00 pp | [+0.00 pp, +0.00 pp] | 0 | 0 | 0 |

Verdict: not paper-worthy; No strict LOMO policy has a positive question-bootstrap lower CI on accuracy delta.

## excluded_tiny_or_smoke_exploratory

Models: 10; debates: 50; standard collapses: 2; standard corrections: 1; probe coverage: 12.0%.

| Policy | Validation | Gate | Delta vs standard | CI95 delta | Prevented | Lost | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard | fixed | 0.0% | +0.00 pp | NA | 0 | 0 | 0 |
| naive_always_freeze | fixed_rule_no_training | 100.0% | +2.00 pp | [-3.77 pp, +11.11 pp] | 2 | 1 | 1 |
| naive_r1_majority_changed | fixed_rule_no_training | 10.0% | -2.00 pp | [-5.46 pp, +0.00 pp] | 0 | 1 | -1 |
| naive_any_r1_flip | fixed_rule_no_training | 26.0% | +0.00 pp | [-4.88 pp, +6.82 pp] | 1 | 1 | 0 |
| learned_r1_percentile_stump | strict_leave_one_model_out | 54.0% | -2.00 pp | [-5.46 pp, +0.00 pp] | 0 | 1 | -1 |
| learned_r1_probe_percentile_stump | strict_leave_one_model_out | 54.0% | -2.00 pp | [-5.46 pp, +0.00 pp] | 0 | 1 | -1 |

Verdict: not paper-worthy; No strict LOMO policy has a positive question-bootstrap lower CI on accuracy delta.

## Interpretation

A positive net means the replay prevented more standard-debate collapses than it discarded corrections. The paper-worthy bar used here is stricter: a strict-LOMO learned policy must have positive net and a question-bootstrap 95% CI for accuracy delta entirely above zero.
