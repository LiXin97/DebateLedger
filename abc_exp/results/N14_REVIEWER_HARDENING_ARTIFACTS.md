# N14 Reviewer-Hardening Artifacts

Zero-API derived artifact for the NeurIPS 2026 submission. This file does not overwrite sealed-vintage results.

## Family-Level Primary Inference

- Family-aggregated Spearman rho: **+0.8929** over G=7 families.
- Exact permutation one-sided p: **31/5040 = 0.0062**.
- Exact permutation two-sided p: **0.0123**.

## N14 Errors-in-Variables

- Method: `parametric_bootstrap_alpha_normal_jeffreys_theta`, seed=20260427, draws=20000.
- Point Spearman: **+0.8295**.
- Latent rank-correlation median: **+0.8022**, 95% interval [+0.7099, +0.8725], Pr(r>0)=1.000.

## Capability/Revision Confound Checks

- `alpha_controlling_init_acc`: rho=+0.6585, p=0.0144, controls=init_acc.
- `alpha_controlling_init_acc_and_revision_proxy`: rho=+0.6671, p=0.0178, controls=init_acc,debate_revision_proxy.
- `alpha_controlling_init_and_final_acc`: rho=+0.6464, p=0.0231, controls=init_acc,final_acc.
- `alpha_controlling_init_final_and_revision_proxy`: rho=+0.6842, p=0.0202, controls=init_acc,final_acc,debate_revision_proxy.
- `alpha_controlling_revision_proxy`: rho=+0.7800, p=0.0017, controls=debate_revision_proxy.

## Baseline/Confound Bakeoff

Only the fingerprint and initial-majority accuracy are pre-debate quantities. Post-debate and outcome-derived controls are included to quantify confounding pressure, not as deployable predictors.

| Predictor | Role | Model-row Spearman | Family-mean exact Spearman |
|---|---|---:|---:|
| 8-probe $\alpha_{\mathrm{tot}}$ | pre-debate fingerprint | +0.829 (n=14, p=0.000243) | +0.893 (G=7, p2=0.0123, p1-pos=0.00615) |
| conditional correction rate | post-debate transition control | +0.502 (n=14, p=0.0676) | +0.571 (G=7, p2=0.2, p1-pos=0.1) |
| raw debate revision proxy | revision-quantity baseline | +0.456 (n=14, p=0.101) | +0.714 (G=7, p2=0.0881, p1-pos=0.044) |
| final debate accuracy | post-debate descriptive control | -0.851 (n=14, p=0.000112) | -0.679 (G=7, p2=0.11, p1-neg=0.0548) |
| initial-majority accuracy | capability proxy | -0.803 (n=14, p=0.000543) | -0.821 (G=7, p2=0.0341, p1-neg=0.0171) |
| raw collapse rate | outcome-derived upper bound | +0.964 (n=14, p=3.1e-08) | +1.000 (G=7, p2=0.000397, p1-pos=0.000198) |

## Per-Model Conditional-Collapse Intervals

| Model | k/n | C_cond | Wilson 95% | Jeffreys 95% |
|---|---:|---:|---:|---:|
| `sonnet-4.5` | 2/93 | 2.15% | [0.59, 7.51] | [0.45, 6.72] |
| `deepseek-v4-flash` | 0/149 | 0.00% | [0.00, 2.51] | [0.00, 1.67] |
| `gemini-3-flash` | 1/170 | 0.59% | [0.10, 3.26] | [0.06, 2.72] |
| `gemma-4-31b-it-awq` | 0/160 | 0.00% | [0.00, 2.34] | [0.00, 1.56] |
| `llama-3.1-8b` | 86/1016 | 8.46% | [6.91, 10.34] | [6.87, 10.29] |
| `gpt-4o-mini` | 8/341 | 2.35% | [1.19, 4.56] | [1.12, 4.38] |
| `gpt-5.4-mini` | 1/143 | 0.70% | [0.12, 3.85] | [0.08, 3.22] |
| `phi-4-mini` | 112/1094 | 10.24% | [8.58, 12.18] | [8.55, 12.14] |
| `qwen3-4b` | 50/979 | 5.11% | [3.90, 6.67] | [3.86, 6.62] |
| `qwen3-8b` | 3/56 | 5.36% | [1.84, 14.61] | [1.53, 13.61] |
| `qwen3.5-4b` | 34/78 | 43.59% | [33.14, 54.64] | [32.98, 54.66] |
| `qwen3.5-9b` | 36/82 | 43.90% | [33.67, 54.68] | [33.53, 54.70] |
| `qwen3.6-27b-fp8` | 11/127 | 8.66% | [4.91, 14.85] | [4.69, 14.49] |
| `qwen3.6-35b-a3b-fp8` | 12/101 | 11.88% | [6.93, 19.63] | [6.66, 19.24] |

## Parser Failure Summary

| Debate trace | states | answer None | debates with init None | debates with final None | answer/final-tag mismatch |
|---|---:|---:|---:|---:|---:|
| `debate_traces_openrouter_deepseek-v4-flash.jsonl` | 3000 | 264 | 41 | 18 | 0 |
| `debate_traces_vllm_gemma-4-31b-it.jsonl` | 3000 | 7 | 2 | 0 | 0 |
| `debate_traces_vllm_qwen3.5-4b.jsonl` | 3000 | 21 | 16 | 1 | 1 |
| `debate_traces_vllm_qwen3.5-9b.jsonl` | 3000 | 20 | 11 | 0 | 0 |
| `debate_traces_vllm_qwen3.6-27b-fp8.jsonl` | 3000 | 11 | 4 | 0 | 0 |
| `debate_traces_vllm_qwen3.6-35b-a3b-fp8.jsonl` | 3000 | 15 | 10 | 0 | 8 |
| `debate_traces_vllm_qwen3_32b_a7_n200.jsonl` | 3000 | 24 | 6 | 3 | 2 |
| `debate_traces_gemini_3_1_flash_lite_n200.jsonl` | 3000 | 1 | 1 | 0 | 0 |
| `debate_traces_openrouter_mistral_small_4_n20.jsonl` | 300 | 0 | 0 | 0 | 0 |

| Alpha trace | rows | probe trials | post-answer None | post-answer None rate |
|---|---:|---:|---:|---:|
| `sa_causal_gemini_3.1_pro_smoke.jsonl` | 7 | 56 | 3 | 5.36% |
| `sa_causal_gemini_3_1_flash_lite_n200.jsonl` | 1800 | 14400 | 677 | 4.70% |
| `sa_causal_gemini_3_1_pro_harness_smoke.jsonl` | 18 | 144 | 1 | 0.69% |
| `sa_causal_gemini_3_1_pro_n200.jsonl` | 1791 | 14328 | 1659 | 11.58% |
| `sa_causal_gemini_3_flash_n200.jsonl` | 1800 | 14400 | 1 | 0.01% |
| `sa_causal_openrouter_grok_4_1_fast_n200.jsonl` | 1799 | 14392 | 33 | 0.23% |
| `sa_causal_openrouter_grok_4_1_fast_smoke.jsonl` | 9 | 72 | 5 | 6.94% |
| `sa_causal_openrouter_hy3_preview_free_n200.jsonl` | 1785 | 14280 | 133 | 0.93% |
| `sa_causal_openrouter_kimi_k2_6_n200.jsonl` | 6 | 48 | 5 | 10.42% |
| `sa_causal_openrouter_kimi_k2_6_smoke.jsonl` | 9 | 72 | 6 | 8.33% |
| `sa_causal_openrouter_kimi_k2_6_smoke.pre_reasoning_20260429.jsonl` | 4 | 32 | 28 | 87.50% |
| `sa_causal_openrouter_llama_3_3_70b_instruct_n200.jsonl` | 1797 | 14376 | 575 | 4.00% |
| `sa_causal_openrouter_llama_4_maverick_n200.jsonl` | 1794 | 14352 | 1316 | 9.17% |
| `sa_causal_openrouter_llama_4_scout_n200.jsonl` | 1800 | 14400 | 1439 | 9.99% |
| `sa_causal_openrouter_minimax_m2_7_n200.jsonl` | 147 | 1176 | 427 | 36.31% |
| `sa_causal_openrouter_minimax_m2_7_smoke.jsonl` | 1 | 8 | 8 | 100.00% |
| `sa_causal_openrouter_mistral_medium_3_1_n200.jsonl` | 550 | 4400 | 72 | 1.64% |
| `sa_causal_openrouter_mistral_small_4_n20.jsonl` | 180 | 1440 | 74 | 5.14% |
| `sa_causal_openrouter_mistral_small_4_n200.jsonl` | 1800 | 14400 | 314 | 2.18% |
| `sa_causal_openrouter_mistral_small_4_n20_paired.jsonl` | 180 | 1440 | 24 | 1.67% |
| `sa_causal_openrouter_qwen3_6_plus_n200.jsonl` | 18 | 144 | 2 | 1.39% |
| `sa_causal_openrouter_qwen3_6_plus_smoke.jsonl` | 9 | 72 | 56 | 77.78% |
| `sa_causal_openrouter_step_3_5_flash_n200.jsonl` | 103 | 824 | 772 | 93.69% |
| `sa_causal_openrouter_step_3_5_flash_smoke.jsonl` | 1 | 8 | 8 | 100.00% |
| `sa_causal_router_deepseek-v4-flash.jsonl` | 1793 | 14344 | 63 | 0.44% |
| `sa_causal_router_deepseek-v4-pro.jsonl` | 1614 | 12912 | 8014 | 62.07% |
| `sa_causal_vllm_gemma-4-31b-it-awq.jsonl` | 1800 | 14400 | 864 | 6.00% |
| `sa_causal_vllm_qwen3-32b.jsonl` | 1788 | 14304 | 119 | 0.83% |
| `sa_causal_vllm_qwen3.5-4b.jsonl` | 1800 | 14400 | 586 | 4.07% |
| `sa_causal_vllm_qwen3.5-9b.jsonl` | 1800 | 14400 | 185 | 1.28% |
| `sa_causal_vllm_qwen3.6-27b-fp8.jsonl` | 1800 | 14400 | 64 | 0.44% |
| `sa_causal_vllm_qwen3.6-35b-a3b-fp8.jsonl` | 1800 | 14400 | 397 | 2.76% |
| `sa_causal_vllm_qwen3_32b_a7_n200.jsonl` | 1800 | 14400 | 194 | 1.35% |

## Parse-Missing Sensitivity

- `current`: rho=+0.8295, two-sided asymptotic p=0.0002432.
- `new_lanes_none_as_flip`: rho=+0.8119, two-sided asymptotic p=0.000421.

## Artifact Provenance

| File | Rows | Timestamp range | Backend/model metadata present | SHA256 prefix |
|---|---:|---|---|---:|
| `sa_causal_router_deepseek-v4-flash.jsonl` | 1793 | 1777344002.008504 to 1777347303.8301806 | openrouter, deepseek-v4-flash | `3a1c482495fa` |
| `sa_causal_vllm_gemma-4-31b-it-awq.jsonl` | 1800 | -- | partial/missing | `eb7fcb5a6f27` |
| `sa_causal_vllm_qwen3.5-4b.jsonl` | 1800 | 1777343987.3511143 to 1777344250.6135895 | local_vllm, qwen3.5-4b | `66c1c711fef9` |
| `sa_causal_vllm_qwen3.5-9b.jsonl` | 1800 | 1777343994.8337178 to 1777344335.921805 | local_vllm, qwen3.5-9b | `4c4338615b0f` |
| `sa_causal_vllm_qwen3.6-27b-fp8.jsonl` | 1800 | 1777344385.7292206 to 1777344385.7293427 | local_vllm, qwen3.6-27b-fp8 | `71c2765221ee` |
| `sa_causal_vllm_qwen3.6-35b-a3b-fp8.jsonl` | 1800 | 1777344774.3709166 to 1777344774.371038 | local_vllm, qwen3.6-35b-a3b-fp8 | `c31b93c8b256` |
| `debate_traces_vllm_qwen3_32b_a7_n200.jsonl` | 200 | 2026-04-27T09:22:56 to 2026-04-27T11:21:36 | local_vllm, qwen3-32b-a7-local | `9a3c0db8c1a7` |
| `debate_traces_gemini_3_1_flash_lite_n200.jsonl` | 200 | 2026-04-26T23:53:42 to 2026-04-26T23:56:02 | gemini_api, models/gemini-3.1-flash-lite-preview | `2cfa125250f6` |
| `debate_traces_openrouter_mistral_small_4_n20.jsonl` | 20 | 2026-04-27T08:59:28 to 2026-04-27T09:04:00 | openrouter, mistral-small-4 | `991e3baae9e8` |
