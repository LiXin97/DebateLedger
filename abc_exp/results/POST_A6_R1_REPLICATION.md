# Post-A6 R1 Replication

Majority-level metrics from `debate_traces_*.jsonl`. Collapse onset uses natural debate-round numbering: first debate round is R1.

## Cohort Comparison

| Cohort | N | Init Maj Correct | Collapses | Cond Collapse | Corrections | Final Acc | Collapse Onset |
|---|---:|---:|---:|---:|---:|---:|---|
| Legacy/partial traces | 82 | 63 | 2 | 3.2% | 5 | 80.5% | R4: 1, unlocalized: 1 |
| Post-A6 200-row traces | 1200 | 662 | 75 | 11.3% | 144 | 60.9% | R1: 40, R2: 10, R3: 11, R4: 14 |

## Pooled All Traces

- N debates: 1282
- Initial majority correct: 725 (56.6%)
- Collapses: 77 (10.6% conditional on initial majority correct)
- Corrections: 149 (26.8% of initial-majority-wrong debates)
- Final accuracy: 797 / 1282 (62.2%)
- Collapse onset: R1: 40, R2: 10, R3: 11, R4: 15, unlocalized: 1

## Per Model

| Model | Cohort | N | Init Maj Correct | Collapses | Cond Collapse | Corrections | Final Acc | Collapse Onset |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3-8B | legacy_or_partial | 1 | 1 | 0 | 0.0% | 0 | 100.0% | - |
| Qwen/Qwen3.5-4B | post_a6_200row | 200 | 67 | 28 | 41.8% | 24 | 31.5% | R1: 18, R2: 3, R3: 3, R4: 4 |
| Qwen/Qwen3.5-9B | post_a6_200row | 200 | 72 | 27 | 37.5% | 25 | 35.0% | R1: 21, R2: 2, R3: 2, R4: 2 |
| Qwen/Qwen3.6-27B-FP8 | post_a6_200row | 200 | 122 | 10 | 8.2% | 36 | 74.0% | R1: 1, R2: 2, R3: 4, R4: 3 |
| Qwen/Qwen3.6-35B-A3B-FP8 | post_a6_200row | 200 | 93 | 10 | 10.8% | 47 | 65.0% | R2: 3, R3: 2, R4: 5 |
| deepseek-v4-flash | post_a6_200row | 200 | 149 | 0 | 0.0% | 7 | 78.0% | - |
| deepseek-v4-pro | legacy_or_partial | 78 | 59 | 1 | 1.7% | 5 | 80.8% | R4: 1 |
| google/gemma-4-31b-it | post_a6_200row | 200 | 159 | 0 | 0.0% | 5 | 82.0% | - |
| kimi-k2.6 | legacy_or_partial | 3 | 3 | 1 | 33.3% | 0 | 66.7% | - |

## Sources

| File | Rows | Cohort |
|---|---:|---|
| `debate_traces_openrouter_deepseek-v4-flash.jsonl` | 200 | post_a6_200row |
| `debate_traces_openrouter_deepseek-v4-pro.jsonl` | 78 | legacy_or_partial |
| `debate_traces_openrouter_kimi-k2.6.jsonl` | 3 | legacy_or_partial |
| `debate_traces_smoke_qwen3-8b.jsonl` | 1 | legacy_or_partial |
| `debate_traces_vllm_gemma-4-31b-it.jsonl` | 200 | post_a6_200row |
| `debate_traces_vllm_qwen3.5-4b.jsonl` | 200 | post_a6_200row |
| `debate_traces_vllm_qwen3.5-9b.jsonl` | 200 | post_a6_200row |
| `debate_traces_vllm_qwen3.6-27b-fp8.jsonl` | 200 | post_a6_200row |
| `debate_traces_vllm_qwen3.6-35b-a3b-fp8.jsonl` | 200 | post_a6_200row |
