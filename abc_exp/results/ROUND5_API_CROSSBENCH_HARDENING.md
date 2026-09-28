# Round-5 API Cross-Benchmark Hardening

Post-hoc reviewer-hardening runs. These are excluded from the primary MMLU-Pro family-level association and are used only as cross-benchmark stress/boundary evidence.

## Run Ledger

| Run | Role | Backend/model | Benchmarks | N | Row cost | Provider notes |
|---|---|---|---|---:|---:|---|
| Mistral Small 4 GPQA full | dense non-MMLU stress test | openrouter / `mistralai/mistral-small-2603` | gpqa | 198 | $3.3500 | {"Mistral": 6017, "Venice": 1521} |
| Gemini 3.1 Flash-Lite GPQA full | Google-family full-benchmark boundary check | gemini / `models/gemini-3.1-flash-lite-preview` | gpqa | 198 | $7.1227 |  |
| DeepSeek V4 Flash GPQA full | OpenRouter full-benchmark stress/boundary replication | openrouter / `deepseek/deepseek-v4-flash` | gpqa | 198 | $2.2635 | latest resume only: {"AtlasCloud": 6, "DeepInfra": 1, "DeepSeek": 12, "Novita": 3, "Parasail": 1}; full routing mixed, see logs |
| Llama 3.3 70B GPQA full | OpenRouter full-benchmark stress/boundary replication | openrouter / `meta-llama/llama-3.3-70b-instruct` | gpqa | 198 | $2.4032 | latest resume only: {"AkashML": 12, "Cloudflare": 2, "DeepInfra": 16, "Inceptron": 5, "Nebius": 16, "Novita": 8, "Parasail": 14, "SambaNova": 4, "WandB": 1}; full routing mixed, see logs |
| Gemini 3.1 Flash-Lite GPQA+TruthfulQA N50 | earlier sparse boundary check | gemini / `models/gemini-3.1-flash-lite-preview` | gpqa, truthfulqa | 100 | $2.6323 |  |
| Gemini 2.5 Flash ARC/GPQA/TQA N20 smoke | low-collapse smoke/boundary check | gemini / `models/gemini-2.5-flash` | arc, gpqa, truthfulqa | 60 | $3.1572 |  |
| Mistral Small 4 GPQA+TruthfulQA N20 smoke | model/benchmark scout before full GPQA | openrouter / `mistralai/mistral-small-2603` | gpqa, truthfulqa | 40 | $0.4736 | {"Mistral": 1343, "Venice": 184, "unknown": 1} |
| Mistral Small 4 GPQA N1 smoke | OpenRouter schema/cost smoke | openrouter / `mistralai/mistral-small-2603` | gpqa | 1 | $0.0120 | {"Mistral": 24, "Venice": 15} |

## Paper-Facing Rows

| Run | init acc | final acc | C^cond (Wilson 95%) | correction (Wilson 95%) | signed utility | alpha | alpha->collapse |
|---|---:|---:|---:|---:|---:|---:|---|
| Mistral Small 4 GPQA full | 60.6% | 61.1% | 20/120 (16.7% [11.1%, 24.3%]) | 21/78 (26.9% [18.3%, 37.7%]) | +1 (+0.51pp) | 0.577 (196/198 rows) | rho=+0.236, p=0.009574, n=120 |
| Gemini 3.1 Flash-Lite GPQA full | 74.2% | 77.8% | 2/147 (1.4% [0.4%, 4.8%]) | 9/51 (17.6% [9.6%, 30.3%]) | +7 (+3.54pp) | 0.533 (198/198 rows) | rho=+0.039, p=0.6403, n=147 |
| DeepSeek V4 Flash GPQA full | 56.1% | 74.7% | 10/111 (9.0% [5.0%, 15.8%]) | 47/87 (54.0% [43.6%, 64.1%]) | +37 (+18.69pp) | 0.255 (161/198 rows) | rho=+0.029, p=0.7833, n=95 |
| Llama 3.3 70B GPQA full | 51.5% | 55.1% | 6/102 (5.9% [2.7%, 12.2%]) | 13/96 (13.5% [8.1%, 21.8%]) | +7 (+3.54pp) | 0.537 (192/198 rows) | rho=+0.061, p=0.544, n=100 |

## Interpretation

- Mistral Small 4 on full GPQA Diamond is the dense non-MMLU stress test: nontrivial conditional collapse and correction both appear, and standard debate is only +1 net debate under equal collapse/correction weights.
- Llama 3.3 70B and DeepSeek V4 Flash add full-GPQA stress rows where collapses are nonzero but corrections dominate under equal weights. These rows strengthen the signed-utility claim, not benchmark-general alpha transfer.
- Gemini GPQA/ARC/TruthfulQA runs remain boundary checks: the full Gemini GPQA row has low collapse despite nonzero alpha, so low event counts should be reported as limits, not transfer success.
- OpenRouter rows must retain provider-routing notes. Full runs may be served by multiple providers and resumed summaries can contain only latest-invocation provider counts, so these are not snapshot-pinned vendor replications.

## Missingness

| Run | alpha-valid rows | missing agent initials | missing debate initials | missing final answers |
|---|---:|---:|---:|---:|
| Mistral Small 4 GPQA full | 196/198 | 23/594 | 4 | 0 |
| Gemini 3.1 Flash-Lite GPQA full | 198/198 | 7/594 | 0 | 0 |
| DeepSeek V4 Flash GPQA full | 161/198 | 296/594 | 40 | 15 |
| Llama 3.3 70B GPQA full | 192/198 | 71/594 | 7 | 3 |
| Gemini 3.1 Flash-Lite GPQA+TruthfulQA N50 | 100/100 | 2/300 | 0 | 0 |
| Gemini 2.5 Flash ARC/GPQA/TQA N20 smoke | 58/60 | 14/180 | 2 | 1 |
| Mistral Small 4 GPQA+TruthfulQA N20 smoke | 40/40 | 4/120 | 0 | 0 |
| Mistral Small 4 GPQA N1 smoke | 1/1 | 0/3 | 0 | 0 |
