# OpenRouter Cross-Benchmark Forecast Replication

Post-hoc reviewer-hardening smoke/expansion run. This is not a new primary headline unless explicitly promoted in the paper text.

Model alias: `llama-3.3-70b-instruct`; model id: `meta-llama/llama-3.3-70b-instruct`.
Benchmarks: gpqa; requested n/benchmark: 198; seed: 20260502.
Observed OpenRouter cost: $0.0241 / $2.00 cap.
Providers: `{"AkashML": 12, "Cloudflare": 2, "DeepInfra": 16, "Inceptron": 5, "Nebius": 16, "Novita": 8, "Parasail": 14, "SambaNova": 4, "WandB": 1}`.

## Summary

| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 198 | 51.5% | 55.1% | 6 | 5.9% | 13 | 0.537 | 0.071 |
| gpqa | 198 | 51.5% | 55.1% | 6 | 5.9% | 13 | 0.537 | 0.071 |
