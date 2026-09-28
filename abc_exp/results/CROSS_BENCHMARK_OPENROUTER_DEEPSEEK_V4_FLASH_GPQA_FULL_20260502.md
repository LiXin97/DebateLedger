# OpenRouter Cross-Benchmark Forecast Replication

Post-hoc reviewer-hardening smoke/expansion run. This is not a new primary headline unless explicitly promoted in the paper text.

Model alias: `deepseek-v4-flash`; model id: `deepseek/deepseek-v4-flash`.
Benchmarks: gpqa; requested n/benchmark: 198; seed: 20260502.
Observed OpenRouter cost: $0.0087 / $1.00 cap.
Providers: `{"AtlasCloud": 6, "DeepInfra": 1, "DeepSeek": 12, "Novita": 3, "Parasail": 1}`.

## Summary

| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 198 | 56.1% | 74.7% | 10 | 9.0% | 47 | 0.255 | -0.029 |
| gpqa | 198 | 56.1% | 74.7% | 10 | 9.0% | 47 | 0.255 | -0.029 |
