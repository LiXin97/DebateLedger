# OpenRouter Cross-Benchmark Forecast Replication

Post-hoc reviewer-hardening smoke/expansion run. This is not a new primary headline unless explicitly promoted in the paper text.

Model alias: `mistral-small-4`; model id: `mistralai/mistral-small-2603`.
Benchmarks: gpqa; requested n/benchmark: 198; seed: 49.
Observed OpenRouter cost: $3.3500 / $8.00 cap.
Providers: `{"Mistral": 6017, "Venice": 1521}`.

## Summary

| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 198 | 60.6% | 61.1% | 20 | 16.7% | 21 | 0.577 | 0.202 |
| gpqa | 198 | 60.6% | 61.1% | 20 | 16.7% | 21 | 0.577 | 0.202 |
