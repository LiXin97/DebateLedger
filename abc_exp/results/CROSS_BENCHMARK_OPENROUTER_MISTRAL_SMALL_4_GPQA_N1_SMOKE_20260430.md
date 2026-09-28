# OpenRouter Cross-Benchmark Forecast Replication

Post-hoc reviewer-hardening smoke/expansion run. This is not a new primary headline unless explicitly promoted in the paper text.

Model alias: `mistral-small-4`; model id: `mistralai/mistral-small-2603`.
Benchmarks: gpqa; requested n/benchmark: 1; seed: 47.
Observed OpenRouter cost: $0.0120 / $1.00 cap.
Providers: `{"Mistral": 24, "Venice": 15}`.

## Summary

| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 1 | 100.0% | 100.0% | 0 | 0.0% | 0 | 0.917 | NA |
| gpqa | 1 | 100.0% | 100.0% | 0 | 0.0% | 0 | 0.917 | NA |
