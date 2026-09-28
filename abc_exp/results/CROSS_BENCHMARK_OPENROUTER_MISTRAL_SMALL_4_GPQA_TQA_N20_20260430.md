# OpenRouter Cross-Benchmark Forecast Replication

Post-hoc reviewer-hardening smoke/expansion run. This is not a new primary headline unless explicitly promoted in the paper text.

Model alias: `mistral-small-4`; model id: `mistralai/mistral-small-2603`.
Benchmarks: gpqa, truthfulqa; requested n/benchmark: 20; seed: 48.
Observed OpenRouter cost: $0.4736 / $3.00 cap.
Providers: `{"Mistral": 1343, "Venice": 184, "unknown": 1}`.

## Summary

| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 40 | 60.0% | 72.5% | 1 | 4.2% | 6 | 0.622 | -0.244 |
| gpqa | 20 | 40.0% | 60.0% | 1 | 12.5% | 5 | 0.548 | -0.300 |
| truthfulqa | 20 | 80.0% | 85.0% | 0 | 0.0% | 1 | 0.696 | NA |
