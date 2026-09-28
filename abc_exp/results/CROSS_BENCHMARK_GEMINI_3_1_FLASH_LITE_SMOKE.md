# Cross-Benchmark Forecast Replication

This is a small, independent Gemini sanity run. It is not a pre-registered headline result.

Model: `models/gemini-3.1-flash-lite-preview`; benchmarks: gpqa, truthfulqa; requested n/benchmark: 1.

## Summary

| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 2 | 50.0% | 50.0% | 0 | 0.0% | 0 | 0.479 | NA |
| gpqa | 1 | 0.0% | 0.0% | 0 | NA | 0 | 0.958 | NA |
| truthfulqa | 1 | 100.0% | 100.0% | 0 | 0.0% | 0 | 0.000 | NA |
