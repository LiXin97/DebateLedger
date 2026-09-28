# Cross-Benchmark Forecast Replication

This is a small, independent Gemini sanity run. It is not a pre-registered headline result.

Model: `models/gemini-3.1-flash-lite-preview`; benchmarks: gpqa, truthfulqa; requested n/benchmark: 2.

## Summary

| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | 75.0% | 75.0% | 0 | 0.0% | 0 | 0.490 | NA |
| gpqa | 2 | 50.0% | 50.0% | 0 | 0.0% | 0 | 0.604 | NA |
| truthfulqa | 2 | 100.0% | 100.0% | 0 | 0.0% | 0 | 0.375 | NA |
