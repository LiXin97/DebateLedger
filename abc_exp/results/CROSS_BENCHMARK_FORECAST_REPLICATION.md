# Cross-Benchmark Forecast Replication

This is a small, independent Gemini sanity run. It is not a pre-registered headline result.

Model: `models/gemini-2.5-flash`; benchmarks: gpqa, truthfulqa; requested n/benchmark: 3.

## Summary

| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 6 | 66.7% | 83.3% | 0 | 0.0% | 1 | 0.658 | NA |
| gpqa | 3 | 33.3% | 66.7% | 0 | 0.0% | 1 | 0.792 | NA |
| truthfulqa | 3 | 100.0% | 100.0% | 0 | 0.0% | 0 | 0.569 | NA |
