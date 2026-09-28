# Cross-Benchmark Forecast Replication

This is a small, independent Gemini sanity run. It is not a pre-registered headline result.

Model: `models/gemini-2.5-flash`; benchmarks: arc, gpqa, truthfulqa; requested n/benchmark: 20.

## Summary

| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 60 | 68.3% | 80.0% | 0 | 0.0% | 7 | 0.433 | NA |
| arc | 20 | 100.0% | 100.0% | 0 | 0.0% | 0 | 0.473 | NA |
| gpqa | 20 | 45.0% | 75.0% | 0 | 0.0% | 6 | 0.395 | NA |
| truthfulqa | 20 | 60.0% | 65.0% | 0 | 0.0% | 1 | 0.427 | NA |
