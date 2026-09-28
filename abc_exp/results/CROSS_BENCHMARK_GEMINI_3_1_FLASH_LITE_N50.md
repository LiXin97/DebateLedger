# Cross-Benchmark Forecast Replication

This is a small, independent Gemini sanity run. It is not a pre-registered headline result.

Model: `models/gemini-3.1-flash-lite-preview`; benchmarks: gpqa, truthfulqa; requested n/benchmark: 50.

## Summary

| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 100 | 73.0% | 75.0% | 1 | 1.4% | 3 | 0.472 | 0.029712319275601688 |
| gpqa | 50 | 62.0% | 66.0% | 1 | 3.2% | 3 | 0.574 | -0.019882542875406316 |
| truthfulqa | 50 | 84.0% | 84.0% | 0 | 0.0% | 0 | 0.371 | NA |
