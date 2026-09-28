# Family-Level Headline Aggregation

Zero-cost diagnostic computed from `D5.2_gate_decision.md` section 1. The N=14 per-model table is collapsed to one row per family before recomputing Spearman association between `alpha_tot` and `C^cond`.

| Aggregation | G | Spearman rho | p (two-sided) | p (one-sided, positive) |
|---|---:|---:|---:|---:|
| Family mean | 7 | +0.8929 | 0.0123 | 0.0062 |
| Family median | 7 | +0.8929 | 0.0123 | 0.0062 |

## Family Aggregates

| Family | n models | mean alpha_tot | mean C^cond | median alpha_tot | median C^cond |
|---|---:|---:|---:|---:|---:|
| Anthropic | 1 | 0.4504 | 2.1505% | 0.4504 | 2.1505% |
| DeepSeek | 1 | 0.1129 | 0.0000% | 0.1129 | 0.0000% |
| Google | 2 | 0.2875 | 0.2941% | 0.2875 | 0.2941% |
| Meta | 1 | 0.7737 | 8.4646% | 0.7737 | 8.4646% |
| OpenAI | 2 | 0.3461 | 1.5227% | 0.3461 | 1.5227% |
| Phi | 1 | 0.4686 | 10.2377% | 0.4686 | 10.2377% |
| Qwen | 6 | 0.6987 | 19.7498% | 0.7100 | 10.2713% |

## Leave-One-Family Sensitivity (Family Mean)

| Dropped family | G remaining | rho | p (two-sided) | p (one-sided, positive) |
|---|---:|---:|---:|---:|
| Anthropic | 6 | +0.8286 | 0.0583 | 0.0292 |
| DeepSeek | 6 | +0.8286 | 0.0583 | 0.0292 |
| Google | 6 | +0.8286 | 0.0583 | 0.0292 |
| Meta | 6 | +1.0000 | 0.0028 | 0.0014 |
| OpenAI | 6 | +0.8286 | 0.0583 | 0.0292 |
| Phi | 6 | +0.9429 | 0.0167 | 0.0083 |
| Qwen | 6 | +0.9429 | 0.0167 | 0.0083 |

Worst family-mean leave-one-family case: drop `Anthropic`, rho = +0.8286.

Note: `C^cond` values are stored and reported in percentage points, matching the D5.2 source table.
