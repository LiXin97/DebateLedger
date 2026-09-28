# Gemini R5 Stability Aggregate

Zero-API aggregation of same-prompt Gemini alpha panels. A true R5 non-determinism check requires repeated panels on separate calendar days; same-day comparisons are reported as smoke/stability diagnostics only.

## Panels

| Panel | records | questions | repeats | init-answer unstable | mean alpha range | max alpha range | post-answer None | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r5_gemini_alpha_panel_20260427_pilot` | 10 | 5 | 2--2 | 0 | 0.0000 | 0.0000 | 5.00% | $0.0609 |
| `r5_gemini_alpha_panel_20260427_day1` | 60 | 20 | 3--3 | 0 | 0.0000 | 0.0000 | 4.38% | $0.3520 |

## Pairwise Shared-Question Drift

| Panel A | Panel B | shared q | alpha pairs | Spearman | mean abs delta | max abs delta | init-answer changes |
|---|---|---:|---:|---:|---:|---:|---:|
| `r5_gemini_alpha_panel_20260427_pilot` | `r5_gemini_alpha_panel_20260427_day1` | 5 | 5 | 1.0000 | 0.0000 | 0.0000 | 0 |
