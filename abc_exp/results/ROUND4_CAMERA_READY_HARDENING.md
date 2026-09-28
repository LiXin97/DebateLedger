# Round-4 Camera-Ready Hardening

Zero-API diagnostics computed from checked-in artifacts. These rows are camera-ready hardening material, not new primary tests.

## EIV-Propagated Held-Out-Family Prediction Intervals

Jeffreys conditional-collapse uncertainty propagated through the six-family LoFO linear predictive fit; alpha is fixed at the measured family mean. Intervals are posterior predictive Monte Carlo quantiles, clipped to [0,100] for displayed coverage.

Coverage after clipping to [0,100]%: 80% 5/7; 95% 7/7.

| Held-out family | observed C^cond | EIV 80% PI clipped | EIV 95% PI clipped | covered 80 | covered 95 |
|---|---:|---:|---:|---:|---:|
| Anthropic | 2.15% | [+0.00, +15.88] | [+0.00, +23.25] | 1 | 1 |
| DeepSeek | 0.00% | [+0.00, +8.32] | [+0.00, +17.62] | 1 | 1 |
| Google | 0.29% | [+0.00, +12.60] | [+0.00, +21.01] | 1 | 1 |
| Meta | 8.46% | [+10.18, +28.39] | [+2.31, +35.95] | 0 | 1 |
| OpenAI | 1.52% | [+0.00, +13.89] | [+0.00, +21.77] | 1 | 1 |
| Phi | 10.24% | [+0.00, +15.01] | [+0.00, +22.60] | 1 | 1 |
| Qwen | 19.75% | [+2.22, +14.60] | [+0.00, +19.76] | 0 | 1 |

## Meta/Qwen Leverage Sensitivity

Model-row family-count inverse-Simpson effective family count: 4.08 over 14 model rows. This describes the model-row imbalance only; the primary inferential table remains the seven-family exact test.

| Sensitivity | G | rho | exact one-sided p | exact two-sided p |
|---|---:|---:|---:|---:|
| Meta dropped | 6 | +1.000 | 0.0014 | 0.0028 |
| Qwen dropped | 6 | +0.943 | 0.0083 | 0.0167 |
| Meta, Qwen dropped | 5 | +1.000 | 0.0083 | 0.0167 |

## R5 Closed-API Stability Panels

R5 remains a drift/nondeterminism audit, not a headline input.

| Panel | records | questions | init unstable | mean alpha range | max alpha range | post-None | cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r5_gemini_alpha_panel_20260427_day1` | 60 | 20 | 0 | 0.0000 | 0.0000 | 4.38% | $0.3520 |
| `r5_gemini_alpha_panel_20260428_day2` | 60 | 20 | 0 | 0.0000 | 0.0000 | 4.38% | $0.3520 |
| `r5_gemini_alpha_panel_20260430_day3` | 60 | 20 | 0 | 0.0000 | 0.0000 | 4.38% | $0.3520 |

| Panel pair | shared q | Spearman | mean abs delta | max abs delta | init-answer changes |
|---|---:|---:|---:|---:|---:|
| `r5_gemini_alpha_panel_20260427_day1` vs `r5_gemini_alpha_panel_20260428_day2` | 20 | 1.0000 | 0.0000 | 0.0000 | 0 |
| `r5_gemini_alpha_panel_20260427_day1` vs `r5_gemini_alpha_panel_20260430_day3` | 20 | 1.0000 | 0.0000 | 0.0000 | 0 |
| `r5_gemini_alpha_panel_20260428_day2` vs `r5_gemini_alpha_panel_20260430_day3` | 20 | 1.0000 | 0.0000 | 0.0000 | 0 |

## 6,525-Row Matched-Tau DG Audit

`pilot_gated_lomo.json` present: False.
Candidate LOMO JSON files found: none.
`per_debate_r1_features.jsonl` size: 0 bytes.
Conclusion: the strict 6,525-row matched-tau DisagreementGate/DRS comparison is not recoverable from this checkout and should remain explicitly disclaimed unless the gated matrix is restored.
