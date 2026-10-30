# Reviewer-Hardening Analyses

Zero-cost diagnostics on the realized N=14 headline cohort. These are intended to block common reviewer confounds, not to create a new primary claim.

## Predictor Bakeoff vs Conditional Collapse

Only the fingerprint and initial-majority accuracy are pre-debate quantities. Final accuracy, correction rate, and raw collapse are included as descriptive controls rather than deployable predictors.

| Predictor | Role | Model-row Spearman | Family-mean exact Spearman | Note |
|---|---|---:|---:|---|
| 8-probe $\alpha_{\mathrm{tot}}$ | pre-debate fingerprint | +0.860 (n=14, p=7.89e-05) | +0.893 (G=7, exact p2=0.0123, p1-pos=0.00615) | predictor of record; measured before debate compute |
| initial-majority accuracy | capability proxy | -0.794 (n=14, p=0.000691) | -0.821 (G=7, exact p2=0.0341, p1-neg=0.0171) | pre-debate capability/difficulty proxy; negative rho means weaker initial panels collapse more |
| final debate accuracy | post-debate descriptive control | -0.934 (n=14, p=1.04e-06) | -0.786 (G=7, exact p2=0.048, p1-neg=0.024) | not deployable as a predictor; included to measure capability confounding pressure |
| conditional correction rate | post-debate transition control | +0.440 (n=14, p=0.115) | +0.571 (G=7, exact p2=0.2, p1-pos=0.1) | not deployable pre-debate; captures the opposing productive-revision transition |
| raw debate revision proxy | revision-quantity baseline | +0.472 (n=14, p=0.0885) | +0.714 (G=7, exact p2=0.0881, p1-pos=0.044) | debate FR for sealed rows and majority-flip rate for post-A6 trace rows |
| raw collapse rate | outcome-derived upper bound | +0.968 (n=14, p=1.44e-08) | +1.000 (G=7, exact p2=0.000397, p1-pos=0.000198) | shares the collapse numerator with C_cond; sanity check only, not a usable predictor |

## Partial Spearman Checks

Rank residualization tests whether the fingerprint remains associated with conditional collapse after removing simple capability or revision-quantity explanations. These are observational sensitivity checks on a small cohort, not new confirmatory tests.

| Check | Result | Controls |
|---|---:|---|
| `alpha_controlling_init_acc` | +0.727 (n=14, p=0.00485) | init_acc |
| `alpha_controlling_init_and_final_acc` | +0.535 (n=14, p=0.0732) | init_acc, final_acc |
| `alpha_controlling_revision_proxy` | +0.816 (n=14, p=0.000672) | debate_revision_proxy |
| `alpha_controlling_init_acc_and_revision_proxy` | +0.740 (n=14, p=0.0059) | init_acc, debate_revision_proxy |
| `alpha_controlling_init_final_and_revision_proxy` | +0.523 (n=14, p=0.0985) | init_acc, final_acc, debate_revision_proxy |

## Family Aggregation

- Family mean aggregation: +0.893 (G=7, exact p2=0.0123, p1-pos=0.00615)
- Family median aggregation: +0.893 (G=7, exact p2=0.0123, p1-pos=0.00615)

## Trace-Available Initial Diversity Diagnostics

Trace-available diagnostic only: six post-A6 rows with raw initial answers (DeepSeek, Gemma, and four Qwen variants). These metrics are measured after the three initial answers are sampled, so they are not substitutes for the single-model pre-debate alpha fingerprint.

| Metric | Spearman vs C^cond | Note |
|---|---:|---|
| initial disagreement rate | +0.928 (n=6, p=0.00767) | fraction of debates where the three initial answers are not unanimous |
| initial unanimity rate | -0.928 (n=6, p=0.00767) | fraction of debates where all parsed initial answers agree |
| normalized initial-answer entropy | +0.928 (n=6, p=0.00767) | mean Shannon entropy of initial answers, normalized by log2(3) |
| at-risk initial disagreement rate | +0.928 (n=6, p=0.00767) | initial disagreement restricted to debates whose initial majority is correct |

Partial alpha check within the same trace-only slice, controlling initial disagreement: +0.196 (n=6, p=0.752). This is underpowered (n=6) and is included only to make the disagreement confound explicit.

| Model | debates | alpha | C^cond | init disagree | init unanimous | init entropy | at-risk disagree |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-v4-flash | 200 | 0.1129 | 0.00% | 4.00% | 76.00% | 0.023 | 0.67% |
| gemma-4-31b-it-awq | 200 | 0.3106 | 0.00% | 12.00% | 87.00% | 0.078 | 5.62% |
| qwen3.5-4b | 200 | 0.6897 | 41.03% | 71.00% | 25.50% | 0.523 | 58.97% |
| qwen3.5-9b | 200 | 0.7303 | 47.56% | 68.00% | 29.00% | 0.500 | 52.44% |
| qwen3.6-27b-fp8 | 200 | 0.7950 | 22.05% | 52.50% | 46.00% | 0.353 | 37.01% |
| qwen3.6-35b-a3b-fp8 | 200 | 0.7595 | 29.70% | 59.00% | 39.00% | 0.423 | 39.60% |

## Selection Utility Diagnostic

Median alpha split: `0.5377`.

| Group | Models | C^cond | Correction | Final acc |
|---|---:|---:|---:|---:|
| low alpha group | 7 | 5.77% | 15.12% | 62.64% |
| high alpha group | 7 | 10.99% | 26.39% | 55.96% |

Accuracy-matched pair window on initial accuracy: +/-0.05.
Low-alpha model has lower conditional collapse in 7/11 matched pairs; mean low-minus-high C^cond delta = -6.17pp.

## Notes

- debate_revision_proxy is debate FR for sealed conditional-collapse rows and majority-flip rate for post-A6 trace rows; use only as a coarse anti-raw-revision diagnostic.
- initial disagreement/diversity metrics are available only for six post-A6 raw-trace rows and are reported as runtime diagnostics, not as the primary selection-time fingerprint.
- final_acc, correction_pct, and raw_collapse_pct are post-debate or outcome-derived controls; they are not deployable pre-debate predictors.
- selection utility is descriptive and zero-cost; it is not a learned deployment policy.
