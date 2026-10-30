# Thinking-off rerun: thinking-on (old) vs thinking-off (new)

R3_alpha = Round 3, alphabetic ties, the endpoint of the paper.

## 1. C^cond (collapses / initial-majority-correct debates)

| | old R3_alpha | new R3_alpha | new, trunc dropped R3_alpha | new, strict alpha R3_alpha |
|---|---:|---:|---:|---:|
| deepseek-v4-flash | 0/149 (0.00%) | 0/149 (0.00%) | 0/149 (0.00%) | 0/149 (0.00%) |
| gemma-4-31b-it-awq | 0/160 (0.00%) | 0/160 (0.00%) | 0/160 (0.00%) | 0/160 (0.00%) |
| qwen3.5-4b | 32/78 (41.03%) | 10/154 (6.49%) | 7/144 (4.86%) | 10/154 (6.49%) |
| qwen3.5-9b | 39/82 (47.56%) | 4/158 (2.53%) | 3/151 (1.99%) | 4/158 (2.53%) |
| qwen3.6-27b-fp8 | 28/127 (22.05%) | 4/166 (2.41%) | 2/162 (1.23%) | 4/166 (2.41%) |
| qwen3.6-35b-a3b-fp8 | 30/101 (29.70%) | 4/164 (2.44%) | 3/162 (1.85%) | 4/164 (2.44%) |
| Qwen family C^cond (%) | 25.13 | 4.06 | 3.40 | 4.06 |
| Qwen family alpha_tot | 0.699 | 0.352 | 0.329 | 0.336 |
| family rho (G=7) | +0.8929 | +0.9286 | +0.8571 | +0.8571 |
| exact p | 0.0123 | 0.0067 | 0.0238 | 0.0238 |
| model-row rho (N=14) | +0.8603 | +0.6227 | +0.7393 | +0.5721 |

alpha_tot of the rerun rows: qwen3.5-4b 0.6897 -> 0.3974 / 0.3424 / 0.3605, qwen3.5-9b 0.7303 -> 0.2721 / 0.2204 / 0.2397, qwen3.6-27b-fp8 0.7950 -> 0.1160 / 0.0991 / 0.1027, qwen3.6-35b-a3b-fp8 0.7595 -> 0.1094 / 0.0950 / 0.0978 (main / trunc dropped / strict)

## 2. Replay cohort: collapses / corrections

| | old R3_alpha | new R3_alpha | new, trunc dropped R3_alpha | new, strict alpha R3_alpha |
|---|---:|---:|---:|---:|
| n | 1255 | 1256 | 1209 | 1256 |
| collapses | 129 | 22 | 15 | 22 |
| corrections | 97 | 23 | 16 | 23 |

## 3. Net per gate (prevented / lost / net, accuracy change)

| | old R3_alpha | new R3_alpha | new, trunc dropped R3_alpha | new, strict alpha R3_alpha |
|---|---:|---:|---:|---:|
| Probe-gated freeze (6,525 sealed, aggregate only) | 29 / 108 / -79 (-1.21pp) | 29 / 108 / -79 (-1.21pp) | 29 / 108 / -79 (-1.21pp) | 29 / 108 / -79 (-1.21pp) |
| Round 1 majority change | 74 / 56 / +18 (+1.43pp) | 16 / 17 / -1 (-0.08pp) | 12 / 11 / +1 (+0.08pp) | 16 / 17 / -1 (-0.08pp) |
| Learned Round 1 stump | 59 / 42 / +17 (+1.35pp) | 12 / 9 / +3 (+0.24pp) | 11 / 5 / +6 (+0.50pp) | 12 / 9 / +3 (+0.24pp) |
| Always freeze | 129 / 97 / +32 (+2.55pp) | 22 / 23 / -1 (-0.08pp) | 15 / 16 / -1 (-0.08pp) | 22 / 23 / -1 (-0.08pp) |
| DisagreementGate | 84 / 82 / +2 (+0.16pp) | 21 / 20 / +1 (+0.08pp) | 14 / 13 / +1 (+0.08pp) | 21 / 20 / +1 (+0.08pp) |
