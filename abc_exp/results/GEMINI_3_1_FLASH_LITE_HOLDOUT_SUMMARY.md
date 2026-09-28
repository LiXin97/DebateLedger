# Gemini 3.1 Flash-Lite Holdout Summary

Model: `models/gemini-3.1-flash-lite-preview`.

## MMLU-Pro High-FR Alpha

- Rows: 1800 over 200 questions.
- Alpha total: 0.3017.
- Alpha by condition: anti_argument=0.3242, anti_social=0.2746, default=0.3065.
- Cost: $12.18.

## MMLU-Pro High-FR Debate

- Debates: 200.
- Initial/final majority accuracy: 82.00% -> 84.50%.
- Majority collapses: 0 / 164 at-risk (0.00%).
- Majority corrections: 5 / 36 initially-wrong (13.89%).
- Majority flip rate: 4.50%.
- Agent-level collapse/correction events: 2 / 7.
- Cost: $3.18.

## Cross-Benchmark Sanity

- GPQA + TruthfulQA N: 100.
- Initial/final accuracy: 73.0% -> 75.0%.
- Collapses: 1 (Ccond=1.37%).
- Mean alpha: 0.4725.

## Interpretation

This is a latest-Gemini Google-family holdout. It is strong as a boundary-condition / floor-collapse result, not as a new-family independent replication of the cross-model alpha-Ccond trend.

Practical paper use: include as a held-out latest-Gemini boundary condition; do not oversell it as strengthening the main cross-family correlation because the collapse rate is at the floor.
