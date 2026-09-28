# Gemini R5 Nondeterminism Alpha Panel

This is a same-prompt alpha stability panel for closed-API drift auditing. It is a pilot unless repeated across multiple calendar days with the same question IDs and seed.

Model: `models/gemini-3.1-flash-lite-preview`; questions: 5; repeats: 2; seed: 20260427.

## Summary

- Records: 10 over 5 questions.
- Questions with initial-answer instability: 0.
- Mean within-question alpha range: 0.0000.
- Max within-question alpha range: 0.0000.
- Probe post-answer None rate: 5.0000%.
- Total estimated cost: $0.0609.

| Question | repeats | unique init answers | alpha values | alpha range | post-answer None |
|---|---:|---:|---:|---:|---:|
| `mmlu_pro_100` | 2 | 1 | 0.750, 0.750 | 0.000 | 0 |
| `mmlu_pro_125` | 2 | 1 | 1.000, 1.000 | 0.000 | 0 |
| `mmlu_pro_342` | 2 | 1 | 0.750, 0.750 | 0.000 | 4 |
| `mmlu_pro_915` | 2 | 1 | 0.750, 0.750 | 0.000 | 0 |
| `mmlu_pro_957` | 2 | 1 | 0.750, 0.750 | 0.000 | 0 |
