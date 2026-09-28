# Gemini R5 Nondeterminism Alpha Panel

This is a same-prompt alpha stability panel for closed-API drift auditing. It is a pilot unless repeated across multiple calendar days with the same question IDs and seed.

Model: `models/gemini-3.1-flash-lite-preview`; questions: 20; repeats: 3; seed: 20260427.

## Summary

- Records: 60 over 20 questions.
- Questions with initial-answer instability: 0.
- Mean within-question alpha range: 0.0000.
- Max within-question alpha range: 0.0000.
- Probe post-answer None rate: 4.3750%.
- Total estimated cost: $0.3520.

| Question | repeats | unique init answers | alpha values | alpha range | post-answer None |
|---|---:|---:|---:|---:|---:|
| `mmlu_pro_100` | 3 | 1 | 0.750, 0.750, 0.750 | 0.000 | 0 |
| `mmlu_pro_1240` | 3 | 1 | 0.750, 0.750, 0.750 | 0.000 | 0 |
| `mmlu_pro_125` | 3 | 1 | 1.000, 1.000, 1.000 | 0.000 | 0 |
| `mmlu_pro_342` | 3 | 1 | 0.750, 0.750, 0.750 | 0.000 | 6 |
| `mmlu_pro_3875` | 3 | 1 | 0.000, 0.000, 0.000 | 0.000 | 0 |
| `mmlu_pro_4151` | 3 | 1 | 1.000, 1.000, 1.000 | 0.000 | 0 |
| `mmlu_pro_4238` | 3 | 1 | 0.625, 0.625, 0.625 | 0.000 | 0 |
| `mmlu_pro_4356` | 3 | 1 | 0.750, 0.750, 0.750 | 0.000 | 0 |
| `mmlu_pro_4882` | 3 | 1 | 0.750, 0.750, 0.750 | 0.000 | 0 |
| `mmlu_pro_5303` | 3 | 1 | 0.250, 0.250, 0.250 | 0.000 | 0 |
| `mmlu_pro_5365` | 3 | 1 | 0.875, 0.875, 0.875 | 0.000 | 0 |
| `mmlu_pro_5386` | 3 | 1 | 0.625, 0.625, 0.625 | 0.000 | 0 |
| `mmlu_pro_5538` | 3 | 1 | 0.875, 0.875, 0.875 | 0.000 | 0 |
| `mmlu_pro_6085` | 3 | 1 | 0.750, 0.750, 0.750 | 0.000 | 0 |
| `mmlu_pro_6402` | 3 | 1 | 1.000, 1.000, 1.000 | 0.000 | 0 |
| `mmlu_pro_6446` | 3 | 1 | 1.000, 1.000, 1.000 | 0.000 | 0 |
| `mmlu_pro_7585` | 3 | 1 | 0.125, 0.125, 0.125 | 0.000 | 3 |
| `mmlu_pro_8885` | 3 | 1 | 0.500, 0.500, 0.500 | 0.000 | 12 |
| `mmlu_pro_915` | 3 | 1 | 0.750, 0.750, 0.750 | 0.000 | 0 |
| `mmlu_pro_957` | 3 | 1 | 0.750, 0.750, 0.750 | 0.000 | 0 |
