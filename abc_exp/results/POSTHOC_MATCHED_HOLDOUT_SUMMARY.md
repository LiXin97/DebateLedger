# Post-Hoc Matched Holdout Summary

These lanes pair the same alpha probe with matched debate traces after the sealed headline cohort. They are diagnostics and external holdouts, not silent additions to the primary preregistered Spearman test.

| Lane | Family | alpha rows | debate rows | qid overlap | alpha | AS | AA | init acc | R3 acc | R4 acc | R3 Ccond | R4 Ccond | R4 correction | debate None | debate cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3 Flash | Google | 1800 | 200 | 200 | 0.1856 | 0.0321 | 0.3844 | 83.00% | 84.00% | 84.00% | 0.60% [0.1, 3.3] (1/166) | 0.60% [0.1, 3.3] (1/166) | 8.82% [3.0, 23.0] (3/34) | 0.03% | $6.92 |
| Gemini 3.1 Flash-Lite | Google | 1800 | 200 | 200 | 0.3017 | 0.2746 | 0.3242 | 82.00% | 84.50% | 84.50% | 0.00% [0.0, 2.3] (0/164) | 0.00% [0.0, 2.3] (0/164) | 13.89% [6.1, 28.7] (5/36) | 0.03% | $3.18 |
| Gemini 3.1 Pro | Google | 1791 | 199 | 199 | 0.2013 | 0.1330 | 0.3197 | 72.86% | 78.39% | 78.89% | 0.00% [0.0, 2.6] (0/145) | 0.00% [0.0, 2.6] (0/145) | 18.92% [9.5, 34.2] (7/37) | 10.42% | $28.79 |
| Mistral Small 4 | Mistral | 1800 | 200 | 200 | 0.3772 | 0.0660 | 0.6448 | 70.00% | 73.50% | 73.00% | 2.86% [1.1, 7.1] (4/140) | 5.00% [2.4, 10.0] (7/140) | 21.67% [13.1, 33.6] (13/60) | 0.07% | $3.02 |
| Grok 4.1 Fast | xAI | 1799 | 200 | 200 | 0.1452 | 0.0683 | 0.1845 | 83.50% | 85.00% | 84.50% | 0.00% [0.0, 2.2] (0/167) | 0.60% [0.1, 3.3] (1/167) | 9.09% [3.1, 23.6] (3/33) | 0.13% | $3.36 |

Ccond is the conditional-collapse rate among debates whose initial plurality answer was correct. Correction is the rate among debates whose initial plurality answer was wrong and whose final plurality answer became correct. Plurality answers drop unparsed agent answers and use a deterministic alphabetical tie-break; rows with no parsed final plurality remain in the denominator but are not counted as collapse or correction. R3 is reported for compatibility with three-round analyses; R4 is the stored final endpoint for these four-round holdout traces.
