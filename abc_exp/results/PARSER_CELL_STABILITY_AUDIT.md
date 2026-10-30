# Parser Cell-Stability Audit

Zero-API audit over checked-in debate traces. The reported strict parser keeps explicit answer patterns but removes the primary parser's permissive last-valid-letter fallback. We also compute a harsher final-tag-only diagnostic in JSON.

## Overall

- Rows audited: **1620**.
- Strict parser labeled both initial and final majorities for **957** rows; **663** rows were unlabeled (40.93%).
- Among strict-labeled rows, transition-cell membership changed for **55** rows (5.75%).
- Collapse label changed among strict-labeled rows: **25**; correction label changed: **29**.
- Primary majority reductions used alphabetic tie-breaking for **321 / 3240** initial/final state reductions (9.91%); **290 / 1620** rows had at least one primary initial/final tie (17.90%).
- Primary collapses/corrections in this open trace subset: **137** / **136**.
- Primary collapses/corrections made unlabeled by strict parsing: **83** / **104**.

## Per Trace

| Source | Rows | Primary tie states | Any tie rows | Strict unlabeled | Labeled cell changes | Collapse changes | Correction changes | Primary C/R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `deepseek-v4-flash` | 200 | 2 | 2 | 20 | 1 | 0 | 1 | 0/1 |
| `gemma-4-31b-it-awq` | 200 | 5 | 5 | 11 | 0 | 0 | 0 | 0/4 |
| `qwen3.5-4b` | 200 | 69 | 63 | 153 | 9 | 5 | 5 | 32/17 |
| `qwen3.5-9b` | 200 | 73 | 62 | 150 | 7 | 4 | 4 | 39/20 |
| `qwen3.6-27b-fp8` | 200 | 37 | 36 | 107 | 13 | 8 | 5 | 28/28 |
| `qwen3.6-35b-a3b-fp8` | 200 | 71 | 65 | 121 | 8 | 4 | 3 | 30/24 |
| `qwen3-32b-a7-n200` | 200 | 55 | 49 | 98 | 16 | 4 | 10 | 8/37 |
| `gemini-3.1-flash-lite-n200` | 200 | 7 | 6 | 3 | 1 | 0 | 1 | 0/5 |
| `mistral-small-4-n20` | 20 | 2 | 2 | 0 | 0 | 0 | 0 | 0/0 |

Interpretation: this is a stability check for trace files present in the open checkout. It does not reconstruct gated full transcripts or sealed lanes that are only available as aggregate transition tables.
