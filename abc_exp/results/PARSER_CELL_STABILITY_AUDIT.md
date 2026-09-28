# Parser Cell-Stability Audit

Zero-API audit over checked-in debate traces. The reported strict parser keeps explicit answer patterns but removes the primary parser's permissive last-valid-letter fallback. We also compute a harsher final-tag-only diagnostic in JSON.

## Overall

- Rows audited: **1620**.
- Strict parser labeled both initial and final majorities for **966** rows; **654** rows were unlabeled (40.37%).
- Among strict-labeled rows, transition-cell membership changed for **51** rows (5.28%).
- Collapse label changed among strict-labeled rows: **20**; correction label changed: **29**.
- Primary majority reductions used alphabetic tie-breaking for **298 / 3240** initial/final state reductions (9.20%); **272 / 1620** rows had at least one primary initial/final tie (16.79%).
- Primary collapses/corrections in this open trace subset: **105** / **166**.
- Primary collapses/corrections made unlabeled by strict parsing: **66** / **133**.

## Per Trace

| Source | Rows | Primary tie states | Any tie rows | Strict unlabeled | Labeled cell changes | Collapse changes | Correction changes | Primary C/R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `deepseek-v4-flash` | 200 | 2 | 2 | 20 | 1 | 0 | 1 | 0/1 |
| `gemma-4-31b-it-awq` | 200 | 5 | 5 | 9 | 1 | 0 | 1 | 0/5 |
| `qwen3.5-4b` | 200 | 72 | 65 | 153 | 9 | 5 | 4 | 33/17 |
| `qwen3.5-9b` | 200 | 71 | 66 | 150 | 5 | 5 | 0 | 38/24 |
| `qwen3.6-27b-fp8` | 200 | 29 | 28 | 103 | 10 | 2 | 8 | 11/32 |
| `qwen3.6-35b-a3b-fp8` | 200 | 54 | 49 | 118 | 8 | 3 | 5 | 13/42 |
| `qwen3-32b-a7-n200` | 200 | 56 | 49 | 98 | 15 | 4 | 9 | 8/40 |
| `gemini-3.1-flash-lite-n200` | 200 | 7 | 6 | 3 | 1 | 0 | 1 | 0/5 |
| `mistral-small-4-n20` | 20 | 2 | 2 | 0 | 1 | 1 | 0 | 2/0 |

Interpretation: this is a stability check for trace files present in the open checkout. It does not reconstruct gated full transcripts or sealed lanes that are only available as aggregate transition tables.
