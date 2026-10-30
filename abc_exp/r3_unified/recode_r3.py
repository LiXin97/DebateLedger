"""Recode debate traces to the paper's endpoint: Round 3 majority with alphabetic ties.

The traces of these rows log a fourth round, and their stored `majority_answer` and `final_answers` come from it.
This script sets `majority_answer`, `final_answers` and `majority_correct` from `round_traces[2]`. `round_traces`
stay intact. It rewrites the files in place, so run it on a working copy (see rebuild_r3.sh). It is idempotent.

usage: recode_r3.py <repo-root>
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
STEMS = ["openrouter_deepseek-v4-flash", "openrouter_deepseek-v4-pro", "vllm_gemma-4-31b-it", "vllm_qwen3.5-4b",
         "vllm_qwen3.5-9b", "vllm_qwen3.6-27b-fp8", "vllm_qwen3.6-35b-a3b-fp8", "vllm_qwen3_32b_a7_n200"]


def alpha_mode(answers):
    v = [str(a).strip() for a in answers if a is not None and str(a).strip()]
    if not v:
        return None
    c = Counter(v)
    return sorted(a for a, n in c.items() if n == max(c.values()))[0]


for stem in STEMS:
    path = ROOT / f"abc_exp/results/debate_traces_{stem}.jsonl"
    out, changed = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        r3 = [x.get("answer") for x in row["round_traces"][2]]
        m = alpha_mode(r3)
        changed += m != row.get("majority_answer")
        row["majority_answer"], row["final_answers"] = m, r3
        row["majority_correct"] = bool(m and m == str(row.get("correct_label") or "").strip())
        out.append(json.dumps(row))
    path.write_text("\n".join(out) + "\n")
    print(stem, len(out), "rows;", changed, "majorities changed")
