"""Write the debate block of qwen3_32b_a7_n200_holdout_summary.json at the paper's endpoint.

Run after recode_r3.py. Recomputes the block from the recoded trace.

usage: holdout_32b_summary.py <repo-root>
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
TRACE = ROOT / "abc_exp/results/debate_traces_vllm_qwen3_32b_a7_n200.jsonl"
SUMMARY = ROOT / "abc_exp/results/qwen3_32b_a7_n200_holdout_summary.json"


def alpha_mode(answers):
    v = [str(a).strip() for a in answers if a is not None and str(a).strip()]
    if not v:
        return None
    c = Counter(v)
    return sorted(a for a, n in c.items() if n == max(c.values()))[0]


rows = [json.loads(x) for x in TRACE.read_text().splitlines() if x.strip()]
n = len(rows)
init = fin = col = cor = flip = 0
for row in rows:
    i, f, y = alpha_mode(row["initial_answers"]), row["majority_answer"], str(row["correct_label"]).strip()
    init += i == y
    fin += f == y
    col += i == y and f != y
    cor += i != y and f == y
    flip += i != f
s = json.loads(SUMMARY.read_text())
s["debate"].update(init_majority_correct=init, final_majority_correct=fin, initial_accuracy_pct=100 * init / n,
                   final_accuracy_pct=100 * fin / n, majority_collapses=col, majority_corrections=cor,
                   conditional_collapse_pct=100 * col / init, conditional_correction_pct=100 * cor / (n - init),
                   majority_flip_rate_pct=100 * flip / n, endpoint="Round 3 majority, alphabetic ties")
s["debate"].pop("agent_level_collapse_events", None)  # defined on the stored last round
s["debate"].pop("agent_level_correction_events", None)
SUMMARY.write_text(json.dumps(s, indent=2) + "\n")
print("qwen3-32b", {k: s["debate"][k] for k in ("final_accuracy_pct", "majority_collapses", "majority_corrections", "majority_flip_rate_pct")})
