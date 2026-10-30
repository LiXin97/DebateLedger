"""Rewrite the four Qwen3.5/3.6 rows of the table in D5.2_gate_decision.md from the recoded traces.

usage: d52_update_rows.py <repo-root>
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("rh", ROOT / "abc_exp/scripts/recompute_headline_n18.py")
m = importlib.util.module_from_spec(spec)
sys.modules["rh"] = m
spec.loader.exec_module(m)
pts = {p.model: p for p in m.load_cohort(ROOT / "abc_exp/results", ROOT / "abc_exp/results",
                                          ROOT / "abc_exp/scripts/cohort_n20_models.yaml", "r3_alpha")}
ROWS = ["qwen3.5-4b", "qwen3.5-9b", "qwen3.6-27b-fp8", "qwen3.6-35b-a3b-fp8"]
path = ROOT / "abc_exp/results/D5.2_gate_decision.md"
lines = path.read_text().split("\n")
done = set()
for i, line in enumerate(lines):
    tok = line.split()
    if tok and tok[0] in ROWS and tok[0] not in done:
        p = pts[tok[0]]
        lines[i] = f"{p.model:<33}{p.family:<12}{p.alpha_tot:.4f}  {p.c_cond:8.4f}%   {p.c_cond_wilson_halfwidth:.4f}"
        done.add(tok[0])
assert done == set(ROWS), done
path.write_text("\n".join(lines))
