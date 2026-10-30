"""Keep the bulk rows (nonzero `total_cost`) of the four Qwen3.5/3.6 probe files; later top-up rows are not used by the paper.

Rewrites the files in place, so run it on a working copy (see rebuild_r3.sh).

usage: bulk_alpha_rows.py <repo-root>
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
for s in ["qwen3.5-4b", "qwen3.5-9b", "qwen3.6-27b-fp8", "qwen3.6-35b-a3b-fp8"]:
    path = ROOT / f"abc_exp/results/sa_causal_vllm_{s}.jsonl"
    keep = [x for x in path.read_text().splitlines() if x.strip() and json.loads(x).get("total_cost")]
    assert len(keep) >= 1773, (s, len(keep))
    path.write_text("\n".join(keep) + "\n")
    print(s, len(keep), "bulk rows")
