#!/usr/bin/env python3
"""Acceptance check of the gate-rerun probe files. Per file: rows vs expected,
missing keys, duplicates, rows without 8 probes, initial / probe truncation, initial None, post None, and the share of
truncated probe replies whose last 600 characters repeat one 30-60 character piece at least 3 times (needs the model
text, so it is 0 on the open tier). Verdict per file: OK, or OVER_2PCT (more than 2% of probe replies hit the cap).

usage: gate_rerun_acceptance.py <probe-dir> [<qids-dir>]   writes <probe-dir>/acceptance.json and prints a table
"""
import json, os, sys
from collections import Counter

O = sys.argv[1]; Q = sys.argv[2] if len(sys.argv) > 2 else O
R = os.path.dirname(os.path.dirname(os.path.abspath(O)))
HF = json.load(open(os.path.join(R, "results/high_fr_question_ids.json")))
exp = {f"sa_causal_vllm_{s}_rerun.jsonl": {(q, c, a) for q in HF for c in ("default", "anti_social", "anti_argument") for a in range(3)}
       for s in ("llama-3.1-8b", "phi-4-mini", "qwen3-4b", "qwen3-8b")}
for s in ("llama-3.1-8b", "phi-4-mini", "qwen3-4b"):
    for k in (0, 1):
        exp[f"gate_default_vllm_{s}_s{k}.jsonl"] = {(x, "default", a) for x in json.load(open(os.path.join(Q, f"qids_{s}_s{k}.json"))) for a in range(3)}


def loop(t):
    tail = t[-600:]
    return any(tail.count(tail[i:i + L]) >= 3 for L in (30, 40, 60) for i in range(0, max(len(tail) - L, 0), 10))


out = {}
for f, keys in exp.items():
    p = os.path.join(O, f)
    if not os.path.exists(p): out[f] = dict(present=False); continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    got = Counter((r["question_id"], r["condition"], int(r["agent_idx"])) for r in rows)
    pr = [x for r in rows for x in r.get("probe_results") or []]
    tr = [x for x in pr if (x.get("usage") or {}).get("finish_reason") == "length"]
    it = sum((r.get("initial_usage") or {}).get("finish_reason") == "length" for r in rows)
    caps = sorted({r.get("max_tokens_probe") for r in rows}, key=str)
    d = dict(present=True, cap_probe=caps, expected=len(keys), rows=len(rows), missing=len(keys - set(got)), duplicates=sum(n > 1 for n in got.values()),
             rows_not_8=sum(len(r.get("probe_results") or []) != 8 for r in rows),
             init_trunc_pct=round(100 * it / max(len(rows), 1), 2), probe_trunc_pct=round(100 * len(tr) / max(len(pr), 1), 2),
             trunc_loop_share=round(sum(loop(x.get("text") or "") for x in tr) / max(len(tr), 1), 2),
             init_none=sum(r.get("initial_answer") is None for r in rows), post_none_pct=round(100 * sum(x.get("post_answer") is None for x in pr) / max(len(pr), 1), 2))
    d["verdict"] = "OVER_2PCT" if d["probe_trunc_pct"] > 2 else "OK"
    d["complete"] = d["duplicates"] == 0 and d["missing"] <= 0.01 * len(keys) and d["rows_not_8"] == 0
    out[f] = d
json.dump(out, open(os.path.join(O, "acceptance.json"), "w"), indent=1)
print(f"{'file':46} {'cap':>6} {'rows':>5} {'miss':>4} {'dup':>3} {'n8':>3} {'initT%':>6} {'prbT%':>6} {'loop':>5} {'iNone':>5} {'pNone%':>6} verdict complete")
for f, d in out.items():
    if not d["present"]: print(f"{f:46} MISSING"); continue
    print(f"{f:46} {str(d['cap_probe'][0] if d['cap_probe'] else '-'):>6} {d['rows']:5d} {d['missing']:4d} {d['duplicates']:3d} {d['rows_not_8']:3d} {d['init_trunc_pct']:6.2f} {d['probe_trunc_pct']:6.2f} {d['trunc_loop_share']:5.2f} {d['init_none']:5d} {d['post_none_pct']:6.2f} {d['verdict']} {d['complete']}")
