"""QC for the thinking-off rerun (2026-10-30): think tags, truncation per stage, parse failures, token lengths,
and alignment with the original files.

    python3 abc_exp/scripts/qc_thinkoff.py --dir abc_exp/results/thinking_off_rerun [--smoke] [--sample 20]

Debate stages are R0..R4 (one generate per agent per round); screen stages are the initial answer, the format-repair
prompts and the eight post-challenge replies. Truncation = finish_reason "length". Think tags are counted in the
generated text only (with enable_thinking=False the template puts an empty think block in the prompt).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[2]
STEMS = ["qwen3.5-4b", "qwen3.5-9b", "qwen3.6-27b-fp8", "qwen3.6-35b-a3b-fp8"]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def tags(text):
    text = text or ""
    return text.count("<think>") + text.count("</think>")


def degenerate(text):
    """Repetition loop: the most common non-empty line is >= 30% of >= 5 lines (the 35B FP8 MoE failure mode)."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return len(lines) >= 5 and Counter(lines).most_common(1)[0][1] >= 0.3 * len(lines)


def pct(v, q):
    v = sorted(v)
    return v[int(q * (len(v) - 1))] if v else None


def stage_line(name, calls):
    n = len(calls)
    if not n:
        return f"  {name:<10} n=0"
    fr = Counter(c["finish_reason"] for c in calls)
    toks = [c["tokens"] for c in calls if c["tokens"] is not None]
    return (f"  {name:<10} n={n:<6} trunc={fr.get('length', 0)} ({fr.get('length', 0) / n:.2%})  "
            f"finish={dict(fr)}  think_tags={sum(c['tags'] for c in calls)}  answer_None={sum(c['none'] for c in calls)}  "
            f"degenerate={sum(degenerate(c['text']) for c in calls if c['text'] is not None)}  "
            f"tokens med/p90/p99/max={pct(toks, .5)}/{pct(toks, .9)}/{pct(toks, .99)}/{max(toks) if toks else None}")


def debate_calls(rows):
    st = defaultdict(list)
    for r in rows:
        for s in r["initial_states"]:
            st["R0"].append(dict(finish_reason=s.get("finish_reason"), tokens=s.get("completion_tokens"),
                                 tags=tags(s["reasoning"]), none=s["answer"] is None, text=s["reasoning"]))
        for rnd in r["round_traces"]:
            for s in rnd:
                st[f"R{s['round_num']}"].append(dict(finish_reason=s.get("finish_reason"), tokens=s.get("completion_tokens"),
                                                     tags=tags(s["reasoning"]), none=s["answer"] is None, text=s["reasoning"]))
    return st


def screen_calls(rows):
    st = defaultdict(list)
    for r in rows:
        u = r.get("initial_usage") or {}
        st["initial"].append(dict(finish_reason=u.get("finish_reason"), tokens=u.get("completion_tokens"),
                                  tags=tags(r.get("initial_text")), none=r["initial_answer"] is None, text=r.get("initial_text")))
        for u in r.get("initial_repair_usages") or []:
            st["repair"].append(dict(finish_reason=u.get("finish_reason"), tokens=u.get("completion_tokens"), tags=0, none=False, text=None))
        for p in r["probe_results"]:
            u = p.get("usage") or {}
            st["probe"].append(dict(finish_reason=u.get("finish_reason"), tokens=u.get("completion_tokens"),
                                    tags=tags(p.get("text")), none=p["post_answer"] is None, text=p.get("text"),
                                    revised=bool(p["revised"])))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(R / "abc_exp/results/thinking_off_rerun"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="print this many random generations (head and tail)")
    a = ap.parse_args()
    d = Path(a.dir)
    orig = R / "abc_exp/results"
    pairs = []
    if a.smoke:
        for k, stem in [("9b", "qwen3.5-9b"), ("27b", "qwen3.6-27b-fp8"), ("35b", "qwen3.6-35b-a3b-fp8")]:
            pairs.append((f"debate {stem}", d / f"smoke/debate_{k}.jsonl", "debate", orig / f"debate_traces_vllm_{stem}.jsonl"))
        pairs.append(("screen qwen3.5-9b", d / "smoke/screen_9b.jsonl", "screen", orig / "sa_causal_vllm_qwen3.5-9b.jsonl"))
    else:
        for stem in STEMS:
            pairs.append((f"debate {stem}", d / f"debate_traces_vllm_{stem}_thinkoff.jsonl", "debate", orig / f"debate_traces_vllm_{stem}.jsonl"))
            pairs.append((f"screen {stem}", d / f"sa_causal_vllm_{stem}_thinkoff.jsonl", "screen", orig / f"sa_causal_vllm_{stem}.jsonl"))
    samples = []
    for label, p, kind, op in pairs:
        rows = load(p)
        print(f"{label}: {len(rows)} rows  ({p.name})")
        if not rows:
            continue
        if kind == "debate":
            o = [json.loads(l)["question_id"] for l in open(op)]
            got = [r["question_id"] for r in rows]
            print(f"  order matches original prefix: {got == o[:len(got)]}; duplicates: {len(got) - len(set(got))}; "
                  f"enable_thinking={Counter(r.get('enable_thinking') for r in rows)}; max_tokens={Counter(r.get('max_tokens') for r in rows)}; "
                  f"vllm_env_override={Counter(json.dumps(r.get('vllm_env_override'), sort_keys=True) for r in rows)}")
            st = debate_calls(rows)
            print(f"  per-agent flips (outcome.collapse / outcome.correction, not the paper's C^cond): "
                  f"{sum(r['outcome']['collapse'] for r in rows)} / {sum(r['outcome']['correction'] for r in rows)}")
        else:
            o = {(r["question_id"], r["condition"], r["agent_idx"]) for r in load(op)}
            got = [(r["question_id"], r["condition"], r["agent_idx"]) for r in rows]
            print(f"  triples in original: {sum(t in o for t in got)}/{len(got)}; duplicates: {len(got) - len(set(got))}; "
                  f"enable_thinking={Counter(r.get('enable_thinking') for r in rows)}; caps={Counter((r.get('max_tokens_initial'), r.get('max_tokens_probe')) for r in rows)}")
            st = screen_calls(rows)
            print(f"  alpha_total mean={statistics.mean(r.get('alpha_total', r.get('flip_rate')) for r in rows):.4f}; "
                  f"vllm_env_override={Counter(json.dumps(r.get('vllm_env_override'), sort_keys=True) for r in rows)}")
            for lab, sel in (("truncated", lambda c: c["finish_reason"] == "length"), ("complete", lambda c: c["finish_reason"] != "length")):
                cs = [c for c in st["probe"] if sel(c)]
                if cs:
                    print(f"  probe replies {lab}: n={len(cs)} revised={sum(c['revised'] for c in cs)} ({sum(c['revised'] for c in cs) / len(cs):.1%})")
        for name in sorted(st):
            print(stage_line(name, st[name]))
            samples += [(label, name, c) for c in st[name] if c["text"] is not None]
    if a.sample and samples:
        random.seed(0)
        for label, name, c in random.sample(samples, min(a.sample, len(samples))):
            t = c["text"]
            print(f"\n--- {label} {name} finish={c['finish_reason']} tokens={c['tokens']} tags={c['tags']}\n"
                  f"HEAD: {t[:200]!r}\nTAIL: {t[-200:]!r}")


if __name__ == "__main__":
    main()
