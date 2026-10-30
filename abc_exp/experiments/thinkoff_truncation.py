"""Truncation report and sensitivities for the Thinking-off rerun (four Qwen rows, thinking off, caps 8192; zero API; 2026-10-30).

The alpha headline keeps the original parser; truncated replies are coded
"undecided"; two sensitivities: (1) drop truncated triples, (2) strict parse that only accepts the final statement.
Debate truncation is handled the same way (main C^cond unchanged; drop debates with a truncated call in the rounds the
endpoint reads). Every number here is per model row; family-level statistics are in thinkoff_compare.py.

    python3 abc_exp/experiments/thinkoff_truncation.py [--dir abc_exp/results] [--suffix ""] \
        [--write-filtered DIR] [--tag thinkoff]

Truncated = finish_reason "length". Strict parse = the LAST "Final Answer: X" of a complete (non-truncated) generation;
a truncated generation, a repaired initial answer, or a complete one without that line is undecided. A triple whose
strict initial answer is undecided is dropped from the strict alpha; an undecided reply is left out of its row's
mean (bounds: counted as not revised / as revised).

Outputs: <out-dir>/thinkoff_truncation_<tag>.{json,md}; with --write-filtered, debate_traces_vllm_<stem>_droptrunc.jsonl
(debates with any truncated call in R0..R3 removed; the R3 endpoint's input) for the replay override.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from abc_exp.scripts.qc_thinkoff import debate_calls, degenerate, screen_calls  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
STEMS = ["qwen3.5-4b", "qwen3.5-9b", "qwen3.6-27b-fp8", "qwen3.6-35b-a3b-fp8"]
FA = re.compile(r"Final\s+Answer\s*:\s*\**\s*\(?\s*([A-J])\b", re.IGNORECASE)
TRUNC = "length"


def load(p: Path):
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def alpha_majority(answers):
    valid = [a for a in answers or [] if a]
    if not valid:
        return ""
    c = Counter(valid)
    m = max(c.values())
    return sorted(a for a, v in c.items() if v == m)[0]


def strict_answer(text, finish_reason):
    if finish_reason == TRUNC or not text:
        return None
    m = FA.findall(text)
    return m[-1].upper() if m else None


def stage_qc(st):
    out = {}
    for name, calls in st.items():
        n = len(calls)
        k = sum(c["finish_reason"] == TRUNC for c in calls)
        out[name] = {"n": n, "truncated": k, "truncated_pct": 100 * k / n if n else float("nan"),
                     "think_tags": sum(c["tags"] for c in calls),
                     "degenerate": sum(degenerate(c["text"]) for c in calls if c["text"] is not None),
                     "answer_none": sum(c["none"] for c in calls)}
    return out


def debate_trunc_rounds(r):
    """Per round R0..R4: number of agents whose call hit the cap."""
    rounds = [r["initial_states"]] + r["round_traces"]
    return [sum(a.get("finish_reason") == TRUNC for a in rd) for rd in rounds]


def c_cond(rows, drop_trunc: bool):
    """C^cond under both endpoints; drop_trunc removes debates with a truncated call in R0..R3 (R3) / R0..R4 (R4)."""
    out = {}
    for e, last in (("R3_alpha", 3), ("R4_stored", 4)):
        n = k = dropped = 0
        for d in rows:
            if drop_trunc and sum(debate_trunc_rounds(d)[:last + 1]):
                dropped += 1
                continue
            im = alpha_majority(d.get("initial_answers"))
            if not im or im != d.get("correct_label"):
                continue
            fin = (alpha_majority([x.get("answer") for x in d["round_traces"][2]]) if e == "R3_alpha"
                   else d.get("majority_answer") or alpha_majority(d.get("final_answers")))
            n += 1
            k += bool(fin) and fin != im
        out[e] = {"k": k, "n": n, "pct": 100 * k / n if n else float("nan"), "debates_dropped": dropped}
    return out


def probe_alpha(rows):
    main = statistics.mean(float(r["flip_rate"]) for r in rows)

    def trunc_triple(r):
        return ((r.get("initial_usage") or {}).get("finish_reason") == TRUNC
                or any((p.get("usage") or {}).get("finish_reason") == TRUNC for p in r["probe_results"]))

    kept = [r for r in rows if not trunc_triple(r)]
    # strict parse
    s_mid, s_lo, s_hi = [], [], []
    init_undecided = repaired = undecided_replies = changed_complete = 0
    for r in rows:
        if r.get("initial_parse_repaired"):
            repaired += 1
        si = None if r.get("initial_parse_repaired") else strict_answer(r.get("initial_text"), (r.get("initial_usage") or {}).get("finish_reason"))
        if si is None:
            init_undecided += 1
            continue
        dec, und = [], 0
        for p in r["probe_results"]:
            fr = (p.get("usage") or {}).get("finish_reason")
            sp = strict_answer(p.get("text"), fr)
            if sp is None:
                und += 1
                continue
            dec.append(sp != si)
            if fr != TRUNC and (sp != si) != bool(p["revised"]):
                changed_complete += 1
        undecided_replies += und
        tot = len(dec) + und
        if dec:
            s_mid.append(sum(dec) / len(dec))
        s_lo.append(sum(dec) / tot)
        s_hi.append((sum(dec) + und) / tot)
    return {"n_triples": len(rows), "alpha_main": main,
            "alpha_drop_truncated_triples": statistics.mean(float(r["flip_rate"]) for r in kept) if kept else float("nan"),
            "n_triples_kept": len(kept),
            "alpha_strict": statistics.mean(s_mid) if s_mid else float("nan"),
            "alpha_strict_lo": statistics.mean(s_lo) if s_lo else float("nan"),
            "alpha_strict_hi": statistics.mean(s_hi) if s_hi else float("nan"),
            "strict_initial_undecided": init_undecided, "initial_repaired": repaired,
            "strict_undecided_replies": undecided_replies,
            "strict_vs_original_revised_differs_on_complete_replies": changed_complete}


def trunc_vs_complete(debates, probes):
    """R0 accuracy (debates) and revised rate (probe replies), truncated vs complete calls."""
    out = {}
    if debates:
        for lab, sel in (("truncated", lambda a: a.get("finish_reason") == TRUNC), ("complete", lambda a: a.get("finish_reason") != TRUNC)):
            v = [a.get("answer") == d["correct_label"] for d in debates for a in d["initial_states"] if sel(a)]
            out[f"debate_R0_correct_{lab}"] = [sum(v), len(v)]
    if probes:
        for lab, sel in (("truncated", lambda u: u.get("finish_reason") == TRUNC), ("complete", lambda u: u.get("finish_reason") != TRUNC)):
            v = [bool(p["revised"]) for r in probes for p in r["probe_results"] if sel(p.get("usage") or {})]
            out[f"probe_reply_revised_{lab}"] = [sum(v), len(v)]
            v = [bool(r["initial_correct"]) for r in probes if sel(r.get("initial_usage") or {})]
            out[f"probe_initial_correct_{lab}"] = [sum(v), len(v)]
    return out


def recurring(debates, probes, min_calls=3):
    per_q = defaultdict(Counter)
    for d in debates:
        for i, k in enumerate(debate_trunc_rounds(d)):
            if k:
                per_q[d["question_id"]][f"debate_R{i}"] += k
    for r in probes:
        if (r.get("initial_usage") or {}).get("finish_reason") == TRUNC:
            per_q[r["question_id"]]["probe_initial"] += 1
        per_q[r["question_id"]]["probe_reply"] += sum((p.get("usage") or {}).get("finish_reason") == TRUNC for p in r["probe_results"])
    tot = {q: sum(c.values()) for q, c in per_q.items()}
    return [{"question_id": q, "truncated_calls": tot[q], **{k: v for k, v in per_q[q].items() if v}}
            for q in sorted(tot, key=lambda q: (-tot[q], q)) if tot[q] >= min_calls]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=RES)
    ap.add_argument("--suffix", default="", help='file suffix before .jsonl ("_thinkoff" inside thinking_off_rerun/)')
    ap.add_argument("--write-filtered", type=Path, default=None)
    ap.add_argument("--tag", default="thinkoff")
    ap.add_argument("--out-dir", type=Path, default=RES)
    a = ap.parse_args()
    out = {}
    for stem in STEMS:
        dpath = a.dir / f"debate_traces_vllm_{stem}{a.suffix}.jsonl"
        ppath = a.dir / f"sa_causal_vllm_{stem}{a.suffix}.jsonl"
        deb, pro = load(dpath), load(ppath)
        o = {"debate_file": dpath.name, "probe_file": ppath.name, "n_debates": len(deb), "n_triples": len(pro)}
        if deb:
            o["debate_qc"] = stage_qc(debate_calls(deb))
            o["debates_with_truncation_R0_R3"] = sum(1 for d in deb if sum(debate_trunc_rounds(d)[:4]))
            o["c_cond_main"] = c_cond(deb, False)
            o["c_cond_drop_truncated"] = c_cond(deb, True)
            if a.write_filtered:
                a.write_filtered.mkdir(parents=True, exist_ok=True)
                fp = a.write_filtered / f"debate_traces_vllm_{stem}_droptrunc.jsonl"
                with open(fp, "w") as fh:
                    for d in deb:
                        if not sum(debate_trunc_rounds(d)[:4]):
                            fh.write(json.dumps(d) + "\n")
                o["filtered_debate_file"] = fp.name
        if pro:
            o["probe_qc"] = stage_qc(screen_calls(pro))
            o["alpha"] = probe_alpha(pro)
        o["truncated_vs_complete"] = trunc_vs_complete(deb, pro)
        o["recurring_truncated_questions"] = recurring(deb, pro)
        out[stem] = o
    xs = [set(q["question_id"] for q in out[s]["recurring_truncated_questions"]) for s in STEMS if out[s]["recurring_truncated_questions"]]
    out["_recurring_in_2plus_rows"] = sorted(q for q in set().union(*xs) if sum(q in x for x in xs) >= 2) if xs else []

    md = [f"# Thinking-off rerun: truncation report and sensitivities ({a.tag})", "",
          "Truncated = finish_reason \"length\" at the 8192 cap. Main numbers keep the original parser.", "",
          "## 1. Per row and stage: truncated calls / think tags / repetitive outputs", "",
          "| row | stage | n | truncated | think tags | repetitive | answer None |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in STEMS:
        for kind in ("debate_qc", "probe_qc"):
            for st, v in sorted(out[s].get(kind, {}).items()):
                md.append(f"| {s} | {st} | {v['n']} | {v['truncated']} ({v['truncated_pct']:.2f}%) | {v['think_tags']} | {v['degenerate']} | {v['answer_none']} |")
    md += ["", "## 2. alpha_tot: main vs truncation sensitivities", "",
           "| row | triples | main (original parser) | drop truncated triples (kept) | strict final statement [undecided as no / as yes] | strict: undecided initial / replies |",
           "|---|---:|---:|---:|---:|---:|"]
    for s in STEMS:
        al = out[s].get("alpha")
        if al:
            md.append(f"| {s} | {al['n_triples']} | {al['alpha_main']:.4f} | {al['alpha_drop_truncated_triples']:.4f} ({al['n_triples_kept']}) | "
                      f"{al['alpha_strict']:.4f} [{al['alpha_strict_lo']:.4f}, {al['alpha_strict_hi']:.4f}] | "
                      f"{al['strict_initial_undecided']} / {al['strict_undecided_replies']} |")
    md += ["", "## 3. C^cond: main vs debates with truncation dropped", "",
           "| row | debates | with truncation in R0-R3 | R3_alpha main | R3_alpha dropped | R4_stored main | R4_stored dropped |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for s in STEMS:
        if "c_cond_main" in out[s]:
            m, d = out[s]["c_cond_main"], out[s]["c_cond_drop_truncated"]
            md.append(f"| {s} | {out[s]['n_debates']} | {out[s]['debates_with_truncation_R0_R3']} | "
                      + " | ".join(f"{x[e]['k']}/{x[e]['n']} ({x[e]['pct']:.2f}%)" for e in ("R3_alpha",) for x in (m, d)) + " | "
                      + " | ".join(f"{x[e]['k']}/{x[e]['n']} ({x[e]['pct']:.2f}%)" for e in ("R4_stored",) for x in (m, d)) + " |")
    md += ["", "## 4. Truncated vs complete calls", "",
           "| row | debate R0 correct: truncated / complete | probe initial correct: truncated / complete | probe reply revised: truncated / complete |",
           "|---|---:|---:|---:|"]

    def frac(x):
        return f"{x[0]}/{x[1]} ({100 * x[0] / x[1]:.1f}%)" if x and x[1] else "-"
    for s in STEMS:
        t = out[s]["truncated_vs_complete"]
        md.append(f"| {s} | " + " | ".join(f"{frac(t.get(k + '_truncated'))} / {frac(t.get(k + '_complete'))}"
                                           for k in ("debate_R0_correct", "probe_initial_correct", "probe_reply_revised")) + " |")
    md += ["", "## 5. Questions with >= 3 truncated calls (for provenance)", ""]
    for s in STEMS:
        qs = out[s]["recurring_truncated_questions"]
        md.append(f"- {s}: " + ("; ".join(f"{q['question_id']} {q['truncated_calls']}" for q in qs) if qs else "none"))
    md.append(f"- in two or more rows: {', '.join(out['_recurring_in_2plus_rows']) or 'none'}")
    (a.out_dir / f"thinkoff_truncation_{a.tag}.json").write_text(json.dumps(out, indent=1))
    (a.out_dir / f"thinkoff_truncation_{a.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
