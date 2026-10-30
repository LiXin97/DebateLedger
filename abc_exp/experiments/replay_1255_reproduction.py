"""Reproduce the paper's 1,255-debate correction-preserving replay from the debate traces (zero API, zero GPU).

Run after abc_exp/r3_unified/recode_r3.py, which sets the Round 3 endpoint (rebuild_r3.sh does both). Seven sources:
DeepSeek-V4-Flash, the first 78 rows of the DeepSeek-V4-Pro file, Gemma-4-31B-it and the four Qwen3.5/3.6 rows. Recomputes
every gate and the weight ladder, cross-checks the counts with an independent raw-JSON pass, checks every number the paper
prints, and reports sensitivity to the endpoint, the tie rule and the cohort.

Usage:
  python3 abc_exp/experiments/replay_1255_reproduction.py --tag r3unified [--strict]

Outputs: abc_exp/results/replay_1255_reproduction_<tag>.json / .md
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]  # abc_exp/
REPO_ROOT = ROOT.parent
DEFAULT_RESULTS = ROOT / "results"
REPLAY_PATH = Path(__file__).resolve().with_name("correction_preserving_policy_replay.py")

_spec = importlib.util.spec_from_file_location("correction_preserving_policy_replay", REPLAY_PATH)
cpr = importlib.util.module_from_spec(_spec)
sys.modules["correction_preserving_policy_replay"] = cpr
_spec.loader.exec_module(cpr)  # import only; its main() is guarded
ORIG_MAJORITY = cpr.majority  # alphabetic-tie majority, kept to restore after patching

# --------------------------------------------------------------------------------------
# The paper's cohort: seven sources.  (stem, max raw rows or None, sha256 of consumed bytes
# of the open-tier trace after recode_r3.py; the full-text tier gives other digests).
# --------------------------------------------------------------------------------------
SOURCES: list[tuple[str, int | None, str]] = [
    ("openrouter_deepseek-v4-flash", None, "d18c4e74ac5cdd67863b9d44599400d0f1c4834e999a13fc57e406584633da19"),
    ("openrouter_deepseek-v4-pro", 78, "e351f56380f56c3c6cd0e162e61630cc55c1d8905a7374cd1a7a2c45a9cf5a39"),
    ("vllm_gemma-4-31b-it", None, "c017517782703e85085ab0633007295c0827e9686e43a360d0ebd73c7a43e839"),
    ("vllm_qwen3.5-4b", None, "2dacc193a41e9ef5b8580227e91c6a090a1b38ab4fb85579916d6f3ef9cebe80"),
    ("vllm_qwen3.5-9b", None, "136ea2e7574bffd146dc257f7e6096653087fc5e43ee5351bfcb33841edc77b0"),
    ("vllm_qwen3.6-27b-fp8", None, "99ef4d144f8f0b2b9154130bde1da7467bcc19c42d0234dc50d3e4b5449d03a2"),
    ("vllm_qwen3.6-35b-a3b-fp8", None, "63339da366d54cea11d8d5d3120540ace36f22afdec5b3a5b6e1c3353ab373a8"),
]
QWEN_ROWS = ["vllm_qwen3.5-4b", "vllm_qwen3.5-9b", "vllm_qwen3.6-27b-fp8", "vllm_qwen3.6-35b-a3b-fp8"]
DS_PRO = "openrouter_deepseek-v4-pro"
WEIGHTS = [0.5, 1.0, 2.0, 4.0, 8.0]
COMMITTED_JSON_COMMIT = "80106ed"  # initial snapshot whose correction_preserving_policy_replay.json is the 1,255 cohort
OUT_PREFIX = "replay_1255_reproduction"


# ------------------------------------------------------------------ small helpers
def majority_first_seen(answers: list[Any]) -> str:
    valid = [str(a).strip() for a in answers if a is not None and str(a).strip()]
    if not valid:
        return ""
    counts = Counter(valid)  # insertion ordered
    top = max(counts.values())
    return next(a for a, c in counts.items() if c == top)


@contextlib.contextmanager
def patched_majority(fn):
    cpr.majority = fn
    try:
        yield
    finally:
        cpr.majority = ORIG_MAJORITY


def read_rows(path: Path, max_rows: int | None) -> tuple[list[dict[str, Any]], str, int]:
    """Non-empty lines up to max_rows; returns (rows, sha256 of consumed bytes, total non-empty lines)."""
    rows: list[dict[str, Any]] = []
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        for raw in fh:
            if raw.strip():
                total += 1
            if max_rows is not None and len(rows) >= max_rows:
                continue  # keep counting total, stop consuming
            h.update(raw)
            if raw.strip():
                rows.append(json.loads(raw))
    return rows, h.hexdigest(), total


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ladder(prevented: int, lost: int, n: int) -> dict[str, Any]:
    """Accuracy delta in pp when a prevented collapse is worth r corrections: (r*P - L)/N."""
    return {
        "delta_pp": {f"{w:g}": (w * prevented - lost) / n * 100.0 for w in WEIGHTS},
        "breakeven_r": (lost / prevented) if prevented else None,
    }


def f2(x: float) -> str:
    s = f"{x:.2f}"
    return "0.00" if s in ("-0.00",) else s


# ------------------------------------------------------------------ record building
def build_records(row_sets, probe_means, probe_norms, endpoint="stored", init_tie="alpha"):
    """row_sets: [(source, rows)].  endpoint in {stored, R4_alpha, R3_alpha}; init_tie in {alpha, first_seen}."""
    recs = []
    ctx = patched_majority(majority_first_seen) if init_tie == "first_seen" else contextlib.nullcontext()
    with ctx:
        for source, rows in row_sets:
            probe_source = cpr.matched_probe_source(source, probe_norms)
            probe_by_q = probe_means.get(probe_source or "", {})
            for row in rows:
                if endpoint != "stored":
                    row = dict(row)
                    if endpoint == "R4_alpha":
                        row["majority_answer"] = ORIG_MAJORITY(cpr.round_answers(row, 3) or row.get("final_answers") or [])
                    elif endpoint == "R3_alpha":
                        row["majority_answer"] = ORIG_MAJORITY(cpr.round_answers(row, 2))
                    else:
                        raise ValueError(endpoint)
                qid = str(row.get("question_id") or "")
                rec = cpr.featurize_trace(row, source, probe_by_q.get(qid))
                if rec is not None:
                    recs.append(rec)
    return recs


def independent_counts(row_sets) -> dict[str, Any]:
    """Counts from raw JSON only (no replay-module code): usable, collapses, corrections, gates."""
    out: dict[str, Any] = {}
    tot = Counter()
    for source, rows in row_sets:
        c = Counter()
        for row in rows:
            ia = [str(a).strip() for a in (row.get("initial_answers") or []) if a is not None and str(a).strip()]
            fa = row.get("final_answers") or []
            lab = str(row.get("correct_label") or "").strip()
            if not row.get("initial_answers") or not fa or not lab or not ia:
                continue
            cnt = Counter(ia)
            top = max(cnt.values())
            init_maj = sorted(a for a, k in cnt.items() if k == top)[0]
            std = str(row.get("majority_answer") or "").strip()
            if not std:
                v = [str(a).strip() for a in fa if a is not None and str(a).strip()]
                if not v:
                    continue
                cc = Counter(v)
                std = sorted(a for a, k in cc.items() if k == max(cc.values()))[0]
            rt0 = (row.get("round_traces") or [None])[0]
            r1_maj = ""
            if isinstance(rt0, dict):  # dict form carries a stored majority
                r1_maj = str(rt0.get("majority") or "").strip()
                r1_list = rt0.get("answers") or []
            elif isinstance(rt0, list):  # these seven files: list of per-agent dicts, no stored majority
                r1_list = [item.get("answer") for item in rt0]
            else:
                r1_list = []
            if not r1_maj:
                v = [str(a).strip() for a in r1_list if a is not None and str(a).strip()]
                if v:
                    cc = Counter(v)
                    r1_maj = sorted(a for a, k in cc.items() if k == max(cc.values()))[0]
            ic, sc = init_maj == lab, std == lab
            col, cor = ic and not sc, (not ic) and sc
            c["usable"] += 1
            c["collapses"] += col
            c["corrections"] += cor
            r1chg = bool(r1_maj) and r1_maj != init_maj
            c["r1chg_gated"] += r1chg
            c["r1chg_prevented"] += r1chg and col
            c["r1chg_lost"] += r1chg and cor
            dg = len(set(ia)) > 1
            c["dg_gated"] += dg
            c["dg_prevented"] += dg and col
            c["dg_lost"] += dg and cor
        out[source] = dict(c)
        tot.update(c)
    out["__total__"] = dict(tot)
    return out


# ------------------------------------------------------------------ evaluation
def gate_row(name, summary, ci=None):
    n = summary["n"]
    return {
        "policy": name,
        "n": n,
        "gate_count": summary.get("gate_count"),
        "gate_rate": summary.get("gate_rate"),
        "acc_delta_pp": summary["acc_delta"] * 100.0 if summary.get("acc_delta") is not None else None,
        "prevented": summary["collapses_prevented"],
        "lost": summary["corrections_lost"],
        "net": summary["net"],
        "ci95_acc_delta_pp": ci,
        "validation": summary.get("validation"),
    }


def ci_pp(boot):
    if not boot:
        return None
    a = boot["acc_delta"]
    return [a["ci_2.5"] * 100.0, a["ci_97.5"] * 100.0]


def dg_gate(rec):
    """DisagreementGate: freeze the initial majority unless the agents initially agree (feature init_unanimous)."""
    return not bool(rec.features.get("init_unanimous"))


def dg_alt_gate(rec):
    """Alternative reading: a missing (None) initial answer also counts as disagreement."""
    return float(rec.features.get("init_agree_frac") or 0.0) < 1.0


def evaluate_variant(records, with_stump=True):
    """Compact policy table, no bootstrap."""
    out: dict[str, Any] = {
        "n": len(records),
        "collapses": sum(r.collapse for r in records),
        "corrections": sum(r.correction for r in records),
    }
    specs = [
        ("always_freeze", lambda r: True),
        ("r1_majority_changed", lambda r: bool(r.features.get("r1_majority_changed"))),
        ("any_r1_flip", lambda r: float(r.features.get("r1_n_flipped") or 0.0) > 0.0),
        ("disagreement_gate", dg_gate),
    ]
    for name, fn in specs:
        s, _ = cpr.evaluate_gate(records, fn)
        out[name] = gate_row(name, s)
    if with_stump:
        s, _, _ = cpr.evaluate_lomo_stump(records, lambda r: r.source, cpr.R1_FEATURES, "r1_only")
        out["learned_r1_stump"] = gate_row("learned_r1_stump", s)
    return out


def per_source_table(records, folds):
    table = {}
    fold_by_model = {f["heldout_model"]: f["test_summary"] for f in folds}
    for source in sorted({r.source for r in records}):
        sub = [r for r in records if r.source == source]
        row: dict[str, Any] = {
            "usable": len(sub),
            "collapses": sum(r.collapse for r in sub),
            "corrections": sum(r.correction for r in sub),
        }
        for name, fn in [
            ("always_freeze", lambda r: True),
            ("r1_majority_changed", lambda r: bool(r.features.get("r1_majority_changed"))),
            ("disagreement_gate", dg_gate),
        ]:
            s, _ = cpr.evaluate_gate(sub, fn)
            row[name] = {"gated": s["gate_count"], "prevented": s["collapses_prevented"], "lost": s["corrections_lost"]}
        fs = fold_by_model.get(source)
        if fs:
            row["learned_r1_stump_heldout"] = {
                "gated": fs["gate_count"], "prevented": fs["collapses_prevented"], "lost": fs["corrections_lost"],
            }
        table[source] = row
    return table


def subtotal(per_source, sources):
    keys = ["usable", "collapses", "corrections"]
    tot: dict[str, Any] = {k: sum(per_source[s][k] for s in sources if s in per_source) for k in keys}
    for pol in ("always_freeze", "r1_majority_changed", "disagreement_gate", "learned_r1_stump_heldout"):
        tot[pol] = {
            k: sum(per_source[s][pol][k] for s in sources if s in per_source and pol in per_source[s])
            for k in ("gated", "prevented", "lost")
        }
    return tot


# ------------------------------------------------------------------ git / existing-artifact comparisons
def git_show(rev_path: str) -> bytes | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "show", rev_path], stderr=subprocess.DEVNULL)
    except Exception:
        return None


def compare_to_committed_json(primary: dict[str, Any]) -> dict[str, Any]:
    blob = git_show(f"{COMMITTED_JSON_COMMIT}:abc_exp/results/correction_preserving_policy_replay.json")
    if blob is None:
        return {"available": False}
    old = json.loads(blob)["cohorts"]["eligible_all_primary"]
    new_by_name = {p["policy"]: p for p in primary["policies"]}
    max_diff = 0.0
    mism: list[str] = []
    fields = ["n", "gate_count", "collapses_prevented", "corrections_lost", "net", "standard_collapses", "standard_corrections"]
    for op in old["policies"]:
        np_ = new_by_name.get(op["policy"])
        if np_ is None:
            mism.append(f"missing policy {op['policy']}")
            continue
        for k in fields + ["acc_delta", "gate_rate"]:
            a, b = op.get(k), np_.get(k)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                d = abs(a - b)
                max_diff = max(max_diff, d)
                if d > 1e-12:
                    mism.append(f"{op['policy']}.{k}: committed {a} vs recomputed {b}")
            elif a != b:
                mism.append(f"{op['policy']}.{k}: committed {a} vs recomputed {b}")
        oc, nc = op.get("question_bootstrap_ci95"), np_.get("question_bootstrap_ci95")
        if oc and nc:
            for q in ("ci_2.5", "ci_97.5", "median"):
                d = abs(oc["acc_delta"][q] - nc["acc_delta"][q])
                max_diff = max(max_diff, d)
                if d > 1e-12:
                    mism.append(f"{op['policy']}.ci.{q}: {oc['acc_delta'][q]} vs {nc['acc_delta'][q]}")
    return {
        "available": True,
        "committed_in": f"{COMMITTED_JSON_COMMIT}:abc_exp/results/correction_preserving_policy_replay.json",
        "committed_header": {"n_records": old["n_records"], "n_models": old["n_models"],
                             "standard_collapses": old["standard_collapses"],
                             "standard_corrections": old["standard_corrections"]},
        "all_policy_fields_equal": not mism and old["n_records"] == primary["n_records"],
        "max_abs_diff": max_diff,
        "mismatches": mism,
        "note": "probe metadata (probe_matched_rows, fill_value of two probe-variant folds) differs because sa_causal_* "
                "files grew after 80106ed; no policy number is affected.",
    }


def describe_existing_json(results_dir: Path) -> dict[str, Any]:
    p = results_dir / "correction_preserving_policy_replay.json"
    if not p.exists():
        return {"available": False}
    d = json.loads(p.read_text())
    c = d["cohorts"]["eligible_all_primary"]
    pol = {x["policy"]: x for x in c["policies"]}
    return {
        "available": True,
        "path": str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p),
        "n_records": c["n_records"],
        "n_models": c["n_models"],
        "models": c["models"],
        "standard_collapses": c["standard_collapses"],
        "standard_corrections": c["standard_corrections"],
        "always_freeze": [pol["naive_always_freeze"][k] for k in ("collapses_prevented", "corrections_lost", "net")],
        "r1_majority_changed": [pol["naive_r1_majority_changed"][k] for k in ("collapses_prevented", "corrections_lost", "net")],
        "r1_majority_changed_delta_pp": pol["naive_r1_majority_changed"]["acc_delta"] * 100.0,
        "learned_r1_stump_delta_pp": pol["learned_r1_percentile_stump"]["acc_delta"] * 100.0,
    }


# ------------------------------------------------------------------ printed-value checks
def printed_checks(n, std, r1, always, stump, dg, boot_stump, lad_r1, lad_st, units):
    """Each entry: (location, quantity, printed, reproduced, match)."""
    rows: list[dict[str, Any]] = []

    def add(loc, what, printed, got):
        rows.append({"where": loc, "quantity": what, "printed": printed, "reproduced": got, "match": printed == got})

    add("paper", "usable debates in cohort", "1255", str(n))
    add("paper", "number of sources in cohort", "7", str(units))
    add("paper", "always-freeze: collapses prevented", "129", str(always["prevented"]))
    add("paper", "always-freeze: corrections lost", "97", str(always["lost"]))
    add("paper", "always-freeze: delta acc (pp)", "2.55", f2(always["acc_delta_pp"]))
    add("paper", "R1-majority-changed: share of debates gated (%)", "23.2", f"{r1['gate_rate'] * 100:.1f}")
    add("paper", "R1-majority-changed: prevented", "74", str(r1["prevented"]))
    add("paper", "R1-majority-changed: lost", "56", str(r1["lost"]))
    add("paper", "R1-majority-changed: net", "18", str(r1["net"]))
    add("paper", "R1-majority-changed: delta acc (pp)", "1.43", f2(r1["acc_delta_pp"]))
    add("paper", "R1-majority-changed: break-even ratio r*", "0.76", f2(lad_r1["breakeven_r"]))
    add("paper", "learned R1 stump (strict LOMO): delta acc (pp)", "1.35", f2(stump["acc_delta_pp"]))
    add("paper", "learned R1 stump: prevented", "59", str(stump["prevented"]))
    add("paper", "learned R1 stump: lost", "42", str(stump["lost"]))
    add("paper", "learned R1 stump: net", "17", str(stump["net"]))
    add("paper", "learned R1 stump: break-even ratio r*", "0.71", f2(lad_st["breakeven_r"]))
    add("paper", "learned R1 stump: bootstrap CI95 lower (pp)", "-0.32", f2(boot_stump[0]))
    add("paper", "learned R1 stump: bootstrap CI95 upper (pp)", "3.04", f2(boot_stump[1]))
    for w, p1, p2 in [("0.5", "-1.51", "-1.00"), ("1", "+1.43", "+1.35"), ("2", "+7.33", "+6.06"), ("4", "+19.12", "+15.46"), ("8", "+42.71", "+34.26")]:
        v1 = lad_r1["delta_pp"][w]
        v2 = lad_st["delta_pp"][w]
        g1 = f"{v1:+.2f}" if v1 > 0 else f2(v1)
        add("paper", f"R1-majority-changed ladder, r={w} (pp)", p1, g1)
        g2 = f"{v2:+.2f}" if v2 > 0 else f2(v2)
        add("paper", f"learned R1 stump ladder, r={w} (pp)", p2, g2)
    add("paper", "DisagreementGate: prevented", "84", str(dg["prevented"]))
    add("paper", "DisagreementGate: lost", "82", str(dg["lost"]))
    add("paper", "DisagreementGate: net", "2", str(dg["net"]))
    add("paper", "DisagreementGate: delta acc (pp)", "0.16", f2(dg["acc_delta_pp"]))
    return rows


# ------------------------------------------------------------------ markdown
def pp(x):
    return "NA" if x is None else f"{x:+.2f}"


def write_markdown(res: dict[str, Any], path: Path) -> None:
    L: list[str] = []
    ok_all = res["all_printed_values_reproduced"]
    L += [
        "# Reproduction of the camera-ready \"1,255 debates\" correction-preserving replay",
        "",
        f"Generated by `abc_exp/experiments/replay_1255_reproduction.py` (zero API, zero GPU). "
        f"Verdict: **{'all printed values reproduced' if ok_all else 'NOT all printed values reproduced (see table)'}**.",
        "",
        "## Which traces and which convention",
        "",
        "Seven sources (R0 + 4 logged rounds; the paper reads Round 3): DeepSeek-V4-Flash, the first 78 rows of the "
        "DeepSeek-V4-Pro partial file, Gemma-4-31B-it, and the four Qwen3.5/3.6 rows. 1,255 usable debates = 999 local vLLM "
        "(Gemma 200 + Qwen rows 799) + 180 DeepSeek-V4-Flash + 76 DeepSeek-V4-Pro. Unusable rows (20 + 2 + 1) have all-None "
        "`initial_answers`.",
        "",
        "Convention: standard endpoint = `majority_answer` recoded to Round 3 with alphabetic ties; initial majority = alphabetic-tie "
        "majority of `initial_answers`; R1 majority = alphabetic-tie majority of the Round-1 agent answers; collapse = R0 majority correct and "
        "endpoint wrong, correction = the reverse; a fired gate returns the R0 majority. DisagreementGate = freeze unless the "
        "three initial answers agree (`init_unanimous == 0`). Weight ladder = (r * prevented - lost) / N, break-even = lost / prevented.",
        "",
        "Why the shipped `correction_preserving_policy_replay.json` does not show this: it globs every `debate_traces_*.jsonl` and now "
        f"holds a {res['existing_json_now'].get('n_models')}-source, {res['existing_json_now'].get('n_records')}-debate cohort; the 1,255 cohort exists "
        f"only in git ({COMMITTED_JSON_COMMIT}) and in this script's output.",
        "",
        "## Source files",
        "",
        "| Stem (`debate_traces_<stem>.jsonl`) | Row cap | Rows read | Usable | sha256 of consumed bytes (first 12) | Matches printed vintage | Qwen3.5/3.6 row |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for s in res["sources"]:
        L.append(f"| {s['stem']} | {s['row_cap'] or 'none'} | {s['rows_read']} (file has {s['rows_in_file']}) | {s['usable']} | "
                 f"`{s['sha256'][:12]}` | {s['hash_matches_expected']}{' (overridden)' if s['overridden'] else ''} | {'yes' if s['is_qwen_row'] else ''} |")
    L += ["", "## Printed values versus reproduced", "",
          "| Where in the camera-ready TeX | Quantity | Printed | Reproduced | Match |", "|---|---|---:|---:|---|"]
    for c in res["printed_checks"]:
        L.append(f"| {c['where']} | {c['quantity']} | {c['printed']} | {c['reproduced']} | {'yes' if c['match'] else '**NO**'} |")
    p = res["primary_policies"]
    L += ["", "## Cohort policy table (N = %d; %d collapses, %d corrections)" % (res["n_records"], res["standard_collapses"], res["standard_corrections"]),
          "", "| Policy | Validation | Gate | Delta acc (pp) | CI95 (pp) | Prevented | Lost | Net |", "|---|---|---:|---:|---|---:|---:|---:|"]
    for r in p:
        ci = r["ci95_acc_delta_pp"]
        L.append(f"| {r['policy']} | {r['validation']} | {r['gate_rate'] * 100:.1f}% | {pp(r['acc_delta_pp'])} | "
                 f"{'NA' if not ci else '[%s, %s]' % (pp(ci[0]), pp(ci[1]))} | {r['prevented']} | {r['lost']} | {r['net']} |")
    L += ["", "## Weight ladder (accuracy delta in pp)", "", "| Control | r=0.5 | r=1 | r=2 | r=4 | r=8 | Break-even r |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, lad in res["weight_ladder"].items():
        L.append(f"| {name} | " + " | ".join(pp(lad['delta_pp'][f'{w:g}']) for w in WEIGHTS) + f" | {lad['breakeven_r']:.2f} |")
    L += ["", "## Independent raw-JSON cross-check", "",
          "Counts recomputed from the JSONL alone (no replay-module code): "
          f"usable {res['independent_check']['total']['usable']}, collapses {res['independent_check']['total']['collapses']}, "
          f"corrections {res['independent_check']['total']['corrections']}, R1-majority-changed prevented/lost "
          f"{res['independent_check']['total']['r1chg_prevented']}/{res['independent_check']['total']['r1chg_lost']}, "
          f"DisagreementGate prevented/lost {res['independent_check']['total']['dg_prevented']}/{res['independent_check']['total']['dg_lost']}. "
          f"Agrees with the replay module: **{res['independent_check']['agrees_with_module']}**.",
          "", "## Equality with the committed JSON from the initial snapshot", ""]
    cm = res["comparison_to_committed_json"]
    if cm.get("available"):
        L.append(f"Recomputed policy table vs `{cm['committed_in']}`: all policy fields equal = **{cm['all_policy_fields_equal']}** "
                 f"(max abs diff {cm['max_abs_diff']:.2e}). {cm['note']}")
    else:
        L.append("git history not available; comparison skipped.")
    L += ["", "## Per-source contribution (what the four Qwen3.5/3.6 re-runs will move)", "",
          "| Source | Usable | Collapses | Corrections | R1-maj-changed gated / prev / lost | DisagreementGate prev / lost | Stump (held-out fold) prev / lost |",
          "|---|---:|---:|---:|---|---|---|"]
    ps = res["per_source"]
    for s in list(ps["rows"]):
        r = ps["rows"][s]
        st = r.get("learned_r1_stump_heldout", {})
        L.append(f"| {s}{' (Qwen row)' if s in QWEN_ROWS else ''} | {r['usable']} | {r['collapses']} | {r['corrections']} | "
                 f"{r['r1_majority_changed']['gated']} / {r['r1_majority_changed']['prevented']} / {r['r1_majority_changed']['lost']} | "
                 f"{r['disagreement_gate']['prevented']} / {r['disagreement_gate']['lost']} | {st.get('prevented', '-')} / {st.get('lost', '-')} |")
    for label in ("four_qwen_rows", "other_three_sources"):
        r = ps[label]
        st = r["learned_r1_stump_heldout"]
        L.append(f"| **{label}** | {r['usable']} | {r['collapses']} | {r['corrections']} | "
                 f"{r['r1_majority_changed']['gated']} / {r['r1_majority_changed']['prevented']} / {r['r1_majority_changed']['lost']} | "
                 f"{r['disagreement_gate']['prevented']} / {r['disagreement_gate']['lost']} | {st['prevented']} / {st['lost']} |")
    L += ["", "The learned stump is trained leave-one-model-out, so replacing the Qwen rows also changes the rule learned for every other fold, not only the Qwen folds.",
          "", "## Sensitivity (labelled sensitivity; the printed numbers are the first row)", "",
          "| Variant | N | Collapses | Corrections | R1-maj prev / lost / net / delta pp | Always-freeze prev / lost | DG prev / lost / net | Stump prev / lost / net / delta pp |",
          "|---|---:|---:|---:|---|---|---|---|"]
    for name, v in res["sensitivity"].items():
        r1, af, dg = v["r1_majority_changed"], v["always_freeze"], v["disagreement_gate"]
        st = v.get("learned_r1_stump")
        stxt = "NA" if not st else f"{st['prevented']} / {st['lost']} / {st['net']} / {pp(st['acc_delta_pp'])}"
        L.append(f"| {name} | {v['n']} | {v['collapses']} | {v['corrections']} | {r1['prevented']} / {r1['lost']} / {r1['net']} / {pp(r1['acc_delta_pp'])} | "
                 f"{af['prevented']} / {af['lost']} | {dg['prevented']} / {dg['lost']} / {dg['net']} | {stxt} |")
    ex = res["existing_json_now"]
    if ex.get("available"):
        L += ["", "Shipped `correction_preserving_policy_replay.json` today (read only): "
              f"{ex['n_models']} sources, {ex['n_records']} debates, {ex['standard_collapses']} collapses, {ex['standard_corrections']} corrections; "
              f"always-freeze prev/lost/net {ex['always_freeze']}; R1-majority-changed prev/lost/net {ex['r1_majority_changed']} "
              f"({ex['r1_majority_changed_delta_pp']:+.2f} pp); learned stump {ex['learned_r1_stump_delta_pp']:+.2f} pp."]
    L += ["", f"Replay module: `{res['replay_module']['path']}` sha256 `{res['replay_module']['sha256'][:12]}`; seed {res['seed']}, bootstrap B = {res['bootstrap_B']}.", ""]
    path.write_text("\n".join(L))


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--tag", default="20261030", help="output suffix; files are replay_1255_reproduction_<tag>.{json,md}")
    ap.add_argument("--override", action="append", default=[], metavar="STEM=PATH",
                    help="use PATH instead of debate_traces_<STEM>.jsonl (e.g. for re-run Qwen rows)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any hash or printed value differs")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    cpr.RES = results_dir  # probe files are read from here
    overrides = dict(o.split("=", 1) for o in args.override)

    probe_means, probe_norms = cpr.load_probe_means()
    row_sets, src_meta = [], []
    hash_ok = True
    for stem, cap, expected in SOURCES:
        path = Path(overrides[stem]) if stem in overrides else results_dir / f"debate_traces_{stem}.jsonl"
        rows, digest, total = read_rows(path, cap)
        row_sets.append((stem, rows))
        matches = digest == expected
        hash_ok &= matches or stem in overrides
        src_meta.append({"stem": stem, "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
                         "row_cap": cap, "rows_read": len(rows), "rows_in_file": total, "sha256": digest,
                         "expected_sha256": expected, "hash_matches_expected": matches, "overridden": stem in overrides,
                         "is_qwen_row": stem in QWEN_ROWS})
        if not matches:
            print(f"WARNING: {stem}: consumed-bytes sha256 {digest[:12]} != printed-vintage {expected[:12]}"
                  f"{' (overridden)' if stem in overrides else ''}", file=sys.stderr)

    records = build_records(row_sets, probe_means, probe_norms)
    for m in src_meta:
        m["usable"] = sum(1 for r in records if r.source == m["stem"])

    # primary cohort: identical code path to the shipped script
    cohort = cpr.cohort_summary("replay_1255", records)
    pol = {p["policy"]: p for p in cohort["policies"]}
    dg_summary, dg_rows = cpr.evaluate_gate(records, dg_gate)
    dg_summary["validation"] = "fixed_rule_no_training"
    dg_boot = cpr.bootstrap_eval(dg_rows)
    dg_alt_summary, _ = cpr.evaluate_gate(records, dg_alt_gate)

    def row_for(name, label=None):
        p = pol[name]
        return gate_row(label or name, p, ci_pp(p.get("question_bootstrap_ci95")))

    primary_policies = [
        row_for("naive_always_freeze"), row_for("naive_r1_majority_changed"), row_for("naive_any_r1_flip"),
        row_for("learned_r1_percentile_stump"), row_for("learned_r1_probe_percentile_stump"),
        gate_row("disagreement_gate", dg_summary, ci_pp(dg_boot)),
    ]
    byname = {r["policy"]: r for r in primary_policies}
    n = len(records)
    lad = {
        "Round 1 majority changed": ladder(byname["naive_r1_majority_changed"]["prevented"], byname["naive_r1_majority_changed"]["lost"], n),
        "Learned Round 1 stump": ladder(byname["learned_r1_percentile_stump"]["prevented"], byname["learned_r1_percentile_stump"]["lost"], n),
        "Always freeze": ladder(byname["naive_always_freeze"]["prevented"], byname["naive_always_freeze"]["lost"], n),
        "DisagreementGate": ladder(dg_summary["collapses_prevented"], dg_summary["corrections_lost"], n),
    }

    checks = printed_checks(
        n, None, byname["naive_r1_majority_changed"], byname["naive_always_freeze"], byname["learned_r1_percentile_stump"],
        byname["disagreement_gate"], byname["learned_r1_percentile_stump"]["ci95_acc_delta_pp"],
        lad["Round 1 majority changed"], lad["Learned Round 1 stump"], len({r.source for r in records}),
    )

    ind = independent_counts(row_sets)
    t = ind["__total__"]
    agrees = (
        t["usable"] == n
        and t["collapses"] == cohort["standard_collapses"]
        and t["corrections"] == cohort["standard_corrections"]
        and t["r1chg_prevented"] == byname["naive_r1_majority_changed"]["prevented"]
        and t["r1chg_lost"] == byname["naive_r1_majority_changed"]["lost"]
        and t["dg_prevented"] == dg_summary["collapses_prevented"]
        and t["dg_lost"] == dg_summary["corrections_lost"]
    )

    folds = cohort["learned_folds"]["learned_r1_percentile_stump"]
    per_source = per_source_table(records, folds)
    qsrc = [s for s in QWEN_ROWS if s in per_source]
    others = [s for s in per_source if s not in QWEN_ROWS]

    sens: dict[str, Any] = {}
    if not args.skip_sensitivity:
        sens["PRINTED: 7 sources, DS-V4-Pro first 78 rows, stored majority recoded to Round 3 alphabetic, alphabetic R0 ties"] = evaluate_variant(records)
        # endpoint / tie-rule grid (labelled sensitivity)
        for endpoint in ("stored", "R4_alpha", "R3_alpha"):
            for init_tie in ("alpha", "first_seen"):
                if endpoint == "stored" and init_tie == "alpha":
                    continue
                recs = build_records(row_sets, probe_means, probe_norms, endpoint=endpoint, init_tie=init_tie)
                sens[f"endpoint={endpoint}, R0 ties={init_tie}"] = evaluate_variant(recs)
        # cohort variants
        full_pro, _, _ = read_rows(results_dir / f"debate_traces_{DS_PRO}.jsonl", None)
        rs_full = [(s, full_pro if s == DS_PRO else rows) for s, rows in row_sets]
        sens["DS-V4-Pro all rows in today's file (81 rows) instead of first 78"] = evaluate_variant(
            build_records(rs_full, probe_means, probe_norms))
        sens["drop DS-V4-Pro (6 sources)"] = evaluate_variant([r for r in records if r.source != DS_PRO])
        sens["local vLLM only (Gemma + 4 Qwen rows, 5 sources)"] = evaluate_variant(
            [r for r in records if r.source.startswith("vllm_")])
        sens["four Qwen3.5/3.6 rows only (4 sources, strict LOMO stump)"] = evaluate_variant(
            [r for r in records if r.source in QWEN_ROWS])
        sens["everything except the four Qwen rows (3 sources; LOMO stump needs >= 4 sources, not reported)"] = evaluate_variant(
            [r for r in records if r.source not in QWEN_ROWS], with_stump=False)

    result = {
        "analysis": "replay_1255_reproduction",
        "date": "2026-10-30",
        "all_printed_values_reproduced": all(c["match"] for c in checks),
        "all_source_hashes_match_printed_vintage": hash_ok,
        "convention": {
            "standard_endpoint": "majority_answer recoded to Round 3, alphabetic ties, by r3_unified/recode_r3.py",
            "initial_majority": "alphabetic-tie majority of initial_answers (R0)",
            "r1_majority": "alphabetic-tie majority of round_traces[0] agent answers (list of per-agent dicts; no stored majority)",
            "disagreement_gate": "freeze to R0 majority unless init_unanimous (all non-empty initial answers equal)",
            "weight_ladder": "(r * prevented - lost) / N, break-even = lost / prevented",
        },
        "seed": cpr.SEED,
        "bootstrap_B": cpr.BOOTSTRAPS,
        "replay_module": {"path": str(REPLAY_PATH.relative_to(REPO_ROOT)), "sha256": sha256_file(REPLAY_PATH)},
        "sources": src_meta,
        "n_records": n,
        "standard_collapses": cohort["standard_collapses"],
        "standard_corrections": cohort["standard_corrections"],
        "primary_policies": primary_policies,
        "weight_ladder": lad,
        "disagreement_gate_alt_definition_missing_answer_counts_as_disagreement": gate_row("dg_alt", dg_alt_summary),
        "printed_checks": checks,
        "independent_check": {"per_source": {k: v for k, v in ind.items() if k != "__total__"}, "total": t, "agrees_with_module": agrees},
        "comparison_to_committed_json": compare_to_committed_json(cohort),
        "existing_json_now": describe_existing_json(results_dir),
        "learned_stump_folds": [
            {"heldout": f["heldout_model"], "kind": f["rule"]["kind"], "feature": f["rule"]["feature"],
             "direction": f["rule"]["direction"], "threshold": f["rule"]["threshold"],
             "train_gated": f["rule"]["gated"], "train_prevented": f["rule"]["prevented"], "train_lost": f["rule"]["lost"],
             "test": {k: f["test_summary"][k] for k in ("n", "gate_count", "collapses_prevented", "corrections_lost", "net")}}
            for f in folds
        ],
        "per_source": {
            "rows": per_source,
            "four_qwen_rows": subtotal(per_source, qsrc),
            "other_three_sources": subtotal(per_source, others),
            "qwen_rows": QWEN_ROWS,
        },
        "sensitivity": sens,
    }

    out_json = results_dir / f"{OUT_PREFIX}_{args.tag}.json"
    out_md = results_dir / f"{OUT_PREFIX}_{args.tag}.md"
    for p in (out_json, out_md):
        if not p.name.startswith(OUT_PREFIX):
            raise SystemExit(f"refusing to write {p}")
    out_json.write_text(json.dumps(result, indent=2, default=float) + "\n")
    write_markdown(result, out_md)

    bad = [c for c in checks if not c["match"]]
    print(f"Wrote {out_json}\nWrote {out_md}")
    print(f"N={n}  printed checks: {len(checks) - len(bad)}/{len(checks)} reproduced; "
          f"source hashes match printed vintage: {hash_ok}; independent cross-check agrees: {agrees}")
    for c in bad:
        print(f"  MISMATCH {c['where']}: {c['quantity']} printed {c['printed']} vs {c['reproduced']}")
    return 1 if args.strict and (bad or not hash_ok or not agrees) else 0


if __name__ == "__main__":
    raise SystemExit(main())
