#!/usr/bin/env python3
"""TeX rows for tab:pilot_gated, tab:weight_sensitivity and tab:control_summary from probe_gated_lomo.py output.
Prints the e5_rule rows (the main analysis) and the comparison rows (always_freeze, raw_fr_gate, nested_lomo), plus
the bounds for unparsed probe replies when the flip / drop files are given.

usage: gate_tex.py <gate_lomo_paper.json> [<gate_lomo_flip.json> <gate_lomo_drop.json>]
"""
import json, sys

NAMES = {"llama-3.1-8b": "Llama-3.1-8B", "phi-4-mini": "Phi-4-mini", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B"}
G = json.load(open(sys.argv[1]))["variants"]
tot = G["always_freeze"]["per_model"]          # always-freeze prevented / lost = the model's collapses / corrections
s = lambda x: f"${x:+.2f}$"
pp = lambda x: f"{x:+.2f}"
# the gate JSON stores pp values rounded to 3 decimals; recompute from counts so the second decimal rounds correctly
dacc = lambda p: 100 * p["net"] / p["n"]          # accuracy change (counts debates outside the ledger)
led = lambda p: p["prevented"] - p["lost"]            # ledger net U(g) = prevented - lost
ladder = lambda p, r: 100 * (r * p["prevented"] - p["lost"]) / p["n"]


def model_rows(v):
    out = []
    for m, r in G[v]["per_model"].items():
        col, cor = tot[m]["prevented"], tot[m]["lost"]
        tau = "n/a" if r["tau"] is None else f"{r['tau']:.2f}"
        ci = f"[{pp(r['ci95'][0])},{pp(r['ci95'][1])}]"
        orc = s(r["oracle"]["dacc_pp"]) if r.get("oracle") else "n/a"
        out.append(f"{NAMES[m]} ({col} col / {cor} corr) & {tau} & {r['mode']} & {r['gate_pct']:.1f}\\% & {r['prevented']}/{col} & {r['lost']}/{cor} & ${led(r):+d}$ & {s(r['dacc_pp'])} {ci} & {orc} \\\\")
    return out


def pooled(v, label):
    p = G[v]["pooled"]
    ci = f"[{pp(p['ci95'][0])},{pp(p['ci95'][1])}]"
    return f"\\textbf{{{label}}} & & & & {p['prevented']}/251 & {p['lost']}/804 & ${led(p):+d}$ & {s(dacc(p))} {ci} & n/a \\\\"


print("% tab:pilot_gated: e5_rule per held-out model, then pooled rows of the main rule and the comparisons")
print("\n".join(model_rows("e5_rule")))
print("\\midrule")
for v, lab in [("e5_rule", "Pooled (frozen LOMO policy)"), ("nested_lomo", "Nested LOMO selection"), ("raw_fr_gate", "Raw mean-FR threshold"), ("always_freeze", "Always freeze")]:
    print(pooled(v, lab))
print("\n% tab:weight_sensitivity rows (r = 0.5, 1, 2, 4, 8, breakeven)")
for v, lab in [("e5_rule", "Probe-gated freeze"), ("always_freeze", "Always freeze"), ("raw_fr_gate", "Raw mean-FR threshold")]:
    p = G[v]["pooled"]
    be = "n/a" if p.get("break_even_r") is None else f"{p['break_even_r']:.2f}"
    print(f"{lab} & " + " & ".join(f"${ladder(p, r):+.2f}$" for r in (0.5, 1, 2, 4, 8)) + f" & {be} \\\\")
print("\n% tab:control_summary rows")
for v, lab, val in [("always_freeze", "Always freeze", "none"), ("e5_rule", "Probe-gated freeze", "LOMO")]:
    p = G[v]["pooled"]; be = "n/a" if p.get("break_even_r") is None else f"{p['break_even_r']:.2f}"
    print(f"{lab} & $6{{,}}525$ OSS & {val} & ${dacc(p):+.2f}$pp & ${p['prevented']}$ & ${p['lost']}$ & ${led(p):+d}$ & ${be}$ \\\\")
if len(sys.argv) == 4:
    print("\n% Unparsed-reply bounds: pooled prevented / lost / net per variant")
    files = {"paper": sys.argv[1], "flip": sys.argv[2], "drop": sys.argv[3]}
    for v in ("e5_rule", "nested_lomo", "raw_fr_gate", "always_freeze"):
        cells = []
        for k, f in files.items():
            p = json.load(open(f))["variants"][v]["pooled"]; cells.append(f"{k} {p['prevented']}/{p['lost']}/{led(p):+d} (acc {p['net']:+d})")
        print(f"% {v}: " + "; ".join(cells))
