#!/usr/bin/env python3
"""
Figure 2 of the paper. Data are read from the panel JSON written by abc_exp/experiments/channel_panel.py
(22 models: 12 API, 10 local open-weight).
Stars mark the unadjusted paired p of AA - AS, as before.
usage: fig2_channel_panel.py <g1_panel.json> <out-stem>
"""
import json, sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colors for conditions
C_DEFAULT = "#999999"      # gray
C_ANTISOC = "#0072B2"      # blue — anti-social (social channel suppressed)
C_ANTIARG = "#D55E00"      # vermillion — anti-argument (argument channel suppressed)

BANDS = {"API models": ("#e3f2fd", "#1565c0", "API models"), "Local open-weight models (vLLM)": ("#f3e5f5", "#6a1b9a", "Local open-weight (vLLM)")}
P = json.load(open(sys.argv[1]))["per_model"]
models = list(P)
fr_d = [P[m]["FR_D"] for m in models]; fr_as = [P[m]["FR_AS"] for m in models]; fr_aa = [P[m]["FR_AA"] for m in models]
p_vals = [P[m]["p"] for m in models]; groups = [P[m]["group"] for m in models]
n = len(models); x = np.arange(n); w = 0.27
fig, ax = plt.subplots(figsize=(5.5, 3.0))
start = 0
for g in dict.fromkeys(groups):
    k = groups.count(g); bg, fg, label = BANDS[g]
    ax.axvspan(start - 0.5, start + k - 0.5, alpha=0.15, color=bg, zorder=0)
    ax.text(start + (k - 1) / 2, 0.90, label, ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=fg, fontstyle="italic")
    if start: ax.axvline(start - 0.5, color="#bbbbbb", linewidth=0.7, linestyle="--", zorder=1)
    start += k
ax.bar(x - w, fr_as, w, color=C_ANTISOC, edgecolor="black", linewidth=0.4, label="Anti-social", zorder=3)
ax.bar(x, fr_d, w, color=C_DEFAULT, edgecolor="black", linewidth=0.4, label="Default", zorder=3)
ax.bar(x + w, fr_aa, w, color=C_ANTIARG, edgecolor="black", linewidth=0.4, label="Anti-argument", zorder=3)
for i in range(n):
    y_top = max(fr_as[i], fr_d[i], fr_aa[i]) + 0.015
    mark = "***" if p_vals[i] < 0.001 else "**" if p_vals[i] < 0.01 else "*" if p_vals[i] < 0.05 else None
    if mark: ax.text(x[i], y_top, mark, ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#333333")
    else: ax.text(x[i], y_top, "n.s.", ha="center", va="bottom", fontsize=6.5, color="#999999", fontstyle="italic")
ax.set_xticks(x)
ax.set_xticklabels([m.replace(" (no reasoning)", "") for m in models], fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.5, n - 0.5); ax.tick_params(axis="x", length=2)
ax.set_ylabel("Flip rate", fontsize=8); ax.set_ylim(0, 0.97)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1)); ax.yaxis.grid(True, linewidth=0.3, alpha=0.4, color="#cccccc", zorder=0); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False, fontsize=7.5, ncol=3, handlelength=1.2, columnspacing=1.2)
fig.subplots_adjust(left=0.08, right=0.995, bottom=0.26, top=0.89)
out = sys.argv[2]
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.04); fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.04, dpi=300)
print(f"Saved {out}.pdf and .png")
