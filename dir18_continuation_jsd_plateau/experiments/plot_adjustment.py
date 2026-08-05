"""Forest plot of the 60-pair association before and after accounting for other pair properties.

Reads results/revisions.json (written by revisions.py); writes plots/adjustment.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import CVD, MARK, RESULTS, PLOTS

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130

KEYS = ["total_w", "w_ctrl_5cov", "w_ctrl_outjsd", "w_ctrl_outjsd_plus5"]
NAMES = ["unadjusted\n(overall association)",
         "after accounting for the five\nmeasured pair properties",
         "after accounting for the\nmodel-output JSD",
         "after accounting for both"]

out = json.load(open(os.path.join(RESULTS, "revisions.json")))
fig, ax = plt.subplots(figsize=(7.0, 3.8))
ypos = np.arange(len(KEYS))[::-1]
for i, k in enumerate(KEYS):
    r, (lo, hi), p = out[k]["rho"], out[k]["ci"], out[k]["p"]
    sig = p < 0.05
    ax.errorbar(r, ypos[i], xerr=[[r - lo], [hi - r]], fmt=MARK[i],
                color=CVD[0] if sig else CVD[1], capsize=4, ms=8,
                markerfacecolor=CVD[0] if sig else "white", lw=1.5)
    ax.annotate(f"$\\rho$={r:+.3f}, p={p:.3g}", (r, ypos[i]), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=8)
ax.axvline(0, color="0.4", lw=1, ls=":")
ax.set_yticks(ypos)
ax.set_yticklabels(NAMES, fontsize=8.5)
ax.set_ylim(-0.6, len(KEYS) - 0.4)
ax.set_xlabel(r"Spearman $\rho$ between corpus next-token JSD and $w$ (95% bootstrap CI)")
ax.set_title("Filled marker: p < 0.05.  Open marker: p > 0.05.  n = 60 pairs.", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "adjustment.png"))
plt.close(fig)
print("wrote plots/adjustment.png")
