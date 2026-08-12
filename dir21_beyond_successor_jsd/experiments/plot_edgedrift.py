"""Figure for the edge-drift test (results/edgedrift.json, edgedrift_summary.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS, CVD

R = json.load(open(f"{RESULTS}/edgedrift.json"))["rows"]
S = json.load(open(f"{RESULTS}/edgedrift_summary.json"))
LINE = S["line_reference"]

# label, color, linestyle
SERIES = [("GPT-2 block 0", "gpt2_block0", CVD[0], "-"),
          ("GPT-2 block 4", "gpt2_block4", CVD[0], "--"),
          ("GPT-2 block 8", "gpt2_block8", CVD[0], ":"),
          ("Pythia-160M", "160m_block0", CVD[1], "-"),
          ("Pythia-410M", "410m_block0", CVD[2], "--"),
          ("Pythia-1.4B", "1.4b_block0", CVD[4], "-.")]

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

for label, key, c, ls in SERIES:
    e = np.sort(np.array(R[key]["edge_curves"], float))
    ax[0].plot(e, np.arange(1, len(e) + 1) / len(e), ls, color=c, lw=1.8, label=label)
ax[0].axvline(LINE, color="0.45", lw=1.1, ls=(0, (6, 3)))
ax[0].text(LINE * 1.03, 0.10, "straight line\n$E = 0.2$", fontsize=8.5, color="0.35")
ax[0].axvline(0.5 * LINE, color="0.75", lw=1.0)
ax[0].set_xscale("log")
ax[0].set_xlim(0.006, 1.0)
ax[0].set_xlabel("edge drift $E$ of one curve (log scale; 0 = flat plateau, 0.2 = straight line)")
ax[0].set_ylabel("fraction of curves with drift $\\leq E$")
ax[0].set_title("How plateau-shaped are the curves? (2,214 per model)")
ax[0].legend(fontsize=8.5, loc="upper left")

f = S["filter"]["gpt2_block0"]
groups = [("all 2,214 curves", "all_curves", "//"), ("plateau curves only\n($E \\leq 0.1$)", "plateau_curves", "..")]
bars = [("GPT-2 split-half\nreliability", "reliability", CVD[0]),
        ("noise ceiling for\nagreement", "ceiling", CVD[3]),
        ("agreement with\nPythia-1.4B", "rho_vs_1_4b", CVD[1])]
x = np.arange(len(bars))
for j, (glabel, gkey, hatch) in enumerate(groups):
    v = [f[gkey][k] for _, k, _ in bars]
    ax[1].bar(x + (j - 0.5) * 0.36, v, 0.34, color=[c for _, _, c in bars], hatch=hatch,
              edgecolor="white", label=glabel)
    for xi, vi in zip(x + (j - 0.5) * 0.36, v):
        ax[1].text(xi, vi + (0.025 if vi >= 0 else -0.055), f"{vi:+.2f}", ha="center", fontsize=9)
ax[1].axhline(0, color="0.3", lw=0.9)
ax[1].set_xticks(x)
ax[1].set_xticklabels([b[0] for b in bars], fontsize=9)
ax[1].set_ylim(-0.35, 1.02)
ax[1].set_ylabel("Spearman $\\rho$")
ax[1].set_title("GPT-2: discarding the non-plateau curves doubles reliability,\n"
                "and the disagreement with Pythia survives it")
ax[1].legend(fontsize=8.5, loc="upper right")

fig.tight_layout()
fig.savefig(f"{PLOTS}/edgedrift.png", dpi=150)
plt.close(fig)
print(f"wrote {PLOTS}/edgedrift.png")
