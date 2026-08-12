"""Figure for the GPT-2 embedding probe refitted against plateau-filtered widths.

Left: the 50-draw shuffled-target null for the all-curve probe, with the observed accuracy and with
the single shuffled draw that the earlier run used as its control.
Right: test accuracy of each probe against its own noise ceiling.

Reads results/gpt2_probe.json and results/embed.json; writes plots/gpt2_probe.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CVD, PLOTS, RESULTS

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)

d = json.load(open(f"{RESULTS}/gpt2_probe.json"))
ref = json.load(open(f"{RESULTS}/embed.json"))["probe_w_block0"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 4.2))

# left: the null distribution the earlier control sampled once
a = d["probe_all_curves"]
ax.hist(a["null_draws"], bins=14, color=CVD[0], edgecolor="black", hatch="//",
        label="shuffled targets (50 draws)")
ax.axvline(a["rho_mean"], color=CVD[1], ls="--", lw=2.2, label="probe, all curves")
ax.axvline(d["stored_single_shuffle"]["reproduced"], color=CVD[2], ls="-.", lw=2.2,
           label="single shuffled draw used earlier")
ax.set_xlabel(r"mean held-out Spearman $\rho$ over 50 train/test splits")
ax.set_ylabel("number of shuffled-target draws")
ax.set_title(f"GPT-2: probe vs its null (permutation $p$ = {a['perm_p']:.2f})")
ax.legend(fontsize=8, loc="upper left")

# right: each probe against the ceiling its target allows
rows = [("GPT-2, all curves\n(reliability 0.32)", d["probe_all_curves"],
         d["ceiling"]["all_curves"], "o"),
        ("GPT-2, plateau curves\n(reliability 0.66)", d["probe_plateau_curves"],
         d["ceiling"]["plateau_curves"], "s"),
        ("GPT-2, 2 corpus statistics\n-> plateau curves", d["probe_corpus_stats"],
         d["ceiling"]["plateau_curves"], "^"),
        ("Pythia-1.4B, strict width\n(reference)", ref, None, "D")]
for k, (lab, r, ceil, mk) in enumerate(rows):
    y = len(rows) - 1 - k
    c = CVD[k % len(CVD)]
    bx.errorbar(r["rho_mean"], y, xerr=r["rho_sd"], fmt=mk, color=c, ms=8, capsize=4, lw=2)
    if "null_sd" in r:
        bx.barh(y, 2 * r["null_sd"], left=r["null_mean"] - r["null_sd"], height=0.44,
                color="0.75", hatch="..", edgecolor="0.35", zorder=0)
    if ceil is not None:
        bx.plot(ceil, y, marker="|", ms=18, mew=2.5, color="0.15")
        bx.annotate("ceiling", (ceil, y), textcoords="offset points", xytext=(3, 9), fontsize=8)
bx.axvline(0, color="0.4", lw=0.8)
bx.set_yticks(range(len(rows)))
bx.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
bx.set_xlim(-0.25, 1.0)
bx.set_ylim(-0.85, 3.5)
bx.set_xlabel(r"mean held-out Spearman $\rho$ ($\pm$ 1 sd over 50 splits)")
bx.set_title("probe accuracy against the ceiling its target allows")
bx.annotate("gray band: shuffled-target null (mean $\\pm$ 1 sd, 50 draws)",
            (0.02, 0.02), xycoords="axes fraction", fontsize=8)

fig.tight_layout()
fig.savefig(f"{PLOTS}/gpt2_probe.png", dpi=150)
plt.close(fig)
print(f"wrote {PLOTS}/gpt2_probe.png")
