"""Figure for the width-definition robustness check. Headless Agg; CVD-safe, green-free palette,
every series also coded by linestyle and marker so the figure survives grayscale printing."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
LS = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]
MK = ["o", "s", "^", "D", "v"]


def main():
    R = json.load(open(os.path.join(RESULTS, "threshold_robustness.json")))
    s = np.array(R["steps"], float)
    levels = R["levels"]
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))

    a = ax[0]
    for k, lv in enumerate(levels):
        r = R["by_level"][f"{lv:.2f}"]
        a.plot(s, r["rho"], ls=LS[k], marker=MK[k], ms=3.5, color=CVD[k],
               label=f"{int(lv * 100)}%/{int(100 - lv * 100)}%")
    a.axhline(0, color="0.35", ls="--", lw=1)
    a.axvspan(8, 32, facecolor="0.5", alpha=0.16, hatch="\\\\", edgecolor="0.4", lw=0)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(-0.3, s.max() * 1.6)
    a.set_xlabel("training step (symlog; 0 shown at left)")
    a.set_ylabel(r"Spearman $\rho$(corpus JSD, width $w_a$)")
    a.set_title("A. Ordering trajectory, five width definitions")
    a.legend(fontsize=7, title="levels", title_fontsize=7, loc="lower left", ncol=2)

    a = ax[1]
    for k, lv in enumerate(levels):
        r = R["by_level"][f"{lv:.2f}"]
        a.plot(s, np.array(r["median_w"]) / r["straight_line_ref"], ls=LS[k], marker=MK[k],
               ms=3.5, color=CVD[k], label=f"{int(lv * 100)}%/{int(100 - lv * 100)}%")
    a.axhline(1.0, color="0.35", ls="--", lw=1)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(-0.3, s.max() * 1.6)
    a.set_xlabel("training step (symlog; 0 shown at left)")
    a.set_ylabel("median $w_a$ / straight-line value $1-2a$")
    a.set_title("B. Sharpening, on a common scale")
    a.legend(fontsize=7, loc="lower left", ncol=2)

    a = ax[2]
    for k, lv in enumerate(levels):
        r = R["by_level"][f"{lv:.2f}"]
        y = len(levels) - 1 - k
        a.plot([r["ordering_after"], r["ordering_by"]], [y + 0.16] * 2, lw=6, solid_capstyle="butt",
               color=CVD[0], alpha=0.85)
        a.plot([r["shape_after"], r["shape_by"]], [y - 0.16] * 2, lw=6, solid_capstyle="butt",
               color=CVD[1], alpha=0.85)
        a.plot([r["ordering_after"], r["ordering_by"]], [y + 0.16] * 2, ls="none", marker="|",
               ms=12, mew=2, color=CVD[0])
        a.plot([r["shape_after"], r["shape_by"]], [y - 0.16] * 2, ls="none", marker="+",
               ms=9, mew=2, color=CVD[1])
        a.annotate("", xy=(r["shape_by"], y), xytext=(r["ordering_by"], y),
                   arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0, ls=":"))
        sep = r["shape_by"] / r["ordering_by"]
        a.text(np.sqrt(r["ordering_by"] * r["shape_by"]), y + 0.34, f"{sep:.0f}x apart",
               ha="center", fontsize=7.5, color="0.3")
    a.plot([], [], lw=6, color=CVD[0], label="divergence-ordering onset bracket")
    a.plot([], [], lw=6, color=CVD[1], label="plateau-shape onset bracket")
    a.set_yticks(range(len(levels)))
    a.set_yticklabels([f"{int(lv * 100)}%/{int(100 - lv * 100)}%" for lv in levels[::-1]])
    a.set_ylim(-1.35, len(levels) - 0.3)
    a.set_xscale("log")
    a.set_xlim(4, 4000)
    a.set_xlabel("training step (log)")
    a.set_ylabel("levels defining $w_a$")
    a.set_title("C. Both brackets, every definition")
    a.legend(fontsize=7, loc="lower center", ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "threshold_robustness.png"))
    plt.close(fig)
    print("wrote plots/threshold_robustness.png")


if __name__ == "__main__":
    main()
