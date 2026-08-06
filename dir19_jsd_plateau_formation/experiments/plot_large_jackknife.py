"""Figure for large_jackknife.py: do the two 1,000-pair clocks survive one carrier sentence?

Headless Agg; CVD-safe (green-free categorical palette, every series also coded by
linestyle/marker).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
LS = ["-", "--", ":", "-."]
MK = ["o", "s", "^", "D"]
KEYS = ["ctx0", "ctx1", "ctx2", "median3"]
NAMES = ["sentence 1 only", "sentence 2 only", "sentence 3 only",
         "median of all three (primary)"]


def main():
    D = json.load(open(os.path.join(RESULTS, "large_jackknife.json")))
    steps = np.array(D["steps"], dtype=float)
    x = np.maximum(steps, 0.5)                      # step 0 plotted at 0.5 on a log axis
    C = D["contexts"]

    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.3))

    # --- A: graded ordering on the 600 middle-divergence pairs ---------------------------------
    a = ax[0]
    a.axvspan(64, 128, alpha=0.22, hatch="\\\\", facecolor="0.4", edgecolor="0.4", lw=0)
    for i, k in enumerate(KEYS):
        a.plot(x, C[k]["rho"], ls=LS[i], marker=MK[i], ms=4, lw=1.8 if i == 3 else 1.1,
               color=CVD[i], label=NAMES[i])
    s95 = C["median3"]["sim95_rho"]
    a.axhspan(-s95, s95, color="0.75", alpha=0.35, hatch="..", edgecolor="0.5", lw=0)
    a.axhline(0, color="0.35", lw=0.8)
    a.set_xscale("log")
    a.set_xlabel("training step (log scale; step 0 drawn at 0.5)")
    a.set_ylabel(r"$\rho$(corpus JSD, width), 600 middle pairs")
    a.set_title("A. Graded ordering: bracket step 64 $\\rightarrow$ 128 in every sentence")
    a.legend(fontsize=7.5, loc="lower left")

    # --- B: ranking lock-in, dpi against this sentence's own final widths ----------------------
    b = ax[1]
    b.axvspan(32, 64, alpha=0.22, hatch="//", facecolor="0.4", edgecolor="0.4", lw=0)
    for i, k in enumerate(KEYS):
        b.plot(x, C[k]["dpi"], ls=LS[i], marker=MK[i], ms=4, lw=1.8 if i == 3 else 1.1,
               color=CVD[i], label=NAMES[i])
    h = C["median3"]["sim_halfwidth_dpi"]
    b.axhspan(-h, h, color="0.75", alpha=0.35, hatch="..", edgecolor="0.5", lw=0)
    b.axhline(0, color="0.35", lw=0.8)
    b.set_xscale("log")
    b.set_xlabel("training step (log scale; step 0 drawn at 0.5)")
    b.set_ylabel(r"$\Delta\pi(s)=\pi(s)-\pi(0)$")
    b.set_title("B. Ranking lock-in: bracket step 32 $\\rightarrow$ 64 in every sentence")
    b.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("1,000-pair bank, one carrier sentence at a time (no median over contexts): "
                 "both onset brackets are unchanged", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(PLOTS, "large_jackknife.png")
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
