"""Figure for the fixed-displacement test: does the landing width track the model's output movement?"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import CVD, PLOTS, RESULTS

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9

STYLE = [("quiet", "o", "quietest combination"),
         ("loud", "s", "loudest combination"),
         ("random", "^", "plain random direction")]


def main():
    d = json.load(open(f"{RESULTS}/quiet.json"))
    rows = d["tokens"]
    base = np.array([rows[s]["base_w"] for s in rows])

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))

    for k, (tag, mk, lab) in enumerate(STYLE):
        b = [rows[s]["dirs"][tag]["bits"] for s in rows]
        dw = [rows[s]["dirs"][tag]["dw"] for s in rows]
        ax[0].scatter(b, dw, s=30, marker=mk, color=CVD[k], alpha=.8, edgecolor="none", label=lab)
    ax[0].axhline(0, color="0.6", lw=.8)
    r = d["rho_bits_dw"]
    ax[0].set_xlabel("output movement the edit actually produced (bits)")
    ax[0].set_ylabel(r"$\Delta \hat w_u$, measured")
    ax[0].set_title("Displacement norm held fixed per token:\n"
                    fr"width change is flat in output movement ($\rho = {r[0]:+.2f}$, p = {r[1]:.2f})",
                    fontsize=9)
    ax[0].legend(fontsize=7, loc="upper right")

    for k, (tag, mk, lab) in enumerate(STYLE):
        w = [rows[s]["dirs"][tag]["w"] for s in rows]
        ax[1].scatter(base, w, s=30, marker=mk, color=CVD[k], alpha=.8, edgecolor="none", label=lab)
    pooled = np.array([[rows[s]["dirs"][t]["w"] for t, _, _ in STYLE] for s in rows]).mean(1)
    rp = spearmanr(base, pooled)
    ax[1].plot(base, np.poly1d(np.polyfit(base, pooled, 1))(base), ls="-", color="0.35", lw=1.3)
    lo, hi = base.min() - .03, max(base.max(), pooled.max()) + .05
    ax[1].plot([lo, hi], [lo, hi], ls=":", color="0.45", lw=1.1)
    ax[1].annotate("no change", (hi * .99, hi * .99), fontsize=6.5, ha="right", va="top",
                   color="0.35", rotation=34)
    ax[1].set_xlim(lo, hi)
    ax[1].set_ylim(lo, hi)
    ax[1].set_xlabel(r"$\hat w_u$ before the edit")
    ax[1].set_ylabel(r"$\hat w_u$ after the edit")
    ax[1].set_title("Spread collapses from 0.083 to 0.02-0.04,\n"
                    fr"but the order partly survives ($\rho = {rp[0]:+.2f}$, p = {rp[1]:.2f})",
                    fontsize=9)
    ax[1].legend(fontsize=7, loc="upper left")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(PLOTS, "quiet.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/quiet.png")


if __name__ == "__main__":
    main()
