"""Figure: three readouts on the same six feature sets, four targets, one panel each.

Each panel is one target. The x axis walks the six feature sets in depth order, from the static
embedding row to the residual stream after block 18. Each line is one readout -- linear ridge, RBF
kernel ridge, k-nearest-neighbour -- fitted on the same 50 train/test splits, with the shaded strip
the widest of the three 50-permutation nulls and the dash-dot line the ceiling that target's own
measurement noise allows.

Writes plots/readout.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7"]
STYLE = [("ridge", "-", "o"), ("krr", "--", "s"), ("knn", ":", "^")]
LABEL = {"ridge": "linear ridge", "krr": "RBF kernel ridge", "knn": "$k$-NN"}
KEYS = ["shape", "width", "width_given_shape", "shape_given_width"]
TITLE = {"shape": "shape  $E_u$", "width": "width  $w_u$",
         "width_given_shape": "width, shape removed  (the open question)",
         "shape_given_width": "shape, width removed"}
XLAB = ["$W_E[u]$", "$m_u$", "block 0", "block 6", "block 12", "block 18"]


def panel(ax, d, key):
    x = np.arange(len(d["sites"]))
    lo = [min(d[s][r][key]["null_mean"] - 2 * d[s][r][key]["null_sd"] for r, _, _ in STYLE)
          for s in d["sites"]]
    hi = [max(d[s][r][key]["null_mean"] + 2 * d[s][r][key]["null_sd"] for r, _, _ in STYLE)
          for s in d["sites"]]
    ax.fill_between(x, lo, hi, color="0.85", edgecolor="0.45", hatch="//", lw=0.8, zorder=1,
                    label="permutation null, $\\pm$2 s.d.")
    for (r, ls, mk), c in zip(STYLE, CVD):
        m = [d[s][r][key]["rho_mean"] for s in d["sites"]]
        e = [d[s][r][key]["rho_sd"] for s in d["sites"]]
        ax.errorbar(x, m, yerr=e, ls=ls, marker=mk, ms=6, lw=1.8, capsize=3, color=c, zorder=3,
                    label=LABEL[r])
    ceil = d[d["sites"][0]]["ridge"][key]["ceiling"]
    ax.axhline(ceil, color="0.2", ls="-.", lw=1.2, zorder=2)
    ax.text(len(x) - 1.05, ceil, f" ceiling {ceil:.2f}", ha="right", va="bottom", fontsize=7.5,
            color="0.2")
    ax.axhline(0, color="0.3", lw=0.8, ls=":")
    ax.set_xticks(x, XLAB, fontsize=8)
    ax.set_title(TITLE[key], fontsize=10)
    ax.set_ylabel("held-out Spearman $\\rho$")
    ax.grid(axis="y", alpha=0.25)


def main():
    d = json.load(open(f"{RESULTS}/readout.json"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key in zip(axes.ravel(), KEYS):
        panel(ax, d, key)
    for ax in axes[1]:
        ax.set_xlabel("feature set (static embedding row, then the residual stream by depth)")
    axes[0, 0].legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.suptitle("Pythia-1.4B, 123 endpoint tokens, 50 shared 80/43 splits: three readouts, "
                 "same features and targets", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{PLOTS}/readout.png", dpi=170)
    plt.close(fig)
    print(f"wrote {PLOTS}/readout.png")


if __name__ == "__main__":
    main()
