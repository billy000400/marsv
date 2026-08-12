"""Figure: what more training tokens buy each target, at every feature set.

One column per feature set (depth order), one row per target. Within a panel the x axis is the number
of training tokens (30, 50, 80, 100 of the 123), the two lines are the two readouts fitted on those
tokens, the shaded strip is the permutation null at that size and the dash-dot line is the ceiling the
target's own measurement noise allows.

Writes plots/curve.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00"]
STYLE = [("ridge", "-", "o"), ("krr", "--", "s")]
LABEL = {"ridge": "linear ridge", "krr": "RBF kernel ridge"}
ROWS = ["width_given_shape", "width", "shape"]
TITLE = {"width_given_shape": "width, shape removed\n(the open question)",
         "width": "width  $w_u$\n(control)", "shape": "shape  $E_u$\n(control)"}
XLAB = ["$W_E[u]$", "$m_u$", "block 0", "block 6", "block 12", "block 18"]


def panel(ax, d, site, key, sizes):
    x = np.arange(len(sizes))
    lo = [min(d[site][str(s)][r][key]["null_mean"] - 2 * d[site][str(s)][r][key]["null_sd"]
              for r, _, _ in STYLE) for s in sizes]
    hi = [max(d[site][str(s)][r][key]["null_mean"] + 2 * d[site][str(s)][r][key]["null_sd"]
              for r, _, _ in STYLE) for s in sizes]
    ax.fill_between(x, lo, hi, color="0.85", edgecolor="0.45", hatch="//", lw=0.8, zorder=1,
                    label="permutation null, $\\pm$2 s.d.")
    for (r, ls, mk), c in zip(STYLE, CVD):
        m = [d[site][str(s)][r][key]["rho_mean"] for s in sizes]
        e = [d[site][str(s)][r][key]["rho_sd"] for s in sizes]
        ax.errorbar(x, m, yerr=e, ls=ls, marker=mk, ms=5, lw=1.6, capsize=3, color=c, zorder=3,
                    label=LABEL[r])
    ceil = d[site][str(sizes[0])]["ridge"][key]["ceiling"]
    ax.axhline(ceil, color="0.2", ls="-.", lw=1.1, zorder=2)
    ax.axhline(0, color="0.3", lw=0.8, ls=":")
    ax.set_xticks(x, [str(s) for s in sizes], fontsize=8)
    ax.grid(axis="y", alpha=0.25)


def main():
    d = json.load(open(f"{RESULTS}/curve.json"))
    sizes, sites = d["train_sizes"], d["sites"]
    fig, axes = plt.subplots(len(ROWS), len(sites), figsize=(15, 8.2), sharey="row")
    for i, key in enumerate(ROWS):
        for j, site in enumerate(sites):
            panel(axes[i, j], d, site, key, sizes)
            if i == 0:
                axes[i, j].set_title(XLAB[j], fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(TITLE[key] + "\n\nheld-out Spearman $\\rho$", fontsize=9)
        ceil = d[sites[0]][str(sizes[0])]["ridge"][key]["ceiling"]
        axes[i, -1].text(len(sizes) - 1.05, ceil, f" ceiling {ceil:.2f} ", ha="right", va="bottom",
                         fontsize=7.5, color="0.2")
    for ax in axes[-1]:
        ax.set_xlabel("training tokens", fontsize=9)
    axes[0, 0].legend(loc="upper left", fontsize=7.5, framealpha=0.95)
    fig.suptitle("Pythia-1.4B, 123 endpoint tokens: held-out accuracy against training-set size, "
                 "50 splits and a 50-permutation null at every size", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{PLOTS}/curve.png", dpi=170)
    plt.close(fig)
    print(f"wrote {PLOTS}/curve.png")


if __name__ == "__main__":
    main()
