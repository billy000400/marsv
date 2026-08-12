"""Figure: the same four probes read from three early sites inside Pythia-1.4B.

One panel per site -- the block-0 MLP output m_u, the post-block-0 residual state x_u, and the static
embedding row W_E[u] as the reference. Each row is one target: the dot is the mean held-out Spearman
correlation over the 50 shared train/test splits, the bar its spread, the shaded strip the
50-permutation null, and the caret the ceiling that target's own measurement noise allows.

Writes plots/early_shape.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
KEYS = ["shape", "width", "width_given_shape", "shape_given_width"]
LAB = {"shape": "shape  $E_u$", "width": "width  $w_u$",
       "width_given_shape": "width, shape removed", "shape_given_width": "shape, width removed"}
TAGS = ["mlp_out", "resid_block0", "embedding"]
TITLE = {"mlp_out": "block-0 MLP output  $m_u$",
         "resid_block0": "post-block-0 residual  $x_u$",
         "embedding": "static embedding row  $W_E[u]$"}


def panel(ax, d, tag):
    ys = np.arange(len(KEYS))[::-1]
    for y, k in zip(ys, KEYS):
        r = d[tag][k]
        lo, hi = r["null_mean"] - r["null_sd"], r["null_mean"] + r["null_sd"]
        ax.barh(y, hi - lo, left=lo, height=0.55, color="0.85", edgecolor="0.45",
                hatch="//", zorder=1, label="null $\\pm$ 1 s.d." if k == KEYS[0] else None)
        ax.errorbar(r["rho_mean"], y, xerr=r["rho_sd"], fmt="o", ms=7, capsize=4, lw=1.8,
                    color=CVD[0], zorder=3, label="probe" if k == KEYS[0] else None)
        ax.plot(r["ceiling"], y, marker="^", ms=9, color=CVD[1], ls="none", zorder=3,
                label="ceiling" if k == KEYS[0] else None)
        if r["perm_p"] > 0.05:
            ax.text(r["rho_mean"] + r["rho_sd"] + 0.03, y, f"p = {r['perm_p']:.2f}", ha="left",
                    va="center", fontsize=7.5, color="0.25")
    ax.axvline(0, color="0.3", lw=0.8, ls=":")
    ax.set_yticks(ys, [LAB[k] for k in KEYS])
    ax.set_xlim(-0.25, 1.02)
    ax.set_xlabel("held-out Spearman $\\rho$ (predicted vs measured target)")
    ax.set_title(TITLE[tag], fontsize=10)
    ax.grid(axis="x", alpha=0.25)


def main():
    d = json.load(open(f"{RESULTS}/early_shape.json"))
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.5), sharex=True, sharey=True)
    for ax, tag in zip(axes, TAGS):
        panel(ax, d, tag)
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.suptitle("Pythia-1.4B, 123 endpoint tokens; shape and width rank them at $\\rho$ = "
                 f"{d['rho_shape_width'][0]:+.3f}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{PLOTS}/early_shape.png", dpi=170)
    plt.close(fig)
    print(f"wrote {PLOTS}/early_shape.png")


if __name__ == "__main__":
    main()
