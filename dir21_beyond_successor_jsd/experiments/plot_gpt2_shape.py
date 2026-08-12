"""Figure: four probes per model -- shape, width, and each with the other removed.

Left GPT-2, right Pythia-1.4B. Each row is one target: the dot is the mean held-out Spearman
correlation over the 50 shared train/test splits, the bar its spread, the shaded strip the
50-permutation null, and the caret the ceiling that target's own measurement noise allows (omitted
where the split-half reliability is negative, so no ceiling is defined).

Writes plots/gpt2_shape.png.
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
TITLE = {"gpt2": "GPT-2 (124M), block 0", "pythia_1.4b": "Pythia-1.4B, block 0"}


def panel(ax, d, tag):
    ys = np.arange(len(KEYS))[::-1]
    for y, k in zip(ys, KEYS):
        r = d[tag][k]
        lo, hi = r["null_mean"] - r["null_sd"], r["null_mean"] + r["null_sd"]
        ax.barh(y, hi - lo, left=lo, height=0.55, color="0.85", edgecolor="0.45",
                hatch="//", zorder=1, label="null $\\pm$ 1 s.d." if k == KEYS[0] else None)
        ax.errorbar(r["rho_mean"], y, xerr=r["rho_sd"], fmt="o", ms=7, capsize=4, lw=1.8,
                    color=CVD[0], zorder=3, label="probe" if k == KEYS[0] else None)
        if r["ceiling"] > 0:
            ax.plot(r["ceiling"], y, marker="^", ms=9, color=CVD[1], ls="none", zorder=3,
                    label="ceiling" if k == KEYS[0] else None)
        else:
            ax.text(0.96, y + 0.28, "no ceiling defined", ha="right", va="bottom", fontsize=7.5,
                    color="0.35", transform=ax.get_yaxis_transform())
    ax.axvline(0, color="0.3", lw=0.8, ls=":")
    ax.set_yticks(ys, [LAB[k] for k in KEYS])
    ax.set_xlim(-0.25, 1.02)
    ax.set_xlabel("held-out Spearman $\\rho$ (predicted vs measured target)")
    ax.set_title(f"{TITLE[tag]}\nshape and width rank the tokens at "
                 f"$\\rho$ = {d[tag]['rho_shape_width'][0]:+.3f}", fontsize=10)
    ax.grid(axis="x", alpha=0.25)


def main():
    d = json.load(open(f"{RESULTS}/gpt2_shape.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6), sharex=True)
    for ax, tag in zip(axes, ["gpt2", "pythia_1.4b"]):
        panel(ax, d, tag)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/gpt2_shape.png", dpi=170)
    plt.close(fig)
    print(f"wrote {PLOTS}/gpt2_shape.png")


if __name__ == "__main__":
    main()
