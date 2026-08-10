"""Figure for the anchor-set swap."""
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


def main():
    sw = json.load(open(f"{RESULTS}/swap.json"))
    tokfx = json.load(open(f"{RESULTS}/explore1.json"))["token_effects"]
    a = np.array(tokfx["effect"])
    names = tokfx["tokens"]
    F = np.array([sw["anchor_width_function"][s] for s in names])
    C = np.array([sw["anchor_width_rare_content"][s] for s in names])

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.9),
                           gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax[0].scatter(F, C, s=18, marker="o", c=CVD[0], alpha=.75, linewidths=0)
    o = np.argsort(a)
    for k in list(range(0, 5, 2)) + list(range(len(a) - 5, len(a), 2)):
        i = o[k]
        ax[0].annotate(names[i].strip(), (F[i], C[i]), fontsize=7, color="0.25",
                       xytext=(5, 3), textcoords="offset points")
    r = spearmanr(F, C)
    ax[0].set_xlabel(r"anchor width vs 6 function words  $\hat w_u^{\,F}$")
    ax[0].set_ylabel(r"anchor width vs 6 rare content words  $\hat w_u^{\,C}$")
    ax[0].set_title("Two disjoint anchor sets agree only in part\n"
                    f"Spearman $\\rho$ = {r[0]:+.2f}, $p$ = {r[1]:.0e}", fontsize=9)

    lab = ["mixed anchors\n(original 6)", "function-word\nanchors", "rare-content\nanchors"]
    orig = np.array([np.nanmedian(json.load(open(f"{RESULTS}/anchor_width.json"))
                                  ["tokens"][s]["w"]) for s in names])
    val = [spearmanr(orig, a)[0], spearmanr(F, a)[0], spearmanr(C, a)[0]]
    ax[1].bar(range(3), val, color=[CVD[0], CVD[3], CVD[1]],
              hatch=[None, "..", "//"], edgecolor="white")
    for i, x in enumerate(val):
        ax[1].text(i, x + .015, f"{x:+.2f}", ha="center", fontsize=8)
    ax[1].set_xticks(range(3), lab, fontsize=8)
    ax[1].set_ylim(0, 0.85)
    ax[1].set_ylabel(r"Spearman $\rho$ with fitted token effect $a_u$")
    ax[1].set_title("...yet each recovers the same token trait", fontsize=9)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(PLOTS, "anchor_swap.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/anchor_swap.png")


if __name__ == "__main__":
    main()
