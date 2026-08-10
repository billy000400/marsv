"""Figure for the transfer test: per-token quantities measured outside the pair bank."""
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
    tokfx = json.load(open(f"{RESULTS}/explore1.json"))["token_effects"]
    a = np.array(tokfx["effect"])
    names = tokfx["tokens"]
    AW = json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"]
    BP = json.load(open(f"{RESULTS}/basin.json"))["tokens"]
    tr = json.load(open(f"{RESULTS}/transfer.json"))
    cv0 = json.load(open(f"{RESULTS}/explore1.json"))["cv_r2"]

    aw = np.array([np.nanmedian(AW[s]["w"]) for s in names])
    ra = np.array([np.nanmedian(BP[s]["radius_anchor"]["0.2"]) for s in names])
    rr = np.array([np.nanmedian(BP[s]["radius_random"]["0.1"]) for s in names])

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.9),
                           gridspec_kw=dict(width_ratios=[1, 1, 1.25]))

    ax[0].scatter(aw, a, s=18, marker="o", c=CVD[0], alpha=.75, linewidths=0)
    o = np.argsort(a)
    for k in list(range(0, 6, 2)) + list(range(len(a) - 6, len(a), 2)):
        i = o[k]
        ax[0].annotate(names[i].strip(), (aw[i], a[i]), fontsize=7, color="0.25",
                       xytext=(5, 3), textcoords="offset points")
    r = spearmanr(aw, a)
    ax[0].set_xlabel(r"anchor width $\hat w_u$ (median vs 6 anchors)")
    ax[0].set_ylabel(r"fitted token effect $a_u$ (from the 1,000-pair bank)")
    ax[0].set_title(f"Width contribution transfers to unseen partners\n"
                    f"Spearman $\\rho$ = {r[0]:+.2f}, $p$ = {r[1]:.0e}", fontsize=9)

    ax[1].scatter(ra, a, s=18, marker="s", c=CVD[1], alpha=.75, linewidths=0,
                  label=f"toward anchor tokens ($\\rho$={spearmanr(ra, a)[0]:+.2f})")
    ax[1].scatter(rr, a, s=18, marker="^", c=CVD[3], alpha=.75, linewidths=0,
                  label=f"random directions ($\\rho$={spearmanr(rr, a)[0]:+.2f})")
    ax[1].set_xlabel("basin radius [radians of great-circle travel]")
    ax[1].set_ylabel(r"fitted token effect $a_u$")
    ax[1].set_title("Absolute-movement radius explains less,\nand only along real-token directions",
                    fontsize=9)
    ax[1].legend(fontsize=7, loc="lower right")

    lab = ["$J$ only", r"measured, 2 params", "fitted, 123 params",
           r"measured + $J$", "fitted + $J$"]
    val = [cv0["corpus_jsd"], tr["cv_r2"]["anchor_sum"], cv0["token_additive"],
           tr["cv_r2"]["anchor_sum_plus_jsd"], cv0["token_additive_plus_jsd"]]
    cols = [CVD[1], CVD[0], CVD[3], CVD[0], CVD[3]]
    hat = ["//", None, "..", None, ".."]
    ax[2].barh(range(len(val)), val, color=cols, hatch=hat, edgecolor="white")
    for i, x in enumerate(val):
        ax[2].text(x + .008, i, f"{x:.3f}", va="center", fontsize=8)
    ax[2].set_yticks(range(len(val)), lab, fontsize=7.5)
    ax[2].set_xlim(0, 0.75)
    ax[2].set_xlabel("held-out $R^2$ for pair width $w$")
    ax[2].set_title("Two measured numbers match 123 fitted ones", fontsize=9)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(os.path.join(PLOTS, "transfer.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/transfer.png")


if __name__ == "__main__":
    main()
