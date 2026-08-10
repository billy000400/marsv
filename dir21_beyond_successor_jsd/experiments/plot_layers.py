"""Figure for the layer sweep of the anchor width."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CVD, PLOTS, RESULTS

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9


def main():
    d = json.load(open(f"{RESULTS}/layers.json"))
    L = d["layers"]
    S = d["summary"]
    rho0 = [S[str(l)]["rho_vs_block0"][0] for l in L]
    rhoa = [S[str(l)]["rho_vs_token_effect"][0] for l in L]
    cv = [S[str(l)]["cv_r2"] for l in L]
    iqr = [S[str(l)]["iqr"] for l in L]

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax[0].plot(L, rho0, marker="o", ls="-", color=CVD[0], label="vs block-0 anchor width")
    ax[0].plot(L, rhoa, marker="s", ls="--", color=CVD[1], label=r"vs fitted token effect $a_u$")
    ax[0].axhline(0, color="0.6", lw=.8)
    for x, y in zip(L, rhoa):
        ax[0].annotate(f"{y:+.2f}", (x, y), fontsize=7, color="0.25",
                       xytext=(0, -12), textcoords="offset points", ha="center")
    ax[0].set_xticks(L)
    ax[0].set_xlabel("block at which the state is interpolated")
    ax[0].set_ylabel(r"Spearman $\rho$ across the 123 tokens")
    ax[0].set_title("Does the token ranking survive moving the site?", fontsize=9)
    ax[0].legend(fontsize=7, loc="lower left")

    med = [S[str(l)]["median"] for l in L]
    ax[1].plot(L, med, marker="o", ls="-", color=CVD[1], label=r"median $\hat w_u$")
    ax[1].plot(L, cv, marker="D", ls="--", color=CVD[0], label="held-out $R^2$ for pair $w$")
    ax[1].plot(L, iqr, marker="^", ls=":", color=CVD[3],
               label=r"IQR of $\hat w_u$ across tokens")
    ax[1].axhline(0.8, color="0.5", lw=.8, ls="-.")
    ax[1].text(0.2, 0.815, "$w = 0.8$: output moves in proportion to $t$", fontsize=7, color="0.35")
    ax[1].set_ylim(0, 0.92)
    ax[1].set_xticks(L)
    ax[1].set_xlabel("block at which the state is interpolated")
    ax[1].set_ylabel("value")
    ax[1].set_title("Transitions flatten and token spread collapses", fontsize=9)
    ax[1].legend(fontsize=7, loc="center left")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(PLOTS, "layer_sweep.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/layer_sweep.png")


if __name__ == "__main__":
    main()
