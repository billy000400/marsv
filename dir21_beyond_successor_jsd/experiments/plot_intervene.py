"""Figure for the embedding-space intervention (a null result)."""
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
    d = json.load(open(f"{RESULTS}/intervene.json"))
    rows = d["tokens"]

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))

    for k, (tag, mk, lab) in enumerate([("probe", "o", "probe direction"),
                                        ("random", "s", "random direction, same step norm")]):
        t = np.array([x["target"] for s in rows for x in rows[s][tag]])
        dw = np.array([x["dw"] for s in rows for x in rows[s][tag]])
        ax[0].scatter(t + (k - .5) * 0.002, dw, s=20, marker=mk, color=CVD[k], alpha=.7,
                      edgecolor="none", label=lab)
    lim = max(abs(np.array(d["targets"]))) * 1.25
    ax[0].plot([-lim, lim], [-lim, lim], ls="--", color="0.5", lw=1.0)
    ax[0].annotate("what a causal direction would give", (lim * .98, lim * .98), fontsize=6.5,
                   ha="right", va="top", color="0.35", rotation=38)
    ax[0].axhline(0, color="0.6", lw=.8)
    ax[0].set_xlim(-lim, lim)
    ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel(r"width change the probe was asked for (width units)")
    ax[0].set_ylabel(r"measured change in $\hat w_u$")
    ax[0].set_title(f"16 tokens x 4 step sizes: measured response\n"
                    fr"mean $|\Delta \hat w_u|$ = {d['mean_abs_dw_probe']:.4f} "
                    f"(random {d['mean_abs_dw_random']:.4f})", fontsize=9)
    ax[0].legend(fontsize=7, loc="upper left")

    slopes = {tag: np.array([np.polyfit([x["target"] for x in rows[s][tag]],
                                        [x["dw"] for x in rows[s][tag]], 1)[0] for s in rows])
              for tag in ("probe", "random")}
    for k, tag in enumerate(("probe", "random")):
        ax[1].scatter(np.full(len(slopes[tag]), k + 1) + np.linspace(-.14, .14, len(slopes[tag])),
                      slopes[tag], s=22, marker=["o", "s"][k], color=CVD[k], alpha=.75,
                      edgecolor="none")
        ax[1].hlines(slopes[tag].mean(), k + .72, k + 1.28, color="0.15", lw=1.6)
    ax[1].axhline(0, color="0.6", lw=.8)
    ax[1].axhline(1.0, ls="--", color="0.4", lw=1.1)
    ax[1].annotate("1.0 = the probe's own prediction", (2.45, 1.0), fontsize=6.5, ha="right",
                   va="bottom", color="0.35")
    ax[1].set_xlim(.5, 2.5)
    ax[1].set_ylim(-.4, 1.15)
    ax[1].set_xticks([1, 2])
    ax[1].set_xticklabels(["probe direction", "random direction"], fontsize=8)
    ax[1].set_ylabel(r"per-token slope of measured $\Delta \hat w_u$ against the requested change")
    ax[1].set_title("A causal handle would sit at 1.0", fontsize=9)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(PLOTS, "intervene.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/intervene.png")


if __name__ == "__main__":
    main()
