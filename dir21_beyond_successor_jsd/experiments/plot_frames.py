"""Figure for the frame-shape control."""
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

TAGS = ["mid-sentence", "question", "list", "code"]
MARK = ["o", "s", "^", "D"]


def main():
    d = json.load(open(f"{RESULTS}/frames.json"))
    names = d["tokens"]
    w0 = np.array(d["w_orig"])

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))

    rho = [d["by_frame"][t]["rho_vs_orig"][0] for t in TAGS]
    x = np.arange(len(TAGS))
    for k, (xi, v) in enumerate(zip(x, rho)):
        ax[0].bar(xi, v, .62, color=CVD[k], hatch=["//", "\\\\", "xx", ".."][k],
                  edgecolor="white")
        ax[0].annotate(f"{v:+.2f}", (xi, v), fontsize=7.5, ha="center",
                       xytext=(0, 3), textcoords="offset points")
    ax[0].axhline(d["within_orig_mean"], ls="--", color="0.3", lw=1.2,
                  label=f"among the three original frames ({d['within_orig_mean']:+.2f})")
    ax[0].axhline(0.46, ls=":", color="0.45", lw=1.1,
                  label="between two disjoint anchor sets (+0.46)")
    ax[0].legend(fontsize=6.5, loc="upper center", ncol=1, framealpha=.9)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(TAGS, fontsize=8)
    ax[0].set_ylim(0, 1.18)
    ax[0].set_xlabel("context the anchor width is measured in")
    ax[0].set_ylabel(r"Spearman $\rho$ with the original ranking (123 tokens)")
    ax[0].set_title("Does the token ranking survive a change of context?", fontsize=9)

    for k, t in enumerate(TAGS):
        y = np.array([d["by_frame"][t]["w"][s] for s in names])
        ax[1].scatter(w0, y, s=13, marker=MARK[k], color=CVD[k], alpha=.7, edgecolor="none",
                      label=f"{t} (median {np.median(y):.2f})")
    lo, hi = 0.30, 0.86
    ax[1].plot([lo, hi], [lo, hi], ls="--", color="0.5", lw=.9)
    ax[1].set_xlim(lo, hi)
    ax[1].set_ylim(lo, hi)
    ax[1].set_xlabel(r"$\hat w_u$ in the three original frames (median {:.2f})".format(
        d["orig_median"]))
    ax[1].set_ylabel(r"$\hat w_u$ in the new context")
    ax[1].set_title("The ordering carries over; the level does not", fontsize=9)
    ax[1].legend(fontsize=7, loc="upper left")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(PLOTS, "frame_control.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/frame_control.png")


if __name__ == "__main__":
    main()
