"""Figure for large_persistence_ref.py: is the 1,000-pair ranking bracket a property of the model
or of the reference checkpoint we call "final"?

Headless Agg; CVD-safe (green-free categorical palette, every series also coded by
linestyle/marker).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

STYLE = [(CVD[0], "-", "o"), (CVD[1], "--", "s"), (CVD[2], "-.", "D")]


def x_of(s):
    return max(s, 0.5)


def main():
    D = json.load(open(os.path.join(RESULTS, "large_persistence_ref.json")))
    refs = sorted(int(r) for r in D["references"])

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))

    # --- A: pi_ref(s) trajectories, one series per reference -----------------------------------
    a = ax[0]
    for (c, ls, mk), ref in zip(STYLE, refs):
        E = D["references"][str(ref)]
        ss = [s for s in E["steps"] if s != ref]           # omit the self-scoring point
        a.plot([x_of(s) for s in ss], [E["rows"][str(s)]["pi"] for s in ss],
               color=c, ls=ls, marker=mk, ms=6, lw=1.7, label=f"reference = step {ref}")
    a.axhline(0, color="0.35", lw=0.9, ls="--")
    a.axvspan(x_of(32), x_of(64), color="0.5", alpha=0.16, hatch="..", ec="0.4", lw=0)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(0.3, 3e5)
    a.set_xlabel("training step")
    a.set_ylabel(r"rank agreement $\pi_{\mathrm{L,ref}}(s)$")
    a.legend(fontsize=7.6, loc="upper left")
    a.set_title("A. 1,000-pair ranking scored against\nthree mature references", fontsize=9.5)

    # --- B: the two bracket checkpoints, per reference -------------------------------------------
    b = ax[1]
    y = np.arange(len(refs))
    for k, (s, mk, ls, off, lab) in enumerate(
            ((32, "o", "-", +0.13, r"$\Delta\pi$ at step 32"),
             (64, "s", "--", -0.13, r"$\Delta\pi$ at step 64"))):
        v = np.array([D["references"][str(r)]["rows"][str(s)]["dpi"] for r in refs])
        h = np.array([D["references"][str(r)]["sim_halfwidth_dpi"] for r in refs])
        b.errorbar(v, y + off, xerr=h, color=CVD[k], ls="none", marker=mk, ms=7,
                   elinewidth=1.6, capsize=3, label=lab)
    b.axvline(0, color="0.35", lw=1.0, ls="--")
    b.set_yticks(y)
    b.set_yticklabels([f"step {r}" for r in refs])
    b.set_ylim(-0.6, len(refs) - 0.4)
    b.set_xlabel(r"$\Delta\pi$ with simultaneous 95% band")
    b.set_ylabel("reference checkpoint")
    b.legend(fontsize=7.6, loc="lower right")
    b.set_title("B. Zero excluded at step 64, covered at\nstep 32, under every reference",
                fontsize=9.5)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "large_persistence_ref.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/large_persistence_ref.png")


if __name__ == "__main__":
    main()
