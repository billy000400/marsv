"""Figure for sentence_jackknife.py: do the three onsets survive dropping to one carrier sentence?

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
LS = ["-", "--", ":", "-."]
MK = ["o", "s", "^", "D"]
KEYS = ["ctx0", "ctx1", "ctx2", "median3"]
NAMES = ["sentence 1 only", "sentence 2 only", "sentence 3 only",
         "median of all three (primary)"]


def main():
    D = json.load(open(os.path.join(RESULTS, "sentence_jackknife.json")))
    steps = np.array(D["steps"], dtype=float)
    C = D["contexts"]

    fig, ax = plt.subplots(2, 2, figsize=(11.6, 8.0))

    # --- A: divergence ordering, one trajectory per carrier sentence --------------------------
    a = ax[0, 0]
    for i, k in enumerate(KEYS):
        a.plot(steps, C[k]["rho"], ls=LS[i], marker=MK[i], ms=4, lw=1.6 if i == 3 else 1.2,
               color=CVD[i], label=NAMES[i])
    a.axvspan(8, 32, alpha=0.22, hatch="\\\\", facecolor="0.4", edgecolor="0.4", lw=0)
    a.axhline(0, color="0.35", lw=0.8)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(-0.3, steps.max() * 1.6)
    a.set_xlabel("training step $s$ (symlog; 0 shown at left)")
    a.set_ylabel(r"Spearman $\rho(J,\,w)$")
    a.set_title("A. Divergence ordering: same step 8$\\to$32 bracket in all three sentences")
    a.legend(loc="lower left", fontsize=7)

    # --- B: global shape ----------------------------------------------------------------------
    a = ax[0, 1]
    for i, k in enumerate(KEYS):
        a.plot(steps, C[k]["median_w"], ls=LS[i], marker=MK[i], ms=4, lw=1.6 if i == 3 else 1.2,
               color=CVD[i], label=NAMES[i])
    a.axhline(0.8, color="0.35", lw=1.0, ls="--")
    a.annotate("straight line, $w=0.8$ (no plateau)", xy=(2, 0.804), fontsize=7, color="0.3")
    a.axvspan(1000, 2000, alpha=0.22, hatch="..", facecolor="0.4", edgecolor="0.4", lw=0)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(-0.3, steps.max() * 1.6)
    a.set_xlabel("training step $s$ (symlog; 0 shown at left)")
    a.set_ylabel(r"median transition width $w$")
    a.set_title("B. Plateau shape: same step 1000$\\to$2000 bracket in all three")
    a.legend(loc="lower left", fontsize=7)

    # --- C: ranking persistence ---------------------------------------------------------------
    a = ax[1, 0]
    env = np.array(C["median3"]["null_pt95"])
    a.fill_between(steps, -env, env, color="0.55", alpha=0.20, hatch="//", edgecolor="0.45", lw=0)
    a.plot(steps, env, ls=":", color="0.45", lw=1.0, label=r"pointwise null 95% $|\pi|$")
    a.plot(steps, -env, ls=":", color="0.45", lw=1.0)
    for i, k in enumerate(KEYS):
        pi = np.array(C[k]["pi"], dtype=float)
        pi[-1] = np.nan                        # the final checkpoint scores against itself
        a.plot(steps, pi, ls=LS[i], marker=MK[i], ms=4, lw=1.6 if i == 3 else 1.2,
               color=CVD[i], label=NAMES[i])
    a.axvspan(64, 128, alpha=0.22, hatch="xx", facecolor="0.4", edgecolor="0.4", lw=0)
    a.axhline(0, color="0.35", lw=0.8)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(-0.3, steps.max() * 1.6)
    a.set_xlabel("training step $s$ (symlog; 0 shown at left)")
    a.set_ylabel(r"rank agreement $\pi(s)$ with that series' own final ranking")
    a.set_title("C. Ranking: sentence 1 alone closes one checkpoint later")
    a.legend(loc="upper left", fontsize=7)

    # --- D: the three brackets per carrier sentence --------------------------------------------
    a = ax[1, 1]
    fields = [("bracket_ordering", "divergence ordering", 0),
              ("bracket_ranking", "ranking becomes final", 1),
              ("bracket_shape", "plateau shape", 2)]
    off = [0.24, 0.0, -0.24]
    for f, (fld, lab, ci) in enumerate(fields):
        for i, k in enumerate(KEYS):
            b = C[k][fld]
            y = (len(KEYS) - 1 - i) + off[f]
            a.plot(b, [y, y], lw=5, solid_capstyle="butt", color=CVD[ci],
                   label=lab if i == 0 else None)
            a.plot(b[1], y, marker=MK[f], ms=6, color=CVD[ci], mec="white", mew=0.7)
            a.annotate(f"{b[0]}$\\to${b[1]}", xy=(b[1] * 1.35, y), fontsize=7,
                       va="center", color="0.25")
    a.set_yticks(range(len(KEYS)))
    a.set_yticklabels(NAMES[::-1], fontsize=8)
    a.set_xscale("log")
    a.set_xlim(4, 2.2e4)
    a.set_ylim(-1.35, len(KEYS) - 0.4)
    a.set_xlabel("training step (log): bracket opens at the left end, closes at the marker")
    a.set_title("D. Every bracket, every carrier sentence")
    a.legend(loc="lower right", fontsize=7)
    a.grid(axis="y", alpha=0)

    fig.tight_layout()
    p = os.path.join(PLOTS, "sentence_jackknife.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
