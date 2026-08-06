"""Figure for persistence.py: when does the per-pair width ranking lock in?

Headless Agg; CVD-safe (green-free categorical palette, every series also coded by
linestyle/marker; sequential cividis for the matrix).
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


def main():
    P = json.load(open(os.path.join(RESULTS, "persistence.json")))
    steps = np.array(P["steps"], dtype=float)
    ref = len(steps) - 1
    k = np.arange(len(steps)) != ref          # the final checkpoint correlates with itself: drop it
    x = steps[k]

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2))

    # --- A: persistence trajectory against the label-permutation null -------------------------
    a = ax[0]
    env = np.array(P["pi_null_pt95"])[k]
    a.fill_between(x, -env, env, color=CVD[2], alpha=0.20, hatch="//", edgecolor=CVD[2], lw=0)
    a.plot(x, env, ls=LS[2], color=CVD[2], lw=1.1, label=r"pointwise null 95% $|\pi|$")
    a.plot(x, -env, ls=LS[2], color=CVD[2], lw=1.1)

    for key, ci_key, lbl, col, ls, mk in [
            ("pi", "pi_ci", r"$\pi(s)=\rho(w_s, w_{\mathrm{final}})$", CVD[0], LS[0], MK[0]),
            ("pip", "pip_ci", r"partial $\pi(s\,|\,J)$: J removed", CVD[1], LS[1], MK[1])]:
        y = np.array(P[key])[k]
        ci = np.array(P[ci_key])[k]
        a.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], ls=ls, marker=mk, ms=4, lw=1.4,
                   color=col, capsize=2, elinewidth=0.8, label=lbl)

    # the ceiling w's own measurement noise puts on pi -- it is ~0.93 everywhere, so a low pi is
    # not attenuation.
    a.plot(x, np.array(P["pi_ceiling"])[k], ls=LS[3], color="0.35", lw=1.2,
           label=r"ceiling from $w$ reliability")

    a.axvspan(8, 32, alpha=0.16, hatch="\\\\", facecolor=CVD[4], edgecolor=CVD[4], lw=0)
    a.axvspan(64, 128, alpha=0.22, hatch="xx", facecolor=CVD[3], edgecolor=CVD[3], lw=0)
    a.annotate("divergence\nordering\n8$\\to$32", xy=(16, -0.30), fontsize=7, ha="center",
               color="0.25")
    a.annotate("ranking locks in\n64$\\to$128", xy=(95, 0.62), xytext=(300, 0.30), fontsize=7,
               ha="center", color="0.25",
               arrowprops=dict(arrowstyle="->", lw=0.9, color="0.35"))
    a.annotate("plateau shape\n1000$\\to$2000", xy=(1400, 0.80), xytext=(4000, 0.42), fontsize=7,
               ha="center", color="0.25",
               arrowprops=dict(arrowstyle="->", lw=0.9, color="0.35"))
    a.axhline(0, color="0.35", lw=0.8)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(-0.3, steps.max() * 1.6)
    a.set_ylim(-0.45, 1.06)
    a.set_xlabel("training step $s$ (symlog; 0 shown at left)")
    a.set_ylabel(r"rank agreement with the final ranking")
    a.set_title("A. When the per-pair width ranking is fixed")
    a.legend(loc="lower right", fontsize=7.5)

    # --- B: full checkpoint x checkpoint ranking-agreement matrix ------------------------------
    a = ax[1]
    M = np.array(P["matrix"])
    im = a.imshow(M, cmap="cividis", vmin=0.0, vmax=1.0, origin="lower")
    lab = [f"{int(s)}" for s in steps]
    tick = np.arange(0, len(steps), 2)
    a.set_xticks(tick, [lab[i] for i in tick], rotation=90, fontsize=7)
    a.set_yticks(tick, [lab[i] for i in tick], fontsize=7)
    a.set_xlabel("training step (checkpoint index, not to scale)")
    a.set_ylabel("training step (checkpoint index, not to scale)")
    a.set_title("B. Ranking agreement between every pair of checkpoints")
    a.grid(False)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    cb.set_label(r"Spearman $\rho(w_{s_1}, w_{s_2})$ over the 60 pairs")
    # mark the block boundary: checkpoints from step 128 onwards agree with each other
    b = list(steps).index(128.0) - 0.5
    for f in (a.axhline, a.axvline):
        f(b, color="#FFFFFF", lw=1.2, ls="--")
    a.annotate("step 128", xy=(b, len(steps) - 1.4), color="#FFFFFF", fontsize=7, ha="left")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ranking_persistence.png"))
    plt.close(fig)
    print("wrote plots/ranking_persistence.png")


if __name__ == "__main__":
    main()
