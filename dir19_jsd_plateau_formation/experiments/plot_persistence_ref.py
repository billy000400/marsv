"""Figure for persistence_ref.py: is the "ranking locks in" bracket a property of the reference?

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
LS = ["-", "--", ":", "-.", (0, (7, 2))]
MK = ["o", "s", "^", "D", "v"]


def main():
    P = json.load(open(os.path.join(RESULTS, "persistence_ref.json")))
    steps = np.array(P["steps"], dtype=float)
    refs = list(P["references"].keys())

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2))

    # --- A: one persistence trajectory per reference checkpoint -------------------------------
    a = ax[0]
    env = np.array(P["references"]["143000"]["null_pt95"])
    a.fill_between(steps, -env, env, color="0.55", alpha=0.20, hatch="//", edgecolor="0.45", lw=0)
    a.plot(steps, env, ls=":", color="0.45", lw=1.0, label="pointwise null 95% $|\\pi|$")
    a.plot(steps, -env, ls=":", color="0.45", lw=1.0)

    for i, rs in enumerate(refs):
        R = P["references"][rs]
        k = np.arange(len(steps)) != R["ref_index"]      # a checkpoint scored against itself is 1
        a.plot(steps[k], np.array(R["pi"])[k], ls=LS[i], marker=MK[i], ms=4, lw=1.4,
               color=CVD[i], label=f"reference = step {int(rs):,}")

    a.axvspan(64, 128, alpha=0.22, hatch="xx", facecolor="0.4", edgecolor="0.4", lw=0)
    a.annotate("ranking locks in\n64$\\to$128 for all five", xy=(95, 0.62), xytext=(600, 0.22),
               fontsize=7, ha="center", color="0.25",
               arrowprops=dict(arrowstyle="->", lw=0.9, color="0.35"))
    a.axhline(0, color="0.35", lw=0.8)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(-0.3, steps.max() * 1.6)
    a.set_ylim(-0.35, 1.06)
    a.set_xlabel("training step $s$ (symlog; 0 shown at left)")
    a.set_ylabel(r"rank agreement $\pi_{\mathrm{ref}}(s)$ with the reference ranking")
    a.set_title("A. Persistence trajectory under five reference checkpoints")
    a.legend(loc="lower right", fontsize=7)

    # --- B: the two bracket checkpoints, per reference -----------------------------------------
    a = ax[1]
    k32 = list(steps).index(32.0)
    k128 = list(steps).index(128.0)
    y = np.arange(len(refs))
    envmax = max(P["references"][rs]["null_pt95"][k32] for rs in refs)
    a.axvspan(-envmax, envmax, facecolor="0.55", alpha=0.20, hatch="//", edgecolor="0.45",
              lw=0,
              label="pointwise 95% chance envelope")
    for j, (kk, lbl, col, ls, mk, off) in enumerate([
            (k32, r"$\pi$ at step 32 (before)", CVD[0], LS[0], MK[0], -0.15),
            (k128, r"$\pi$ at step 128 (after)", CVD[1], LS[1], MK[1], +0.15)]):
        v = np.array([P["references"][rs]["pi"][kk] for rs in refs])
        ci = np.array([P["references"][rs]["pi_ci"][kk] for rs in refs])
        a.errorbar(v, y + off, xerr=[v - ci[:, 0], ci[:, 1] - v], ls="none", marker=mk, ms=5,
                   color=col, capsize=2.5, elinewidth=1.0, label=lbl)
    vp = np.array([P["references"][rs]["pip"][k32] for rs in refs])
    a.plot(vp, y, ls="none", marker=MK[2], ms=5, mfc="none", color=CVD[2],
           label=r"$\pi^{\perp}$ at step 32 ($J$ removed)")
    a.axvline(0, color="0.35", lw=0.8)
    a.set_yticks(y, [f"step {int(r):,}" for r in refs], fontsize=8)
    a.set_ylim(-0.6, len(refs) + 0.5)
    a.set_xlim(-0.45, 1.02)
    a.set_xlabel(r"rank agreement with the reference ranking")
    a.set_ylabel("reference checkpoint")
    a.set_title("B. Step 32 stays inside chance, step 128 outside it")
    a.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "reference_robustness.png"))
    plt.close(fig)
    print("wrote plots/reference_robustness.png")


if __name__ == "__main__":
    main()
