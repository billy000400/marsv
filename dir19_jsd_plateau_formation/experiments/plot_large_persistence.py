"""Figure for large_persistence.py: are the graded ordering and the ranking lock-in one event?

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


def x_of(s):
    return max(s, 0.5)


def main():
    D = json.load(open(os.path.join(RESULTS, "large_persistence.json")))
    B = json.load(open(os.path.join(RESULTS, "bulk_onset.json")))
    steps = D["steps"]
    xs = np.array([x_of(s) for s in steps])
    R = D["rows"]

    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.5))

    # --- A: pi and pi_perp on the 1,000-pair bank ----------------------------------------------
    a = ax[0]
    for key, cikey, lab, c, ls, mk in (
            ("pi", "pi_ci", r"$\pi$ (agreement with final ranking)", CVD[0], "-", "o"),
            ("pi_perp", "pi_perp_ci", r"$\pi^{\perp}$ ($J$ partialled out)", CVD[1], "--", "s")):
        v = np.array([R[str(s)][key] for s in steps])
        lo = np.array([R[str(s)][cikey][0] for s in steps])
        hi = np.array([R[str(s)][cikey][1] for s in steps])
        a.errorbar(xs, v, yerr=[v - lo, hi - v], color=c, ls=ls, marker=mk, ms=6, lw=1.6,
                   capsize=2.5, label=lab)
    a.axhline(0, color="0.35", lw=0.9, ls="--")
    br = D["bracket_dpi"]
    if br and br[0] is not None:
        a.axvspan(x_of(br[0]), x_of(br[1]), color="0.5", alpha=0.16, hatch="..", ec="0.4", lw=0)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(0.3, 3e5)
    a.set_xlabel("training step")
    a.set_ylabel("rank agreement with step-143000 widths")
    a.legend(fontsize=7.6, loc="upper left")
    a.set_title("A. The 1,000-pair ranking, scored\nagainst the final one", fontsize=9.5)

    # --- B: dpi with the simultaneous band -----------------------------------------------------
    b = ax[1]
    d = np.array([R[str(s)]["dpi"] for s in steps])
    lo = np.array([R[str(s)]["dpi_sim_lo"] for s in steps])
    hi = np.array([R[str(s)]["dpi_sim_hi"] for s in steps])
    b.errorbar(xs, d, yerr=[d - lo, hi - d], color=CVD[2], ls="-.", marker="D", ms=5.5, lw=1.6,
               capsize=2.5, label=r"$\Delta\pi(s)=\pi(s)-\pi(0)$, simultaneous 95% band")
    b.axhline(0, color="0.35", lw=0.9, ls="--")
    if br and br[0] is not None:
        b.axvspan(x_of(br[0]), x_of(br[1]), color="0.5", alpha=0.16, hatch="..", ec="0.4", lw=0)
    b.set_xscale("symlog", linthresh=1)
    b.set_xlim(0.3, 3e5)
    b.set_xlabel("training step")
    b.set_ylabel(r"$\Delta\pi$ (ranking acquired since step 0)")
    b.legend(fontsize=7.4, loc="upper left")
    b.set_title("B. Onset rule on the same bank:\nranking acquired by step 64", fontsize=9.5)

    # --- C: both clocks on the same bank, as a fraction of their final value -------------------
    c = ax[2]
    mid = B["subsets"]["middle3"]["steps"]
    rho = np.array([mid[str(s)]["rho"] for s in steps])
    rho_f = rho[-1]
    c.plot(xs, rho / rho_f, color=CVD[1], ls="--", marker="s", ms=6, lw=1.7,
           label=r"graded ordering $\rho(J,w)$, middle 600 pairs")
    c.plot(xs, d / d[-1], color=CVD[2], ls="-.", marker="D", ms=5.5, lw=1.7,
           label=r"ranking $\Delta\pi(s)$")
    c.axhline(0, color="0.35", lw=0.9, ls="--")
    c.axhline(1, color="0.6", lw=0.9, ls=":")
    if br and br[0] is not None:
        c.axvspan(x_of(br[0]), x_of(br[1]), color="0.5", alpha=0.16, hatch="..", ec="0.4", lw=0)
    mbr = B["subsets"]["middle3"]["bracket"]
    if mbr and mbr[0] is not None:
        c.axvspan(x_of(mbr[0]), x_of(mbr[1]), color="0.5", alpha=0.16, hatch="xx", ec="0.4", lw=0)
    c.set_xscale("symlog", linthresh=1)
    c.set_xlim(0.3, 3e5)
    c.set_xlabel("training step")
    c.set_ylabel("fraction of the step-143000 value")
    c.legend(fontsize=7.4, loc="lower right")
    c.set_title("C. Both clocks run inside\nsteps 32–128", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "large_persistence.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/large_persistence.png")


if __name__ == "__main__":
    main()
