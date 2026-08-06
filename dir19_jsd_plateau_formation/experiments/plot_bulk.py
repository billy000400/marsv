"""Figure for bulk_onset.py: when does the GRADED (middle-range) ordering arrive?

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
    D = json.load(open(os.path.join(RESULTS, "bulk_onset.json")))
    steps = D["steps"]
    xs = np.array([x_of(s) for s in steps])

    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.5))

    # --- A: rho trajectories, full bank vs middle three quintiles ------------------------------
    a = ax[0]
    style = [("all", "all 1,000 pairs", CVD[0], "-", "o"),
             ("middle3", "middle three quintiles (600)", CVD[1], "--", "s")]
    for k, lab, c, ls, mk in style:
        d = D["subsets"][k]
        r = np.array([d["steps"][str(s)]["rho"] for s in steps])
        lo = np.array([d["steps"][str(s)]["ci"][0] for s in steps])
        hi = np.array([d["steps"][str(s)]["ci"][1] for s in steps])
        a.errorbar(xs, r, yerr=[r - lo, hi - r], color=c, ls=ls, marker=mk, ms=6, lw=1.6,
                   capsize=2.5, label=lab)
        a.axhline(-d["sim95"], color=c, ls=":", lw=1.0)
    a.axhspan(-D["subsets"]["all"]["sim95"], D["subsets"]["all"]["sim95"], color="0.6",
              alpha=0.18, hatch="..", ec="0.45", lw=0)
    a.axhline(0, color="0.35", lw=0.9, ls="--")
    for k, hatch in (("all", "\\\\"), ("middle3", "xx")):
        br = D["subsets"][k]["bracket"]
        if br and br[0] is not None:
            a.axvspan(x_of(br[0]), x_of(br[1]), color="0.5", alpha=0.16, hatch=hatch, ec="0.4",
                      lw=0)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlim(0.3, 3e5)
    a.set_xlabel("training step")
    a.set_ylabel(r"Spearman $\rho(J,\,w)$")
    a.legend(fontsize=7.6, loc="lower left")
    a.set_title("A. The graded middle of the range fills in\nbetween step 32 and step 256", fontsize=9.5)

    # --- B: group-level separation of the top quintile ----------------------------------------
    b = ax[1]
    g = D["group_gap"]
    obs = np.array([g[str(s)]["obs"] for s in steps])
    lo = np.array([g[str(s)]["ci"][0] for s in steps])
    hi = np.array([g[str(s)]["ci"][1] for s in steps])
    b.errorbar(xs, obs, yerr=[obs - lo, hi - obs], color=CVD[2], ls="-.", marker="D", ms=5.5,
               lw=1.6, capsize=2.5, label=r"median $w$(Q5) $-$ median $w$(Q1–Q4)")
    b.axhline(0, color="0.35", lw=0.9, ls="--")
    b.set_xscale("symlog", linthresh=1)
    b.set_yscale("symlog", linthresh=0.005)
    b.set_xlim(0.3, 3e5)
    b.set_xlabel("training step")
    b.set_ylabel(r"width gap $G_s$ (units of $t$)")
    b.legend(fontsize=7.6, loc="lower left")
    b.set_title("B. The top quintile is already separated\nas a group at step 32", fontsize=9.5)

    # --- C: the clocks on one timeline --------------------------------------------------------
    c = ax[2]
    rows = [("plateau shape (60-pair)", 1000, 2000, CVD[3], "//"),
            ("graded ordering, middle range\n(1,000-pair)", *D["subsets"]["middle3"]["bracket"][:2],
             CVD[1], "xx"),
            ("ranking becomes final (60-pair)", 64, 128, CVD[4], ".."),
            ("ordering, whole bank (1,000-pair)", *D["subsets"]["all"]["bracket"][:2], CVD[0],
             "\\\\")]
    for i, (lab, s0, s1, col, hatch) in enumerate(rows):
        y = len(rows) - 1 - i
        c.barh(y, x_of(s1) - x_of(s0), left=x_of(s0), height=0.5, color=col, alpha=0.55,
               hatch=hatch, ec="0.25", lw=0.8)
        c.text(x_of(s1) * 1.35, y, f"{s0} → {s1}", va="center", fontsize=8)
    c.set_yticks(range(len(rows)))
    c.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
    c.set_xscale("log")
    c.set_xlim(4, 3e5)
    c.set_xlabel("training step (onset bracket)")
    c.set_title("C. Four clocks, in order", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "bulk_onset.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/bulk_onset.png")


if __name__ == "__main__":
    main()
