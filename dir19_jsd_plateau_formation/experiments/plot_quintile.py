"""Figure for quintile_loo.py + quintile_large.py: which pairs carry the step-32 ordering?

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

ORDER = ["all", "drop_Q1", "drop_Q2", "drop_Q3", "drop_Q4", "drop_Q5", "middle3"]
LABEL = ["all 60 pairs", "drop Q1 (lowest $J$)", "drop Q2", "drop Q3", "drop Q4",
         "drop Q5 (highest $J$)", "middle three only"]


def main():
    S = json.load(open(os.path.join(RESULTS, "quintile_loo.json")))
    L = json.load(open(os.path.join(RESULTS, "quintile_large.json")))
    sub, ctrl = S["subsets"], S["random_drop_control"]

    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.6))
    y = np.arange(len(ORDER))[::-1]

    # --- A: 60-pair rho at step 32 per subset, against the size-matched random-drop envelope ---
    a = ax[0]
    for i, k in enumerate(ORDER):
        d = sub[k]
        lo, hi = d["band_32"]
        cut = (k in ("drop_Q5", "middle3"))
        a.plot([lo, hi], [y[i], y[i]], color=CVD[1] if cut else CVD[0], lw=1.4,
               ls="--" if cut else "-")
        a.plot(d["rho_32"], y[i], marker="s" if cut else "o", ms=7,
               color=CVD[1] if cut else CVD[0], mfc="none" if cut else None, mew=1.6)
        n = str(d["n"])
        if n in ctrl:
            p = ctrl[n]["pct"]
            a.plot([p[0], p[4]], [y[i] + 0.26] * 2, color="0.35", lw=5, alpha=0.35,
                   solid_capstyle="butt")
            a.plot(p[2], y[i] + 0.26, marker="|", ms=9, color="0.25")
    a.axvline(0, color="0.35", lw=0.9, ls=":")
    a.set_yticks(y)
    a.set_yticklabels(LABEL)
    a.set_xlabel(r"Spearman $\rho(J,\,w)$ at step 32")
    a.set_title("A. 60-pair bank: only the top quintile\nis load-bearing", fontsize=9.5)

    # --- B: per-quintile median width change over step 8 -> 32 --------------------------------
    b = ax[1]
    q = S["per_quintile"]
    xs = np.arange(1, 6)
    dw = np.array([r["dw"] for r in q]) * 1e3
    err = np.array([[r["dw"] - r["dw_ci"][0], r["dw_ci"][1] - r["dw"]] for r in q]).T * 1e3
    b.errorbar(xs, dw, yerr=err, fmt="o", ms=6, lw=1.4, capsize=3, color=CVD[0], ls="-")
    b.plot(xs[-1], dw[-1], marker="s", ms=10, mfc="none", mew=1.8, color=CVD[1], ls="none",
           label="Q5: the only quintile that sharpens")
    b.axhline(0, color="0.35", lw=0.9, ls=":")
    b.set_xticks(xs)
    b.set_xticklabels([f"Q{r['q']}\n$J\\approx${r['J_med']:.2f}" for r in q])
    b.set_xlabel("corpus-divergence quintile (median $J$, bits)")
    b.set_ylabel(r"median $\Delta w$, step 8 $\to$ 32 ($\times 10^{-3}$)")
    b.set_title("B. 60-pair bank: the sharpening\nis confined to Q5", fontsize=9.5)
    b.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # --- C: 1,000-pair bank, same subsets, step 32 against the mature checkpoint ---------------
    c = ax[2]
    for j, (s, mk, ls, lab) in enumerate([("32", "o", "-", "step 32"),
                                          ("143000", "s", "--", "step 143000 (mature)")]):
        v = [L["subsets"][k][s]["rho"] for k in ORDER]
        c.plot(v, y, marker=mk, ms=6, ls=ls, lw=1.3, color=CVD[j], label=lab)
    env = L["subsets"]["all"]["32"]["null95"]
    c.axvspan(-env, env, alpha=0.25, hatch="//", facecolor="0.45", edgecolor="0.45", lw=0,
              label=r"95% endpoint-label null for $|\rho|$")
    c.axvline(0, color="0.35", lw=0.9, ls=":")
    c.set_yticks(y)
    c.set_yticklabels([f"{l}\n(n={L['subsets'][k]['32']['n']})" for l, k in zip(LABEL, ORDER)],
                      fontsize=7.5)
    c.set_xlabel(r"Spearman $\rho(J,\,w)$")
    c.set_title("C. 1,000-pair bank: the bulk relation is absent\nat step 32, present at the end", fontsize=9.5)
    c.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    p = os.path.join(PLOTS, "quintile_dependence.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
