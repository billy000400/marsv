"""Figure for the exploratory depth sweep: third-token rate and plateau flatness vs interpolation
block, joining the preregistered blocks 0/2/4/6 to the exploratory blocks 12/18/24/30.

plots/depth_sweep.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ALPHAS, PLOTS, RESULTS
from cvd_style import CVD, REF_DIAG, REF_RULE, use_cvd

use_cvd()
PRE = [0, 2, 4, 6]


def main():
    a = json.load(open(os.path.join(RESULTS, "analysis.json")))["by_layer"]
    e = json.load(open(os.path.join(RESULTS, "depth_extension.json")))["by_layer"]
    z = np.load(os.path.join(RESULTS, "matthew_d_curves.npz"))
    pri_rho = np.load(os.path.join(RESULTS, "depth_extension_rho.npz"))

    # flatness of the preregistered blocks, from the primary d(t) run
    import pickle
    rows = pickle.load(open(os.path.join(RESULTS, "screen_primary.pkl"), "rb"))["rows"]
    d, kin, kout, ci = z["d_cand"], z["kin"], z["kout"], z["cand_idx"]
    rho0 = np.array([(d[i, kin[i]:kout[i] + 1].max() - d[i, kin[i]:kout[i] + 1].min())
                     / (ALPHAS[kout[i]] - ALPHAS[kin[i]]) for i in range(len(ci))])
    lay0 = np.array([rows[i]["layer"] for i in ci])

    blocks, rate, lo, hi, medrho, sub, sublo, subhi, pre = [], [], [], [], [], [], [], [], []
    for l in PRE:
        s = a[str(l)]
        m = lay0 == l
        blocks.append(l); rate.append(s["rate"]); lo.append(s["ci"][0]); hi.append(s["ci"][1])
        medrho.append(float(np.median(rho0[m])))
        k = int((rho0[m] < 0.5).sum())
        sub.append(k / s["n_eligible"])
        from common import wilson
        a_, b_ = wilson(k, s["n_eligible"])
        sublo.append(a_); subhi.append(b_); pre.append(True)
    for l in sorted(int(x) for x in e):
        s = e[str(l)]
        blocks.append(l); rate.append(s["rate"]); lo.append(s["ci"][0]); hi.append(s["ci"][1])
        medrho.append(s["median_rho"]); sub.append(s["subplateau_rate"])
        sublo.append(s["subplateau_ci"][0]); subhi.append(s["subplateau_ci"][1]); pre.append(False)

    blocks = np.array(blocks)
    npre = np.array(pre)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    def two_tone(ax, y, ylo, yhi, label, color, ls, mk):
        err = [np.clip(np.array(y) - np.array(ylo), 0, None),
               np.clip(np.array(yhi) - np.array(y), 0, None)]
        ax.errorbar(blocks, y, yerr=err,
                    color=color, ls=ls, marker=mk, lw=2, ms=6, capsize=3, label=label)

    two_tone(a1, np.array(rate) * 100, np.array(lo) * 100, np.array(hi) * 100,
             "persistent third top-1 token (solid, circles)", CVD[0], "-", "o")
    two_tone(a1, np.array(sub) * 100, np.array(sublo) * 100, np.array(subhi) * 100,
             "true sub-plateau, ρ < 0.5 (dashed, squares)", CVD[1], "--", "s")
    a1.axvspan(-0.5, 6.5, color="0.88", hatch="//", ec="0.6", lw=0, zorder=0)
    a1.annotate("preregistered\nblocks 0–6", (0.02, 0.62), xycoords="axes fraction", fontsize=8)
    a1.annotate("exploratory\nblocks 12–30", (0.60, 0.62), xycoords="axes fraction", fontsize=8)
    a1.set_xlabel("interpolation block L of GPT-2 Large (36 blocks)")
    a1.set_ylabel("% of eligible paths")
    a1.set_title("How often, and how often as a real plateau?", fontsize=10)
    a1.legend(fontsize=8, loc="upper left")

    a2.plot(blocks[npre], np.array(medrho)[npre], color=CVD[0], ls="-", marker="o", lw=2, ms=6,
            label="preregistered blocks (solid, circles)")
    a2.plot(blocks[~npre], np.array(medrho)[~npre], color=CVD[1], ls="--", marker="s", lw=2, ms=6,
            label="exploratory blocks (dashed, squares)")
    a2.axhline(1.0, label="ρ = 1: as steep as the no-plateau diagonal", **REF_DIAG)
    a2.axhline(0.5, label="ρ = 0.5: sub-plateau line", **REF_RULE)
    a2.set_xlabel("interpolation block L of GPT-2 Large (36 blocks)")
    a2.set_ylabel("median flatness ρ of the C window")
    a2.set_ylim(0, max(medrho) * 1.1)
    a2.set_title("Are the third regions getting flatter with depth?", fontsize=10)
    a2.legend(fontsize=8)
    for ax in (a1, a2):
        ax.grid(alpha=0.25)
        ax.set_xticks(list(blocks))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "depth_sweep.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"block": blocks.tolist(),
                      "rate_pct": [round(100 * r, 2) for r in rate],
                      "subplateau_pct": [round(100 * s, 2) for s in sub],
                      "median_rho": [round(m, 2) for m in medrho]}, indent=1))


if __name__ == "__main__":
    main()
