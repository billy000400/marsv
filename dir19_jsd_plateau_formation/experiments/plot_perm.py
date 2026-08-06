"""Figure for the permutation tests (permtest.py). Headless Agg; CVD-safe palette, every series
also coded by linestyle/marker so the figure survives grayscale printing."""
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


def xaxis(ax, steps):
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.3, max(steps) * 1.6)
    ax.set_xlabel("training step (symlog; 0 shown at left)")


def null_band(ax, x, env, label):
    """Two-sided null envelope +/- env, drawn as a hatched band with an explicit edge."""
    x = np.asarray(x, dtype=float)
    e = np.asarray(env, dtype=float)
    ax.fill_between(x, -e, e, color=CVD[2], alpha=0.20, hatch="//", edgecolor=CVD[2], linewidth=0)
    ax.plot(x, e, ls=LS[2], color=CVD[2], lw=1.1, label=label)
    ax.plot(x, -e, ls=LS[2], color=CVD[2], lw=1.1)


def main():
    P = json.load(open(os.path.join(RESULTS, "permutation.json")))
    c, iv, q = P["small_cross"], P["small_interval"], P["large_qap"]
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.8))

    # A -- cross-sectional rho against the label-permutation null, 60-pair bank.
    a = ax[0]
    s = np.array(c["steps"], dtype=float)
    null_band(a, s, c["null_pt95"], "pointwise null 95% $|\\rho|$")
    a.axhline(c["null_env95"], ls=LS[1], color=CVD[1], lw=1.2,
              label="simultaneous null 95% (19 ckpts)")
    a.axhline(-c["null_env95"], ls=LS[1], color=CVD[1], lw=1.2)
    a.plot(s, c["rho"], ls=LS[0], marker=MK[0], ms=4, color=CVD[0], label=r"observed $\rho(J,w)$")
    a.axvspan(8, 32, alpha=0.16, hatch="\\\\", facecolor=CVD[4], edgecolor=CVD[4], lw=0)
    a.axhline(0, color="0.35", lw=0.8)
    xaxis(a, s)
    a.set_ylim(-0.78, 0.72)
    a.set_ylabel(r"Spearman $\rho$(corpus JSD $J$, width $w$)")
    a.set_title("A. 60-pair bank: ordering vs. chance")
    a.legend(loc="upper right", fontsize=7)

    # B -- interval statistic against the same null.
    a = ax[1]
    ends = np.array([float(v.split("->")[1]) for v in iv["intervals"]])
    ok = np.array([r is not None for r in iv["rho"]])
    rho = np.array([np.nan if r is None else r for r in iv["rho"]], dtype=float)
    null_band(a, ends[ok], np.array(iv["null_pt95"])[ok], "pointwise null 95%")
    a.plot(ends[ok], rho[ok], ls=LS[0], marker=MK[0], ms=4, color=CVD[0],
           label=r"observed $\rho(J,\Delta w)$")
    for name, lbl, ty, ha in [("8->32", "8$\\to$32: selective,\nno sharpening", -0.80, "left"),
                              ("512->1000", "512$\\to$1000: biggest\nsharpening, blind", 0.62,
                               "center")]:
        k = iv["intervals"].index(name)
        a.annotate(lbl, xy=(ends[k], rho[k]), xytext=(ends[k], ty), fontsize=7, ha=ha,
                   arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"))
    a.axhline(0, color="0.35", lw=0.8)
    xaxis(a, ends)
    a.set_ylim(-1.00, 0.90)
    a.set_ylabel(r"$\rho$($J$, width change $\Delta w$ in interval)")
    a.set_title("B. Which intervals are divergence-selective")
    a.legend(loc="upper left", fontsize=7)

    # C -- 1,000-pair bank under the endpoint-label (QAP) null.
    a = ax[2]
    sq = np.array(q["steps"], dtype=float)
    null_band(a, sq, q["null_pt95"], "endpoint-label null 95%")
    a.plot(sq, q["rho"], ls=LS[0], marker=MK[0], ms=5, color=CVD[0],
           label=r"observed $\rho(J,w)$, 1000 pairs")
    for i, (x, y, p) in enumerate(zip(sq, q["rho"], q["p"])):
        ha = "left" if i == 0 else ("right" if i == len(sq) - 1 else "center")
        off = (2, 9) if i < 3 else (-4 if i else 2, 9)
        a.annotate(f"p={p:.3f}" if p >= 1e-3 else "p<0.001", xy=(x, y), xytext=off,
                   textcoords="offset points", fontsize=7, ha=ha, color="0.25")
    a.axvspan(8, 32, alpha=0.16, hatch="\\\\", facecolor=CVD[4], edgecolor=CVD[4], lw=0)
    a.axhline(0, color="0.35", lw=0.8)
    xaxis(a, sq)
    a.set_ylim(-0.72, 0.28)
    a.set_ylabel(r"Spearman $\rho$(corpus JSD $J$, width $w$)")
    a.set_title("C. 1,000-pair bank: endpoint-label null")
    a.legend(loc="lower left", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "permutation_null.png"))
    plt.close(fig)
    print("wrote plots/permutation_null.png")


if __name__ == "__main__":
    main()
