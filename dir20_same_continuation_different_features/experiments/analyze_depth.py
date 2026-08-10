"""S5: does the plateau (and the inverted JSD association) depend on depth below the patch?

Compares the mined-bank sweep run with the patch at block 0, 12 and 20. Same 200 pairs per
model in every case; only the patch site moves. Writes results/depth_analysis.json and
plots/depth_effect.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from analyze_bank import THR, LIN, cluster_boot, rho_of
from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
MODELS = ["gpt2-medium", "pythia-410m"]
LAYERS = [0, 12, 20]
STYLE = {"gpt2-medium": dict(color=CVD[0], ls="-", marker="o"),
         "pythia-410m": dict(color=CVD[1], ls="--", marker="s")}


def load_bank(m, layer):
    suf = "" if layer == 0 else f"_L{layer}"
    return json.load(open(os.path.join(RESULTS, f"bank_{m}{suf}.json")))


def main():
    rng = np.random.default_rng(0)
    stats = {}
    for m in MODELS:
        stats[m] = {}
        for L in LAYERS:
            rows = load_bank(m, L)
            clusters = {}
            for r in rows:
                clusters.setdefault(r["prefix_idx"], []).append(r)
            wtv = np.array([r["wtv"] for r in rows])
            x = [r["jsd"] for r in rows]
            rho, p = spearmanr(x, wtv)
            lo, hi = cluster_boot(clusters, lambda rr: rho_of(rr, "wtv"), rng)
            uns = [r for r in rows if r["jsd"] < 0.65]
            rho_u, p_u = spearmanr([r["jsd"] for r in uns], [r["wtv"] for r in uns])
            cl_u = {}
            for r in uns:
                cl_u.setdefault(r["prefix_idx"], []).append(r)
            lo_u, hi_u = cluster_boot(cl_u, lambda rr: rho_of(rr, "wtv"), rng)
            w = [r["w"] for r in rows if r["w"] is not None]
            stats[m][L] = dict(
                n=len(rows), n_below=23 - L,
                median_wtv=float(np.median(wtv)),
                q1_wtv=float(np.percentile(wtv, 25)), q3_wtv=float(np.percentile(wtv, 75)),
                frac_sharp=float(np.mean(wtv < THR["wtv"])),
                median_w=float(np.median(w)), n_w=len(w),
                frac_mono=float(np.mean([r["mono"] for r in rows])),
                max_endpoint_err=max(max(r["endpoint_err"]) for r in rows),
                rho_wtv=float(rho), p_wtv=float(p), ci=[lo, hi],
                rho_wtv_unsat=float(rho_u), p_wtv_unsat=float(p_u), n_unsat=len(uns),
                ci_unsat=[lo_u, hi_u])
            print(m, L, json.dumps(stats[m][L]))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for m in MODELS:
        s = [stats[m][L] for L in LAYERS]
        st = STYLE[m]
        ax[0].plot(LAYERS, [q["median_wtv"] for q in s], lw=2, ms=8, mec="k", mew=0.6,
                   label=m, **st)
        ax[0].fill_between(LAYERS, [q["q1_wtv"] for q in s], [q["q3_wtv"] for q in s],
                           color=st["color"], alpha=0.16,
                           hatch="//" if m == MODELS[0] else "\\\\", edgecolor=st["color"],
                           linewidth=0.0)
        ax[1].errorbar(LAYERS, [q["rho_wtv_unsat"] for q in s],
                       yerr=[[q["rho_wtv_unsat"] - q["ci_unsat"][0] for q in s],
                             [q["ci_unsat"][1] - q["rho_wtv_unsat"] for q in s]],
                       lw=2, ms=8, mec="k", mew=0.6, capsize=4, label=m, **st)
    ax[0].axhline(LIN["wtv"], color="0.35", ls=(0, (4, 3)), lw=1.4)
    ax[0].text(0.5, LIN["wtv"] + 0.012, "linear response (0.5)", fontsize=7.5, color="0.3")
    ax[0].axhline(THR["wtv"], color="0.1", ls=":", lw=1.4)
    ax[0].text(0.5, THR["wtv"] - 0.014, "sharp threshold (0.25)", fontsize=7.5, color="0.1",
               va="top")
    ax[0].set_ylim(0, 0.62)
    ax[0].set_ylabel("$w_{\\mathrm{TV}}$ at final logits (median, IQR band)")
    ax[0].set_title("Plateau strength vs patch depth", fontsize=10)
    ax[1].axhline(0, color="0.35", ls=(0, (4, 3)), lw=1.4)
    ax[1].text(0.5, 0.02, "no association", fontsize=7.5, color="0.3")
    ax[1].set_ylabel("Spearman $\\rho$(JSD, $w_{\\mathrm{TV}}$), JSD $<0.65$ pairs")
    ax[1].set_title("Divergence-sharpness association vs patch depth", fontsize=10)
    for a in ax:
        a.set_xticks(LAYERS)
        a.set_xticklabels([f"block {L}\n({23 - L} blocks below)" for L in LAYERS], fontsize=8)
        a.set_xlabel("patch site")
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "depth_effect.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(RESULTS, "depth_analysis.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("wrote depth_analysis.json + plots/depth_effect.png")


if __name__ == "__main__":
    main()
