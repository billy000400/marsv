"""S7: is the plateau set by the ABSOLUTE number of blocks below the patch, or by the
FRACTION of the stack below it?

Experiment 4 moved the patch inside three 24-block models, which cannot separate the two
readings. Here the same mined-bank sweep is run in three members of the GPT-2 family with
12, 24 and 36 blocks (identical tokenizer and pretraining corpus), at patch sites chosen so
that the models line up either on blocks-below or on fraction-of-stack-below.

Writes results/scaling_analysis.json and plots/depth_scaling.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from analyze_bank import THR, LIN, cluster_boot, rho_of
from common import N_BLOCKS, PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)

# patch sites per model; blocks below = N_BLOCKS - 1 - L
SITES = {"gpt2-small": [0, 6, 8, 10],
         "gpt2-medium": [0, 12, 20],
         "gpt2-large": [0, 12, 18, 24, 31]}
FAMILY = ["gpt2-small", "gpt2-medium", "gpt2-large"]
STYLE = {"gpt2-small": dict(color=CVD[0], ls="-", marker="o"),
         "gpt2-medium": dict(color=CVD[1], ls="--", marker="s"),
         "gpt2-large": dict(color=CVD[2], ls=":", marker="^")}

# Levels at which the three models are matched under each reading (model -> patch site).
MATCH_ABS = {"11 blocks below": {"gpt2-small": 0, "gpt2-medium": 12, "gpt2-large": 24},
             "3-4 blocks below": {"gpt2-small": 8, "gpt2-medium": 20, "gpt2-large": 31}}
MATCH_FRAC = {"whole stack below (f=1.0)":
              {"gpt2-small": 0, "gpt2-medium": 0, "gpt2-large": 0},
              "half the stack below (f=0.46-0.49)":
              {"gpt2-small": 6, "gpt2-medium": 12, "gpt2-large": 18},
              "tail of the stack below (f=0.09-0.13)":
              {"gpt2-small": 10, "gpt2-medium": 20, "gpt2-large": 31}}


def load_bank(m, layer):
    suf = "" if layer == 0 else f"_L{layer}"
    return json.load(open(os.path.join(RESULTS, f"bank_{m}{suf}.json")))


def site_stats(m, L, rng):
    rows = load_bank(m, L)
    wtv = np.array([r["wtv"] for r in rows])
    w = [r["w"] for r in rows if r["w"] is not None]
    uns = [r for r in rows if r["jsd"] < 0.65]
    rho_u, p_u = spearmanr([r["jsd"] for r in uns], [r["wtv"] for r in uns])
    cl = {}
    for r in uns:
        cl.setdefault(r["prefix_idx"], []).append(r)
    lo, hi = cluster_boot(cl, lambda rr: rho_of(rr, "wtv"), rng)
    nb = N_BLOCKS[m]
    return dict(model=m, layer=L, n=len(rows), n_blocks=nb,
                blocks_below=nb - 1 - L, frac_below=(nb - 1 - L) / (nb - 1),
                median_wtv=float(np.median(wtv)),
                q1_wtv=float(np.percentile(wtv, 25)),
                q3_wtv=float(np.percentile(wtv, 75)),
                frac_sharp=float(np.mean(wtv < THR["wtv"])),
                median_w=float(np.median(w)), n_w=len(w),
                frac_mono=float(np.mean([r["mono"] for r in rows])),
                max_endpoint_err=max(max(r["endpoint_err"]) for r in rows),
                rho_wtv_unsat=float(rho_u), p_wtv_unsat=float(p_u), n_unsat=len(uns),
                ci_unsat=[lo, hi])


def spread(stats, levels):
    """Across-model range of median w_TV at each matched level."""
    out = {}
    for name, sel in levels.items():
        vals = {m: stats[m][L]["median_wtv"] for m, L in sel.items()}
        out[name] = dict(values=vals, range=float(max(vals.values()) - min(vals.values())))
    return out


def main():
    rng = np.random.default_rng(0)
    stats = {m: {L: site_stats(m, L, rng) for L in SITES[m]} for m in FAMILY}
    for m in FAMILY:
        for L in SITES[m]:
            s = stats[m][L]
            print(f"{m:12s} L{L:<3d} below={s['blocks_below']:<3d} f={s['frac_below']:.3f} "
                  f"med_wtv={s['median_wtv']:.3f} sharp={s['frac_sharp']:.1%} "
                  f"rho={s['rho_wtv_unsat']:+.2f}")

    res = dict(sites=stats,
               matched_absolute=spread(stats, MATCH_ABS),
               matched_fraction=spread(stats, MATCH_FRAC))
    res["mean_range_absolute"] = float(np.mean(
        [v["range"] for v in res["matched_absolute"].values()]))
    res["mean_range_fraction"] = float(np.mean(
        [v["range"] for v in res["matched_fraction"].values()]))
    print(json.dumps({k: res[k] for k in
                      ["matched_absolute", "matched_fraction",
                       "mean_range_absolute", "mean_range_fraction"]}, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for m in FAMILY:
        s = [stats[m][L] for L in SITES[m]]
        st = STYLE[m]
        lab = f"{m} ({N_BLOCKS[m]} blocks)"
        ax[0].plot([q["blocks_below"] for q in s], [q["median_wtv"] for q in s],
                   lw=2, ms=8, mec="k", mew=0.6, label=lab, **st)
        ax[1].plot([q["frac_below"] for q in s], [q["median_wtv"] for q in s],
                   lw=2, ms=8, mec="k", mew=0.6, label=lab, **st)
    for a, xl, tt in [(ax[0], "blocks below the patch site",
                       "Matched on absolute depth below the patch"),
                      (ax[1], "fraction of the stack below the patch site",
                       "Matched on fraction of the stack below")]:
        a.axhline(LIN["wtv"], color="0.35", ls=(0, (4, 3)), lw=1.4)
        a.axhline(THR["wtv"], color="0.1", ls=":", lw=1.4)
        a.set_xlabel(xl)
        a.set_title(tt, fontsize=10)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="upper right")
        a.set_ylim(0, 0.62)
    ax[0].text(0.5, 0.035, f"mean across-model spread at matched levels: "
               f"{res['mean_range_absolute']:.3f}", fontsize=8.5, color="0.15")
    ax[1].text(0.06, 0.035, f"mean across-model spread at matched levels: "
               f"{res['mean_range_fraction']:.3f}", fontsize=8.5, color="0.15")
    ax[0].text(1.0, LIN["wtv"] + 0.012, "linear response (0.5)", fontsize=7.5, color="0.3")
    ax[0].text(1.0, THR["wtv"] - 0.014, "sharp threshold (0.25)", fontsize=7.5, color="0.1",
               va="top")
    ax[0].set_ylabel("$w_{\\mathrm{TV}}$ at final logits (median of 200 pairs)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "depth_scaling.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(RESULTS, "scaling_analysis.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("wrote scaling_analysis.json + plots/depth_scaling.png")


if __name__ == "__main__":
    main()
