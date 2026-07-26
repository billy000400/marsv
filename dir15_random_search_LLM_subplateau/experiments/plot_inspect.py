"""Figures for S4 (continuation stability) and S5 (nearest natural activations)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import PLOTS, RESULTS
from cvd_style import CVD, use_cvd

use_cvd()
HATCH = ["//", "\\\\", "..", "xx"]
LABEL = {"A": "A-region point", "C": "C-region point", "B": "B-region point",
         "natural": "natural context (control)"}


def main():
    d = json.load(open(os.path.join(RESULTS, "inspection.json")))
    nb, keys = d["neighbours"], ["A", "C", "B", "natural"]

    rng = np.random.default_rng(3)
    summ = {}
    for k in keys:
        c = np.array(nb[k]["raw_cos"])
        f = np.array(nb[k]["raw_frac"])
        bs_c = [np.median(rng.choice(c, len(c))) for _ in range(2000)]
        bs_f = [np.mean(rng.choice(f, len(f))) for _ in range(2000)]
        summ[k] = {"n": len(c), "median_cos_dist_nn": float(np.median(c)),
                   "median_cos_ci": [float(np.percentile(bs_c, 2.5)),
                                     float(np.percentile(bs_c, 97.5))],
                   "mean_agree10": float(f.mean()),
                   "agree10_ci": [float(np.percentile(bs_f, 2.5)),
                                  float(np.percentile(bs_f, 97.5))]}
    json.dump(summ, open(os.path.join(RESULTS, "neighbor_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, k in enumerate(keys):
        v = np.array(nb[k]["raw_cos"])
        ax[0].hist(v, bins=np.linspace(0, max(0.6, v.max()), 40), histtype="step", lw=2,
                   color=CVD[i], ls=["-", "--", "-.", ":"][i], label=LABEL[k], density=True)
    ax[0].set_xlabel("cosine distance to nearest natural activation  $1-\\cos$")
    ax[0].set_ylabel("density")
    ax[0].set_title("how far each query sits from the natural bank")
    ax[0].legend(fontsize=8)

    means = [nb[k]["frac_same_top1_mean"] for k in keys]
    sems = [nb[k]["frac_same_top1_sd"] / np.sqrt(nb[k]["n"]) for k in keys]
    for i, k in enumerate(keys):
        ax[1].bar(i, means[i], yerr=1.96 * sems[i], color=CVD[i], hatch=HATCH[i], edgecolor="k",
                  capsize=4, ecolor="k")
    ax[1].set_xticks(range(len(keys)))
    ax[1].set_xticklabels([LABEL[k].replace(" (", "\n(").replace(" point", "") for k in keys],
                          fontsize=8)
    ax[1].set_ylabel("fraction of top-10 neighbours\npredicting the query's own top-1 token")
    ax[1].set_title("label agreement of the natural neighbourhood")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "natural_neighbor_comparison.png"), dpi=150)
    plt.close(fig)

    cont = d["continuations"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(cont))
    pref = [c["C_common_prefix_tokens"] for c in cont]
    ax.bar(x, pref, color=CVD[0], hatch="//", edgecolor="k")
    ax.axhline(1, **{"color": "k", "ls": ":", "lw": 1.6})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c['C']!r}\nblock {c['layer']}" for c in cont], fontsize=7)
    ax.set_ylabel("identical greedy tokens across\nthe first / middle / last C alpha")
    ax.set_xlabel("inspected candidate (3 top-scoring, then 3 random)")
    ax.set_title("continuation reproducibility inside the C region (20 tokens decoded)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "continuation_stability.png"), dpi=150)
    plt.close(fig)
    print("wrote natural_neighbor_comparison.png, continuation_stability.png")


if __name__ == "__main__":
    main()
