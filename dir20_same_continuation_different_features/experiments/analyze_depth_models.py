"""S15: does the top-of-stack collapse of the plateau reproduce outside gpt2-large?

depth_curve.py was re-run with MKEY=gpt2-small / gpt2-medium and SITES=0,1,2,3,4. The head-set
ablation is under-powered in those two models (block-0 effects +0.015 / +0.005), so the primary
readout here is the UNABLATED switch width w_TV as the patch moves down one block at a time, plus
the fraction of the available headroom (0.5 - w_TV at block 0) that the first b blocks close.
Writes plots/depth_models.png and results/depth_models.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

FILES = [("gpt2-large", "depth_curve.json", "^", ":"),
         ("gpt2-medium", "depth_curve_gpt2-medium.json", "s", "--"),
         ("gpt2-small", "depth_curve_gpt2-small.json", "o", "-")]
SITES = [0, 1, 2, 3, 4]
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def main():
    out = {}
    for mk, fn, _, _ in FILES:
        s = json.load(open(os.path.join(RESULTS, fn)))["sites"]
        base = [s[str(L)]["stats"]["median_wtv"]["base"] for L in SITES]
        head = 0.5 - base[0]
        out[mk] = dict(
            n_block=json.load(open(os.path.join(RESULTS, fn)))["meta"]["n_block"],
            base=base, headroom=head,
            closed=[(b - base[0]) / head for b in base],
            delta=[s[str(L)]["stats"]["median_delta"] for L in SITES],
            ci=[s[str(L)]["stats"]["ci_delta"] for L in SITES],
            p=[s[str(L)]["stats"]["wilcoxon_p"] for L in SITES],
            max_err=max(s[str(L)]["stats"]["max_endpoint_err"] for L in SITES))
    json.dump(out, open(os.path.join(RESULTS, "depth_models.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for i, (mk, _, mrk, ls) in enumerate(FILES):
        r = out[mk]
        lab = f"{mk} ({r['n_block']} blocks)"
        ax[0].plot(SITES, r["base"], color=CVD[i], marker=mrk, ls=ls, lw=2.2, ms=8, mec="k",
                   mew=0.6, label=lab)
        ax[1].plot(SITES, 100 * np.array(r["closed"]), color=CVD[i], marker=mrk, ls=ls, lw=2.2,
                   ms=8, mec="k", mew=0.6, label=lab)
    ax[0].axhline(0.5, color="0.35", ls=(0, (4, 3)), lw=1.4)
    ax[0].text(0.05, 0.508, "linear response 0.5", fontsize=8, color="0.35")
    ax[0].set_ylim(0.15, 0.56)
    ax[0].set_ylabel("median transition width $w_{TV}$ (no ablation)")
    ax[0].set_title("A. The switch weakens with every block removed\nfrom the path below the patch",
                    fontsize=10)
    ax[1].set_ylabel("% of the block-0 headroom closed")
    ax[1].set_title("B. How much of the available compression\nthe top blocks supply", fontsize=10)
    for a in ax:
        a.set_xticks(SITES)
        a.set_xlabel("patch site (block index $L$); $L=0$ is the full stack below the patch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("60 low-JSD pairs per model, block-0 SLERP patch moved down one block at a time",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(PLOTS, "depth_models.png"), dpi=140)
    plt.close(fig)
    for mk in out:
        print(mk, "closed@4 = %.1f%%  closed@1 = %.1f%%  worst err %.1e"
              % (100 * out[mk]["closed"][4], 100 * out[mk]["closed"][1], out[mk]["max_err"]))


if __name__ == "__main__":
    main()
