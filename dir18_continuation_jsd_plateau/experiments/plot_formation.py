"""Formation figure: when during training does the JSD -> sharpness relationship appear?"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import CVD, LS, MARK, arr, boot_spearman
from common import PLOTS, RESULTS

STEPS = [0, 1000, 8000, 32000, 64000, 143000]
XPOS = [300, 1000, 8000, 32000, 64000, 143000]  # step 0 placed left of the log axis

if __name__ == "__main__":
    rows, xs = [], []
    for s, x in zip(STEPS, XPOS):
        p = os.path.join(RESULTS, f"assay_step{s}.json")
        if not os.path.exists(p):
            continue
        A = json.load(open(p))
        j, w = arr(A["rows"], "jsd_B"), arr(A["rows"], "w")
        oj = arr(A["rows"], "out_jsd_med")
        r, lo, hi, n, _ = boot_spearman(j, w)
        ro = boot_spearman(j, oj)[0]
        rows.append(dict(step=s, rho=r, lo=lo, hi=hi, median_w=float(np.nanmedian(w)),
                         iqr_w=float(np.nanpercentile(w, 75) - np.nanpercentile(w, 25)),
                         rho_outjsd=ro))
        xs.append(x)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.1))
    rho = [r["rho"] for r in rows]
    lo = [r["lo"] for r in rows]
    hi = [r["hi"] for r in rows]
    ax[0].fill_between(xs, lo, hi, color=CVD[0], alpha=0.18)
    ax[0].plot(xs, rho, ls=LS[0], marker=MARK[0], color=CVD[0], lw=1.7,
               label=r"$\rho(JSD_B,\ w)$  (sharpness)")
    ax[0].plot(xs, [r["rho_outjsd"] for r in rows], ls=LS[1], marker=MARK[1], color=CVD[1], lw=1.7,
               label=r"$\rho(JSD_B,\ JSD_{\mathrm{out}})$  (output)")
    ax[0].axhline(0, color="0.5", lw=0.8, ls=":")
    ax[0].set_xscale("log")
    ax[0].set_xticks(XPOS)
    ax[0].set_xticklabels(["0", "1k", "8k", "32k", "64k", "143k"])
    ax[0].set_xlabel("training step (log scale; step 0 drawn at the left edge)")
    ax[0].set_ylabel(r"Spearman $\rho$ with corpus $JSD_B$")
    ax[0].set_title("When the corpus predictor starts to work")
    ax[0].legend(frameon=False, fontsize=8, loc="center left")

    ax[1].plot(xs, [r["median_w"] for r in rows], ls=LS[0], marker=MARK[0], color=CVD[0], lw=1.7,
               label="median $w$")
    ax[1].fill_between(xs, [r["median_w"] - r["iqr_w"] / 2 for r in rows],
                       [r["median_w"] + r["iqr_w"] / 2 for r in rows],
                       color=CVD[0], alpha=0.18, hatch="//", label="median $\\pm$ IQR/2")
    ax[1].axhline(0.8, color=CVD[1], lw=1.2, ls=LS[1], label="linear response ($w = 0.8$)")
    ax[1].set_xscale("log")
    ax[1].set_xticks(XPOS)
    ax[1].set_xticklabels(["0", "1k", "8k", "32k", "64k", "143k"])
    ax[1].set_xlabel("training step (log scale; step 0 drawn at the left edge)")
    ax[1].set_ylabel("transition width $w$ (smaller = sharper)")
    ax[1].set_title("Plateaus sharpen throughout training")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "formation.png"))
    plt.close(fig)

    json.dump(rows, open(os.path.join(RESULTS, "formation.json"), "w"), indent=2)
    for r in rows:
        print(f"step {r['step']:>6}: rho={r['rho']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]  "
              f"median w={r['median_w']:.3f} (IQR {r['iqr_w']:.3f})  rho_out={r['rho_outjsd']:+.3f}")
