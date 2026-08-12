"""Forest plot: the amended analysis against the pre-registered independent replication.

Reads results/matched_metrics.json (amended, wikitext test split) and
results/matched_metrics_rep.json (replication, wikitext train split, bank size fixed in
advance) and draws each one's median Delta w with its bootstrap 95% CI against the gate
threshold. Writes plots/replication_forest.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def main():
    am = json.load(open(os.path.join(RESULTS, "matched_metrics.json")))
    rep = json.load(open(os.path.join(RESULTS, "matched_metrics_rep.json")))
    rows = [
        ("Amended analysis\n(test split, n=%d)" % am["n_contrasts"], am, CVD[0], "o"),
        ("Independent replication\n(train split, n=%d)" % rep["n_contrasts"], rep, CVD[1], "s"),
    ]

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.2))

    # --- forest ------------------------------------------------------------------
    for k, (lab, s, col, mk) in enumerate(rows):
        y = len(rows) - 1 - k
        lo, hi = s["ci95"]
        ax[0].plot([lo, hi], [y, y], "-", color=col, lw=2.4)
        ax[0].plot([s["median_dw"]], [y], mk, color=col, ms=9)
        ax[0].text(0.012, y + 0.16, f"{s['median_dw']:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                   va="center", fontsize=9, color=col)
    ax[0].axvline(0, color="0.35", ls=":", lw=1.5)
    ax[0].axvline(-0.05, color="0.35", ls="--", lw=1.5)
    ax[0].text(-0.049, -0.5, "gate:\nmedian $\\leq -0.05$", fontsize=8, color="0.3")
    ax[0].set_yticks(range(len(rows)))
    ax[0].set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
    ax[0].set_ylim(-0.75, len(rows) - 0.2)
    ax[0].set_xlim(min(r[1]["ci95"][0] for r in rows) - 0.02, 0.075)
    ax[0].set_xlabel(r"median $\Delta w = w_{TV}(\mathrm{high}\,F) - w_{TV}(\mathrm{low}\,F)$")
    ax[0].set_title("Effect size and 95% CI in both banks")

    # --- median widths per group --------------------------------------------------
    x = np.arange(2)
    w = 0.34
    for k, (lab, s, col, mk) in enumerate(rows):
        vals = [s["w_tv_low"][0], s["w_tv_high"][0]]
        ax[1].bar(x + (k - 0.5) * w, vals, w, color=col,
                  hatch=["//", "xx"][k], edgecolor="k", lw=0.6,
                  label=lab.replace("\n", " "))
        for xi, v in zip(x + (k - 0.5) * w, vals):
            ax[1].text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax[1].axhline(0.5, color="0.35", ls=":", lw=1.5)
    ax[1].text(1.35, 0.505, "linear response", fontsize=8, color="0.3")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["low-$F$ member", "high-$F$ member"])
    ax[1].set_ylabel(r"median transition width $w_{TV}$")
    ax[1].set_ylim(0, 0.56)
    ax[1].set_title("Median width by group in each bank")
    ax[1].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "replication_forest.png"), dpi=150)
    plt.close(fig)
    print("wrote plots/replication_forest.png")
    for lab, s, _, _ in rows:
        print(f"{lab.splitlines()[0]}: n={s['n_contrasts']} med={s['median_dw']:+.4f} "
              f"CI={s['ci95']} frac={s['frac_predicted_sign']:.3f} "
              f"p={s['perm_p']:.4f} verdict={s['verdict']}")


if __name__ == "__main__":
    main()
