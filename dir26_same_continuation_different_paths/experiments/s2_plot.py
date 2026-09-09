"""S2 plot: d(t) for the one pair that passed the completion screen."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def main():
    os.makedirs(PLOTS, exist_ok=True)
    z = np.load(os.path.join(RESULTS, "p1_copy_vs_recall_interp.npz"))
    t, d = z["t"], z["d"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(t, d, "-o", color=CVD[0], ms=3.5, lw=1.8, label="d(t)")
    ax.axhline(0.5, color="0.55", ls=":", lw=1.2)
    ax.text(0.02, 0.52, "midpoint d = 0.5", color="0.35", fontsize=8)
    ax.set_xlabel("interpolation position t  (0 = prompt A endpoint, 1 = prompt B endpoint)")
    ax.set_ylabel("relative logit distance d(t)")
    ax.set_title("Pair 1: copy-available vs factual-recall, GPT-2 Large, block 0")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "p1_copy_vs_recall_dt.png"), dpi=160)
    plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
