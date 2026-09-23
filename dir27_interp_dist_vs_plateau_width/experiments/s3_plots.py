"""S3/S4: aggregate sweep shards -> results/sweep.csv; Figure 1 (D vs W) and Figure 2 (W_final hist + sorted)."""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from common import CVD, MODEL, PLOTS, RESULTS

BACKSTEP = 0.05


def load_sweep():
    sh = [np.load(p) for p in sorted(glob.glob(os.path.join(RESULTS, "sweep", "shard_*.npz")))]
    ids = np.concatenate([s["ids"] for s in sh])
    D = np.concatenate([s["D"] for s in sh])
    st = np.concatenate([s["stats"] for s in sh])
    # stats columns: (t10, t90, W, ncross10, ncross90, backstep) for mid, then final
    flag = ((st[:, [3, 4, 9, 10]] > 1).any(1)) | (st[:, [5, 11]] > BACKSTEP).any(1)
    return ids, D, st, flag


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids, D, st, flag = load_sweep()
    Wm, Wf = st[:, 2], st[:, 8]
    with open(os.path.join(RESULTS, "sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token_id", "token", "D", "W_mid", "W_final", "t10_mid", "t90_mid",
                    "t10_final", "t90_final", "flag_nonmonotonic"])
        for i in range(len(ids)):
            w.writerow([ids[i], repr(tok.decode([int(ids[i])])), f"{D[i]:.4f}", f"{Wm[i]:.4f}",
                        f"{Wf[i]:.4f}", f"{st[i, 0]:.4f}", f"{st[i, 1]:.4f}", f"{st[i, 6]:.4f}",
                        f"{st[i, 7]:.4f}", int(flag[i])])
    print(f"{len(ids)} tokens, flagged {flag.sum()} ({flag.mean():.1%})")

    # Figure 1: D vs W at block 18 and block 35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, W, name in ((axes[0], Wm, "(a) block 18 (middle)"), (axes[1], Wf, "(b) block 35 (final)")):
        ax.scatter(D[flag], W[flag], s=4, alpha=0.25, color=CVD[1], marker="x", lw=0.4,
                   label="flagged (non-monotonic)")
        ax.scatter(D[~flag], W[~flag], s=2, alpha=0.08, color=CVD[0], marker="o", lw=0,
                   label="clean curve")
        ax.set_title(name)
        ax.set_xlabel("layer-0 L2 distance D from ' big'")
    axes[0].set_ylabel("transition width W = t$_{0.9}$ − t$_{0.1}$")
    leg = axes[0].legend(loc="upper right", markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig1_distance_vs_width.png"), dpi=130)
    plt.close(fig)

    # Figure 2: W_final histogram and sorted view
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(Wf, bins=200, color=CVD[0])
    axes[0].set_xlabel("final-layer width W_final")
    axes[0].set_ylabel("number of tokens")
    axes[0].set_title("(a) histogram, 200 bins")
    srt = np.sort(Wf)
    axes[1].plot(np.arange(len(srt)), srt, color=CVD[0], lw=1.5)
    axes[1].set_xlabel("token rank (sorted by W_final)")
    axes[1].set_ylabel("final-layer width W_final")
    axes[1].set_title("(b) sorted widths")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig2_final_width_distribution.png"), dpi=130)
    plt.close(fig)

    # quick textual summary by W_final bands
    for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1)):
        sel = (Wf >= lo) & (Wf < hi)
        ex = [tok.decode([int(i)]) for i in ids[sel][np.argsort(D[sel])][:12]]
        print(f"W_final [{lo},{hi}): n={sel.sum()}, median D={np.median(D[sel]) if sel.any() else 0:.2f}, "
              f"lowest-D examples {ex}")


if __name__ == "__main__":
    main()
