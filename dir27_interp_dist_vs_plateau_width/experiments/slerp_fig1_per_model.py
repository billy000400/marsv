"""Feedback 7 (repair): per-model Figure 1 analogue for the SLERP-path sweeps (slerp_sweep.py).

Same layout and style as the linear Figure 1 (s3_plots.py / gen_plots.py): panel (a) middle block, panel (b) final
block, flagged (non-monotonic) curves as crosses, clean curves as dots. x = length of the injected SLERP path,
y = transition width measured along that path. Panels whose x range is stretched by a few far points show only the
majority region (X_CROP); the number of tokens not drawn is printed on the panel.
Reads results/block0_resid_post/slerp/<key>.csv; writes plots/block0_resid_post/slerp/<key>_fig1_slerp_distance_vs_width.png.

Usage: python experiments/slerp_fig1_per_model.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CVD, PLOTS, RESULTS

MODELS = [("gpt2-large", "GPT-2 Large", 18, 35), ("gpt2-xl", "GPT2-XL", 24, 47),
          ("pythia-1.4b", "Pythia-1.4B", 12, 23), ("llama-3.1-8b", "Llama-3.1-8B", 16, 31),
          ("qwen3-8b-base", "Qwen3-8B-Base", 18, 35)]
# Linear Figure 1 crops (Llama D 1-2.5, Qwen3 D 10-55) stretched by the ~1.1 median ratio of path length to D
X_CROP = {"llama-3.1-8b": (1.0, 2.75), "qwen3-8b-base": (10, 60)}
XLABEL = "SLERP path length from ' big' (block-0 resid_post, L2)"
OUT = os.path.join(PLOTS, "slerp")


def main():
    os.makedirs(OUT, exist_ok=True)
    for key, name, bm, bf in MODELS:
        rows = list(csv.DictReader(open(os.path.join(RESULTS, "slerp", f"{key}.csv"), newline="")))
        L, Wm, Wf = (np.array([float(r[c]) for r in rows]) for c in ("L_slerp", "W_mid", "W_final"))
        flag = np.array([r["flag_nonmonotonic"] == "1" for r in rows])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
        for ax, W, t in zip(axes, (Wm, Wf), (f"(a) block {bm} (middle)", f"(b) block {bf} (final)")):
            ax.scatter(L[flag], W[flag], s=4, alpha=0.25, color=CVD[1], marker="x", lw=0.4,
                       label="flagged (non-monotonic)")
            ax.scatter(L[~flag], W[~flag], s=2, alpha=0.08, color=CVD[0], marker="o", lw=0, label="clean curve")
            ax.set_title(f"{name} {t}, SLERP path")
            ax.set_xlabel(XLABEL)
        axes[0].set_ylabel("transition width W = t$_{0.9}$ − t$_{0.1}$")
        if key in X_CROP:
            lo, hi = X_CROP[key]
            axes[0].set_xlim(lo, hi)
            n_out = int(((L < lo) | (L > hi)).sum())
            for ax in axes:
                ax.text(0.98, 0.02, f"{n_out} of {len(L)} tokens outside this x range, not drawn",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
            print(f"{key}: x range {lo}-{hi}, {n_out} of {len(L)} not drawn; L min {L.min():.2f} max {L.max():.2f}")
        leg = axes[0].legend(loc="upper right", markerscale=4)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{key}_fig1_slerp_distance_vs_width.png"), dpi=110)
        plt.close(fig)
        print(f"{key}: n {len(L)}, flagged {flag.mean():.1%}, median W_mid {np.median(Wm):.3f}, W_final {np.median(Wf):.3f}")


if __name__ == "__main__":
    main()
