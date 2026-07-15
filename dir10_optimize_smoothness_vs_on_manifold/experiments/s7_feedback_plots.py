"""S7 feedback plots (no model) — addresses human_feedback_07140930.md points 1-3.

1. Regenerate the PCA scatter with an EXPLICIT legend explaining the star markers
   (= the 7 ground-truth weekday centroids) vs. dots (= the 49 individual prompt
   activations, 7 per weekday).
3. Cumulative-explained-variance ("scree") plot vs. number of PCs, to show whether
   2-3 PCs are representative of the layer-28 weekday geometry.

All from saved arrays in results/weekday_setup.npz — no GPU, no model.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import common as C

setup = np.load(C.RESULTS + "/weekday_setup.npz")
Z = setup["Z"]                       # [49, 48] PCA coords
centroids = setup["centroids"]       # [7, 48]
gt = setup["gt_idx"]                 # [49]
ev = setup["explained_variance"]     # [48]
spline_pts = setup["spline_pts"]     # dense periodic spline in PCA coords

ratio = ev / ev.sum()
cum = np.cumsum(ratio)

# ---- Fig A: PCA scatter with explicit star legend ----
cmap = plt.cm.hsv
fig, ax = plt.subplots(figsize=(7.4, 6.0))
for d in range(7):
    m = gt == d
    ax.scatter(Z[m, 0], Z[m, 1], color=cmap(d / 7), s=30, alpha=0.85,
               edgecolor="none", label=None)
    ax.scatter(centroids[d, 0], centroids[d, 1], color=cmap(d / 7),
               s=210, marker="*", edgecolor="k", linewidth=0.9, zorder=5)
    ax.annotate(C.WEEKDAYS[d][:3], (centroids[d, 0], centroids[d, 1]),
                fontsize=9, weight="bold", ha="center", va="center",
                xytext=(0, 14), textcoords="offset points")
ax.plot(spline_pts[:, 0], spline_pts[:, 1], "-", color="0.35", lw=1.3,
        alpha=0.8, label="periodic cubic spline")
# legend proxies
from matplotlib.lines import Line2D
proxies = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markersize=8, label="individual prompt activation (7 per weekday)"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
           markeredgecolor="k", markersize=15,
           label="weekday centroid = mean of its 7 prompts (★)"),
    Line2D([0], [0], color="0.35", lw=1.3, label="fitted periodic cubic spline"),
]
ax.legend(handles=proxies, fontsize=8.5, loc="best")
ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}% var)")
ax.set_title("Weekday activations @ layer 28 (PCA)\n"
             "★ = 7 ground-truth weekday centroids; ● = 49 prompts (7/weekday)")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(C.PLOTS + "/s2_pca_weekday_manifold.png", dpi=130)
plt.close(fig)

# ---- Fig B: cumulative explained variance ----
fig, ax = plt.subplots(figsize=(7.2, 5.0))
k = np.arange(1, len(ratio) + 1)
ax.bar(k, ratio * 100, color="#8aa", alpha=0.6, label="per-PC variance %")
ax2 = ax.twinx()
ax2.plot(k, cum * 100, "o-", color="#b2182b", ms=3, label="cumulative %")
for thr, txt in [(2, cum[1]), (3, cum[2]), (10, cum[9]), (32, cum[31])]:
    ax2.annotate(f"{txt*100:.0f}%", (thr, cum[thr-1]*100),
                 fontsize=8, color="#b2182b",
                 xytext=(2, 6), textcoords="offset points")
    ax2.axvline(thr, color="k", ls=":", lw=0.6, alpha=0.4)
ax.set_xlabel("number of principal components")
ax.set_ylabel("per-PC explained variance (%)")
ax2.set_ylabel("cumulative explained variance (%)", color="#b2182b")
ax2.set_ylim(0, 101)
ax.set_title("Layer-28 weekday activations: explained variance vs. #PCs\n"
             "PC1–2 capture only ~31%; PC1–3 ~44% — the geometry is high-dim")
ax.legend(loc="center right", fontsize=8)
ax2.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(C.PLOTS + "/s2_pca_cumvar.png", dpi=130)
plt.close(fig)

print(f"PC1-2 cumvar = {cum[1]*100:.1f}%   PC1-3 = {cum[2]*100:.1f}%   "
      f"PC1-10 = {cum[9]*100:.1f}%   PC1-32 = {cum[31]*100:.1f}%")
print(f"#PCs for 90% var = {int(np.searchsorted(cum, 0.90)+1)}")
print("saved s2_pca_weekday_manifold.png (relabeled) + s2_pca_cumvar.png")
