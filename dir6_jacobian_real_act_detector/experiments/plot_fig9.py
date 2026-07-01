"""Render fig9 (Phase 2c sink-confound control) from results/position_stratified_metrics.csv.

Standalone so the figure is reproducible from the cached CSV (the original generator was not persisted).
Grouped bars: per-baseline MACRO AUROC, WITHSINK vs NONSINK — the headline being norm/mean_l2 collapse
while Mahalanobis/kNN are robust.
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); PLOTS = os.path.join(ROOT, "plots")

rows = list(csv.DictReader(open(os.path.join(RES, "position_stratified_metrics.csv"))))
macro = {c: {} for c in ("WITHSINK", "NONSINK")}
for r in rows:
    if r["family"] == "MACRO":
        macro[r["condition"]][r["baseline"]] = float(r["auroc"])

bases = ["norm", "mean_l2", "mahalanobis", "pca_recon", "coord_quantile", "knn_distance"]
x = np.arange(len(bases)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x - w / 2, [macro["WITHSINK"][b] for b in bases], w, label="WITHSINK (all positions)", color="#4C72B0")
ax.bar(x + w / 2, [macro["NONSINK"][b] for b in bases], w, label="NONSINK (pos≥1)", color="#DD8452")
ax.axhline(0.5, color="crimson", ls="--", lw=1, label="chance")
for xi, b in zip(x, bases):
    ax.annotate(f'{macro["WITHSINK"][b]:.2f}', (xi - w / 2, macro["WITHSINK"][b]), ha="center", va="bottom", fontsize=8)
    ax.annotate(f'{macro["NONSINK"][b]:.2f}', (xi + w / 2, macro["NONSINK"][b]), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(bases, rotation=20, ha="right")
ax.set_ylabel("macro AUROC"); ax.set_ylim(0.4, 1.02)
ax.set_title("Phase 2c: sink-confound control (document-level split)\nnorm/mean_l2 collapse; Mahalanobis/kNN robust")
ax.legend(loc="lower left", fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig9_position_stratified.png"), dpi=110)
plt.close(fig)
print("wrote plots/fig9_position_stratified.png")
