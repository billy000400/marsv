"""fig13 — paired-bootstrap 95% CIs on the Phase-6b manifold-repair KL deltas."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); PLOTS = os.path.join(ROOT, "plots")

rows = json.load(open(os.path.join(RES, "manifold_repair_ci_summary.json")))["rows"]
labels = [r["comparison"].replace(" vs ", "\nvs ") for r in rows]
delta = np.array([r["delta_KL"] for r in rows])
lo = np.array([r["ci_lo"] for r in rows]); hi = np.array([r["ci_hi"] for r in rows])
sig = [r["sig_treatment_better"] for r in rows]
err = np.vstack([delta - lo, hi - delta])
colors = ["#2a8f4f" if s else "#b0b0b0" for s in sig]

fig, ax = plt.subplots(figsize=(8.2, 4.4))
x = np.arange(len(rows))
ax.bar(x, delta, color=colors, width=0.6, zorder=2)
ax.errorbar(x, delta, yerr=err, fmt="none", ecolor="black", elinewidth=1.4, capsize=5, zorder=3)
ax.axhline(0, color="black", lw=1)
for i, r in enumerate(rows):
    ax.text(i, hi[i] + 0.02, ("sig" if r["sig_treatment_better"] else "n.s."),
            ha="center", va="bottom", fontsize=9,
            color="#2a8f4f" if r["sig_treatment_better"] else "#888")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel(r"$\Delta$KL = KL(reference) $-$ KL(kNN step)")
ax.set_title("Phase 6b: paired bootstrap (N=300, B=5000) — kNN manifold step vs matched random / corrupted\n"
             r"$\Delta$>0 (green) = kNN step has lower downstream KL; error bars = 95% CI", fontsize=9.5)
ax.margins(y=0.18)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig13_manifold_repair_ci.png"), dpi=130)
plt.close(fig)
print("wrote fig13_manifold_repair_ci.png")
