"""Figure for the MLP-gain intervention (mlp_gain_probe.py).

Left: dose-response of the transition width w_{10->90} against the MLP gain g, for the early
blocks 1-4 and the late blocks 8-11 (median + IQR band). Right: paired per-pair change in w
relative to the unmodified model, one box per condition.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD, use_cvd, REF_DIAG, REF_RULE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
INIT_W = 0.803   # median width at the step-0 checkpoint (Experiment 5 learned-vs-init control)

use_cvd()
S = json.load(open(os.path.join(RES, "mlp_gain_summary.json")))["summary"]
raw = np.load(os.path.join(RES, "mlp_gain_raw.npz"))
C = S["conditions"]
gains = [0.0, 0.5, 1.0, 1.5]

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

for k, (tag, label, ls, mk, col) in enumerate([
        ("early", "MLP gain on blocks 1-4", "-", "o", CVD[0]),
        ("late", "MLP gain on blocks 8-11", "--", "s", CVD[1])]):
    med, lo, hi = [], [], []
    for g in gains:
        name = "baseline" if g == 1.0 else f"{tag}_g{g}"
        med.append(C[name]["median_w"])
        lo.append(C[name]["iqr_w"][0])
        hi.append(C[name]["iqr_w"][1])
    ax[0].fill_between(gains, lo, hi, color=col, alpha=0.16, lw=0)
    ax[0].plot(gains, med, ls=ls, marker=mk, color=col, lw=2, label=label)
    ax[0].annotate(label.split(" on ")[1], (gains[-1], med[-1]), textcoords="offset points",
                   xytext=(6, -2 + 8 * k), fontsize=8, color=col)

ax[0].axhline(INIT_W, **REF_DIAG)
ax[0].text(0.02, INIT_W - 0.035, "untrained model (step 0), w = 0.803", fontsize=8, color="0.35")
ax[0].axhline(0.25, **REF_RULE)
ax[0].text(0.02, 0.20, "strict plateau rule, w <= 0.25", fontsize=8)
ax[0].axvline(1.0, color="0.7", lw=1, zorder=0)
ax[0].set_xlabel("MLP-branch gain $g$ (1.0 = unmodified model)")
ax[0].set_ylabel("transition width $w_{10\\to90}$")
ax[0].set_title("A. Scaling the early MLPs sets the sharpness")
ax[0].set_ylim(0.1, 0.95)
ax[0].legend(fontsize=8, loc="center right")

names = [f"early_g{g}" for g in [0.0, 0.5, 1.5]] + [f"late_g{g}" for g in [0.0, 0.5, 1.5]]
data = [raw[n + "_w"] - raw["baseline_w"] for n in names]
bp = ax[1].boxplot(data, vert=True, widths=0.6, showfliers=False, patch_artist=True,
                   medianprops=dict(color="k", lw=1.6))
for i, box in enumerate(bp["boxes"]):
    box.set_facecolor(CVD[0] if i < 3 else CVD[1])
    box.set_alpha(0.55)
    box.set_hatch("//" if i < 3 else "..")
ax[1].axhline(0.0, color="0.4", lw=1.2, ls=":")
ax[1].set_xticklabels(["1-4\ng=0", "1-4\ng=0.5", "1-4\ng=1.5",
                       "8-11\ng=0", "8-11\ng=0.5", "8-11\ng=1.5"], fontsize=8)
ax[1].set_xlabel("intervened blocks and gain (hatched //: early, ..: late)")
ax[1].set_ylabel("paired change in width  $\\Delta w$")
ax[1].set_title("B. Per-pair change vs the unmodified model")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "mlp_gain_intervention.png"), dpi=150)
plt.close(fig)
print("wrote plots/mlp_gain_intervention.png")
