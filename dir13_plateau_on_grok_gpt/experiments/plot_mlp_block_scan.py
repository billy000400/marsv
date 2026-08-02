"""Figure for the per-block MLP ablation scan (mlp_block_scan.py).

Left:   which of blocks 1-4 carries the sharpness -- median w (IQR bar) for the unmodified model,
        each single-block MLP deletion, and all four deleted, against the untrained reference.
Middle: the mediation test -- per-pair widening dw against the per-pair change in the endpoint
        plausibility max(p(A),p(B)), for the all-four deletion (the largest effect available).
Right:  does the decision structure survive? fraction of pairs whose two endpoints still predict
        different next characters, and the median |t* - t_flip| gap, per condition.
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
S = json.load(open(os.path.join(RES, "mlp_block_scan_summary.json")))["summary"]
raw = np.load(os.path.join(RES, "mlp_block_scan_raw.npz"))
C = S["conditions"]

CONDS = ["baseline", "block1_g0", "block2_g0", "block3_g0", "block4_g0", "early_all_g0"]
TICKS = ["none\n(unmodified)", "block 1", "block 2", "block 3", "block 4", "blocks\n1-4"]
x = np.arange(len(CONDS))

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.5))

# ---- left: median width per condition ---------------------------------------------------------
med = np.array([C[c]["median_w"] for c in CONDS])
lo = np.array([C[c]["iqr_w"][0] for c in CONDS])
hi = np.array([C[c]["iqr_w"][1] for c in CONDS])
ax[0].errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", ms=8, lw=2, capsize=5,
               color=CVD[0], label="median $w_{10\\to90}$ (bar = IQR)")
ax[0].axhline(INIT_W, **REF_DIAG)
ax[0].axhline(med[0], **REF_RULE)
ax[0].annotate("untrained model (step 0): 0.803", (len(CONDS) - 1, INIT_W), ha="right",
               va="bottom", fontsize=8, color="0.35")
ax[0].annotate(f"unmodified: {med[0]:.3f}", (0.05, med[0]), ha="left", va="bottom", fontsize=8)
for i in range(1, len(CONDS)):
    ax[0].annotate(f"{C[CONDS[i]]['frac_of_group_effect']*100:.0f}%", (x[i], hi[i]),
                   textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8,
                   color=CVD[1])
ax[0].set_xticks(x)
ax[0].set_xticklabels(TICKS, fontsize=8)
ax[0].set_xlabel("MLP branch deleted (gain $g=0$) in block(s)")
ax[0].set_ylabel("transition width $w_{10\\to90}$")
ax[0].set_ylim(0.25, 0.92)
ax[0].set_title("A. No single early block carries the sharpness")
ax[0].legend(loc="upper left", fontsize=8)

# ---- middle: mediation test ------------------------------------------------------------------
dw = raw["early_all_g0_w"] - raw["baseline_w"]
dmaxp = raw["early_all_g0_maxp"] - raw["baseline_maxp"]
ax[1].scatter(dmaxp, dw, s=22, color=CVD[0], marker="o", alpha=0.65, edgecolor="none",
              label="one pair (blocks 1-4 deleted)")
ax[1].axhline(0.0, **REF_DIAG)
ax[1].axvline(0.0, color="0.45", ls=":", lw=1.2)
ax[1].set_xlabel(r"change in endpoint plausibility $\Delta\max(p_A,p_B)$")
ax[1].set_ylabel(r"change in width $\Delta w_{10\to90}$")
ax[1].set_title("B. The widening does not track plausibility\n"
                fr"Spearman $\rho$ = {C['early_all_g0']['rho_dw_vs_dmax_p']:.2f}")
ax[1].legend(loc="upper right", fontsize=8)

# ---- right: decision structure ----------------------------------------------------------------
differ = np.array([C[c]["frac_endpoints_differ"] for c in CONDS])
gap = np.array([C[c]["median_abs_tstar_minus_tflip"] for c in CONDS])
ax[2].plot(x, differ, ls="-", marker="o", lw=2, color=CVD[0],
           label="endpoints predict different characters")
ax[2].set_ylim(0.0, 1.05)
ax[2].set_ylabel("fraction of the 150 pairs", color=CVD[0])
ax[2].tick_params(axis="y", labelcolor=CVD[0])
ax2 = ax[2].twinx()
ax2.plot(x, gap, ls="--", marker="s", lw=2, color=CVD[1],
         label=r"median $|t^* - t_{\mathrm{flip}}|$")
ax2.set_ylim(0.0, 0.30)
ax2.set_ylabel(r"median $|t^* - t_{\mathrm{flip}}|$ (dashed, squares)", color=CVD[1])
ax2.tick_params(axis="y", labelcolor=CVD[1])
ax[2].set_xticks(x)
ax[2].set_xticklabels(TICKS, fontsize=8)
ax[2].set_xlabel("MLP branch deleted (gain $g=0$) in block(s)")
ax[2].set_title("C. The decision survives; its alignment with $t^*$ does not")
h1, l1 = ax[2].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, loc="center left", fontsize=8)

for a in ax:
    a.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "mlp_block_scan.png"), dpi=150)
plt.close(fig)
print("wrote plots/mlp_block_scan.png")
