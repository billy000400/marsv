"""Figures for the context control (context_sweep.py)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvd_style import CVD, REF_DIAG, REF_RULE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(ROOT, "results", "context_sweep_summary.json")))
PLOTS = os.path.join(ROOT, "plots")
DIAG, STRICT = 0.80, 0.25

ctx = S["per_context"]
W = [np.array(d["widths"]) for d in ctx]
pc = np.array([d["p_comma"] for d in ctx])
med = np.array([d["median_w"] for d in ctx])
rho = np.array([d["spearman_rho_w_vs_pnext"] for d in ctx])
is_ref = np.array([d["is_reference"] for d in ctx])
order = np.argsort(pc)                      # contexts ordered by how likely a comma is there
lab = [("ref" if ctx[i]["is_reference"] else f"C{i}") + f"\n{pc[i]:.0e}" for i in order]

# ---- Figure 1: width distribution per context + median vs p(comma) -------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
bp = ax[0].boxplot([W[i] for i in order], widths=0.6, patch_artist=True,
                   medianprops=dict(color="k", lw=2), flierprops=dict(marker=".", ms=4))
for j, i in enumerate(order):
    bp["boxes"][j].set(facecolor=CVD[1] if is_ref[i] else CVD[0], alpha=0.75,
                       hatch="xx" if is_ref[i] else "//", edgecolor="k")
ax[0].axhline(DIAG, label=f"straight line, no plateau ({DIAG}) — dashed", **REF_DIAG)
ax[0].axhline(STRICT, label=f"strict plateau rule (≤ {STRICT}) — dotted", **REF_RULE)
ax[0].set_xticklabels(lab, fontsize=8)
ax[0].set_xlabel("context (ordered by the model's probability of a comma there, shown below each box;\n"
                 "\"ref\" = \"The house was \", the context used for every earlier plateau number)")
ax[0].set_ylabel("transition width $w_{10\\to90}$")
ax[0].set_ylim(0, 0.9)
ax[0].set_title("Every context gives plateau-shaped curves, none linear\n(64 comma→character pairs per context)")
ax[0].legend(fontsize=8, loc="upper left")

ax[1].plot(pc[~is_ref], med[~is_ref], "o", color=CVD[0], ms=9, label="held-out contexts (circles)")
ax[1].plot(pc[is_ref], med[is_ref], "D", color=CVD[1], ms=11,
           label="\"The house was \" reference (diamond)")
ax[1].set_xscale("log")
ax[1].axhline(DIAG, label=f"straight line ({DIAG}) — dashed", **REF_DIAG)
ax[1].axhline(STRICT, label=f"strict plateau rule (≤ {STRICT}) — dotted", **REF_RULE)
ax[1].set_xlabel("model probability of a comma in that context (log scale)")
ax[1].set_ylabel("median transition width over the 64 pairs")
ax[1].set_ylim(0, 0.9)
ax[1].set_title("Sharpness does not track how plausible the fixed endpoint is\n"
                f"Spearman ρ = {S['rho_medianw_vs_pcomma']:.2f} "
                f"(p = {S['rho_medianw_vs_pcomma_p']:.2f}, n = {len(ctx)})")
ax[1].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "context_widths.png"), dpi=130)
plt.close(fig)

# ---- Figure 2: does the width-vs-probability predictor replicate? --------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
y = np.arange(len(ctx))
for j, i in enumerate(order):
    ax[0].barh(j, rho[i], color=CVD[1] if is_ref[i] else CVD[0], alpha=0.8,
               hatch="xx" if is_ref[i] else "//", edgecolor="k")
ax[0].axvline(0, color="k", lw=1)
ax[0].axvline(np.median(rho), color="k", ls="-.", lw=2,
              label=f"median over contexts = {np.median(rho):.2f} (dash-dot)")
ax[0].set_yticks(y); ax[0].set_yticklabels(lab, fontsize=8)
ax[0].set_xlabel("Spearman ρ between transition width and the model's probability of the target character")
ax[0].set_ylabel("context (ordered by p(comma), as in Figure above)")
ax[0].set_title("The 'sharper for expected characters' effect replicates in sign\n"
                "(9/9 contexts negative) but its size varies a lot")
ax[0].legend(fontsize=8, loc="lower left")

for i in range(len(ctx)):
    p = np.maximum(np.array(ctx[i]["p_next"]), 1e-24)
    if is_ref[i]:
        continue
    ax[1].scatter(p, W[i], s=13, color=CVD[0], marker="o", alpha=0.45, edgecolor="none")
i_ref = int(np.where(is_ref)[0][0])
ax[1].scatter(np.maximum(np.array(ctx[i_ref]["p_next"]), 1e-24), W[i_ref], s=34,
              color=CVD[1], marker="D", edgecolor="w", lw=0.4,
              label="\"The house was \" reference (diamonds)")
ax[1].scatter([], [], s=13, color=CVD[0], marker="o", label="8 held-out contexts (circles)")
ax[1].set_xscale("log")
ax[1].axhline(DIAG, label=f"straight line ({DIAG}) — dashed", **REF_DIAG)
ax[1].axhline(STRICT, label=f"strict plateau rule (≤ {STRICT}) — dotted", **REF_RULE)
ax[1].set_xlabel("model probability of the target character in its context (log scale)")
ax[1].set_ylabel("transition width $w_{10\\to90}$")
ax[1].set_ylim(0, 0.9)
ax[1].set_title(f"All {S['pooled_n_pairs']} pairs pooled: ρ = {S['pooled_rho_w_vs_pnext']:.2f}, "
                f"median width {S['pooled_median_w']:.2f}, 0 linear")
ax[1].legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "context_rho.png"), dpi=130)
plt.close(fig)
print("saved plots/context_widths.png, plots/context_rho.png")
