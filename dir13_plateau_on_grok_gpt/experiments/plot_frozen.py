"""Figure for the frozen-block training test (train_frozen.py + frozen_assay.py).

Top row:      raw d(t) curves (the primary evidence) for the same 20 pairs under five models -- the
              reference run untrained / at step 2500 / fully trained, and the two frozen-block runs
              at matched validation accuracy.
Bottom left:  median transition width per condition, against the untrained and trained references.
Bottom middle: depth control -- median width by interpolation block for the trained reference and
              the two final frozen models, i.e. whether the sharpening moved to another depth.
Bottom right: validation accuracy across training for the three runs, showing where the frozen runs
              reach the reference run's final accuracy (the "matched" checkpoints assayed above).
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

use_cvd()
S = json.load(open(os.path.join(RES, "frozen_assay_summary.json")))["summary"]
raw = np.load(os.path.join(RES, "frozen_assay_raw.npz"))
C = S["conditions"]
ts = raw["ts"]

CURVE_CONDS = [("ref_init", "reference, untrained (step 0)"),
               ("ref_matched_step", "reference at step 2500"),
               ("ref_trained", "reference, trained (step 30000)"),
               ("frozen_early_matched", "blocks 1-4 frozen, matched acc"),
               ("frozen_late_matched", "blocks 8-11 frozen, matched acc")]
BAR_CONDS = [c for c in ["ref_init", "ref_matched_step", "frozen_early_matched",
                         "frozen_late_matched", "frozen_early_last", "frozen_late_last",
                         "ref_trained"] if c in C]
BAR_TICKS = {"ref_init": "reference\nuntrained", "ref_trained": "reference\ntrained\n(30000)",
             "ref_matched_step": "reference\nstep 2500",
             "frozen_early_matched": "frozen 1-4\nmatched acc", "frozen_early_last": "frozen 1-4\nfinal",
             "frozen_late_matched": "frozen 8-11\nmatched acc", "frozen_late_last": "frozen 8-11\nfinal"}

fig = plt.figure(figsize=(16.5, 8.4))
gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.30)

# ---- top: raw d(t) curves ---------------------------------------------------------------------
for k, (cond, title) in enumerate(CURVE_CONDS):
    ax = fig.add_subplot(gs[0, k])
    if cond not in C:
        ax.set_axis_off()
        continue
    D = raw[cond + "_curves"]
    for row in D:
        ax.plot(ts, row, color=CVD[0], lw=0.9, alpha=0.45)
    ax.plot(ts, np.median(D, axis=0), color=CVD[1], lw=2.6, ls="--", label="median of 20")
    ax.plot([0, 1], [0, 1], **REF_DIAG)
    ax.set_xlabel("interpolation position $t$")
    if k == 0:
        ax.set_ylabel("relative distance $d(t)$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{title}\nmedian $w$ = {C[cond]['median_w']:.3f}", fontsize=9)
    ax.grid(alpha=0.3)
    if k == 0:
        ax.legend(fontsize=7, loc="upper left")
        ax.annotate("straight path\n(no plateau)", (0.62, 0.30), fontsize=7, color="0.35")

# ---- bottom left: median width per condition --------------------------------------------------
axb = fig.add_subplot(gs[1, :2])
x = np.arange(len(BAR_CONDS))
med = np.array([C[c]["median_w"] for c in BAR_CONDS])
lo = np.array([C[c]["iqr_w"][0] for c in BAR_CONDS])
hi = np.array([C[c]["iqr_w"][1] for c in BAR_CONDS])
axb.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", ms=8, lw=2, capsize=5, color=CVD[0],
             label="median $w_{10\\to90}$ (bar = IQR)")
axb.axhline(C["ref_init"]["median_w"], **REF_DIAG)
axb.axhline(C["ref_trained"]["median_w"], **REF_RULE)
axb.annotate(f"untrained reference: {C['ref_init']['median_w']:.3f}",
             (len(BAR_CONDS) - 0.6, C["ref_init"]["median_w"]), ha="right", va="top",
             fontsize=8, color="0.35")
axb.annotate(f"trained reference: {C['ref_trained']['median_w']:.3f}",
             (-0.4, C["ref_trained"]["median_w"]), ha="left", va="bottom", fontsize=8)
for i, c in enumerate(BAR_CONDS):
    axb.annotate(f"{med[i]:.3f}", (x[i], hi[i] + 0.02), ha="center", fontsize=8)
axb.set_xticks(x)
axb.set_xticklabels([BAR_TICKS[c] for c in BAR_CONDS], fontsize=8)
axb.set_ylabel("transition width $w_{10\\to90}$")
axb.set_ylim(0, 1.05)
axb.set_title("Freezing four blocks slows the sharpening but does not prevent it", fontsize=10)
axb.grid(alpha=0.3, axis="y")
axb.legend(fontsize=8, loc="lower left")

# ---- bottom middle: depth control -------------------------------------------------------------
axd = fig.add_subplot(gs[1, 2])
DEPTH_LABELS = {"ref_trained": ("reference, trained", 0, "-", "o"),
                "frozen_early_last": ("blocks 1-4 frozen", 1, "--", "s"),
                "frozen_late_last": ("blocks 8-11 frozen", 3, "-.", "^")}
for cond, (lab, ci, ls, mk) in DEPTH_LABELS.items():
    d = C.get(cond, {}).get("depth_median_w")
    if not d:
        continue
    bl = sorted(int(k) for k in d)
    axd.plot(bl, [d[str(b)] for b in bl], color=CVD[ci], ls=ls, marker=mk, ms=7, lw=2, label=lab)
axd.axhline(C["ref_init"]["median_w"], **REF_DIAG)
axd.set_xlabel("interpolation block")
axd.set_ylabel("median $w_{10\\to90}$")
axd.set_ylim(0, 1.05)
axd.set_title("Where is the path sharpened?", fontsize=10)
axd.grid(alpha=0.3)
axd.legend(fontsize=7, loc="lower right")

# ---- bottom right: validation accuracy across training ----------------------------------------
axa = fig.add_subplot(gs[1, 3:])
runs = [("grok_char", "reference (all blocks train)", 0, "-"),
        ("frozen_early", "blocks 1-4 frozen at init", 1, "--"),
        ("frozen_late", "blocks 8-11 frozen at init", 3, "-.")]
ref_acc = S["ref_final_val_acc"]
for tag, lab, ci, ls in runs:
    p = os.path.join(RES, f"train_hist_{tag}.json")
    if not os.path.exists(p):
        continue
    h = json.load(open(p))
    axa.plot(h["step"], h["val_acc"], color=CVD[ci], ls=ls, lw=2, label=lab)
axa.axhline(ref_acc, **REF_RULE)
axa.annotate(f"reference final val acc = {ref_acc:.3f}", (1.5, ref_acc + 0.008), fontsize=8)
for tag, ci, mk in (("frozen_early", 1, "s"), ("frozen_late", 3, "^")):
    key = f"{tag}_matched"
    if key in C and C[key].get("val_acc") is not None:
        axa.plot([C[key]["step"]], [C[key]["val_acc"]], marker=mk, ms=11, mfc="none", mew=2,
                 color=CVD[ci], ls="none",
                 label=f"assayed checkpoint (step {C[key]['step']})")
axa.set_xscale("symlog", linthresh=100)
axa.set_xlabel("optimization step (symlog, linear below 100)")
axa.set_ylabel("validation next-character accuracy")
axa.set_ylim(0, 0.62)
axa.set_title("All three runs reach the same validation accuracy", fontsize=10)
axa.grid(alpha=0.3)
axa.legend(fontsize=8, loc="lower right")

fig.savefig(os.path.join(PLOTS, "frozen_blocks.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote plots/frozen_blocks.png")
