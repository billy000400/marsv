"""Figure for the frozen-block training test (train_frozen.py + frozen_assay.py).

Top row:      raw d(t) curves (the primary evidence) for the same 20 pairs -- the reference run
              untrained / at step 2500 / fully trained, plus each frozen-block run present in the
              summary at its final checkpoint (one panel each).
Bottom left:  median transition width per condition, against the untrained and trained references.
Bottom middle: depth control -- median width by interpolation block for the trained reference and
              each final frozen model, i.e. whether the sharpening moved to another depth.
Bottom right: validation accuracy across training for every run, showing where the frozen runs
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
DASH5 = (0, (3, 1, 1, 1, 1, 1))  # 5th line style: the four named ones are already taken
DASH6 = (0, (5, 1, 1, 1))         # 6th line style
DASH7 = (0, (1, 1))               # 7th line style
DASH8 = (0, (4, 2, 1, 2))         # 8th line style
DASH9 = (0, (7, 2))               # 9th line style (long dashes; distinct from the default "--")
DASH10 = (0, (6, 1, 2, 1))        # 10th line style
# The CVD palette holds five hues (cvd_style.CVD) and CLAUDE.md forbids inventing a sixth. Ten
# conditions now appear, so the assignment follows THREE small multiples: the trained reference is
# neutral black in all of them (it is the anchor, not a category); the six FIVE-BLOCK runs -- the ones
# carrying the position result -- are split by where their trainable window sits, upper-stack windows
# (6-10, 4-8, 2-6) in the first panel and bottom windows plus the one non-contiguous trainable set
# (1-5, 0-4, and 0+8-11) in the second, so neither panel exceeds three hues; the four other freeze
# sizes go to gray at four lightnesses in the third. frozen_high and frozen_deep share CVD[0] because
# they never appear in the same panel. Every series also has its own dash pattern and marker, so all
# eleven stay distinguishable in grayscale.
GRAYS = ("0.15", "0.38", "0.58", "0.74")
# condition -> (label, colour, linestyle, marker)
STYLE = {"ref_trained":         ("reference, trained",        "black", "-",    "o"),
         "frozen_high_last":    ("blocks 0-5, 11 frozen",     CVD[0], DASH10, "h"),
         "frozen_mid_last":     ("blocks 0-3, 9-11 frozen",   CVD[1], DASH5,  "v"),
         "frozen_mid_off_last": ("blocks 0-1, 7-11 frozen",   CVD[2], DASH9,  "<"),
         "frozen_mid_low_last": ("blocks 0, 6-11 frozen",     CVD[3], DASH7,  ">"),
         "frozen_deep_last":    ("blocks 1-7 frozen",         CVD[0], ":",    "D"),
         "frozen_mirror_last":  ("blocks 5-11 frozen",        CVD[4], DASH6,  "P"),
         "frozen_early_last":   ("blocks 1-4 frozen",        GRAYS[0], "--",  "s"),
         "frozen_late_last":    ("blocks 8-11 frozen",       GRAYS[1], "-.",  "^"),
         "frozen_mid3_last":    ("blocks 0-4, 8-11 frozen",  GRAYS[2], DASH7, "X"),
         "frozen_two_last":     ("blocks 1-10 frozen",       GRAYS[3], DASH8, "*")}
# three depth small-multiples: six five-block runs split by where the trainable window sits, then the
# other freeze sizes -- no panel carries more than three hues plus the black reference anchor
DEPTH_GROUPS = [("Five trainable blocks, upper-stack windows",
                 ["ref_trained", "frozen_high_last", "frozen_mid_last", "frozen_mid_off_last"]),
                ("Five trainable blocks, bottom windows and the split set",
                 ["ref_trained", "frozen_mid_low_last", "frozen_mirror_last", "frozen_deep_last"]),
                ("Other freeze sizes",
                 ["ref_trained", "frozen_early_last", "frozen_late_last", "frozen_mid3_last",
                  "frozen_two_last"])]
S = json.load(open(os.path.join(RES, "frozen_assay_summary.json")))["summary"]
raw = np.load(os.path.join(RES, "frozen_assay_raw.npz"))
C = S["conditions"]
ts = raw["ts"]

CURVE_CONDS = [("ref_init", "reference, untrained (step 0)"),
               ("ref_matched_step", "reference at step 2500"),
               ("ref_trained", "reference, trained (step 30000)"),
               ("frozen_early_last", "blocks 1-4 frozen, final (30000)"),
               ("frozen_late_last", "blocks 8-11 frozen, final (30000)"),
               ("frozen_deep_last", "blocks 1-7 frozen, final (30000)"),
               ("frozen_mid_last", "blocks 0-3, 9-11 frozen, final (30000)"),
               ("frozen_mid3_last", "blocks 0-4, 8-11 frozen, final (30000)"),
               ("frozen_mid_off_last", "blocks 0-1, 7-11 frozen, final (30000)"),
               ("frozen_mid_low_last", "blocks 0, 6-11 frozen, final (30000)"),
               ("frozen_high_last", "blocks 0-5, 11 frozen, final (30000)"),
               ("frozen_mirror_last", "blocks 5-11 frozen, final (30000)"),
               ("frozen_two_last", "blocks 1-10 frozen, final (30000)")]
CURVE_CONDS = [c for c in CURVE_CONDS if c[0] in C]
BAR_CONDS = [c for c in ["ref_init", "ref_matched_step", "frozen_early_matched",
                         "frozen_late_matched", "frozen_deep_matched", "frozen_mid_matched",
                         "frozen_mid3_matched", "frozen_mid_off_matched",
                         "frozen_mid_low_matched", "frozen_high_matched", "frozen_mirror_matched",
                         "frozen_early_last", "frozen_late_last", "frozen_deep_last",
                         "frozen_mid_last", "frozen_mid3_last", "frozen_mid_off_last",
                         "frozen_mid_low_last", "frozen_high_last", "frozen_mirror_last",
                         "frozen_two_matched", "frozen_two_last",
                         "ref_trained"] if c in C]
BAR_TICKS = {"ref_init": "reference\nuntrained", "ref_trained": "reference\ntrained\n(30000)",
             "ref_matched_step": "reference\nstep 2500",
             "frozen_early_matched": "frozen 1-4\nmatched acc", "frozen_early_last": "frozen 1-4\nfinal",
             "frozen_late_matched": "frozen 8-11\nmatched acc", "frozen_late_last": "frozen 8-11\nfinal",
             "frozen_deep_matched": "frozen 1-7\nmatched acc", "frozen_deep_last": "frozen 1-7\nfinal",
             "frozen_mid_matched": "frozen 0-3,\n9-11\nmatched acc",
             "frozen_mid_last": "frozen 0-3,\n9-11\nfinal",
             "frozen_mid3_matched": "frozen 0-4,\n8-11\nmatched acc",
             "frozen_mid3_last": "frozen 0-4,\n8-11\nfinal",
             "frozen_mid_off_matched": "frozen 0-1,\n7-11\nmatched acc",
             "frozen_mid_off_last": "frozen 0-1,\n7-11\nfinal",
             "frozen_mid_low_matched": "frozen 0,\n6-11\nmatched acc",
             "frozen_mid_low_last": "frozen 0,\n6-11\nfinal",
             "frozen_high_matched": "frozen 0-5,\n11\nmatched acc",
             "frozen_high_last": "frozen 0-5,\n11\nfinal",
             "frozen_mirror_matched": "frozen 5-11\nmatched acc",
             "frozen_mirror_last": "frozen 5-11\nfinal",
             "frozen_two_matched": "frozen 1-10\nmatched acc",
             "frozen_two_last": "frozen 1-10\nfinal"}
NC = len(CURVE_CONDS)

fig = plt.figure(figsize=(3.25 * NC, 8.4))
gs = fig.add_gridspec(2, NC, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.32)

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
axb = fig.add_subplot(gs[1, :NC - 5])
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
axb.set_title("How much freezing costs depends on WHERE the trainable blocks sit, not how many",
              fontsize=10)
axb.grid(alpha=0.3, axis="y")
axb.legend(fontsize=8, loc="lower left")

# ---- bottom middle: depth control -------------------------------------------------------------
for gi, (gtitle, conds) in enumerate(DEPTH_GROUPS):
    axd = fig.add_subplot(gs[1, NC - 5 + gi])
    for cond in conds:
        d = C.get(cond, {}).get("depth_median_w")
        if not d:
            continue
        lab, col, ls, mk = STYLE[cond]
        bl = sorted(int(k) for k in d)
        axd.plot(bl, [d[str(b)] for b in bl], color=col, ls=ls, marker=mk, ms=7, lw=2, label=lab)
    axd.axhline(C["ref_init"]["median_w"], **REF_DIAG)
    axd.set_xlabel("interpolation block")
    if gi == 0:
        axd.set_ylabel("median $w_{10\\to90}$")
    axd.set_ylim(0, 1.05)
    axd.set_title(f"Where is the path sharpened?\n{gtitle}", fontsize=9)
    axd.grid(alpha=0.3)
    axd.legend(fontsize=7, loc="lower right")

# ---- bottom right: validation accuracy across training ----------------------------------------
axa = fig.add_subplot(gs[1, NC - 2:])
runs = [("grok_char", "ref_trained", "reference (all blocks train)"),
        ("frozen_early", "frozen_early_last", "blocks 1-4 frozen at init"),
        ("frozen_late", "frozen_late_last", "blocks 8-11 frozen at init"),
        ("frozen_deep", "frozen_deep_last", "blocks 1-7 frozen at init"),
        ("frozen_mid", "frozen_mid_last", "blocks 0-3, 9-11 frozen at init"),
        ("frozen_mid3", "frozen_mid3_last", "blocks 0-4, 8-11 frozen at init"),
        ("frozen_mid_off", "frozen_mid_off_last", "blocks 0-1, 7-11 frozen at init"),
        ("frozen_mid_low", "frozen_mid_low_last", "blocks 0, 6-11 frozen at init"),
        ("frozen_high", "frozen_high_last", "blocks 0-5, 11 frozen at init"),
        ("frozen_mirror", "frozen_mirror_last", "blocks 5-11 frozen at init"),
        ("frozen_two", "frozen_two_last", "blocks 1-10 frozen at init")]
ref_acc = S["ref_final_val_acc"]
for tag, skey, lab in runs:
    p = os.path.join(RES, f"train_hist_{tag}.json")
    if not os.path.exists(p):
        continue
    _, col, ls, _ = STYLE[skey]
    h = json.load(open(p))
    axa.plot(h["step"], h["val_acc"], color=col, ls=ls, lw=2, label=lab)
axa.axhline(ref_acc, **REF_RULE)
axa.annotate(f"reference final val acc = {ref_acc:.3f}", (1.5, ref_acc + 0.008), fontsize=8)
first = True
for tag, skey, _ in runs[1:]:
    key = f"{tag}_matched"
    if key in C and C[key].get("val_acc") is not None:
        _, col, _, mk = STYLE[skey]
        # one legend entry for the whole marker family; the per-run steps are given in the caption
        axa.plot([C[key]["step"]], [C[key]["val_acc"]], marker=mk, ms=11, mfc="none", mew=2,
                 color=col, ls="none",
                 label="matched-accuracy checkpoint (assayed)" if first else None)
        first = False
axa.set_xscale("symlog", linthresh=100)
axa.set_xlabel("optimization step (symlog, linear below 100)")
axa.set_ylabel("validation next-character accuracy")
axa.set_ylim(0, 0.62)
axa.set_title("Every run reaches the reference validation accuracy", fontsize=10)
axa.grid(alpha=0.3)
axa.legend(fontsize=8, loc="lower right")

fig.savefig(os.path.join(PLOTS, "frozen_blocks.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote plots/frozen_blocks.png")
