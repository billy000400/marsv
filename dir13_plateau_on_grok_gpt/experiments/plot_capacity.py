"""Depth or capacity? Median transition width against trainable blocks and trainable parameters.

The five frozen runs made width monotone in trainable depth, but each of them also removed trainable
parameters, so the two explanations were confounded. The narrow run (n_embd 192, nothing frozen) breaks
the tie: it has frozen_early's trainable-parameter budget with the reference's full trainable depth.

The large marker for each run is its FIRST checkpoint to reach the reference run's final validation
accuracy (0.5502), so all seven are compared at matched task performance; the small open square joined
to it by a dotted line is the same run at the end of training, showing the ordering is not an artifact
of the matching rule.

Left panel: median w vs number of trainable blocks. Right panel: the same runs vs trainable
parameters. Whichever x-axis orders the points is the variable that sets plateau sharpness.

CVD-safe (CLAUDE.md rule 13): two families separated by colour AND marker AND fill; no red/green.
"""
import json, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD, REF_DIAG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

C = json.load(open(os.path.join(RES, "frozen_assay_summary.json")))["summary"]["conditions"]
TOTAL = 8378640  # reference-run parameter count (results/train_meta_frozen_deep.json)

ck = torch.load("/tmp/dir13_frozen/checkpoints_narrow192/ckpt_matched.pt",
                map_location="cpu", weights_only=False)
NARROW = sum(v.numel() for v in ck["model"].values())

# (condition key, label, trainable blocks, trainable params, all-12-trainable?)
RUNS = [("ref_matched_step",      "reference\n(240 wide)",  12, TOTAL,            True),
        ("narrow192_matched",     "narrow 192",             12, NARROW,           True),
        ("frozen_early_matched",  "frozen 1-4",              8, TOTAL - 2777280,  False),
        ("frozen_late_matched",   "frozen 8-11",             8, TOTAL - 2777280,  False),
        ("frozen_deep_matched",   "frozen 1-7",              5, TOTAL - 4860240,  False),
        ("frozen_mirror_matched", "frozen 5-11",             5, TOTAL - 4860240,  False),
        ("frozen_two_matched",    "frozen 1-10",             2, TOTAL - 6943200,  False)]
RUNS = [r for r in RUNS if r[0] in C]

untrained = C["ref_init"]["median_w"]
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))

for ax, xi, xlabel in ((axes[0], 2, "trainable transformer blocks (of 12)"),
                       (axes[1], 3, "trainable parameters (millions)")):
    ax.axhline(untrained, **REF_DIAG)
    ax.annotate(f"untrained network, $w$={untrained:.2f} (no plateau)",
                (0.98, untrained - 0.010), xycoords=("axes fraction", "data"),
                ha="right", va="top", fontsize=8, color="0.35")
    for key, lab, nb, npar, full in RUNS:
        x = nb if xi == 2 else npar / 1e6
        if xi == 2 and nb == 12:  # the two all-trainable runs share a block count; separate them
            x += 0.50 if key.startswith("ref") else -0.50
        med = C[key]["median_w"]
        lo, hi = C[key]["iqr_w"]
        st = dict(color=CVD[0], marker="o", ms=9, mfc=CVD[0]) if full else \
             dict(color=CVD[1], marker="D", ms=8, mfc="white")
        ax.errorbar([x], [med], yerr=[[med - lo], [hi - med]], capsize=4, elinewidth=1.3,
                    lw=0, mew=1.6, **st)
        trained = C.get(key.replace("_matched_step", "_trained").replace("_matched", "_last"))
        if trained is not None:  # same run at the end of training, second framing
            dx = (0.34 if xi == 2 else 0.22) * (1 if full else -1)
            ax.plot([x, x + dx], [med, trained["median_w"]], color=st["color"], lw=0.9, ls=":",
                    zorder=1)
            ax.plot([x + dx], [trained["median_w"]], color=st["color"], marker="s", ms=5.5,
                    mfc="white", mew=1.4, lw=0, zorder=3)
        below = key == "frozen_late_matched" or (key == "narrow192_matched" and xi == 3)
        dy = -0.055 if below else 0.032
        lx = x - 0.45 if (xi == 3 and key == "narrow192_matched") else x  # clear of its own square
        ax.annotate(lab, (lx, med + dy), ha="center", va="bottom" if dy > 0 else "top", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0.24, 0.94)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("median transition width $w$\n(lower = sharper plateau)")
axes[0].set_xticks([2, 5, 8, 12])
axes[0].invert_xaxis()
axes[0].set_title("ordered by trainable DEPTH", fontsize=10)
axes[1].set_title("NOT ordered by trainable CAPACITY", fontsize=10)

h = [plt.Line2D([], [], color=CVD[0], marker="o", ms=9, lw=0, label="all 12 blocks trainable"),
     plt.Line2D([], [], color=CVD[1], marker="D", ms=8, lw=0, mfc="white",
                label="some blocks frozen at init"),
     plt.Line2D([], [], color="0.35", marker="s", ms=5.5, mfc="white", mew=1.4, ls=":", lw=0.9,
                label="same run, end of training (small square)")]
axes[0].legend(handles=h, fontsize=8, loc="lower right", framealpha=0.95)

fig.suptitle("Plateau sharpness tracks how many blocks can train, not how many parameters they hold\n"
             "(large markers: matched validation accuracy 0.550; small squares: end of training)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(PLOTS, "capacity_vs_depth.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote plots/capacity_vs_depth.png",
      {k: C[k]["median_w"] for k, *_ in RUNS}, "narrow_params", NARROW)
