"""Plots for Exp 3/4.
1) matthew_<tag>_by_checkpoint.png — d(t) at final logits, interp block 0: one panel per frozen
   checkpoint phase, both pairs overlaid (+ diagonal reference).
2) joint_timeline_<tag>.png — top: Figure-9 curves (test acc, adv acc, LC train/test/random);
   bottom: transition width w_10->90 (final logits, interp block 0) per pair vs step.
Usage: plot_matthew_ckpts.py <matthew_summary.json> <matthew_raw.npz> <fig9.json> <tag>
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvd_style import CVD, LINESTYLES, MARKERS, REF_DIAG, REF_RULE

summ = json.load(open(sys.argv[1]))
raw = np.load(sys.argv[2])
fig9 = json.load(open(sys.argv[3]))
tag = sys.argv[4]
ts = raw["ts"]
pairs = list(summ[0]["pairs"].keys())
steps = [r["step"] for r in summ]

# ---- panel-per-checkpoint raw curves (interp block 0, final logits) ----
n = len(steps)
fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.9), sharey=True)
axes = np.atleast_1d(axes)
for ax, st in zip(axes, steps):
    for pi, p in enumerate(pairs):
        d = raw[f"s{st}|{p}|L0|logits"]
        ax.plot(ts, d, lw=2, color=CVD[pi], ls=LINESTYLES[pi], marker=MARKERS[pi],
                ms=3, markevery=6, label=f"{p} ({LINESTYLES[pi]})")
    ax.plot([0, 1], [0, 1], **REF_DIAG)
    ax.set_title(f"step {st}", fontsize=10)
    ax.set_xlabel("t")
axes[0].set_ylabel("d(t) (final logits)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Matthew d(t), interp block 0 — {tag}", y=1.02)
fig.tight_layout()
fig.savefig(f"plots/matthew_{tag}_by_checkpoint.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---- joint timeline ----
f9steps = np.array([r["step"] for r in fig9])
x9 = np.maximum(f9steps, 1)
fig, (a1, a3) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
a2 = a1.twinx()
for name, color, ls in [("train", CVD[0], "-"), ("test", CVD[1], "--"), ("random", CVD[2], "-.")]:
    m = np.array([r["lc"][name]["mean"] for r in fig9])
    ci = np.array([r["lc"][name]["ci99"] for r in fig9])
    a1.plot(x9, m, color=color, ls=ls, lw=2, label=f"LC {name} ({ls})")
    a1.fill_between(x9, m - ci, m + ci, color=color, alpha=0.25)
a2.plot(x9, [r["clean_acc"] for r in fig9], "k-", marker="o", ms=4, lw=2,
        label="test acc, clean (black, circles)")
a2.plot(x9, [r["adv_acc"] for r in fig9], "k:", marker="s", ms=4, lw=2,
        label="adv acc, PGD $\\epsilon$=0.03 (black dotted, squares)")
a1.set_ylabel("local complexity")
a2.set_ylabel("next-token accuracy"); a2.set_ylim(0, 1)
h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
a1.grid(alpha=0.3)

xm = np.maximum(np.array(steps), 1)
for pi, p in enumerate(pairs):
    ws = []
    for r in summ:
        iv = r["pairs"][p]["interp"]
        key = 0 if 0 in iv else "0"
        ws.append(iv[key]["w_logits"])
    a3.plot(xm, ws, color=CVD[pi], ls=LINESTYLES[pi], marker=MARKERS[pi], lw=2,
            label=f"w ({p}) [{LINESTYLES[pi]}, {MARKERS[pi]}]")
a3.axhline(0.8, label="diagonal (no plateau) w=0.8", **REF_DIAG)
a3.axhline(0.25, label="plateau rule w=0.25", **REF_RULE)
a3.set_xscale("log")
a3.set_xlabel("training step (log scale; step 0 shown at 1)")
a3.set_ylabel("transition width $w_{10\\to 90}$")
a3.set_ylim(0, 1)
a3.legend(fontsize=8)
a3.grid(alpha=0.3)
fig.suptitle(f"Grokking metrics vs plateau width on one timeline — {tag}")
fig.tight_layout()
fig.savefig(f"plots/joint_timeline_{tag}.png", dpi=120)
plt.close(fig)
print("saved plots for", tag)
