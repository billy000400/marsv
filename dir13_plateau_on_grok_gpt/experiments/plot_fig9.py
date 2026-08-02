"""Plot Figure-9-style grokking curves from a fig9.py output JSON:
test accuracy + eps=0.03 PGD adversarial accuracy + train/test/random local complexity
on one (log) training axis, with 99% CIs, annotated with the corrected Figure-9 gate
landmarks from fig9_verdict.analyze (first local LC minimum, the local maximum that opens the
second descent, the sustained robustness onset and the clean-accuracy peak).
Usage: plot_fig9.py <json> <out_png> <title>
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvd_style import CVD
from fig9_verdict import analyze

recs = json.load(open(sys.argv[1]))
V = analyze(recs)
out, title = sys.argv[2], sys.argv[3]
steps = np.array([r["step"] for r in recs])
x = np.maximum(steps, 1)  # log axis; step 0 plotted at 1

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
for name, color, ls in [("train", CVD[0], "-"), ("test", CVD[1], "--"), ("random", CVD[2], "-.")]:
    m = np.array([r["lc"][name]["mean"] for r in recs])
    ci = np.array([r["lc"][name]["ci99"] for r in recs])
    ax1.plot(x, m, color=color, ls=ls, lw=2, label=f"LC {name} ({ls})")
    ax1.fill_between(x, m - ci, m + ci, color=color, alpha=0.25)
ax2.plot(x, [r["clean_acc"] for r in recs], "k-", marker="o", ms=4, lw=2,
         label="test accuracy, clean (black, circles)")
ax2.plot(x, [r["adv_acc"] for r in recs], "k:", marker="s", ms=4, lw=2,
         label="adv accuracy, PGD $\\epsilon$=0.03 (black dotted, squares)")

# --- corrected Figure-9 gate landmarks (operator feedback #4) -------------------------------
lc_test = np.array([r["lc"]["test"]["mean"] for r in recs])
marks = [(V["first_local_min_step"], "1st LC local min", "-.", "v"),
         (V["second_descent_onset_step"], "2nd-descent onset (LC local max)", "--", "^"),
         (V["sustained_adv_onset_step"], "sustained adv. robustness onset", ":", None),
         (V["clean_acc_peak_step"], "clean-accuracy peak", (0, (3, 1, 1, 1, 1, 1)), None)]
for k, (st, lab, ls, mk) in enumerate(marks):
    if st is None:
        continue
    xs = max(st, 1)
    ax1.axvline(xs, color="0.35", ls=ls, lw=1.4, zorder=0)
    ax1.annotate(lab, xy=(xs, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(0, -12 - 13 * k), textcoords="offset points",
                 ha="center", va="top", fontsize=7.5, color="0.15",
                 bbox=dict(fc="white", ec="0.6", lw=0.5, alpha=0.85, pad=1.4))
    if mk is not None:
        ax1.plot([xs], [lc_test[steps.tolist().index(st)]], marker=mk, ms=10,
                 mfc="none", mec="k", mew=1.6, ls="none", zorder=5)
ax1.text(0.985, 0.03, f"corrected Figure-9 gate: {V['verdict']}", transform=ax1.transAxes,
         ha="right", va="bottom", fontsize=10, fontweight="bold",
         bbox=dict(fc="white", ec="0.3", lw=1.0, alpha=0.9))

ax1.set_xscale("log")
ax1.set_xlabel("training step (log scale; step 0 shown at 1)")
ax1.set_ylabel("local complexity (sign-crossing units, sum over 12 GeLU layers)")
ax2.set_ylabel("next-token accuracy")
ax2.set_ylim(0, 1)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=9)
ax1.grid(alpha=0.3)
plt.title(title)
fig.tight_layout()
fig.savefig(out, dpi=120)
plt.close(fig)
print("saved", out)
