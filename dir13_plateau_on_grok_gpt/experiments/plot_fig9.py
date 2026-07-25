"""Plot Figure-9-style grokking curves from a fig9.py output JSON:
test accuracy + eps=0.03 PGD adversarial accuracy + train/test/random local complexity
on one (log) training axis, with 99% CIs. Usage: plot_fig9.py <json> <out_png> <title>
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvd_style import CVD

recs = json.load(open(sys.argv[1]))
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
