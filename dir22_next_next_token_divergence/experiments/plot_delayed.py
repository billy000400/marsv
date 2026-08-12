"""Three figures for the delayed-successor experiment."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(ROOT, "plots")

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

d = json.load(open(os.path.join(ROOT, "results", "delayed.json")))
t = np.array(d["t"])
p_is = np.array(d["p_is"])
dd = np.array(d["d_delayed"])
os.makedirs(PLOTS, exist_ok=True)

XLAB = "interpolation position  t   (0 = ' Japan', 1 = ' Germany')"

# --- Fig 1: immediate readout stays put -------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(t, p_is, "-", marker="o", ms=3, markevery=8, color=CVD[0],
        label="p(' is') at the interpolated position")
ax.set_ylim(0, 1)
ax.set_xlabel(XLAB)
ax.set_ylabel("probability of ' is'")
ax.set_title("Immediate prediction is unchanged along the whole path")
ax.annotate(f"range {p_is.min():.3f}-{p_is.max():.3f};\ntop-1 is ' is' at all 101 points",
            xy=(0.5, p_is.min()), xytext=(0.30, 0.55), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=.8, color="0.35"))
ax.grid(alpha=.3)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "immediate_prediction.png"))
plt.close(fig)

# --- Fig 2: relative logit distance at the delayed readout ------------------
t10 = t[np.flatnonzero(dd >= 0.1)[0]]
t90 = t[np.flatnonzero(dd >= 0.9)[0]]
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(t, t, ":", color="0.45", label="linear reference  d = t   (w = 0.80)")
ax.plot(t, dd, "-", marker="^", ms=3.5, markevery=6, color=CVD[1],
        label=f"delayed readout, after ' is'  (w = {d['w_delayed']:.2f})")
ax.axhline(0.1, lw=.6, color="0.75")
ax.axhline(0.9, lw=.6, color="0.75")
ax.axvspan(t10, t90, color="0.85", zorder=0)
ax.text((t10 + t90) / 2, 0.04, f"transition\nwidth w = {d['w_delayed']:.2f}",
        ha="center", fontsize=8, color="0.25")
ax.set_xlabel(XLAB)
ax.set_ylabel("relative logit distance  d(t)")
ax.set_title("Delayed logits: flat, sharp switch, flat")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "delayed_distance.png"))
plt.close(fig)

# --- Fig 3: delayed Tokyo/Berlin probabilities ------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(t, d["p_tokyo"], "-", marker="o", ms=3, markevery=6, color=CVD[0], label="p(' Tokyo')")
ax.plot(t, d["p_berlin"], "--", marker="s", ms=3, markevery=6, color=CVD[1], label="p(' Berlin')")
ax.axvline(d["flip_t_delayed"], lw=.8, ls="-.", color="0.45")
ax.text(d["flip_t_delayed"] + 0.015, 0.45, f"top-1 flips at t = {d['flip_t_delayed']:.2f}",
        fontsize=8, color="0.25")
ax.set_ylim(-0.03, 1.0)
ax.set_xlabel(XLAB)
ax.set_ylabel("probability at the delayed readout")
ax.set_title("The capital swaps in one step, not gradually")
ax.legend(loc="center right", fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "delayed_tokens.png"))
plt.close(fig)
print("saved 3 figures")
