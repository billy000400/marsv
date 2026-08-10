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
os.makedirs(PLOTS, exist_ok=True)

# --- Fig 1: immediate readout — the endpoint condition fails ----------------
z = np.load(os.path.join(ROOT, "results", "logits_immediate.npy"))
e = np.exp(z - z.max(1, keepdims=True))
p_top1 = (e / e.sum(1, keepdims=True)).max(1)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.semilogy(t, p_top1, "-", marker="o", ms=3, markevery=10, color=CVD[0],
            label='p(top-1) — top-1 is " =" at every t')
ax.semilogy(t, d["p_means"], "--", marker="s", ms=3, markevery=10, color=CVD[1],
            label='p(" means") — the successor the plan assumed')
ax.set_xlabel("interpolation position  t   (0 = ' A', 1 = ' B')")
ax.set_ylabel("probability (log scale)")
ax.set_title("Immediate readout: GPT-2 Large never predicts ' means'")
ax.legend(loc="center right", fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "immediate_readout.png"))
plt.close(fig)

# --- Fig 2: relative logit distance, immediate vs delayed -------------------
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(t, t, ":", color="0.45", label="linear reference  d = t  (w = 0.80)")
ax.plot(t, d["d_immediate"], "-", marker="o", ms=3, markevery=8, color=CVD[0],
        label=f"immediate readout  (w = {d['w_immediate']:.2f})")
ax.plot(t, d["d_delayed"], "--", marker="^", ms=3, markevery=8, color=CVD[1],
        label=f"delayed readout, after ' means'  (w = {d['w_delayed']:.2f})")
ax.axhline(0.1, lw=.6, color="0.7")
ax.axhline(0.9, lw=.6, color="0.7")
ax.set_xlabel("interpolation position  t   (0 = ' A', 1 = ' B')")
ax.set_ylabel("relative logit distance  d(t)")
ax.set_title("Plateau shape survives one token of propagation")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "delayed_distance.png"))
plt.close(fig)

# --- Fig 3: delayed cat/dog probabilities — the codebook is never used ------
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(t, d["p_cat"], "-", marker="o", ms=3, markevery=8, color=CVD[0], label="p(' cat')")
ax.plot(t, d["p_dog"], "--", marker="^", ms=3, markevery=8, color=CVD[1], label="p(' dog')")
ax.set_xlabel("interpolation position  t   (0 = ' A', 1 = ' B')")
ax.set_ylabel("probability at the delayed readout")
ax.set_title("No codebook lookup: ' cat' leads ' dog' at every t")
ax.legend(loc="center right", fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "delayed_tokens.png"))
plt.close(fig)
print("saved 3 figures")
