"""Figure for neuron_bigram.py: what the residual half of the path-bending units responds to.

Three panels:
  (a) how much of a unit's corpus response the current character explains, for the recruits the
      character ranking finds, the ones it misses, and all units;
  (b) the missed recruits still bend the path, at matched set size;
  (c) conditioning tuning on the previous character does not recover them -- a like-for-like
      k = 32 comparison on the 84 pairs where both profiles are well estimated.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD, HATCHES, REF_DIAG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

S = json.load(open(os.path.join(RES, "neuron_bigram_summary.json")))
R = np.load(os.path.join(RES, "neuron_bigram_raw.npz"))
sc = R["share_cur"]
found, missed = R["found_u"], R["missed_u"]
C = S["causal"]

fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5))

# (a) current-character share of corpus response ------------------------------------------------
ax = axes[0]
grid = np.linspace(0, 1, 200)
groups = [("recruits the ranking finds", sc[found], 0),
          ("recruits it misses", sc[missed], 1),
          ("all 3,840 units", sc, 2)]
for lab, v, i in groups:
    ax.plot(grid, [(v <= g).mean() for g in grid], color=CVD[i], ls=["-", "--", ":"][i], lw=2.2,
            label=f"{lab} (median {np.median(v):.2f})")
ax.set_xlabel("fraction of a unit's corpus response explained by the current character alone")
ax.set_ylabel("cumulative fraction of units")
ax.set_title("(a) The missed units are context-dependent")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.25, lw=0.6)

# (b) matched-size ablation of found vs missed recruits ------------------------------------------
ax = axes[1]
fm = C["found_vs_missed_recruits"]
vals = [100 * fm["found_recovered_frac"], 100 * fm["missed_recovered_frac"]]
bars = ax.bar([0, 1], vals, color=[CVD[0], CVD[1]], width=0.6, edgecolor="k", lw=0.8)
for b, h in zip(bars, HATCHES[:2]):
    b.set_hatch(h)
for x, v in zip([0, 1], vals):
    ax.text(x, v + 1.0, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels([f"found\n({fm['k_each']} units)", f"missed\n({fm['k_each']} units)"])
ax.set_ylim(0, max(vals) * 1.35)
ax.set_ylabel("% of the trained→untrained width gap removed")
ax.set_title("(b) Missed recruits carry real bend, less of it")
ax.grid(alpha=0.25, lw=0.6, axis="y")

# (c) like-for-like selection rules at k = 32 -----------------------------------------------------
ax = axes[2]
r = C["restricted_to_well_sampled_pairs_k32"]
names = ["random", "one global\ntop-32", "previous-char\nconditioned", "current-char\nconditioned",
         "per-pair fitted\n(ceiling)"]
keys = ["random", "global", "ctx", "all_context", "pair_ranked"]
vals = [100 * r[k] for k in keys]
cols = ["0.6", CVD[3], CVD[1], CVD[0], "0.35"]
bars = ax.bar(range(5), vals, color=cols, width=0.62, edgecolor="k", lw=0.8)
for b, h in zip(bars, ["", "..", "\\\\", "//", ""]):
    if h:
        b.set_hatch(h)
for x, v in zip(range(5), vals):
    ax.text(x, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
ax.axhline(100 * r["all_context"], **REF_DIAG)
ax.set_xticks(range(5))
ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylim(0, max(vals) * 1.3)
ax.set_ylabel("% of the width gap removed at $k=32$")
ax.set_title(f"(c) Bigram conditioning does not help ({r['n_pairs']} pairs)")
ax.grid(alpha=0.25, lw=0.6, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "neuron_bigram.png"), dpi=150)
plt.close(fig)
print("wrote plots/neuron_bigram.png")
