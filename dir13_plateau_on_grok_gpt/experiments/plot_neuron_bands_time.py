"""Figure for S24h: the band decomposition over training (neuron_bands_time.py) and the identity of
the head units across the same checkpoints (neuron_head_identity.py)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
MK = ["o", "s", "^", "D", "v"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

s = json.load(open(os.path.join(RES, "neuron_bands_time_summary.json")))
ident = json.load(open(os.path.join(RES, "neuron_head_identity_summary.json")))
C = s["common_subset"]["checkpoints"]
n = s["common_subset"]["n_pairs"]
edges = s["band_edges"]
steps = np.array([c["step"] for c in C])

fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
ax = axes.ravel()

# (a) how much of the bend the blocks-1-4 units carry, and what the bands add up to
ax[0].plot(steps, [100 * c["rho_all_units"] for c in C], ls=LS[0], color=CVD[0], marker=MK[0],
           lw=2, ms=7, label="all 3,840 units linearized at once")
ax[0].plot(steps, [100 * c["sum_of_band_alone"] for c in C], ls=LS[1], color=CVD[1], marker=MK[1],
           lw=2, ms=7, label="sum of the six bands measured alone")
ax[0].set_xscale("log")
ax[0].set_xlabel("training step (log)")
ax[0].set_ylabel("% of the trained$\\rightarrow$untrained width gap removed")
ax[0].set_title("(a) the units carry more of the bend\nas the plateau sharpens")
ax[0].legend(loc="upper left", fontsize=8.5)
ax[0].grid(alpha=0.25, which="both")

# (b) the redundancy ratio itself
ax[1].plot(steps, [c["redundancy_ratio"] for c in C], ls=LS[0], color=CVD[0], marker=MK[0],
           lw=2, ms=7, label="median over the %d pairs" % n)
ax[1].axhline(1.0, color="0.45", ls="--", lw=1.4)
ax[1].text(900, 1.022, "1.0 = the bands add up exactly (no overlap)", fontsize=8.5, color="0.3")
ax[1].set_xscale("log")
ax[1].set_ylim(0.95, 1.35)
ax[1].set_xlabel("training step (log)")
ax[1].set_ylabel("(sum of band-alone effects) / (all-units effect)")
ax[1].set_title("(b) redundancy is present from the start\nand does not grow steadily")
ax[1].legend(loc="lower right", fontsize=8.5)
ax[1].grid(alpha=0.25, which="both")

# (c) which band gains over training
for bi in range(5):
    ax[2].plot(steps, [100 * c["rho_band_alone"][bi] for c in C], ls=LS[bi], color=CVD[bi],
               marker=MK[bi], lw=2, ms=6,
               label=f"ranks {edges[bi]}–{edges[bi+1]} ({edges[bi+1]-edges[bi]} units)")
ax[2].set_xscale("log")
ax[2].set_xlabel("training step (log)")
ax[2].set_ylabel("% of the gap removed by that band alone")
ax[2].set_title("(c) the head strengthens; the deep tail\n(ranks 2,048–3,840, not shown) stays at 0")
ax[2].legend(loc="upper left", fontsize=8)
ax[2].grid(alpha=0.25, which="both")

# (d) are the top slots held by the same units throughout?
for j, k in enumerate(ident["k_list"]):
    v = ident["overlap_with_final"][str(k)]
    ax[3].plot(ident["steps"], [100 * x / k for x in v], ls=LS[j], color=CVD[j], marker=MK[j],
               lw=2, ms=7, label=f"top-{k} set (chance {100*k/3840:.1f}%)")
ax[3].set_xscale("log")
ax[3].set_ylim(0, 105)
ax[3].set_xlabel("training step (log)")
ax[3].set_ylabel("% of that checkpoint's top-$k$ set\nalso in the step-30,000 top-$k$ set")
ax[3].set_title("(d) the head is progressively re-selected,\nnot just amplified")
ax[3].legend(loc="upper left", fontsize=8.5)
ax[3].grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "neuron_bands_time.png"), dpi=150)
plt.close(fig)
print("wrote plots/neuron_bands_time.png")
