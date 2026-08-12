"""Figure for S24h: band decomposition of the unit ranking (neuron_bands.py)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

s = json.load(open(os.path.join(RES, "neuron_bands_summary.json")))
raw = np.load(os.path.join(RES, "neuron_bands_raw.npz"))
B = s["bands"]
lab = [f"{b['edge_lo']}–{b['edge_hi']}\n({b['size']} units)" for b in B]
x = np.arange(len(B))

fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.9))

# (a) each band on its own, against its marginal contribution and its within-region control
w = 0.27
for j, (key, name, col, hat) in enumerate([
        ("rho_alone", "band linearized alone", CVD[0], "//"),
        ("rho_marginal_in_prefix", "band's marginal gain inside the nested prefix", CVD[1], "\\\\"),
        ("rho_within_region_same_size", "random units of the same size from the same region",
         CVD[2], "..")]):
    ax[0].bar(x + (j - 1) * w, [100 * b[key] for b in B], w, color=col, hatch=hat,
              edgecolor="white", label=name)
ax[0].axhline(100 * s["additivity"]["all_units"], color="black", ls="-.", lw=1.4)
ax[0].text(-0.45, 100 * s["additivity"]["all_units"] - 12.0,
           "all 3,840 units\nlinearized (ceiling)", fontsize=8.5)
ax[0].set_xticks(x)
ax[0].set_xticklabels(lab, fontsize=8.5)
ax[0].set_ylim(0, 96)
ax[0].set_xlabel("importance-rank band")
ax[0].set_ylabel("% of the trained$\\rightarrow$untrained width gap removed")
ax[0].set_title("(a) every band bends the path on its own —\nand does so redundantly")
ax[0].legend(loc="upper right", fontsize=8.0, framealpha=0.95)
ax[0].grid(alpha=0.25, axis="y")

# (b) the same numbers per unit: how fast does a unit's worth decay down the ranking?
eff = np.array([1000 * b["rho_alone"] / b["size"] for b in B])
effr = np.array([1000 * b["rho_within_region_same_size"] / b["size"] for b in B])
ax[1].plot(x, np.maximum(eff, 1e-3), "-", color=CVD[0], marker="o", lw=2, ms=7,
           label="ranked band")
ax[1].plot(x, np.maximum(effr, 1e-3), "--", color=CVD[2], marker="^", lw=2, ms=7,
           label="random units, same size, same region")
ax[1].annotate("$\\leq 0$: the last band\nbends nothing", xy=(x[-1], 1e-3), xytext=(3.15, 3e-3),
               fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=1.1))
ax[1].set_yscale("log")
ax[1].set_xticks(x)
ax[1].set_xticklabels(lab, fontsize=8.5)
ax[1].set_xlabel("importance-rank band")
ax[1].set_ylabel("% of the gap removed per 1,000 units (log)")
ax[1].set_title("(b) a unit's worth falls ~500-fold down the ranking,\nbut stays above chance to rank 2,048")
ax[1].legend(loc="lower left", fontsize=8.5)
ax[1].grid(alpha=0.25, which="both")

# (c) is the tail a different kind of unit? the fitted text description of each band's units
prof = s["character_profile"]["bands"]
pr = np.load(os.path.join(RES, "neuron_probe_raw.npz"))
band_of = raw["band_of"]
for j, (key, name, col, ls, mk) in enumerate([
        ("r2_full", "full description (chars + context + interactions)", CVD[0], "-", "o"),
        ("r2_1", "current character alone", CVD[1], "--", "s")]):
    med = [np.median(pr[key][band_of == bi]) for bi in range(len(B))]
    lo = [np.percentile(pr[key][band_of == bi], 25) for bi in range(len(B))]
    hi = [np.percentile(pr[key][band_of == bi], 75) for bi in range(len(B))]
    ax[2].plot(x, med, ls, color=col, marker=mk, lw=2, ms=7, label=name)
    ax[2].fill_between(x, lo, hi, color=col, alpha=0.15,
                       hatch=["//", "\\\\"][j], edgecolor=col, lw=0)
ax[2].set_xticks(x)
ax[2].set_xticklabels([f"{p['edge_lo']}–{p['edge_hi']}\n({p['n_units_assigned']} units)"
                       for p in prof], fontsize=8.5)
ax[2].set_ylim(0, 1.02)
ax[2].set_xlabel("band of the unit's BEST rank over the 150 pairs")
ax[2].set_ylabel("held-out $R^2$ of the fitted text description")
ax[2].set_title("(c) the tail is a continuum of less describable units,\nnot a separate population")
ax[2].legend(loc="upper right", fontsize=8.5)
ax[2].grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "neuron_bands.png"), dpi=150)
plt.close(fig)
print("wrote plots/neuron_bands.png")
