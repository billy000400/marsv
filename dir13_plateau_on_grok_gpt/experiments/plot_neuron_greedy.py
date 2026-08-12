"""Figure for S24g: residual-corrected (greedy) unit selection (neuron_greedy.py)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

s = json.load(open(os.path.join(RES, "neuron_greedy_summary.json")))
raw = np.load(os.path.join(RES, "neuron_greedy_raw.npz"))
scale = json.load(open(os.path.join(RES, "neuron_scale_summary.json")))
K, R = s["k_list"], s["r_list"]
best_unit = {32: scale["rules"]["endpoint"]["recovered_frac"][1],
             128: scale["rules"]["probe_w"]["recovered_frac"][2]}
style = {32: (CVD[0], "-", "o"), 128: (CVD[1], "--", "s")}

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))

# (a) does re-measuring between rounds buy anything?
for k in K:
    col, ls, mk = style[k]
    v = [100 * s["greedy"][str(k)][str(r)]["recovered_frac"] for r in R]
    ax[0].plot(R, v, ls=ls, color=col, marker=mk, lw=2, ms=7, label=f"greedy, $k={k}$")
    ax[0].axhline(100 * best_unit[k], color=col, ls=(0, (1, 1)), lw=1.4)
    ax[0].text(8.15, 100 * best_unit[k], f" best per-unit\n rule, $k={k}$",
               fontsize=8, color=col, va="center")
ax[0].axhline(100 * s["all_units_ceiling"], color="black", ls="-.", lw=1.4)
ax[0].text(1.05, 100 * s["all_units_ceiling"] - 2.6, "all 3,840 units linearized (ceiling)",
           fontsize=8.5)
ax[0].set_xscale("log", base=2)
ax[0].set_xticks(R)
ax[0].set_xticklabels([str(r) for r in R])
ax[0].set_xlim(0.95, 11.5)
ax[0].set_ylim(40, 92)
ax[0].set_xlabel("selection rounds $R$ ($R=1$ is the one-shot pair-fitted ranking)")
ax[0].set_ylabel("% of the trained$\\rightarrow$untrained width gap removed")
ax[0].set_title("(a) re-measuring helps at $k=128$,\nnot at $k=32$")
ax[0].legend(loc="lower right", fontsize=9)
ax[0].grid(alpha=0.25)

# (b) per-pair paired effect of 8 rounds vs 1
for k in K:
    col, ls, mk = style[k]
    d = raw[f"w_k{k}_R8"] - raw[f"w_k{k}_R1"]
    x = np.sort(d)
    ax[1].plot(x, np.arange(1, len(x) + 1) / len(x), ls, color=col, lw=2, marker=mk, ms=4,
               markevery=12,
               label=f"$k={k}$ (median {np.median(d):+.4f}, "
                     f"{100 * s['frac_pairs_R8_ge_R1'][str(k)]:.0f}% not worse)")
ax[1].axvline(0.0, color="black", lw=1.2)
ax[1].set_xlabel("change in transition width $w$, 8 rounds minus 1 round")
ax[1].set_ylabel("fraction of the 150 pairs")
ax[1].set_title("(b) the $k=128$ gain is broad, not a few pairs")
ax[1].legend(loc="upper left", fontsize=8.5)
ax[1].grid(alpha=0.25)

# (c) how different is the greedy set from the one-shot set?
for k in K:
    col, ls, mk = style[k]
    v = [100 * s["greedy"][str(k)][str(r)]["median_overlap_with_R1"] / k for r in R]
    ax[2].plot(R, v, ls=ls, color=col, marker=mk, lw=2, ms=7, label=f"$k={k}$")
ax[2].set_xscale("log", base=2)
ax[2].set_xticks(R)
ax[2].set_xticklabels([str(r) for r in R])
ax[2].set_ylim(60, 102)
ax[2].set_xlabel("selection rounds $R$")
ax[2].set_ylabel("% of the greedy set also in the one-shot set")
ax[2].set_title("(c) greedy swaps a fifth of the set\nfor a 3.4-point gain at $k=128$")
ax[2].legend(loc="lower left", fontsize=9)
ax[2].grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "neuron_greedy.png"), dpi=150)
plt.close(fig)
print("wrote plots/neuron_greedy.png")
