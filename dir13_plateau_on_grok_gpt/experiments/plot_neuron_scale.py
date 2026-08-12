"""Figure for S24f: which scale the chord intervention reads (neuron_scale.py)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

s = json.load(open(os.path.join(RES, "neuron_scale_summary.json")))
raw = np.load(os.path.join(RES, "neuron_scale_raw.npz"))
ctl = np.load(os.path.join(RES, "neuron_probe_control_raw.npz"))
prb = np.load(os.path.join(RES, "neuron_probe_raw.npz"))
path = json.load(open(os.path.join(RES, "neuron_path_summary.json")))
K, k32 = s["k_list"], s["k_list"].index(32)
R, ref = s["rules"], s["reference_at_k32"]

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))

# (a) every rule at k = 32, activation scale against residual-displacement scale
bars = [("write norm alone (pair-blind)", R["wnorm"]["recovered_frac"][k32], "0.6", "///"),
        ("random 32", 0.0118, "0.6", "///"),
        ("character profile, standardized", ref["character_rule"], CVD[2], "\\\\"),
        ("pair-fitted top-32 (curvature)", ref["pair_fitted"], "black", ""),
        ("endpoint swing $\\times$ write norm", R["endpoint_w"]["recovered_frac"][k32], CVD[1], ".."),
        ("character profile $\\times$ write norm", R["raw_char_w"]["recovered_frac"][k32], CVD[1], ".."),
        ("probe $\\times$ write norm", R["probe_w"]["recovered_frac"][k32], CVD[1], ".."),
        ("character profile (activation units)", ref["raw_char"], CVD[0], ""),
        ("probe at context (activation units)", ref["probe"], CVD[0], ""),
        ("measured endpoint swing (oracle)", R["endpoint"]["recovered_frac"][k32], CVD[3], "")]
ypos = np.arange(len(bars))
for y, (lab, v, col, hh) in zip(ypos, bars):
    ax[0].barh(y, 100 * v, color=col, hatch=hh, edgecolor="black", lw=0.6)
    ax[0].text(100 * v + 1, y, f"{100*v:.1f}%", va="center", fontsize=9)
ax[0].set_yticks(ypos)
ax[0].set_yticklabels([b[0] for b in bars], fontsize=8.5)
ax[0].set_xlim(0, 74)
ax[0].set_xlabel("% of the trained$\\rightarrow$untrained width gap removed")
ax[0].set_title("(a) at $k=32$: the write norm adds nothing,\nand the oracle ties the corpus rule")
ax[0].grid(alpha=0.25, axis="x")

# (b) the paired per-pair cost of converting each score to residual-displacement units
deltas = [("character profile", raw["w_raw_char_w"][k32] - ctl["w_raw_char"][k32], CVD[0], "-", "o"),
          ("probe at context", raw["w_probe_w"][k32] - prb["w_probe"][k32], CVD[1], "--", "s"),
          ("measured endpoint swing", raw["w_endpoint_w"][k32] - raw["w_endpoint"][k32],
           CVD[2], ":", "^")]
for lab, d, col, ls, mk in deltas:
    x = np.sort(d)
    ax[1].plot(x, np.arange(1, len(x) + 1) / len(x), ls, color=col, lw=2,
               marker=mk, ms=4, markevery=12,
               label=f"{lab} (median {np.median(d):+.3f})")
ax[1].axvline(0.0, color="black", lw=1.2)
ax[1].set_xlabel("change in transition width $w$ when the score is multiplied by the write norm")
ax[1].set_ylabel("fraction of the 150 pairs")
ax[1].set_title("(b) per pair, weighting by the write norm\nmostly moves widths the wrong way")
ax[1].legend(loc="upper left", fontsize=8.5)
ax[1].grid(alpha=0.25)

# (c) the pair-blind floor, against the informative rules, as k grows
kl_path = path["k_list"]
idx = [kl_path.index(k) for k in K]
curves = [("measured endpoint swing (oracle)", R["endpoint"]["recovered_frac"], CVD[0], "-", "o"),
          ("character profile (activation units)",
           json.load(open(os.path.join(RES, "neuron_probe_control.json")))["raw_char"]["recovered_frac"],
           CVD[1], "--", "s"),
          ("pair-fitted top-$k$ (curvature)",
           [path["curves"]["pair"]["recovered_frac"][i] for i in idx], "black", "-.", "D"),
          ("write norm alone (pair-blind)", R["wnorm"]["recovered_frac"], CVD[2], ":", "^"),
          ("random $k$", [path["curves"]["random"]["recovered_frac"][i] for i in idx],
           "0.55", (0, (1, 1)), "v")]
for lab, v, col, ls, mk in curves:
    ax[2].plot(K, 100 * np.array(v), ls=ls, color=col, marker=mk, lw=2, ms=6, label=lab)
ax[2].set_xscale("log", base=2)
ax[2].set_xticks(K)
ax[2].set_xticklabels([str(k) for k in K])
ax[2].set_xlabel("units linearized per pair, $k$ (of 3,840)")
ax[2].set_ylabel("% of the width gap removed")
ax[2].set_title("(c) which units write hardest says almost\nnothing about which units bend a path")
ax[2].legend(loc="upper left", fontsize=8.5)
ax[2].grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "neuron_scale.png"), dpi=150)
plt.close(fig)
print("wrote plots/neuron_scale.png")
