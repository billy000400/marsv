"""Figure for S24e: a fitted description of the path-bending units (neuron_probe*.py)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

s = json.load(open(os.path.join(RES, "neuron_probe_summary.json")))
ctl = json.load(open(os.path.join(RES, "neuron_probe_control.json")))
raw = np.load(os.path.join(RES, "neuron_probe_raw.npz"))
orders = ["1", "2", "4", "8", "full"]
xs = np.arange(len(orders))
groups = [("found by the character rule", raw["found_u"], CVD[0], "-", "o"),
          ("missed by the character rule", raw["missed_u"], CVD[1], "--", "s"),
          ("all 3,840 units", np.arange(3840), "0.45", ":", "^")]

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

# (a) how much of a unit a short character window explains
for lab, u, col, ls, mk in groups:
    med = [np.median(raw[f"r2_{o}"][u]) for o in orders]
    lo = [np.percentile(raw[f"r2_{o}"][u], 25) for o in orders]
    hi = [np.percentile(raw[f"r2_{o}"][u], 75) for o in orders]
    ax[0].plot(xs, med, ls, color=col, marker=mk, lw=2, ms=6, label=lab)
    ax[0].fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
ax[0].set_xticks(xs)
ax[0].set_xticklabels(["1\n(current)", "2", "4", "8", "8 +\nbigram"])
ax[0].set_xlabel("characters in the probe's window")
ax[0].set_ylabel("held-out $R^2$ of the fitted probe")
ax[0].set_title("(a) the missed recruits are describable,\nbut only with context")
ax[0].set_ylim(0, 1.02)
ax[0].legend(loc="lower right", fontsize=8.5)
ax[0].grid(alpha=0.25)

# (b) every blind selection rule at k = 32, against the two in-domain references
k32 = 1
rules = [("random 32", 0.0118, "0.6", "///"),
         ("one global 32 (assay-ranked)", 0.1901, "0.6", "///"),
         ("bigram profile, standardized", 0.2246, CVD[2], "\\\\"),
         ("character profile, standardized", 0.2894, CVD[2], "\\\\"),
         ("probe at context, standardized", ctl["probe_z"]["recovered_frac"][k32], CVD[0], ".."),
         ("character profile, raw scale", ctl["raw_char"]["recovered_frac"][k32], CVD[0], ".."),
         ("probe at context, raw scale", s["causal"]["probe_selected"]["recovered_frac"][k32], CVD[1], "")]
ypos = np.arange(len(rules))
for y, (lab, v, col, hh) in zip(ypos, rules):
    ax[1].barh(y, 100 * v, color=col, hatch=hh, edgecolor="black", lw=0.6)
    ax[1].text(100 * v + 1, y, f"{100*v:.1f}%", va="center", fontsize=9)
ax[1].axvline(100 * 0.5091, color="black", ls="--", lw=1.4)
ax[1].text(100 * 0.5091 - 1.5, -0.75, "pair-fitted top-32 (50.9%)", ha="right", fontsize=8.5)
ax[1].set_yticks(ypos)
ax[1].set_yticklabels([r[0] for r in rules], fontsize=8.5)
ax[1].set_xlim(0, 74)
ax[1].set_xlabel("% of the trained$\\rightarrow$untrained width gap removed")
ax[1].set_title("(b) blind selection rules at $k=32$:\nkeeping each unit's own scale is what matters")
ax[1].grid(alpha=0.25, axis="x")

# (c) the same rules as a function of how many units are linearized
kl = s["causal"]["k_list"]
curves = [("probe at context, raw scale", s["causal"]["probe_selected"]["recovered_frac"], CVD[1], "-", "o"),
          ("character profile, raw scale", ctl["raw_char"]["recovered_frac"], CVD[0], "--", "s"),
          ("character profile, standardized", s["causal"]["character_rule"]["recovered_frac"], CVD[2], ":", "^"),
          ("pair-fitted top-$k$", s["causal"]["reference_rules"]["pair"]["recovered_frac"], "black", "-.", "D"),
          ("random $k$", s["causal"]["reference_rules"]["random"]["recovered_frac"], "0.55", (0, (1, 1)), "v")]
for lab, v, col, ls, mk in curves:
    ax[2].plot(kl, 100 * np.array(v), ls=ls, color=col, marker=mk, lw=2, ms=6, label=lab)
ax[2].set_xscale("log", base=2)
ax[2].set_xticks(kl)
ax[2].set_xticklabels([str(k) for k in kl])
ax[2].set_xlabel("units linearized per pair, $k$ (of 3,840)")
ax[2].set_ylabel("% of the width gap removed")
ax[2].set_title("(c) the ordering holds at every set size")
ax[2].legend(loc="upper left", fontsize=8.5)
ax[2].grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "neuron_probe.png"), dpi=150)
plt.close(fig)
print("wrote plots/neuron_probe.png")
