"""Figure for neuron_path.py: how many MLP units carry the bend, and are they shared across pairs?

Three panels:
  A  median transition width vs the number k of linearized units, one series per selection rule
     (per-pair top-k / one global top-k / random), against the trained and untrained references;
  B  per-pair smallest k that recovers half of that pair's own trained->untrained width gap;
  C  reuse: how many of the 150 pairs each unit appears in the top-32 of.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD, LINESTYLES, MARKERS, REF_DIAG, REF_RULE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

S = json.load(open(os.path.join(RES, "neuron_path_summary.json")))
R = np.load(os.path.join(RES, "neuron_path_raw.npz"))
ks = np.array(S["k_list"], dtype=float)
wb, wi = S["baseline"]["median_w"], S["untrained"]["median_w"]

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))

# ---- A: width vs number of linearized units -------------------------------------------------
labels = {"pair": "per-pair top-$k$", "global": "one global top-$k$", "random": "random $k$"}
for i, m in enumerate(["pair", "global", "random"]):
    c = S["curves"][m]
    ax[0].plot(ks, c["median_w"], color=CVD[i], ls=LINESTYLES[i], marker=MARKERS[i], ms=5,
               lw=2, label=labels[m])
    ax[0].fill_between(ks, c["iqr_lo"], c["iqr_hi"], color=CVD[i], alpha=0.13, lw=0)
ax[0].axhline(wi, **REF_DIAG)
ax[0].text(1.15, wi + 0.012, f"untrained network ({wi:.2f})", fontsize=8, color="0.35")
ax[0].axhline(wb, **REF_RULE)
ax[0].text(1.15, wb - 0.045, f"unmodified model ({wb:.2f})", fontsize=8, color="k")
ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("number of MLP units linearized along the path, $k$\n(of %d units in blocks 1–4)"
                 % S["n_units"])
ax[0].set_ylabel("median transition width $w_{10\\to90}$")
ax[0].set_title("A. A few units carry most of the bend")
ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_ylim(0.30, 0.87)

# ---- B: per-pair k for half the gap ----------------------------------------------------------
k50 = R["k50"][np.isfinite(R["k50"])]
edges = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=float)
ax[1].hist(k50, bins=edges, color=CVD[0], edgecolor="k", hatch="//", lw=0.8)
med = float(np.median(k50))
ax[1].axvline(med, color=CVD[1], ls="--", lw=2)
ax[1].text(med * 1.15, ax[1].get_ylim()[1] * 0.92, f"median {med:.0f}", color=CVD[1], fontsize=9)
ax[1].set_xscale("log", base=2)
ax[1].set_xlabel("smallest $k$ recovering half of that pair's width gap")
ax[1].set_ylabel("number of pairs (of %d)" % S["n_pairs"])
ax[1].set_title("B. Half the bend costs tens of units")

# ---- C: reuse across pairs -------------------------------------------------------------------
counts = R["counts"]
seen = counts[counts > 0]
ax[2].hist(seen, bins=np.arange(0.5, seen.max() + 1.5), color=CVD[3], edgecolor="k",
           hatch="..", lw=0.8)
ax[2].set_yscale("log")
ax[2].set_xlabel("number of pairs a unit is in the top-32 of")
ax[2].set_ylabel("number of units (log scale)")
ax[2].set_title("C. A shared pool, but a pair-dependent subset")
ax[2].text(0.97, 0.93, f"{S['reuse_top32']['n_distinct_units']} of {S['n_units']} units ever used\n"
                       f"{100*(1-S['reuse_top32']['frac_units_in_1_pair']):.0f}% of them used by ≥2 pairs\n"
                       f"most reused unit: {S['reuse_top32']['max_pairs_one_unit']} of "
                       f"{S['n_pairs']} pairs\nmedian overlap with the\nglobal top-32: "
                       f"{S['reuse_top32']['median_overlap_with_global32']:.0f}/32",
           transform=ax[2].transAxes, ha="right", va="top", fontsize=8.5)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "neuron_path.png"), dpi=150)
plt.close(fig)
print("wrote plots/neuron_path.png")
