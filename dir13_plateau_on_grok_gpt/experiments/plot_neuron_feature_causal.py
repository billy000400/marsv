"""Figure for neuron_feature_causal.py: units chosen by corpus tuning alone, then linearized.

One panel: recovered fraction of the trained->untrained width gap against the number of linearized
units, for the held-out corpus-tuning rule and the three rules neuron_path.py already ran.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD, LINESTYLES, MARKERS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

C = json.load(open(os.path.join(RES, "neuron_feature_causal.json")))
ks = np.array(C["k_list"], dtype=float)

series = [("corpus tuning $|z_a-z_b|$ (held out)", C["tuning_selected"]["recovered_frac"], 0),
          ("assay-ranked per-pair top-$k$ (fitted)", C["reference_rules"]["pair"]["recovered_frac"], 1),
          ("one global top-$k$ (assay-derived)", C["reference_rules"]["global"]["recovered_frac"], 2),
          ("random $k$", C["reference_rules"]["random"]["recovered_frac"], 3)]

fig, ax = plt.subplots(figsize=(6.6, 4.8))
for lab, y, i in series:
    ax.plot(ks, 100 * np.array(y), color=CVD[i], ls=LINESTYLES[i % 4], marker=MARKERS[i % 4],
            ms=6, lw=2, label=lab)
ax.annotate(f"{100*C['tuning_selected']['recovered_frac'][1]:.1f}%",
            (32, 100 * C["tuning_selected"]["recovered_frac"][1]), textcoords="offset points",
            xytext=(6, -14), fontsize=9, fontweight="bold", color=CVD[0])
ax.set_xscale("log", base=2)
ax.set_xlabel("number of MLP units linearized along the path, $k$ (of 3,840 in blocks 1–4)")
ax.set_ylabel("% of the trained→untrained width gap removed")
ax.set_title("Units picked from ordinary text alone carry the bend")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.25, lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "neuron_feature_causal.png"), dpi=150)
plt.close(fig)
print("wrote plots/neuron_feature_causal.png")
