"""Figure for the readout-rebalancing intervention (rebalance_probe.py)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD, use_cvd, HATCHES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
use_cvd()

D = json.load(open(os.path.join(ROOT, "results", "rebalance_summary.json")))
S, P = D["summary"], D["pairs"]
col = lambda k: np.array([p[k] for p in P], dtype=float)
t_star, t_gap, t_eq, t_half = col("t_star"), col("t_gap"), col("t_gap_eq"), col("t_gap_half")
c_half, span = np.abs(col("c_half")), col("gap_span")
m = np.isfinite(t_gap) & np.isfinite(t_eq) & np.isfinite(t_half) & np.isfinite(t_star)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))

# (a) where the decision boundary sits under the three readouts, vs the plateau midpoint
bins = np.linspace(0, 1, 41)
series = [("plateau midpoint $t^*$ (all 3 readouts)", t_star[m], CVD[0], "//", "-"),
          ("decision boundary, unmodified", t_gap[m], CVD[1], "\\\\", "--"),
          ("decision boundary, equalised bias", t_eq[m], CVD[2], "..", "-."),
          ("decision boundary, midpoint-forced bias", t_half[m], CVD[4], "xx", ":")]
for lab, v, c, h, ls in series:
    ax[0].hist(v, bins=bins, histtype="step", lw=2.0, ls=ls, color=c, label=lab)
ax[0].set_xlabel("position along the interpolation path  $t$")
ax[0].set_ylabel("number of character pairs")
ax[0].set_title("(a) decision boundary vs plateau midpoint")
ax[0].legend(fontsize=7.5, loc="upper left")

# (b) how far the boundary moves as a function of the bias actually applied
ax[1].axhline(0.0, **{"color": "0.45", "ls": "--", "lw": 1.2})
ax[1].scatter(c_half[m] / span[m], (t_half - t_gap)[m], s=6, alpha=0.35,
              color=CVD[0], marker="o", label="midpoint-forced bias")
ax[1].scatter(np.abs(col("c_eq"))[m] / span[m], (t_eq - t_gap)[m], s=6, alpha=0.35,
              color=CVD[1], marker="^", label="equalised bias")
ax[1].set_xlabel("bias applied, as a fraction of the endpoint logit-gap span $|c|/|g(0)-g(1)|$")
ax[1].set_ylabel("shift of the decision boundary  $t_{gap}^{c}-t_{gap}$")
ax[1].set_title("(b) boundary shift vs bias size")
ax[1].legend(fontsize=8, loc="lower right", markerscale=2.5)
ax[1].text(0.03, 0.95, f"$d(t)$ unchanged to {S['d_invariance_max_abs_dev']:.1e}\n"
                       f"(width $w$ and $t^*$ exactly invariant)",
           transform=ax[1].transAxes, va="top", fontsize=8,
           bbox=dict(boxstyle="round", fc="0.94", ec="0.6"))

fig.suptitle(f"Readout rebalancing: {S['n_pairs_used']} character pairs, block {S['block']}, "
             f"step {S['step']}", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "plots", "rebalance_readout.png"), dpi=150)
plt.close(fig)
print("wrote plots/rebalance_readout.png")
