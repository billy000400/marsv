"""Figure 19: dose-response for the block-0 MLP against an output-matched random control."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, PLOTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]

d = json.load(open(f"{RESULTS}/dose.json"))
rows = d["rows"]
base_sd = d["base_sd"]
arms = {"mlp": ("block-0 MLP -> mean", "-", "o", CVD[0]),
        "ctrl": ("random dir, matched bits", "--", "s", CVD[1])}

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for a, (lab, ls, mk, col) in arms.items():
    r = sorted([x for x in rows if x["arm"] == a], key=lambda x: x["bits"])
    b = [x["bits"] for x in r]
    ax[0].plot(b, [x["rho"] for x in r], ls, marker=mk, color=col, label=lab)
    ax[1].plot(b, [x["sd"] for x in r], ls, marker=mk, color=col, label=lab)
ax[0].axhline(0, color="0.6", lw=0.8)
ax[0].set_ylabel(r"$\rho$(unperturbed $\hat w_u$, perturbed $\hat w_u$)")
ax[0].set_title("ordering survival")
ax[1].axhline(base_sd, color="0.4", lw=0.9, ls=":")
ax[1].text(0.02, base_sd * 1.03, "unperturbed sd", transform=ax[1].get_yaxis_transform(),
           fontsize=8, color="0.3")
ax[1].set_ylabel(r"sd of $\hat w_u$ across the 12 tokens")
ax[1].set_title("across-token spread")
for a_ in ax:
    a_.set_xlabel("output movement (bits, mean JSD vs unperturbed)")
    a_.set_xscale("log")
    a_.legend(fontsize=8, loc="best")
    a_.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{PLOTS}/dose.png", dpi=150)
plt.close(fig)
print("wrote plots/dose.png")

m = [x for x in rows if x["arm"] == "mlp"]
c = [x for x in rows if x["arm"] == "ctrl"]
for x, y in zip(m, c):
    print(f"alpha {x['dose']:.2f} bits {x['bits']:.3f}/{y['bits']:.3f} "
          f"rho {x['rho']:+.2f}/{y['rho']:+.2f} sd {x['sd']:.3f}/{y['sd']:.3f}")
