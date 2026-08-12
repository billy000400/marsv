"""Figure + numbers for the partial-transplant rank sweep (results/mlp_rank.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS, CVD

D = json.load(open(f"{RESULTS}/mlp_rank.json"))
rows = D["rows"]
base = np.array(D["base_w"])
n = len(base)
off = ~np.eye(n, dtype=bool)

for r in rows:
    W = np.array(r["w"])
    r["mean_w"] = float(np.nanmean(W[off]))
    r["sd_w"] = float(np.nanstd(W[off], ddof=1))

pca = [r for r in rows if r["basis"] == "pca"]
rnd = [r for r in rows if r["basis"] == "random"]
tail = [r for r in rows if r["basis"] == "tail"]
full = pca[-1]

print("basis   k   var kept  slope   rho    mean w   sd w   bits")
for r in pca + rnd + tail:
    v = "  --  " if r["var_retained"] is None else f"{r['var_retained']:.3f} "
    print(f"{r['basis']:6s} {r['k']:3d}  {v}  {r['slope']:+.3f}  {r['rho']:+.3f}  "
          f"{r['mean_w']:.3f}  {r['sd_w']:.3f}  {r['median_bits']:.3f}")
top64 = [r for r in pca if r["k"] == 64][0]
tail58 = [r for r in tail if r["k"] == 58][0]
print(f"\ntop-64 ({top64['var_retained']:.2f} of the variance) slope {top64['slope']:+.3f} "
      f"+ tail-58 slope {tail58['slope']:+.3f} = {top64['slope'] + tail58['slope']:+.3f}, "
      f"against {full['slope']:+.3f} for the whole vector")
print(f"baseline mean {base.mean():.3f} sd {base.std(ddof=1):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

kk = [r["k"] for r in pca]
ax[0].axhline(full["slope"], color="0.6", ls="-.", lw=1)
ax[0].plot(kk, [r["slope"] for r in pca], "-o", color=CVD[0], label="top $k$ principal components")
ax[0].plot([r["k"] for r in tail], [r["slope"] for r in tail], ":^", color=CVD[2],
           label="bottom $k$ components")
ax[0].plot([r["k"] for r in rnd], [r["slope"] for r in rnd], "--s", color=CVD[1],
           label="random $k$-dim subspace")
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xscale("log")
ax[0].set_xlabel("number of transplanted directions $k$")
ax[0].set_ylabel("transfer slope on the donor's width")
ax[0].set_title("only the complete vector transfers")
ax[0].legend(fontsize=7.5, loc="upper left")

ax[1].plot([r["var_retained"] for r in pca], [r["slope"] for r in pca], "-o", color=CVD[0],
           label="top $k$ components")
ax[1].plot([r["var_retained"] for r in tail], [r["slope"] for r in tail], ":^", color=CVD[2],
           label="bottom $k$ components")
ax[1].plot([0, 1], [0, full["slope"]], "k--", lw=1, label="transfer $\\propto$ variance kept")
ax[1].set_xlabel("share of across-token variance of $m$ transplanted")
ax[1].set_ylabel("transfer slope on the donor's width")
ax[1].set_title("transfer is not proportional to variance")
ax[1].legend(fontsize=7.5, loc="upper left")

ax[2].axhline(base.mean(), color="0.6", ls="-.", lw=1)
ax[2].errorbar([r["median_bits"] for r in pca], [r["mean_w"] for r in pca],
               yerr=[r["sd_w"] for r in pca], fmt="-o", color=CVD[0],
               label="top $k$ components")
for r in pca:
    ax[2].annotate(f"{r['k']}", (r["median_bits"], r["mean_w"]), fontsize=7,
                   textcoords="offset points", xytext=(4, 5))
ax[2].set_xscale("symlog", linthresh=1e-3)
ax[2].set_xlabel("output movement of the partial transplant (bits)")
ax[2].set_ylabel(r"mean $\hat w$ over the 132 transplants")
ax[2].set_title("partial transplants disturb without transferring")
ax[2].legend(fontsize=7.5, loc="lower right")

fig.tight_layout()
fig.savefig(f"{PLOTS}/mlp_rank.png", dpi=150)
plt.close(fig)
print("wrote plots/mlp_rank.png")
