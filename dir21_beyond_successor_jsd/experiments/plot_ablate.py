"""Figure 18: does mean-ablating any single early component destroy the per-token width trait?"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import CVD, PLOTS, RESULTS

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)

r = json.load(open(f"{RESULTS}/ablate.json"))
base = np.array(r["base_w"])
rows = r["rows"]
base_sd = float(np.nanstd(base, ddof=1))

blk = np.array([x["block"] for x in rows])
sd = np.array([x["sd"] for x in rows])
bits = np.array([x["bits"] for x in rows])
rho = np.array([stats.spearmanr(base, np.array(x["w"]))[0] for x in rows])
is_mlp = np.array([x["comp"] == "mlp" for x in rows])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

rng = np.random.default_rng(0)
jit = rng.uniform(-0.22, 0.22, len(rows))
ax[0].scatter(blk[~is_mlp] + jit[~is_mlp], sd[~is_mlp], s=26, marker="o",
              facecolors="none", edgecolors=CVD[0], label="attention head")
ax[0].scatter(blk[is_mlp], sd[is_mlp], s=95, marker="D", color=CVD[1], label="MLP")
ax[0].axhline(base_sd, ls="-.", color="0.3", lw=1.4)
ax[0].text(5.4, base_sd, "  unablated", va="bottom", ha="right", color="0.3", fontsize=9)
ax[0].set_xlabel("block containing the ablated component")
ax[0].set_ylabel(r"sd of $\hat w_u$ across the 12 tokens")
ax[0].set_title("Spread of the per-token trait after ablation")
ax[0].set_ylim(0, max(sd.max(), base_sd) * 1.25)
ax[0].legend(frameon=False, fontsize=9, loc="lower right")

ax[1].scatter(np.maximum(bits[~is_mlp], 1e-5), rho[~is_mlp], s=26, marker="o",
              facecolors="none", edgecolors=CVD[0], label="attention head")
ax[1].scatter(np.maximum(bits[is_mlp], 1e-5), rho[is_mlp], s=95, marker="D", color=CVD[1],
              label="MLP")
ax[1].axhline(1.0, ls="-.", color="0.3", lw=1.4)
ax[1].set_xscale("log")
ax[1].set_xlabel("output movement caused by the ablation (bits, log scale)")
ax[1].set_ylabel(r"$\rho$(unablated $\hat w_u$, ablated $\hat w_u$)")
ax[1].set_title("Ordering survives every single-component ablation")
ax[1].set_ylim(-0.1, 1.08)
ax[1].legend(frameon=False, fontsize=9, loc="lower left")

k = int(np.argmin(rho))
ax[2].scatter(base, np.array(rows[k]["w"]), s=60, marker="o", facecolors="none",
              edgecolors=CVD[0], label=f"most damaging: block {rows[k]['block']} "
                                       f"{rows[k]['comp']} ($\\rho={rho[k]:+.2f}$)")
k2 = int(np.argmax(bits))
ax[2].scatter(base, np.array(rows[k2]["w"]), s=60, marker="s", color=CVD[1],
              label=f"loudest: block {rows[k2]['block']} {rows[k2]['comp']} "
                    f"({bits[k2]:.2f} bits, $\\rho={rho[k2]:+.2f}$)")
lim = [min(base.min(), 0.4) - 0.02, max(base.max(), 0.7) + 0.02]
ax[2].plot(lim, lim, ls=":", color="0.4", lw=1.2)
ax[2].set_xlim(lim)
ax[2].set_ylim(lim)
ax[2].set_xlabel(r"$\hat w_u$ with nothing ablated")
ax[2].set_ylabel(r"$\hat w_u$ after the ablation")
ax[2].set_title("The two worst cases, token by token")
ax[2].legend(frameon=False, fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(f"{PLOTS}/ablate.png", dpi=150)
plt.close(fig)

summ = dict(n_components=len(rows), base_sd=base_sd, base_mean=float(base.mean()),
            min_sd=float(sd.min()), min_sd_comp=f"block {blk[int(np.argmin(sd))]} "
                                                f"{rows[int(np.argmin(sd))]['comp']}",
            min_rho=float(rho.min()), min_rho_comp=f"block {rows[k]['block']} {rows[k]['comp']}",
            max_bits=float(bits.max()), max_bits_comp=f"block {rows[k2]['block']} {rows[k2]['comp']}",
            median_rho=float(np.median(rho)), median_sd=float(np.median(sd)),
            rho_bits_vs_rho=float(stats.spearmanr(bits, rho)[0]),
            rho_bits_vs_sd=float(stats.spearmanr(bits, sd)[0]),
            mlp_sd=[float(x) for x in sd[is_mlp]], mlp_rho=[float(x) for x in rho[is_mlp]],
            mlp_bits=[float(x) for x in bits[is_mlp]])
json.dump(summ, open(f"{RESULTS}/ablate_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
