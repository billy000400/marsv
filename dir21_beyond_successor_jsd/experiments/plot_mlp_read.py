"""Figure + numbers for the block-0 MLP read/transplant experiment (results/mlp_read.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from common import PLOTS, RESULTS, CVD

D = json.load(open(f"{RESULTS}/mlp_read.json"))
S = D["transplant_summary"]
rows = D["transplant"]
toks = json.load(open(f"{RESULTS}/mode_split.json"))["tokens"].keys()
toks = list(toks)
base = np.array(S["base_w"])
o = np.argsort(base)                      # narrow -> wide
name = [toks[i] for i in o]

W = np.full((len(toks), len(toks)), np.nan)   # W[recipient, donor]
B = np.full_like(W, np.nan)
idx = {s: i for i, s in enumerate(toks)}
for r in rows:
    if r["donor"] in idx:
        W[idx[r["recipient"]], idx[r["donor"]]] = r["w"]
        B[idx[r["recipient"]], idx[r["donor"]]] = r["bits"]
Wo = W[np.ix_(o, o)]
bo = base[o]

# does the transplanted width follow the donor's m_u or the recipient's remaining state?
rho_by_recipient = [float(spearmanr(np.delete(bo, i), np.delete(Wo[i], i)).statistic)
                    for i in range(len(o))]
rho_by_donor = [float(spearmanr(np.delete(bo, j), np.delete(Wo[:, j], j)).statistic)
                for j in range(len(o))]
slope_by_recipient = [float(np.polyfit(np.delete(bo, i), np.delete(Wo[i], i), 1)[0])
                      for i in range(len(o))]

off = ~np.eye(len(o), dtype=bool)
g = Wo.copy()
tot = np.nanvar(g[off])
var_donor = np.nanvar(np.array([np.nanmean(np.delete(Wo[:, j], j)) for j in range(len(o))]))
var_recip = np.nanvar(np.array([np.nanmean(np.delete(Wo[i], i)) for i in range(len(o))]))

print("PROBE (test rho +- sd over 50 splits, 80 train / 43 test tokens)")
for k in ("mlp_out", "resid_block0", "embedding"):
    p = D[f"probe_{k}"]
    print(f"  {k:14s} rho {p['rho_mean']:+.3f} +- {p['rho_sd']:.3f}  R2 {p['r2_mean']:+.3f} "
          f"  null rho {p['null_rho_mean']:+.3f}")
print(f"  m_u across the three frames: mean cosine {D['m_frame_cosine']:.4f}")

print("\nTRANSPLANT (12 recipients x 11 donors, frame 1)")
print(f"  self-transplant reproduces the baseline: rho = {S['rho_base_vs_self']:+.3f}, "
      f"max |diff| = {np.max(np.abs(np.array(S['self_w']) - np.array(S['base_w']))):.4f}")
print(f"  width follows the DONOR's m_u: per-recipient rho mean {np.mean(rho_by_recipient):+.3f} "
      f"(min {np.min(rho_by_recipient):+.2f}), Wilcoxon p = {wilcoxon(rho_by_recipient).pvalue:.4f}, "
      f"slope {np.mean(slope_by_recipient):+.3f}")
print(f"  width follows the RECIPIENT's own state: per-donor rho mean {np.mean(rho_by_donor):+.3f} "
      f"(Wilcoxon p = {wilcoxon(rho_by_donor).pvalue:.4f})")
print(f"  between-donor variance {var_donor:.5f} vs between-recipient variance {var_recip:.5f} "
      f"(ratio {var_donor / var_recip:.1f}x), total off-diagonal variance {tot:.5f}")
print(f"  median output movement of a cross transplant: {S['median_bits']:.3f} bits")
print(f"  mean-donor (m replaced by the 12-token mean) width: "
      f"{np.mean(S['mean_donor_w']):.3f} +- {np.std(S['mean_donor_w'], ddof=1):.3f}, "
      f"vs baseline {base.mean():.3f} +- {base.std(ddof=1):.3f}")

fig, ax = plt.subplots(1, 4, figsize=(19, 5.0))

keys = ["embedding", "mlp_out", "resid_block0"]
lab = ["static\nembedding $W_E[u]$", "block-0 MLP\noutput $m_u$", "post-block-0\nstate $x_u$"]
mu = [D[f"probe_{k}"]["rho_mean"] for k in keys]
sd = [D[f"probe_{k}"]["rho_sd"] for k in keys]
nul = [D[f"probe_{k}"]["null_rho_mean"] for k in keys]
hat = ["//", "\\\\", ".."]
for i in range(3):
    ax[0].bar(i, mu[i], yerr=sd[i], color=CVD[i], hatch=hat[i], edgecolor="k", linewidth=0.6)
    ax[0].bar(i + 0.0, nul[i], width=0.4, color="0.75", hatch="xx", edgecolor="k", linewidth=0.4)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xticks(range(3))
ax[0].set_xticklabels(lab, fontsize=8)
ax[0].set_ylabel(r"held-out Spearman $\rho$ with $\hat w_u$")
ax[0].set_title("what can be read off? (gray = shuffled targets)")

im = ax[1].imshow(Wo, cmap="cividis", origin="lower")
ax[1].set_xticks(range(len(o)))
ax[1].set_xticklabels(name, rotation=70, fontsize=7)
ax[1].set_yticks(range(len(o)))
ax[1].set_yticklabels(name, fontsize=7)
ax[1].set_xlabel("donor of $m_u$ (narrow $\\rightarrow$ wide)")
ax[1].set_ylabel("recipient (narrow $\\rightarrow$ wide)")
ax[1].set_title(r"$\hat w$ after transplant")
fig.colorbar(im, ax=ax[1], fraction=0.046)

for i in range(len(o)):
    ax[2].plot(bo, Wo[i], "-", color="0.8", lw=0.8, zorder=1)
ax[2].scatter(np.tile(bo, len(o)), Wo.ravel(), s=12, color=CVD[0], marker="o", zorder=2,
              label="cross transplant")
ax[2].scatter(bo, np.diag(Wo), s=45, color=CVD[1], marker="D", zorder=3,
              label="self transplant (= baseline)")
lim = [bo.min() - 0.02, bo.max() + 0.02]
ax[2].plot(lim, lim, "k--", lw=1, label=r"$y = x$ (full transfer)")
ax[2].set_xlabel(r"donor's own width $\hat w_{\mathrm{donor}}$")
ax[2].set_ylabel(r"recipient's width after transplant")
ax[2].set_title(f"slope {np.mean(slope_by_recipient):+.2f} on the donor")
ax[2].legend(fontsize=7.5, loc="upper left")

ax[3].plot(np.arange(len(o)), sorted(rho_by_recipient), "-o", color=CVD[0],
           label=r"vs donor's width (recipient fixed)")
ax[3].plot(np.arange(len(o)), sorted(rho_by_donor), "--s", color=CVD[1],
           label=r"vs recipient's width (donor fixed)")
ax[3].axhline(0, color="k", lw=0.8)
ax[3].axhline(1, color="0.75", ls="-.", lw=1)
ax[3].set_xlabel("the 12 tokens, sorted by their own value")
ax[3].set_ylabel(r"Spearman $\rho$ over the 11 partners")
ax[3].set_title("which half of the state sets the width?")
ax[3].legend(fontsize=7.5, loc="center right")

fig.tight_layout()
fig.savefig(f"{PLOTS}/mlp_read.png", dpi=150)
plt.close(fig)
print("wrote plots/mlp_read.png")
