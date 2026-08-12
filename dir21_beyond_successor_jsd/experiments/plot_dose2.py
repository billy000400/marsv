"""Figure + numbers for the per-token-matched dose-response (results/dose2.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from common import PLOTS, RESULTS, CVD

D = json.load(open(f"{RESULTS}/dose2.json"))
rows = D["rows"]
base = np.array(D["base_w"])
toks = D["tokens"]
nt = len(toks)
alphas = sorted({r["alpha"] for r in rows})


def arm(match, a):
    return [r for r in rows if r["match"] == match and r["alpha"] == a]


mlp = [arm("none", a)[0] for a in alphas]
ctok = [arm("per_token", a) for a in alphas]
cmean = [arm("mean", a)[0] for a in alphas]

bits = np.array([r["bits"] for r in mlp])
rho_m = np.array([r["rho"] for r in mlp])
rho_c = np.array([np.mean([r["rho"] for r in g]) for g in ctok])
rho_c_sd = np.array([np.std([r["rho"] for r in g], ddof=1) for g in ctok])
rho_g = np.array([r["rho"] for r in cmean])
bits_g = np.array([r["bits"] for r in cmean])
sd_m = np.array([r["sd"] for r in mlp])
sd_c = np.array([np.mean([r["sd"] for r in g]) for g in ctok])

# per-token match quality: achieved control bits / MLP bits, per token
ratio_g = np.array([np.array(r["bits_tok"]) / np.array(cm["bits_target"][:nt])
                    for r, cm in zip(cmean, cmean)])                     # (n_alpha, 12)
ratio_t = np.array([np.mean([np.array(r["bits_tok"]) / np.array(r["bits_target"][:nt])
                             for r in g], axis=0) for g in ctok])

dw_m = np.array([np.abs(np.array(r["w"]) - base) for r in mlp])
dw_c = np.array([np.mean([np.abs(np.array(r["w"]) - base) for r in g], axis=0) for g in ctok])
pw = np.array([float(wilcoxon(dw_m[i], dw_c[i]).pvalue) for i in range(len(alphas))])


def cross(x, y, level=0.6):
    """Output movement at which rho falls through `level` (log-x interpolation)."""
    for i in range(1, len(x)):
        if y[i - 1] >= level > y[i]:
            f = (y[i - 1] - level) / (y[i - 1] - y[i])
            return float(np.exp(np.log(x[i - 1]) + f * (np.log(x[i]) - np.log(x[i - 1]))))
    return float("nan")


print("alpha  bits    rho_mlp  rho_ctrl(pertok)   rho_ctrl(mean-matched, bits)   sd_m  sd_c")
for i, a in enumerate(alphas):
    print(f"{a:5.2f} {bits[i]:7.4f}  {rho_m[i]:+.2f}   {rho_c[i]:+.2f} +- {rho_c_sd[i]:.2f}"
          f"        {rho_g[i]:+.2f} ({bits_g[i]:.4f})            {sd_m[i]:.3f} {sd_c[i]:.3f}")
print("\nper-token match ratio (achieved ctrl bits / MLP bits on the same token)")
for i, a in enumerate(alphas):
    print(f"{a:5.2f}  mean-matched min/med/max {ratio_g[i].min():.2f}/{np.median(ratio_g[i]):.2f}/"
          f"{ratio_g[i].max():.2f}   per-token {ratio_t[i].min():.3f}/"
          f"{np.median(ratio_t[i]):.3f}/{ratio_t[i].max():.3f}")
print("\npaired |dw| per token (mean over seeds), Wilcoxon")
for i, a in enumerate(alphas):
    print(f"{a:5.2f} bits {bits[i]:.4f}: mlp {dw_m[i].mean():.3f} ctrl {dw_c[i].mean():.3f} "
          f"p {pw[i]:.4f}")
print(f"\nrho=0.6 crossing: mlp {cross(bits, rho_m):.4f} bits, "
      f"per-token control {cross(bits, rho_c):.4f} bits "
      f"(ratio {cross(bits, rho_c) / cross(bits, rho_m):.2f}x); "
      f"mean-matched control {cross(bits_g, rho_g):.4f} bits "
      f"(ratio {cross(bits_g, rho_g) / cross(bits, rho_m):.2f}x)")
# level-free version of the paired test: deviation of each token's change from the arm's own mean
print("\npaired |dw - mean(dw)| per token (level shift removed), Wilcoxon")
for i, a in enumerate(alphas):
    sm = np.array(mlp[i]["w"]) - base
    sc = np.mean([np.array(r["w"]) - base for r in ctok[i]], axis=0)
    cm, cc = np.abs(sm - sm.mean()), np.abs(sc - sc.mean())
    print(f"{a:5.2f} bits {bits[i]:.4f}: mlp {cm.mean():.4f} ctrl {cc.mean():.4f} "
          f"p {float(wilcoxon(cm, cc).pvalue):.4f}")

live = [i for i, b in enumerate(bits) if b <= 0.03]
n_below = sum(1 for i in live for r in ctok[i] if rho_m[i] < r["rho"])
print(f"\nlive band ({len(live)} rungs, <= 0.03 bits): MLP rho below its matched control in "
      f"{n_below}/{len(live) * len(ctok[0])} rung x seed comparisons")

print(f"\nspread of MLP per-token bits at alpha=1: "
      f"{np.min(mlp[-1]['bits_tok']):.3f}-{np.max(mlp[-1]['bits_tok']):.3f} "
      f"(ratio {np.max(mlp[-1]['bits_tok']) / np.min(mlp[-1]['bits_tok']):.1f}x)")

fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))

ax[0].axhline(1.0, color="0.75", ls="-.", lw=1)
ax[0].plot(bits, rho_m, "-o", color=CVD[0], label="block-0 MLP dose")
ax[0].errorbar(bits, rho_c, yerr=rho_c_sd, fmt="--s", color=CVD[1],
               label="random control, per-token matched")
ax[0].plot(bits_g, rho_g, ":^", color=CVD[2], label="random control, mean-matched (old)")
ax[0].set_xscale("log")
ax[0].set_xlabel("output movement (bits)")
ax[0].set_ylabel(r"$\rho$(width before, width after)")
ax[0].set_title("ordering: MLP vs matched control")
ax[0].legend(fontsize=7.5, loc="lower left")

ax[1].axhline(D["base_sd"], color="0.75", ls="-.", lw=1)
ax[1].plot(bits, sd_m, "-o", color=CVD[0], label="block-0 MLP dose")
ax[1].plot(bits, sd_c, "--s", color=CVD[1], label="random control, per-token matched")
ax[1].set_xscale("log")
ax[1].set_xlabel("output movement (bits)")
ax[1].set_ylabel(r"sd of $\hat w_u$ across the 12 tokens")
ax[1].set_title("level spread")
ax[1].legend(fontsize=7.5, loc="lower left")

for i, a in enumerate(alphas):
    ax[2].scatter(np.full(nt, bits[i]) * 0.93, ratio_g[i], s=14, marker="^",
                  facecolors="none", edgecolors=CVD[2], linewidths=0.9,
                  label="mean-matched (old)" if i == 0 else None)
    ax[2].scatter(np.full(nt, bits[i]) * 1.07, ratio_t[i], s=14, marker="s", color=CVD[1],
                  label="per-token matched" if i == 0 else None)
ax[2].axhline(1.0, color="0.4", ls="--", lw=1)
ax[2].set_xscale("log")
ax[2].set_yscale("log")
ax[2].set_xlabel("output movement of the MLP dose (bits)")
ax[2].set_ylabel("control bits / MLP bits, same token")
ax[2].set_title("how well is the control matched?")
ax[2].legend(fontsize=7.5, loc="lower left")

k = alphas.index(0.3)
o = np.argsort(base)
ax[3].plot(np.arange(nt), dw_m[k][o], "-o", color=CVD[0], label="block-0 MLP dose")
ax[3].plot(np.arange(nt), dw_c[k][o], "--s", color=CVD[1], label="per-token-matched control")
ax[3].set_xticks(np.arange(nt))
ax[3].set_xticklabels([toks[j] for j in o], rotation=70, fontsize=7)
ax[3].set_ylabel(r"$|\Delta \hat w_u|$")
ax[3].set_title(f"per-token change at {bits[k]:.4f} bits (p = {pw[k]:.3f})")
ax[3].legend(fontsize=7.5)

fig.tight_layout()
fig.savefig(f"{PLOTS}/dose.png", dpi=150)
plt.close(fig)
print("wrote plots/dose.png")
