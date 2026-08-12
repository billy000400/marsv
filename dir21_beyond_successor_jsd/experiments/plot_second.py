"""Figures for the second-model replication (results/second_*.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from common import PLOTS, RESULTS, CVD

S = json.load(open(f"{RESULTS}/second_summary.json"))
TAGS = S["tags"]
SIZE = {"160m": 0.16, "410m": 0.41, "1b": 1.0, "1.4b": 1.4}
names = S["tokens"]

raw = {}
for tag in TAGS[:-1]:
    raw[tag] = json.load(open(f"{RESULTS}/second_{tag}.json"))["w_raw"]
raw["1.4b"] = {s: v["w"] for s, v in
               json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
W = {tag: np.array([np.nanmedian(raw[tag][s]) for s in names]) for tag in TAGS}
pred = json.load(open(f"{RESULTS}/embed.json"))["probe_pred"]
P = np.array([pred[s] for s in names])

# ---------------------------------------------------------------- Figure: cross-model agreement
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

for j, tag in enumerate(["160m", "410m", "1b"]):
    ok = np.isfinite(W[tag]) & np.isfinite(W["1.4b"])
    ax[0].scatter(W["1.4b"][ok], W[tag][ok], s=13, color=CVD[j], marker=["o", "s", "^"][j],
                  alpha=0.75, label=f"Pythia {tag} "
                                    fr"($\rho$ = {S['pairwise'][f'{tag}|1.4b']['rho']:+.2f})")
ax[0].set_xlabel(r"$\hat w_u$ measured in Pythia-1.4B")
ax[0].set_ylabel(r"$\hat w_u$ measured in the other model")
ax[0].set_title("per-token width, model vs model")
ax[0].legend(fontsize=8, loc="upper left")

x = [SIZE[t] for t in TAGS]
rel = [S["reliability"][t]["spearman_brown"] for t in TAGS]
rho = [1.0 if t == "1.4b" else S["pairwise"][f"{t}|1.4b"]["rho"] for t in TAGS]
dis = [1.0 if t == "1.4b" else S["pairwise"][f"{t}|1.4b"]["disattenuated"] for t in TAGS]
ax[1].plot(x, rho, "-o", color=CVD[0], label=r"agreement with 1.4B, raw $\rho$")
ax[1].plot(x, dis, "--s", color=CVD[1], label="same, divided by the noise ceiling")
ax[1].plot(x, rel, ":^", color=CVD[2], label="measurement reliability of this model")
ax[1].axhline(1.0, color="0.75", ls="-.", lw=1)
ax[1].set_xscale("log")
ax[1].set_xticks(x)
ax[1].set_xticklabels(TAGS)
ax[1].minorticks_off()
ax[1].set_ylim(0, 1.15)
ax[1].set_xlabel("model size (non-embedding parameters, log scale)")
ax[1].set_ylabel(r"Spearman $\rho$ over the 123 tokens")
ax[1].set_title("agreement rises with size, then saturates")
ax[1].legend(fontsize=8, loc="lower right")

for j, tag in enumerate(TAGS):
    ok = np.isfinite(W[tag])
    ax[2].scatter(P[ok], W[tag][ok], s=12, color=CVD[j], marker=["o", "s", "^", "D"][j],
                  alpha=0.7, label=f"{tag} "
                                   fr"($\rho$ = {S['lookup_transfer'][tag]['rho']:+.2f})")
ax[2].set_xlabel(r"width predicted by the Pythia-1.4B embedding lookup, $\tilde w_u$")
ax[2].set_ylabel(r"$\hat w_u$ measured in each model")
ax[2].set_title("does the free lookup transfer across models?")
ax[2].legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(f"{PLOTS}/cross_model.png", dpi=150)
plt.close(fig)
print("wrote plots/cross_model.png")

# ---------------------------------------------------------------- Figure: probe + component sweep
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

pm = S["per_model"]
r = [pm[t]["probe"]["rho"] for t in TAGS]
e = [pm[t]["probe"]["sd"] for t in TAGS]
n = [pm[t]["probe"]["null"] for t in TAGS]
ax[0].errorbar(x, r, yerr=e, fmt="-o", color=CVD[0], capsize=3,
               label="ridge probe from this model's $W_E$")
ax[0].plot(x, n, "--s", color=CVD[1], label="shuffled-target control")
ax[0].axhline(0, color="0.75", lw=1)
ax[0].set_ylim(-0.35, 0.95)
ax[0].set_xscale("log")
ax[0].set_xticks(x)
ax[0].set_xticklabels(TAGS)
ax[0].minorticks_off()
ax[0].set_xlabel("model size (non-embedding parameters, log scale)")
ax[0].set_ylabel(r"held-out $\rho$(predicted, measured $\hat w_u$)")
ax[0].set_title("the embedding probe, refitted inside each model")
ax[0].legend(fontsize=8, loc="lower right")

for j, tag in enumerate(["160m", "410m", "1b"]):
    a = pm[tag]["ablate"]
    keys = [f"{L}:{c}" for L in range(6) for c in ("mlp", "attn") if f"{L}:{c}" in a]
    v = [a[k]["sd"] for k in keys]
    ax[1].plot(np.arange(len(keys)), v, "-" + "os^"[j], color=CVD[j], ms=5,
               label=f"Pythia {tag}")
    ax[1].axhline(pm[tag]["base_sd"], color=CVD[j], ls=":", lw=1)
ax[1].set_xticks(np.arange(12))
ax[1].set_xticklabels([f"b{L} {c}" for L in range(6) for c in ("MLP", "attn")],
                      rotation=70, fontsize=7.5)
ax[1].set_ylabel(r"sd of $\hat w_u$ across the 12 tokens after ablation")
ax[1].set_title("only the block-0 MLP flattens the ordering (dotted: unablated)")
ax[1].legend(fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(f"{PLOTS}/second_repl.png", dpi=150)
plt.close(fig)
print("wrote plots/second_repl.png")

# ---------------------------------------------------------------- Figure: matched control, 410M
C = json.load(open(f"{RESULTS}/second_ctrl_410m.json"))
base = np.array(C["base_w"])
alphas = C["alphas"]
mlp = [[r for r in C["rows"] if r["seed"] is None and r["alpha"] == a][0] for a in alphas]
ctl = [[r for r in C["rows"] if r["seed"] is not None and r["alpha"] == a] for a in alphas]
bits = np.array([r["bits"] for r in mlp])
rho_m = np.array([r["rho"] for r in mlp])
rho_c = np.array([np.mean([r["rho"] for r in g]) for g in ctl])
rho_c_sd = np.array([np.std([r["rho"] for r in g], ddof=1) for g in ctl])

D14 = json.load(open(f"{RESULTS}/dose2.json"))
a14 = sorted({r["alpha"] for r in D14["rows"]})
m14 = [[r for r in D14["rows"] if r["match"] == "none" and r["alpha"] == a][0] for a in a14]
c14 = [[r for r in D14["rows"] if r["match"] == "per_token" and r["alpha"] == a] for a in a14]
b14 = np.array([r["bits"] for r in m14])
rm14 = np.array([r["rho"] for r in m14])
rc14 = np.array([np.mean([r["rho"] for r in g]) for g in c14])

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax[0].plot(bits, rho_m, "-o", color=CVD[0], label="410M: block-0 MLP dose")
ax[0].errorbar(bits, rho_c, yerr=rho_c_sd, fmt="--s", color=CVD[1], capsize=3,
               label="410M: movement-matched random control")
ax[0].plot(b14, rm14, ":^", color=CVD[2], lw=1.2, ms=4, label="1.4B: block-0 MLP dose")
ax[0].plot(b14, rc14, "-.D", color=CVD[3], lw=1.2, ms=4, label="1.4B: matched control")
ax[0].axhline(1.0, color="0.75", ls="-.", lw=1)
ax[0].set_xscale("log")
ax[0].set_xlabel("output movement (bits, log scale)")
ax[0].set_ylabel(r"$\rho$(width before, width after)")
ax[0].set_title("does the dose damage the ordering more than a matched disturbance?")
ax[0].legend(fontsize=7.5, loc="lower left")

pw = np.array([p["p_centred"] for p in C["paired"]])
dm = np.array([p["mlp_move_centred"] for p in C["paired"]])
dc = np.array([p["ctrl_move_centred"] for p in C["paired"]])
i = np.arange(len(alphas))
ax[1].bar(i - 0.18, dm, 0.36, color=CVD[0], hatch="//", edgecolor="white",
          label="block-0 MLP dose")
ax[1].bar(i + 0.18, dc, 0.36, color=CVD[1], hatch="..", edgecolor="white",
          label="matched random control")
for k in i:
    ax[1].text(k, max(dm[k], dc[k]) + 0.002, f"p={pw[k]:.2f}", ha="center", fontsize=7)
ax[1].set_xticks(i)
ax[1].set_xticklabels([f"{b:.3f}" for b in bits])
ax[1].set_xlabel("output movement (bits)")
ax[1].set_ylabel(r"mean $|\Delta \hat w_u -$ arm mean$|$")
ax[1].set_title("level-free per-token movement, Pythia-410M")
ax[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{PLOTS}/second_ctrl.png", dpi=150)
plt.close(fig)
print("wrote plots/second_ctrl.png")

print("\n410M matched control, per rung:")
for k, a in enumerate(alphas):
    print(f"  alpha {a}: bits {bits[k]:.4f}  rho_mlp {rho_m[k]:+.2f}  "
          f"rho_ctrl {rho_c[k]:+.2f} +- {rho_c_sd[k]:.2f}  "
          f"p_centred {pw[k]:.3f}")
live = [k for k, b in enumerate(bits) if b <= 0.05]
below = sum(1 for k in live for r in ctl[k] if rho_m[k] < r["rho"])
print(f"  live band (<= 0.05 bits, {len(live)} rungs): MLP below its control in "
      f"{below}/{len(live) * len(ctl[0])} rung x seed comparisons")
