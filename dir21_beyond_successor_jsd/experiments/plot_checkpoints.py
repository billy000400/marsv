"""Figures for the training-checkpoint sweep (results/checkpoints_summary.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS, CVD

S = json.load(open(f"{RESULTS}/checkpoints_summary.json"))
R = S["rows"]
step = np.array([r["step"] for r in R], float)
FIRST = step[step > 0].min()
X = np.where(step == 0, FIRST / 2.5, step)       # step 0 drawn off-scale on the left of the log axis
LAB = ["0" if s == 0 else (f"{s / 1000:g}k" if s >= 1000 else f"{s:g}") for s in step]
get = lambda k: np.array([r[k] for r in R], float)
FINAL = len(R) - 1                               # the final checkpoint compares with itself


def xaxis(ax):
    ax.set_xscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels(LAB, fontsize=7, rotation=45)
    ax.minorticks_off()
    ax.axvline(FIRST / 1.6, color="0.8", lw=1)   # visual break between init and trained checkpoints
    ax.set_xlabel("training step (log scale; step 0 = random initialisation, off-scale)")


# ------------------------------------------------- Figure: when does the ordering appear?
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

rho, dis, rel = get("rho_final"), get("disattenuated"), get("reliability")
m = np.ones(len(R), bool)
m[FINAL] = False                                  # self-comparison carries no information
ax[0].plot(X[m], rho[m], "-o", color=CVD[0], label=r"agreement with step143000, raw $\rho$")
ax[0].plot(X[m], dis[m], "--s", color=CVD[1], label="same, divided by the noise ceiling")
ax[0].plot(X, rel, ":^", color=CVD[2], label="measurement reliability of this checkpoint")
ax[0].axhline(1.0, color="0.75", ls="-.", lw=1)
ax[0].axhline(0.0, color="0.75", lw=0.8)
ax[0].set_ylim(-0.15, 1.15)
ax[0].set_ylabel(r"Spearman $\rho$ over the 123 tokens")
ax[0].set_title("the ordering appears between step 16 and step 512")
ax[0].legend(fontsize=8, loc="lower right")
xaxis(ax[0])

med, sd = get("median_w"), get("sd_w")
ax[1].plot(X, med, "-o", color=CVD[0], label=r"median $\hat w_u$ (level)")
ax[1].fill_between(X, med - sd, med + sd, color=CVD[0], alpha=0.15)
ax2 = ax[1].twinx()
ax2.plot(X, sd, "--s", color=CVD[1], label=r"sd of $\hat w_u$ across tokens (spread)")
ax2.set_ylabel(r"sd of $\hat w_u$ across the 123 tokens", color=CVD[1])
ax2.set_ylim(0, 0.09)
ax[1].set_ylim(0.5, 0.9)
ax[1].set_ylabel(r"median anchor width $\hat w_u$")
ax[1].set_title("spread appears first, then the level sharpens")
h1, l1 = ax[1].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
xaxis(ax[1])

fig.tight_layout()
fig.savefig(f"{PLOTS}/ckpt_emergence.png", dpi=150)
plt.close(fig)

# ------------------------------------------------- Figure: is it a corpus statistic?
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax[0].plot(X, -get("rho_logcount"), "-o", color=CVD[0],
           label=r"$-\rho$ with log$_{10}$ unigram count")
ax[0].plot(X, -get("rho_entropy"), "--s", color=CVD[1],
           label=r"$-\rho$ with successor entropy")
ax[0].plot(X[m], get("partial_final")[m], ":^", color=CVD[2],
           label="agreement with step143000, both partialled out")
ax[0].plot(X[m], rho[m], "-.D", color=CVD[3], label="agreement with step143000, raw")
ax[0].axhline(0.0, color="0.75", lw=0.8)
ax[0].set_ylim(-0.15, 1.05)
ax[0].set_ylabel(r"Spearman $\rho$ over the 123 tokens")
ax[0].set_title("frequency first, then the part frequency cannot explain")
ax[0].legend(fontsize=8, loc="lower right")
xaxis(ax[0])

pr, ps = get("probe_rho"), get("probe_sd")
ax[1].plot(X, pr, "-o", color=CVD[0], label="probe refitted inside this checkpoint")
ax[1].fill_between(X, pr - ps, pr + ps, color=CVD[0], alpha=0.15)
ax[1].plot(X, get("rho_lookup14"), "--s", color=CVD[1],
           label="fixed lookup read off Pythia-1.4B's embeddings")
ax[1].plot(X, get("probe_null"), ":^", color=CVD[2], label="shuffled-target control")
ax[1].axhline(0.0, color="0.75", lw=0.8)
ax[1].set_ylim(-0.25, 1.0)
ax[1].set_ylabel(r"Spearman $\rho$ with this checkpoint's measured $\hat w_u$")
ax[1].set_title("a mature model's lookup reads it first")
ax[1].legend(fontsize=8, loc="lower right")
xaxis(ax[1])

fig.tight_layout()
fig.savefig(f"{PLOTS}/ckpt_source.png", dpi=150)
plt.close(fig)

print("wrote plots/ckpt_emergence.png, plots/ckpt_source.png")
