"""Figure for neuron_feature.py: what the path-bending units detect, and whether that predicts
which interpolation paths recruit them.

Four panels:
  A  AUROC per pair for each ranking rule (corpus-tuning rules vs the pair-blind controls);
  B  recruitment rate against the differential-tuning decile (calibration);
  C  tuning sharpness of the 668 pool units vs the other 3,172;
  D  rank-ordered character-tuning profiles of the three most-reused units, against the pool and
     non-pool medians.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD, LINESTYLES, MARKERS, REF_DIAG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

S = json.load(open(os.path.join(RES, "neuron_feature_summary.json")))
R = np.load(os.path.join(RES, "neuron_feature_raw.npz"), allow_pickle=True)
z, ok_chars = R["z_ok"], R["char_n"] >= S["restricted"]["min_char_n"]
pool = R["pool"]

fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.6))

# ---- A: AUROC by ranking rule ----------------------------------------------------------------
methods = ["differential", "max", "global_imp", "mean_act", "random"]
names = ["tuning\n$|z_a-z_b|$", "tuning\n$\\max(z_a,z_b)$",
         "global\nimportance", "overall\nactivity", "random"]
rng = np.random.default_rng(0)
for i, m in enumerate(methods):
    v = R[f"auroc_{m}"]
    ax[0].scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=7, alpha=0.35,
                  color=CVD[i % len(CVD)], marker=MARKERS[i % len(MARKERS)], lw=0)
    ax[0].plot([i - 0.32, i + 0.32], [v.mean()] * 2, color="k", lw=2.4)
    ax[0].annotate(f"{v.mean():.3f}", (i, v.mean()), textcoords="offset points",
                   xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")
ax[0].axhline(0.5, **REF_DIAG)
ax[0].text(-0.42, 0.512, "chance", fontsize=8, color="0.35", ha="left")
ax[0].set_xticks(range(len(methods)))
ax[0].set_xticklabels(names, fontsize=8.5)
ax[0].set_ylabel("AUROC: does the ranking find that pair's top-32?")
ax[0].set_ylim(0.35, 1.02)
ax[0].set_title("A. Natural-text tuning predicts recruitment")

# ---- B: calibration by differential-tuning decile ---------------------------------------------
dec = np.array(S["decile_recruit_rate"])
ax[1].bar(np.arange(1, 11), 100 * dec, color=CVD[0], edgecolor="k", hatch="//", lw=0.8)
ax[1].axhline(100 * S["chance_recruit_rate"], **REF_DIAG)
ax[1].text(10.4, 100 * S["chance_recruit_rate"] * 1.15, "chance (0.83%)", fontsize=8,
           color="0.35", ha="right")
ax[1].set_yscale("log")
ax[1].set_xlabel("decile of differential tuning $|z_a-z_b|$ (1 = strongest)")
ax[1].set_ylabel("% of units recruited into the pair's top-32")
ax[1].set_title("B. Recruitment falls off with tuning")

# ---- C: tuning sharpness, pool vs the rest ----------------------------------------------------
sharp = R["sharp"]
bins = np.linspace(0, sharp.max(), 50)
ax[2].hist(sharp[~pool], bins=bins, density=True, color=CVD[3], alpha=0.75, hatch="..",
           edgecolor="k", lw=0.5, label=f"never recruited (n={int((~pool).sum())})")
ax[2].hist(sharp[pool], bins=bins, density=True, color=CVD[1], alpha=0.6, hatch="\\\\",
           edgecolor="k", lw=0.5, label=f"pool: recruited by ≥1 pair (n={int(pool.sum())})")
for k, (v, c, ls) in enumerate([(S["tuning_sharpness"]["nonpool_median"], CVD[3], "--"),
                                (S["tuning_sharpness"]["pool_median"], CVD[1], "-")]):
    ax[2].axvline(v, color=c, ls=ls, lw=2)
ax[2].set_xlabel("tuning sharpness $\\max_c |z_c(j)|$ of unit $j$")
ax[2].set_ylabel("density of units")
ax[2].legend(fontsize=8, loc="upper right")
ax[2].set_title("C. Recruited units are more sharply tuned")

# ---- D: tuning profiles of the most-reused units ----------------------------------------------
ex = S["examples"][:3]
for i, e in enumerate(ex):
    prof = np.sort(z[ok_chars, e["unit"]])[::-1]
    ax[3].plot(np.arange(1, len(prof) + 1), prof, color=CVD[i], ls=LINESTYLES[i], lw=2,
               label=f"unit {e['unit']} (block {e['block']}), top char '{e['top_chars'][0]}', "
                     f"{e['n_pairs_recruiting']}/150 pairs")
prof_pool = np.median(np.sort(z[np.ix_(ok_chars, pool)], axis=0)[::-1], axis=1)
prof_non = np.median(np.sort(z[np.ix_(ok_chars, ~pool)], axis=0)[::-1], axis=1)
ax[3].plot(np.arange(1, len(prof_pool) + 1), prof_pool, color="0.25", ls="-", lw=1.6,
           marker="^", ms=3.5, label="pool median")
ax[3].plot(np.arange(1, len(prof_non) + 1), prof_non, color="0.55", ls=":", lw=1.6,
           label="never-recruited median")
ax[3].axhline(0, color="0.7", lw=0.8)
ax[3].set_xlabel("characters ranked by that unit's own tuning $z_c(j)$")
ax[3].set_ylabel("standardized mean activation $z_c(j)$")
ax[3].legend(fontsize=7.5, loc="upper right")
ax[3].set_title("D. Tuning is sharp and character-specific")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "neuron_feature.png"), dpi=150)
plt.close(fig)
print("wrote plots/neuron_feature.png")
