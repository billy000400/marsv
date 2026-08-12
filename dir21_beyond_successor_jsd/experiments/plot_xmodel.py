"""Figures for the cross-tokenizer test (results/xmodel_summary.json, gpt2_sites.json, xcurves.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS, CVD

S = json.load(open(f"{RESULTS}/xmodel_summary.json"))
names = S["tokens"]
raw = {t: json.load(open(f"{RESULTS}/xwidth_{t}.json"))["raw"] for t in S["tags"]}
W = {t: np.array([np.median(np.array(raw[t][s]["w_env"], float)) for s in names]) for t in S["tags"]}

# ------------------------------------------------- Figure A: curve shape and rank agreement
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))

C = json.load(open(f"{RESULTS}/xcurves.json"))
g = np.array(C["grid"])
for k, s in enumerate(C["tokens"][:3]):
    ax[0].plot(g, C["pythia"][s]["d"], "-", color=CVD[0], lw=1.6,
               label="Pythia-1.4B" if k == 0 else None)
    ax[0].plot(g, C["gpt2"][s]["d"], "--", color=CVD[1], lw=1.6,
               label="GPT-2 small" if k == 0 else None)
ax[0].axhline(0.1, color="0.8", lw=0.8)
ax[0].axhline(0.9, color="0.8", lw=0.8)
ax[0].set_xlabel("interpolation position $t$ (0 = token, 1 = anchor ' close')")
ax[0].set_ylabel("relative output distance $d(t)$")
ax[0].set_title("Pythia's curves rise once; GPT-2's wander")
ax[0].legend(fontsize=9, loc="lower right")

lim = (0.25, 0.95)
for k, (t, ttl) in enumerate([("410m", "Pythia-410M vs Pythia-1.4B"),
                              ("gpt2", "GPT-2 small vs Pythia-1.4B")]):
    a = ax[k + 1]
    key = f"{t}|1.4b" if t == "410m" else "gpt2|1.4b"
    p = S["pairwise"][key]
    a.plot(W["1.4b"], W[t], "o" if t == "410m" else "s", color=CVD[0] if t == "410m" else CVD[1],
           ms=4, alpha=0.75)
    a.plot(lim, lim, color="0.75", ls=":", lw=1)
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_xlabel(r"Pythia-1.4B envelope width $\hat w_u$")
    a.set_ylabel(rf"{'Pythia-410M' if t == '410m' else 'GPT-2 small'} envelope width $\hat w_u$")
    a.set_title(rf"{ttl}: $\rho$ = {p['rho']:+.2f}"
                f"\n(ceiling {p['ceiling']:.2f}, disattenuated {p['disattenuated']:+.2f}, "
                f"n = {p['n']})")

fig.tight_layout()
fig.savefig(f"{PLOTS}/xmodel_agreement.png", dpi=150)
plt.close(fig)

# ------------------------------------------------- Figure B: is it the site? GPT-2 depth sweep
R = json.load(open(f"{RESULTS}/gpt2_sites.json"))["rows"]
L = np.array([r["layer"] for r in R], float)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax[0].plot(L, [r["valid"] for r in R], "-o", color=CVD[0],
           label="fraction of curves passing dir18's strict validity")
ax[0].plot(L, [r["reliability"] for r in R], "--s", color=CVD[1],
           label="split-half reliability of the per-token width")
ax[0].axhline(1.0, color=CVD[0], ls=":", lw=1.2)
ax[0].axhline(S["reliability"]["1.4b"]["spearman_brown"], color=CVD[1], ls="-.", lw=1.2)
ax[0].text(8.1, 1.01, "Pythia-1.4B validity", fontsize=7, ha="right", color=CVD[0])
ax[0].text(8.1, S["reliability"]["1.4b"]["spearman_brown"] + 0.02, "Pythia-1.4B reliability",
           fontsize=7, ha="right", color=CVD[1])
ax[0].set_ylim(-0.1, 1.12)
ax[0].set_xlabel("GPT-2 block whose output is interpolated")
ax[0].set_ylabel("fraction / Spearman $\\rho$")
ax[0].set_title("GPT-2's curves clean up with depth,\nbut the measurement stays unreliable")
ax[0].legend(fontsize=8, loc="center right")

ax[1].plot(L, [r["rho_pythia"] for r in R], "-^", color=CVD[2],
           label=r"$\rho$ with Pythia-1.4B's ranking")
ax[1].plot(L, [r["median_w"] for r in R], "--D", color=CVD[3],
           label=r"median envelope width $\hat w_u$ (level)")
ax[1].axhline(0.0, color="0.75", lw=0.8)
ax[1].axhline(S["reliability"]["1.4b"]["median"], color=CVD[3], ls=":", lw=1.2)
ax[1].text(8.1, S["reliability"]["1.4b"]["median"] + 0.02, "Pythia-1.4B level", fontsize=7,
           ha="right", color=CVD[3])
ax[1].set_ylim(-0.35, 0.8)
ax[1].set_xlabel("GPT-2 block whose output is interpolated")
ax[1].set_ylabel(r"Spearman $\rho$ / width")
ax[1].set_title("no site in GPT-2 recovers Pythia's ordering")
ax[1].legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(f"{PLOTS}/gpt2_sites.png", dpi=150)
plt.close(fig)
print("wrote plots/xmodel_agreement.png, plots/gpt2_sites.png")
