"""Figures for the S1-S3 exploration. Headless; writes plots/*.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import CVD, PLOTS, RESULTS, load, token_index
from explore1 import contrasts, cv_r2

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9

GATE = 0.2


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), bbox_inches="tight")
    plt.close(fig)


def fig_scatter(t, con):
    m = t["out_jsd_min"] >= GATE
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax[0].scatter(t["jsd_B"][m], t["w"][m], s=9, marker="o", c=CVD[0], alpha=.35,
                  linewidths=0, label=f"passes movement gate (n={m.sum()})")
    ax[0].scatter(t["jsd_B"][~m], t["w"][~m], s=22, marker="s", facecolors="none",
                  edgecolors=CVD[1], linewidths=.9, label=f"gated out (n={(~m).sum()})")
    for c in con[:8]:
        i, j = c["narrow"], c["wide"]
        ax[0].plot([t["jsd_B"][i], t["jsd_B"][j]], [t["w"][i], t["w"][j]],
                   color="0.25", lw=1.1, zorder=3)
        ax[0].scatter([t["jsd_B"][i], t["jsd_B"][j]], [t["w"][i], t["w"][j]],
                      s=26, marker="D", c="0.15", zorder=4)
    ax[0].plot([], [], color="0.25", lw=1.1, marker="D", ms=4,
               label="top-8 matched contrasts")
    ax[0].axhline(0.8, ls=":", c="0.4", lw=1)
    ax[0].text(0.13, 0.807, "w = 0.8: output moves in proportion to t", fontsize=7, color="0.35")
    ax[0].set_xlabel("corpus successor JSD  $J(u,v)$  [bits]")
    ax[0].set_ylabel("transition width  $w$")
    ax[0].set_title("Width vs corpus JSD, 1,000 pairs")
    ax[0].legend(fontsize=7, loc="lower left", framealpha=.9)

    ax[1].scatter(t["out_jsd_med"][m], t["w"][m], s=9, marker="o", c=CVD[0], alpha=.35,
                  linewidths=0, label="passes gate")
    ax[1].scatter(t["out_jsd_med"][~m], t["w"][~m], s=22, marker="s", facecolors="none",
                  edgecolors=CVD[1], linewidths=.9, label="gated out")
    ax[1].axvline(GATE, ls="--", c="0.25", lw=1.2)
    ax[1].text(GATE + .01, 0.34, "gate: min-frame\nendpoint movement\n= 0.2 bits", fontsize=7)
    ax[1].set_xlabel(r"endpoint output movement  $JSD_{\mathrm{out}}$  [bits]")
    ax[1].set_ylabel("transition width  $w$")
    ax[1].set_title("Where normalised $w$ stops being informative")
    ax[1].legend(fontsize=7, loc="upper right", framealpha=.9)
    save(fig, "scatter_and_gate.png")


def fig_contrast_curves(t, curves, grid, con):
    top = []
    seen = set()
    for c in con:                                    # keep distinct pairs across panels
        if c["narrow"] in seen or c["wide"] in seen:
            continue
        seen |= {c["narrow"], c["wide"]}
        top.append(c)
        if len(top) == 3:
            break
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for ax, c in zip(axs, top):
        for k, (idx, col, ls, lab) in enumerate(((c["narrow"], CVD[0], "-", "narrow"),
                                                 (c["wide"], CVD[1], "--", "wide"))):
            for f in range(3):
                ax.plot(grid, curves[idx, f], color=col, ls=ls, lw=1.3, alpha=.85,
                        label=(lab + f" $w$={t['w'][idx]:.2f}") if f == 0 else None)
        ax.axhline(0.1, ls=":", c="0.6", lw=.8)
        ax.axhline(0.9, ls=":", c="0.6", lw=.8)
        ax.set_xlabel("interpolation position $t$")
        ax.set_title(f"{c['narrow_pair']}   vs\n{c['wide_pair']}\n"
                     f"$J$={c['jsd']:.2f}  "
                     r"$JSD_{\mathrm{out}}$=" f"{c['out_jsd_n']:.2f}/{c['out_jsd_w']:.2f} bits",
                     fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
    axs[0].set_ylabel("output-distance score  $d(t)$")
    save(fig, "contrast_curves.png")


def fig_cvr2(res, ceiling):
    names = ["corpus_jsd", "out_jsd", "pair_covariates", "token_additive",
             "token_additive_plus_jsd", "token_additive_plus_jsd_outjsd",
             "token_additive_plus_jsd_geometry"]
    lab = ["corpus JSD $J$", r"model-output JSD $JSD_{\mathrm{out}}$",
           "5 pair covariates + $J$", "token-additive only",
           "token-additive + $J$", "token-additive + $J$ + $JSD_{\\mathrm{out}}$",
           "token-additive + $J$ + $JSD_{\\mathrm{out}}$ + block-0 geometry"]
    v = [res[n] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    cols = [CVD[1] if "token" not in n else CVD[0] for n in names]
    hat = ["//" if "token" not in n else None for n in names]
    ax.barh(range(len(v)), v, color=cols, hatch=hat, edgecolor="white")
    for i, x in enumerate(v):
        ax.text(x + .008, i, f"{x:.3f}", va="center", fontsize=8)
    ax.axvline(ceiling, ls="--", c="0.2", lw=1.3)
    ax.text(ceiling - .01, -0.7, f"reproducibility ceiling {ceiling:.2f}",
            ha="right", fontsize=8)
    ax.bar(0, 0, color=CVD[1], hatch="//", edgecolor="white",
           label="pair-level predictors only")
    ax.bar(0, 0, color=CVD[0], label="includes the per-token term $a_u + a_v$")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_yticks(range(len(v)), lab, fontsize=8)
    ax.set_xlabel("held-out $R^2$ for $w$ (5-fold over pairs)")
    ax.set_xlim(0, 1)
    save(fig, "cv_r2.png")


def fig_token_effects(tok):
    a = np.array(tok["effect"])
    lf = np.array(tok["logf"])
    names = tok["tokens"]
    o = np.argsort(a)
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.9),
                           gridspec_kw=dict(width_ratios=[1.55, 1]))
    ax[0].plot(np.arange(len(a)), a[o], marker="o", ms=3, lw=1, color=CVD[0])
    for k in list(range(0, 8, 2)) + list(range(len(a) - 10, len(a), 3)):
        lo = k < 8
        ax[0].annotate(names[o[k]].strip(), (k, a[o][k]), fontsize=7,
                       ha="left" if lo else "right", va="center",
                       xytext=(8 if lo else -8, 0), textcoords="offset points",
                       color="0.25")
    ax[0].set_xlabel("token, ranked by fitted width contribution")
    ax[0].set_ylabel(r"token effect $a_u$ (width units)")
    ax[0].set_title("Each token carries its own width contribution")
    ax[1].scatter(lf, a, s=16, marker="^", c=CVD[2], alpha=.8, linewidths=0)
    r = spearmanr(lf, a)
    ax[1].set_xlabel(r"token corpus log$_{10}$ frequency")
    ax[1].set_ylabel(r"token effect $a_u$")
    ax[1].set_title(f"Frequency explains only part of it\nSpearman $\\rho$={r[0]:+.2f}  "
                    f"$p$={r[1]:.1e}")
    save(fig, "token_effects.png")


def fig_pred(t, m):
    y = t["w"][m]
    toks, ia, ib = token_index(t)
    ia, ib = ia[m], ib[m]
    n = len(y)
    T = np.zeros((n, len(toks)))
    T[np.arange(n), ia] += 1
    T[np.arange(n), ib] += 1
    one = np.ones((n, 1))
    J = t["jsd_B"][m][:, None]
    r_j, p_j = cv_r2(np.hstack([one, J]), y)
    r_t, p_t = cv_r2(np.hstack([one, T, J]), y)
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 4), sharex=True, sharey=True)
    for a, p, r, ti, col, mk in ((ax[0], p_j, r_j, "corpus JSD only", CVD[1], "s"),
                                 (ax[1], p_t, r_t, "token-additive + corpus JSD", CVD[0], "o")):
        a.scatter(p, y, s=9, marker=mk, c=col, alpha=.35, linewidths=0)
        a.plot([.3, .85], [.3, .85], ls="--", c="0.3", lw=1)
        a.set_title(f"{ti}\nheld-out $R^2$ = {r:.3f}", fontsize=9)
        a.set_xlabel("predicted $w$ (held out)")
    ax[0].set_ylabel("observed $w$")
    save(fig, "prediction.png")


def main():
    t, curves, grid = load()
    res = json.load(open(os.path.join(RESULTS, "explore1.json")))
    m = t["out_jsd_min"] >= GATE
    con = contrasts(t, GATE)
    fig_scatter(t, con)
    fig_contrast_curves(t, curves, grid, con)
    fig_cvr2(res["cv_r2"], res["reliability"]["ceiling_r2"])
    fig_token_effects(res["token_effects"])
    fig_pred(t, m)
    print("wrote 5 figures")


if __name__ == "__main__":
    main()
