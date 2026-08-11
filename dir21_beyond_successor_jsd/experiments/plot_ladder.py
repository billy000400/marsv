"""Figure for the displacement-norm ladder: quiet against loud directions as the step grows."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import CVD, PLOTS, RESULTS

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9

STYLE = [("quiet", "o", "-", "quietest of 24 directions"),
         ("loud", "s", "--", "loudest of 24 directions"),
         ("random", "^", ":", "plain random direction")]


def main():
    d = json.load(open(f"{RESULTS}/ladder.json"))
    rows, norms = d["tokens"], d["norms"]
    keys = [f"{c:g}" for c in norms]
    base = np.array([rows[s]["base_w"] for s in rows])

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2))

    for k, (tag, mk, ls, lab) in enumerate(STYLE):
        m = [np.mean([rows[s]["rungs"][q][tag]["w"] for s in rows]) for q in keys]
        e = [np.std([rows[s]["rungs"][q][tag]["w"] for s in rows], ddof=1) / np.sqrt(len(rows))
             for q in keys]
        ax[0].errorbar(norms, m, yerr=e, marker=mk, ls=ls, color=CVD[k], lw=1.4, ms=5,
                       capsize=2.5, label=lab)
        ax[0].annotate(tag, (norms[-1] * 1.04, m[-1]), fontsize=7, color=CVD[k], va="center")
    ax[0].axhline(base.mean(), color="0.45", lw=1.0, ls="-.")
    ax[0].annotate(fr"unedited mean $\hat w_u$ = {base.mean():.3f}", (norms[1], base.mean()),
                   fontsize=6.5, color="0.35", va="top")
    ax[0].set_xscale("log")
    ax[0].set_xlim(norms[0] * 0.8, norms[-1] * 1.45)
    ax[0].set_xticks(norms)
    ax[0].set_xticklabels([f"{c:g}" for c in norms])
    ax[0].set_xticks([], minor=True)
    ax[0].set_xlabel("displacement norm of the embedding edit (log scale)")
    ax[0].set_ylabel(r"mean $\hat w_u$ after the edit")
    ax[0].set_title("Quiet directions hold width to larger displacements\n"
                    "than loud ones (mean over 12 tokens, error bars 1 s.e.)", fontsize=9)
    ax[0].legend(fontsize=7, loc="upper left")

    top = keys[-1]
    lab_r = {}
    for k, (tag, mk, ls, lab) in enumerate(STYLE[:2]):
        w = np.array([rows[s]["rungs"][top][tag]["w"] for s in rows])
        b = np.median([rows[s]["rungs"][top][tag]["bits"] for s in rows])
        rr = spearmanr(base, w)
        lab_r[tag] = rr
        ax[1].scatter(base, w, s=32, marker=mk, color=CVD[k], alpha=.85, edgecolor="none",
                      label=f"{lab} ({b:.3f} bits), " + fr"$\rho = {rr[0]:+.2f}$")
        ax[1].plot(base, np.poly1d(np.polyfit(base, w, 1))(base), ls=ls, color=CVD[k], lw=1.3)
    lo, hi = base.min() - .03, 0.75
    ax[1].plot([lo, hi], [lo, hi], ls=":", color="0.45", lw=1.1)
    ax[1].annotate("no change", (hi * .99, hi * .99), fontsize=6.5, ha="right", va="top",
                   color="0.35", rotation=36)
    ax[1].set_xlim(lo, hi)
    ax[1].set_ylim(lo, hi)
    ax[1].set_xlabel(r"$\hat w_u$ before the edit")
    ax[1].set_ylabel(fr"$\hat w_u$ after an edit of norm {norms[-1]:g}")
    ax[1].set_title(f"At the top rung (norm {norms[-1]:g}) the quiet direction keeps the\n"
                    "token ordering; the loud one destroys it", fontsize=9)
    ax[1].legend(fontsize=7, loc="upper left")
    bits = [rows[s]["rungs"][q][t]["bits"] for s in rows for q in keys for t, _, _, _ in STYLE]
    dw = [rows[s]["rungs"][q][t]["dw"] for s in rows for q in keys for t, _, _, _ in STYLE]
    r = spearmanr(bits, dw)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(PLOTS, "ladder.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"rho(bits, dw) over {len(bits)} edits: {r[0]:+.3f} (p={r[1]:.1e})")
    for q in keys:
        o = d["by_norm"][q]
        print(f"norm {q:>4}: " + "  ".join(
            f"{t} {o['bits'][t]:.4f}b w {o['w'][t]:.3f}+-{o['w_sd'][t]:.3f}"
            for t in ("quiet", "loud", "random"))
            + f" | quiet-vs-loud p={o['quiet_vs_loud_p']:.3f}"
              f" rho(bits,dw)={o['rho_bits_dw'][0]:+.2f} (p={o['rho_bits_dw'][1]:.2f})"
              f" rho(base,after) quiet {o['rho_base']['quiet'][0]:+.2f}"
              f" loud {o['rho_base']['loud'][0]:+.2f}")
    print("wrote plots/ladder.png")


if __name__ == "__main__":
    main()
