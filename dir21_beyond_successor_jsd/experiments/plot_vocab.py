"""Figure for the vocabulary-wide test of the embedding lookup."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CVD, PLOTS, RESULTS

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9

CLS = [("word", "words outside the pool", "o"), ("fragment", "subword fragments", "s"),
       ("symbol", "punctuation / numerals", "^"), ("capitalised", "capitalised names", "D")]


def main():
    d = json.load(open(f"{RESULTS}/vocab.json"))
    w0 = {s: float(np.nanmedian(v["w"]))
          for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    rows = d["tokens"]

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.0))

    for k, (cls, lab, mk) in enumerate(CLS):
        xs = [rows[s]["pred"] for s in rows if rows[s]["cls"] == cls]
        ys = [rows[s]["w_med"] for s in rows if rows[s]["cls"] == cls]
        ax[0].scatter(xs, ys, s=34, marker=mk, color=CVD[k], alpha=.85,
                      edgecolor="white", linewidth=.4, label=lab)
    lo = min(min(rows[s]["pred"] for s in rows), min(rows[s]["w_med"] for s in rows)) - .02
    hi = max(max(rows[s]["pred"] for s in rows), max(rows[s]["w_med"] for s in rows)) + .02
    ax[0].plot([lo, hi], [lo, hi], ls="--", color="0.5", lw=.9)
    ax[0].axhspan(min(w0.values()), max(w0.values()), color="0.85", alpha=.45, zorder=0)
    ax[0].annotate("range of the 123 pool tokens", (hi - .005, min(w0.values())), fontsize=6.5,
                   ha="right", va="bottom", color="0.35")
    ax[0].set_xlim(lo, hi)
    ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel(r"$\hat w_u$ predicted from the static embedding (lookup)")
    ax[0].set_ylabel(r"$\hat w_u$ measured at block 0")
    ax[0].set_title(fr"32 tokens from outside the pool: $\rho = {d['rho_all'][0]:+.2f}$ "
                    f"(p = {d['rho_all'][1]:.0e})", fontsize=9)
    ax[0].legend(fontsize=7, loc="upper left")

    groups = [("123 pool tokens", list(w0.values()), "..")] + [
        (lab, [rows[s]["w_med"] for s in rows if rows[s]["cls"] == cls], h)
        for (cls, lab, _), h in zip(CLS, ["//", "\\\\", "xx", "--"])]
    bp = ax[1].boxplot([g[1] for g in groups], widths=.6, patch_artist=True,
                       medianprops=dict(color="0.15"))
    for k, (patch, g) in enumerate(zip(bp["boxes"], groups)):
        patch.set_facecolor(CVD[(k - 1) % len(CVD)] if k else "0.75")
        patch.set_alpha(.75)
        patch.set_hatch(g[2])
        patch.set_edgecolor("white")
    for k, g in enumerate(groups):
        ax[1].scatter(np.full(len(g[1]), k + 1) + np.linspace(-.12, .12, len(g[1])), g[1],
                      s=6, color="0.2", alpha=.5, zorder=3)
    ax[1].set_xticks(range(1, len(groups) + 1))
    ax[1].set_xticklabels([g[0].replace(" ", "\n", 1) for g in groups], fontsize=7)
    ax[1].axhline(0.8, ls="-.", color="0.4", lw=.9)
    ax[1].annotate("$w = 0.8$: proportional response", (len(groups) + .45, .8), fontsize=6.5,
                   ha="right", va="bottom", color="0.35")
    ax[1].set_ylabel(r"measured anchor width $\hat w_u$")
    ax[1].set_title("Where each token class sits", fontsize=9)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(PLOTS, "vocab_probe.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/vocab_probe.png")


if __name__ == "__main__":
    main()
