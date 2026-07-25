"""Figure for the confirmatory replication: bank 1 (n=4/class) vs bank 2 (n=8/class)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvd_style import CVD, LINESTYLES, MARKERS, REF_DIAG, use_cvd
from sweep import load_meta

PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
CLASSES = ["none", "random", "unrelated", "relevant"]
PAIR = "big->in"


def widths(meta, cls, L=0):
    return [meta["summaries"][f"{cid}|{PAIR}|L{L}|logits"]["width"]
            for cid, c in meta["contexts"].items() if c["class"] == cls]


def main():
    os.makedirs(PLOTS, exist_ok=True)
    use_cvd()
    m1, m2 = load_meta("exp2"), load_meta("bank2")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    for bi, (name, meta, n) in enumerate((("bank 1 (n=4/class)", m1, 4),
                                          ("bank 2 (n=8/class)", m2, 8))):
        for ci, cl in enumerate(CLASSES):
            w = widths(meta, cl)
            if not w:
                continue
            x = ci + (bi - 0.5) * 0.34
            ax.scatter(x + np.linspace(-0.07, 0.07, len(w)), w, s=42, marker=MARKERS[bi],
                       facecolor="none" if bi else CVD[bi], edgecolor=CVD[bi], linewidths=1.5,
                       zorder=3, label=name if ci == 1 else None)
            ax.plot([x - 0.12, x + 0.12], [np.median(w)] * 2, color=CVD[bi], lw=2.5, zorder=2)
    ax.axhline(0.8, **REF_DIAG)
    ax.text(3.45, 0.815, "straight line", fontsize=8, color="0.35", ha="right")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_xlabel("context class")
    ax.set_ylabel("final-logit transition width $w$")
    ax.set_title("big → in, patch at block 0: two independent prefix banks\n"
                 "(bank 2 has no prefix in common with bank 1)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="center right")

    ax = axes[1]
    Ls = np.arange(m2["n_layer"])
    for ci, cl in enumerate(CLASSES[1:], start=1):
        W = np.array([[m2["summaries"][f"{cid}|{PAIR}|L{L}|logits"]["width"] for L in Ls]
                      for cid, c in m2["contexts"].items() if c["class"] == cl])
        ax.plot(Ls, np.median(W, axis=0), color=CVD[ci], ls=LINESTYLES[ci], lw=2.2,
                marker=MARKERS[ci], ms=4, markevery=3, label=f"{cl} (median of 8)")
        ax.fill_between(Ls, W.min(0), W.max(0), color=CVD[ci], alpha=0.13)
    ax.axhline(0.8, **REF_DIAG)
    ax.set_xlabel("interpolation (patched) block $L$")
    ax.set_ylabel("final-logit transition width $w$")
    ax.set_title("bank 2 across the whole layer sweep\n(shaded band = min–max over the 8 prefixes)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "context_bank2_replication.png"), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(PLOTS, "context_bank2_replication.png"))


if __name__ == "__main__":
    main()
