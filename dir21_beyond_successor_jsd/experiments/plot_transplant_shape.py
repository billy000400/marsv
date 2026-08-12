"""Figure 33: does the m_u transplant move the recipient's curve shape as well as its width?

(a) and (b) are the same 132 cross transplants scored two ways: the recipient's post-transplant width,
and its post-transplant edge drift, each against the donor's own baseline value. One thin line per
recipient joins its 11 donors. (c) is the paired per-recipient comparison of the two scorings.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, PLOTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
W_C, E_C = CVD[0], CVD[1]


def panel(ax, res, key, base_by, color, marker, label, xlabel, ylabel):
    toks = res["tokens"]
    rows = [x for x in res["transplant"] if x["donor"] in toks and x["donor"] != x["recipient"]]
    for r in toks:
        rr = sorted([x for x in rows if x["recipient"] == r], key=lambda x: base_by[x["donor"]])
        xs = [base_by[x["donor"]] for x in rr]
        ys = [float(np.nanmedian(x[key])) for x in rr]
        ax.plot(xs, ys, "-", color=color, marker=marker, ms=3.0, lw=0.7, alpha=0.55)
    lim = [min(base_by.values()), max(base_by.values())]
    pad = 0.06 * (lim[1] - lim[0])
    ax.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad], "k--", lw=1.0,
            label="donor's own value (slope 1)")
    s = res["score"][key]
    ax.set_title(f"{label}\nmean per-recipient slope {s['mean_slope']:+.3f}, "
                 rf"$\rho$ {s['mean_rho']:+.3f}", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left")


def main():
    res = json.load(open(f"{RESULTS}/transplant_shape.json"))
    toks = res["tokens"]
    bw = dict(zip(toks, res["score"]["w"]["base"]))
    be = dict(zip(toks, res["score"]["edge"]["base"]))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    panel(axes[0], res, "w", bw, W_C, "o", "(a) width transports",
          "donor's own baseline width $w$", "recipient's post-transplant width $w$")
    panel(axes[1], res, "edge", be, E_C, "s", "(b) shape transports",
          "donor's own baseline edge drift $E$", "recipient's post-transplant edge drift $E$")

    ax = axes[2]
    stats = [("slope", "transported slope"), ("partial", "partial " + r"$\rho$" + "\n(other donor\nproperty held)")]
    for gi, (stat, name) in enumerate(stats):
        wv = np.array(res["score"]["w"][f"per_recipient_{stat}"])
        ev = np.array(res["score"]["edge"][f"per_recipient_{stat}"])
        x0, x1 = gi * 1.6, gi * 1.6 + 0.55
        for a, b in zip(wv, ev):
            ax.plot([x0, x1], [a, b], "-", color="0.6", lw=0.7, zorder=1)
        ax.plot([x0] * len(wv), wv, "o", color=W_C, ms=5, label="width $w$" if gi == 0 else None,
                zorder=2)
        ax.plot([x1] * len(ev), ev, "s", color=E_C, ms=5, label="edge drift $E$" if gi == 0 else None,
                zorder=2)
        p = res["paired"][stat]["wilcoxon_p"]
        ax.text(gi * 1.6 + 0.275, 1.09, f"p = {p:.4f}", ha="center", fontsize=7)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.text(0.5, 1.015, "complete transport", fontsize=6.5, ha="center",
            transform=ax.get_yaxis_transform())
    ax.set_xticks([0.275, 1.875])
    ax.set_xticklabels([s[1] for s in stats], fontsize=7.5)
    ax.set_xlim(-0.35, 2.3)
    ax.set_ylim(0.0, 1.16)
    ax.set_ylabel("value per recipient token (12 tokens)", fontsize=8)
    ax.set_title("(c) scored on width vs on shape,\npaired over the same 12 recipients", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    fig.savefig(f"{PLOTS}/transplant_shape.png", dpi=150)
    plt.close(fig)
    print(f"wrote {PLOTS}/transplant_shape.png")


if __name__ == "__main__":
    main()
