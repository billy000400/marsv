"""Figures for Experiment 2 — fixed endpoint pair, four context classes."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvd_style import CVD, LINESTYLES, MARKERS, REF_DIAG, use_cvd
from sweep import RES, load

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "..", "plots")
CLASSES = ["none", "random", "unrelated", "relevant"]
PAIRS = ["big->large", "big->in"]
CSTYLE = {cl: dict(color=CVD[i], ls=LINESTYLES[i]) for i, cl in enumerate(CLASSES)}


def class_of(cid):
    return "".join(ch for ch in cid if not ch.isdigit())


def fig_raw(curves, ts, ctx, L=0):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.0), sharex=True, sharey=True)
    for r, p in enumerate(PAIRS):
        for c, cl in enumerate(CLASSES):
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], **REF_DIAG)
            cids = [k for k in ctx if class_of(k) == cl]
            for j, cid in enumerate(cids):
                ax.plot(ts, curves[f"{cid}|{p}|L{L}|logits"], lw=1.8, marker=MARKERS[j % 4],
                        ms=3, markevery=6, label=f"'{ctx[cid]['prefix']}'" or "(empty)",
                        **CSTYLE[cl])
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
            if r == 0:
                ax.set_title(f"{cl} context", fontsize=10)
            if r == 1:
                ax.set_xlabel("interpolation step $t$")
        axes[r, 0].set_ylabel(f"{p.replace('->', ' → ')}\nrelative distance $d(t)$")
    fig.suptitle(f"Final-logit $d(t)$, patch at block {L}: same endpoint pair, four context classes\n"
                 "(grey dashed = straight-line reference; legend gives the exact frozen prefix)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fixed_transition_contexts.png"), dpi=150)
    plt.close(fig)


def fig_by_layer(head, ctx, n_layer=36):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    Ls = np.arange(n_layer)
    for ax, p in zip(axes, PAIRS):
        for cl in CLASSES:
            cids = [k for k in ctx if class_of(k) == cl]
            W = np.array([[head["exp2"][f"{cid}|{p}"][str(L)]["width"] for L in Ls]
                          for cid in cids])
            for row in W:
                ax.plot(Ls, row, lw=0.8, alpha=0.45, **CSTYLE[cl])
            ax.plot(Ls, np.median(W, axis=0), lw=2.4, marker=MARKERS[CLASSES.index(cl)],
                    ms=4, markevery=3, label=f"{cl} (median of {len(cids)})", **CSTYLE[cl])
        ax.axhline(0.8, **REF_DIAG)
        ax.set_title(p.replace("->", " → "), fontsize=11)
        ax.set_xlabel("interpolation (patched) block $L$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("final-logit transition width $w$")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Context effect on plateau sharpness at every interpolation block\n"
                 "(thin lines = individual frozen prefixes; dashed = straight-line value 0.8)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "context_effect_by_layer.png"), dpi=150)
    plt.close(fig)


def fig_strip(head, ctx, L=0):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for pi, p in enumerate(PAIRS):
        for ci, cl in enumerate(CLASSES):
            cids = [k for k in ctx if class_of(k) == cl]
            w = [head["exp2"][f"{cid}|{p}"][str(L)]["width"] for cid in cids]
            x = ci + (pi - 0.5) * 0.32
            ax.scatter([x + 0.03 * (j - 1.5) for j in range(len(w))], w, s=46,
                       marker=MARKERS[pi], facecolor="none" if pi else CVD[pi],
                       edgecolor=CVD[pi], linewidths=1.6, zorder=3,
                       label=p.replace("->", " → ") if ci == 0 else None)
            ax.plot([x - 0.11, x + 0.11], [np.median(w)] * 2, color=CVD[pi], lw=2.5, zorder=2)
    ax.axhline(0.8, **REF_DIAG)
    ax.text(3.42, 0.815, "straight line", fontsize=8, color="0.35", ha="right")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_xlabel("context class")
    ax.set_ylabel("final-logit transition width $w$")
    ax.set_title(f"Transition width by context class (patch at block {L})\n"
                 "each point = one frozen prefix; bar = class median", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "context_width_summary.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    use_cvd()
    curves, meta = load("exp2")
    with open(os.path.join(RES, "headline.json")) as f:
        head = json.load(f)
    ts = np.array(meta["ts"])
    ctx = meta["contexts"]
    ctx["none"]["prefix"] = "(no context)"
    fig_raw(curves, ts, ctx, L=head["headline_layer"])
    fig_by_layer(head, ctx, meta["n_layer"])
    fig_strip(head, ctx, L=head["headline_layer"])
    print("wrote 3 figures to", PLOTS)


if __name__ == "__main__":
    main()
