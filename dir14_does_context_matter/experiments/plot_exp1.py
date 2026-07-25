"""Figures for Experiment 1 — fixed context `The house was`, two endpoint pairs."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvd_style import CVD, LINESTYLES, REF_DIAG, use_cvd
from sweep import load

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "..", "plots")
PAIRS = ["big->large", "big->in"]
STYLE = {p: dict(color=CVD[i], ls=LINESTYLES[i], lw=2.0) for i, p in enumerate(PAIRS)}


def fig_raw_curves(curves, ts, layers=(0, 6, 12, 18, 24, 30)):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    for ax, L in zip(axes.ravel(), layers):
        ax.plot([0, 1], [0, 1], **REF_DIAG)
        for p in PAIRS:
            ax.plot(ts, curves[f"{p}|L{L}|logits"], label=p.replace("->", " → "), **STYLE[p])
        ax.set_title(f"patch at block {L}", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("interpolation step $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("relative distance $d(t)$")
    axes[0, 0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Final-logit $d(t)$ under the fixed context 'The house was'\n"
                 "(grey dashed = straight-line reference, no plateau)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fixed_context_endpoint_pairs.png"), dpi=150)
    plt.close(fig)


def fig_width_by_layer(summ, n_layer=36):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    Ls = np.arange(n_layer)
    for p in PAIRS:
        w = [summ[f"{p}|L{L}|logits"]["width"] for L in Ls]
        loc = [summ[f"{p}|L{L}|logits"]["location"] for L in Ls]
        lab = p.replace("->", " → ")
        axes[0].plot(Ls, w, marker="o", ms=3.5, **STYLE[p], label=lab)
        axes[1].plot(Ls, loc, marker="s", ms=3.5, **STYLE[p], label=lab)
        pl = [L for L in Ls if summ[f"{p}|L{L}|logits"]["plateau"]]
        if pl:
            axes[0].plot(pl, [w[L] for L in pl], "o", ms=8, mfc="none",
                         mec=STYLE[p]["color"], mew=1.6)
    axes[0].axhline(0.8, **REF_DIAG)
    axes[0].set_ylabel("transition width $w$")
    axes[0].set_title("width of the final-logit transition\n(open ring = frozen plateau rule fires;"
                      " dashed = straight-line value 0.8)", fontsize=9)
    axes[1].axhline(0.5, **REF_DIAG)
    axes[1].set_ylabel("transition location $t_{1/2}$")
    axes[1].set_title("location of the final-logit transition", fontsize=9)
    for ax in axes:
        ax.set_xlabel("interpolation (patched) block $L$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "exp1_width_by_layer.png"), dpi=150)
    plt.close(fig)


def fig_depth_emergence(summ, interp_layers=(0, 10, 20), n_layer=36):
    fig, axes = plt.subplots(1, len(interp_layers), figsize=(11, 3.6), sharey=True)
    for ax, L in zip(axes, interp_layers):
        rec = np.arange(L + 1, n_layer)
        for p in PAIRS:
            w = [summ[f"{p}|L{L}|L{r}.resid_post"]["width"] for r in rec]
            ax.plot(rec, w, marker="o", ms=3.5, **STYLE[p], label=p.replace("->", " → "))
        ax.axhline(0.8, **REF_DIAG)
        ax.set_title(f"patch at block {L}", fontsize=10)
        ax.set_xlabel("recording block (resid_post)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("transition width $w$")
    axes[0].legend(fontsize=9)
    fig.suptitle("Sharpening with depth: transition width at each downstream residual-stream site",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "exp1_depth_emergence.png"), dpi=150)
    plt.close(fig)


def fig_site_types(summ, L=0, n_layer=36):
    """Does one component type create the sharp transition, or does it build up everywhere?"""
    sites = ["attn_out", "resid_mid", "mlp_post", "mlp_out", "resid_post"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    rec = np.arange(L + 1, n_layer)
    for ax, p in zip(axes, PAIRS):
        for i, site in enumerate(sites):
            w = [summ[f"{p}|L{L}|L{r}.{site}"]["width"] for r in rec]
            ax.plot(rec, w, color=CVD[i], ls=LINESTYLES[i % 4], lw=1.8,
                    marker=["o", "s", "^", "D", "v"][i], ms=3.5, markevery=4, label=site)
        ax.axhline(0.8, **REF_DIAG)
        ax.set_title(p.replace("->", " → "), fontsize=11)
        ax.set_xlabel("recording block")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("transition width $w$")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Transition width at all five recording sites (patch at block {L}, "
                 "context 'The house was')", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "exp1_site_types.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    use_cvd()
    curves, meta = load("exp1")
    ts = np.array(meta["ts"])
    fig_raw_curves(curves, ts)
    fig_width_by_layer(meta["summaries"], meta["n_layer"])
    fig_depth_emergence(meta["summaries"], n_layer=meta["n_layer"])
    fig_site_types(meta["summaries"], n_layer=meta["n_layer"])
    print("wrote 4 figures to", PLOTS)


if __name__ == "__main__":
    main()
