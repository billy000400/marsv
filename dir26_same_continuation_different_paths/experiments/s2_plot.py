"""S2 plots: d(t) for the pairs that passed the completion screen on Qwen2.5-3B, plus the
GPT-2 Large / Qwen2.5-3B comparison for the one pair testable on both models."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
STYLE = [("-", "o"), ("--", "s"), (":", "^")]


def load(key, mk):
    z = np.load(os.path.join(RESULTS, f"{key}_interp_{mk}.npz"))
    return z["t"], z["d"]


def panel(ax, series, title):
    for i, (lab, t, d) in enumerate(series):
        ls, mk = STYLE[i]
        ax.plot(t, d, ls, marker=mk, color=CVD[i], ms=3.2, lw=1.8, label=lab)
    ax.axhline(0.5, color="0.55", ls=(0, (1, 3)), lw=1.0)
    ax.set_xlabel("interpolation position t  (0 = prompt A endpoint, 1 = prompt B endpoint)")
    ax.set_ylabel("relative logit distance d(t)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")


def main():
    os.makedirs(PLOTS, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    panel(ax, [("pair 1: copy vs recall (solid, circles)", *load("p1_copy_vs_recall", "qwen")),
               ("pair 3: capital vs largest city (dashed, squares)",
                *load("p3_relation_repaired", "qwen")),
               ("pair 4: Hamlet vs Macbeth (dotted, triangles)", *load("p4_entity", "qwen"))],
          "Qwen2.5-3B, block-0 residual stream, full-sequence interpolation")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "qwen_dt_three_pairs.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    panel(ax, [("Qwen2.5-3B (solid, circles)", *load("p1_copy_vs_recall", "qwen")),
               ("GPT-2 Large (dashed, squares)", *load("p1_copy_vs_recall", "gpt2"))],
          "Pair 1 (copy-available vs factual-recall): both models")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "p1_gpt2_vs_qwen_dt.png"), dpi=160)
    plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
