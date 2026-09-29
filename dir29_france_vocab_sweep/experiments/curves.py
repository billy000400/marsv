"""Representative final-layer y(t) curves (PLAN's d(t)) for chosen tokens, read from the saved sweep shards.

Usage: curves.py "<panel a tokens as python list>" "<panel b tokens as python list>"
Panel (a): W <= 0.3 examples; panel (b): W > 0.3 examples. Up to 5 tokens per panel.
"""
import ast
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from analyze import PLOTS, load
from sweep import d27

STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"), ("-", "v")]


def main():
    a, b = ast.literal_eval(sys.argv[1]), ast.literal_eval(sys.argv[2])
    tok = AutoTokenizer.from_pretrained(d27.MODEL)
    ids, D, st, yf, flag = load()
    pos = {tok.decode([int(i)]): k for k, i in enumerate(ids)}
    Wf = st[:, 8]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, words, title in ((axes[0], a, "(a) W ≤ 0.3 examples"), (axes[1], b, "(b) W > 0.3 examples")):
        for j, w in enumerate(words):
            k = pos[w]
            ls, mk = STYLES[j]
            ax.plot(d27.TS, yf[k], ls=ls, marker=mk, markevery=10, ms=4, color=d27.CVD[j], lw=1.5,
                    label=f"{w!r}  W={Wf[k]:.3f}")
            print(f"{w!r}: W_final={Wf[k]:.4f} D={D[k]:.3f} flag={int(flag[k])}")
        ax.axhline(0.1, color="0.6", lw=0.8, ls=":")
        ax.axhline(0.9, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("interpolation position t (0 = ' France', 1 = token B)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("final-layer progress y(t)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "france_representative_curves.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
