"""S5: representative final-layer y(t) curves from different W_final regimes and from the modal group."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from common import CVD, MODEL, PLOTS, RESULTS, TS

STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"), ("-", "v")]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    sh = [np.load(p) for p in sorted(glob.glob(os.path.join(RESULTS, "sweep", "shard_*.npz")))]
    ids = np.concatenate([s["ids"] for s in sh])
    yf = np.concatenate([s["y_final"] for s in sh]).astype(np.float32)
    Wf = np.concatenate([s["stats"] for s in sh])[:, 8]
    D = np.concatenate([s["D"] for s in sh])
    pos = {tok.decode([int(i)]): k for k, i in enumerate(ids)}

    # (a) one common word per width regime; (b) five tokens from the modal histogram bin
    regime = [" of", " the", " good", " great", " large"]
    hist, edges = np.histogram(Wf, bins=200)
    b = hist.argmax()
    lo, hi = edges[b], edges[b + 1]
    modal = [w for w in (" house", " car", " red", " dog", " money", " table", " music", " water",
                         " city", " blue") if lo <= Wf[pos[w]] < hi][:5]
    if len(modal) < 5:
        cand = np.where((Wf >= lo) & (Wf < hi))[0]
        cand = cand[np.argsort(-D[cand])]  # deterministic fill
        modal += [tok.decode([int(ids[k])]) for k in cand if tok.decode([int(ids[k])]) not in modal][:5 - len(modal)]
    print(f"modal bin [{lo:.4f},{hi:.4f}) n={hist[b]}; modal examples {modal}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, words, title in ((axes[0], regime, "(a) tokens from different width regimes"),
                             (axes[1], modal, "(b) five tokens from the most common width bin")):
        for j, w in enumerate(words):
            k = pos[w]
            ls, mk = STYLES[j]
            ax.plot(TS, yf[k], ls=ls, marker=mk, markevery=10, ms=4, color=CVD[j], lw=1.5,
                    label=f"{w!r}  W={Wf[k]:.3f}, D={D[k]:.2f}")
            print(f"{title[:3]} {w!r}: W_final={Wf[k]:.4f} D={D[k]:.3f}")
        ax.axhline(0.1, color="0.6", lw=0.8, ls=":")
        ax.axhline(0.9, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("interpolation position t (0 = ' big', 1 = token B)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("final-layer progress y(t)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig3_representative_curves.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
