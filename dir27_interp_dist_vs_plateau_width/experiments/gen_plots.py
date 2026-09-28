"""Feedback 3: per-model Figures 1-3 analogues, sweep CSV, W_final band table and summary numbers.
Usage: gen_plots.py MODEL_KEY  ->  results/gen/<key>.csv, results/gen/<key>_summary.json, plots/gen/<key>_fig{1,2,3}.png
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from common import CVD
from gen_common import GEN, GPLOTS, MODELS, NAMES

BACKSTEP = 0.05
BANDS = ((0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01))


def scatter(ax, D, W, flag, sel):
    ax.scatter(D[sel & flag], W[sel & flag], s=4, alpha=0.25, color=CVD[1], marker="x", lw=0.4,
               label="flagged (non-monotonic)")
    ax.scatter(D[sel & ~flag], W[sel & ~flag], s=2, alpha=0.08, color=CVD[0], marker="o", lw=0,
               label="clean curve")
    ax.set_xlabel("layer-0 L2 distance D from ' big'")


def legend(ax):
    leg = ax.legend(loc="upper right", markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1)


def main(key):
    tok = AutoTokenizer.from_pretrained(MODELS[key][0])
    meta = json.load(open(os.path.join(GEN, key, "meta.json")))
    sh = [np.load(p) for p in sorted(glob.glob(os.path.join(GEN, key, "shard_*.npz")))]
    ids = np.concatenate([s["ids"] for s in sh])
    D = np.concatenate([s["D"] for s in sh])
    st = np.concatenate([s["stats"] for s in sh])
    flag = ((st[:, [3, 4, 9, 10]] > 1).any(1)) | (st[:, [5, 11]] > BACKSTEP).any(1)
    Wm, Wf = st[:, 2], st[:, 8]
    name, bm, bf = NAMES[key], meta["mid_block"], meta["final_block"]
    with open(os.path.join(GEN, f"{key}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token_id", "token", "D", "W_mid", "W_final", "flag_nonmonotonic"])
        for i in range(len(ids)):
            w.writerow([ids[i], repr(tok.decode([int(ids[i])])), f"{D[i]:.4f}", f"{Wm[i]:.4f}",
                        f"{Wf[i]:.4f}", int(flag[i])])

    titles = (f"(a) block {bm} (middle)", f"(b) block {bf} (final)")
    # Figure 1 analogue: D vs W, both readout blocks
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, W, t in zip(axes, (Wm, Wf), titles):
        scatter(ax, D, W, flag, np.ones_like(flag))
        ax.set_title(f"{name} {t}")
    axes[0].set_ylabel("transition width W = t$_{0.9}$ − t$_{0.1}$")
    legend(axes[0])
    fig.tight_layout()
    fig.savefig(os.path.join(GPLOTS, f"{key}_fig1_distance_vs_width.png"), dpi=110)
    plt.close(fig)

    # Figure 2 analogue: zoom on the 5th-95th percentile of D, y range = min-max W in that range
    lo, hi = np.percentile(D, [5, 95])
    sel = (D >= lo) & (D <= hi)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, W, t in zip(axes, (Wm, Wf), titles):
        scatter(ax, D, W, flag, sel)
        pad = 0.02 * (W[sel].max() - W[sel].min())
        ax.set_xlim(lo, hi)
        ax.set_ylim(W[sel].min() - pad, W[sel].max() + pad)
        ax.set_title(f"{name} {t}, zoom")
    axes[0].set_ylabel("transition width W = t$_{0.9}$ − t$_{0.1}$")
    legend(axes[0])
    fig.tight_layout()
    fig.savefig(os.path.join(GPLOTS, f"{key}_fig2_zoom_distance_vs_width.png"), dpi=110)
    plt.close(fig)

    # Figure 3 analogue: W_final histogram and sorted view
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(Wf, bins=200, color=CVD[0])
    axes[0].set_xlabel("final-block width W_final")
    axes[0].set_ylabel("number of tokens")
    axes[0].set_title(f"{name} (a) histogram, 200 bins")
    axes[1].plot(np.arange(len(Wf)), np.sort(Wf), color=CVD[0], lw=1.5)
    axes[1].set_xlabel("token rank (sorted by W_final)")
    axes[1].set_ylabel("final-block width W_final")
    axes[1].set_title(f"{name} (b) sorted widths")
    fig.tight_layout()
    fig.savefig(os.path.join(GPLOTS, f"{key}_fig3_final_width_distribution.png"), dpi=110)
    plt.close(fig)

    # S4-style band table (lowest-D examples) and the numbers used in the text
    bands = []
    for a, b in BANDS:
        s = (Wf >= a) & (Wf < b)
        ex = [tok.decode([int(i)]) for i in ids[s][np.argsort(D[s])][:8]]
        bands.append(dict(band=[a, b], n=int(s.sum()), median_D=float(np.median(D[s])) if s.any() else None,
                          lowest_D_examples=ex))
    q = np.percentile(D, [0, 20, 40, 60, 80, 100])
    dbins = []
    for i in range(5):
        s = (D >= q[i]) & (D <= q[i + 1])
        dbins.append(dict(D_range=[float(q[i]), float(q[i + 1])], n=int(s.sum()),
                          median_W_mid=float(np.median(Wm[s])), median_W_final=float(np.median(Wf[s]))))
    hist, edges = np.histogram(Wf, bins=200)
    k = int(hist.argmax())
    large = [i for i in ids if tok.decode([int(i)]) == " large"]
    summ = dict(model=key, meta=meta, n_tokens=int(len(ids)), D_median=float(np.median(D)),
                D_p5_p95=[float(lo), float(hi)], zoom_W_range_mid=[float(Wm[sel].min()), float(Wm[sel].max())],
                zoom_W_range_final=[float(Wf[sel].min()), float(Wf[sel].max())],
                median_W_mid=float(np.median(Wm)), median_W_final=float(np.median(Wf)),
                frac_final_narrower=float((Wf < Wm).mean()), frac_flagged=float(flag.mean()),
                frac_flagged_final=float(((st[:, [9, 10]] > 1).any(1) | (st[:, 11] > BACKSTEP)).mean()),
                hist_peak_W=[float(edges[k]), float(edges[k + 1])], hist_peak_n=int(hist[k]),
                W_final_p90=float(np.percentile(Wf, 90)), W_final_p99=float(np.percentile(Wf, 99)),
                W_final_large=float(Wf[ids == large[0]][0]) if large else None,
                D_bins=dbins, bands=bands,
                widest=[(tok.decode([int(i)]), float(w)) for i, w in
                        zip(ids[np.argsort(-Wf)][:20], np.sort(Wf)[::-1][:20])])
    json.dump(summ, open(os.path.join(GEN, f"{key}_summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("meta",)}, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main(sys.argv[1])
