"""Aggregate the ' France' sweep -> per-token CSV, both sides of W_final = 0.3 as sorted lists, histogram.

Flag rule and columns are Direction 27's (s3_plots.py); histogram conventions are Direction 28's.
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from sweep import PROMPT, ROOT, d27

THR = 0.3
BACKSTEP = 0.05
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")


def load():
    sh = [np.load(p) for p in sorted(glob.glob(os.path.join(RES, "sweep", "shard_*.npz")))]
    ids = np.concatenate([s["ids"] for s in sh])
    D = np.concatenate([s["D"] for s in sh])
    st = np.concatenate([s["stats"] for s in sh])
    yf = np.concatenate([s["y_final"] for s in sh]).astype(np.float32)
    flag = ((st[:, [3, 4, 9, 10]] > 1).any(1)) | (st[:, [5, 11]] > BACKSTEP).any(1)
    return ids, D, st, yf, flag


def write_list(path, order, ids, dec, Wf, D, flag):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token_id", "token", "W_final", "D", "flag_nonmonotonic"])
        for r, i in enumerate(order, 1):
            w.writerow([r, ids[i], repr(dec[i]), f"{Wf[i]:.4f}", f"{D[i]:.4f}", int(flag[i])])


def main():
    tok = AutoTokenizer.from_pretrained(d27.MODEL)
    ids, D, st, yf, flag = load()
    assert len(ids) == 50256, len(ids)
    Wf = st[:, 8]
    dec = [tok.decode([int(i)]) for i in ids]
    with open(os.path.join(RES, "france_tokens.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "token_id", "token", "D", "W_mid", "W_final", "t10_final", "t90_final",
                    "flag_nonmonotonic"])
        for i in range(len(ids)):
            w.writerow([PROMPT, ids[i], repr(dec[i]), f"{D[i]:.4f}", f"{st[i, 2]:.4f}", f"{Wf[i]:.4f}",
                        f"{st[i, 6]:.4f}", f"{st[i, 7]:.4f}", int(flag[i])])
    le = np.where(Wf <= THR)[0]
    le = le[np.argsort(Wf[le], kind="stable")]
    gt = np.where(Wf > THR)[0]
    gt = gt[np.argsort(-Wf[gt], kind="stable")]
    write_list(os.path.join(RES, "france_W_le_0.3.csv"), le, ids, dec, Wf, D, flag)
    write_list(os.path.join(RES, "france_W_gt_0.3.csv"), gt, ids, dec, Wf, D, flag)
    summary = dict(prompt=PROMPT, n=int(len(ids)), n_flagged=int(flag.sum()),
                   n_le=int(len(le)), n_gt=int(len(gt)), n_le_flagged=int(flag[le].sum()),
                   n_gt_flagged=int(flag[gt].sum()), n_gt_D_gt4=int((D[gt] > 4).sum()),
                   W_pct={p: float(np.percentile(Wf, p)) for p in (1, 5, 25, 50, 75, 95, 99)},
                   W_min=float(Wf.min()), W_max=float(Wf.max()))
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    print(summary)

    bins = np.linspace(0, np.ceil(Wf.max() * 10) / 10, 101)
    ymax = np.histogram(Wf, bins)[0].max() * 12
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(Wf, bins=bins, color=d27.CVD[0], edgecolor="none")
    ax.axvline(THR, color="0.2", ls="--", lw=1)
    ax.text(THR - 0.01, ymax / 2.2, f"{len(le):,} tokens\nW ≤ 0.3", fontsize=8, ha="right")
    ax.text(THR + 0.01, ymax / 2.2, f"{len(gt):,} tokens\nW > 0.3", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.8, ymax)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("final transition width W (block 35)")
    ax.set_ylabel("number of tokens (log scale)")
    ax.set_title(f"\"{PROMPT}\": ' France' → B ({len(Wf):,} tokens)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "france_vocab_width_histogram.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
