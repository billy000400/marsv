"""S16: is gpt2-medium's shallow top-of-stack rate a property of the model or of its bank?

S15 measured C(b), the share of each model's block-0 headroom to the linear response that is closed
after removing b blocks from below the patch, on the low-JSD banks (JSD < 0.1). gpt2-large closed
60.7% after four blocks and gpt2-small 51.1%, but gpt2-medium only 18.6%. Those banks are mined per
model, so "gpt2-medium is the outlier" is confounded with "gpt2-medium's low-JSD bank is different".

This re-runs the identical blocks 0-4 sweep, unablated (the S15 primary readout), on the S4
corpus-mined bank instead, which spans the full JSD range rather than JSD < 0.1. If C(4) keeps its
ordering across the two banks, the rate is a model property.

Writes results/bank_depth.json and plots/bank_depth.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze import tv_width
from common import PLOTS, RESULTS, load
from depth_gap import run_at
from mine_pairs import SEED, get_prefixes

MODELS = ["gpt2-medium", "gpt2-large", "gpt2-small"]
SITES = [0, 1, 2, 3, 4]
N_PAIRS = int(os.environ.get("N_PAIRS", 60))
N_ALPHA = 101
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
MSTYLE = {"gpt2-medium": dict(color=CVD[1], marker="s", ls="--"),
          "gpt2-large": dict(color=CVD[2], marker="^", ls=":"),
          "gpt2-small": dict(color=CVD[0], marker="o", ls="-.")}


def main():
    alphas = np.linspace(0, 1, N_ALPHA)
    path = os.path.join(RESULTS, "bank_depth.json")
    out = json.load(open(path)) if os.path.exists(path) else {}

    for mkey in MODELS:
        if mkey in out:
            continue
        # gpt2-large has no S4 bank of its own; the mined bank is per model, so use the one that
        # exists for it (S7 ran the same miner for the depth-mismatched GPT-2 models).
        bank = json.load(open(os.path.join(RESULTS, f"bank_{mkey}.json")))
        idx = np.linspace(0, len(bank) - 1, min(N_PAIRS, len(bank))).round().astype(int)
        rows = [bank[i] for i in sorted(set(idx.tolist()))]

        tok, m = load(mkey)
        prefixes = get_prefixes(tok, np.random.default_rng(SEED))
        n_block = len(m.transformer.h)
        print(f"{mkey}: {len(rows)} pairs, {n_block} blocks, sites {SITES}", flush=True)

        rec = {}
        for L in SITES:
            w, err = [], 0.0
            for i, r in enumerate(rows):
                pre = prefixes[r["prefix_idx"]]
                ia = tok.convert_tokens_to_ids(r["tok_a"])
                ib = tok.convert_tokens_to_ids(r["tok_b"])
                d = run_at(m, pre + [ia], pre + [ib], alphas, L)
                w.append(tv_width(alphas, d))
                err = max(err, float(max(abs(d[0]), abs(d[-1] - 1))))
                if i % 30 == 0:
                    print(f"    L{L} pair {i}/{len(rows)}", flush=True)
            rec[str(L)] = dict(median_wtv=float(np.median(w)), n=len(w), max_endpoint_err=err,
                               f=(n_block - 1 - L) / (n_block - 1), wtv=[float(x) for x in w])
            print(mkey, L, rec[str(L)]["median_wtv"], flush=True)

        base = rec["0"]["median_wtv"]
        head = 0.5 - base
        rec["C"] = {str(L): (rec[str(L)]["median_wtv"] - base) / head for L in SITES}
        rec["meta"] = dict(n_block=n_block, n_pairs=len(rows), headroom=head,
                           median_jsd=float(np.median([r["jsd"] for r in rows])))
        out[mkey] = rec
        json.dump(out, open(path, "w"), indent=1)
        del m
        torch.cuda.empty_cache()

    figure(out)
    return out


def figure(out):
    low = json.load(open(os.path.join(RESULTS, "depth_models.json")))
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.6))
    b = np.array(SITES)

    for mkey in MODELS:
        st = MSTYLE[mkey]
        y = [out[mkey][str(L)]["median_wtv"] for L in SITES]
        ax[0].plot(b, y, lw=2.2, ms=8, mec="k", mew=0.6, label=f"{mkey} (wide-JSD bank)", **st)
        ax[1].plot(b, [100 * out[mkey]["C"][str(L)] for L in SITES], lw=2.2, ms=8, mec="k", mew=0.6,
                   label=f"{mkey} (wide-JSD bank)", **st)
        if mkey in low:
            c = [100 * v for v in low[mkey]["closed"]]
            ax[1].plot(b, c, color=st["color"], marker=st["marker"], ls="-", lw=1.3, ms=5,
                       alpha=0.55, label=f"{mkey} (low-JSD bank, S15)")

    ax[0].axhline(0.5, color="0.35", ls=(0, (4, 3)), lw=1.4)
    ax[0].set_ylim(0, 0.56)
    ax[0].set_ylabel("median transition width $w_{TV}$")
    ax[0].set_title("A. The unablated switch, blocks removed from below the patch\n"
                    "(dashed = linear response 0.5)", fontsize=10)
    ax[1].axhline(0.0, color="0.35", lw=1.0)
    ax[1].set_ylabel("headroom closed $C(b)$ (%)")
    ax[1].set_title("B. Same curve normalised by each model's own block-0 headroom,\n"
                    "against the low-JSD banks of S15", fontsize=10)
    for a in ax:
        a.set_xlabel("blocks removed from below the patch, $b$ (patch at block $b$)")
        a.set_xticks(b)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="lower right" if a is ax[0] else "upper left")
    fig.suptitle("Blocks 0-4 on the corpus-mined wide-JSD bank (S4), 60 pairs per model", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(PLOTS, "bank_depth.png"), dpi=140)
    plt.close(fig)
    print("wrote plots/bank_depth.png")


if __name__ == "__main__":
    main()
