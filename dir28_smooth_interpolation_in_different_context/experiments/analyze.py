"""Aggregate shards per context -> per-token CSV, sorted W_final > 0.3 lists, histograms, cross-context table.

Flag rule and column layout are Direction 27's (s3_plots.py): a curve is flagged when it crosses 0.1 or 0.9
more than once at either block, or y(t) drops by more than 0.05 between neighbouring steps.
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

from sweep import PROMPTS, ROOT, d27

CTX = ["s1", "s2", "s3", "s4"]
THR = 0.3
BACKSTEP = 0.05
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=d27.CVD)


def load(ctx):
    sh = [np.load(p) for p in sorted(glob.glob(os.path.join(RES, ctx, "shard_*.npz")))]
    ids = np.concatenate([s["ids"] for s in sh])
    D = np.concatenate([s["D"] for s in sh])
    st = np.concatenate([s["stats"] for s in sh])
    flag = ((st[:, [3, 4, 9, 10]] > 1).any(1)) | (st[:, [5, 11]] > BACKSTEP).any(1)
    return ids, D, st, flag


def main():
    tok = AutoTokenizer.from_pretrained(d27.MODEL)
    data, summary = {}, {}
    for c in CTX:
        ids, D, st, flag = load(c)
        assert len(ids) == 50256, (c, len(ids))
        Wf = st[:, 8]
        data[c] = dict(zip(ids.tolist(), Wf.tolist()))
        dec = [tok.decode([int(i)]) for i in ids]
        with open(os.path.join(RES, f"{c}_tokens.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["context_id", "prompt", "token_id", "token", "D", "W_mid", "W_final",
                        "t10_final", "t90_final", "flag_nonmonotonic"])
            for i in range(len(ids)):
                w.writerow([c, PROMPTS[c], ids[i], repr(dec[i]), f"{D[i]:.4f}", f"{st[i, 2]:.4f}",
                            f"{Wf[i]:.4f}", f"{st[i, 6]:.4f}", f"{st[i, 7]:.4f}", int(flag[i])])
        sel = np.where(Wf > THR)[0]
        sel = sel[np.argsort(-Wf[sel])]
        with open(os.path.join(RES, f"{c}_W_gt_0.3.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "token_id", "token", "W_final", "D", "flag_nonmonotonic"])
            for r, i in enumerate(sel, 1):
                w.writerow([r, ids[i], repr(dec[i]), f"{Wf[i]:.4f}", f"{D[i]:.4f}", int(flag[i])])
        summary[c] = dict(prompt=PROMPTS[c], n=int(len(ids)), n_flagged=int(flag.sum()),
                          n_gt=int(len(sel)), n_gt_flagged=int(flag[sel].sum()),
                          n_gt_rare_D_gt4=int((D[sel] > 4).sum()),
                          W_pct={p: float(np.percentile(Wf, p)) for p in (5, 25, 50, 75, 95)},
                          W_max=float(Wf.max()))
        print(c, summary[c])

    # identical bins and axes for all four histograms
    allW = np.concatenate([np.array(list(data[c].values())) for c in CTX])
    bins = np.linspace(0, np.ceil(allW.max() * 10) / 10, 101)
    ymax = max(np.histogram(list(data[c].values()), bins)[0].max() for c in CTX) * 1.5
    for k, c in enumerate(CTX, 1):
        W = np.array(list(data[c].values()))
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.hist(W, bins=bins, color=d27.CVD[0], edgecolor="none")
        ax.axvline(THR, color="0.2", ls="--", lw=1)
        ax.text(THR + 0.01, ymax / 3, f"W = 0.3\n{summary[c]['n_gt']} tokens above", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(0.8, ymax)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("final transition width W (block 35)")
        ax.set_ylabel("number of tokens (log scale)")
        ax.set_title(f"Section {k}: \"{PROMPTS[c]}\" ({len(W):,} tokens)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, f"section{k}_width_histogram.png"), dpi=130)
        plt.close(fig)

    # cross-context comparison of the W > 0.3 sets
    sets = {c: {i for i, w in data[c].items() if w > THR} for c in CTX}
    union = set().union(*sets.values())
    rows = []
    for i in union:
        ws = [data[c][i] for c in CTX]
        rows.append(dict(token_id=i, token=repr(tok.decode([i])), n_ctx=sum(w > THR for w in ws),
                         **{f"W_{c}": round(w, 4) for c, w in zip(CTX, ws)},
                         min_W=round(min(ws), 4), max_W=round(max(ws), 4)))
    rows.sort(key=lambda r: (-r["n_ctx"], -r["max_W"]))
    with open(os.path.join(RES, "cross_context.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary["cross"] = {f"in_{n}_contexts": sum(r["n_ctx"] == n for r in rows) for n in (1, 2, 3, 4)}
    summary["cross"]["unique_per_context"] = {
        c: sum(1 for r in rows if r["n_ctx"] == 1 and r[f"W_{c}"] == r["max_W"]) for c in CTX}
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    print(summary["cross"])


if __name__ == "__main__":
    main()
