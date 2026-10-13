"""How much do the per-pair measurements vary across the three fixed sentence frames? (feedback #8)

No GPU: reads the committed per-frame transition widths (`w_ctx`) and per-frame model-output JSDs
(`out_jsd`) of the trained 1.4B model for the 1,000-pair set and the controlled 60-pair set.
Range = per pair, maximum minus minimum over the three frames.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130

SETS = [("1,000-pair set", "assay_large_step143000.json", "pair_manifest_large.json"),
        ("controlled 60-pair set", "assay_step143000_t256.json", None)]


def load(assay, manifest):
    rows = json.load(open(os.path.join(RESULTS, assay)))["rows"]
    if manifest:
        pairs = json.load(open(os.path.join(RESULTS, manifest)))["pairs"]
        J = np.array([pairs[r["pair_idx"]]["jsd_B"] for r in rows])
    else:
        J = np.array([r["jsd_B"] for r in rows])
    W = np.array([r["w_ctx"] for r in rows], float)       # pairs x 3 frames
    O = np.array([r["out_jsd"] for r in rows], float)
    assert np.isfinite(W).all() and np.isfinite(O).all()
    names = [r["a_str"] + "/" + r["b_str"] for r in rows]
    return J, W, O, names


def describe(x):
    q = np.percentile(x, [25, 50, 75, 90])
    return dict(median=float(q[1]), q25=float(q[0]), q75=float(q[2]), p90=float(q[3]),
                max=float(x.max()))


def stats(J, W, O, names):
    out = {"n_pairs": len(J)}
    for key, X in (("w", W), ("out_jsd", O)):
        med, rng = np.median(X, 1), np.ptp(X, 1)
        iqr = float(np.subtract(*np.percentile(med, [75, 25])))
        full = float(np.ptp(med))
        worst = np.argsort(-rng)[:3]
        out[key] = dict(
            range=describe(rng),
            between_pair_iqr=iqr, between_pair_full_range=full,
            median_range_over_between_pair_iqr=float(np.median(rng) / iqr),
            share_pairs_range_above_between_pair_iqr=float((rng > iqr).mean()),
            frame_medians=[float(v) for v in np.median(X, 0)],
            rho_J_single_frame=[float(spearmanr(J, X[:, c]).statistic) for c in range(3)],
            rho_J_frame_median=float(spearmanr(J, med).statistic),
            rho_J_frame_min=float(spearmanr(J, X.min(1)).statistic),
            rho_J_frame_max=float(spearmanr(J, X.max(1)).statistic),
            rho_between_frames={f"{a}-{b}": float(spearmanr(X[:, a], X[:, b]).statistic)
                                for a, b in ((0, 1), (0, 2), (1, 2))},
            rho_range_vs_J=float(spearmanr(J, rng).statistic),
            widest_range_pairs=[dict(pair=names[i], range=float(rng[i]),
                                     values=[float(v) for v in X[i]]) for i in worst])
    out["rho_outjsd_w_single_frame"] = [float(spearmanr(O[:, c], W[:, c]).statistic)
                                        for c in range(3)]
    out["rho_outjsd_w_frame_median"] = float(spearmanr(np.median(O, 1), np.median(W, 1)).statistic)
    return out


def hist(data, key, xlabel, fname):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for a, (label, J, W, O, _) in zip(ax, data):
        X = W if key == "w" else O
        rng, med = np.ptp(X, 1), np.median(X, 1)
        iqr = np.subtract(*np.percentile(med, [75, 25]))
        a.hist(rng, bins=30, color=CVD[0], edgecolor="white", lw=0.4)
        a.axvline(np.median(rng), color="0.1", ls="--", lw=1.5,
                  label=f"median range = {np.median(rng):.3f}")
        a.axvline(iqr, color=CVD[1], ls=":", lw=2.2,
                  label=f"spread between pairs (IQR) = {iqr:.3f}")
        a.set_xlabel(xlabel)
        a.set_ylabel("number of token pairs")
        a.set_title(f"{label} ({len(rng)} pairs)")
        a.legend(fontsize=8, framealpha=0.9, edgecolor="none", loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname))
    plt.close(fig)


def errorbars(data):
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9))
    for row, (label, J, W, O, _) in enumerate(data):
        big = len(J) > 200
        kw = dict(fmt="o", ms=2.2 if big else 4, color=CVD[0], ecolor="0.35",
                  elinewidth=0.35 if big else 0.9, alpha=0.75 if big else 0.95, capsize=0)
        w = np.median(W, 1)
        werr = [w - W.min(1), W.max(1) - w]
        o = np.median(O, 1)
        oerr = [o - O.min(1), O.max(1) - o]
        ax[row, 0].errorbar(J, w, yerr=werr, **kw)
        ax[row, 0].set_xlabel("corpus next-token JSD $J$ [bits] (one value per pair, no frame range)")
        ax[row, 0].set_title(f"{label}: $w$ against corpus JSD\n"
                             f"Spearman $\\rho$ = {spearmanr(J, w).statistic:+.3f}")
        ax[row, 1].errorbar(o, w, yerr=werr, xerr=oerr, **kw)
        ax[row, 1].set_xlabel("model-output JSD [bits] (point = median of 3 frames)")
        ax[row, 1].set_title(f"{label}: $w$ against model-output JSD\n"
                             f"Spearman $\\rho$ = {spearmanr(o, w).statistic:+.3f}")
        for a in ax[row]:
            a.set_ylabel("transition width $w$ (point = median of 3 frames)")
            a.set_ylim(0.1, 0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "frame_jsd_w_errorbars.png"))
    plt.close(fig)


def main():
    data = [(label, *load(a, m)) for label, a, m in SETS]
    res = {"frames": json.load(open(os.path.join(RESULTS, "pair_manifest_large.json")))["contexts"],
           "range_definition": "per pair, max minus min over the three sentence frames"}
    for label, J, W, O, names in data:
        res[label] = stats(J, W, O, names)
    json.dump(res, open(os.path.join(RESULTS, "frame_variation.json"), "w"), indent=1)
    hist(data, "w", "range of transition width $w$ across the three frames (max $-$ min)",
         "frame_range_w_hist.png")
    hist(data, "out_jsd", "range of model-output JSD across the three frames (max $-$ min) [bits]",
         "frame_range_jsd_hist.png")
    errorbars(data)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
