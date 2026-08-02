"""S5: primary correlation, output-JSD validation, sensitivity model, and every figure."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from common import DATA, PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
MARK = ["o", "s", "^", "D", "v"]
LS = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
BOOT = 10_000
RNG = np.random.default_rng(0)


def boot_spearman(x, y, n=BOOT):
    """Endpoint-aware bootstrap: the bank is endpoint-disjoint, so resampling PAIRS resamples
    endpoints as intact clusters -- no endpoint appears in two pairs."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    r = spearmanr(x, y).statistic
    bs = np.empty(n)
    for i in range(n):
        k = RNG.integers(0, len(x), len(x))
        bs[i] = spearmanr(x[k], y[k]).statistic
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(r), float(lo), float(hi), int(m.sum()), float(spearmanr(x, y).pvalue)


def partial_spearman(x, y, covs):
    """Spearman of x and y after linear adjustment for `covs`, all on ranks."""
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(covs), axis=0)
    rx, ry = rankdata(x[m]), rankdata(y[m])
    C = np.column_stack([rankdata(c[m]) for c in covs] + [np.ones(m.sum())])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    return float(spearmanr(ex, ey).statistic), int(m.sum())


def load_assay(tag):
    return json.load(open(os.path.join(RESULTS, f"assay_{tag}.json")))


def arr(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


# ---------------------------------------------------------------- figures
def fig_reliability(rel, npz):
    jA, jB, sh = npz["jsd_A"], npz["jsd_B"], npz["splithalf"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    ax[0].scatter(jA, jB, s=3, alpha=0.15, color=CVD[0], marker="o", edgecolors="none")
    lim = [min(jA.min(), jB.min()), max(jA.max(), jB.max())]
    ax[0].plot(lim, lim, ls="--", color="0.35", lw=1, label="y = x")
    ax[0].set_xlabel("$JSD_A$ (split A, bits)")
    ax[0].set_ylabel("$JSD_B$ (split B, bits)")
    ax[0].set_title(f"Between-token JSD, two disjoint corpus splits\nSpearman = {rel['spearman_A_B']:.3f}"
                    f"  (n = {len(jA):,} pairs)")
    ax[0].legend(frameon=False, fontsize=8)
    bins = np.linspace(0, max(jB.max(), sh.max()), 60)
    ax[1].hist(jB, bins=bins, color=CVD[0], histtype="stepfilled", alpha=0.55, hatch="//",
               label=f"between-token $JSD_B$ (median {np.median(jB):.3f})")
    ax[1].hist(sh, bins=bins, color=CVD[1], histtype="stepfilled", alpha=0.55, hatch="\\\\",
               label=f"same-token split-half (median {np.median(sh):.3f})")
    ax[1].set_xlabel("JSD (bits)")
    ax[1].set_ylabel("count")
    ax[1].set_title(f"Signal vs sampling-noise floor\nratio = {rel['splithalf_ratio']:.3f} (need < 0.25)")
    ax[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "jsd_reliability.png"))
    plt.close(fig)


def fig_jsd_vs_width(assays, labels):
    fig, ax = plt.subplots(1, len(assays), figsize=(4.6 * len(assays), 4.1), squeeze=False)
    for k, (A, lab) in enumerate(zip(assays, labels)):
        a = ax[0, k]
        x, y = arr(A["rows"], "jsd_B"), arr(A["rows"], "w")
        b = arr(A["rows"], "bin").astype(int)
        for q in range(5):
            m = b == q
            a.scatter(x[m], y[m], s=26, color=CVD[q], marker=MARK[q], alpha=0.85,
                      edgecolors="none", label=f"$JSD_A$ quintile {q+1}")
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() > 10:  # running median of w in 5 equal-count JSD_B bins
            xs, ys = x[good], y[good]
            e = np.quantile(xs, np.linspace(0, 1, 6))
            bi = np.clip(np.digitize(xs, e[1:-1]), 0, 4)
            a.plot([np.median(xs[bi == q]) for q in range(5)],
                   [np.median(ys[bi == q]) for q in range(5)],
                   color="0.25", ls="--", lw=1.4, marker="x", ms=6, label="running median")
        r, lo, hi, n, p = boot_spearman(x, y)
        a.set_xlabel("$JSD_B$: corpus continuation divergence (bits)")
        a.set_ylabel("transition width $w$  (smaller = sharper plateau)")
        a.set_title(f"{lab}\nSpearman $\\rho$ = {r:+.3f}  [{lo:+.2f}, {hi:+.2f}]  n = {n}")
        if k == 0:
            a.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "jsd_vs_width.png"))
    plt.close(fig)


def fig_width_by_bin(assays, labels):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    off = np.linspace(-0.18, 0.18, len(assays))
    for k, (A, lab) in enumerate(zip(assays, labels)):
        w = arr(A["rows"], "w")
        b = arr(A["rows"], "bin").astype(int)
        data = [w[(b == q) & np.isfinite(w)] for q in range(5)]
        pos = np.arange(5) + off[k]
        bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.4))
        for patch in bp["boxes"]:
            patch.set_facecolor(CVD[k])
            patch.set_alpha(0.55)
            patch.set_hatch(["//", "\\\\", "..", "xx"][k])
        for q in range(5):
            ax.scatter(np.full(len(data[q]), pos[q]), data[q], s=9, color=CVD[k],
                       marker=MARK[k], alpha=0.7, edgecolors="none",
                       label=lab if q == 0 else None)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Q{q+1}" for q in range(5)])
    ax.set_xlabel("$JSD_A$ quintile of the frozen bank (Q1 = most similar continuations)")
    ax.set_ylabel("transition width $w$")
    ax.set_title("Plateau sharpness by corpus-divergence bin")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "width_by_jsd_bin.png"))
    plt.close(fig)


def fig_reference_curves(A, tag, man):
    grid = np.array(A["grid"])
    curves = np.load(os.path.join(RESULTS, f"curves_{tag}.npy"))
    x = arr(A["rows"], "jsd_B")
    order = np.argsort(x)
    pick = list(order[:3]) + list(order[-3:])
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for n, i in enumerate(pick):
        r = A["rows"][i]
        d = np.nanmedian(curves[i], axis=0)
        grp = "low" if n < 3 else "high"
        ax.plot(grid, d, ls=LS[0] if grp == "low" else LS[1], marker=MARK[n % 5], ms=3,
                color=CVD[0] if grp == "low" else CVD[1], lw=1.4, alpha=0.9,
                label=f"{r['a_str'].strip()}/{r['b_str'].strip()}  $JSD_B$={r['jsd_B']:.2f} ({grp})")
    ax.axhline(0.1, color="0.6", lw=0.8, ls=":")
    ax.axhline(0.9, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("interpolation position $t$ (block-0 residual, SLERP)")
    ax.set_ylabel("relative logit distance $d(t)$")
    ax.set_title("Raw plateau curves: 3 lowest vs 3 highest $JSD_B$ pairs")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "reference_curves.png"))
    plt.close(fig)


def fig_output_jsd(A):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    x, y = arr(A["rows"], "jsd_B"), arr(A["rows"], "out_jsd_med")
    b = arr(A["rows"], "bin").astype(int)
    for q in range(5):
        m = b == q
        ax.scatter(x[m], y[m], s=26, color=CVD[q], marker=MARK[q], alpha=0.85, edgecolors="none",
                   label=f"quintile {q+1}")
    r, lo, hi, n, p = boot_spearman(x, y)
    ax.set_xlabel("$JSD_B$: corpus continuation divergence (bits)")
    ax.set_ylabel("model output JSD in carrier context (bits)")
    ax.set_title(f"Does corpus JSD predict a distinction the model learned?\n"
                 f"Spearman $\\rho$ = {r:+.3f}  [{lo:+.2f}, {hi:+.2f}]  n = {n}")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "output_jsd_validation.png"))
    plt.close(fig)


def fig_block_scan():
    p = os.path.join(RESULTS, "block_scan.json")
    if not os.path.exists(p):
        return None
    bs = json.load(open(p))
    blocks = bs["blocks"]
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for gi, grp in enumerate(["low", "high"]):
        W = np.array([[r["w_by_block"][str(L)] for L in blocks]
                      for r in bs["rows"] if r["group"] == grp], dtype=float)
        med = np.nanmedian(W, axis=0)
        ax.plot(blocks, med, ls=LS[gi], marker=MARK[gi], color=CVD[gi], lw=1.6,
                label=f"{grp} $JSD_B$ pairs (n={W.shape[0]})")
        for row in W:
            ax.plot(blocks, row, ls=LS[gi], color=CVD[gi], lw=0.6, alpha=0.3)
    ax.set_xticks(blocks)
    ax.set_xlabel("patched block $L$ (residual stream after this block is interpolated)")
    ax.set_ylabel("transition width $w$")
    ax.set_title("Sharpness needs downstream blocks")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "block_scan.png"))
    plt.close(fig)
    return bs


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    rel = json.load(open(os.path.join(RESULTS, "jsd_reliability.json")))
    npz = np.load(os.path.join(DATA, "reliability_bank.npz"))
    man = json.load(open(os.path.join(RESULTS, "pair_manifest.json")))
    fig_reliability(rel, npz)

    tags, labels = [], []
    for t, lab in [("step143000", "pythia-1.4b-deduped, step 143000 (trained)"),
                   ("step0", "pythia-1.4b-deduped, step 0 (untrained)"),
                   ("step143000_410m", "pythia-410m-deduped, step 143000")]:
        if os.path.exists(os.path.join(RESULTS, f"assay_{t}.json")):
            tags.append(t)
            labels.append(lab)
    assays = [load_assay(t) for t in tags]
    fig_jsd_vs_width(assays, labels)
    fig_width_by_bin(assays, labels)
    fig_reference_curves(assays[0], tags[0], man)
    fig_output_jsd(assays[0])
    bs = fig_block_scan()

    summary = dict(reliability=rel, bank=dict(
        n_pairs=len(man["pairs"]), topk_used=man["topk_used"],
        balance_p_logfreq=man["balance_kruskal_p_logfreq"],
        balance_p_surprisal=man["balance_kruskal_p_surprisal"]), checkpoints={})
    for t, lab, A in zip(tags, labels, assays):
        x, y = arr(A["rows"], "jsd_B"), arr(A["rows"], "w")
        r, lo, hi, n, p = boot_spearman(x, y)
        oj = arr(A["rows"], "out_jsd_med")
        ro, lo_o, hi_o, no, po = boot_spearman(x, oj)
        cov = [np.log10(np.array([man["pairs"][rr["pair_idx"]]["count_a"] *
                                  man["pairs"][rr["pair_idx"]]["count_b"] for rr in A["rows"]],
                                 dtype=float)) / 2,
               np.array([(man["pairs"][rr["pair_idx"]]["ent_a"] +
                          man["pairs"][rr["pair_idx"]]["ent_b"]) / 2 for rr in A["rows"]]),
               np.array([(man["pairs"][rr["pair_idx"]]["surp_a"] +
                          man["pairs"][rr["pair_idx"]]["surp_b"]) / 2 for rr in A["rows"]]),
               np.array([np.median(rr["cos0"]) for rr in A["rows"]]),
               np.array([np.median(rr["dist0"]) for rr in A["rows"]])]
        pr, npart = partial_spearman(x, y, cov)
        bw = {q: float(np.nanmedian(y[arr(A["rows"], "bin").astype(int) == q])) for q in range(5)}
        inval = {q: float(1 - np.isfinite(y[arr(A["rows"], "bin").astype(int) == q]).mean())
                 for q in range(5)}
        summary["checkpoints"][t] = dict(
            label=lab, n_pairs=A["n_pairs"], valid_curve_rate=A["valid_curve_rate"],
            median_w=A["median_w"], iqr_w=A["iqr_w"],
            max_endpoint_relerr=A["max_endpoint_relerr"],
            spearman_jsdB_w=r, ci=[lo, hi], p=p, n=n,
            spearman_jsdB_outjsd=ro, ci_outjsd=[lo_o, hi_o], p_outjsd=po,
            partial_spearman_jsdB_w=pr, n_partial=npart,
            median_w_by_bin=bw, invalid_rate_by_bin=inval)
    if bs:
        summary["block_scan"] = {str(L): float(np.nanmedian(
            [r["w_by_block"][str(L)] for r in bs["rows"]])) for L in bs["blocks"]}
    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary["checkpoints"].items()}, indent=2)[:3000])
