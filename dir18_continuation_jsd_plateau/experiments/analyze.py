"""S5: primary correlation, output-JSD validation, sensitivity model, and every figure.

PRIMARY bank = the prespecified top-256 endpoint filter (`pair_manifest_top256.json`, tags `*_t256`).
SECONDARY, post-hoc bank = the relaxed top-512 filter (`pair_manifest.json`, tags without a suffix).
All widths come from the strict validity re-scoring in `qc_<tag>.json` (see curve_metrics.py).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from common import DATA, PLOTS, RESULTS
from curve_metrics import E_LINEAR

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
MARK = ["o", "s", "^", "D", "v"]
LS = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
BOOT = 10_000
RNG = np.random.default_rng(0)

PRIMARY = [("step143000_t256", "pythia-1.4b-deduped, step 143000 (trained)"),
           ("step0_t256", "pythia-1.4b-deduped, step 0 (untrained)"),
           ("step143000_410m_t256", "pythia-410m-deduped, step 143000")]
SECONDARY = [("step143000", "1.4b step 143000"), ("step0", "1.4b step 0"),
             ("step143000_410m", "410m step 143000")]
FORM_STEPS = [0, 1000, 8000, 32000, 64000, 143000]


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


def qc(tag):
    return json.load(open(os.path.join(RESULTS, f"qc_{tag}.json")))


def arr(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def covariates(rows, man):
    p = [man["pairs"][r["pair_idx"]] for r in rows]
    return [np.log10(np.array([q["count_a"] * q["count_b"] for q in p], dtype=float)) / 2,
            np.array([(q["ent_a"] + q["ent_b"]) / 2 for q in p]),
            np.array([(q["surp_a"] + q["surp_b"]) / 2 for q in p]),
            np.array([np.median(r["cos0"]) for r in rows]),
            np.array([np.median(r["dist0"]) for r in rows])]


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


def fig_all_curves(tags, labels):
    """Every raw d(t) curve -- one panel per JSD_A quintile, both checkpoints, all 3 contexts."""
    fig, ax = plt.subplots(2, 5, figsize=(15.5, 6.0), sharex=True, sharey=True)
    for row, (tag, lab) in enumerate(zip(tags, labels)):
        Q = qc(tag)
        C = np.load(os.path.join(RESULTS, f"curves_{tag}.npy"))
        grid = np.linspace(0, 1, C.shape[2])
        for q in range(5):
            a = ax[row, q]
            idx = [n for n, r in enumerate(Q["rows"]) if r["bin"] == q]
            for n in idx:
                for ci in range(C.shape[1]):
                    a.plot(grid, C[n, ci], color=CVD[q], lw=0.6, alpha=0.45,
                           ls=LS[ci], solid_capstyle="butt")
            med = np.nanmedian(np.array([C[n, ci] for n in idx for ci in range(C.shape[1])]), axis=0)
            a.plot(grid, med, color="0.15", lw=1.8, ls="-", marker=MARK[q], ms=3, markevery=6,
                   label="bin median")
            a.axhline(0.1, color="0.6", lw=0.7, ls=":")
            a.axhline(0.9, color="0.6", lw=0.7, ls=":")
            w = arr([Q["rows"][n] for n in idx], "w")
            a.set_title(f"Q{q+1}  n={len(idx)} pairs, {len(idx)*C.shape[1]} curves\n"
                        f"median $w$ = {np.nanmedian(w):.3f}", fontsize=9)
            if row == 1:
                a.set_xlabel("interpolation position $t$")
            if q == 0:
                a.set_ylabel(f"{lab}\nrelative logit distance $d(t)$", fontsize=8)
                a.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Every raw curve in the frozen top-256 bank (100% pass the strict validity criteria)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "all_curves.png"))
    plt.close(fig)


def fig_reference_curves(tag):
    Q = qc(tag)
    C = np.load(os.path.join(RESULTS, f"curves_{tag}.npy"))
    grid = np.linspace(0, 1, C.shape[2])
    x = arr(Q["rows"], "jsd_B")
    order = np.argsort(x)
    pick = list(order[:3]) + list(order[-3:])
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for n, i in enumerate(pick):
        r = Q["rows"][i]
        grp = "low" if n < 3 else "high"
        col = CVD[0] if grp == "low" else CVD[1]
        for ci in range(C.shape[1]):  # every carrier context drawn separately, no averaging
            ax.plot(grid, C[i, ci], ls=LS[0] if grp == "low" else LS[1], color=col, lw=1.0,
                    alpha=0.75, marker=MARK[n % 3] if ci == 0 else None, ms=3,
                    label=(f"{r['a_str'].strip()}/{r['b_str'].strip()}  "
                           f"$JSD_B$={r['jsd_B']:.2f} ({grp})") if ci == 0 else None)
    ax.axhline(0.1, color="0.6", lw=0.8, ls=":")
    ax.axhline(0.9, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("interpolation position $t$ (block-0 residual, SLERP)")
    ax.set_ylabel("relative logit distance $d(t)$")
    ax.set_title("Raw curves, 3 lowest vs 3 highest $JSD_B$ pairs\n(all 3 carrier contexts drawn, no averaging)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "reference_curves.png"))
    plt.close(fig)


def fig_jsd_vs_width(Qs, labels):
    fig, ax = plt.subplots(1, len(Qs), figsize=(4.6 * len(Qs), 4.1), squeeze=False)
    for k, (Q, lab) in enumerate(zip(Qs, labels)):
        a = ax[0, k]
        x, y = arr(Q["rows"], "jsd_B"), arr(Q["rows"], "w")
        b = arr(Q["rows"], "bin").astype(int)
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
        a.set_ylabel("transition width $w$  (smaller = sharper)")
        a.set_title(f"{lab}\nSpearman $\\rho$ = {r:+.3f}  [{lo:+.2f}, {hi:+.2f}]  n = {n}")
        if k == 0:
            a.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "jsd_vs_width.png"))
    plt.close(fig)


def fig_width_by_bin(Qs, labels):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    off = np.linspace(-0.18, 0.18, len(Qs))
    for k, (Q, lab) in enumerate(zip(Qs, labels)):
        w = arr(Q["rows"], "w")
        b = arr(Q["rows"], "bin").astype(int)
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
    ax.set_title("Transition width by corpus-divergence bin")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "width_by_jsd_bin.png"))
    plt.close(fig)


def fig_edge_drift(Qs, labels):
    """Is the curve a PLATEAU (flat ends + jump) or just a shallow ramp? Compare edge drift E to
    the no-plateau reference E_LINEAR."""
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.1))
    bins = np.linspace(0, 0.28, 40)
    hatch = ["//", "\\\\", ".."]
    for k, (Q, lab) in enumerate(zip(Qs, labels)):
        e = arr(Q["rows"], "edge_drift")
        ax[0].hist(e, bins=bins, color=CVD[k], histtype="stepfilled", alpha=0.5, hatch=hatch[k],
                   label=f"{lab} (median {np.median(e):.3f})")
    ax[0].axvline(E_LINEAR, color="0.15", ls="--", lw=1.4)
    ax[0].text(E_LINEAR - 0.004, ax[0].get_ylim()[1] * 0.75, "no-plateau reference\n$d(t)=t$",
               ha="right", fontsize=7.5)
    ax[0].set_xlabel("edge drift $E$  (0 = perfectly flat endpoint regions)")
    ax[0].set_ylabel("number of pairs")
    ax[0].set_title("Are the curves actually plateaus?")
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")

    Q = Qs[0]
    x, e, w = arr(Q["rows"], "jsd_B"), arr(Q["rows"], "edge_drift"), arr(Q["rows"], "w")
    ax[1].scatter(w, e, s=26, color=CVD[0], marker="o", alpha=0.85, edgecolors="none",
                  label=f"{labels[0]}  (Spearman {spearmanr(w, e).statistic:+.3f})")
    e0, w0 = arr(Qs[1]["rows"], "edge_drift"), arr(Qs[1]["rows"], "w")
    ax[1].scatter(w0, e0, s=26, color=CVD[1], marker="s", alpha=0.85, edgecolors="none",
                  label=f"{labels[1]}")
    ax[1].axhline(E_LINEAR, color="0.15", ls="--", lw=1.2)
    ax[1].set_xlabel("transition width $w$")
    ax[1].set_ylabel("edge drift $E$")
    ax[1].set_title("Flatness and width carry the same pair-level information")
    ax[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "edge_drift.png"))
    plt.close(fig)


def fig_output_jsd(Q):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    x, y = arr(Q["rows"], "jsd_B"), arr(Q["rows"], "out_jsd_med")
    b = arr(Q["rows"], "bin").astype(int)
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


def fig_bank_comparison(prim, sec):
    """Prespecified top-256 bank vs the post-hoc top-512 bank, same analysis."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    labs = ["1.4B step143000", "1.4B step 0", "410M step143000"]
    xs = np.arange(3)
    for k, (res, name, mk, ls, off) in enumerate(
            [(prim, "top-256 bank (prespecified, n=60)", "o", "none", -0.09),
             (sec, "top-512 bank (post-hoc, n=75)", "s", "none", 0.09)]):
        r = np.array([res[t]["spearman_jsdB_w"] for t in res])
        lo = np.array([res[t]["ci"][0] for t in res])
        hi = np.array([res[t]["ci"][1] for t in res])
        ax.errorbar(xs + off, r, yerr=[r - lo, hi - r], fmt=mk, ls=ls, color=CVD[k], capsize=4,
                    lw=1.4, ms=7, label=name)
    ax.axhline(0, color="0.4", lw=1, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_ylabel(r"Spearman $\rho$($JSD_B$, $w$)")
    ax.set_title("The prespecified bank gives the same conclusion, slightly stronger")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "bank_comparison.png"))
    plt.close(fig)


def fig_formation(form):
    steps = [f["step"] for f in form]
    xpos = [3e2 if s == 0 else s for s in steps]  # step 0 drawn at the left edge of the log axis
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.1))
    r = np.array([f["rho_w"] for f in form])
    lo = np.array([f["ci_w"][0] for f in form])
    hi = np.array([f["ci_w"][1] for f in form])
    ax[0].plot(xpos, r, ls="-", marker="o", color=CVD[0], lw=1.6, label=r"$\rho$($JSD_B$, $w$)")
    ax[0].fill_between(xpos, lo, hi, color=CVD[0], alpha=0.18, hatch="//", edgecolor=CVD[0])
    ax[0].plot(xpos, [f["rho_out"] for f in form], ls="--", marker="s", color=CVD[1], lw=1.6,
               label=r"$\rho$($JSD_B$, model output JSD)")
    ax[0].axhline(0, color="0.4", lw=1, ls=":")
    ax[0].set_xscale("log")
    ax[0].set_xticks(xpos)
    ax[0].set_xticklabels([str(s) for s in steps], fontsize=8)
    ax[0].set_xlabel("training step (log scale; step 0 at the left edge)")
    ax[0].set_ylabel(r"Spearman $\rho$ with corpus $JSD_B$")
    ax[0].set_title("The corpus predictor is at full strength by step 1000")
    ax[0].legend(frameon=False, fontsize=8, loc="center right")

    med = np.array([f["median_w"] for f in form])
    iqr = np.array([f["iqr_w"] for f in form])
    ax[1].plot(xpos, med, ls="-", marker="o", color=CVD[0], lw=1.6, label="median $w$")
    ax[1].fill_between(xpos, med - iqr / 2, med + iqr / 2, color=CVD[0], alpha=0.18, hatch="//",
                       edgecolor=CVD[0], label="median $\\pm$ IQR/2")
    ax[1].axhline(0.8, color="0.15", ls="--", lw=1.2)
    ax[1].text(1.2e3, 0.808, "linear-response value 0.8", fontsize=7.5)
    ax[1].set_xscale("log")
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels([str(s) for s in steps], fontsize=8)
    ax[1].set_xlabel("training step (log scale; step 0 at the left edge)")
    ax[1].set_ylabel("transition width $w$")
    ax[1].set_title("Transitions keep sharpening throughout training")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "formation.png"))
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


# ---------------------------------------------------------------- analysis
def analyse(tags_labels, manifest):
    man = json.load(open(os.path.join(RESULTS, manifest)))
    out = {}
    for tag, lab in tags_labels:
        if not os.path.exists(os.path.join(RESULTS, f"qc_{tag}.json")):
            continue
        Q = qc(tag)
        x, y = arr(Q["rows"], "jsd_B"), arr(Q["rows"], "w")
        e, oj = arr(Q["rows"], "edge_drift"), arr(Q["rows"], "out_jsd_med")
        b = arr(Q["rows"], "bin").astype(int)
        r, lo, hi, n, p = boot_spearman(x, y)
        ro, lo_o, hi_o, no, po = boot_spearman(x, oj)
        re_, lo_e, hi_e, ne, pe = boot_spearman(x, e)
        cov = covariates(Q["rows"], man)
        pr, npart = partial_spearman(x, y, cov)
        out[tag] = dict(
            label=lab, n_pairs=Q["n_pairs"],
            valid_pair_rate=Q["valid_pair_rate"], valid_ctx_rate=Q["valid_curve_rate_context"],
            invalid_pair_rate_by_bin=Q["invalid_pair_rate_by_bin"],
            fail_counts_context=Q["fail_counts_context"], max_backslide=Q["max_backslide_overall"],
            median_w=Q["median_w"], iqr_w=Q["iqr_w"],
            median_edge_drift=Q["median_edge_drift"], edge_drift_linear=E_LINEAR,
            spearman_jsdB_w=r, ci=[lo, hi], p=p, n=n,
            spearman_jsdB_outjsd=ro, ci_outjsd=[lo_o, hi_o], p_outjsd=po,
            spearman_jsdB_edge=re_, ci_edge=[lo_e, hi_e], p_edge=pe,
            partial_spearman_jsdB_w=pr, n_partial=npart,
            spearman_w_edge=float(spearmanr(y, e).statistic),
            median_w_by_bin={str(q): float(np.nanmedian(y[b == q])) for q in range(5)})
    return out


def per_context_rho(tag, manifest):
    """rho computed separately inside each carrier context (contexts are not independent)."""
    Q = qc(tag)
    x = arr(Q["rows"], "jsd_B")
    n_ctx = len(Q["rows"][0]["ctx"])
    return [float(spearmanr(x, np.array([r["ctx"][c]["w"] for r in Q["rows"]],
                                        dtype=float), nan_policy="omit").statistic)
            for c in range(n_ctx)]


if __name__ == "__main__":
    rel = json.load(open(os.path.join(RESULTS, "jsd_reliability.json")))
    npz = np.load(os.path.join(DATA, "reliability_bank.npz"))
    fig_reliability(rel, npz)

    prim = analyse(PRIMARY, "pair_manifest_top256.json")
    sec = analyse(SECONDARY, "pair_manifest.json")
    tags = [t for t, _ in PRIMARY if t in prim]
    Qs = [qc(t) for t in tags]
    labels = ["trained 1.4B (step143000)", "untrained 1.4B (step 0)", "410M (step143000)"]

    fig_all_curves(tags[:2], labels[:2])
    fig_reference_curves(tags[0])
    fig_jsd_vs_width(Qs, [prim[t]["label"] for t in tags])
    fig_width_by_bin(Qs, labels)
    fig_edge_drift(Qs, labels)
    fig_output_jsd(Qs[0])
    fig_bank_comparison(prim, sec)
    bs = fig_block_scan()

    form = []
    for s in FORM_STEPS:
        tag = f"step{s}_t256"
        if not os.path.exists(os.path.join(RESULTS, f"qc_{tag}.json")):
            continue
        Q = qc(tag)
        x, y, oj = arr(Q["rows"], "jsd_B"), arr(Q["rows"], "w"), arr(Q["rows"], "out_jsd_med")
        r, lo, hi, n, p = boot_spearman(x, y)
        form.append(dict(step=s, rho_w=r, ci_w=[lo, hi], p_w=p,
                         rho_out=float(spearmanr(x, oj).statistic),
                         median_w=Q["median_w"], iqr_w=Q["iqr_w"],
                         median_edge_drift=Q["median_edge_drift"],
                         valid_pair_rate=Q["valid_pair_rate"]))
    if len(form) > 2:
        fig_formation(form)

    man = json.load(open(os.path.join(RESULTS, "pair_manifest_top256.json")))
    summary = dict(
        reliability=rel,
        bank=dict(n_pairs=len(man["pairs"]), topk_used=man["topk_used"],
                  per_bin=[sum(p["bin"] == q for p in man["pairs"]) for q in range(5)],
                  n_eligible_endpoints=man["n_eligible_endpoints"],
                  balance_p_logfreq=man["balance_kruskal_p_logfreq"],
                  balance_p_surprisal=man["balance_kruskal_p_surprisal"]),
        primary=prim, secondary_post_hoc=sec, formation=form,
        per_context_rho=per_context_rho(tags[0], "pair_manifest_top256.json"))
    if bs:
        summary["block_scan"] = {str(L): float(np.nanmedian(
            [r["w_by_block"][str(L)] for r in bs["rows"]])) for L in bs["blocks"]}
    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"), indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "invalid_pair_rate_by_bin"}
                      for k, v in prim.items()}, indent=2)[:2500])
    print("formation:", json.dumps(form, indent=1)[:1200])
