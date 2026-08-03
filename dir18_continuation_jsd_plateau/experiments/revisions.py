"""Operator-requested revisions: learned-sharpening outcome, mediation ladder, late reversal,
and the word-fragment sensitivity check.

Writes results/revisions.json and plots/mediation.png. Run BEFORE analyze.py (which reads
revisions.json for the fragment-dropped series in the bank-comparison figure).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

from analyze import BOOT, CVD, MARK, RESULTS, PLOTS, arr, boot_spearman, covariates, qc

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
RNG = np.random.default_rng(1)
FRAGMENT = "un"  # the only bank endpoint that is a word-start fragment, not a complete word


def partial(x, y, covs):
    """Spearman of x and y after linear adjustment for covs (all on ranks), with p and bootstrap CI."""
    def stat(x, y, covs):
        rx, ry = rankdata(x), rankdata(y)
        C = np.column_stack([rankdata(c) for c in covs] + [np.ones(len(x))])
        ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
        ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
        return spearmanr(ex, ey)

    s = stat(x, y, covs)
    bs = np.empty(BOOT)
    for i in range(BOOT):
        k = RNG.integers(0, len(x), len(x))
        bs[i] = stat(x[k], y[k], [c[k] for c in covs]).statistic
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(rho=float(s.statistic), p=float(s.pvalue), ci=[float(lo), float(hi)], n=len(x))


def main():
    Qt, Q0 = qc("step143000_t256"), qc("step0_t256")
    man = json.load(open(os.path.join(RESULTS, "pair_manifest_top256.json")))
    x = arr(Qt["rows"], "jsd_B")
    w, w0 = arr(Qt["rows"], "w"), arr(Q0["rows"], "w")
    oj = arr(Qt["rows"], "out_jsd_med")
    dw = w - w0                      # learned sharpening: how much training narrowed the transition
    cov5 = covariates(Qt["rows"], man)

    out = {}
    r, lo, hi, n, p = boot_spearman(x, w)
    out["total_w"] = dict(rho=r, ci=[lo, hi], p=p, n=n)
    r, lo, hi, n, p = boot_spearman(x, dw)
    out["learned_sharpening_dw"] = dict(rho=r, ci=[lo, hi], p=p, n=n,
                                        median_dw=float(np.median(dw)))
    out["dw_ctrl_outjsd"] = partial(x, dw, [oj])
    out["dw_ctrl_outjsd_plus5"] = partial(x, dw, [oj] + cov5)
    out["w_ctrl_outjsd"] = partial(x, w, [oj])
    out["w_ctrl_5cov"] = partial(x, w, cov5)
    out["w_ctrl_outjsd_plus5"] = partial(x, w, [oj] + cov5)

    # --- late reversal: is the 64k -> 143k rebound in median w real at the pair level?
    Q64 = qc("step64000_t256")
    w64 = arr(Q64["rows"], "w")
    wil = wilcoxon(w, w64)  # paired, same 60 pairs
    out["late_reversal"] = dict(
        median_w_64k=float(np.median(w64)), median_w_final=float(np.median(w)),
        n_blunter=int((w > w64).sum()), n_sharper=int((w < w64).sum()), n=len(w),
        wilcoxon_stat=float(wil.statistic), wilcoxon_p=float(wil.pvalue),
        median_delta=float(np.median(w - w64)))

    # --- sensitivity: drop the one pair whose endpoint is a word-start FRAGMENT, not a full word
    keep = np.array([FRAGMENT not in (rw["a_str"].strip(), rw["b_str"].strip())
                     for rw in Qt["rows"]])
    out["fragment_pair"] = [f"{rw['a_str'].strip()}/{rw['b_str'].strip()}" for rw in Qt["rows"]
                            if FRAGMENT in (rw["a_str"].strip(), rw["b_str"].strip())]
    drop = {}
    for tag, lab in [("step143000_t256", "1.4B step143000"), ("step0_t256", "1.4B step 0"),
                     ("step143000_410m_t256", "410M step143000")]:
        Q = qc(tag)
        xx, yy = arr(Q["rows"], "jsd_B")[keep], arr(Q["rows"], "w")[keep]
        r, lo, hi, n, p = boot_spearman(xx, yy)
        drop[tag] = dict(label=lab, spearman_jsdB_w=r, ci=[lo, hi], p=p, n=n)
    out["drop_fragment"] = drop

    json.dump(out, open(os.path.join(RESULTS, "revisions.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    fig_mediation(x, dw, out)
    return out


def fig_mediation(x, dw, out):
    """Left: the learned-sharpening outcome. Right: how far the association survives adjustment."""
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))
    Qt = qc("step143000_t256")
    b = arr(Qt["rows"], "bin").astype(int)
    for q in range(5):
        m = b == q
        ax[0].scatter(x[m], dw[m], s=26, color=CVD[q], marker=MARK[q], alpha=0.85,
                      edgecolors="none", label=f"$JSD_A$ quintile {q+1}")
    ax[0].axhline(0, color="0.4", lw=1, ls=":")
    s = out["learned_sharpening_dw"]
    ax[0].set_xlabel("$JSD_B$: corpus continuation divergence (bits)")
    ax[0].set_ylabel(r"$\Delta w = w_{\rm trained} - w_{\rm step\,0}$" "\n(more negative = training sharpened it more)")
    ax[0].set_title("Learned sharpening\n"
                    f"Spearman $\\rho$ = {s['rho']:+.3f}  [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]  n = {s['n']}")
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower left")

    keys = ["total_w", "learned_sharpening_dw", "w_ctrl_outjsd", "w_ctrl_outjsd_plus5"]
    names = ["total: $\\rho(JSD_B,\\ w)$\n(headline, unadjusted)",
             "learned sharpening:\n$\\rho(JSD_B,\\ \\Delta w)$",
             "$w$, adjusted for the\nmediator (model output JSD)",
             "$w$, adjusted for output\nJSD + 5 covariates"]
    r = [out[k]["rho"] for k in keys]
    lo = [out[k]["ci"][0] for k in keys]
    hi = [out[k]["ci"][1] for k in keys]
    ypos = np.arange(len(keys))[::-1]
    for i, k in enumerate(keys):
        sig = out[k]["p"] < 0.05
        ax[1].errorbar(r[i], ypos[i], xerr=[[r[i] - lo[i]], [hi[i] - r[i]]],
                       fmt=MARK[i], color=CVD[0] if sig else CVD[1], capsize=4, ms=8,
                       markerfacecolor=CVD[0] if sig else "white", lw=1.5)
        ax[1].annotate(f"$\\rho$={r[i]:+.3f}, p={out[k]['p']:.3g}", (r[i], ypos[i]),
                       textcoords="offset points", xytext=(0, 11), ha="center", fontsize=7.5)
    ax[1].axvline(0, color="0.4", lw=1, ls=":")
    ax[1].set_yticks(ypos)
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_ylim(-0.6, len(keys) - 0.4)
    ax[1].set_xlabel(r"Spearman $\rho$ with corpus $JSD_B$ (95% bootstrap CI)")
    ax[1].set_title("The fully adjusted association is not significant\n"
                    "(filled marker: p < 0.05; open marker: p > 0.05)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "mediation.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
