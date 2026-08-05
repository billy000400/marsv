"""Endpoint-clustered inference for the secondary ~1000-pair bank (endpoints recur).

With endpoint reuse the pairs are not independent, so a naive Spearman p-value is invalid. We report:
  * the point estimate rho(J_holdout, w);
  * a dyadic (pigeonhole) endpoint bootstrap CI -- resample the 123 endpoints with replacement and
    weight each pair by the product of its two endpoints' multiplicities;
  * an endpoint-label permutation (QAP) p-value -- relabel endpoints, so each assayed pair keeps its
    measured w but takes the corpus JSD of a different endpoint pair;
  * binned medians over the JSD range, to see whether the relationship is monotone or curved.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
MARK = ["o", "s", "^", "D", "v"]

N_BOOT = 4000
N_PERM = 4000
SEED = 11


def spearman_w(x, y, wt):
    """Weighted Spearman: ranks of the weighted multiset, computed via replication weights."""
    keep = wt > 0
    x, y, wt = x[keep], y[keep], wt[keep]
    rx, ry = weighted_rank(x, wt), weighted_rank(y, wt)
    mx = np.average(rx, weights=wt)
    my = np.average(ry, weights=wt)
    cov = np.average((rx - mx) * (ry - my), weights=wt)
    return cov / np.sqrt(np.average((rx - mx) ** 2, weights=wt)
                         * np.average((ry - my) ** 2, weights=wt))


def weighted_rank(x, wt):
    """Mid-rank of each value in the multiset that repeats element i exactly wt[i] times."""
    o = np.argsort(x, kind="mergesort")
    cw = np.cumsum(wt[o])
    r = np.empty_like(x, dtype=float)
    r[o] = cw - (wt[o] - 1) / 2.0   # mid-rank of the block of tied copies
    return r


def figure(jB, w, binq, res, prim):
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for q in range(5):
        m = binq == q
        ax[0].scatter(jB[m], w[m], s=9, color=CVD[q], marker=MARK[q], alpha=0.55,
                      edgecolors="none", label=f"JSD group {q+1}")
    b = res["binned"]
    ax[0].errorbar([x["jsd"] for x in b], [x["w"] for x in b],
                   yerr=[[x["w"] - x["w_lo"] for x in b], [x["w_hi"] - x["w"] for x in b]],
                   fmt="x--", color="0.15", lw=1.5, ms=7, capsize=3,
                   label="median $w$ in 10 non-overlapping\nJSD bins (bars = IQR)")
    ax[0].set_xlabel("corpus next-token JSD $J(u,v)$ [bits] (measurement sample)")
    ax[0].set_ylabel("transition width $w$  (smaller = sharper)")
    ax[0].set_title(f"{res['n_pairs']} pairs built from {res['n_endpoints']} tokens\n"
                    f"Spearman $\\rho$ = {res['rho']:+.3f}")
    ax[0].legend(frameon=False, fontsize=7, loc="lower left")

    names = [f"{prim['n']}-pair controlled set\n(no token reused; bootstrap over pairs)",
             f"{res['n_pairs']}-pair set, uncertainty accounting\nfor tokens reused across pairs",
             f"{res['n_pairs']}-pair set, uncertainty ignoring\ntoken reuse (invalid here)"]
    vals = [(prim["rho"], prim["ci"]), (res["rho"], res["boot_ci"]), (res["rho"], res["naive_ci"])]
    for k, ((r, ci), nm) in enumerate(zip(vals, names)):
        ax[1].errorbar([r], [2 - k], xerr=[[r - ci[0]], [ci[1] - r]], fmt=MARK[k], color=CVD[k],
                       capsize=4, ms=7, lw=1.6)
    ax[1].set_yticks([2, 1, 0])
    ax[1].set_yticklabels(names, fontsize=7)
    ax[1].axvline(0, color="0.5", ls=":", lw=1)
    ax[1].set_xlabel("Spearman $\\rho$($J$, $w$) with 95% CI")
    ax[1].set_title("Accounting for token reuse widens the interval\n"
                    f"token-relabelling permutation $p$ = {res['perm_p']:.4f}")
    ax[1].set_ylim(-0.6, 2.6)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "large_bank.png"))
    plt.close(fig)


def main(tag="step143000"):
    man = json.load(open(os.path.join(RESULTS, "pair_manifest_large.json")))
    assay = json.load(open(os.path.join(RESULTS, f"assay_large_{tag}.json")))
    rows = assay["rows"]
    pairs = man["pairs"]
    E = np.array(man["eligible_endpoints"])
    pos = {int(e): k for k, e in enumerate(E)}
    Jm = np.array(man["jsd_holdout_matrix"])

    w = np.array([r["w"] for r in rows])
    jB = np.array([pairs[r["pair_idx"]]["jsd_B"] for r in rows])
    jA = np.array([pairs[r["pair_idx"]]["jsd_A"] for r in rows])
    oj = np.array([r["out_jsd_med"] for r in rows])
    ea = np.array([pos[pairs[r["pair_idx"]]["ep_a"]] for r in rows])
    eb = np.array([pos[pairs[r["pair_idx"]]["ep_b"]] for r in rows])
    ok = np.isfinite(w)
    w, jB, jA, oj, ea, eb = w[ok], jB[ok], jA[ok], oj[ok], ea[ok], eb[ok]
    n = len(w)
    print(f"n = {n} valid pairs over {len(set(ea) | set(eb))} endpoints")

    rho = spearmanr(jB, w).statistic
    rho_naive_p = spearmanr(jB, w).pvalue
    rho_out = spearmanr(jB, oj).statistic
    rho_sel = spearmanr(jA, w).statistic

    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        m = np.bincount(rng.integers(0, len(E), len(E)), minlength=len(E)).astype(float)
        boot[b] = spearman_w(jB, w, m[ea] * m[eb])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    perm = np.empty(N_PERM)
    for b in range(N_PERM):
        p = rng.permutation(len(E))
        perm[b] = spearmanr(Jm[p[ea], p[eb]], w).statistic
    n_extreme = int((np.abs(perm) >= abs(rho)).sum())
    p_perm = max(n_extreme / N_PERM, 1.0 / N_PERM)

    # naive (invalid) bootstrap over pairs, for the size of the dependence correction
    nb = np.empty(1000)
    for b in range(1000):
        i = rng.integers(0, n, n)
        nb[b] = spearmanr(jB[i], w[i]).statistic
    nlo, nhi = np.percentile(nb, [2.5, 97.5])

    q = np.quantile(jB, np.linspace(0, 1, 11))
    q[-1] += 1e-9
    bid = np.clip(np.digitize(jB, q[1:-1]), 0, 9)
    binned = [dict(k=int(k), n=int((bid == k).sum()), jsd=float(np.median(jB[bid == k])),
                   w=float(np.median(w[bid == k])),
                   w_lo=float(np.percentile(w[bid == k], 25)),
                   w_hi=float(np.percentile(w[bid == k], 75))) for k in range(10)]

    out = dict(n_pairs=n, n_endpoints=int(len(set(ea) | set(eb))),
               rho=float(rho), naive_p=float(rho_naive_p),
               boot_ci=[float(lo), float(hi)], boot_sd=float(boot.std()),
               naive_ci=[float(nlo), float(nhi)], naive_sd=float(nb.std()),
               perm_p=p_perm, n_perm=N_PERM, n_perm_extreme=n_extreme, perm_sd=float(perm.std()),
               perm_q975=float(np.percentile(np.abs(perm), 97.5)),
               rho_selection_split=float(rho_sel), rho_out_jsd=float(rho_out),
               median_w=float(np.median(w)), iqr_w=float(np.percentile(w, 75) - np.percentile(w, 25)),
               valid_rate=float(ok.mean()), binned=binned,
               boot_samples=[float(x) for x in boot[:500]])
    name = "large_bank.json" if tag == "step143000" else f"large_bank_{tag}.json"
    json.dump(out, open(os.path.join(RESULTS, name), "w"), indent=2)

    if tag == "step143000":  # the figure carries the primary checkpoint only
        prim = json.load(open(os.path.join(RESULTS, "summary.json")))["primary"]["step143000_t256"]
        figure(jB, w, np.array([pairs[r["pair_idx"]]["bin"] for r in rows])[ok], out,
               dict(rho=prim["spearman_jsdB_w"], ci=prim["ci"], n=prim["n"]))
    print(f"rho(J_holdout, w) = {rho:.4f}")
    print(f"  endpoint-cluster bootstrap 95% CI [{lo:.3f}, {hi:.3f}]  (sd {boot.std():.3f})")
    print(f"  naive pair bootstrap  95% CI [{nlo:.3f}, {nhi:.3f}]  (sd {nb.std():.3f})")
    print(f"  endpoint-label permutation p = {p_perm:.4g}   (naive p = {rho_naive_p:.2g}, invalid)")
    print(f"  rho(selection split) = {rho_sel:.4f}   rho(J, out_jsd) = {rho_out:.4f}")
    print(f"  median w {np.median(w):.3f}  valid rate {ok.mean():.3f}")
    for b in binned:
        print(f"   bin {b['k']}: n={b['n']:3d}  J={b['jsd']:.3f}  median w={b['w']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "step143000")
