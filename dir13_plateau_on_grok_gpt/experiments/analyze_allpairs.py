"""Experiment 5 / S9b-S9c — turn the raw all-pairs sweep into decidable numbers.

Reads results/allpairs_summary.json (written by allpairs_sweep.py) and appends an "analysis" block:
  * diagnostics + swap-symmetry check;
  * per-character med_w / flat_frac / strict_frac  (S9b, "is each character in its own plateau?");
  * variance decomposition w_ij ~ a_i + a_j with a permutation null  (verdict i / ii / iii);
  * readout-decision test  (t* vs first argmax flip, number of argmax regions);
  * depth and learned-vs-init controls;
  * plausibility correlations, incl. partial correlations against endpoint separation.
"""
import os, sys, json
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
GRID = 1.0 / 49                       # one interpolation step of the 50-point grid


def char_class(c):
    if c in " \n":
        return "space"
    if c.isupper():
        return "upper"
    if c.islower():
        return "lower"
    return "punct"


def rank(x):
    return stats.rankdata(x)


def partial_spearman(x, y, z):
    """Spearman correlation of x,y after linearly removing rank(z) from both rank vectors."""
    rx, ry, rz = rank(x), rank(y), rank(z)
    Z = np.stack([np.ones_like(rz), rz], 1)
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p)


def var_decomp(idx_i, idx_j, w, V, n_perm=200, seed=0):
    """Least-squares fit w_ij ~ mu + a_i + a_j; returns R^2, adjusted R^2, a permutation null,
    and the fitted per-character effects. The design is rank-deficient by one (a common shift can
    move between mu and the a's), so lstsq's minimum-norm solution is used."""
    n = len(w)
    X = np.zeros((n, V + 1))
    X[:, 0] = 1.0
    X[np.arange(n), 1 + idx_i] += 1.0
    X[np.arange(n), 1 + idx_j] += 1.0
    rk = np.linalg.matrix_rank(X)

    def r2(y):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ beta
        return 1.0 - res @ res / ((y - y.mean()) @ (y - y.mean())), beta

    R2, beta = r2(w)
    adj = 1 - (1 - R2) * (n - 1) / (n - rk)
    rng = np.random.default_rng(seed)
    null = [r2(rng.permutation(w))[0] for _ in range(n_perm)]
    return {"r2": float(R2), "adj_r2": float(adj), "rank": int(rk), "n": int(n),
            "null_r2_mean": float(np.mean(null)), "null_r2_p99": float(np.percentile(null, 99)),
            "effects": [float(b) for b in beta[1:]], "mu": float(beta[0])}


def main():
    S = json.load(open(os.path.join(RES, "allpairs_summary.json")))
    chars = S["chars"]
    V = S["vocab_size"]
    fin = S["final_block0"]
    A = {}

    # ---- diagnostics -------------------------------------------------------------------------
    d0 = np.array([p["d0"] for p in fin]); d1 = np.array([p["d1"] for p in fin])
    bad = (d0 >= 1e-3) | (d1 <= 1 - 1e-3)
    A["diagnostics"] = {
        "n_pairs": len(fin), "n_dropped_endpoint": int(bad.sum()),
        "max_d0": float(d0.max()), "min_d1": float(d1.min()),
        "max_endpoint_err": float(max(p["ep_err"] for p in fin)),
        "max_prefix_err": float(max(p["prefix_err"] for p in fin)),
        "n_width_undefined": int(sum(p["w"] is None for p in fin)),
        "max_iso_dev": float(max(p["iso_dev"] for p in fin)),
        "n_nonmonotone_dev_gt_0.10": int(sum(p["iso_dev"] > 0.10 for p in fin))}
    assert bad.sum() == 0, "endpoint reproduction failed for some pairs"

    # ---- swap symmetry -----------------------------------------------------------------------
    fwd = {(p["i"], p["j"]): p for p in fin}
    dw = [abs(sw["w"] - fwd[tuple(ref)]["w"])
          for sw, ref in zip(S["swap"], S["swap_of"]) if sw["w"] is not None]
    A["symmetry"] = {"n": len(dw), "median_abs_dw": float(np.median(dw)),
                     "p90_abs_dw": float(np.percentile(dw, 90)), "max_abs_dw": float(np.max(dw)),
                     "median_w": float(np.median([p["w"] for p in fin if p["w"] is not None]))}

    # ---- per-pair vectors --------------------------------------------------------------------
    ii = np.array([p["i"] for p in fin]); jj = np.array([p["j"] for p in fin])
    w = np.array([p["w"] if p["w"] is not None else np.nan for p in fin])
    t_lo = np.array([p["t_lo"] for p in fin], dtype=float)
    t_hi = np.array([p["t_hi"] for p in fin], dtype=float)
    t_star = np.array([p["t_star"] for p in fin], dtype=float)
    t_flip = np.array([np.nan if p["t_flip"] is None else p["t_flip"] for p in fin])
    n_am = np.array([p["n_argmax"] for p in fin])
    strict = np.array([p["plateau"] for p in fin])
    sep = np.array(S["logit_sep"])
    pn = np.array(S["p_next"])
    pA, pB = pn[ii], pn[jj]

    # ---- per-character statistics (S9b) ------------------------------------------------------
    per_char = []
    for c in range(V):
        as_A = ii == c
        as_B = jj == c
        m = as_A | as_B
        flat = np.where(as_A[m], t_lo[m] >= 0.10, t_hi[m] <= 0.90)
        per_char.append({"char": chars[c], "idx": c, "class": char_class(chars[c]),
                         "n": int(m.sum()), "med_w": float(np.nanmedian(w[m])),
                         "iqr_w": [float(np.nanpercentile(w[m], 25)), float(np.nanpercentile(w[m], 75))],
                         "flat_frac": float(np.mean(flat)),
                         "strict_frac": float(np.mean(strict[m])),
                         "p_next": float(pn[c])})
    A["per_char"] = per_char
    mw = np.array([p["med_w"] for p in per_char])
    ff = np.array([p["flat_frac"] for p in per_char])
    A["per_char_summary"] = {
        "med_w_range": [float(mw.min()), float(mw.max())], "med_w_median": float(np.median(mw)),
        "flat_frac_range": [float(ff.min()), float(ff.max())], "flat_frac_median": float(np.median(ff)),
        "n_char_flat_frac_ge_0.5": int((ff >= 0.5).sum()),
        "n_char_strict_frac_ge_0.5": int(sum(p["strict_frac"] >= 0.5 for p in per_char)),
        "argmin_med_w": chars[int(np.argmin(mw))], "argmax_med_w": chars[int(np.argmax(mw))],
        "spearman_med_w_vs_p_next": [float(x) for x in
                                     stats.spearmanr(mw, np.log10(np.array([p["p_next"] for p in per_char]) + 1e-30))]}

    # ---- variance decomposition (verdict i / ii / iii) ----------------------------------------
    ok = np.isfinite(w)
    A["var_decomp"] = var_decomp(ii[ok], jj[ok], w[ok], V)
    A["var_decomp"]["resid_frac"] = 1.0 - A["var_decomp"]["r2"]

    # ---- readout-decision test (S9c.1) -------------------------------------------------------
    # Sharper form of the test: are ALL next-character decision changes confined to the transition
    # window [t_lo, t_hi]? If so the two flat arms of d(t) are each a single readout decision region.
    R = np.load(os.path.join(RES, "allpairs_raw.npz"))
    tsg = R["ts"]
    in_win, flat_const = [], []
    for p in fin:
        am = R[f"final|L0|am|{p['i']}_{p['j']}"]
        f = np.nonzero(am[1:] != am[:-1])[0]
        if len(f) == 0 or p["t_lo"] is None:
            continue
        tf = 0.5 * (tsg[f] + tsg[f + 1])
        in_win.append(float(np.mean((tf >= p["t_lo"]) & (tf <= p["t_hi"]))))
        lo_arm, hi_arm = am[tsg < p["t_lo"]], am[tsg > p["t_hi"]]
        flat_const.append(bool(len(np.unique(lo_arm)) <= 1 and len(np.unique(hi_arm)) <= 1))
    A["readout_window"] = {"n": len(in_win),
                           "mean_frac_flips_in_window": float(np.mean(in_win)),
                           "frac_pairs_all_flips_in_window": float(np.mean(np.array(in_win) == 1.0)),
                           "frac_pairs_flat_arms_single_decision": float(np.mean(flat_const))}

    have = np.isfinite(t_flip) & np.isfinite(t_star)
    diff = t_star[have] - t_flip[have]
    A["readout"] = {
        "n": int(have.sum()), "median_abs_diff": float(np.median(np.abs(diff))),
        "frac_within_1_grid": float(np.mean(np.abs(diff) <= GRID)),
        "frac_within_2_grid": float(np.mean(np.abs(diff) <= 2 * GRID)),
        "median_signed_diff": float(np.median(diff)),
        "n_argmax_hist": {str(k): int((n_am == k).sum()) for k in np.unique(n_am)},
        "frac_exactly_2_argmax": float(np.mean(n_am == 2)),
        "median_n_argmax": float(np.median(n_am)),
        "frac_endpoints_are_own_argmax": float(np.mean(
            (np.array([p["am0"] for p in fin]) == ii) & (np.array([p["am1"] for p in fin]) == jj))),
        "spearman_w_vs_n_argmax": [float(x) for x in stats.spearmanr(w[ok], n_am[ok])]}

    # ---- depth control (S9c.2) ---------------------------------------------------------------
    dep = [tuple(p) for p in S["depth_pairs"]]
    depw = {"0": [fwd[p]["w"] for p in dep if fwd[p]["w"] is not None]}
    for L, rows in S["depth"].items():
        depw[L] = [r["w"] for r in rows if r["w"] is not None]
    A["depth"] = {L: {"n": len(v), "median_w": float(np.median(v)),
                      "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}
                  for L, v in depw.items()}

    # ---- learned-vs-init control -------------------------------------------------------------
    ini = S["init_block0"]
    wi = np.array([p["w"] if p["w"] is not None else np.nan for p in ini])
    oi = np.isfinite(wi)
    A["init_control"] = {
        "n": int(oi.sum()), "median_w_init": float(np.median(wi[oi])),
        "median_w_final": float(np.median(w[ok])),
        "iqr_init": [float(np.percentile(wi[oi], 25)), float(np.percentile(wi[oi], 75))],
        "iqr_final": [float(np.percentile(w[ok], 25)), float(np.percentile(w[ok], 75))],
        "strict_frac_init": float(np.mean([p["plateau"] for p in ini])),
        "strict_frac_final": float(np.mean(strict)),
        "frac_init_near_linear_w_ge_0.7": float(np.mean(wi[oi] >= 0.7)),
        "frac_final_near_linear_w_ge_0.7": float(np.mean(w[ok] >= 0.7)),
        "mannwhitney_p": float(stats.mannwhitneyu(wi[oi], w[ok]).pvalue),
        "median_n_argmax_init": float(np.median([p["n_argmax"] for p in ini]))}

    # ---- plausibility confound ---------------------------------------------------------------
    lp = np.log10(pn + 1e-30)
    maxp = np.maximum(pA, pB)
    dlogp = np.abs(lp[ii] - lp[jj])
    A["plausibility"] = {
        "spearman_w_vs_max_p": [float(x) for x in stats.spearmanr(w[ok], maxp[ok])],
        "spearman_w_vs_abs_dlogp": [float(x) for x in stats.spearmanr(w[ok], dlogp[ok])],
        "spearman_w_vs_logit_sep": [float(x) for x in stats.spearmanr(w[ok], sep[ok])],
        "partial_w_vs_max_p_given_sep": partial_spearman(w[ok], maxp[ok], sep[ok]),
        "partial_w_vs_logit_sep_given_max_p": partial_spearman(w[ok], sep[ok], maxp[ok]),
        "spearman_tstar_vs_dlogp_signed": [float(x) for x in
                                           stats.spearmanr(t_star[ok], (lp[ii] - lp[jj])[ok])]}

    S["analysis"] = A
    json.dump(S, open(os.path.join(RES, "allpairs_summary.json"), "w"), indent=1)

    print(json.dumps({k: v for k, v in A.items() if k not in ("per_char",)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
