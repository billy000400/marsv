"""Do the three onsets depend on the three carrier sentences inherited from dir18?

Every width in this direction is a median over three fixed carrier sentences: the same token pair is
patched into three different contexts and the three d(t) curves are summarised into one w per pair.
Averaging is what makes w reliable enough to correlate at step 32 (the three contexts agree at
r_bar = 0.83 there), but it also means a reader cannot tell whether the timing is a property of the
model or of one lucky sentence frame. If, say, only the third frame carried the divergence ordering,
the median over three would still show it and the report would be reading one context's quirk as a
training-time fact.

So we redo the whole scan three times, once per carrier sentence, with NO averaging: w_c and E_c come
from that single context's curve. All three prespecified onset rules are re-run verbatim on each:

  ordering   rho_c(s)  = Spearman(J, w_c(s)),  simultaneous 95% band over the 20 checkpoints;
             bracket = (last checkpoint whose band contains 0, first of two consecutive below 0)
  shape      median w_c(s) band below the straight-line 0.8 AND median E_c(s) band below E_LINEAR,
             for two consecutive checkpoints
  ranking    pi_c(s) = Spearman(w_c(s), w_c(final)), family-wise label-permutation p < 0.05 for two
             consecutive checkpoints

A single context is a NOISIER measurement than the three-context median, so the honest expectation is
that brackets can only move LATER (attenuation), never earlier. The question this answers is whether
they move at all, and whether any one frame disagrees with the other two about the ORDER of events.

CPU only; reads the saved curves. Writes results/sentence_jackknife.json.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

import curve_metrics
from analyze import GRID, load_checkpoints
from common import RESULTS

B_BOOT = 4000
B_PERM = 20000
RNG = np.random.default_rng(37)
E_LINEAR = curve_metrics.E_LINEAR
W_LINEAR = 0.8


def rank_rows(X):
    """Ranks along axis 1 (rows are bootstrap draws / permutations)."""
    return rankdata(X, axis=1)


def corr_rows(A, B):
    """Pearson correlation of matching rows of two equally shaped matrices."""
    a = A - A.mean(1, keepdims=True)
    b = B - B.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def simultaneous(stat_boot, stat):
    """95% band covering the WHOLE trajectory (max deviation across checkpoints)."""
    dev = np.nanmax(np.abs(stat_boot - stat[None, :]), axis=1)
    c = float(np.nanpercentile(dev, 95))
    return stat - c, stat + c


def bracket_ordering(lo, hi):
    """Last checkpoint whose band contains 0, then first of two consecutive bands below 0."""
    for k in range(len(hi) - 1):
        if hi[k] < 0 and hi[k + 1] < 0:
            prev = [j for j in range(k) if lo[j] <= 0 <= hi[j]]
            return (prev[-1] if prev else None), k
    return None, None


def bracket_shape(med_hi, e_hi):
    """Two consecutive checkpoints below BOTH straight-line references, bands excluded."""
    for k in range(len(med_hi) - 1):
        if (med_hi[k] < W_LINEAR and med_hi[k + 1] < W_LINEAR
                and e_hi[k] < E_LINEAR and e_hi[k + 1] < E_LINEAR):
            return (k - 1 if k > 0 else None), k
    return None, None


def bracket_ranking(p_fw, ref_k):
    """Two consecutive checkpoints with family-wise permutation p < 0.05, before the reference."""
    for k in range(1, ref_k):
        if p_fw[k] < 0.05 and k + 1 < ref_k and p_fw[k + 1] < 0.05:
            return k - 1, k
    return None, None


def main():
    cps = load_checkpoints()
    steps = np.array([c["step"] for c in cps])
    J = np.array([r["jsd_B"] for r in cps[0]["rows"]], float)
    n_ck, n_p, n_c = len(cps), len(J), cps[0]["curves"].shape[1]
    print(f"{n_ck} checkpoints x {n_p} pairs x {n_c} carrier sentences")

    # per-context width and edge drift, NO median over contexts
    W = np.full((n_c, n_ck, n_p), np.nan)
    E = np.full((n_c, n_ck, n_p), np.nan)
    for k, a in enumerate(cps):
        cur = a["curves"]
        for c in range(n_c):
            for i in range(n_p):
                m = curve_metrics.metrics(cur[i, c], GRID)
                W[c, k, i], E[c, k, i] = m["w"], m["edge_drift"]
    print("invalid (NaN) widths per context:", [int(np.isnan(W[c]).sum()) for c in range(n_c)])

    idx = RNG.integers(0, n_p, size=(B_BOOT, n_p))
    perm = np.array([RNG.permutation(n_p) for _ in range(B_PERM)])
    rJ = rankdata(J)
    ref = n_ck - 1

    out = {"steps": [int(s) for s in steps], "n_pairs": n_p, "contexts": {}}
    for c in list(range(n_c)) + ["median3"]:
        if c == "median3":
            Wc, Ec = np.nanmedian(W, 0), np.nanmedian(E, 0)
        else:
            Wc, Ec = W[c], E[c]

        # --- ordering: rho(J, w) with a simultaneous band from the paired pair-bootstrap ---
        rho = np.array([float(np.corrcoef(rJ, rankdata(Wc[k]))[0, 1]) for k in range(n_ck)])
        rJb = rank_rows(J[idx])
        rho_b = np.empty((B_BOOT, n_ck))
        med_b = np.empty((B_BOOT, n_ck))
        e_b = np.empty((B_BOOT, n_ck))
        for k in range(n_ck):
            Wb = Wc[k][idx]
            rho_b[:, k] = corr_rows(rJb, rank_rows(Wb))
            med_b[:, k] = np.median(Wb, axis=1)
            e_b[:, k] = np.median(Ec[k][idx], axis=1)
        rho_lo, rho_hi = simultaneous(rho_b, rho)
        b_ord = bracket_ordering(rho_lo, rho_hi)

        # --- shape: median w and median edge drift against their straight-line references ---
        med_w = np.median(Wc, axis=1)
        med_e = np.median(Ec, axis=1)
        _, medw_hi = simultaneous(med_b, med_w)
        _, mede_hi = simultaneous(e_b, med_e)
        b_shape = bracket_shape(medw_hi, mede_hi)

        # --- ranking: pi(s) against this context's own final widths, permutation family-wise ---
        R = np.array([rankdata(w) - rankdata(w).mean() for w in Wc])
        rref = R[ref]
        n_ref = np.linalg.norm(rref)
        pi = np.array([float(R[k] @ rref / (np.linalg.norm(R[k]) * n_ref)) for k in range(n_ck)])
        pi_p = np.empty((B_PERM, n_ck))
        for k in range(n_ck):
            pi_p[:, k] = (R[k][perm] @ rref) / (np.linalg.norm(R[k]) * n_ref)
        keep = np.arange(n_ck) != ref
        maxnull = np.nanmax(np.abs(pi_p[:, keep]), axis=1)
        p_fw = np.array([(np.sum(maxnull >= abs(pi[k])) + 1) / (B_PERM + 1) for k in range(n_ck)])
        b_rank = bracket_ranking(p_fw, ref)

        key = "median3" if c == "median3" else f"ctx{c}"
        out["contexts"][key] = dict(
            rho=[float(x) for x in rho],
            rho_lo=[float(x) for x in rho_lo], rho_hi=[float(x) for x in rho_hi],
            median_w=[float(x) for x in med_w], median_w_hi=[float(x) for x in medw_hi],
            median_e=[float(x) for x in med_e], median_e_hi=[float(x) for x in mede_hi],
            pi=[float(x) for x in pi], p_fw=[float(x) for x in p_fw],
            null_pt95=[float(x) for x in np.nanpercentile(np.abs(pi_p), 95, axis=0)],
            bracket_ordering=[None if b_ord[0] is None else int(steps[b_ord[0]]),
                              None if b_ord[1] is None else int(steps[b_ord[1]])],
            bracket_shape=[None if b_shape[0] is None else int(steps[b_shape[0]]),
                           None if b_shape[1] is None else int(steps[b_shape[1]])],
            bracket_ranking=[None if b_rank[0] is None else int(steps[b_rank[0]]),
                             None if b_rank[1] is None else int(steps[b_rank[1]])],
        )
        print(f"{key}: ordering {out['contexts'][key]['bracket_ordering']} "
              f"shape {out['contexts'][key]['bracket_shape']} "
              f"ranking {out['contexts'][key]['bracket_ranking']}")

    # how much each context agrees with the others about per-pair width, per checkpoint
    agree = []
    for k in range(n_ck):
        rs = [float(np.corrcoef(rankdata(W[i, k]), rankdata(W[j, k]))[0, 1])
              for i in range(n_c) for j in range(i + 1, n_c)]
        agree.append(float(np.mean(rs)))
    out["ctx_agreement_rbar"] = agree

    with open(os.path.join(RESULTS, "sentence_jackknife.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/sentence_jackknife.json")


if __name__ == "__main__":
    main()
