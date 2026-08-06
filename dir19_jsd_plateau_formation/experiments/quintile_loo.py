"""Is the step-32 divergence ordering carried by the extremes of the divergence range?

Result 1 reports rho(J, w) = -0.428 at step 32 across 60 pairs whose corpus next-token JSD J spans
0.137 to 0.942 bits. A rank correlation over a wide range can be produced entirely by its tails: if
only the lowest-J and highest-J pairs differ, the "ordering" is a two-group contrast dressed up as a
graded relationship, and the onset time would then be the time at which those two extreme groups
separate rather than the time at which J starts ranking pairs.

So we re-run the ordering onset rule on subsets that remove divergence groups:

    - leave-one-quintile-out: drop Q1 ... Q5 in turn (n = 46 to 50),
    - middle three only: drop BOTH tails, Q1 and Q5 (n = 36, J in 0.514 to 0.756 bits).

For each subset we recompute the full 20-checkpoint trajectory rho_s, its SIMULTANEOUS 95% band
(trajectory bootstrap over the subset's pairs, same resampled pairs at every checkpoint), a
label-permutation null, the prespecified ordering bracket (last checkpoint whose band includes zero,
then the first of two consecutive checkpoints whose band lies below zero), and the interval statistic
rho(J, dw) over step 8 -> 32 that Result 2 uses to show the ordering is being created there.

CPU only; reads results/per_pair_trajectories.npz written by analyze.py.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

from common import RESULTS

B_BOOT = 4000
B_PERM = 20000
RNG = np.random.default_rng(2600)


def corr_rows(A, b):
    """Pearson correlation of each row of A with vector b (inputs already rank-transformed)."""
    Ac = A - A.mean(axis=1, keepdims=True)
    bc = b - b.mean()
    return (Ac @ bc) / (np.linalg.norm(Ac, axis=1) * np.linalg.norm(bc) + 1e-300)


def paired_corr(A, Bm):
    """Row-wise correlation between two equally shaped rank matrices."""
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = Bm - Bm.mean(axis=1, keepdims=True)
    return ((Ac * Bc).sum(1) /
            (np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1) + 1e-300))


def analyse(J, W, steps):
    """rho trajectory, simultaneous band, permutation nulls and the ordering bracket for one subset."""
    n_ck, n = W.shape
    R = rankdata(W, axis=1)
    rJ = rankdata(J)
    rho = corr_rows(R, rJ)

    bi = RNG.integers(0, n, size=(B_BOOT, n))
    Jb = rankdata(J[bi], axis=1)
    rho_b = np.empty((B_BOOT, n_ck))
    for k in range(n_ck):
        rho_b[:, k] = paired_corr(rankdata(W[k][bi], axis=1), Jb)
    c = float(np.percentile(np.max(np.abs(rho_b - rho[None, :]), axis=1), 95))
    lo, hi = rho - c, rho + c

    perm = np.array([RNG.permutation(n) for _ in range(B_PERM)])
    rJc = rJ - rJ.mean()
    rho_p = np.empty((B_PERM, n_ck))
    for k in range(n_ck):
        Rc = R[k] - R[k].mean()
        rho_p[:, k] = (rJc[perm] @ Rc) / (np.linalg.norm(rJc) * np.linalg.norm(Rc))
    maxnull = np.max(np.abs(rho_p), axis=1)
    p = [(np.sum(np.abs(rho_p[:, k]) >= abs(rho[k])) + 1) / (B_PERM + 1) for k in range(n_ck)]
    p_fw = [(np.sum(maxnull >= abs(rho[k])) + 1) / (B_PERM + 1) for k in range(n_ck)]

    below = hi < 0
    br = None
    for k in range(n_ck - 1):
        if below[k] and below[k + 1]:
            prev = [j for j in range(k) if not below[j]]
            br = [int(steps[max(prev)]) if prev else None, int(steps[k])]
            break

    # interval statistic over step 8 -> 32 (indices 4 and 5), the interval that CREATES the ordering
    d = W[5] - W[4]
    rho_d = float(corr_rows(rankdata(d)[None, :], rJ)[0])
    rho_d_b = paired_corr(rankdata(d[bi], axis=1), Jb)
    rd_p = (rJc[perm] @ (rankdata(d) - rankdata(d).mean())) / (
        np.linalg.norm(rJc) * np.linalg.norm(rankdata(d) - rankdata(d).mean()))
    return dict(
        n=int(n), rho=[float(x) for x in rho],
        rho_sim_lo=[float(x) for x in lo], rho_sim_hi=[float(x) for x in hi],
        sim_c=c, p=[float(x) for x in p], p_fw=[float(x) for x in p_fw],
        null_pt95=[float(x) for x in np.percentile(np.abs(rho_p), 95, axis=0)],
        null_sim95=float(np.percentile(maxnull, 95)),
        bracket=br,
        rho_32=float(rho[5]), band_32=[float(lo[5]), float(hi[5])], p_fw_32=float(p_fw[5]),
        rho_dw_8_32=rho_d,
        rho_dw_8_32_ci=[float(np.percentile(rho_d_b, 2.5)), float(np.percentile(rho_d_b, 97.5))],
        rho_dw_8_32_p=float((np.sum(np.abs(rd_p) >= abs(rho_d)) + 1) / (B_PERM + 1)),
        median_dw_8_32=float(np.median(d)),
        J_lo=float(J.min()), J_hi=float(J.max()),
    )


def random_subset_control(J, W, k_ck, sizes, n_draw=4000):
    """What does dropping the SAME NUMBER of pairs at random do to rho at checkpoint k_ck?

    Removing a quintile removes both pairs and divergence range, and either alone attenuates a rank
    correlation. This control holds the sample size fixed and randomises WHICH pairs go, so a
    quintile-drop value outside the random envelope is about that quintile, not about n.
    """
    n = len(J)
    rJ, rw = rankdata(J), None
    out = {}
    for m in sizes:
        vals = np.empty(n_draw)
        for b in range(n_draw):
            s = RNG.choice(n, size=m, replace=False)
            rj, rw = rankdata(J[s]), rankdata(W[k_ck][s])
            a, c = rj - rj.mean(), rw - rw.mean()
            vals[b] = (a @ c) / (np.linalg.norm(a) * np.linalg.norm(c))
        out[str(m)] = dict(mean=float(vals.mean()),
                           pct=[float(np.percentile(vals, q)) for q in (2.5, 5, 50, 95, 97.5)],
                           values_sd=float(vals.std()))
        out[str(m)]["_vals"] = vals
    return out


def per_quintile(J, W, bins, steps):
    """Median w at step 32 and median change over step 8 -> 32 within each divergence quintile."""
    d = W[5] - W[4]
    rows = []
    for q in range(5):
        m = bins == q
        db = np.array([np.median(d[m][RNG.integers(0, m.sum(), m.sum())]) for _ in range(4000)])
        rows.append(dict(q=q + 1, n=int(m.sum()), J_med=float(np.median(J[m])),
                         w32=float(np.median(W[5][m])), w8=float(np.median(W[4][m])),
                         dw=float(np.median(d[m])),
                         dw_ci=[float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]))
    return rows


def main():
    z = np.load(os.path.join(RESULTS, "per_pair_trajectories.npz"))
    steps, J, W, bins = z["steps"], z["J"], z["W"], z["bins"]

    subsets = [("all", np.ones(len(J), bool))]
    for q in range(5):
        subsets.append((f"drop_Q{q + 1}", bins != q))
    subsets.append(("middle3", (bins != 0) & (bins != 4)))

    out = {"steps": [int(s) for s in steps], "n_boot": B_BOOT, "n_perm": B_PERM, "subsets": {}}
    for name, m in subsets:
        r = analyse(J[m], W[:, m], steps)
        out["subsets"][name] = r
        print(f"{name:>9} n={r['n']:>2} J[{r['J_lo']:.3f},{r['J_hi']:.3f}]  "
              f"rho32={r['rho_32']:+.3f} band[{r['band_32'][0]:+.3f},{r['band_32'][1]:+.3f}] "
              f"p_fw={r['p_fw_32']:.4f}  bracket={r['bracket']}  "
              f"rho(J,dw 8->32)={r['rho_dw_8_32']:+.3f} p={r['rho_dw_8_32_p']:.4f}")

    # size-matched random-drop control at step 32 (index 5)
    ctrl = random_subset_control(J, W, 5, sizes=sorted({out["subsets"][k]["n"] for k in
                                                       ("drop_Q1", "drop_Q5", "middle3")}))
    for name in ("drop_Q1", "drop_Q5", "middle3"):
        obs, m = out["subsets"][name]["rho_32"], out["subsets"][name]["n"]
        v = ctrl[str(m)]["_vals"]
        out["subsets"][name]["rand_drop_pct"] = float((np.sum(v <= obs) + 1) / (len(v) + 1))
        print(f"  {name}: rho32={obs:+.3f} vs random {m}-pair subsets "
              f"median {np.median(v):+.3f} [{np.percentile(v, 2.5):+.3f},"
              f"{np.percentile(v, 97.5):+.3f}]  pct={out['subsets'][name]['rand_drop_pct']:.3f}")
    out["random_drop_control"] = {k: {kk: vv for kk, vv in d.items() if kk != "_vals"}
                                  for k, d in ctrl.items()}
    out["per_quintile"] = per_quintile(J, W, bins, steps)
    for r in out["per_quintile"]:
        print(f"  Q{r['q']} (J~{r['J_med']:.2f}, n={r['n']}): w(8)={r['w8']:.4f} "
              f"w(32)={r['w32']:.4f}  dw={r['dw']:+.5f} "
              f"[{r['dw_ci'][0]:+.5f},{r['dw_ci'][1]:+.5f}]")

    json.dump(out, open(os.path.join(RESULTS, "quintile_loo.json"), "w"), indent=2)
    print("wrote results/quintile_loo.json")


if __name__ == "__main__":
    main()
