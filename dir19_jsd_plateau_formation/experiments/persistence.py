"""Is the step-32 micro-ordering the SAME pair-level ordering that later becomes the plateau?

The report establishes that corpus divergence J ranks pairs by width w from step 32, when the
curves are still straight lines. That leaves the obvious objection open: a rank correlation with J
does not say the step-32 ranking of the 60 pairs is the ranking the mature model ends up with. It
could be a J-shaped coincidence on a width spread of 0.006 that the model later re-derives from
scratch.

So we score PERSISTENCE: how well the per-pair width ranking at step s predicts the ranking at the
final checkpoint,

    pi(s) = Spearman(w_s, w_final),

and its J-free part, the partial Spearman controlling for corpus divergence,

    pi_partial(s) = partial Spearman(w_s, w_final | J),

which asks whether early width carries final-ordering information BEYOND what J already explains.
Inference matches the rest of the direction: a paired bootstrap over pairs (same resampled pairs at
every checkpoint, simultaneous band from the max deviation) and a label-permutation null in which
ONE permutation of the 60 pair labels is applied to the whole trajectory, so the family-wise
statement respects the across-checkpoint dependence.

Also writes the full checkpoint-by-checkpoint rank-correlation matrix of w, which shows when the
ordering locks in. CPU only; reads results/per_pair_trajectories.npz written by analyze.py.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata, spearmanr

from common import RESULTS

B_BOOT = 4000
B_PERM = 20000
RNG = np.random.default_rng(19)


def rank(x):
    return rankdata(x)


def partial_spearman(x, y, z):
    """Spearman of x and y with z removed: rank all three, then correlate OLS residuals."""
    rx, ry, rz = rank(x), rank(y), rank(z)
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def simultaneous(stat_boot, stat):
    dev = np.nanmax(np.abs(stat_boot - stat[None, :]), axis=1)
    c = float(np.nanpercentile(dev, 95))
    return stat - c, stat + c, c


def pct_ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def main():
    z = np.load(os.path.join(RESULTS, "per_pair_trajectories.npz"))
    steps, J, W = z["steps"], z["J"], z["W"]        # W is (n_ckpt, n_pairs)
    n_ck, n = W.shape
    ref = n_ck - 1                                  # final checkpoint, step 143000
    print(f"{n_ck} checkpoints x {n} pairs; reference = step {steps[ref]}")

    # --- observed statistics ---
    pi = np.array([spearmanr(W[k], W[ref]).statistic for k in range(n_ck)])
    pip = np.array([partial_spearman(W[k], W[ref], J) for k in range(n_ck)])

    # --- paired bootstrap over pairs (same resample at every checkpoint) ---
    idx = RNG.integers(0, n, size=(B_BOOT, n))
    pi_b = np.array([[spearmanr(W[k][i], W[ref][i]).statistic for k in range(n_ck)] for i in idx])
    pip_b = np.array([[partial_spearman(W[k][i], W[ref][i], J[i]) for k in range(n_ck)]
                      for i in idx])
    pi_lo, pi_hi, c_pi = simultaneous(pi_b, pi)
    pip_lo, pip_hi, c_pip = simultaneous(pip_b, pip)

    # --- label-permutation null: ONE permutation of the pair labels for the whole trajectory ---
    # w_s is relabelled, w_final is not, so the null destroys pair identity while keeping both
    # marginal width distributions exactly as observed.
    # Ranks are fixed under relabelling, so both statistics reduce to dot products of centred
    # rank vectors -- computed in closed form here, identical to the scipy calls above.
    perm = np.array([RNG.permutation(n) for _ in range(B_PERM)])
    R = np.array([rank(w) - rank(w).mean() for w in W])            # centred rank of w_s
    rj = rank(J) - rank(J).mean()
    u = rj / np.linalg.norm(rj)
    rref = R[ref]
    ey = rref - (rref @ u) * u                                     # w_final with J projected out
    n_ref, n_ey = np.linalg.norm(rref), np.linalg.norm(ey)
    pi_p = np.empty((B_PERM, n_ck))
    pip_p = np.empty((B_PERM, n_ck))
    for k in range(n_ck):
        Rp = R[k][perm]                                            # (B_PERM, n)
        nk = np.linalg.norm(R[k])
        pi_p[:, k] = (Rp @ rref) / (nk * n_ref)
        a = Rp @ u
        pip_p[:, k] = (Rp @ ey) / (np.sqrt(np.maximum(nk ** 2 - a ** 2, 1e-12)) * n_ey)
    # the reference checkpoint correlates with itself; it carries no information and is excluded
    # from every null summary and from the family-wise maximum.
    keep = np.arange(n_ck) != ref
    null_pt = np.nanpercentile(np.abs(pi_p), 95, axis=0)
    null_sim = float(np.nanpercentile(np.nanmax(np.abs(pi_p[:, keep]), axis=1), 95))
    p = np.array([(np.sum(np.abs(pi_p[:, k]) >= abs(pi[k])) + 1) / (B_PERM + 1)
                  for k in range(n_ck)])
    maxnull = np.nanmax(np.abs(pi_p[:, keep]), axis=1)
    p_fw = np.array([(np.sum(maxnull >= abs(pi[k])) + 1) / (B_PERM + 1) for k in range(n_ck)])

    null_pt_pp = np.nanpercentile(np.abs(pip_p), 95, axis=0)
    p_pp = np.array([(np.sum(np.abs(pip_p[:, k]) >= abs(pip[k])) + 1) / (B_PERM + 1)
                     for k in range(n_ck)])
    maxnull_pp = np.nanmax(np.abs(pip_p[:, keep]), axis=1)
    p_fw_pp = np.array([(np.sum(maxnull_pp >= abs(pip[k])) + 1) / (B_PERM + 1)
                        for k in range(n_ck)])

    # --- full checkpoint x checkpoint ranking-agreement matrix ---
    Msim = np.array([[spearmanr(W[a], W[b]).statistic for b in range(n_ck)] for a in range(n_ck)])

    # --- reliability of w, and the attenuation ceiling it puts on pi ---
    # At step 32 the whole spread of w is 0.006, so a low pi could just mean w is measured too
    # noisily there to correlate with anything. The three carrier sentences are three independent
    # measurements of the same pair, so their agreement estimates that noise: r_bar is the mean
    # pairwise Spearman between per-context widths, Spearman-Brown-corrected to the 3-context
    # median actually used. pi can then be at most sqrt(rel_s * rel_final) even if the underlying
    # rankings are identical, which is the ceiling reported alongside the observed value.
    import curve_metrics
    from analyze import GRID, load_checkpoints

    cps = load_checkpoints()
    rel, rbar = [], []
    for a in cps:
        cur = a["curves"]
        wc = np.array([[curve_metrics.metrics(cur[i, c], GRID)["w"] for i in range(cur.shape[0])]
                       for c in range(cur.shape[1])])            # (n_ctx, n_pairs)
        rs = [spearmanr(wc[i], wc[j]).statistic
              for i in range(len(wc)) for j in range(i + 1, len(wc))]
        r = float(np.mean(rs))
        rbar.append(r)
        rel.append(float(3 * r / (1 + 2 * r)) if r > 0 else 0.0)  # Spearman-Brown, k = 3
    rel = np.array(rel)
    ceiling = np.sqrt(np.clip(rel * rel[ref], 0, None))

    out = dict(
        steps=[int(s) for s in steps], n_pairs=int(n), reference_step=int(steps[ref]),
        n_boot=B_BOOT, n_perm=B_PERM,
        pi=[float(x) for x in pi], pi_ci=[pct_ci(pi_b[:, k]) for k in range(n_ck)],
        pi_sim_lo=[float(x) for x in pi_lo], pi_sim_hi=[float(x) for x in pi_hi], pi_sim_c=c_pi,
        pi_null_pt95=[float(x) for x in null_pt], pi_null_sim95=null_sim,
        pi_p=[float(x) for x in p], pi_p_fw=[float(x) for x in p_fw],
        pip=[float(x) for x in pip], pip_ci=[pct_ci(pip_b[:, k]) for k in range(n_ck)],
        pip_sim_lo=[float(x) for x in pip_lo], pip_sim_hi=[float(x) for x in pip_hi],
        pip_sim_c=c_pip, pip_null_pt95=[float(x) for x in null_pt_pp],
        pip_p=[float(x) for x in p_pp], pip_p_fw=[float(x) for x in p_fw_pp],
        matrix=[[float(v) for v in row] for row in Msim],
        ctx_rbar=[float(x) for x in rbar], reliability=[float(x) for x in rel],
        pi_ceiling=[float(x) for x in ceiling],
        pi_disattenuated=[float(pi[k] / ceiling[k]) if ceiling[k] > 0 else None
                          for k in range(n_ck)],
    )
    json.dump(out, open(os.path.join(RESULTS, "persistence.json"), "w"), indent=2)

    for k in range(n_ck):
        print(f"step {steps[k]:>6}  pi {pi[k]:+.3f} [{pi_lo[k]:+.3f},{pi_hi[k]:+.3f}] "
              f"p={p[k]:.4f} p_fw={p_fw[k]:.4f} | partial {pip[k]:+.3f} "
              f"p={p_pp[k]:.4f} p_fw={p_fw_pp[k]:.4f}")
    print(f"null: pointwise 95% |pi| ~ {null_pt[0]:.3f}, simultaneous {null_sim:.3f}")
    print("\nreliability of w (3 carrier sentences) and the attenuation ceiling on pi:")
    for k in range(n_ck):
        print(f"step {steps[k]:>6}  r_bar {rbar[k]:+.3f}  rel {rel[k]:.3f}  "
              f"ceiling {ceiling[k]:.3f}  observed pi {pi[k]:+.3f}")


if __name__ == "__main__":
    main()
