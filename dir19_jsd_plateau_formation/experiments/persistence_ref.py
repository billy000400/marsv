"""Does the "ranking becomes final" clock depend on WHICH checkpoint we call final?

Result 8 scores every checkpoint's per-pair width ranking against step 143000's and finds the
ranking locks in between step 64 and step 128. That bracket is defined relative to one arbitrary
reference: the last released checkpoint. If the model's ordering kept drifting late in training,
scoring against step 8000 or step 64000 instead could move the bracket, and the "third clock" would
be a property of the reference rather than of the model.

So we re-run the whole persistence analysis with five reference checkpoints (8000, 32000, 64000,
128000, 143000), using exactly the statistics and the onset rule of Result 8:

    pi_ref(s)      = Spearman(w_s, w_ref)
    pi_perp_ref(s) = partial Spearman(w_s, w_ref | J)

with a paired bootstrap over pairs and the same one-permutation-per-trajectory label null. The
onset bracket is (last checkpoint before, first of two consecutive checkpoints with family-wise
p < 0.05), searched only over checkpoints strictly before the reference.

CPU only; reads results/per_pair_trajectories.npz written by analyze.py.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

from common import RESULTS

B_BOOT = 4000
B_PERM = 20000
REF_STEPS = [8000, 32000, 64000, 128000, 143000]
RNG = np.random.default_rng(1900)


def corr_rows(A, b):
    """Pearson correlation of every row of A with vector b (both already rank-transformed)."""
    Ac = A - A.mean(axis=1, keepdims=True)
    bc = b - b.mean()
    return (Ac @ bc) / (np.linalg.norm(Ac, axis=1) * np.linalg.norm(bc) + 1e-300)


def partial_rows(A, b, z):
    """Partial correlation of every row of A with b, controlling z (all rank-transformed)."""
    Ac = A - A.mean(axis=1, keepdims=True)
    bc, zc = b - b.mean(), z - z.mean()
    u = zc / np.linalg.norm(zc)
    ea = Ac - np.outer(Ac @ u, u)
    eb = bc - (bc @ u) * u
    return (ea @ eb) / (np.linalg.norm(ea, axis=1) * np.linalg.norm(eb) + 1e-300)


def bracket(steps, p_fw, ref_k):
    """(last non-significant, first of two consecutive significant) among checkpoints < ref."""
    lim = ref_k                       # search s strictly before the reference
    for k in range(1, lim):
        if p_fw[k] < 0.05 and (k + 1 < lim and p_fw[k + 1] < 0.05):
            return [int(steps[k - 1]), int(steps[k])]
    return None


def main():
    z = np.load(os.path.join(RESULTS, "per_pair_trajectories.npz"))
    steps, J, W = z["steps"], z["J"], z["W"]
    n_ck, n = W.shape
    R = np.array([rankdata(w) for w in W])            # (n_ck, n) ranks of w at each checkpoint
    rJ = rankdata(J)

    boot_idx = RNG.integers(0, n, size=(B_BOOT, n))
    perm = np.array([RNG.permutation(n) for _ in range(B_PERM)])

    # bootstrap ranks, recomputed inside each resample (ties change the ranks)
    Rb = np.array([rankdata(W[k][boot_idx], axis=1) for k in range(n_ck)])   # (n_ck, B, n)
    Jb = rankdata(J[boot_idx], axis=1)

    out = {"steps": [int(s) for s in steps], "n_pairs": int(n),
           "n_boot": B_BOOT, "n_perm": B_PERM, "references": {}}

    for rs in REF_STEPS:
        ref = int(np.where(steps == rs)[0][0])
        pi = corr_rows(R, R[ref])
        pip = partial_rows(R, R[ref], rJ)

        pi_b = np.empty((B_BOOT, n_ck))
        pip_b = np.empty((B_BOOT, n_ck))
        for k in range(n_ck):
            A, Bref, Zb = Rb[k], Rb[ref], Jb
            Ac = A - A.mean(axis=1, keepdims=True)
            Bc = Bref - Bref.mean(axis=1, keepdims=True)
            Zc = Zb - Zb.mean(axis=1, keepdims=True)
            pi_b[:, k] = ((Ac * Bc).sum(1) /
                          (np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1) + 1e-300))
            u = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-300)
            ea = Ac - (Ac * u).sum(1, keepdims=True) * u
            eb = Bc - (Bc * u).sum(1, keepdims=True) * u
            pip_b[:, k] = ((ea * eb).sum(1) /
                           (np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1) + 1e-300))

        # label-permutation null: one permutation of the 60 pair labels per trajectory
        rref = R[ref] - R[ref].mean()
        u = (rJ - rJ.mean()) / np.linalg.norm(rJ - rJ.mean())
        ey = rref - (rref @ u) * u
        pi_p = np.empty((B_PERM, n_ck))
        for k in range(n_ck):
            Rp = (R[k] - R[k].mean())[perm]
            nk = np.linalg.norm(R[k] - R[k].mean())
            pi_p[:, k] = (Rp @ rref) / (nk * np.linalg.norm(rref))
        keep = np.arange(n_ck) != ref
        maxnull = np.nanmax(np.abs(pi_p[:, keep]), axis=1)
        p = np.array([(np.sum(np.abs(pi_p[:, k]) >= abs(pi[k])) + 1) / (B_PERM + 1)
                      for k in range(n_ck)])
        p_fw = np.array([(np.sum(maxnull >= abs(pi[k])) + 1) / (B_PERM + 1) for k in range(n_ck)])

        br = bracket(steps, p_fw, ref)
        k32 = int(np.where(steps == 32)[0][0])
        out["references"][str(rs)] = dict(
            ref_index=ref,
            pi=[float(x) for x in pi],
            pi_ci=[[float(np.percentile(pi_b[:, k], 2.5)),
                    float(np.percentile(pi_b[:, k], 97.5))] for k in range(n_ck)],
            pip=[float(x) for x in pip],
            pip_ci=[[float(np.percentile(pip_b[:, k], 2.5)),
                     float(np.percentile(pip_b[:, k], 97.5))] for k in range(n_ck)],
            p=[float(x) for x in p], p_fw=[float(x) for x in p_fw],
            null_pt95=[float(x) for x in np.percentile(np.abs(pi_p), 95, axis=0)],
            null_sim95=float(np.percentile(maxnull, 95)),
            bracket=br,
            pi_at_32=float(pi[k32]), pip_at_32=float(pip[k32]), p_at_32=float(p[k32]),
        )
        print(f"ref step {rs:>6}: bracket {br}  pi(32)={pi[k32]:+.3f} (p={p[k32]:.3f})  "
              f"pi_perp(32)={pip[k32]:+.3f}  pi(128)={pi[k32 + 2]:+.3f} "
              f"p_fw={p_fw[k32 + 2]:.4f}")

    json.dump(out, open(os.path.join(RESULTS, "persistence_ref.json"), "w"), indent=2)
    print("wrote results/persistence_ref.json")


if __name__ == "__main__":
    main()
