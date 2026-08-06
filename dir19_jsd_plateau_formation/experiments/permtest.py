"""Permutation inference for the divergence-ordering claims (PLAN S5: 'endpoint-label permutation').

Everything so far rests on bootstrap intervals. A bootstrap asks how much the statistic would wobble
if we redrew the pairs; a permutation test asks the complementary question -- how large a |rho| this
design produces when corpus divergence carries no information at all about transition width. That
second question is the one that matters for the earliest checkpoints, where the whole ordered spread
in w is 0.006 and a reader is entitled to ask whether rank structure that fine is simply what any
labelling would give.

Three tests, all on already-saved curves (no GPU):

  T1  60-pair bank, cross-sectional rho_s.  Pairs are endpoint-disjoint, so the exact null relabels
      pairs: permute J across the 60 pairs.  ONE permutation is applied at every checkpoint, so the
      null trajectory keeps the same across-checkpoint dependence as the observed one and a
      max-|rho| envelope gives a family-wise (simultaneous) p-value over all 19 checkpoints.

  T2  60-pair bank, interval statistic rho(J, dw).  Same permutations, applied to the within-interval
      width change.  This is the test that carries the dissociation claim.

  T3  1,000-pair bank, ENDPOINT-LABEL permutation (QAP).  Its 1,000 pairs are built from only 123
      endpoint tokens, so pairs sharing a token are dependent and permuting pairs would be invalid.
      Instead we permute the 123 endpoint LABELS: pair (u,v) keeps its measured w, but its divergence
      is looked up as J[pi(u), pi(v)] in the frozen 123x123 held-out JSD matrix.  The measured widths,
      the pairing graph and the per-endpoint use counts are all untouched -- only the correspondence
      between token identity and divergence is broken.

Writes results/permutation.json.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata, spearmanr

from common import RESULTS

B = 20000
RNG = np.random.default_rng(11)
EXCLUDE_STEPS = {16}                      # mislabelled released revision; see ckpt_qc.py
LARGE_STEPS = [0, 8, 32, 64000, 143000]


def rho_from_ranks(ra, rb):
    """Spearman for pre-ranked vectors; rb may be a (B, n) stack of null ranks."""
    a = ra - ra.mean()
    b = rb - rb.mean(axis=-1, keepdims=True)
    return (b @ a) / np.sqrt((a * a).sum() * (b * b).sum(axis=-1))


def two_sided_p(obs, null):
    """Permutation p with the standard +1 correction, so p is never 0."""
    return (1.0 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1.0 + null.size)


def small_bank():
    """T1 and T2 on the frozen 60-pair bank."""
    z = np.load(os.path.join(RESULTS, "per_pair_trajectories.npz"))
    keep = np.array([s not in EXCLUDE_STEPS for s in z["steps"]])
    steps, W, J = z["steps"][keep], z["W"][keep], z["J"]
    n = J.size

    rJ = rankdata(J)
    perms = np.stack([RNG.permutation(n) for _ in range(B)])       # one relabelling per replicate
    rJ_null = rJ[perms]                                            # (B, 60)

    # T1: cross-sectional rho at every checkpoint, same permutation across the whole trajectory.
    obs, null = [], []
    for k in range(len(steps)):
        rw = rankdata(W[k])
        obs.append(rho_from_ranks(rw, rJ[None, :])[0])
        null.append(rho_from_ranks(rw, rJ_null))
    obs = np.array(obs)
    null = np.stack(null)                                          # (n_ckpt, B)
    maxnull = np.abs(null).max(axis=0)                             # family-wise over checkpoints
    t1 = {
        "steps": steps.tolist(),
        "rho": obs.tolist(),
        "p_pointwise": [two_sided_p(obs[k], null[k]) for k in range(len(steps))],
        "p_familywise": [float((1 + np.sum(maxnull >= abs(o) - 1e-12)) / (1 + B)) for o in obs],
        "null_env95": float(np.quantile(maxnull, 0.95)),           # simultaneous null envelope
        "null_pt95": np.quantile(np.abs(null), 0.95, axis=1).tolist(),
    }

    # T2: interval statistic rho(J, dw) for every adjacent pair of measured checkpoints.
    # step 0 -> 1 is a degenerate interval: the two checkpoints give bit-identical curves (one
    # warmup step at lr = 1.4e-7), so every dw is exactly 0 and no rank correlation exists.
    iobs, inull, ivalid = [], [], []
    for k in range(1, len(steps)):
        dw = W[k] - W[k - 1]
        if np.ptp(dw) == 0.0:
            iobs.append(np.nan)
            inull.append(np.zeros(B))
            ivalid.append(False)
            continue
        rd = rankdata(dw)
        iobs.append(rho_from_ranks(rd, rJ[None, :])[0])
        inull.append(rho_from_ranks(rd, rJ_null))
        ivalid.append(True)
    iobs = np.array(iobs)
    inull = np.stack(inull)
    imax = np.abs(inull[np.array(ivalid)]).max(axis=0)
    t2 = {
        "intervals": [f"{steps[k-1]}->{steps[k]}" for k in range(1, len(steps))],
        "median_dw": [float(np.median(W[k] - W[k - 1])) for k in range(1, len(steps))],
        "degenerate": [not v for v in ivalid],
        "rho": [None if np.isnan(o) else float(o) for o in iobs],
        "p_pointwise": [None if np.isnan(iobs[k]) else two_sided_p(iobs[k], inull[k])
                        for k in range(len(iobs))],
        "p_familywise": [None if np.isnan(o) else
                         float((1 + np.sum(imax >= abs(o) - 1e-12)) / (1 + B)) for o in iobs],
        "null_pt95": np.quantile(np.abs(inull), 0.95, axis=1).tolist(),
    }
    return t1, t2


def large_bank():
    """T3: endpoint-label (QAP) permutation on the frozen 1,000-pair bank."""
    man = json.load(open(os.path.join(RESULTS, "pair_manifest_large.json")))
    M = np.asarray(man["jsd_holdout_matrix"], dtype=float)         # (123, 123) held-out corpus JSD
    n_ep = M.shape[0]
    # pairs store ep_a/ep_b as top-256 vocabulary slots; the matrix is indexed by position in
    # eligible_endpoints, so map through that list.
    pos = {e: i for i, e in enumerate(man["eligible_endpoints"])}
    ea = np.array([pos[p["ep_a"]] for p in man["pairs"]])
    eb = np.array([pos[p["ep_b"]] for p in man["pairs"]])
    jb = np.array([p["jsd_B"] for p in man["pairs"]])
    jm = M[ea, eb]                                                 # same quantity, from the matrix
    lookup_err = float(np.abs(jm - jb).max())                      # matrix must reproduce jsd_B

    perms = np.stack([RNG.permutation(n_ep) for _ in range(B)])    # (B, 123) endpoint relabellings
    Jnull = M[perms[:, ea], perms[:, eb]]                          # (B, 1000) relabelled divergences
    rJ_null = rankdata(Jnull, axis=1)
    # score observed and null from the SAME source (the matrix), so the JSON's rounding cannot
    # differ between them; rho_check confirms this matches the jsd_B-based value we report.
    rJ = rankdata(jm)
    rJ_check = rankdata(jb)

    out = {"lookup_max_err": lookup_err, "n_endpoints": n_ep, "steps": [], "rho": [],
           "rho_from_jsdB": [], "p": [], "p_familywise": [], "null_pt95": []}
    obs_all, null_all = [], []
    for s in LARGE_STEPS:
        rows = json.load(open(os.path.join(RESULTS, f"assay_large_step{s}.json")))["rows"]
        w = np.array([r["w"] for r in rows])
        rw = rankdata(w)
        obs_all.append(rho_from_ranks(rw, rJ[None, :])[0])
        out["rho_from_jsdB"].append(float(rho_from_ranks(rw, rJ_check[None, :])[0]))
        null_all.append(rho_from_ranks(rw, rJ_null))
    obs_all = np.array(obs_all)
    null_all = np.stack(null_all)
    maxnull = np.abs(null_all).max(axis=0)
    for k, s in enumerate(LARGE_STEPS):
        out["steps"].append(s)
        out["rho"].append(float(obs_all[k]))
        out["p"].append(two_sided_p(obs_all[k], null_all[k]))
        out["p_familywise"].append(float((1 + np.sum(maxnull >= abs(obs_all[k]) - 1e-12)) / (1 + B)))
        out["null_pt95"].append(float(np.quantile(np.abs(null_all[k]), 0.95)))
    out["null_env95"] = float(np.quantile(maxnull, 0.95))
    return out


if __name__ == "__main__":
    t1, t2 = small_bank()
    t3 = large_bank()
    res = {"n_perm": B, "small_cross": t1, "small_interval": t2, "large_qap": t3}
    json.dump(res, open(os.path.join(RESULTS, "permutation.json"), "w"), indent=1)

    print(f"matrix/jsd_B lookup max err = {t3['lookup_max_err']:.2e}  (must be ~0)")
    print("\n60-pair cross-sectional (T1)   null |rho| simultaneous 95% envelope = "
          f"{t1['null_env95']:.3f}")
    for s, r, p, pf in zip(t1["steps"], t1["rho"], t1["p_pointwise"], t1["p_familywise"]):
        if s <= 128 or s in (1000, 2000, 143000):
            print(f"  step {s:>6}  rho={r:+.3f}  p={p:.4f}  p_fw={pf:.4f}")
    print("\n60-pair interval rho(J, dw) (T2)")
    for iv, dw, r, p, pf in zip(t2["intervals"], t2["median_dw"], t2["rho"],
                                t2["p_pointwise"], t2["p_familywise"]):
        if iv in ("8->32", "512->1000", "4000->8000", "32000->64000"):
            print(f"  {iv:>14}  med dw={dw:+.4f}  rho={r:+.3f}  p={p:.4f}  p_fw={pf:.4f}")
    print(f"\n1000-pair endpoint-label QAP (T3)  null envelope = {t3['null_env95']:.3f}")
    for s, r, p, pf, q in zip(t3["steps"], t3["rho"], t3["p"], t3["p_familywise"],
                              t3["null_pt95"]):
        print(f"  step {s:>6}  rho={r:+.3f}  p={p:.4f}  p_fw={pf:.4f}  null95|rho|={q:.3f}")
