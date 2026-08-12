"""Is the width-specific component missing from the representations, or only out of reach at n = 80?

Every cell of patterns 41, 44, 45 and 46 -- six feature sets, three readouts -- sits inside its own
permutation null for the part of the crossing width that curve shape does not already explain. One
caveat now applies to all of them and to nothing else: each probe is fitted on 80 of the 123 endpoint
tokens. This run measures what that costs by moving the training size and holding everything else
fixed.

Same 123 tokens, same six feature sets, same four targets, same ceilings, same permutation seeds. Only
the number of training tokens changes: 30, 50, 80, 100 (test halves 93, 73, 43, 23). Two readouts --
linear ridge and RBF kernel ridge, the strict generalisation of it -- each tuned by the 5-fold CV
inside the training half that the incumbent probe uses.

  width residual climbing towards its null boundary with size -> the probe is underpowered and the
      curve says how many tokens a decisive test needs;
  flat at chance while the control targets climb -> more tokens of this kind will not expose it.

Both readouts are linear in the training targets once the features, split and hyper-parameter are
fixed, so each split's fits are built once as matrices and applied to all 4 targets x 51 target
vectors (real plus 50 permutations). Ridge uses the dual form X'(XX' + lam I)^-1, algebraically the
same estimator as embed_probe.probe but with 123x123 inverses instead of 2049x2049 -- the tie-check at
n_train = 80 confirms it reproduces the published rho AND the published null draws.

Costs 2 x 369 single-token forward passes to rebuild the six feature sets (cached in
results/curve_features.npz); the rest is CPU algebra.

Writes results/curve.json.
"""
import json
import os
from collections import Counter

import numpy as np
import torch
from scipy.stats import spearmanr

from common import RESULTS
from embed_probe import ALPHAS
from gpt2_shape_probe import N_PERM
from readout_probe import (SITES, TARGETS, features, fit_krr, folds_of, krr_maps, split_plan,
                           targets_and_ceilings)

torch.set_num_threads(2)

SIZES = [30, 50, 80, 100]
CACHE = f"{RESULTS}/curve_features.npz"


def ridge_maps(F, tr, te, ip):
    """Per split, the linear map from training targets to predictions, for every ridge penalty.

    embed_probe.probe solves (X'X + lam I)^-1 X'y with X standardised on the training half and a
    leading ones column; X'(XX' + lam I)^-1 y is the same vector of coefficients, so predictions are
    a fixed matrix times the centred training targets and every target reuses one inverse.
    """
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
    Xtr = np.hstack([np.ones((len(tr), 1)), (F[tr] - mu) / sd])
    Xte = np.hstack([np.ones((len(te), 1)), (F[te] - mu) / sd])
    Gtr, Gte = Xtr @ Xtr.T, Xte @ Xtr.T
    folds = folds_of(ip)
    maps = {}
    for lam in ALPHAS:
        inner = [(ite, itr, Gtr[np.ix_(ite, itr)]
                  @ np.linalg.inv(Gtr[np.ix_(itr, itr)] + lam * np.eye(len(itr))))
                 for ite, itr in folds]
        maps[float(lam)] = (inner, Gte @ np.linalg.inv(Gtr + lam * np.eye(len(tr))))
    return maps


def scores(plan, maps, y):
    rho, hp = [], []
    for (tr, te, _), m in zip(plan, maps):
        p, h = fit_krr(m, y, tr, te)
        rho.append(0.0 if np.std(p) == 0 else float(spearmanr(p, y[te])[0]))
        hp.append(h)
    return np.array(rho), hp


def run(plan, maps, y, ceil):
    """One probe: the 50 splits at this training size, then the same 50 permutations as a null."""
    rho, hp = scores(plan, maps, y)
    null = np.array([scores(plan, maps, np.random.default_rng(1000 + j).permutation(y))[0].mean()
                     for j in range(N_PERM)])
    reliability, rel_interval = ceil
    c = float(np.sqrt(max(reliability, 0)))
    return dict(
        reliability=reliability, reliability_ci=rel_interval, ceiling=c,
        rho_mean=float(rho.mean()), rho_sd=float(rho.std()),
        disattenuated=float(rho.mean() / c) if c > 0 else None,
        null_mean=float(null.mean()), null_sd=float(null.std()), null_max=float(null.max()),
        perm_p=float(((null >= rho.mean()).sum() + 1) / (len(null) + 1)),
        hyper=str(Counter(map(str, hp)).most_common(1)[0][0]),
        rho_per_split=[float(x) for x in rho], null_draws=[float(x) for x in null])


def cached_features(names):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    f = features(names)
    np.savez(CACHE, **f)
    return f


def main():
    names = json.load(open(f"{RESULTS}/edgedrift.json"))["rows"]["1.4b_block0"]["tokens"]
    y, ceil = targets_and_ceilings(names)
    feats = cached_features(names)
    assert set(feats) == set(SITES), sorted(feats)

    res = {"tokens": names, "n_tokens": len(names), "train_sizes": SIZES, "n_split": 50,
           "n_perm": N_PERM, "sites": SITES, "targets": TARGETS,
           "ridge_alphas": [float(a) for a in ALPHAS]}

    for site in SITES:
        F = feats[site]
        res[site] = {"n_features": int(F.shape[1])}
        print(f"\n--- {site} ({F.shape[1]} features) ---", flush=True)
        for size in SIZES:
            plan = split_plan(len(names), n_train=size)
            rm = [ridge_maps(F, *s) for s in plan]
            km = [krr_maps(F, *s) for s in plan]
            res[site][str(size)] = {"ridge": {}, "krr": {}}
            for name in TARGETS:
                for readout, maps in (("ridge", rm), ("krr", km)):
                    r = run(plan, maps, y[name], ceil[name])
                    res[site][str(size)][readout][name] = r
                    print(f"[{site:12s} n={size:3d} {readout:5s}] {name:17s} "
                          f"rho {r['rho_mean']:+.3f} +- {r['rho_sd']:.3f} (ceiling {r['ceiling']:.3f}"
                          f" -> {r['disattenuated']:+.3f}); null {r['null_mean']:+.3f} +- "
                          f"{r['null_sd']:.3f}, permutation p {r['perm_p']:.3f}; "
                          f"modal hyper {r['hyper']}", flush=True)

    # tie-check: at n_train = 80 the dual-form ridge column must reproduce the published probe, both
    # the real rho and all 50 null draws, or the learning curve is not anchored to the earlier runs.
    ref = json.load(open(f"{RESULTS}/early_shape.json"))
    deep = json.load(open(f"{RESULTS}/deep_shape.json"))
    ref.update({f"block{b}": deep[f"block{b}"] for b in (6, 12, 18)})
    res["tie_check_ridge_n80"] = {
        f"{s}:{name}": dict(
            now=res[s]["80"]["ridge"][name]["rho_mean"], published=ref[s][name]["rho_mean"],
            diff=res[s]["80"]["ridge"][name]["rho_mean"] - ref[s][name]["rho_mean"],
            null_diff=float(np.max(np.abs(np.array(res[s]["80"]["ridge"][name]["null_draws"])
                                          - np.array(ref[s][name]["null_draws"])))))
        for s in SITES for name in TARGETS}
    t = res["tie_check_ridge_n80"]
    print(f"\ntie-check at n_train=80 over {len(t)} probes: largest |rho diff| "
          f"{max(abs(d['diff']) for d in t.values()):.4f}, largest |null draw diff| "
          f"{max(d['null_diff'] for d in t.values()):.4f}")

    json.dump(res, open(f"{RESULTS}/curve.json", "w"), indent=1)
    print(f"\nwrote {RESULTS}/curve.json")


if __name__ == "__main__":
    main()
