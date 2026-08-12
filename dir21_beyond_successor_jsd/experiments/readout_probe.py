"""Is the width-specific component unreadable, or only unreadable BY A LINEAR PROBE?

Six feature sets have now been tested for the part of the crossing width that curve shape does not
already explain -- the static embedding row, the block-0 MLP output, and the residual stream after
blocks 0, 6, 12 and 18 -- and every one of them came out inside its own permutation null (patterns 41,
44, 45). Every statement of that negative carries the same caveat: it bounds ridge regression with
2,048 features and 80 training tokens, not every readout. This run tests the caveat.

Features, tokens, targets, ceilings, the 50 train/test splits and the 50 permutations are all held
fixed at exactly what those runs used. Only the readout changes:

  ridge   the incumbent linear probe (embed_probe.probe), refit here as a tie-check
  krr     RBF kernel ridge -- a nonlinear function of the same features; the kernel width (median
          heuristic x {0.25 .. 4}) and the penalty are tuned on the training half of each split by
          the same 5-fold CV that tunes the ridge penalty
  knn     k-nearest-neighbour regression on the same standardised features, k tuned the same way

Above the null at any of the six sites -> linearity was the binding constraint and the negative was a
statement about ridge. Inside it everywhere -> the negative stops depending on the word "linear".

Kernels and neighbour orderings depend only on the features and the split, never on the target, so
each site's maps are built once and applied to all four targets and all 51 target vectors (real plus
50 permutations); that is what makes 24 probes x 3 readouts x 51 fits affordable on CPU.

Costs 2 x 369 single-token forward passes to rebuild the six feature sets; no interpolation curves.

Writes results/readout.json.
"""
import json
from collections import Counter

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from common import RESULTS
from edgedrift_analysis import rel, widths
from embed_probe import ALPHAS, N_SPLIT
from gpt2_shape_probe import EDGE_CUT, N_PERM, N_TRAIN, rel_ci, rel_resid, resid
from mlp_read import probe

torch.set_num_threads(2)

TARGETS = ["shape", "width", "width_given_shape", "shape_given_width"]
SITES = ["embedding", "mlp_out", "resid_block0", "block6", "block12", "block18"]
GAMMA_MULT = [0.25, 0.5, 1.0, 2.0, 4.0]     # multipliers on the median-heuristic kernel width
KS = [1, 3, 5, 9, 15, 25, 39]               # neighbour counts, tuned per split


def split_plan(n, seed=1, n_split=N_SPLIT, n_train=N_TRAIN):
    """Exactly the split sequence embed_probe.probe draws: outer permutation, then inner fold order."""
    rng = np.random.default_rng(seed)
    plan = []
    for _ in range(n_split):
        perm = rng.permutation(n)
        tr, te = perm[:n_train], perm[n_train:]
        ip = rng.permutation(len(tr))
        plan.append((tr, te, ip))
    return plan


def sqdist(A, B):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T
    return np.maximum(d, 0)


def folds_of(ip, n_fold=5):
    return [(ip[f::n_fold], np.setdiff1d(ip, ip[f::n_fold])) for f in range(n_fold)]


def krr_maps(F, tr, te, ip):
    """Per split, the linear map from training targets to predictions, for every (kernel width, penalty).

    Kernel ridge predicts K_be (K_aa + lam I)^-1 (y_a - ybar) + ybar, which is linear in y once the
    kernel is fixed -- so the inverse is taken once here and reused for every target and permutation.
    """
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
    Xtr, Xte = (F[tr] - mu) / sd, (F[te] - mu) / sd
    Dtr, Dte = sqdist(Xtr, Xtr), sqdist(Xte, Xtr)
    med = float(np.median(Dtr[np.triu_indices(len(tr), 1)]))
    folds = folds_of(ip)
    maps = {}
    for g in GAMMA_MULT:
        Ktr, Kte = np.exp(-g * Dtr / med), np.exp(-g * Dte / med)
        for lam in ALPHAS:
            inner = [(ite, itr, Ktr[np.ix_(ite, itr)]
                      @ np.linalg.inv(Ktr[np.ix_(itr, itr)] + lam * np.eye(len(itr))))
                     for ite, itr in folds]
            maps[(g, float(lam))] = (inner, Kte @ np.linalg.inv(Ktr + lam * np.eye(len(tr))))
    return maps


def knn_maps(F, tr, te, ip):
    """Per split, the neighbour ordering of every held-out point over its candidate training points."""
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
    Xtr, Xte = (F[tr] - mu) / sd, (F[te] - mu) / sd
    Dtr, Dte = sqdist(Xtr, Xtr), sqdist(Xte, Xtr)
    inner = [(ite, itr[np.argsort(Dtr[np.ix_(ite, itr)], 1)]) for ite, itr in folds_of(ip)]
    return inner, np.argsort(Dte, 1)


def cv_score(y_tr, pred):
    return 1 - ((y_tr - pred) ** 2).sum() / ((y_tr - y_tr.mean()) ** 2).sum()


def fit_krr(maps, y, tr, te):
    ytr, ym = y[tr], y[tr].mean()
    best, best_key, best_fin = -np.inf, None, None
    for key, (inner, fin) in maps.items():
        pred = np.empty(len(tr))
        for ite, itr, A in inner:
            pred[ite] = A @ (ytr[itr] - ym) + ym
        s = cv_score(ytr, pred)
        if s > best:
            best, best_key, best_fin = s, key, fin
    return best_fin @ (ytr - ym) + ym, best_key


def fit_knn(maps, y, tr, te):
    inner, fin = maps
    ytr = y[tr]
    best, best_k = -np.inf, KS[0]
    for k in KS:
        pred = np.empty(len(tr))
        for ite, nbr in inner:
            pred[ite] = ytr[nbr[:, :k]].mean(1)
        s = cv_score(ytr, pred)
        if s > best:
            best, best_k = s, k
    return ytr[fin[:, :best_k]].mean(1), best_k


def scores(plan, maps, fit, y):
    """Test-half Spearman rho per split for one readout and one target vector."""
    rho, hp = [], []
    for (tr, te, _), m in zip(plan, maps):
        p, h = fit(m, y, tr, te)
        rho.append(0.0 if np.std(p) == 0 else float(spearmanr(p, y[te])[0]))
        hp.append(h)
    return np.array(rho), hp


def run(res, site, readout, name, plan, maps, fit, y, reliability, rel_interval):
    """One probe: the 50 shared splits, then the same 50 permutations of the target as a null."""
    rho, hp = scores(plan, maps, fit, y)
    null = np.array([scores(plan, maps, fit,
                            np.random.default_rng(1000 + j).permutation(y))[0].mean()
                     for j in range(N_PERM)])
    ceil = float(np.sqrt(max(reliability, 0)))
    res[site][readout][name] = dict(
        reliability=reliability, reliability_ci=rel_interval, ceiling=ceil,
        rho_mean=float(rho.mean()), rho_sd=float(rho.std()),
        disattenuated=float(rho.mean() / ceil) if ceil > 0 else None,
        null_mean=float(null.mean()), null_sd=float(null.std()), null_max=float(null.max()),
        perm_p=float(((null >= rho.mean()).sum() + 1) / (len(null) + 1)),
        hyper=str(Counter(map(str, hp)).most_common(1)[0][0]),
        rho_per_split=[float(x) for x in rho], null_draws=[float(x) for x in null])
    r = res[site][readout][name]
    print(f"[{site:12s} {readout:5s}] {name:17s} rho {rho.mean():+.3f} +- {rho.std():.3f} "
          f"(ceiling {ceil:.3f} -> {r['disattenuated']:+.3f}); null {null.mean():+.3f} +- "
          f"{null.std():.3f}, permutation p {r['perm_p']:.3f}; modal hyper {r['hyper']}", flush=True)


def run_ridge(res, site, name, F, y, reliability, rel_interval, published):
    """The incumbent linear probe, refit here on the same splits.

    Its 50-permutation null is the one published for this site and target in patterns 44/45. The null
    is a deterministic function of the features, the splits and the permutation seeds, all three of
    which this run holds fixed, so refitting it would return the same 50 numbers at 8 minutes per
    site. The refit rho is the check that the features and splits really are the same ones.
    """
    rho, _, lam = probe(F, y, np.random.default_rng(1), N_TRAIN)
    null = np.array(published["null_draws"])
    ceil = float(np.sqrt(max(reliability, 0)))
    res[site]["ridge"][name] = dict(
        reliability=reliability, reliability_ci=rel_interval, ceiling=ceil,
        rho_mean=float(rho.mean()), rho_sd=float(rho.std()),
        disattenuated=float(rho.mean() / ceil) if ceil > 0 else None,
        null_mean=float(null.mean()), null_sd=float(null.std()), null_max=float(null.max()),
        perm_p=float(((null >= rho.mean()).sum() + 1) / (len(null) + 1)),
        hyper=f"alpha={lam:.3g}",
        rho_per_split=[float(x) for x in rho], null_draws=[float(x) for x in null])
    r = res[site]["ridge"][name]
    print(f"[{site:12s} ridge] {name:17s} rho {rho.mean():+.3f} +- {rho.std():.3f} "
          f"(ceiling {ceil:.3f} -> {r['disattenuated']:+.3f}); null {null.mean():+.3f} +- "
          f"{null.std():.3f}, permutation p {r['perm_p']:.3f}; modal hyper {r['hyper']}", flush=True)


def paired(res, site, name, a, b):
    """Same 50 splits, same features, same target, two readouts: is `a` above `b` split by split?"""
    d = np.array(res[site][a][name]["rho_per_split"]) - np.array(res[site][b][name]["rho_per_split"])
    res.setdefault("paired_readouts", {})[f"{site}:{name}:{a}_minus_{b}"] = dict(
        mean=float(d.mean()), sd=float(d.std()), frac_a_higher=float((d > 0).mean()),
        p=float(wilcoxon(d).pvalue))
    print(f"paired [{site:12s}] {name:17s} {a} - {b}: {d.mean():+.3f} +- {d.std():.3f}, {a} higher "
          f"in {(d > 0).mean():.0%} of 50 splits (Wilcoxon p {wilcoxon(d).pvalue:.3g})", flush=True)


def features(names):
    """The six feature sets of patterns 44 and 45, rebuilt with their own code."""
    from early_shape_probe import features as early_features
    from deep_shape_probe import features as deep_features
    f = early_features(names)
    f.update({k: v for k, v in deep_features(names).items() if k != "block0"})
    return f


def targets_and_ceilings(names):
    row = json.load(open(f"{RESULTS}/edgedrift.json"))["rows"]["1.4b_block0"]
    raw = json.load(open(f"{RESULTS}/xwidth_1.4b.json"))["raw"]
    e_curves = np.array(row["edge_curves"], float).reshape(len(names), -1)
    w_pl = np.where(e_curves <= EDGE_CUT, widths(raw, names), np.nan)
    y = {"shape": np.median(e_curves, 1), "width": np.nanmedian(w_pl, 1)}
    y["width_given_shape"] = resid(y["width"], y["shape"])
    y["shape_given_width"] = resid(y["shape"], y["width"])
    n = len(names)
    ceil = {
        "shape": (rel(e_curves)[0], rel_ci(lambda i: rel(e_curves[i])[0], n)),
        "width": (rel(w_pl)[0], rel_ci(lambda i: rel(w_pl[i])[0], n)),
        "width_given_shape": (rel_resid(w_pl, e_curves)[0],
                              rel_ci(lambda i: rel_resid(w_pl[i], e_curves[i])[0], n)),
        "shape_given_width": (rel_resid(e_curves, w_pl)[0],
                              rel_ci(lambda i: rel_resid(e_curves[i], w_pl[i])[0], n)),
    }
    return y, ceil


def main():
    names = json.load(open(f"{RESULTS}/edgedrift.json"))["rows"]["1.4b_block0"]["tokens"]
    y, ceil = targets_and_ceilings(names)
    plan = split_plan(len(names))

    res = {"tokens": names, "n_tokens": len(names), "n_train": N_TRAIN, "n_split": N_SPLIT,
           "n_perm": N_PERM, "edge_cut": EDGE_CUT, "sites": SITES,
           "gamma_multipliers": GAMMA_MULT, "k_grid": KS,
           "ridge_alphas": [float(a) for a in ALPHAS]}

    ref = json.load(open(f"{RESULTS}/early_shape.json"))
    deep = json.load(open(f"{RESULTS}/deep_shape.json"))
    ref.update({f"block{b}": deep[f"block{b}"] for b in (6, 12, 18)})

    feats = features(names)
    assert set(feats) == set(SITES), sorted(feats)

    for site in SITES:
        F = feats[site]
        res[site] = {"n_features": int(F.shape[1]), "ridge": {}, "krr": {}, "knn": {}}
        print(f"\n--- {site} ({F.shape[1]} features) ---", flush=True)
        km = [krr_maps(F, *s) for s in plan]
        nm = [knn_maps(F, *s) for s in plan]
        for name in TARGETS:
            run_ridge(res, site, name, F, y[name], ceil[name][0], ceil[name][1], ref[site][name])
            run(res, site, "krr", name, plan, km, fit_krr, y[name], ceil[name][0], ceil[name][1])
            run(res, site, "knn", name, plan, nm, fit_knn, y[name], ceil[name][0], ceil[name][1])
        for name in TARGETS:
            paired(res, site, name, "krr", "ridge")
            paired(res, site, name, "knn", "ridge")

    # tie-check: the refit ridge column must reproduce patterns 44 and 45, or the features or the
    # splits differ and the three readouts are not being compared on the same footing.
    res["tie_check_ridge"] = {
        f"{s}:{name}": dict(now=res[s]["ridge"][name]["rho_mean"],
                            published=ref[s][name]["rho_mean"],
                            diff=res[s]["ridge"][name]["rho_mean"] - ref[s][name]["rho_mean"])
        for s in SITES for name in TARGETS}
    worst = max(abs(d["diff"]) for d in res["tie_check_ridge"].values())
    print(f"\ntie-check, ridge now vs patterns 44/45: largest absolute difference {worst:.4f} "
          f"over {len(res['tie_check_ridge'])} probes")

    json.dump(res, open(f"{RESULTS}/readout.json", "w"), indent=1)
    print(f"\nwrote {RESULTS}/readout.json")


if __name__ == "__main__":
    main()
