"""D6 Phase 4 — learned realness detectors with LEAVE-ONE-FAMILY-OUT (Gate 4).

Pure numpy (no sklearn). Tests whether a learned detector generalizes to an UNSEEN
corruption family or merely memorizes generator shortcuts.

Leakage controls:
- Real positives and the real activations used to DERIVE negatives are DISJOINT row blocks
  (shuffle_coord / norm_pert are tied to specific reals -> must not also be positives).
- Train/test real blocks are separated by a gap.
- Detector sees ONLY standardized raw activations (no layer/norm/source metadata).
- Standardization stats fit on TRAIN region only.

Models: logistic regression (full-batch GD + L2) and a 1-hidden-layer MLP (ReLU, minibatch).
Protocol per held-out family F: train on real + the OTHER 3 families, test on real + F.
Compare held-out-family AUROC to the kNN baseline bar (macro 0.913 from Phase 2).
"""
import os, json, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
SRC = os.path.join(ROOT, "..", "dir3_manifold", "data")
LAYER = 6
SEED = 0
rng = np.random.default_rng(SEED)
N = 10_000           # per block
SHRINK = 0.05
FAMS = ["iso_gauss", "cov_gauss", "shuffle_coord", "norm_pert"]


def auroc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]; ranks_sorted = np.arange(1, len(s) + 1, dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(len(s)); ranks[order] = ranks_sorted
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def make_negs(real_block, mu, L_chol, global_std, fit_std):
    """Build the 4 negative families derived from / matched to real_block (each [N,D])."""
    Nb, D = real_block.shape
    out = {}
    out["iso_gauss"] = (mu + global_std * rng.standard_normal((Nb, D))).astype(np.float32)
    z = rng.standard_normal((Nb, D)).astype(np.float32)
    out["cov_gauss"] = (mu + z @ L_chol.T).astype(np.float32)
    sc = real_block.copy()
    for i in range(Nb):
        sc[i] = sc[i, rng.permutation(D)]
    out["shuffle_coord"] = sc
    noise = rng.standard_normal((Nb, D)).astype(np.float32)
    pert = real_block + 0.5 * np.linalg.norm(real_block, axis=1, keepdims=True) / np.sqrt(D) * noise
    pert *= np.linalg.norm(real_block, axis=1, keepdims=True) / np.linalg.norm(pert, axis=1, keepdims=True)
    out["norm_pert"] = pert.astype(np.float32)
    return out


def logreg(X, y, Xte, l2=1e-2, lr=0.5, iters=400):
    n, d = X.shape
    w = np.zeros(d, dtype=np.float32); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        gw = X.T @ g / n + l2 * w
        gb = g.mean()
        w -= lr * gw; b -= lr * gb
    return Xte @ w + b  # logit score (higher=fake)


def mlp(X, y, Xte, h=64, l2=1e-3, lr=0.05, epochs=15, bs=512):
    n, d = X.shape
    W1 = (rng.standard_normal((d, h)) * np.sqrt(2.0 / d)).astype(np.float32)
    b1 = np.zeros(h, dtype=np.float32)
    W2 = (rng.standard_normal((h, 1)) * np.sqrt(2.0 / h)).astype(np.float32)
    b2 = 0.0
    idx = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, bs):
            bi = idx[s:s + bs]
            xb = X[bi]; yb = y[bi]
            a1 = xb @ W1 + b1
            r1 = np.maximum(a1, 0)
            z = (r1 @ W2).ravel() + b2
            p = 1.0 / (1.0 + np.exp(-z))
            g = (p - yb) / len(bi)
            gW2 = r1.T @ g[:, None] + l2 * W2
            gb2 = g.sum()
            gr1 = g[:, None] * W2.ravel()[None, :]
            ga1 = gr1 * (a1 > 0)
            gW1 = xb.T @ ga1 + l2 * W1
            gb1 = ga1.sum(0)
            W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    a1 = np.maximum(Xte @ W1 + b1, 0)
    return (a1 @ W2).ravel() + b2


def main():
    t0 = time.time()
    a = np.load(os.path.join(SRC, f"acts_layer{LAYER}.npy"), mmap_mode="r")
    D = a.shape[1]
    # disjoint row blocks
    train_pos = np.asarray(a[0:N], dtype=np.float32)
    train_negsrc = np.asarray(a[N:2 * N], dtype=np.float32)
    test_pos = np.asarray(a[3 * N:4 * N], dtype=np.float32)
    test_negsrc = np.asarray(a[4 * N:5 * N], dtype=np.float32)

    # fit stats on the train region (pos+negsrc)
    fitX = np.concatenate([train_pos, train_negsrc], 0)
    mu = fitX.mean(0)
    Xc = fitX - mu
    cov = (Xc.T @ Xc) / (len(fitX) - 1)
    cov_s = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + 1e-3 * np.eye(D, dtype=np.float32)
    L_chol = np.linalg.cholesky(cov_s).astype(np.float32)
    global_std = float(Xc.std())
    fit_std = fitX.std(0) + 1e-6

    train_negs = make_negs(train_negsrc, mu, L_chol, global_std, fit_std)
    test_negs = make_negs(test_negsrc, mu, L_chol, global_std, fit_std)
    print(f"[{time.time()-t0:.0f}s] built train/test negatives", flush=True)

    # standardize using train-positive stats only (detector sees only standardized acts)
    std_mu = train_pos.mean(0); std_sd = train_pos.std(0) + 1e-6
    def Z(x): return ((x - std_mu) / std_sd).astype(np.float32)
    Ztrain_pos, Ztest_pos = Z(train_pos), Z(test_pos)
    Ztrain_neg = {f: Z(train_negs[f]) for f in FAMS}
    Ztest_neg = {f: Z(test_negs[f]) for f in FAMS}

    results = []
    for held in FAMS:
        train_fams = [f for f in FAMS if f != held]
        Xneg = np.concatenate([Ztrain_neg[f] for f in train_fams], 0)
        # balance: replicate positives to match neg count
        reps = int(np.ceil(len(Xneg) / len(Ztrain_pos)))
        Xpos = np.tile(Ztrain_pos, (reps, 1))[:len(Xneg)]
        Xtr = np.concatenate([Xpos, Xneg], 0)
        ytr = np.concatenate([np.zeros(len(Xpos)), np.ones(len(Xneg))]).astype(np.float32)
        perm = rng.permutation(len(Xtr)); Xtr, ytr = Xtr[perm], ytr[perm]

        # held-out test: real_test vs held-family_test
        Xte = np.concatenate([Ztest_pos, Ztest_neg[held]], 0)
        yte = np.concatenate([np.zeros(len(Ztest_pos)), np.ones(len(Ztest_neg[held]))])

        s_lr = logreg(Xtr, ytr, Xte)
        s_mlp = mlp(Xtr, ytr, Xte)
        au_lr = auroc(yte, s_lr); au_mlp = auroc(yte, s_mlp)
        results.append({"held_out_family": held, "logreg_auroc": round(au_lr, 4),
                        "mlp_auroc": round(au_mlp, 4)})
        print(f"[{time.time()-t0:.0f}s] held={held:14s} logreg={au_lr:.3f} mlp={au_mlp:.3f}", flush=True)

    # held-IN reference: train on all 4, test on all 4 (pooled)
    Xneg_all = np.concatenate([Ztrain_neg[f] for f in FAMS], 0)
    reps = int(np.ceil(len(Xneg_all) / len(Ztrain_pos)))
    Xpos = np.tile(Ztrain_pos, (reps, 1))[:len(Xneg_all)]
    Xtr = np.concatenate([Xpos, Xneg_all], 0)
    ytr = np.concatenate([np.zeros(len(Xpos)), np.ones(len(Xneg_all))]).astype(np.float32)
    perm = rng.permutation(len(Xtr)); Xtr, ytr = Xtr[perm], ytr[perm]
    Xte_neg = np.concatenate([Ztest_neg[f] for f in FAMS], 0)
    Xte = np.concatenate([Ztest_pos, Xte_neg], 0)
    yte = np.concatenate([np.zeros(len(Ztest_pos)), np.ones(len(Xte_neg))])
    s_lr = logreg(Xtr, ytr, Xte); s_mlp = mlp(Xtr, ytr, Xte)
    heldin = {"logreg_auroc": round(auroc(yte, s_lr), 4), "mlp_auroc": round(auroc(yte, s_mlp), 4)}
    print(f"[{time.time()-t0:.0f}s] held-IN (all4): logreg={heldin['logreg_auroc']} mlp={heldin['mlp_auroc']}", flush=True)

    macro_lr = round(float(np.mean([r["logreg_auroc"] for r in results])), 4)
    macro_mlp = round(float(np.mean([r["mlp_auroc"] for r in results])), 4)

    with open(os.path.join(RES, "detector_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["held_out_family", "logreg_auroc", "mlp_auroc"])
        w.writeheader()
        for r in results:
            w.writerow(r)
        w.writerow({"held_out_family": "LOFO_MACRO", "logreg_auroc": macro_lr, "mlp_auroc": macro_mlp})
        w.writerow({"held_out_family": "HELD_IN_all4", **heldin})
    with open(os.path.join(RES, "detector_summary.json"), "w") as f:
        json.dump({"layer": LAYER, "lofo": results, "lofo_macro": {"logreg": macro_lr, "mlp": macro_mlp},
                   "held_in_all4": heldin, "knn_baseline_macro": 0.913,
                   "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"DONE LOFO macro logreg={macro_lr} mlp={macro_mlp} (kNN bar 0.913)", flush=True)


if __name__ == "__main__":
    main()
