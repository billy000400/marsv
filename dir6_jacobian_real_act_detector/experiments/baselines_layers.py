"""D6 — extend Phase-2 baselines to layers {3,6,9} + 3 HARDER norm-matched negatives.

Tests (a) whether the "local density (kNN) is the realness signal" story holds across depth,
and (b) whether harder manifold-aware negatives finally stress kNN.

New negative families (all norm-matched to their source real act):
- interp        : convex combo lambda*a+(1-lambda)*b of two distinct real eval acts, renormed
                  to a's norm. (Average of two reals -> off the (curved) manifold but near reals.)
- tangent_pert  : perturbation confined to the top-K global PCA subspace (on-manifold-ish; HARD).
- orth_pert     : perturbation confined to the complement of the top-K subspace (off-manifold; EASY).
Plus the original 4 (iso_gauss, cov_gauss, shuffle_coord, norm_pert) for continuity.

Pure numpy. Writes results/baseline_layers_metrics.csv.
"""
import os, json, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
SRC = os.path.join(ROOT, "..", "dir3_manifold", "data")
LAYERS = [3, 6, 9]
SEED = 0
N_TRAIN = 30_000
N_EVAL = 10_000
GAP = 5_000
KNN_SUB = 5_000
PCA_K = 64        # for pca_recon baseline
TAN_K = 50        # tangent subspace dim for tangent/orth perturbations
SHRINK = 0.05
PERT_REL = 0.5    # perturbation magnitude relative to per-act rms


def auroc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort"); s = scores[order]
    ranks_sorted = np.arange(1, len(s) + 1, dtype=np.float64); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0; i = j + 1
    ranks = np.empty(len(s)); ranks[order] = ranks_sorted
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def renorm_to(x, target_norm):
    return x * (target_norm / np.linalg.norm(x, axis=1, keepdims=True))


def run_layer(layer, rng, t0):
    a = np.load(os.path.join(SRC, f"acts_layer{layer}.npy"), mmap_mode="r")
    D = a.shape[1]
    train = np.asarray(a[:N_TRAIN], dtype=np.float32)
    es = N_TRAIN + GAP
    real = np.asarray(a[es:es + N_EVAL], dtype=np.float32)
    mu = train.mean(0); Xc = train - mu
    cov = (Xc.T @ Xc) / (len(train) - 1)
    cov_s = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + 1e-3 * np.eye(D, dtype=np.float32)
    cov_inv = np.linalg.inv(cov_s).astype(np.float32)
    L_chol = np.linalg.cholesky(cov_s).astype(np.float32)
    gstd = float(Xc.std())
    evals, evecs = np.linalg.eigh(cov)
    Vall = evecs[:, ::-1]                 # [D,D] desc
    V = Vall[:, :PCA_K]                   # pca_recon subspace
    Vt = Vall[:, :TAN_K]                  # tangent subspace
    coord_sd = train.std(0) + 1e-6
    onorm = np.linalg.norm(real, axis=1, keepdims=True)

    fams = {}
    fams["iso_gauss"] = (mu + gstd * rng.standard_normal((N_EVAL, D))).astype(np.float32)
    fams["cov_gauss"] = (mu + rng.standard_normal((N_EVAL, D)).astype(np.float32) @ L_chol.T).astype(np.float32)
    sc = real.copy()
    for i in range(N_EVAL):
        sc[i] = sc[i, rng.permutation(D)]
    fams["shuffle_coord"] = sc
    noise = rng.standard_normal((N_EVAL, D)).astype(np.float32)
    rms = onorm / np.sqrt(D)
    fams["norm_pert"] = renorm_to(real + PERT_REL * rms * noise, onorm).astype(np.float32)
    # interp between two distinct reals, renormed to a's norm
    b = real[rng.permutation(N_EVAL)]
    lam = rng.uniform(0.2, 0.8, (N_EVAL, 1)).astype(np.float32)
    fams["interp"] = renorm_to(lam * real + (1 - lam) * b, onorm).astype(np.float32)
    # tangent / orthogonal perturbations (mean-centered direction, projected)
    g = rng.standard_normal((N_EVAL, D)).astype(np.float32)
    tan = (g @ Vt) @ Vt.T                 # component in top-K subspace
    orth = g - tan                        # complement
    tan = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-6)
    orth = orth / (np.linalg.norm(orth, axis=1, keepdims=True) + 1e-6)
    mag = PERT_REL * onorm
    fams["tangent_pert"] = renorm_to(real + mag * tan, onorm).astype(np.float32)
    fams["orth_pert"] = renorm_to(real + mag * orth, onorm).astype(np.float32)

    knn_tr = train[rng.choice(len(train), KNN_SUB, replace=False)]
    knn_sqn = (knn_tr ** 2).sum(1)

    def score_all(X):
        Xc_ = X - mu; out = {}
        out["norm"] = np.linalg.norm(X, axis=1)
        out["mean_l2"] = np.linalg.norm(Xc_, axis=1)
        out["mahalanobis"] = np.einsum("ij,jk,ik->i", Xc_, cov_inv, Xc_)
        proj = Xc_ @ V; out["pca_recon"] = ((Xc_ - proj @ V.T) ** 2).sum(1)
        out["coord_quantile"] = np.abs(Xc_ / coord_sd).mean(1)
        xsq = (X ** 2).sum(1); knn = np.empty(len(X), np.float32); CH = 2000
        for i in range(0, len(X), CH):
            d2 = xsq[i:i + CH, None] + knn_sqn[None, :] - 2.0 * (X[i:i + CH] @ knn_tr.T)
            knn[i:i + CH] = np.sqrt(np.maximum(d2.min(1), 0))
        out["knn_distance"] = knn
        return out

    pos = score_all(real)
    BL = list(pos.keys())
    # norm orientation via pooled negs
    negsc = {f: score_all(fams[f]) for f in fams}
    pooled = np.concatenate([negsc[f]["norm"] for f in fams])
    lab = np.concatenate([np.zeros(N_EVAL), np.ones(len(pooled))])
    nsign = 1.0 if auroc(lab, np.concatenate([pos["norm"], pooled])) >= 0.5 else -1.0

    rows = []
    for f in fams:
        for bl in BL:
            ps = pos[bl] * (nsign if bl == "norm" else 1.0)
            ns = negsc[f][bl] * (nsign if bl == "norm" else 1.0)
            labels = np.concatenate([np.zeros(len(ps)), np.ones(len(ns))])
            rows.append({"layer": layer, "family": f, "baseline": bl,
                         "auroc": round(auroc(labels, np.concatenate([ps, ns])), 4)})
    print(f"[{time.time()-t0:.0f}s] layer{layer} done ({len(fams)} fams)", flush=True)
    return rows, BL, list(fams.keys())


def main():
    t0 = time.time(); rng = np.random.default_rng(SEED)
    allrows = []; BL = FAMS = None
    for L in LAYERS:
        r, BL, FAMS = run_layer(L, rng, t0)
        allrows += r
    with open(os.path.join(RES, "baseline_layers_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "family", "baseline", "auroc"])
        w.writeheader(); [w.writerow(r) for r in allrows]
    # print compact table per layer: kNN + best baseline per family
    for L in LAYERS:
        print(f"\n=== layer {L} (AUROC) ===", flush=True)
        for fam in FAMS:
            vals = {r["baseline"]: r["auroc"] for r in allrows if r["layer"] == L and r["family"] == fam}
            best = max(vals.items(), key=lambda kv: kv[1])
            print(f"  {fam:14s} knn={vals['knn_distance']:.3f}  best={best[1]:.3f}({best[0]})", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s -> results/baseline_layers_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
