"""D6 MVP — Phases 0/1/2 in pure numpy (no sklearn/scipy/transformers needed).

Reuses cached REAL GPT-2 residual activations from dir3_manifold:
  ../dir3_manifold/data/acts_layer{3,6,9}.npy  -> [200000, 768] float16, all token
  positions pooled, RAW (not centered/normalized).  resid_post = hidden_states[L+1].

Builds a leakage-resistant real-vs-synthetic benchmark and runs strong statistical
baselines. Negative families (minimum viable suite from PLAN fallback):
  - iso_gauss        : isotropic Gaussian (mean + global_std * N(0,1))
  - cov_gauss        : full mean/cov-matched Gaussian (shrinkage cov, cholesky)
  - shuffle_coord    : real test act with its 768 coords permuted (norm preserved exactly)
  - norm_pert        : real test act + Gaussian noise, rescaled to original norm (norm preserved)

Baselines (fit on TRAIN real positives only): norm, mean_l2, mahalanobis(shrinkage),
pca_recon, knn_distance(1-NN to train subsample), coord_quantile.

Leakage control: split the 200k rows into contiguous TRAIN/TEST blocks (with a gap) so
same-document token rows stay within one split. Negatives derived from TEST positives only.
All metrics AUROC + AUPRC, per family + macro. Self-implemented (no sklearn).
"""
import os, json, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
os.makedirs(os.path.join(RES, "acts"), exist_ok=True)
SRC = os.path.join(ROOT, "..", "dir3_manifold", "data")

SEED = 0
rng = np.random.default_rng(SEED)
LAYER = 6
N_TRAIN = 50_000      # train real positives (fit baselines)
N_EVAL = 10_000       # test real positives, and per negative family
KNN_TRAIN_SUB = 5_000 # train subsample for kNN
PCA_K = 64            # PCA components for recon-error baseline
SHRINK = 0.05         # shrinkage toward diagonal for covariance


# ---------- metrics (no sklearn) ----------
def auroc(labels, scores):
    """AUROC with label=1 = anomaly/fake. Rank-sum (Mann-Whitney)."""
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = ranks_sorted
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def auprc(labels, scores):
    """Average precision, label=1 = anomaly/fake."""
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    # integrate precision over recall steps
    rprev = 0.0; ap = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - rprev); rprev = r
    return float(ap)


# ---------- load + split ----------
def load_real(layer):
    path = os.path.join(SRC, f"acts_layer{layer}.npy")
    a = np.load(path, mmap_mode="r")
    return a  # [200000,768] float16 memmap


def main():
    t0 = time.time()
    a = load_real(LAYER)
    N, D = a.shape
    # contiguous split with a gap to reduce document leakage across the boundary
    GAP = 5_000
    train = np.asarray(a[:N_TRAIN], dtype=np.float32)
    eval_start = N_TRAIN + GAP
    real_eval = np.asarray(a[eval_start:eval_start + N_EVAL], dtype=np.float32)
    print(f"[{time.time()-t0:.0f}s] loaded layer{LAYER} train={train.shape} eval={real_eval.shape}", flush=True)

    # fit statistics on TRAIN
    mu = train.mean(0)
    Xc = train - mu
    cov = (Xc.T @ Xc) / (len(train) - 1)
    diag = np.diag(np.diag(cov))
    cov_s = (1 - SHRINK) * cov + SHRINK * diag
    # add tiny ridge for invertibility
    cov_s += 1e-3 * np.eye(D, dtype=np.float32)
    cov_inv = np.linalg.inv(cov_s).astype(np.float32)
    L_chol = np.linalg.cholesky(cov_s).astype(np.float32)
    global_std = Xc.std()
    train_norm_mean = np.linalg.norm(train, axis=1).mean()
    # PCA via eigdecomp of cov
    evals, evecs = np.linalg.eigh(cov)  # ascending
    V = evecs[:, ::-1][:, :PCA_K]       # top-K components [D,K]
    # coord quantile reference: per-coord mean/std from train
    coord_mu = mu; coord_sd = train.std(0) + 1e-6

    # ---------- negatives, all matched to N_EVAL, derived from real_eval where applicable ----------
    fams = {}
    fams["iso_gauss"] = (mu + global_std * rng.standard_normal((N_EVAL, D))).astype(np.float32)
    z = rng.standard_normal((N_EVAL, D)).astype(np.float32)
    fams["cov_gauss"] = (mu + z @ L_chol.T).astype(np.float32)
    # shuffle coords of each real eval act (norm preserved exactly)
    sc = real_eval.copy()
    for i in range(N_EVAL):
        sc[i] = sc[i, rng.permutation(D)]
    fams["shuffle_coord"] = sc
    # norm-preserving perturbation: add noise then rescale to original norm
    noise = rng.standard_normal((N_EVAL, D)).astype(np.float32)
    perturbed = real_eval + 0.5 * np.linalg.norm(real_eval, axis=1, keepdims=True) / np.sqrt(D) * noise
    orig_norm = np.linalg.norm(real_eval, axis=1, keepdims=True)
    perturbed *= orig_norm / np.linalg.norm(perturbed, axis=1, keepdims=True)
    fams["norm_pert"] = perturbed.astype(np.float32)

    print(f"[{time.time()-t0:.0f}s] built {len(fams)} negative families", flush=True)

    # ---------- baseline scorers (anomaly = higher means MORE fake) ----------
    knn_train = train[rng.choice(len(train), KNN_TRAIN_SUB, replace=False)]
    knn_train_sqn = (knn_train ** 2).sum(1)

    def score_all(X):
        Xc_ = X - mu
        out = {}
        out["norm"] = np.linalg.norm(X, axis=1)                       # orient later
        out["mean_l2"] = np.linalg.norm(Xc_, axis=1)                  # higher=fake
        out["mahalanobis"] = np.einsum("ij,jk,ik->i", Xc_, cov_inv, Xc_)  # higher=fake
        proj = Xc_ @ V
        recon = proj @ V.T
        out["pca_recon"] = ((Xc_ - recon) ** 2).sum(1)               # higher=fake
        # coord quantile: mean squared z over coords beyond the bulk (use abs-z mean)
        zc = (X - coord_mu) / coord_sd
        out["coord_quantile"] = np.abs(zc).mean(1)                   # higher=fake
        # 1-NN distance to train subsample, chunked
        knn = np.empty(len(X), dtype=np.float32)
        xsq = (X ** 2).sum(1)
        CH = 2000
        for i in range(0, len(X), CH):
            xb = X[i:i + CH]
            d2 = xsq[i:i + CH, None] + knn_train_sqn[None, :] - 2.0 * (xb @ knn_train.T)
            knn[i:i + CH] = np.sqrt(np.maximum(d2.min(1), 0))
        out["knn_distance"] = knn                                    # higher=fake
        return out

    pos_scores = score_all(real_eval)
    print(f"[{time.time()-t0:.0f}s] scored positives", flush=True)
    BASELINES = list(pos_scores.keys())

    # norm orientation: pick sign so AUROC>=0.5 against the easiest (iso_gauss) family
    rows = []
    family_scores_cache = {}
    for fam, Xn in fams.items():
        family_scores_cache[fam] = score_all(Xn)
        print(f"[{time.time()-t0:.0f}s] scored {fam}", flush=True)

    # determine norm sign once using all families pooled
    pooled_neg_norm = np.concatenate([family_scores_cache[f]["norm"] for f in fams])
    lab = np.concatenate([np.zeros(N_EVAL), np.ones(len(pooled_neg_norm))])
    sc_pooled = np.concatenate([pos_scores["norm"], pooled_neg_norm])
    norm_sign = 1.0 if auroc(lab, sc_pooled) >= 0.5 else -1.0

    metrics = []
    for fam in fams:
        for b in BASELINES:
            ps = pos_scores[b]; ns = family_scores_cache[fam][b]
            if b == "norm":
                ps = norm_sign * ps; ns = norm_sign * ns
            labels = np.concatenate([np.zeros(len(ps)), np.ones(len(ns))])  # 1=fake
            scores = np.concatenate([ps, ns])
            metrics.append({
                "family": fam, "baseline": b,
                "auroc": round(auroc(labels, scores), 4),
                "auprc": round(auprc(labels, scores), 4),
            })

    # macro average per baseline
    for b in BASELINES:
        aus = [m["auroc"] for m in metrics if m["baseline"] == b]
        aps = [m["auprc"] for m in metrics if m["baseline"] == b]
        metrics.append({"family": "MACRO", "baseline": b,
                        "auroc": round(float(np.mean(aus)), 4),
                        "auprc": round(float(np.mean(aps)), 4)})

    # write metrics csv
    import csv
    mp = os.path.join(RES, "baseline_metrics.csv")
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "baseline", "auroc", "auprc"])
        w.writeheader(); [w.writerow(r) for r in metrics]

    # dataset / negative summary
    fam_summary = []
    for fam, Xn in fams.items():
        fam_summary.append({
            "family": fam, "n": int(len(Xn)),
            "norm_mean": round(float(np.linalg.norm(Xn, axis=1).mean()), 3),
            "norm_std": round(float(np.linalg.norm(Xn, axis=1).std()), 3),
        })
    fam_summary.append({"family": "real_eval", "n": int(N_EVAL),
                        "norm_mean": round(float(np.linalg.norm(real_eval, axis=1).mean()), 3),
                        "norm_std": round(float(np.linalg.norm(real_eval, axis=1).std()), 3)})
    with open(os.path.join(RES, "negative_family_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "n", "norm_mean", "norm_std"])
        w.writeheader(); [w.writerow(r) for r in fam_summary]

    summary = {
        "layer": LAYER, "d_model": D, "n_total_rows": int(N),
        "n_train": N_TRAIN, "n_eval_per_class": N_EVAL, "gap": GAP,
        "families": list(fams.keys()), "baselines": BASELINES,
        "pca_k": PCA_K, "shrink": SHRINK, "seed": SEED,
        "train_norm_mean": round(float(train_norm_mean), 3),
        "real_eval_norm_mean": round(float(np.linalg.norm(real_eval, axis=1).mean()), 3),
        "norm_sign": norm_sign,
        "elapsed_s": round(time.time() - t0, 1),
        "note": "real positives reused from dir3_manifold acts_layer6.npy; pos0 sink tokens (norm~3000) NOT filtered",
    }
    with open(os.path.join(RES, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # audit.json (Phase 0)
    audit = {
        "direction": 6, "phase": "P0-audit + P1 + P2 (MVP)",
        "env": {"numpy": np.__version__, "transformers": False, "sklearn": False,
                "scipy": False, "torch_available": True, "gpu": "A10 sm_86"},
        "reused_real_acts": "../dir3_manifold/data/acts_layer{3,6,9}.npy [200000,768] f16",
        "no_prior_d6_results": True,
        "reusable_scripts": ["../dir9_ood/experiments/plateau_score.py (model+KL/Jacobian patterns)",
                              "../dir3_manifold/experiments/collect_acts.py (act caching)"],
        "caveat_dir9": "scalar Jacobian/plateau OOD signals were weak; don't rely on jac-norm alone",
        "caveat_pos0_sink": "row0-type sink tokens have norm~3000; norm shortcut + Mahalanobis artifact risk",
    }
    with open(os.path.join(RES, "audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    print("=== MACRO AUROC by baseline ===", flush=True)
    for m in metrics:
        if m["family"] == "MACRO":
            print(f"  {m['baseline']:16s} auroc={m['auroc']:.3f} auprc={m['auprc']:.3f}", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s -> {mp}", flush=True)


if __name__ == "__main__":
    main()
