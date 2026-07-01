"""D6 Phase 2c — token-position (sink-token) confound control + document-level split.

The main benchmark (mvp_benchmark.py) reuses acts_layer6.npy [200000,768] with ALL token
positions pooled and pos-0 "attention-sink" tokens (norm ~3000) NOT filtered — RESULTS.md flags
this as the key open caveat (norm / Mahalanobis confound). dir3 also cached a POSITION-INDEXED
sample:
  ../dir3_manifold/data/acts_layer6_pos.npy    [80000,768] f16  (resid_post L6, FineWeb)
  ../dir3_manifold/data/acts_layer6_posidx.npy [80000] int16    (absolute token position)
The layout is document-major: each document is a contiguous run of positions 0,1,2,... So we can
(1) do a genuine DOCUMENT-LEVEL train/eval split (stronger than the contiguous+gap heuristic), and
(2) filter pos-0 sink tokens.

We re-run the exact Phase-2 baseline benchmark (same 4 negative families, same 6 baselines) in two
conditions and compare, to test whether the headline discrimination story ("norm is a shortcut,
kNN best, no single baseline dominates") is robust once the sink confound is removed:
  - WITHSINK : reals = all positions (incl. pos-0 sink), doc-level split
  - NONSINK  : reals = pos>=1 only (sink removed), doc-level split
"""
import os, json, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "plots")
os.makedirs(RES, exist_ok=True); os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(ROOT, "..", "dir3_manifold", "data")

SEED = 0
rng = np.random.default_rng(SEED)
N_EVAL = 12_000       # eval real positives, and per negative family (capped for kNN cost)
N_TRAIN = 40_000      # train real positives (fit baselines)
KNN_TRAIN_SUB = 5_000
PCA_K = 64
SHRINK = 0.05


def auroc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def build_and_score(real_train, real_eval, D, tag, t0):
    """Fit baselines on real_train, build 4 negative families from real_eval, return AUROC dict."""
    mu = real_train.mean(0)
    Xc = real_train - mu
    cov = (Xc.T @ Xc) / (len(real_train) - 1)
    diag = np.diag(np.diag(cov))
    cov_s = (1 - SHRINK) * cov + SHRINK * diag + 1e-3 * np.eye(D, dtype=np.float32)
    cov_inv = np.linalg.inv(cov_s).astype(np.float32)
    L_chol = np.linalg.cholesky(cov_s).astype(np.float32)
    global_std = Xc.std()
    evals, evecs = np.linalg.eigh(cov)
    V = evecs[:, ::-1][:, :PCA_K]
    coord_mu = mu; coord_sd = real_train.std(0) + 1e-6

    # negative families (identical construction to mvp_benchmark)
    fams = {}
    fams["iso_gauss"] = (mu + global_std * rng.standard_normal((N_EVAL, D))).astype(np.float32)
    z = rng.standard_normal((N_EVAL, D)).astype(np.float32)
    fams["cov_gauss"] = (mu + z @ L_chol.T).astype(np.float32)
    sc = real_eval.copy()
    for i in range(N_EVAL):
        sc[i] = sc[i, rng.permutation(D)]
    fams["shuffle_coord"] = sc
    noise = rng.standard_normal((N_EVAL, D)).astype(np.float32)
    perturbed = real_eval + 0.5 * np.linalg.norm(real_eval, axis=1, keepdims=True) / np.sqrt(D) * noise
    orig_norm = np.linalg.norm(real_eval, axis=1, keepdims=True)
    perturbed *= orig_norm / np.linalg.norm(perturbed, axis=1, keepdims=True)
    fams["norm_pert"] = perturbed.astype(np.float32)

    knn_train = real_train[rng.choice(len(real_train), KNN_TRAIN_SUB, replace=False)]
    knn_train_sqn = (knn_train ** 2).sum(1)

    def score_all(X):
        Xc_ = X - mu
        out = {}
        out["norm"] = np.linalg.norm(X, axis=1)
        out["mean_l2"] = np.linalg.norm(Xc_, axis=1)
        out["mahalanobis"] = np.einsum("ij,jk,ik->i", Xc_, cov_inv, Xc_)
        proj = Xc_ @ V; recon = proj @ V.T
        out["pca_recon"] = ((Xc_ - recon) ** 2).sum(1)
        zc = (X - coord_mu) / coord_sd
        out["coord_quantile"] = np.abs(zc).mean(1)
        knn = np.empty(len(X), dtype=np.float32); xsq = (X ** 2).sum(1)
        for i in range(0, len(X), 2000):
            xb = X[i:i + 2000]
            d2 = xsq[i:i + 2000, None] + knn_train_sqn[None, :] - 2.0 * (xb @ knn_train.T)
            knn[i:i + 2000] = np.sqrt(np.maximum(d2.min(1), 0))
        out["knn_distance"] = knn
        return out

    pos_scores = score_all(real_eval)
    BASELINES = list(pos_scores.keys())
    fam_scores = {f: score_all(Xn) for f, Xn in fams.items()}
    print(f"[{time.time()-t0:.0f}s] {tag}: scored eval + {len(fams)} families", flush=True)

    # norm orientation from pooled negatives
    pooled = np.concatenate([fam_scores[f]["norm"] for f in fams])
    lab = np.concatenate([np.zeros(N_EVAL), np.ones(len(pooled))])
    norm_sign = 1.0 if auroc(lab, np.concatenate([pos_scores["norm"], pooled])) >= 0.5 else -1.0

    out_rows = []
    macro = {b: [] for b in BASELINES}
    for fam in fams:
        for b in BASELINES:
            ps = pos_scores[b]; ns = fam_scores[fam][b]
            if b == "norm":
                ps = norm_sign * ps; ns = norm_sign * ns
            labels = np.concatenate([np.zeros(len(ps)), np.ones(len(ns))])
            a = auroc(labels, np.concatenate([ps, ns]))
            out_rows.append({"condition": tag, "family": fam, "baseline": b, "auroc": round(a, 4)})
            macro[b].append(a)
    for b in BASELINES:
        out_rows.append({"condition": tag, "family": "MACRO", "baseline": b,
                         "auroc": round(float(np.mean(macro[b])), 4)})
    return out_rows, {b: round(float(np.mean(macro[b])), 4) for b in BASELINES}, \
        {"real_eval_norm_mean": round(float(np.linalg.norm(real_eval, axis=1).mean()), 2),
         "cov_gauss_norm_mean": round(float(np.linalg.norm(fams["cov_gauss"], axis=1).mean()), 2)}


def main():
    t0 = time.time()
    acts = np.load(os.path.join(SRC, "acts_layer6_pos.npy"), mmap_mode="r")
    posidx = np.load(os.path.join(SRC, "acts_layer6_posidx.npy"))
    N, D = acts.shape
    posidx = posidx[:N]
    # document boundaries: a new doc starts at each pos-0
    doc_starts = np.where(posidx == 0)[0]
    n_docs = len(doc_starts)
    doc_id = np.zeros(N, dtype=np.int32)
    doc_id[doc_starts] = 1
    doc_id = np.cumsum(doc_id) - 1               # 0..n_docs-1, contiguous per document
    # document-level split: even docs -> train, odd docs -> eval (no token shared across split)
    is_train_doc = (np.arange(n_docs) % 2 == 0)
    train_mask = is_train_doc[doc_id]
    eval_mask = ~train_mask
    sink_mask = (posidx == 0)
    print(f"[{time.time()-t0:.0f}s] N={N} docs={n_docs} sink(pos0)={int(sink_mask.sum())} "
          f"train_docs={int(is_train_doc.sum())} eval_docs={int((~is_train_doc).sum())}", flush=True)

    A = np.asarray(acts, dtype=np.float32)       # 80000x768 f32 = 245 MB, within RAM budget
    norms = np.linalg.norm(A, axis=1)
    sink_norm = float(norms[sink_mask].mean()); nonsink_norm = float(norms[~sink_mask].mean())

    all_rows = []; macros = {}; norm_info = {}
    for tag, keep in [("WITHSINK", np.ones(N, dtype=bool)), ("NONSINK", ~sink_mask)]:
        tr_idx = np.where(train_mask & keep)[0]
        ev_idx = np.where(eval_mask & keep)[0]
        tr = A[tr_idx[:N_TRAIN]]
        ev = A[ev_idx[:N_EVAL]]
        rows, macro, ninfo = build_and_score(tr, ev, D, tag, t0)
        all_rows += rows; macros[tag] = macro; norm_info[tag] = ninfo

    with open(os.path.join(RES, "position_stratified_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "family", "baseline", "auroc"])
        w.writeheader(); [w.writerow(r) for r in all_rows]

    summary = {
        "layer": 6, "d_model": D, "n_rows": int(N), "n_docs": int(n_docs),
        "split": "document-level (even docs train / odd docs eval); no token shared across split",
        "sink_token_pos0_count": int(sink_mask.sum()),
        "sink_pos0_norm_mean": round(sink_norm, 1),
        "nonsink_norm_mean": round(nonsink_norm, 1),
        "n_train": N_TRAIN, "n_eval_per_class": N_EVAL, "seed": SEED,
        "macro_auroc": macros, "norm_info": norm_info,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(RES, "position_stratified_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== MACRO AUROC (WITHSINK -> NONSINK) ===", flush=True)
    for b in macros["WITHSINK"]:
        print(f"  {b:16s} {macros['WITHSINK'][b]:.3f} -> {macros['NONSINK'][b]:.3f}", flush=True)
    print(f"sink norm {sink_norm:.0f} vs non-sink {nonsink_norm:.0f}", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
