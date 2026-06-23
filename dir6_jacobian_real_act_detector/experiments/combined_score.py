"""D6 capstone — does a COMBINED realness score (statistical ⊕ functional) generalize across
families under leave-one-family-out, and can it lift the hard `interp` family above 0.61?

Per-activation scores on a shared layer-6 eval set (real + 4 families):
  statistical : mahalanobis, knn_distance
  functional  : entropy, plateau_kl   (from continue-forward probe, GPU)
Then:
  - per-family AUROC of each single score (reference)
  - COMBINED logistic over the 4 standardized scores, LEAVE-ONE-FAMILY-OUT (train on 3, test on
    held-out) -> tests whether a stacked realness score generalizes, incl. to interp.
Pure numpy stats + torch for functional. Writes results/combined_metrics.csv.
"""
import os, json, time, csv
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import GPT2LMHeadModel

torch.set_num_threads(2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.45)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); SRC = os.path.join(ROOT, "..", "dir3_manifold", "data")
LAYER = 6; N_TRAIN = 30_000; N_EVAL = 2_000; GAP = 5_000
SHRINK = 0.05; TAN_K = 50; PERT_REL = 0.5; KNN_SUB = 5_000
EPS_PLATEAU = 0.02; N_PLATEAU = 4
FAMS = ["cov_gauss", "norm_pert", "interp", "tangent_pert"]
rng = np.random.default_rng(0)


def auroc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort"); s = scores[order]
    rs = np.arange(1, len(s) + 1, dtype=np.float64); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rs[i:j + 1] = (i + 1 + j + 1) / 2.0; i = j + 1
    ranks = np.empty(len(s)); ranks[order] = rs
    npp, nn = labels.sum(), len(labels) - labels.sum()
    return float((ranks[labels == 1].sum() - npp * (npp + 1) / 2.0) / (npp * nn)) if npp and nn else float("nan")


def renorm(x, tn): return x * (tn / np.linalg.norm(x, axis=1, keepdims=True))


def logreg(X, y, Xte, l2=1e-2, lr=0.3, iters=600):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b))); g = p - y
        w -= lr * (X.T @ g / len(X) + l2 * w); b -= lr * g.mean()
    return Xte @ w + b


class Continuer:
    def __init__(self, layer):
        self.m = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
        self.blocks = self.m.transformer.h[layer + 1:]
        self.ln_f = self.m.transformer.ln_f; self.head = self.m.lm_head

    @torch.no_grad()
    def _logits(self, x):
        h = x.unsqueeze(1)
        for blk in self.blocks:
            r = blk(h); h = r[0] if isinstance(r, tuple) else r
        return self.head(self.ln_f(h)).squeeze(1)

    @torch.no_grad()
    def feats(self, X, bs=256):
        ent, pk = [], []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(DEVICE).float()
            lg = self._logits(xb); lp = torch.log_softmax(lg, -1); p = lp.exp()
            ent.append((-(p * lp).sum(-1)).cpu().numpy())
            xn = xb.norm(dim=1, keepdim=True); kl = torch.zeros(len(xb), device=DEVICE)
            for _ in range(N_PLATEAU):
                nz = torch.randn_like(xb); nz = nz / nz.norm(dim=1, keepdim=True) * xn * EPS_PLATEAU
                kl += (p * (lp - torch.log_softmax(self._logits(xb + nz), -1))).sum(-1)
            pk.append((kl / N_PLATEAU).cpu().numpy())
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        return np.concatenate(ent), np.concatenate(pk)


def main():
    t0 = time.time()
    a = np.load(os.path.join(SRC, f"acts_layer{LAYER}.npy"), mmap_mode="r"); D = a.shape[1]
    train = np.asarray(a[:N_TRAIN], dtype=np.float32)
    real = np.asarray(a[N_TRAIN + GAP:N_TRAIN + GAP + N_EVAL], dtype=np.float32)
    mu = train.mean(0); Xc = train - mu
    cov = (Xc.T @ Xc) / (len(train) - 1)
    cov_s = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + 1e-3 * np.eye(D, dtype=np.float32)
    cov_inv = np.linalg.inv(cov_s).astype(np.float32)
    L_chol = np.linalg.cholesky(cov_s).astype(np.float32)
    evals, evecs = np.linalg.eigh(cov); Vt = evecs[:, ::-1][:, :TAN_K]
    knn_tr = train[rng.choice(len(train), KNN_SUB, replace=False)]; knn_sqn = (knn_tr ** 2).sum(1)

    on = np.linalg.norm(real, axis=1, keepdims=True)
    fams = {}
    fams["cov_gauss"] = (mu + rng.standard_normal((N_EVAL, D)).astype(np.float32) @ L_chol.T).astype(np.float32)
    fams["norm_pert"] = renorm(real + PERT_REL * on / np.sqrt(D) * rng.standard_normal((N_EVAL, D)).astype(np.float32), on).astype(np.float32)
    b = real[rng.permutation(N_EVAL)]; lam = rng.uniform(0.2, 0.8, (N_EVAL, 1)).astype(np.float32)
    fams["interp"] = renorm(lam * real + (1 - lam) * b, on).astype(np.float32)
    g = rng.standard_normal((N_EVAL, D)).astype(np.float32); tan = (g @ Vt) @ Vt.T
    tan = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-6)
    fams["tangent_pert"] = renorm(real + PERT_REL * on * tan, on).astype(np.float32)

    cont = Continuer(LAYER)

    def stat_scores(X):
        Xc_ = X - mu
        maha = np.einsum("ij,jk,ik->i", Xc_, cov_inv, Xc_)
        xsq = (X ** 2).sum(1); knn = np.empty(len(X), np.float32)
        for i in range(0, len(X), 2000):
            d2 = xsq[i:i + 2000, None] + knn_sqn[None, :] - 2 * (X[i:i + 2000] @ knn_tr.T)
            knn[i:i + 2000] = np.sqrt(np.maximum(d2.min(1), 0))
        return maha, knn

    def all4(X):
        m, k = stat_scores(X); e, p = cont.feats(X)
        return np.stack([m, k, e, p], 1)  # [N,4] mahalanobis,knn,entropy,plateau_kl

    Sreal = all4(real)
    Sfam = {f: all4(fams[f]) for f in FAMS}
    print(f"[{time.time()-t0:.0f}s] all scores computed", flush=True)
    FEAT_NAMES = ["mahalanobis", "knn", "entropy", "plateau_kl"]

    # standardize features by real-eval stats
    fmu = Sreal.mean(0); fsd = Sreal.std(0) + 1e-9
    Zr = (Sreal - fmu) / fsd
    Zf = {f: (Sfam[f] - fmu) / fsd for f in FAMS}

    rows = []
    # single-feature per-family AUROC (oriented) for reference
    for f in FAMS:
        for j, nm in enumerate(FEAT_NAMES):
            au = auroc(np.r_[np.zeros(N_EVAL), np.ones(N_EVAL)], np.r_[Sreal[:, j], Sfam[f][:, j]])
            rows.append({"family": f, "score": nm, "auroc": round(max(au, 1 - au), 4)})

    # COMBINED logistic, leave-one-family-out
    combo = {}
    for held in FAMS:
        tr_f = [x for x in FAMS if x != held]
        Xneg = np.concatenate([Zf[x] for x in tr_f], 0)
        reps = int(np.ceil(len(Xneg) / len(Zr)))
        Xpos = np.tile(Zr, (reps, 1))[:len(Xneg)]
        Xtr = np.r_[Xpos, Xneg]; ytr = np.r_[np.zeros(len(Xpos)), np.ones(len(Xneg))]
        Xte = np.r_[Zr, Zf[held]]; yte = np.r_[np.zeros(N_EVAL), np.ones(N_EVAL)]
        s = logreg(Xtr, ytr, Xte)
        au = auroc(yte, s)
        combo[held] = round(max(au, 1 - au), 4)
        rows.append({"family": held, "score": "COMBINED_lofo", "auroc": combo[held]})

    with open(os.path.join(RES, "combined_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "score", "auroc"]); w.writeheader()
        [w.writerow(r) for r in rows]
    macro_combo = round(float(np.mean(list(combo.values()))), 4)
    with open(os.path.join(RES, "combined_summary.json"), "w") as f:
        json.dump({"layer": LAYER, "features": FEAT_NAMES, "combined_lofo": combo,
                   "combined_lofo_macro": macro_combo, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)

    print("\n=== per-family AUROC (oriented) ===", flush=True)
    print(f"{'family':14s}" + "".join(f"{n:>13s}" for n in FEAT_NAMES) + f"{'COMBINED':>13s}", flush=True)
    for f in FAMS:
        vals = {r["score"]: r["auroc"] for r in rows if r["family"] == f}
        print(f"{f:14s}" + "".join(f"{vals[n]:>13.3f}" for n in FEAT_NAMES) + f"{combo[f]:>13.3f}", flush=True)
    print(f"\nCOMBINED LOFO macro = {macro_combo}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
