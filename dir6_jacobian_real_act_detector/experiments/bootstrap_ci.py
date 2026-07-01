"""D6 rigor pass — bootstrap 95% CIs on the Phase-3 capstone single-score AUROCs.

The deliverable's load-bearing claims rest on BORDERLINE AUROCs (functional entropy/plateau-KL on
`interp` ~0.60-0.61 vs chance 0.50; two-sided Mahalanobis on interp ~0.68; kNN on cov_gauss ~0.97).
None had error bars — PLAN.md lists "Unstable conclusions from small N -> report bootstrap CIs" as an
open risk. This script re-builds the EXACT combined_score.py eval set (same SEED=0, N_EVAL=2000, same
4 families and 4 scores: mahalanobis, knn, entropy, plateau_kl) and puts a paired bootstrap 95% CI on
each single-score AUROC.

Method: compute raw scores ONCE. For each (family, score) fix the orientation sign from the full-sample
directed AUROC (a-priori modeling choice, NOT re-estimated per resample -> no upward bias near 0.5), then
resample the 2000 real-eval rows and 2000 family rows WITH REPLACEMENT B=2000 times, recomputing the
directed AUROC each draw. Report full-sample oriented AUROC, 2.5/97.5 percentile CI, and whether the CI
lower bound exceeds 0.50 (significantly-above-chance discrimination).

GPU (A10) for the functional features, VRAM capped per BUDGET; pure numpy for stats + bootstrap.
"""
import os, json, time, csv
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel

torch.set_num_threads(2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.225)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); PLOTS = os.path.join(ROOT, "plots")
SRC = os.path.join(ROOT, "..", "dir3_manifold", "data")
LAYER = 6; N_TRAIN = 30_000; N_EVAL = 2_000; GAP = 5_000
SHRINK = 0.05; TAN_K = 50; PERT_REL = 0.5; KNN_SUB = 5_000
EPS_PLATEAU = 0.02; N_PLATEAU = 4
FAMS = ["cov_gauss", "norm_pert", "interp", "tangent_pert"]
FEAT_NAMES = ["mahalanobis", "knn", "entropy", "plateau_kl"]
B = 2000
rng = np.random.default_rng(0)          # matches combined_score.py family construction
boot_rng = np.random.default_rng(12345)  # independent stream for resampling


def fast_auroc_dir(pos, neg):
    """Directed AUROC P(pos > neg), tie-aware (needed under bootstrap duplicates)."""
    s = np.concatenate([pos, neg])
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    avg_rank = start + (counts + 1) / 2.0        # 1-indexed average rank per tie group
    ranks = avg_rank[inv]
    npos, nneg = len(pos), len(neg)
    return (ranks[:npos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def renorm(x, tn): return x * (tn / np.linalg.norm(x, axis=1, keepdims=True))


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

    def all4(X):
        Xc_ = X - mu
        maha = np.einsum("ij,jk,ik->i", Xc_, cov_inv, Xc_).astype(np.float64)
        xsq = (X ** 2).sum(1); knn = np.empty(len(X), np.float64)
        for i in range(0, len(X), 2000):
            d2 = xsq[i:i + 2000, None] + knn_sqn[None, :] - 2 * (X[i:i + 2000] @ knn_tr.T)
            knn[i:i + 2000] = np.sqrt(np.maximum(d2.min(1), 0))
        e, p = cont.feats(X)
        return np.stack([maha, knn, e.astype(np.float64), p.astype(np.float64)], 1)  # [N,4]

    Sreal = all4(real)
    Sfam = {f: all4(fams[f]) for f in FAMS}
    print(f"[{time.time()-t0:.0f}s] scores computed; bootstrapping B={B}", flush=True)

    rows = []
    for f in FAMS:
        for j, nm in enumerate(FEAT_NAMES):
            pos = Sreal[:, j]; neg = Sfam[f][:, j]
            au_dir = fast_auroc_dir(neg, pos)          # label=1 is the fake family
            sign = 1.0 if au_dir >= 0.5 else -1.0       # fix orientation from full sample
            au_full = au_dir if sign > 0 else 1.0 - au_dir
            boots = np.empty(B)
            for bidx in range(B):
                ri = boot_rng.integers(0, N_EVAL, N_EVAL)
                fi = boot_rng.integers(0, N_EVAL, N_EVAL)
                boots[bidx] = fast_auroc_dir(sign * neg[fi], sign * pos[ri])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"family": f, "score": nm,
                         "auroc": round(float(au_full), 4),
                         "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                         "sig_above_chance": bool(lo > 0.5)})
            print(f"  {f:13s} {nm:12s} AUROC {au_full:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
                  f"  {'sig>0.5' if lo > 0.5 else 'n.s.'}", flush=True)

    with open(os.path.join(RES, "bootstrap_ci.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "score", "auroc", "ci_lo", "ci_hi", "sig_above_chance"])
        w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(RES, "bootstrap_ci_summary.json"), "w") as fh:
        json.dump({"layer": LAYER, "n_eval": N_EVAL, "n_boot": B, "seed_family": 0, "seed_boot": 12345,
                   "orientation": "fixed from full-sample directed AUROC (two-sided reporting)",
                   "rows": rows, "elapsed_s": round(time.time() - t0, 1)}, fh, indent=2)

    # --- figure: AUROC with 95% CI error bars, one panel per family ---
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    colors = {"mahalanobis": "#4C72B0", "knn": "#55A868", "entropy": "#C44E52", "plateau_kl": "#8172B3"}
    for ax, f in zip(axes, FAMS):
        fr = [r for r in rows if r["family"] == f]
        y = np.arange(len(fr))
        au = [r["auroc"] for r in fr]
        lo = [r["auroc"] - r["ci_lo"] for r in fr]; hi = [r["ci_hi"] - r["auroc"] for r in fr]
        ax.errorbar(au, y, xerr=[lo, hi], fmt="o", capsize=4,
                    color="#333", ecolor="#888", markersize=7, zorder=3)
        for yi, r in zip(y, fr):
            ax.plot(r["auroc"], yi, "o", color=colors[r["score"]], markersize=7, zorder=4)
        ax.axvline(0.5, color="crimson", ls="--", lw=1, label="chance")
        ax.set_yticks(y); ax.set_yticklabels([r["score"] for r in fr] if f == FAMS[0] else [])
        ax.set_title(f); ax.set_xlim(0.35, 1.02); ax.set_xlabel("AUROC (95% CI)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Phase-3 capstone single-score AUROC with bootstrap 95% CIs (N=2000/family, B=2000)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PLOTS, "fig10_bootstrap_ci.png"), dpi=110)
    plt.close(fig)
    print(f"\nDONE {time.time()-t0:.0f}s -> results/bootstrap_ci.csv, plots/fig10_bootstrap_ci.png", flush=True)


if __name__ == "__main__":
    main()
