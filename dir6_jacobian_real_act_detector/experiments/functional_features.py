"""D6 Phase 3/5 — FUNCTIONAL probe where statistical baselines fail (interp, tangent_pert).

Treat each cached activation x as resid_post (hidden_states[L+1]) at a single position, run the
remaining GPT-2 blocks h[L+1:] + ln_f + lm_head -> logits (verified to reconstruct real logits
exactly when given full context). Per-activation functional features:
  - entropy : Shannon entropy of next-token distribution
  - msp     : max softmax prob (1 - MSP is the classic OOD score)
  - plateau_kl : mean output KL(p || p') under small random activation perturbations
                 (eps * ||x|| ); low KL = flat "plateau", high = sensitive. dir9-style.
  - logit_max : max logit (pre-softmax scale)
Compare real vs {interp, tangent_pert, cov_gauss, norm_pert} via AUROC for each feature, against
the statistical ceiling (interp/tangent_pert were ~chance for all stats).

GPU (A10), VRAM capped 0.45, torch threads 2. Pure-forward, no grad.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
SRC = os.path.join(ROOT, "..", "dir3_manifold", "data")
LAYER = 6
SEED = 0
N_TRAIN = 30_000
N_EVAL = 2_000
GAP = 5_000
SHRINK = 0.05
TAN_K = 50
PERT_REL = 0.5
EPS_PLATEAU = 0.02     # perturbation radius (fraction of ||x||) for plateau-KL
N_PLATEAU = 4
rng = np.random.default_rng(SEED)


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
    np_, nn_ = labels.sum(), len(labels) - labels.sum()
    if np_ == 0 or nn_ == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn_))


def renorm(x, tn):
    return x * (tn / np.linalg.norm(x, axis=1, keepdims=True))


def build_families(real, train):
    D = real.shape[1]
    mu = train.mean(0); Xc = train - mu
    cov = (Xc.T @ Xc) / (len(train) - 1)
    cov_s = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + 1e-3 * np.eye(D, dtype=np.float32)
    L_chol = np.linalg.cholesky(cov_s).astype(np.float32)
    evals, evecs = np.linalg.eigh(cov); Vt = evecs[:, ::-1][:, :TAN_K]
    on = np.linalg.norm(real, axis=1, keepdims=True)
    fams = {}
    fams["cov_gauss"] = (mu + rng.standard_normal((len(real), D)).astype(np.float32) @ L_chol.T).astype(np.float32)
    noise = rng.standard_normal((len(real), D)).astype(np.float32)
    fams["norm_pert"] = renorm(real + PERT_REL * on / np.sqrt(D) * noise, on).astype(np.float32)
    b = real[rng.permutation(len(real))]
    lam = rng.uniform(0.2, 0.8, (len(real), 1)).astype(np.float32)
    fams["interp"] = renorm(lam * real + (1 - lam) * b, on).astype(np.float32)
    g = rng.standard_normal((len(real), D)).astype(np.float32)
    tan = (g @ Vt) @ Vt.T; tan = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-6)
    fams["tangent_pert"] = renorm(real + PERT_REL * on * tan, on).astype(np.float32)
    return fams


class Continuer:
    def __init__(self, layer):
        self.m = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
        self.blocks = self.m.transformer.h[layer + 1:]
        self.ln_f = self.m.transformer.ln_f
        self.head = self.m.lm_head

    @torch.no_grad()
    def logits(self, x):  # x: [B,768] torch on device -> [B,V]
        h = x.unsqueeze(1)  # [B,1,768]
        for blk in self.blocks:
            r = blk(h)
            h = r[0] if isinstance(r, tuple) else r
        h = self.ln_f(h)
        return self.head(h).squeeze(1)  # [B,V]

    @torch.no_grad()
    def features(self, X, bs=256):
        feats = {k: [] for k in ["entropy", "msp", "plateau_kl", "logit_max"]}
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(DEVICE).float()
            lg = self.logits(xb)
            lp = torch.log_softmax(lg, -1); p = lp.exp()
            feats["entropy"].append((-(p * lp).sum(-1)).cpu().numpy())
            feats["msp"].append(p.max(-1).values.cpu().numpy())
            feats["logit_max"].append(lg.max(-1).values.cpu().numpy())
            xn = xb.norm(dim=1, keepdim=True)
            kls = torch.zeros(len(xb), device=DEVICE)
            for _ in range(N_PLATEAU):
                noise = torch.randn_like(xb)
                noise = noise / noise.norm(dim=1, keepdim=True) * xn * EPS_PLATEAU
                lp2 = torch.log_softmax(self.logits(xb + noise), -1)
                kls += (p * (lp - lp2)).sum(-1)  # KL(p||p')
            feats["plateau_kl"].append((kls / N_PLATEAU).cpu().numpy())
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        return {k: np.concatenate(v) for k, v in feats.items()}


def main():
    t0 = time.time()
    a = np.load(os.path.join(SRC, f"acts_layer{LAYER}.npy"), mmap_mode="r")
    train = np.asarray(a[:N_TRAIN], dtype=np.float32)
    es = N_TRAIN + GAP
    real = np.asarray(a[es:es + N_EVAL], dtype=np.float32)
    fams = build_families(real, train)
    print(f"[{time.time()-t0:.0f}s] families built; device={DEVICE}", flush=True)

    cont = Continuer(LAYER)
    fr = cont.features(real)
    print(f"[{time.time()-t0:.0f}s] real features done", flush=True)
    FEATS = list(fr.keys())
    rows = []
    fam_feats = {}
    for fam, X in fams.items():
        ff = cont.features(X); fam_feats[fam] = ff
        print(f"[{time.time()-t0:.0f}s] {fam} features done", flush=True)
        for feat in FEATS:
            ps, ns = fr[feat], ff[feat]
            # orient so AUROC>=0.5 (functional anomaly direction unknown a priori per feat)
            lab = np.concatenate([np.zeros(len(ps)), np.ones(len(ns))])
            sc = np.concatenate([ps, ns])
            au = auroc(lab, sc); au = max(au, 1 - au)
            rows.append({"family": fam, "feature": feat, "auroc_oriented": round(au, 4)})

    with open(os.path.join(RES, "functional_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "feature", "auroc_oriented"])
        w.writeheader(); [w.writerow(r) for r in rows]

    # best functional feature per family + combined (logreg on the 4 functional features, in-sample
    # ceiling check — small, just to see if features jointly separate interp)
    summary = {"layer": LAYER, "n_eval": N_EVAL, "eps_plateau": EPS_PLATEAU,
               "stat_ceiling_note": "interp/tangent_pert were ~0.5 for ALL statistical baselines",
               "per_family": {}}
    for fam in fams:
        fam_rows = [r for r in rows if r["family"] == fam]
        best = max(fam_rows, key=lambda r: r["auroc_oriented"])
        summary["per_family"][fam] = {"best_feature": best["feature"],
                                      "best_auroc": best["auroc_oriented"],
                                      "all": {r["feature"]: r["auroc_oriented"] for r in fam_rows}}
    summary["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RES, "functional_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== functional AUROC (oriented) real-vs-family ===", flush=True)
    print(f"{'family':14s}" + "".join(f"{x:>13s}" for x in FEATS), flush=True)
    for fam in fams:
        vals = {r["feature"]: r["auroc_oriented"] for r in rows if r["family"] == fam}
        print(f"{fam:14s}" + "".join(f"{vals[x]:>13.3f}" for x in FEATS), flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s -> results/functional_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
