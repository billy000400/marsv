"""D6 Phase 5 — CONTEXT-AWARE functional validation (claim 3: prediction).

Addresses the main limitation of the single-position probe. We now corrupt the layer-6 resid_post
at the LAST position of a real prompt, continue the REAL forward pass through blocks 7..11 + ln_f +
head, and measure the TRUE in-context downstream degradation KL(clean || corrupted). Then ask: does a
realness score predict that degradation BEYOND distance-to-original?

Corruption severity sweeps (per prompt):
  - noise   : x + s * rms * gaussian  (off-manifold)        s in {0,.25,.5,1,2}
  - interp  : (1-s)*x + s*x_other     (toward another real) s in {0,.2,.4,.6,.8}
Both renormed to the clean activation's norm (norm-matched, so norm is not a confound).

Realness scores per corrupted activation: maha_twosided = |maha - real_median_maha| (anomaly),
entropy, plateau_kl (single-position functional), knn_distance, and the control dist_to_orig=||dx||.
Downstream metric: KL(clean_p || corrupt_p) from the in-context continuation.

Reports Spearman corr(score, KL) and PARTIAL Spearman controlling for dist_to_orig. A score with
nonzero partial correlation predicts degradation beyond mere movement distance (claim-3 evidence).
"""
import os, json, time, csv
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_num_threads(2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.45)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); SRC = os.path.join(ROOT, "..", "dir3_manifold", "data")
TXT = os.path.join(ROOT, "..", "dir9_ood", "data", "fineweb_sample.txt")
LAYER = 6; SEQ = 64; N_PROMPTS = 400; N_TRAIN = 30_000; SHRINK = 0.05; KNN_SUB = 5_000
EPS_PLATEAU = 0.02; N_PLATEAU = 4
NOISE_S = [0.0, 0.25, 0.5, 1.0, 2.0]
INTERP_S = [0.0, 0.2, 0.4, 0.6, 0.8]
rng = np.random.default_rng(0)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def partial_spearman(score, kl, ctrl):
    """Spearman of residuals of score and kl after linear regression on rank(ctrl)."""
    def rank(z): r = np.argsort(np.argsort(z)).astype(float); return r
    rs, rk, rc = rank(score), rank(kl), rank(ctrl)
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]
        beta = np.linalg.lstsq(x1, y, rcond=None)[0]
        return y - x1 @ beta
    return spearman(resid(rs, rc), resid(rk, rc))


def main():
    t0 = time.time()
    # fit statistical reference on TRAIN reals
    a = np.load(os.path.join(SRC, f"acts_layer{LAYER}.npy"), mmap_mode="r"); D = a.shape[1]
    train = np.asarray(a[:N_TRAIN], dtype=np.float32)
    mu = train.mean(0); Xc = train - mu; cov = (Xc.T @ Xc) / (len(train) - 1)
    cov_s = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + 1e-3 * np.eye(D, dtype=np.float32)
    cov_inv = np.linalg.inv(cov_s).astype(np.float32)
    maha_train = np.einsum("ij,jk,ik->i", Xc, cov_inv, Xc)
    real_med_maha = float(np.median(maha_train))
    knn_tr = train[rng.choice(len(train), KNN_SUB, replace=False)]; knn_sqn = (knn_tr ** 2).sum(1)
    muD = torch.from_numpy(mu).to(DEVICE); covinvD = torch.from_numpy(cov_inv).to(DEVICE)

    # model + prompts
    m = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    with open(TXT) as f:
        texts = [ln.strip() for ln in f if len(ln.strip()) > 200][:N_PROMPTS]
    blocks = m.transformer.h[LAYER + 1:]; ln_f = m.transformer.ln_f; head = m.lm_head

    @torch.no_grad()
    def continue_from(x):  # x [B,768] -> logits [B,V]
        h = x.unsqueeze(1)
        for blk in blocks:
            r = blk(h); h = r[0] if isinstance(r, tuple) else r
        return head(ln_f(h)).squeeze(1)

    @torch.no_grad()
    def plateau(x):
        lg = continue_from(x); lp = torch.log_softmax(lg, -1); p = lp.exp()
        ent = (-(p * lp).sum(-1))
        xn = x.norm(dim=1, keepdim=True); kl = torch.zeros(len(x), device=DEVICE)
        for _ in range(N_PLATEAU):
            nz = torch.randn_like(x); nz = nz / nz.norm(dim=1, keepdim=True) * xn * EPS_PLATEAU
            kl += (p * (lp - torch.log_softmax(continue_from(x + nz), -1))).sum(-1)
        return ent.cpu().numpy(), (kl / N_PLATEAU).cpu().numpy(), lp  # clean log-probs returned

    # collect clean last-position activations
    clean_acts = []
    bs = 64
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", truncation=True, max_length=SEQ, padding="max_length")
        ids = enc["input_ids"].to(DEVICE); mask = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hs = out.hidden_states[LAYER + 1]  # [B,SEQ,768]
        last = mask.sum(1) - 1
        clean_acts.append(hs[torch.arange(len(ids)), last].float().cpu().numpy())
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    clean = np.concatenate(clean_acts, 0).astype(np.float32)
    Np = len(clean)
    print(f"[{time.time()-t0:.0f}s] {Np} clean last-pos activations", flush=True)

    def knn_dist(X):
        xsq = (X ** 2).sum(1); out = np.empty(len(X), np.float32)
        for i in range(0, len(X), 2000):
            d2 = xsq[i:i + 2000, None] + knn_sqn[None, :] - 2 * (X[i:i + 2000] @ knn_tr.T)
            out[i:i + 2000] = np.sqrt(np.maximum(d2.min(1), 0))
        return out

    rows_corr = []
    other = clean[rng.permutation(Np)]  # partner reals for interp
    for mode, sweep in [("noise", NOISE_S), ("interp", INTERP_S)]:
        recs = []  # (maha_two, entropy, plateau, knn, dist, kl)
        for s in sweep:
            if mode == "noise":
                rms = np.linalg.norm(clean, axis=1, keepdims=True) / np.sqrt(D)
                corr = clean + s * rms * rng.standard_normal((Np, D)).astype(np.float32)
            else:
                corr = (1 - s) * clean + s * other
            on = np.linalg.norm(clean, axis=1, keepdims=True)
            corr = (corr * (on / np.linalg.norm(corr, axis=1, keepdims=True))).astype(np.float32)
            # downstream KL(clean||corrupt) via continue-forward (clean s=0 -> ~0)
            xt = torch.from_numpy(corr).to(DEVICE).float()
            ct = torch.from_numpy(clean).to(DEVICE).float()
            with torch.no_grad():
                lp_clean = torch.log_softmax(continue_from(ct), -1)
                lp_corr = torch.log_softmax(continue_from(xt), -1)
                kl = (lp_clean.exp() * (lp_clean - lp_corr)).sum(-1).cpu().numpy()
            ent, pk, _ = plateau(xt)
            Xc_ = corr - mu
            maha = np.einsum("ij,jk,ik->i", Xc_, cov_inv, Xc_)
            maha_two = np.abs(maha - real_med_maha)
            knn = knn_dist(corr)
            dist = np.linalg.norm(corr - clean, axis=1)
            for j in range(Np):
                recs.append((maha_two[j], ent[j], pk[j], knn[j], dist[j], kl[j]))
            print(f"[{time.time()-t0:.0f}s] {mode} s={s} meanKL={kl.mean():.3f}", flush=True)
        recs = np.array(recs)
        names = ["maha_twosided", "entropy", "plateau_kl", "knn_distance", "dist_to_orig"]
        kl = recs[:, 5]
        for k, nm in enumerate(names):
            sp = spearman(recs[:, k], kl)
            pp = partial_spearman(recs[:, k], kl, recs[:, 4]) if nm != "dist_to_orig" else float("nan")
            rows_corr.append({"sweep": mode, "score": nm,
                              "spearman_kl": round(sp, 4),
                              "partial_spearman_ctrl_dist": (round(pp, 4) if pp == pp else "")})

    with open(os.path.join(RES, "functional_prediction.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sweep", "score", "spearman_kl", "partial_spearman_ctrl_dist"])
        w.writeheader(); [w.writerow(r) for r in rows_corr]
    with open(os.path.join(RES, "functional_prediction_summary.json"), "w") as f:
        json.dump({"layer": LAYER, "n_prompts": Np, "noise_s": NOISE_S, "interp_s": INTERP_S,
                   "rows": rows_corr, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)

    print("\n=== Spearman(score, downstream KL) / partial controlling dist_to_orig ===", flush=True)
    for r in rows_corr:
        print(f"  {r['sweep']:7s} {r['score']:14s} rho={r['spearman_kl']:>7}  partial={r['partial_spearman_ctrl_dist']}", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
