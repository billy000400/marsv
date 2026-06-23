"""D6 Phase 5 (CORRECTED) — genuinely IN-CONTEXT downstream validation (claim 3).

Codex review 20260623T024606Z finding #1: the original `context_validation.py` (and
`causal_repair.py`, `functional_features.py`) continued the forward pass by feeding a single
position `[B,1,768]` through the late GPT-2 blocks. That drops the prompt's previous-token hidden
states, so later-layer attention cannot attend to context. The measured KL was therefore
single-position late-block continuation, NOT full-context continuation.

This script fixes that with a FORWARD HOOK on transformer block `LAYER` (whose output is
resid_post@LAYER = hidden_states[LAYER+1]). During a FULL model forward over the real prompt, the
hook overwrites ONLY the last real-token residual with the corrupted activation; we then read the
last-position next-token logits. This is the genuine in-context downstream distribution.

We re-run the exact Phase-5 analysis (two norm-matched severity sweeps, the same realness scores)
and recompute Spearman / partial-Spearman(ctrl dist_to_orig) to see whether claim 3 — functional
scores predict downstream KL beyond movement distance — survives the correction.
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
BS = 64
rng = np.random.default_rng(0)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def partial_spearman(score, kl, ctrl):
    def rank(z): return np.argsort(np.argsort(z)).astype(float)
    rs, rk, rc = rank(score), rank(kl), rank(ctrl)
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]
        beta = np.linalg.lstsq(x1, y, rcond=None)[0]
        return y - x1 @ beta
    return spearman(resid(rs, rc), resid(rk, rc))


def main():
    t0 = time.time()
    # ---- statistical reference on TRAIN reals (same as original) ----
    a = np.load(os.path.join(SRC, f"acts_layer{LAYER}.npy"), mmap_mode="r"); D = a.shape[1]
    train = np.asarray(a[:N_TRAIN], dtype=np.float32)
    mu = train.mean(0); Xc = train - mu; cov = (Xc.T @ Xc) / (len(train) - 1)
    cov_s = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + 1e-3 * np.eye(D, dtype=np.float32)
    cov_inv = np.linalg.inv(cov_s).astype(np.float32)
    maha_train = np.einsum("ij,jk,ik->i", Xc, cov_inv, Xc)
    real_med_maha = float(np.median(maha_train))
    knn_tr = train[rng.choice(len(train), KNN_SUB, replace=False)]; knn_sqn = (knn_tr ** 2).sum(1)

    # ---- model + prompts ----
    m = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    with open(TXT) as f:
        texts = [ln.strip() for ln in f if len(ln.strip()) > 200][:N_PROMPTS]

    # tokenize once; cache ids/mask/last_idx
    all_ids, all_mask, all_last = [], [], []
    for i in range(0, len(texts), BS):
        enc = tok(texts[i:i + BS], return_tensors="pt", truncation=True,
                  max_length=SEQ, padding="max_length")
        all_ids.append(enc["input_ids"]); all_mask.append(enc["attention_mask"])
        all_last.append(enc["attention_mask"].sum(1) - 1)
    ids = torch.cat(all_ids).to(DEVICE); mask = torch.cat(all_mask).to(DEVICE)
    last = torch.cat(all_last).to(DEVICE)
    Np = ids.shape[0]

    block = m.transformer.h[LAYER]  # output == resid_post@LAYER == hidden_states[LAYER+1]

    # in-context logits at last position, with the last-token residual at LAYER replaced by `rep`.
    # rep=None  -> unmodified (clean) forward.
    def hooked_last_logits(rep_batch, ids_b, mask_b, last_b):
        handle = None
        if rep_batch is not None:
            rb = rep_batch
            lb = last_b
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[torch.arange(h.shape[0], device=h.device), lb] = rb.to(h.dtype)
                if isinstance(out, tuple):
                    return (h,) + tuple(out[1:])
                return h
            handle = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = m(input_ids=ids_b, attention_mask=mask_b).logits
        finally:
            if handle is not None:
                handle.remove()
        return logits[torch.arange(logits.shape[0], device=logits.device), last_b]  # [B,V]

    # ---- extract clean last-position activations (full forward) ----
    clean = np.empty((Np, D), np.float32)
    captured = {}
    def cap_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h
    for i in range(0, Np, BS):
        sl = slice(i, i + BS)
        hh = block.register_forward_hook(cap_hook)
        with torch.no_grad():
            m(input_ids=ids[sl], attention_mask=mask[sl])
        hh.remove()
        h = captured["h"]
        lb = last[sl]
        clean[sl] = h[torch.arange(h.shape[0], device=h.device), lb].float().cpu().numpy()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print(f"[{time.time()-t0:.0f}s] {Np} clean last-pos activations (in-context capture)", flush=True)

    def knn_dist(X):
        xsq = (X ** 2).sum(1); out = np.empty(len(X), np.float32)
        for i in range(0, len(X), 2000):
            d2 = xsq[i:i + 2000, None] + knn_sqn[None, :] - 2 * (X[i:i + 2000] @ knn_tr.T)
            out[i:i + 2000] = np.sqrt(np.maximum(d2.min(1), 0))
        return out

    # batched in-context downstream KL + functional features for a full corrupted set
    def downstream(corr):
        """corr [Np,D] -> (kl[Np], entropy[Np], plateau_kl[Np]) all genuinely in-context."""
        kl = np.empty(Np, np.float32); ent = np.empty(Np, np.float32); pk = np.empty(Np, np.float32)
        for i in range(0, Np, BS):
            sl = slice(i, i + BS)
            ids_b, mask_b, last_b = ids[sl], mask[sl], last[sl]
            rep = torch.from_numpy(corr[sl]).to(DEVICE).float()
            lp_clean = torch.log_softmax(hooked_last_logits(None, ids_b, mask_b, last_b), -1)
            lp_corr = torch.log_softmax(hooked_last_logits(rep, ids_b, mask_b, last_b), -1)
            p = lp_corr.exp()
            kl[sl] = (lp_clean.exp() * (lp_clean - lp_corr)).sum(-1).cpu().numpy()
            ent[sl] = (-(p * lp_corr).sum(-1)).cpu().numpy()
            # in-context plateau: perturb the injected residual, KL of last-pos output
            xn = rep.norm(dim=1, keepdim=True)
            acc = torch.zeros(rep.shape[0], device=DEVICE)
            for _ in range(N_PLATEAU):
                nz = torch.randn_like(rep); nz = nz / nz.norm(dim=1, keepdim=True) * xn * EPS_PLATEAU
                lp_p = torch.log_softmax(hooked_last_logits(rep + nz, ids_b, mask_b, last_b), -1)
                acc += (p * (lp_corr - lp_p)).sum(-1)
            pk[sl] = (acc / N_PLATEAU).cpu().numpy()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        return kl, ent, pk

    rows_corr = []; sanity = {}
    other = clean[rng.permutation(Np)]
    for mode, sweep in [("noise", NOISE_S), ("interp", INTERP_S)]:
        recs = []
        for s in sweep:
            if mode == "noise":
                rms = np.linalg.norm(clean, axis=1, keepdims=True) / np.sqrt(D)
                corr = clean + s * rms * rng.standard_normal((Np, D)).astype(np.float32)
            else:
                corr = (1 - s) * clean + s * other
            on = np.linalg.norm(clean, axis=1, keepdims=True)
            corr = (corr * (on / np.linalg.norm(corr, axis=1, keepdims=True))).astype(np.float32)
            kl, ent, pk = downstream(corr)
            Xc_ = corr - mu
            maha = np.einsum("ij,jk,ik->i", Xc_, cov_inv, Xc_)
            maha_two = np.abs(maha - real_med_maha)
            knn = knn_dist(corr)
            dist = np.linalg.norm(corr - clean, axis=1)
            for j in range(Np):
                recs.append((maha_two[j], ent[j], pk[j], knn[j], dist[j], kl[j]))
            sanity[f"{mode}_s{s}_meanKL"] = round(float(kl.mean()), 4)
            print(f"[{time.time()-t0:.0f}s] {mode} s={s} meanKL={kl.mean():.4f}", flush=True)
        recs = np.array(recs)
        names = ["maha_twosided", "entropy", "plateau_kl", "knn_distance", "dist_to_orig"]
        kl = recs[:, 5]
        for k, nm in enumerate(names):
            sp = spearman(recs[:, k], kl)
            pp = partial_spearman(recs[:, k], kl, recs[:, 4]) if nm != "dist_to_orig" else float("nan")
            rows_corr.append({"sweep": mode, "score": nm, "spearman_kl": round(sp, 4),
                              "partial_spearman_ctrl_dist": (round(pp, 4) if pp == pp else "")})

    with open(os.path.join(RES, "functional_prediction_v2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sweep", "score", "spearman_kl", "partial_spearman_ctrl_dist"])
        w.writeheader(); [w.writerow(r) for r in rows_corr]
    with open(os.path.join(RES, "functional_prediction_v2_summary.json"), "w") as f:
        json.dump({"layer": LAYER, "n_prompts": Np, "method": "in-context forward hook (full context)",
                   "noise_s": NOISE_S, "interp_s": INTERP_S, "sanity_meanKL": sanity,
                   "rows": rows_corr, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)

    print("\n=== IN-CONTEXT Spearman(score, downstream KL) / partial controlling dist_to_orig ===", flush=True)
    for r in rows_corr:
        print(f"  {r['sweep']:7s} {r['score']:14s} rho={r['spearman_kl']:>7}  partial={r['partial_spearman_ctrl_dist']}", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
