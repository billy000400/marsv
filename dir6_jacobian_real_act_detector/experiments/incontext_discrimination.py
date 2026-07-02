"""D6 Phase 3b — IN-CONTEXT discrimination benchmark (top-cited open limitation).

All prior discrimination (Phase 2/3) scored each activation as a STANDALONE vector, and the
functional probe that partially caught the hard `interp` family (entropy/plateau-KL AUROC ~0.61)
was computed OUT-OF-CONTEXT (single position pushed through the late GPT-2 blocks — the same
single-position artifact the codex review flagged for the prediction/causality phases). The open
question: does placing a candidate activation at the last-token position of its *native prompt*
(full context, attention over real previous tokens) sharpen the INTRINSIC functional realness
signal enough to discriminate real from the too-central `interp` and moment-matched `cov_gauss`
negatives?

We only use INTRINSIC scores — features computable WITHOUT knowing the true clean activation:
  functional : entropy, msp(=max softmax prob), plateau_kl (local output sensitivity to a small
               activation perturbation; low = functional plateau = "real-like").
  statistical: maha_twosided (|maha - median_real_maha|), knn_distance (to train reals).
Downstream-KL-to-clean is deliberately EXCLUDED: for real positives it is exactly 0, so it is a
distance-to-original giveaway, not an intrinsic realness score.

For every score we report AUROC real-vs-family both IN-CONTEXT (forward hook, full prompt) and
OUT-OF-CONTEXT (single position through late blocks) so the two are directly comparable on the
identical candidate sets. Orientation of each score is fixed on a held-out reference (real should
score LOW on plateau_kl/maha_two/knn, HIGH on entropy) so AUROC>=0.5 always means "separates".

GPU (A10), VRAM capped 0.225 (4-agent share), torch threads 2, pure-forward no-grad, halve BS on OOM.
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
    torch.cuda.set_per_process_memory_fraction(0.225)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); SRC = os.path.join(ROOT, "..", "dir3_manifold", "data")
TXT = os.path.join(ROOT, "..", "dir9_ood", "data", "fineweb_sample.txt")
LAYER = 6; SEQ = 64; N_PROMPTS = 300; N_TRAIN = 30_000; SHRINK = 0.05; KNN_SUB = 5_000
EPS_PLATEAU = 0.02; N_PLATEAU = 6; TAN_K = 50; PERT_REL = 0.5
BS = 48
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
    np_, nn_ = labels.sum(), len(labels) - labels.sum()
    if np_ == 0 or nn_ == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn_))


def renorm(x, tn):
    return (x * (tn / np.linalg.norm(x, axis=1, keepdims=True))).astype(np.float32)


def main():
    t0 = time.time()
    # ---- statistical reference on TRAIN reals ----
    a = np.load(os.path.join(SRC, f"acts_layer{LAYER}.npy"), mmap_mode="r"); D = a.shape[1]
    train = np.asarray(a[:N_TRAIN], dtype=np.float32)
    mu = train.mean(0); Xc = train - mu; cov = (Xc.T @ Xc) / (len(train) - 1)
    cov_s = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + 1e-3 * np.eye(D, dtype=np.float32)
    cov_inv = np.linalg.inv(cov_s).astype(np.float32)
    L_chol = np.linalg.cholesky(cov_s).astype(np.float32)
    real_med_maha = float(np.median(np.einsum("ij,jk,ik->i", Xc, cov_inv, Xc)))
    knn_tr = train[rng.choice(len(train), KNN_SUB, replace=False)]; knn_sqn = (knn_tr ** 2).sum(1)
    evals, evecs = np.linalg.eigh(cov); Vt = evecs[:, ::-1][:, :TAN_K]

    def maha_two(X):
        Xc_ = X - mu
        return np.abs(np.einsum("ij,jk,ik->i", Xc_, cov_inv, Xc_) - real_med_maha)

    def knn_dist(X):
        xsq = (X ** 2).sum(1); out = np.empty(len(X), np.float32)
        for i in range(0, len(X), 2000):
            d2 = xsq[i:i + 2000, None] + knn_sqn[None, :] - 2 * (X[i:i + 2000] @ knn_tr.T)
            out[i:i + 2000] = np.sqrt(np.maximum(d2.min(1), 0))
        return out

    # ---- model + prompts ----
    m = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    with open(TXT) as f:
        texts = [ln.strip() for ln in f if len(ln.strip()) > 200][:N_PROMPTS]
    all_ids, all_mask, all_last = [], [], []
    for i in range(0, len(texts), BS):
        enc = tok(texts[i:i + BS], return_tensors="pt", truncation=True,
                  max_length=SEQ, padding="max_length")
        all_ids.append(enc["input_ids"]); all_mask.append(enc["attention_mask"])
        all_last.append(enc["attention_mask"].sum(1) - 1)
    ids = torch.cat(all_ids).to(DEVICE); mask = torch.cat(all_mask).to(DEVICE)
    last = torch.cat(all_last).to(DEVICE); Np = ids.shape[0]

    block = m.transformer.h[LAYER]
    late = m.transformer.h[LAYER + 1:]; ln_f = m.transformer.ln_f; lm_head = m.lm_head

    def hooked_last_logits(rep_batch, ids_b, mask_b, last_b):
        handle = None
        if rep_batch is not None:
            rb, lb = rep_batch, last_b
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[torch.arange(h.shape[0], device=h.device), lb] = rb.to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            handle = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = m(input_ids=ids_b, attention_mask=mask_b).logits
        finally:
            if handle is not None:
                handle.remove()
        return logits[torch.arange(logits.shape[0], device=logits.device), last_b]

    def ooc_logits(rep_batch):
        # single position [B,1,D] through late blocks + ln_f + lm_head (out-of-context)
        h = rep_batch.unsqueeze(1)
        with torch.no_grad():
            for blk in late:
                out = blk(h)
                h = out[0] if isinstance(out, tuple) else out
            return lm_head(ln_f(h))[:, 0]

    # ---- capture clean last-position activations (real positives, in-context) ----
    clean = np.empty((Np, D), np.float32); cap = {}
    def cap_hook(module, inp, out):
        cap["h"] = out[0] if isinstance(out, tuple) else out
    for i in range(0, Np, BS):
        sl = slice(i, i + BS); hh = block.register_forward_hook(cap_hook)
        with torch.no_grad():
            m(input_ids=ids[sl], attention_mask=mask[sl])
        hh.remove(); h = cap["h"]; lb = last[sl]
        clean[sl] = h[torch.arange(h.shape[0], device=h.device), lb].float().cpu().numpy()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print(f"[{time.time()-t0:.0f}s] {Np} real in-context activations captured", flush=True)

    # ---- build matched negative families from the captured reals ----
    on = np.linalg.norm(clean, axis=1, keepdims=True)
    fams = {}
    fams["cov_gauss"] = renorm(mu + rng.standard_normal((Np, D)).astype(np.float32) @ L_chol.T, on)
    b = clean[rng.permutation(Np)]; lam = rng.uniform(0.2, 0.8, (Np, 1)).astype(np.float32)
    fams["interp"] = renorm(lam * clean + (1 - lam) * b, on)
    g = rng.standard_normal((Np, D)).astype(np.float32)
    tan = (g @ Vt) @ Vt.T; tan = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-6)
    fams["tangent_pert"] = renorm(clean + PERT_REL * on * tan, on)

    # ---- functional feature extractors (intrinsic; no reference to clean) ----
    def feats(cand, in_context):
        ent = np.empty(Np, np.float32); msp = np.empty(Np, np.float32); pk = np.empty(Np, np.float32)
        for i in range(0, Np, BS):
            sl = slice(i, i + BS)
            rep = torch.from_numpy(cand[sl]).to(DEVICE).float()
            if in_context:
                lg = hooked_last_logits(rep, ids[sl], mask[sl], last[sl])
            else:
                lg = ooc_logits(rep)
            lp = torch.log_softmax(lg, -1); p = lp.exp()
            ent[sl] = (-(p * lp).sum(-1)).cpu().numpy(); msp[sl] = p.max(-1).values.cpu().numpy()
            xn = rep.norm(dim=1, keepdim=True); acc = torch.zeros(rep.shape[0], device=DEVICE)
            for _ in range(N_PLATEAU):
                nz = torch.randn_like(rep); nz = nz / nz.norm(dim=1, keepdim=True) * xn * EPS_PLATEAU
                if in_context:
                    lgp = hooked_last_logits(rep + nz, ids[sl], mask[sl], last[sl])
                else:
                    lgp = ooc_logits(rep + nz)
                acc += (p * (lp - torch.log_softmax(lgp, -1))).sum(-1)
            pk[sl] = (acc / N_PLATEAU).cpu().numpy()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        return {"entropy": ent, "msp": msp, "plateau_kl": pk}

    # score orientation: for real, plateau_kl/maha_two/knn/msp should be LOW, entropy HIGH.
    # AUROC labels: real=1. Sign flips so that "more real-like" -> larger score for AUROC.
    SIGN = {"entropy": +1, "msp": -1, "plateau_kl": -1, "maha_twosided": -1, "knn_distance": -1}

    print(f"[{time.time()-t0:.0f}s] computing functional features (real + 3 families x 2 contexts)...", flush=True)
    real_ic = feats(clean, True); real_ooc = feats(clean, False)
    real_maha = maha_two(clean); real_knn = knn_dist(clean)

    rows = []
    for fam, cand in fams.items():
        f_ic = feats(cand, True); f_ooc = feats(cand, False)
        c_maha = maha_two(cand); c_knn = knn_dist(cand)
        lab = np.concatenate([np.ones(Np), np.zeros(Np)])
        # statistical (context-independent)
        for nm, rv, cv in [("maha_twosided", real_maha, c_maha), ("knn_distance", real_knn, c_knn)]:
            sc = SIGN[nm] * np.concatenate([rv, cv])
            rows.append({"family": fam, "score": nm, "context": "n/a", "auroc": round(auroc(lab, sc), 4)})
        # functional (in-context vs out-of-context)
        for nm in ["entropy", "msp", "plateau_kl"]:
            for ctx, rf, cf in [("in_context", real_ic, f_ic), ("out_of_context", real_ooc, f_ooc)]:
                sc = SIGN[nm] * np.concatenate([rf[nm], cf[nm]])
                rows.append({"family": fam, "score": nm, "context": ctx, "auroc": round(auroc(lab, sc), 4)})
        print(f"[{time.time()-t0:.0f}s] {fam} done", flush=True)

    with open(os.path.join(RES, "incontext_discrimination.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "score", "context", "auroc"])
        w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(RES, "incontext_discrimination_summary.json"), "w") as f:
        json.dump({"layer": LAYER, "n_prompts": Np, "families": list(fams),
                   "n_plateau": N_PLATEAU, "eps_plateau": EPS_PLATEAU,
                   "method": "forward-hook inject at last real-token pos (in-context) vs single-position late blocks (out-of-context)",
                   "rows": rows, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)

    print("\n=== AUROC real-vs-family (>=0.5 separates; functional shown in vs out of context) ===", flush=True)
    for fam in fams:
        print(f"[{fam}]", flush=True)
        for r in rows:
            if r["family"] == fam:
                print(f"    {r['score']:14s} {r['context']:14s} auroc={r['auroc']}", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
