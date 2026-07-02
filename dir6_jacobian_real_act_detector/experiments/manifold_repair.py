"""D6 Phase 6b — MANIFOLD / denoising-prior causal repair (claim 4, the open lever).

Phase 6 showed naive gradient descent on a SCALAR realness score (Mahalanobis, plateau-KL) fails as a
causal objective: it Goodharts and makes downstream behaviour WORSE than a matched random move. The
PLAN's remaining open alternative is a "manifold / denoising-prior causal objective". Here we test the
simplest nonparametric version: a kNN manifold-PROJECTION (a mean-shift / denoising step) that moves the
corrupted activation toward the mean of its k nearest REAL train activations — using the empirical
real-activation manifold as the prior, WITHOUT any reference to the clean target (objective-free).

Setup mirrors causal_repair_v2 exactly (in-context full-model forward hook overwriting only the
last-token resid_post@L6; norm-matched Gaussian corruption s=1). All controls are matched to the kNN
projection's OWN L2 move so the comparison is at equal movement.

Methods:
  - corrupted(start)                : no move (reference).
  - knn_project(k=16, 1 step)       : move all the way to the mean of the 16 nearest train reals.
  - knn_meanshift(k=16, 3 steps)    : iterate the projection 3x (re-find neighbours each step).
  - random_move(knn1-matched)       : random direction, matched to knn_project move.
  - shrink_clean(oracle, knn1-match): move toward the TRUE clean act, matched to knn_project move (ceiling).
External objective-free metrics: in-context KL(clean||x) and NLL of clean argmax (lower=better).
Claim 4 (manifold form) holds iff knn_project beats its matched random control.
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
LAYER = 6; SEQ = 64; N_PROMPTS = 300; N_TRAIN = 30_000
NOISE_S = 1.0; K = 16; MS_STEPS = 3; BS = 48
rng = np.random.default_rng(0); torch.manual_seed(0)


def main():
    t0 = time.time()
    a = np.load(os.path.join(SRC, f"acts_layer{LAYER}.npy"), mmap_mode="r"); D = a.shape[1]
    train = np.asarray(a[:N_TRAIN], dtype=np.float32)
    trainD = torch.from_numpy(train).to(DEVICE)           # [N_TRAIN, D] real manifold sample
    train_sqn = (trainD * trainD).sum(1)                  # precomputed ||t||^2 for cdist

    m = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    with open(TXT) as f:
        texts = [ln.strip() for ln in f if len(ln.strip()) > 200][:N_PROMPTS]
    block = m.transformer.h[LAYER]

    def cont(rep, ids_b, mask_b, last_b):
        lb = last_b
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone(); h[torch.arange(h.shape[0], device=h.device), lb] = rep.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handle = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = m(input_ids=ids_b, attention_mask=mask_b).logits
        finally:
            handle.remove()
        return logits[torch.arange(logits.shape[0], device=logits.device), lb]

    def knn_mean(xD):
        """Mean of the K nearest train reals to each row of xD (euclidean). No self-reference."""
        xsq = (xD * xD).sum(1, keepdim=True)                       # [B,1]
        d2 = xsq + train_sqn[None, :] - 2.0 * (xD @ trainD.T)      # [B,N_TRAIN]
        idx = torch.topk(d2, K, dim=1, largest=False).indices      # [B,K]
        return trainD[idx].mean(1)                                 # [B,D]

    # capture clean last-pos activations in context
    ids_all, mask_all, last_all, clean_all = [], [], [], []
    cap = {}
    def caphook(module, inp, out):
        cap["h"] = out[0] if isinstance(out, tuple) else out
    for i in range(0, len(texts), BS):
        enc = tok(texts[i:i + BS], return_tensors="pt", truncation=True, max_length=SEQ, padding="max_length")
        ib = enc["input_ids"].to(DEVICE); mb = enc["attention_mask"].to(DEVICE); lb = mb.sum(1) - 1
        hh = block.register_forward_hook(caphook)
        with torch.no_grad():
            m(input_ids=ib, attention_mask=mb)
        hh.remove()
        h = cap["h"]; cl = h[torch.arange(h.shape[0], device=h.device), lb].float()
        ids_all.append(ib); mask_all.append(mb); last_all.append(lb); clean_all.append(cl.cpu().numpy())
    Np = sum(len(c) for c in clean_all)
    print(f"[{time.time()-t0:.0f}s] {Np} clean acts (in-context capture)", flush=True)

    muD = trainD.mean(0)
    agg = {}
    def add(name, kl, nll, move, dcl, dmn):
        agg.setdefault(name, []).append(np.stack([kl, nll, move, dcl, dmn], 1))

    for bi in range(len(ids_all)):
        ib, mb, lb = ids_all[bi], mask_all[bi], last_all[bi]
        cl_np = clean_all[bi]; B = len(cl_np)
        clD = torch.from_numpy(cl_np).to(DEVICE)
        on = np.linalg.norm(cl_np, axis=1, keepdims=True); rms = on / np.sqrt(D)
        corr = cl_np + NOISE_S * rms * rng.standard_normal((B, D)).astype(np.float32)
        corr = (corr * (on / np.linalg.norm(corr, axis=1, keepdims=True))).astype(np.float32)
        corrD = torch.from_numpy(corr).to(DEVICE)

        lp_clean = torch.log_softmax(cont(clD, ib, mb, lb), -1)
        clean_argmax = lp_clean.argmax(-1); p_clean = lp_clean.exp()

        def ext(xD):
            lp = torch.log_softmax(cont(xD, ib, mb, lb), -1)
            kl = (p_clean * (lp_clean - lp)).sum(-1)
            nll = -lp.gather(1, clean_argmax[:, None]).squeeze(1)
            return kl.cpu().numpy(), nll.cpu().numpy()

        rep_knn1 = knn_mean(corrD)                          # single projection step (full move to kNN mean)
        x = corrD.clone()
        for _ in range(MS_STEPS):                           # iterated mean-shift
            x = knn_mean(x)
        rep_ms = x

        budget1 = (rep_knn1 - corrD).norm(dim=1, keepdim=True)
        def matched(direction, bud):
            u = direction / (direction.norm(dim=1, keepdim=True) + 1e-8); return corrD + bud * u
        g = torch.randn_like(corrD)
        knn_dir = rep_knn1 - corrD                          # partial steps along the kNN-mean direction
        methods = {
            "corrupted(start)": corrD,
            "knn_project(t=0.10)": corrD + 0.10 * knn_dir,
            "knn_project(t=0.25)": corrD + 0.25 * knn_dir,
            "knn_project(t=0.50)": corrD + 0.50 * knn_dir,
            "knn_project(t=1.00)": rep_knn1,
            "knn_meanshift(k16,3step)": rep_ms,
            # random controls matched to each partial-step L2 move (direction test)
            "random_move(t=0.25-matched)": matched(g, 0.25 * budget1),
            "random_move(t=0.50-matched)": matched(g, 0.50 * budget1),
            "random_move(knn1-matched)": matched(g, budget1),
            # oracle ceiling at t=0.25 move (best partial-step budget)
            "shrink_clean(oracle,t=0.25-matched)": matched(clD - corrD, 0.25 * budget1),
            "shrink_clean(oracle,knn1-matched)": matched(clD - corrD, budget1),
        }
        for name, xD in methods.items():
            kl, nll = ext(xD)
            move = (xD - corrD).norm(dim=1).cpu().numpy()
            dcl = (xD - clD).norm(dim=1).cpu().numpy(); dmn = (xD - muD[None, :]).norm(dim=1).cpu().numpy()
            add(name, kl, nll, move, dcl, dmn)
        print(f"[{time.time()-t0:.0f}s] batch {bi+1}/{len(ids_all)} (knn1 move {budget1.mean():.1f})", flush=True)

    rows = []
    for name, parts in agg.items():
        M = np.concatenate(parts, 0)
        rows.append({"method": name, "ext_KL_clean": round(float(M[:, 0].mean()), 4),
                     "ext_NLL_cleanargmax": round(float(M[:, 1].mean()), 4),
                     "move_from_corrupt": round(float(M[:, 2].mean()), 2),
                     "dist_to_clean": round(float(M[:, 3].mean()), 2),
                     "dist_to_mean": round(float(M[:, 4].mean()), 2)})
    with open(os.path.join(RES, "manifold_repair_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(RES, "manifold_repair_summary.json"), "w") as f:
        json.dump({"layer": LAYER, "n": Np, "noise_s": NOISE_S, "k": K, "meanshift_steps": MS_STEPS,
                   "n_train": N_TRAIN, "method": "in-context full-model forward hook; kNN manifold projection",
                   "rows": rows, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)

    print("\n=== MANIFOLD repair: in-context KL(clean||x) & NLL clean-argmax (lower=better) ===", flush=True)
    print(f"{'method':36s}{'extKL':>9}{'extNLL':>9}{'move':>8}{'d_clean':>9}{'d_mean':>9}", flush=True)
    for r in rows:
        print(f"{r['method']:36s}{r['ext_KL_clean']:>9}{r['ext_NLL_cleanargmax']:>9}{r['move_from_corrupt']:>8}{r['dist_to_clean']:>9}{r['dist_to_mean']:>9}", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
