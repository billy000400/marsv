"""D6 Phase 7 (S9) — STEERING-PRESERVATION test of the Phase-6b manifold repair (claim 5).

Phase 6b found the FIRST objective-free causal repair: a small (t~0.25) kNN manifold-projection step
toward nearby REAL activations lowers downstream KL of a *randomly corrupted* activation below both the
corrupted start and a matched-size random move. Claim 5 asks the harder, Direction-1 question: when the
"corruption" is an intentional STEERING edit, can the same manifold step improve validity WHILE
PRESERVING the intended steering effect — and crucially, can it do so better than the trivial control of
simply shrinking the steering coefficient (alpha)?

Setup (identical in-context harness to manifold_repair.py):
  - Real positives: last-token resid_post@L6 of N FineWeb prompts, captured in context (forward hook).
  - Manifold: 30k real train activations from the dir3 L6 cache (kNN prior, objective-free).
  - Steering vector v = mean(resid@L6 last-tok of POSITIVE-sentiment sentences)
                       - mean(resid@L6 last-tok of NEGATIVE-sentiment sentences)  (difference-of-means,
    the standard "contrastive activation addition" Direction-1 construction).
  - Steered activation at coefficient alpha:  x_s = x0 + alpha * v.

Decomposition of the output change (logit space, vocab-mean-centred to kill the softmax constant):
  - Intended readout direction d_hat = normalised per-prompt output response of steering in the LINEAR
    (small-alpha) regime:  d_lin = (cont(x0 + a_small*v) - cont(x0)) / a_small ,  d_hat = d_lin/||d_lin||.
  - For any candidate x:  dL = cont(x) - cont(x0);
        E(x)  = <dL, d_hat>            achieved steering effect (higher = more of the intended edit)
        C(x)  = ||dL - E*d_hat||       OFF-TARGET collateral (output change the steering did NOT intend)
  - Validity proxies (all EXTERNAL to the manifold objective, which only moves toward real neighbours):
        C (off-target collateral), next-token entropy H, dist_to_mean, knn_dist to real manifold.

Methods, all starting from the fully-steered x_s at a fixed alpha_full:
  - steered(full)           : x_s.
  - alpha_shrink(f)         : x0 + f*alpha_full*v   for f in a grid  -> the "merely shrink alpha" control.
                              (For a linear v this is IDENTICAL to projecting x_s toward the original x0.)
  - manifold(t)             : x_s + t*(knn_mean(x_s) - x_s)  for t in a grid  -> the Phase-6b repair.
  - random(t-matched)       : x_s + random dir of the SAME L2 size as manifold(t)  -> direction control.

Claim 5 holds iff manifold(t) reaches a given achieved effect E with LOWER collateral C (or better
external validity) than the alpha_shrink frontier at the SAME E — i.e. the repair is NOT explained by
just shrinking alpha. If the alpha_shrink frontier dominates, correction adds nothing over shrinking
alpha (an honest null for claim 5, which the PLAN accepts).
"""
import os, json, time, csv
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_num_threads(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.18)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); SRC = os.path.join(ROOT, "..", "dir3_manifold", "data")
TXT = os.path.join(ROOT, "..", "dir9_ood", "data", "fineweb_sample.txt")
LAYER = 6; SEQ = 64; N_PROMPTS = 200; N_TRAIN = 30_000
K = 16; BS = 40
A_SMALL_FRAC = 0.12                       # small-alpha fraction defining the linear readout d_hat
SHRINK_GRID = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
T_GRID = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
rng = np.random.default_rng(0); torch.manual_seed(0)

POS = [
    "This is absolutely wonderful and I love it.", "What a fantastic, delightful experience.",
    "I am so happy and grateful today.", "The movie was brilliant and deeply moving.",
    "Everything turned out perfectly, it was amazing.", "She smiled with pure joy and excitement.",
    "A truly beautiful, uplifting and inspiring story.", "I feel great, everything is going so well.",
    "The food was delicious and the service excellent.", "It was the best day of my entire life.",
    "Wonderful news, we are thrilled and overjoyed.", "A charming, heartwarming and joyful film.",
    "I adore this place, it is simply magnificent.", "Their kindness made me incredibly happy.",
    "Such a splendid, cheerful and pleasant afternoon.", "The results were superb and very encouraging.",
    "I could not be more delighted with the outcome.", "A gorgeous sunny day full of laughter.",
    "This gift is perfect, thank you so much.", "We had a marvelous, unforgettable celebration.",
]
NEG = [
    "This is absolutely terrible and I hate it.", "What an awful, miserable experience.",
    "I am so sad and hopeless today.", "The movie was boring and deeply disappointing.",
    "Everything fell apart, it was a disaster.", "He frowned with pure anger and disgust.",
    "A truly ugly, depressing and dreadful story.", "I feel awful, everything is going so wrong.",
    "The food was disgusting and the service rude.", "It was the worst day of my entire life.",
    "Terrible news, we are devastated and heartbroken.", "A grim, painful and joyless film.",
    "I despise this place, it is simply horrible.", "Their cruelty made me incredibly miserable.",
    "Such a dreary, gloomy and unpleasant afternoon.", "The results were poor and very discouraging.",
    "I could not be more disappointed with the outcome.", "A bleak grey day full of tears.",
    "This gift is useless, what a waste.", "We had a dreadful, forgettable ordeal.",
]


def main():
    t0 = time.time()
    a = np.load(os.path.join(SRC, f"acts_layer{LAYER}.npy"), mmap_mode="r"); D = a.shape[1]
    trainD = torch.from_numpy(np.asarray(a[:N_TRAIN], dtype=np.float32)).to(DEVICE)
    train_sqn = (trainD * trainD).sum(1)

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
        L = logits[torch.arange(logits.shape[0], device=logits.device), lb]
        return L - L.mean(-1, keepdim=True)          # vocab-mean-centre: kill softmax constant

    def capture_last(prompts):
        """Mean last-token resid@L6 (in-context) over a list of prompts."""
        cap = {}
        def caphook(module, inp, out):
            cap["h"] = out[0] if isinstance(out, tuple) else out
        outs = []
        for i in range(0, len(prompts), BS):
            enc = tok(prompts[i:i + BS], return_tensors="pt", truncation=True, max_length=SEQ,
                      padding="max_length")
            ib = enc["input_ids"].to(DEVICE); mb = enc["attention_mask"].to(DEVICE); lb = mb.sum(1) - 1
            hh = block.register_forward_hook(caphook)
            with torch.no_grad():
                m(input_ids=ib, attention_mask=mb)
            hh.remove()
            h = cap["h"]; outs.append(h[torch.arange(h.shape[0], device=h.device), lb].float().cpu().numpy())
        return np.concatenate(outs, 0)

    # --- build steering vector (difference of means, contrastive activation addition) ---
    v = torch.from_numpy(capture_last(POS).mean(0) - capture_last(NEG).mean(0)).to(DEVICE)
    print(f"[{time.time()-t0:.0f}s] steering vector ||v||={v.norm():.2f}", flush=True)

    def knn_mean(xD):
        xsq = (xD * xD).sum(1, keepdim=True)
        d2 = xsq + train_sqn[None, :] - 2.0 * (xD @ trainD.T)
        d2, idx = torch.topk(d2, K, dim=1, largest=False)
        return trainD[idx].mean(1), d2.clamp_min(0).sqrt().mean(1)   # kNN mean, mean kNN distance

    # capture clean prompt activations in context
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
        h = cap["h"]
        ids_all.append(ib); mask_all.append(mb); last_all.append(lb)
        clean_all.append(h[torch.arange(h.shape[0], device=h.device), lb].float().cpu().numpy())
    Np = sum(len(c) for c in clean_all)

    # choose alpha_full: full-steer L2 move ~= 0.8 * typical clean-activation norm (strong, off-manifold)
    clean_norm = np.concatenate([np.linalg.norm(c, axis=1) for c in clean_all]).mean()
    alpha_full = float(0.8 * clean_norm / v.norm().item())
    print(f"[{time.time()-t0:.0f}s] {Np} clean acts; clean_norm~{clean_norm:.0f}; alpha_full={alpha_full:.3f}"
          f" (||alpha*v||~{alpha_full*v.norm().item():.0f})", flush=True)

    muD = trainD.mean(0)
    agg = {}
    def add(name, arr):
        agg.setdefault(name, []).append(arr)

    for bi in range(len(ids_all)):
        ib, mb, lb = ids_all[bi], mask_all[bi], last_all[bi]
        clD = torch.from_numpy(clean_all[bi]).to(DEVICE); B = clD.shape[0]

        L0 = cont(clD, ib, mb, lb)
        p0 = torch.softmax(L0, -1)
        # linear readout direction d_hat (per prompt), from small-alpha steering
        d_lin = (cont(clD + (A_SMALL_FRAC * alpha_full) * v, ib, mb, lb) - L0) / (A_SMALL_FRAC * alpha_full)
        d_hat = d_lin / (d_lin.norm(dim=1, keepdim=True) + 1e-8)

        x_s = clD + alpha_full * v                          # full steer
        knn_m, _ = knn_mean(x_s); knn_dir = knn_m - x_s
        budget = knn_dir.norm(dim=1, keepdim=True)
        g = torch.randn_like(x_s)

        methods = {"steered(full)": x_s}
        for f in SHRINK_GRID:
            methods[f"alpha_shrink(f={f:.2f})"] = clD + (f * alpha_full) * v
        for t in T_GRID:
            methods[f"manifold(t={t:.2f})"] = x_s + t * knn_dir
            u = g / (g.norm(dim=1, keepdim=True) + 1e-8)
            methods[f"random(t={t:.2f}-matched)"] = x_s + (t * budget) * u

        for name, xD in methods.items():
            Lx = cont(xD, ib, mb, lb); dL = Lx - L0
            E = (dL * d_hat).sum(1)
            C = (dL - E[:, None] * d_hat).norm(dim=1)
            px = torch.softmax(Lx, -1)
            H = -(px * torch.log(px + 1e-12)).sum(1)
            klc = (p0 * (torch.log(p0 + 1e-12) - torch.log_softmax(Lx, -1))).sum(1)
            _, kdist = knn_mean(xD)
            dmn = (xD - muD[None, :]).norm(dim=1)
            arr = torch.stack([E, C, H, klc, kdist, dmn, (xD - clD).norm(dim=1)], 1).cpu().numpy()
            add(name, arr)
        print(f"[{time.time()-t0:.0f}s] batch {bi+1}/{len(ids_all)}", flush=True)

    cols = ["E_effect", "C_offtarget", "entropy", "KL_from_clean", "knn_dist", "dist_to_mean", "move"]
    rows = []
    perprompt = {}
    for name, parts in agg.items():
        M = np.concatenate(parts, 0); perprompt[name] = M.astype(np.float32)
        r = {"method": name}
        for j, c in enumerate(cols):
            r[c] = round(float(M[:, j].mean()), 4)
        rows.append(r)
    with open(os.path.join(RES, "steering_repair_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    np.savez(os.path.join(RES, "steering_repair_perprompt.npz"), cols=np.array(cols), **perprompt)
    with open(os.path.join(RES, "steering_repair_summary.json"), "w") as f:
        json.dump({"layer": LAYER, "n": Np, "n_train": N_TRAIN, "k": K, "alpha_full": alpha_full,
                   "a_small_frac": A_SMALL_FRAC, "steer_vec_norm": float(v.norm()),
                   "clean_norm": float(clean_norm), "cols": cols, "rows": rows,
                   "method": "in-context forward hook; diff-of-means steering; kNN manifold repair vs alpha-shrink",
                   "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)

    print("\n=== STEERING repair: effect E vs off-target C & validity (per method) ===", flush=True)
    hdr = f"{'method':30s}" + "".join(f"{c:>13}" for c in cols)
    print(hdr, flush=True)
    for r in rows:
        print(f"{r['method']:30s}" + "".join(f"{r[c]:>13.3f}" for c in cols), flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
