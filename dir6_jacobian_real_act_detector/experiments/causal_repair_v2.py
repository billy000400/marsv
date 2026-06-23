"""D6 Phase 6 (CORRECTED) — genuinely IN-CONTEXT causal repair (claim 4).

Fixes the same single-position bug as context_validation_v2 (codex finding #1): the continuation is
now a FULL model forward with a forward hook that overwrites ONLY the last-token resid_post@LAYER
with the candidate activation, so later-layer attention attends to the real prompt context. Gradients
for the differentiable realness scores flow back through the late blocks to the injected residual.

Also addresses codex finding #2 (func_descent control not move-matched): we report a random control
matched to the maha-descent move budget AND a separate random control matched to the func-descent
move budget, so each descent is compared at its OWN matched L2 distance.

Repairs (start from corrupted, Gaussian noise s=1, norm-matched):
  - maha_descent : Adam descent on Mahalanobis distance to the train distribution (no model needed).
  - func_descent : Adam descent on in-context plateau-KL (output sensitivity), via the model.
Controls (matched L2 move):
  - shrink_mean / shrink_clean(oracle) / random_move  matched to maha budget;
  - random_move(func-matched)                         matched to func budget.
External objective-free metrics: KL(clean ‖ x) and NLL of clean argmax, both in-context.
Claim 4 holds only if a realness-improving repair beats its matched random control.
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
LAYER = 6; SEQ = 64; N_PROMPTS = 300; N_TRAIN = 30_000; SHRINK = 0.05
NOISE_S = 1.0; STEPS = 40; LR = 0.5; EPS_PLATEAU = 0.02; N_PLATEAU = 4; BS = 60
rng = np.random.default_rng(0); torch.manual_seed(0)


def main():
    t0 = time.time()
    a = np.load(os.path.join(SRC, f"acts_layer{LAYER}.npy"), mmap_mode="r"); D = a.shape[1]
    train = np.asarray(a[:N_TRAIN], dtype=np.float32)
    mu = train.mean(0); Xc = train - mu; cov = (Xc.T @ Xc) / (len(train) - 1)
    cov_s = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + 1e-3 * np.eye(D, dtype=np.float32)
    cov_inv = np.linalg.inv(cov_s).astype(np.float32)
    muD = torch.from_numpy(mu).to(DEVICE); cinvD = torch.from_numpy(cov_inv).to(DEVICE)

    m = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    with open(TXT) as f:
        texts = [ln.strip() for ln in f if len(ln.strip()) > 200][:N_PROMPTS]
    block = m.transformer.h[LAYER]

    def cont(rep, ids_b, mask_b, last_b, grad):
        """Full in-context forward; overwrite last-token resid@LAYER with rep -> last-pos logits."""
        lb = last_b
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone(); h[torch.arange(h.shape[0], device=h.device), lb] = rep.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handle = block.register_forward_hook(hook)
        try:
            if grad:
                logits = m(input_ids=ids_b, attention_mask=mask_b).logits
            else:
                with torch.no_grad():
                    logits = m(input_ids=ids_b, attention_mask=mask_b).logits
        finally:
            handle.remove()
        return logits[torch.arange(logits.shape[0], device=logits.device), lb]

    # tokenize + capture clean last-pos activations
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
    clean = np.concatenate(clean_all, 0).astype(np.float32); Np = len(clean)
    print(f"[{time.time()-t0:.0f}s] {Np} clean acts (in-context capture)", flush=True)

    def maha_of(x):
        d = x - muD; return torch.einsum("ij,jk,ik->i", d, cinvD, d)

    agg = {}  # method -> list of (kl, nll, move, d_clean, d_mean) arrays per batch
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

        lp_clean = torch.log_softmax(cont(clD, ib, mb, lb, grad=False), -1)
        clean_argmax = lp_clean.argmax(-1)
        p_clean = lp_clean.exp()

        def ext(xD):
            lp = torch.log_softmax(cont(xD, ib, mb, lb, grad=False), -1)
            kl = (p_clean * (lp_clean - lp)).sum(-1)
            nll = -lp.gather(1, clean_argmax[:, None]).squeeze(1)
            return kl.cpu().numpy(), nll.cpu().numpy()

        # ---- maha descent (no model) ----
        x = corrD.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=LR)
        for _ in range(STEPS):
            opt.zero_grad(); maha_of(x).sum().backward(); opt.step()
        rep_maha = x.detach()

        # ---- func descent (in-context plateau-KL) ----
        x = corrD.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=LR)
        for _ in range(STEPS):
            opt.zero_grad()
            lp = torch.log_softmax(cont(x, ib, mb, lb, grad=True), -1); pdet = lp.exp().detach()
            loss = 0.0
            for _ in range(N_PLATEAU):
                nz = torch.randn_like(x); nz = nz / nz.norm(dim=1, keepdim=True) * x.norm(dim=1, keepdim=True) * EPS_PLATEAU
                lp_p = torch.log_softmax(cont(x + nz, ib, mb, lb, grad=True), -1)
                loss = loss + (pdet * (lp - lp_p)).sum(-1).sum()
            (loss / N_PLATEAU).backward(); opt.step()
        rep_func = x.detach()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        budget = (rep_maha - corrD).norm(dim=1, keepdim=True)
        budget_f = (rep_func - corrD).norm(dim=1, keepdim=True)
        def matched(direction, bud):
            u = direction / (direction.norm(dim=1, keepdim=True) + 1e-8); return corrD + bud * u
        g = torch.randn_like(corrD); gf = torch.randn_like(corrD)
        methods = {
            "corrupted(start)": corrD, "maha_descent": rep_maha, "func_descent": rep_func,
            "shrink_mean(matched)": matched(muD[None, :] - corrD, budget),
            "shrink_clean(oracle,matched)": matched(clD - corrD, budget),
            "random_move(maha-matched)": matched(g, budget),
            "random_move(func-matched)": matched(gf, budget_f),
        }
        for name, xD in methods.items():
            kl, nll = ext(xD)
            move = (xD - corrD).norm(dim=1).cpu().numpy()
            dcl = (xD - clD).norm(dim=1).cpu().numpy(); dmn = (xD - muD[None, :]).norm(dim=1).cpu().numpy()
            add(name, kl, nll, move, dcl, dmn)
        print(f"[{time.time()-t0:.0f}s] batch {bi+1}/{len(ids_all)} done (move maha {budget.mean():.1f} func {budget_f.mean():.1f})", flush=True)

    rows = []
    for name, parts in agg.items():
        M = np.concatenate(parts, 0)
        rows.append({"method": name, "ext_KL_clean": round(float(M[:, 0].mean()), 4),
                     "ext_NLL_cleanargmax": round(float(M[:, 1].mean()), 4),
                     "move_from_corrupt": round(float(M[:, 2].mean()), 2),
                     "dist_to_clean": round(float(M[:, 3].mean()), 2),
                     "dist_to_mean": round(float(M[:, 4].mean()), 2)})

    with open(os.path.join(RES, "repair_metrics_v2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(RES, "repair_v2_summary.json"), "w") as f:
        json.dump({"layer": LAYER, "n": Np, "noise_s": NOISE_S, "steps": STEPS,
                   "method": "in-context full-model forward hook", "rows": rows,
                   "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)

    print("\n=== IN-CONTEXT repair: external KL(clean||x) & NLL clean-argmax (lower=better) ===", flush=True)
    print(f"{'method':32s}{'extKL':>9}{'extNLL':>9}{'move':>8}{'d_clean':>9}{'d_mean':>9}", flush=True)
    for r in rows:
        print(f"{r['method']:32s}{r['ext_KL_clean']:>9}{r['ext_NLL_cleanargmax']:>9}{r['move_from_corrupt']:>8}{r['dist_to_clean']:>9}{r['dist_to_mean']:>9}", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
