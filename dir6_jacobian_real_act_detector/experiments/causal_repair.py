"""D6 Phase 6 — CAUSAL repair (claim 4). Does improving a realness score causally recover downstream
behavior better than matched-distance controls?

Setup (reuses Phase-5 in-context machinery): take real last-position resid_post of 300 prompts,
corrupt with Gaussian noise (s=1.0, norm-matched). The TRUE clean continuation logits are known, so
the external metric is KL(clean ‖ repaired) — and we ALSO report next-token-loss-vs-clean-argmax,
a metric NOT in any objective (reward-hacking check).

Repair methods (all start from the corrupted activation, all reported at MATCHED L2 move budget):
  - maha_descent : gradient descent to reduce Mahalanobis distance to the train distribution
                   (differentiable realness score), step until move budget reached.
  - func_descent : gradient descent to reduce plateau-KL functional anomaly (output sensitivity).
Controls (moved the SAME L2 distance as maha_descent, per-activation):
  - shrink_mean  : move toward the global train mean (matched distance).
  - shrink_clean : move toward the TRUE clean activation (oracle upper bound, matched distance).
  - random_move  : random direction (matched distance).
Claim 4 holds if a realness-improving repair lowers external KL MORE than random/shrink_mean at the
same move distance (and ideally approaches the shrink_clean oracle) without just shrinking to mean.
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
NOISE_S = 1.0; STEPS = 40; LR = 0.5; EPS_PLATEAU = 0.02; N_PLATEAU = 4
rng = np.random.default_rng(0); torch.manual_seed(0)


def main():
    t0 = time.time()
    a = np.load(os.path.join(SRC, f"acts_layer{LAYER}.npy"), mmap_mode="r"); D = a.shape[1]
    train = np.asarray(a[:N_TRAIN], dtype=np.float32)
    mu = train.mean(0); Xc = train - mu; cov = (Xc.T @ Xc) / (len(train) - 1)
    cov_s = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov)) + 1e-3 * np.eye(D, dtype=np.float32)
    cov_inv = np.linalg.inv(cov_s).astype(np.float32)
    muD = torch.from_numpy(mu).to(DEVICE)
    cinvD = torch.from_numpy(cov_inv).to(DEVICE)

    m = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    with open(TXT) as f:
        texts = [ln.strip() for ln in f if len(ln.strip()) > 200][:N_PROMPTS]
    blocks = m.transformer.h[LAYER + 1:]; ln_f = m.transformer.ln_f; head = m.lm_head

    def continue_from(x):  # x [B,768] (grad ok) -> logits [B,V]
        h = x.unsqueeze(1)
        for blk in blocks:
            r = blk(h); h = r[0] if isinstance(r, tuple) else r
        return head(ln_f(h)).squeeze(1)

    # clean last-pos activations
    clean = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], return_tensors="pt", truncation=True, max_length=SEQ, padding="max_length")
        ids = enc["input_ids"].to(DEVICE); mask = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hs = out.hidden_states[LAYER + 1]; last = mask.sum(1) - 1
        clean.append(hs[torch.arange(len(ids)), last].float().cpu().numpy())
    clean = np.concatenate(clean, 0).astype(np.float32)
    Np = len(clean)
    cleanD = torch.from_numpy(clean).to(DEVICE)
    with torch.no_grad():
        lp_clean = torch.log_softmax(continue_from(cleanD), -1)
        clean_argmax = lp_clean.argmax(-1)
    print(f"[{time.time()-t0:.0f}s] {Np} clean acts", flush=True)

    # corrupt: noise, norm-matched
    on = np.linalg.norm(clean, axis=1, keepdims=True)
    rms = on / np.sqrt(D)
    corr = clean + NOISE_S * rms * rng.standard_normal((Np, D)).astype(np.float32)
    corr = (corr * (on / np.linalg.norm(corr, axis=1, keepdims=True))).astype(np.float32)
    corrD = torch.from_numpy(corr).to(DEVICE)

    def ext_metrics(xD):
        with torch.no_grad():
            lp = torch.log_softmax(continue_from(xD), -1)
            kl = (lp_clean.exp() * (lp_clean - lp)).sum(-1)          # external KL (objective-free)
            nll = -lp.gather(1, clean_argmax[:, None]).squeeze(1)    # NLL of clean argmax (objective-free)
        return kl.cpu().numpy(), nll.cpu().numpy()

    def maha_of(xD):
        d = xD - muD
        return torch.einsum("ij,jk,ik->i", d, cinvD, d)

    def plateau_of(xD):
        lg = continue_from(xD); lp = torch.log_softmax(lg, -1); p = lp.exp().detach()
        xn = xD.norm(dim=1, keepdim=True); kl = 0.0
        for _ in range(N_PLATEAU):
            nz = torch.randn_like(xD); nz = nz / nz.norm(dim=1, keepdim=True) * xn * EPS_PLATEAU
            kl = kl + (p * (lp - torch.log_softmax(continue_from(xD + nz), -1))).sum(-1)
        return kl / N_PLATEAU

    def descend(score_fn):
        x = corrD.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=LR)
        for _ in range(STEPS):
            opt.zero_grad()
            loss = score_fn(x).sum()
            loss.backward()
            opt.step()
        return x.detach()

    rep_maha = descend(maha_of)
    rep_func = descend(plateau_of)
    # per-activation move budget from maha repair
    budget = (rep_maha - corrD).norm(dim=1, keepdim=True)
    print(f"[{time.time()-t0:.0f}s] descents done; mean move {budget.mean().item():.2f}", flush=True)

    def matched_move(direction):  # direction [B,768] (unnormalized) -> move corr by `budget`
        u = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)
        return corrD + budget * u

    rep_smean = matched_move(muD[None, :] - corrD)
    rep_sclean = matched_move(cleanD - corrD)
    g = torch.randn_like(corrD)
    rep_rand = matched_move(g)
    # also match func repair move budget for fair func comparison
    budget_f = (rep_func - corrD).norm(dim=1, keepdim=True)

    methods = {
        "corrupted(start)": corrD,
        "maha_descent": rep_maha,
        "func_descent": rep_func,
        "shrink_mean(matched)": rep_smean,
        "shrink_clean(oracle,matched)": rep_sclean,
        "random_move(matched)": rep_rand,
    }
    rows = []
    for name, xD in methods.items():
        kl, nll = ext_metrics(xD)
        move = (xD - corrD).norm(dim=1).mean().item()
        d_clean = (xD - cleanD).norm(dim=1).mean().item()
        d_mean = (xD - muD[None, :]).norm(dim=1).mean().item()
        rows.append({"method": name, "ext_KL_clean": round(float(kl.mean()), 4),
                     "ext_NLL_cleanargmax": round(float(nll.mean()), 4),
                     "move_from_corrupt": round(move, 2),
                     "dist_to_clean": round(d_clean, 2), "dist_to_mean": round(d_mean, 2)})

    with open(os.path.join(RES, "repair_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(RES, "repair_summary.json"), "w") as f:
        json.dump({"layer": LAYER, "n": Np, "noise_s": NOISE_S, "steps": STEPS,
                   "func_move_budget_mean": round(float(budget_f.mean()), 2),
                   "rows": rows, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)

    print("\n=== repair: external KL(clean||x) & NLL of clean argmax (lower=better), matched move ===", flush=True)
    print(f"{'method':30s}{'extKL':>9}{'extNLL':>9}{'move':>8}{'d_clean':>9}{'d_mean':>9}", flush=True)
    for r in rows:
        print(f"{r['method']:30s}{r['ext_KL_clean']:>9}{r['ext_NLL_cleanargmax']:>9}{r['move_from_corrupt']:>8}{r['dist_to_clean']:>9}{r['dist_to_mean']:>9}", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
