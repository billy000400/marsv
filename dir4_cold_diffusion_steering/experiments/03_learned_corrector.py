"""Iteration 3 — LEARNED, downstream-supervised projection-preserving corrector.

Exp 2 showed a statistical manifold prior (Gaussian Mahalanobis) is the wrong training
target: the analytic corrector lowered D_M but WORSENED the LM. The obvious fix, per that
result, is to train the correction against the DOWNSTREAM LM loss directly. This script
does exactly that and asks the decisive question:

    At matched steering projection α|v| along v, can ANY orthogonal correction
    r (optimized end-to-end through the frozen upper LM) beat raw steering z=h+αv
    on ΔLM loss?

Parameterization (projection-preserving by construction):
    ĥ = z + P_{v⊥} r_θ(h, z, α),      z = h + α v
so ⟨ĥ - h, v̂⟩ = α|v| exactly — the corrector can only move orthogonal to v, it can
never erase the steer (unlike the naive-inversion control).

Training: r_θ is a 4-layer MLP. For each batch we patch ĥ into resid_post at block 6 and
run the FROZEN upper GPT-2 (blocks 7-11 + ln_f + lm_head), backprop the real next-token
cross-entropy into r_θ only (h detached; model weights frozen). α ~ U(0.5, 8) per step.
Loss = LM_CE(ĥ) + λ_near · ⟨‖P_{v⊥}r‖²⟩ (a light minimal-correction regularizer).

Eval (held-out docs) compares, at matched projection, ΔLM for raw / analytic cov_corr /
learned corrector, plus projection retention and D_M. Model: GPT-2 small, resid_post block 6.
Outputs: results/03_learned_corrector.json + plots/03_learned_corrector.png.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import resid_post, fineweb_texts, load_model, DATA, DEVICE
from projections import cov_aligned_shift, normalize_vector

LAYER = 6
HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------- Gaussian manifold stats (shared with exp 1/2) ----------
def gaussian_stats(H):
    mu = H.mean(0)
    Xc = H - mu
    cov = (Xc.T @ Xc) / (len(H) - 1)
    cov += 1e-3 * np.eye(cov.shape[0], dtype=cov.dtype)
    prec = np.linalg.inv(cov).astype(np.float32)
    return mu.astype(np.float32), cov.astype(np.float32), prec


def mahalanobis(H, mu, prec):
    Xc = H - mu
    return np.sqrt(np.einsum("nd,dk,nk->n", Xc, prec, Xc))


# ---------- corrector ----------
class Corrector(nn.Module):
    """r_θ(h, z, α): 4-layer MLP. Inputs scaled to O(1). Last layer zero-init so the
    corrector starts as the identity (ĥ = z = raw steering) and improves from there."""
    def __init__(self, d=768, hidden=1024, n_layers=4):
        super().__init__()
        dims = [2 * d + 1] + [hidden] * (n_layers - 1) + [d]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, z, alpha):
        a = torch.full_like(h[..., :1], alpha / 8.0)
        x = torch.cat([h / 100.0, z / 100.0, a], dim=-1)
        return self.net(x)


def make_hat(corr, h, alpha, vt, vhat_t):
    """ĥ = z + P_{v⊥} r_θ(h, z, α). h may require or not require grad; steering fixed."""
    z = h + alpha * vt
    r = corr(h, z, alpha)
    r_perp = r - (r @ vhat_t).unsqueeze(-1) * vhat_t
    return z + r_perp, r_perp


# hook that swaps resid_post at LAYER for fn(out0)
class FuncPatcher:
    def __init__(self, model, layer, fn):
        self.block = model.transformer.h[layer]; self.fn = fn; self.handle = None
    def _hook(self, module, inp, out):
        if isinstance(out, tuple):
            return (self.fn(out[0]),) + out[1:]
        return self.fn(out)
    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook); return self
    def __exit__(self, *a):
        self.handle.remove()


def batched_ids(tok, texts, seq_len, batch):
    """Yield (ids, mask) tensors for tokenized text batches."""
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                  max_length=seq_len, padding="max_length")
        yield enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)


@torch.no_grad()
def lm_loss_fn(texts, layer, fn, seq_len=128, batch=16):
    model, tok = load_model()
    total, count = 0.0, 0
    for ids, mask in batched_ids(tok, texts, seq_len, batch):
        with FuncPatcher(model, layer, fn):
            logits = model(ids, attention_mask=mask).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]; m = mask[:, 1:].bool()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total += nll[m].sum().item(); count += m.sum().item()
    return total / count


def train_corrector(corr, vt, vhat_t, train_texts, seq_len=64, batch=8,
                    epochs=6, lr=1e-3, lam_near=0.05, amin=0.5, amax=8.0):
    model, tok = load_model()
    opt = torch.optim.Adam(corr.parameters(), lr=lr)
    rng = np.random.RandomState(SEED)
    corr.train()
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(train_texts))
        texts = [train_texts[i] for i in order]
        for ids, mask in batched_ids(tok, texts, seq_len, batch):
            alpha = float(rng.uniform(amin, amax))
            store = {}

            def fn(h):
                h = h.detach()                       # cut grad into lower blocks
                hat, r_perp = make_hat(corr, h, alpha, vt, vhat_t)
                store["rp"] = r_perp
                return hat

            with FuncPatcher(model, LAYER, fn):
                logits = model(ids, attention_mask=mask).logits
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tgt = ids[:, 1:]; m = mask[:, 1:].bool()
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lm = nll[m].mean()
            near = (store["rp"] / 100.0).pow(2).sum(-1).mean()
            loss = lm + lam_near * near
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 25 == 0:
                print(f"  ep{ep} step{step:4d} a={alpha:4.1f} LM={lm.item():.3f} "
                      f"near={near.item():.3f}", flush=True)
    corr.eval()
    return corr


@torch.no_grad()
def corrector_acts(corr, Hs, alpha, vt, vhat_t, batch=4096):
    """Apply the learned corrector to a numpy sample of clean acts -> numpy ĥ."""
    out = []
    for i in range(0, len(Hs), batch):
        h = torch.tensor(Hs[i:i + batch], device=DEVICE)
        hat, _ = make_hat(corr, h, alpha, vt, vhat_t)
        out.append(hat.cpu().numpy())
    return np.concatenate(out, 0)


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)                      # freeze LM
    print("device:", DEVICE, flush=True)

    d = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))
    v = d["v"].astype(np.float32)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE)
    vhat_t = torch.tensor(vhat, device=DEVICE)
    print(f"|v| = {vnorm:.2f}", flush=True)

    # clean Gaussian (fit set), disjoint from train/eval
    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=16)
    mu, cov, prec = gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(mahalanobis(Hs, mu, prec).mean())
    print(f"clean D_M={dm_clean:.2f} mean|h|={hnorm:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]                 # 300 docs, disjoint from eval/fit
    eval_texts = all_texts[400:500]                  # same held-out set as exp 2

    # ---- train ----
    corr = Corrector().to(DEVICE)
    print(f"corrector params: {sum(p.numel() for p in corr.parameters())/1e6:.2f}M", flush=True)
    train_corrector(corr, vt, vhat_t, train_texts)

    # ---- eval ----
    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]
    methods = ["raw", "cov_corr", "learned"]
    out = {m: {"dm": [], "dlm": [], "retention": []} for m in methods}

    clean_loss = lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    for a in alphas:
        delta_cov = cov_aligned_shift(v, cov, a).astype(np.float32)
        delta_cov_t = torch.tensor(delta_cov, device=DEVICE)

        def f_raw(h):  return h + a * vt
        def f_cov(h):  return h + delta_cov_t
        def f_learn(h):
            hat, _ = make_hat(corr, h, a, vt, vhat_t); return hat
        fns = {"raw": f_raw, "cov_corr": f_cov, "learned": f_learn}

        Z = Hs + a * v[None, :]
        Zc = Hs + delta_cov[None, :]
        Zl = corrector_acts(corr, Hs, a, vt, vhat_t)
        acts = {"raw": Z, "cov_corr": Zc, "learned": Zl}

        for m in methods:
            out[m]["dm"].append(float(mahalanobis(acts[m], mu, prec).mean()))
            out[m]["retention"].append(float(((acts[m] - Hs) @ vhat).mean()))
            out[m]["dlm"].append(lm_loss_fn(eval_texts, LAYER, fns[m]) - clean_loss)
        print(f"alpha={a:>4}  ΔLM raw={out['raw']['dlm'][-1]:+.3f} "
              f"cov={out['cov_corr']['dlm'][-1]:+.3f} learned={out['learned']['dlm'][-1]:+.3f}"
              f"  | D_M raw={out['raw']['dm'][-1]:.1f} learned={out['learned']['dm'][-1]:.1f}",
              flush=True)

    res = dict(layer=LAYER, v_norm=vnorm, mean_h_norm=hnorm, dm_clean=dm_clean,
               clean_loss=clean_loss, alphas=alphas, methods=methods, results=out,
               n_fit_tokens=int(len(H)), n_train_texts=len(train_texts),
               n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "03_learned_corrector.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    C = {"raw": "#c0392b", "cov_corr": "#27ae60", "learned": "#2980b9"}
    L = {"raw": "raw steer  h+αv", "cov_corr": "analytic cov-aligned",
         "learned": "learned corrector (LM-supervised)"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for m in methods:
        ax[0].plot(alphas, out[m]["dlm"], "o-", color=C[m], label=L[m])
    ax[0].axhline(0, ls="--", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) LM degradation (lower is better)"); ax[0].legend(fontsize=8)
    for m in methods:
        ax[1].plot(alphas, out[m]["dm"], "o-", color=C[m], label=L[m])
    ax[1].axhline(dm_clean, ls="--", color="k", lw=1, label=f"real acts ({dm_clean:.1f})")
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("mean Mahalanobis $D_M$")
    ax[1].set_title("(b) off-manifold distance"); ax[1].legend(fontsize=8)
    for m in methods:
        ax[2].plot(alphas, out[m]["retention"], "o-", color=C[m], label=L[m])
    ax[2].set_xlabel(r"steering strength $\alpha$")
    ax[2].set_ylabel(r"projection retention $\langle \hat h - h,\ \hat v\rangle$")
    ax[2].set_title("(c) steering projection (all matched)"); ax[2].legend(fontsize=8)
    fig.suptitle("Learned downstream-supervised corrector vs raw steering (GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "03_learned_corrector.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
