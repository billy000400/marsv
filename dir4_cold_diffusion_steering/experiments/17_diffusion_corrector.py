"""Iteration 17 — A REAL diffusion / iterative corrector (acts on human feedback #1).

The central critique of this direction ("Cold-Diffusion Steering") is that the flagship
corrector is a ONE-SHOT MLP, not the iterative diffusion model of the GLP arxiv paper. This
script builds the actual diffusion-style machinery and compares three correctors at MATCHED
steering projection α|v| on the SAME held-out eval set (GPT-2 small, resid_post block 6,
DiffMean sentiment vector), reusing the Exp-3 pipeline verbatim:

  (1) ONE-SHOT MLP           — the incumbent (Exp 3): ĥ = z + P_{v⊥} r_θ(h,z,α), 1 forward.
  (2) COLD-DIFFUSION ITER    — NEW. The SAME-capacity network, weight-shared, applied over
        K correction steps with a step embedding: x_K=z; x_{k-1}=x_k+(1/K)P_{v⊥}g_θ(h,x,α,t_k);
        ĥ=x_0. Projection along v is preserved at EVERY step by construction (each increment ⟂v),
        so it can never erase the steer. Trained by UNROLLING the K steps and backpropping the
        frozen upper-LM next-token CE into g_θ (the iterative analogue of Exp 3). This is a
        genuine flow-matching-style iterative corrector: a time-conditioned velocity field
        integrated to transport z to an LM-safe activation.
  (3) GLP GAUSSIAN PRIOR     — NEW baseline = the "generic Gaussian-noise GLP teacher" the
        proposal names. A real DDPM (cosine schedule, ε-prediction) trained on CLEAN standardized
        layer-6 activations with GAUSSIAN-noise corruption (NOT steering), pure MSE, no LM in the
        loop. To correct z we run SDEdit-style post-processing (noise z to t_start, DDIM-denoise
        back to 0), which projects z toward the learned clean-activation manifold. Because the
        prior is unconditional it PARTIALLY ERASES the steer (measured: as-is projection
        retention); for the matched-ΔLM comparison we re-impose the target projection α|v| along v̂.

Research questions answered:
  RQ1  Does steering-corruption training beat generic Gaussian-noise GLP denoising? (2 vs 3)
  RQ2  Does the iterative diffusion structure add anything over the one-shot MLP? (2 vs 1)
  RQ3  Does the unconditional GLP prior erase the steering projection? (3, as-is retention)

Outputs: results/17_diffusion_corrector.json + plots/17_diffusion_corrector.png.
"""
import os, json, math, importlib.util
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import resid_post, fineweb_texts, load_model, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(PLOTS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

exp03 = _load("exp03", "03_learned_corrector.py")
Corrector      = exp03.Corrector
train_corrector= exp03.train_corrector
make_hat       = exp03.make_hat
corrector_acts = exp03.corrector_acts
lm_loss_fn     = exp03.lm_loss_fn
gaussian_stats = exp03.gaussian_stats
mahalanobis    = exp03.mahalanobis
FuncPatcher    = exp03.FuncPatcher
batched_ids    = exp03.batched_ids
LAYER          = exp03.LAYER


# =====================================================================================
# (2) Cold-Diffusion iterative corrector — same-capacity net, weight-shared over K steps
# =====================================================================================
class IterCorrector(nn.Module):
    """g_θ(h, x, α, t): 4-layer MLP; same capacity as the one-shot Corrector plus a scalar
    step input t∈[0,1). Last layer zero-init so each step starts as the identity."""
    def __init__(self, d=768, hidden=1024, n_layers=4):
        super().__init__()
        dims = [2 * d + 2] + [hidden] * (n_layers - 1) + [d]   # inputs: h, x, α, t
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, x, alpha, t_frac):
        a = torch.full_like(x[..., :1], alpha / 8.0)
        t = torch.full_like(x[..., :1], t_frac)
        inp = torch.cat([h / 100.0, x / 100.0, a, t], dim=-1)
        return self.net(inp)


def make_hat_iter(g, h, alpha, vt, vhat_t, K):
    """Iterative projection-preserving correction. Returns (ĥ, net orthogonal correction)."""
    z = h + alpha * vt
    x = z
    for k in range(K):
        r = g(h, x, alpha, k / K)
        r_perp = r - (r @ vhat_t).unsqueeze(-1) * vhat_t
        x = x + (1.0 / K) * r_perp
    return x, (x - z)


def train_iter(g, vt, vhat_t, train_texts, K, seq_len=64, batch=8,
               epochs=6, lr=1e-3, lam_near=0.05):
    model, tok = load_model()
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    rng = np.random.RandomState(SEED)
    g.train(); step = 0
    for ep in range(epochs):
        order = rng.permutation(len(train_texts))
        texts = [train_texts[i] for i in order]
        for ids, mask in batched_ids(tok, texts, seq_len, batch):
            alpha = float(rng.uniform(0.5, 8.0)); store = {}

            def fn(h):
                h = h.detach()
                hat, rp = make_hat_iter(g, h, alpha, vt, vhat_t, K)
                store["rp"] = rp; return hat

            with FuncPatcher(model, LAYER, fn):
                logits = model(ids, attention_mask=mask).logits
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tgt = ids[:, 1:]; m = mask[:, 1:].bool()
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lm = nll[m].mean()
            near = (store["rp"] / 100.0).pow(2).sum(-1).mean()
            loss = lm + lam_near * near
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if step % 25 == 0:
                print(f"  [iter K={K}] ep{ep} step{step:4d} a={alpha:4.1f} "
                      f"LM={lm.item():.3f} near={near.item():.3f}", flush=True)
    g.eval(); return g


@torch.no_grad()
def iter_acts(g, Hs, alpha, vt, vhat_t, K, batch=4096):
    out = []
    for i in range(0, len(Hs), batch):
        h = torch.tensor(Hs[i:i + batch], device=DEVICE)
        hat, _ = make_hat_iter(g, h, alpha, vt, vhat_t, K)
        out.append(hat.cpu().numpy())
    return np.concatenate(out, 0)


# =====================================================================================
# (3) GLP-style Gaussian-noise diffusion PRIOR over clean activations (real DDPM)
# =====================================================================================
def abar(t):
    """Cosine noise schedule ᾱ(t), t∈[0,1] (Nichol & Dhariwal 2021). t may be tensor/float."""
    s = 0.008
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=DEVICE)
    f = lambda u: torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
    return f(t) / f(torch.zeros((), device=t.device))


class TimeEmb(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.dim = dim
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000.0), dim // 2)))

    def forward(self, t):                     # t: [...,1] in [0,1]
        args = t * self.freqs                 # broadcast -> [..., dim/2]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class GLPDenoiser(nn.Module):
    """ε-prediction network over STANDARDIZED clean activations. Generic Gaussian diffusion."""
    def __init__(self, d=768, hidden=1024, temb=64):
        super().__init__()
        self.temb = TimeEmb(temb)
        dims = [d + temb, hidden, hidden, d]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):                  # x:[...,d]  t: scalar or [...,1]
        if not torch.is_tensor(t) or t.dim() == 0:
            t = torch.full_like(x[..., :1], float(t))
        e = self.temb(t)
        return self.net(torch.cat([x, e], dim=-1))


def train_glp(net, H_std, epochs=40, batch=512, lr=2e-4):
    """Train the DDPM ε-predictor on standardized clean activations. No LM in the loop."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X = torch.tensor(H_std, device=DEVICE)
    n = len(X); rng = torch.Generator(device="cpu"); rng.manual_seed(SEED)
    net.train(); step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=rng).to(DEVICE)
        for i in range(0, n, batch):
            xb = X[perm[i:i + batch]]
            t = torch.rand(len(xb), 1, device=DEVICE)
            ab = abar(t)
            eps = torch.randn(xb.shape, device=DEVICE)
            xt = torch.sqrt(ab) * xb + torch.sqrt(1 - ab) * eps
            pred = net(xt, t)
            loss = (pred - eps).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if ep % 5 == 0:
            print(f"  [glp] ep{ep} mse={loss.item():.4f}", flush=True)
    net.eval(); return net


@torch.no_grad()
def glp_correct(net, z, mu_t, sd_t, t0, n_steps, gen):
    """SDEdit-style GLP post-processing of a steered activation z toward the clean manifold.
    Standardize, forward-noise to t0, DDIM-denoise to 0, destandardize. Works on any leading
    shape (per-vector network). Returns corrected activation (same shape as z)."""
    zs = (z - mu_t) / sd_t
    ab0 = abar(t0)
    eps = torch.randn(zs.shape, device=zs.device, generator=gen)
    x = torch.sqrt(ab0) * zs + torch.sqrt(1 - ab0) * eps
    grid = torch.linspace(t0, 0.0, n_steps + 1, device=zs.device)
    for i in range(n_steps):
        tc, tn = grid[i], grid[i + 1]
        ab_c, ab_n = abar(tc), abar(tn)
        e = net(x, tc)
        x0 = (x - torch.sqrt(1 - ab_c) * e) / torch.sqrt(ab_c)
        x = torch.sqrt(ab_n) * x0 + torch.sqrt(1 - ab_n) * e
    return x * sd_t + mu_t


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    d = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))
    v = d["v"].astype(np.float32)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)
    print(f"|v| = {vnorm:.2f}", flush=True)

    # clean Gaussian + standardization stats (fit set), disjoint from train/eval
    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=16)
    mu, cov, prec = gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(mahalanobis(Hs, mu, prec).mean())
    mu_d = H.mean(0).astype(np.float32)
    sd_d = (H.std(0) + 1e-6).astype(np.float32)
    mu_t = torch.tensor(mu_d, device=DEVICE); sd_t = torch.tensor(sd_d, device=DEVICE)
    print(f"clean D_M={dm_clean:.2f} mean|h|={hnorm:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]     # 300 docs (= Exp 3)
    eval_texts  = all_texts[400:500]     # held-out 100 (= Exp 3)

    K = 8
    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]

    # ---- train (1) one-shot MLP (Exp 3 incumbent, retrained here for a matched eval) ----
    print("== train one-shot MLP ==", flush=True)
    oneshot = Corrector().to(DEVICE)
    train_corrector(oneshot, vt, vhat_t, train_texts)

    # ---- train (2) cold-diffusion iterative corrector ----
    print(f"== train cold-diffusion iterative corrector (K={K}) ==", flush=True)
    g = IterCorrector().to(DEVICE)
    train_iter(g, vt, vhat_t, train_texts, K)

    # ---- train (3) GLP Gaussian-noise diffusion prior (no LM) ----
    print("== train GLP Gaussian diffusion prior ==", flush=True)
    H_std = (Hs - mu_d[None, :]) / sd_d[None, :]
    glp = GLPDenoiser().to(DEVICE)
    train_glp(glp, H_std)
    p_one = sum(p.numel() for p in oneshot.parameters()) / 1e6
    p_it  = sum(p.numel() for p in g.parameters()) / 1e6
    p_glp = sum(p.numel() for p in glp.parameters()) / 1e6
    print(f"params: one-shot {p_one:.2f}M  iter {p_it:.2f}M  glp {p_glp:.2f}M", flush=True)

    # ---- pick GLP SDEdit t_start by steelmanning it (lowest ΔLM at α=4,8) ----
    clean_loss = lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)
    N_DDIM = 20

    def glp_fn(a, t0, reimpose=True):
        def fn(h):
            gen = torch.Generator(device=DEVICE); gen.manual_seed(SEED)
            z = h + a * vt
            hg = glp_correct(glp, z, mu_t, sd_t, t0, N_DDIM, gen)
            if reimpose:
                proj = ((hg - h) @ vhat_t).unsqueeze(-1)
                hg = hg + (a * vnorm - proj) * vhat_t
            return hg
        return fn

    t0_grid = [0.15, 0.25, 0.40]
    t0_score = {}
    for t0 in t0_grid:
        s = 0.0
        for a in (4.0, 8.0):
            s += lm_loss_fn(eval_texts, LAYER, glp_fn(a, t0)) - clean_loss
        t0_score[t0] = s / 2
        print(f"  GLP t0={t0} mean ΔLM(α=4,8) = {t0_score[t0]:+.3f}", flush=True)
    t0_best = min(t0_score, key=t0_score.get)
    print(f"GLP best t_start = {t0_best}", flush=True)

    # ---- full matched-projection eval ----
    methods = ["raw", "oneshot", "iter", "glp"]
    out = {m: {"dm": [], "dlm": [], "retention": []} for m in methods}
    glp_asis_ret = []       # GLP projection retention WITHOUT re-imposing (erasure test)

    @torch.no_grad()
    def glp_acts(Hs_np, a, t0, reimpose=True):
        gen = torch.Generator(device=DEVICE); gen.manual_seed(SEED)
        res = []
        for i in range(0, len(Hs_np), 4096):
            h = torch.tensor(Hs_np[i:i + 4096], device=DEVICE)
            z = h + a * vt
            hg = glp_correct(glp, z, mu_t, sd_t, t0, N_DDIM, gen)
            if reimpose:
                proj = ((hg - h) @ vhat_t).unsqueeze(-1)
                hg = hg + (a * vnorm - proj) * vhat_t
            res.append(hg.cpu().numpy())
        return np.concatenate(res, 0)

    for a in alphas:
        def f_raw(h):  return h + a * vt
        def f_one(h):  hat, _ = make_hat(oneshot, h, a, vt, vhat_t); return hat
        def f_it(h):   hat, _ = make_hat_iter(g, h, a, vt, vhat_t, K); return hat
        fns = {"raw": f_raw, "oneshot": f_one, "iter": f_it, "glp": glp_fn(a, t0_best)}

        Z  = Hs + a * v[None, :]
        Zo = corrector_acts(oneshot, Hs, a, vt, vhat_t)
        Zi = iter_acts(g, Hs, a, vt, vhat_t, K)
        Zg = glp_acts(Hs, a, t0_best, reimpose=True)
        Zg_asis = glp_acts(Hs, a, t0_best, reimpose=False)
        acts = {"raw": Z, "oneshot": Zo, "iter": Zi, "glp": Zg}

        glp_asis_ret.append(float(((Zg_asis - Hs) @ vhat).mean()))
        for m in methods:
            out[m]["dm"].append(float(mahalanobis(acts[m], mu, prec).mean()))
            out[m]["retention"].append(float(((acts[m] - Hs) @ vhat).mean()))
            out[m]["dlm"].append(lm_loss_fn(eval_texts, LAYER, fns[m]) - clean_loss)
        print(f"alpha={a:>4} ΔLM raw={out['raw']['dlm'][-1]:+.3f} "
              f"one={out['oneshot']['dlm'][-1]:+.3f} iter={out['iter']['dlm'][-1]:+.3f} "
              f"glp={out['glp']['dlm'][-1]:+.3f} | glp as-is ret={glp_asis_ret[-1]:.1f}"
              f" (target {a*vnorm:.1f})", flush=True)

    # recovery fractions (vs raw)
    def rec(m):
        return [100.0 * (1 - out[m]["dlm"][i] / out["raw"]["dlm"][i]) if out["raw"]["dlm"][i] > 1e-3
                else float("nan") for i in range(len(alphas))]
    recov = {m: rec(m) for m in ["oneshot", "iter", "glp"]}

    res = dict(layer=LAYER, v_norm=vnorm, mean_h_norm=hnorm, dm_clean=dm_clean,
               clean_loss=clean_loss, alphas=alphas, K=K, n_ddim=N_DDIM,
               t0_grid=t0_grid, t0_score={str(k): v for k, v in t0_score.items()},
               t0_best=t0_best, params=dict(oneshot=p_one, iter=p_it, glp=p_glp),
               methods=methods, results=out, glp_asis_retention=glp_asis_ret,
               glp_target_projection=[a * vnorm for a in alphas], recovery=recov)
    with open(os.path.join(RESULTS, "17_diffusion_corrector.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    C = {"raw": "#c0392b", "oneshot": "#7f8c8d", "iter": "#2980b9", "glp": "#8e44ad"}
    L = {"raw": "raw steer  h+αv", "oneshot": "one-shot MLP (Exp 3)",
         "iter": f"cold-diffusion iterative (K={K})", "glp": "GLP Gaussian prior (SDEdit)"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for m in methods:
        ax[0].plot(alphas, out[m]["dlm"], "o-", color=C[m], label=L[m])
    ax[0].axhline(0, ls="--", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) LM degradation, matched projection (lower better)"); ax[0].legend(fontsize=8)

    for m in ["oneshot", "iter", "glp"]:
        ax[1].plot(alphas, recov[m], "o-", color=C[m], label=L[m])
    ax[1].axhline(0, ls="--", color="k", lw=1); ax[1].set_ylim(-30, 110)
    ax[1].set_xlabel(r"steering strength $\alpha$")
    ax[1].set_ylabel("fluency recovery vs raw (%)")
    ax[1].set_title("(b) recovery of raw steering's damage"); ax[1].legend(fontsize=8)

    tgt = [a * vnorm for a in alphas]
    ax[2].plot(alphas, tgt, "k--", lw=1.3, label="target α|v| (matched)")
    ax[2].plot(alphas, glp_asis_ret, "o-", color=C["glp"], label="GLP prior, as-is")
    ax[2].plot(alphas, out["iter"]["retention"], "s-", color=C["iter"],
               label="cold-diffusion iter (preserved)")
    ax[2].set_xlabel(r"steering strength $\alpha$")
    ax[2].set_ylabel(r"steering projection $\langle \hat h-h,\ \hat v\rangle$")
    ax[2].set_title("(c) GLP prior erases the steer; iterative preserves it"); ax[2].legend(fontsize=8)

    fig.suptitle("A real diffusion corrector: cold-diffusion (steering corruption) vs one-shot MLP "
                 "vs GLP Gaussian prior (GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "17_diffusion_corrector.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
