"""Iteration 2 — analytic projection-preserving corrector vs raw steering.

Tests the ColdSteer success criterion at MATCHED steering projection: does the
projection-preserving corrector  ĥ = z + P_{v⊥} r  reduce off-manifold damage
(Mahalanobis D_M and ΔLM loss) versus raw steering z = h + αv?

Corrector under test (`cov_corr`): the ANALYTIC optimal r for a Gaussian activation
model — the min-whitened-norm constant shift achieving projection α|v| along v,
  Δ = Σ v̂ · α|v| / (v̂^T Σ v̂)   (see projections.cov_aligned_shift).
It preserves the v-projection exactly, so it is a matched-projection comparison to raw.

Baselines / controls:
  raw              : z = h + αv                              (the thing to beat)
  norm_clip        : rescale z per-token to the clean mean norm (does NOT preserve proj.)
  naive_inversion  : ĥ = h  (erase the steer) — negative control; proj. retention = 0.

Model: GPT-2 small (124M), resid_post block 6. Vector: DiffMean sentiment (from iter 1).
Outputs: results/02_corrector.json + plots/02_corrector.png.
"""
import os, json
import numpy as np
import torch
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


class FuncPatcher:
    """Replace resid_post at `layer` by fn(h) for every token during a forward pass."""
    def __init__(self, model, layer, fn):
        self.block = model.transformer.h[layer]
        self.fn = fn
        self.handle = None

    def _hook(self, module, inp, out):
        if isinstance(out, tuple):
            return (self.fn(out[0]),) + out[1:]
        return self.fn(out)

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self.handle.remove()


@torch.no_grad()
def lm_loss_fn(texts, layer, fn, seq_len=128, batch=16):
    """Mean next-token CE (nats) with resid_post at `layer` replaced by fn(h)."""
    model, tok = load_model()
    total, count = 0.0, 0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  max_length=seq_len, padding="max_length")
        ids = enc.input_ids.to(DEVICE)
        mask = enc.attention_mask.to(DEVICE)
        with FuncPatcher(model, layer, fn):
            logits = model(ids, attention_mask=mask).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]
        m = mask[:, 1:].bool()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total += nll[m].sum().item()
        count += m.sum().item()
    return total / count


def main():
    load_model()
    print("device:", DEVICE, flush=True)

    # steering vector from iter 1 (keeps the two experiments consistent)
    d = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))
    v = d["v"].astype(np.float32)
    vnorm = float(np.linalg.norm(v))
    vhat = normalize_vector(v)
    print(f"|v| = {vnorm:.2f}", flush=True)

    # clean activation Gaussian
    fit_texts = fineweb_texts(400)
    H = resid_post(fit_texts, LAYER, seq_len=128, batch=16)
    mu, cov, prec = gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(mahalanobis(Hs, mu, prec).mean())
    print(f"clean D_M = {dm_clean:.2f}, mean|h| = {hnorm:.2f}", flush=True)

    eval_texts = fineweb_texts(500)[400:500]
    alphas = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

    vt = torch.tensor(v, device=DEVICE)
    vhat_t = torch.tensor(vhat, device=DEVICE)

    methods = ["raw", "cov_corr", "norm_clip", "naive_inversion"]
    out = {m: {"dm": [], "dlm": [], "retention": []} for m in methods}

    base_loss = {}  # per-method alpha=0 loss (all equal clean, but keep clean)
    for a in alphas:
        delta_cov = cov_aligned_shift(v, cov, a).astype(np.float32)
        delta_cov_t = torch.tensor(delta_cov, device=DEVICE)

        # ---- numpy activation-level transforms on Hs ----
        Z = Hs + a * v[None, :]
        Zc = Hs + delta_cov[None, :]
        znc = Z * (hnorm / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8))
        acts = {"raw": Z, "cov_corr": Zc, "norm_clip": znc, "naive_inversion": Hs}

        # ---- torch per-token fns for the LM ----
        def f_raw(h):      return h + a * vt
        def f_cov(h):      return h + delta_cov_t
        def f_nc(h):       return (h + a * vt) * (hnorm / (h + a * vt).norm(dim=-1, keepdim=True).clamp_min(1e-8))
        def f_inv(h):      return h
        fns = {"raw": f_raw, "cov_corr": f_cov, "norm_clip": f_nc, "naive_inversion": f_inv}

        for m in methods:
            A = acts[m]
            out[m]["dm"].append(float(mahalanobis(A, mu, prec).mean()))
            out[m]["retention"].append(float(((A - Hs) @ vhat).mean()))
            loss = lm_loss_fn(eval_texts, LAYER, fns[m])
            if a == 0.0:
                base_loss[m] = loss
            out[m]["dlm"].append(loss - base_loss[m])
        print(f"alpha={a:>4}  raw D_M={out['raw']['dm'][-1]:6.2f} dLM={out['raw']['dlm'][-1]:+.3f}"
              f"  | cov D_M={out['cov_corr']['dm'][-1]:6.2f} dLM={out['cov_corr']['dlm'][-1]:+.3f}",
              flush=True)

    res = dict(layer=LAYER, v_norm=vnorm, mean_h_norm=hnorm, dm_clean=dm_clean,
               alphas=alphas, methods=methods, results=out,
               n_fit_tokens=int(len(H)), n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "02_corrector.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    C = {"raw": "#c0392b", "cov_corr": "#27ae60", "norm_clip": "#e67e22", "naive_inversion": "#7f8c8d"}
    L = {"raw": "raw steer  h+αv", "cov_corr": "ColdSteer (cov-aligned)",
         "norm_clip": "norm-clip", "naive_inversion": "naive inversion (control)"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    for m in methods:
        ax[0].plot(alphas, out[m]["dm"], "o-", color=C[m], label=L[m])
    ax[0].axhline(dm_clean, ls="--", color="k", lw=1, label=f"real acts ({dm_clean:.1f})")
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel("mean Mahalanobis $D_M$")
    ax[0].set_title("(a) off-manifold distance"); ax[0].legend(fontsize=8)

    for m in ["raw", "cov_corr", "norm_clip"]:
        ax[1].plot(alphas, out[m]["dlm"], "o-", color=C[m], label=L[m])
    ax[1].axhline(0.0, ls="--", color="k", lw=1)
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[1].set_title("(b) LM degradation"); ax[1].legend(fontsize=8)

    for m in ["raw", "cov_corr", "norm_clip", "naive_inversion"]:
        ax[2].plot(alphas, out[m]["retention"], "o-", color=C[m], label=L[m])
    ax[2].set_xlabel(r"steering strength $\alpha$")
    ax[2].set_ylabel(r"projection retention $\langle \hat h - h,\ \hat v\rangle$")
    ax[2].set_title("(c) steering projection preserved"); ax[2].legend(fontsize=8)

    fig.suptitle("Projection-preserving corrector vs raw steering (GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "02_corrector.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
