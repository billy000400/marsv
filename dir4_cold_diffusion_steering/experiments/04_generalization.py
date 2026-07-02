"""Iteration 4 — GENERALIZATION: does the learned corrector extrapolate BEYOND its
training range of steering strengths?

Exp 3 trained r_θ with α ~ U(0.5, 8) and showed it beats raw steering at every α in
[1, 8] on held-out text. The obvious generalization question: the training never saw
α > 8, so does the SAME corrector still help at α = 10, 12 (out-of-training-range), or
does it collapse once we push past where it was trained?

We reuse Exp 3's Corrector, training loop, and eval helpers verbatim (imported), train
identically (α ~ U(0.5, 8), same seed / data), then evaluate at
    α ∈ {1, 2, 4, 6, 8 | 10, 12}
where α ≤ 8 is in-range and α ∈ {10, 12} is strictly beyond training. Matched projection
by construction (ĥ = z + P_{v⊥}r_θ ⇒ retention = α|v|). Metrics: ΔLM, D_M vs raw steering.
Outputs: results/04_generalization.json + plots/04_generalization.png.
"""
import os, json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import resid_post, fineweb_texts, load_model, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")

# reuse Exp 3 code verbatim (DRY: same Corrector / training / eval path)
_spec = importlib.util.spec_from_file_location(
    "exp03", os.path.join(HERE, "03_learned_corrector.py"))
exp03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp03)
Corrector = exp03.Corrector
train_corrector = exp03.train_corrector
corrector_acts = exp03.corrector_acts
gaussian_stats = exp03.gaussian_stats
mahalanobis = exp03.mahalanobis
lm_loss_fn = exp03.lm_loss_fn
make_hat = exp03.make_hat
LAYER = exp03.LAYER

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
AMAX_TRAIN = 8.0                      # training upper bound (α ~ U(0.5, 8))


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

    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=16)
    mu, cov, prec = gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(mahalanobis(Hs, mu, prec).mean())
    print(f"clean D_M={dm_clean:.2f} mean|h|={hnorm:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]                 # same as Exp 3
    eval_texts = all_texts[400:500]                  # same held-out set

    corr = Corrector().to(DEVICE)
    print(f"corrector params: {sum(p.numel() for p in corr.parameters())/1e6:.2f}M", flush=True)
    train_corrector(corr, vt, vhat_t, train_texts)   # α ~ U(0.5, 8), identical to Exp 3

    # ---- eval, including OUT-OF-RANGE α ----
    alphas = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    methods = ["raw", "learned"]
    out = {m: {"dm": [], "dlm": [], "retention": []} for m in methods}

    clean_loss = lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    for a in alphas:
        def f_raw(h):  return h + a * vt
        def f_learn(h):
            hat, _ = make_hat(corr, h, a, vt, vhat_t); return hat
        fns = {"raw": f_raw, "learned": f_learn}

        Z = Hs + a * v[None, :]
        Zl = corrector_acts(corr, Hs, a, vt, vhat_t)
        acts = {"raw": Z, "learned": Zl}
        for m in methods:
            out[m]["dm"].append(float(mahalanobis(acts[m], mu, prec).mean()))
            out[m]["retention"].append(float(((acts[m] - Hs) @ vhat).mean()))
            out[m]["dlm"].append(lm_loss_fn(eval_texts, LAYER, fns[m]) - clean_loss)
        tag = "  (extrap)" if a > AMAX_TRAIN else ""
        print(f"alpha={a:>4}{tag}  ΔLM raw={out['raw']['dlm'][-1]:+.3f} "
              f"learned={out['learned']['dlm'][-1]:+.3f}  | D_M raw={out['raw']['dm'][-1]:.1f} "
              f"learned={out['learned']['dm'][-1]:.1f}", flush=True)

    res = dict(layer=LAYER, v_norm=vnorm, mean_h_norm=hnorm, dm_clean=dm_clean,
               clean_loss=clean_loss, alphas=alphas, amax_train=AMAX_TRAIN,
               methods=methods, results=out, n_train_texts=len(train_texts),
               n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "04_generalization.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    C = {"raw": "#c0392b", "learned": "#2980b9"}
    Lb = {"raw": "raw steer  h+αv", "learned": "learned corrector (trained α≤8)"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in methods:
        ax[0].plot(alphas, out[m]["dlm"], "o-", color=C[m], label=Lb[m])
    ax[0].axhline(0, ls="--", color="k", lw=1)
    ax[0].axvspan(AMAX_TRAIN, alphas[-1], color="0.85", zorder=0)
    ax[0].text(AMAX_TRAIN + 0.1, ax[0].get_ylim()[1] * 0.92, "extrapolation\n(α>8, untrained)",
               fontsize=8, va="top")
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) LM degradation (lower is better)"); ax[0].legend(fontsize=8, loc="upper left")
    for m in methods:
        ax[1].plot(alphas, out[m]["dm"], "o-", color=C[m], label=Lb[m])
    ax[1].axhline(dm_clean, ls="--", color="k", lw=1, label=f"real acts ({dm_clean:.1f})")
    ax[1].axvspan(AMAX_TRAIN, alphas[-1], color="0.85", zorder=0)
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("mean Mahalanobis $D_M$")
    ax[1].set_title("(b) off-manifold distance"); ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Generalization beyond the training range: learned corrector at α up to 12 "
                 "(GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "04_generalization.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
