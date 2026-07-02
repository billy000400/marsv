"""Iteration 1 — motivating phenomenon for ColdSteer.

Claim under test: raw linear activation steering z = h + alpha*v pushes GPT-2
activations OFF the activation manifold as alpha grows, which (a) inflates a
Gaussian Mahalanobis distance to real activations and (b) degrades next-token
LM loss. This motivates a projection-preserving corrector.

Model: GPT-2 small (124M), resid_post at block 6 (middle of 12 blocks).
Steering vector: DiffMean sentiment direction (positive - negative), raw units.
Outputs: data/ artifacts + plots/01_offmanifold_phenomenon.png + results json.
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import resid_post, lm_loss, fineweb_texts, load_model, DATA, DEVICE

LAYER = 6
os.makedirs(DATA, exist_ok=True)
PLOTS = os.path.join(os.path.dirname(__file__), "..", "plots")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

POS = [
    "This movie was absolutely wonderful and I loved every minute.",
    "What a fantastic experience, everything was perfect.",
    "I am so happy with this, it exceeded all my expectations.",
    "The food was delicious and the service was excellent.",
    "A brilliant, beautiful, and deeply moving performance.",
    "I feel great today, everything is going wonderfully.",
    "This is the best product I have ever purchased.",
    "Such a delightful and charming little book.",
    "The team did an amazing job and I am thrilled.",
    "Warm, kind, and genuinely lovely people.",
    "The weather is gorgeous and the view is stunning.",
    "I highly recommend this, it is truly excellent.",
    "A joyful celebration full of laughter and love.",
    "The results were outstanding and everyone was pleased.",
    "This restaurant is superb, the flavors are incredible.",
    "A wonderful gift that made my whole day brighter.",
    "The concert was magnificent and the crowd was ecstatic.",
    "I am grateful for such a positive and uplifting message.",
    "Everything about this trip was perfect and relaxing.",
    "A splendid success that we are all proud of.",
]
NEG = [
    "This movie was absolutely terrible and I hated every minute.",
    "What an awful experience, everything was ruined.",
    "I am so disappointed with this, it failed all my expectations.",
    "The food was disgusting and the service was horrible.",
    "A dull, ugly, and deeply depressing performance.",
    "I feel awful today, everything is going wrong.",
    "This is the worst product I have ever purchased.",
    "Such a dreadful and annoying little book.",
    "The team did a terrible job and I am furious.",
    "Cold, cruel, and genuinely unpleasant people.",
    "The weather is miserable and the view is bleak.",
    "I strongly warn against this, it is truly awful.",
    "A sorrowful ordeal full of tears and pain.",
    "The results were disastrous and everyone was upset.",
    "This restaurant is dreadful, the flavors are revolting.",
    "A worthless gift that made my whole day darker.",
    "The concert was atrocious and the crowd was angry.",
    "I resent such a negative and depressing message.",
    "Everything about this trip was ruined and stressful.",
    "A pathetic failure that we are all ashamed of.",
]


def diffmean_vector():
    hp = resid_post(POS, LAYER, seq_len=32, batch=8)   # pooled below via mean over all tokens
    hn = resid_post(NEG, LAYER, seq_len=32, batch=8)
    v = hp.mean(0) - hn.mean(0)
    return v.astype(np.float32)


def gaussian_stats(H):
    mu = H.mean(0)
    Xc = H - mu
    cov = (Xc.T @ Xc) / (len(H) - 1)
    cov += 1e-3 * np.eye(cov.shape[0], dtype=cov.dtype)  # ridge for invertibility
    prec = np.linalg.inv(cov).astype(np.float32)
    return mu.astype(np.float32), prec


def mahalanobis(H, mu, prec):
    Xc = H - mu
    return np.sqrt(np.einsum("nd,dk,nk->n", Xc, prec, Xc))


def main():
    load_model()
    print("device:", DEVICE, flush=True)

    # ---- steering vector ----
    v = diffmean_vector()
    vnorm = float(np.linalg.norm(v))
    print(f"DiffMean sentiment |v| = {vnorm:.2f}", flush=True)

    # ---- clean activation statistics (fit Gaussian) ----
    fit_texts = fineweb_texts(400)
    H = resid_post(fit_texts, LAYER, seq_len=128, batch=16)
    print(f"fit activations: {H.shape}", flush=True)
    mu, prec = gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    print(f"mean |h| = {hnorm:.2f}", flush=True)

    # baseline Mahalanobis of real activations (subsample for speed)
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(mahalanobis(Hs, mu, prec).mean())

    # ---- alpha sweep ----
    alphas = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    eval_texts = fineweb_texts(500)[400:500]   # held-out from the fit set

    dm, norm_ratio, dlm = [], [], []
    base_loss = None
    vt = torch.tensor(v, device=DEVICE)
    for a in alphas:
        Z = Hs + a * v[None, :]
        dm.append(float(mahalanobis(Z, mu, prec).mean()))
        norm_ratio.append(float(np.linalg.norm(Z, axis=1).mean() / hnorm))
        loss = lm_loss(eval_texts, LAYER, a * vt, seq_len=128, batch=16)
        if a == 0.0:
            base_loss = loss
        dlm.append(loss - base_loss)
        print(f"alpha={a:>4}  D_M={dm[-1]:7.2f}  |z|/|h|={norm_ratio[-1]:.3f}  dLM={dlm[-1]:+.4f}", flush=True)

    res = dict(layer=LAYER, v_norm=vnorm, mean_h_norm=hnorm,
               dm_clean=dm_clean, alphas=alphas,
               mahalanobis=dm, norm_ratio=norm_ratio, dlm=dlm,
               n_fit_tokens=int(len(H)), n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "01_offmanifold.json"), "w") as f:
        json.dump(res, f, indent=2)

    # persist vector + stats for later iterations
    np.savez(os.path.join(DATA, "sentiment_vec_layer6.npz"),
             v=v, mu=mu, mean_h_norm=hnorm)

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(alphas, dm, "o-", color="#c0392b")
    ax[0].axhline(dm_clean, ls="--", color="gray", label=f"real acts ({dm_clean:.1f})")
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel("mean Mahalanobis dist.")
    ax[0].set_title("(a) off-manifold distance"); ax[0].legend()

    ax[1].plot(alphas, norm_ratio, "o-", color="#2980b9")
    ax[1].axhline(1.0, ls="--", color="gray")
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel(r"$\|z\|/\|h\|$")
    ax[1].set_title("(b) activation norm inflation")

    ax[2].plot(alphas, dlm, "o-", color="#8e44ad")
    ax[2].axhline(0.0, ls="--", color="gray")
    ax[2].set_xlabel(r"steering strength $\alpha$"); ax[2].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[2].set_title("(c) LM degradation")

    fig.suptitle("Raw linear steering z = h + αv goes off-manifold (GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "01_offmanifold_phenomenon.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
