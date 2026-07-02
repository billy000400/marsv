"""Iteration 12 — layer robustness: is the ColdSteer result a block-6 artifact?

Every prior experiment steers/corrects at resid_post block 6. A reviewer's first
question is whether the two headline facts are specific to that layer:

    (P) raw steering z = h + alpha*v goes off-manifold and raises LM loss, and
    (C) an LM-supervised projection-preserving corrector r_theta recovers most of
        that damage at matched projection.

We replicate the flagship Exp-3 pipeline UNCHANGED at three layers — block 3 (early),
block 6 (mid, = Exp 3, a reproducibility check), block 9 (late). At each layer we:
  1. build the DiffMean sentiment vector v at that layer (same POS/NEG prompts),
  2. fit the clean Gaussian (same 400 docs),
  3. measure the raw phenomenon (Delta LM, D_M vs alpha),
  4. train the identical 4-layer corrector against the downstream LM loss (same
     recipe/seed/hyperparams; only the hook layer changes),
  5. evaluate learned Delta LM / D_M / retention at matched projection.

Reuses Exp 3's Corrector / train_corrector / eval helpers by importing that module
and swapping its module-global LAYER per layer. Model: GPT-2 small.
Outputs: results/12_layer_robustness.json + plots/12_layer_robustness.png.
"""
import os, json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import resid_post, fineweb_texts, load_model, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

exp03 = _load("exp03", "03_learned_corrector.py")
exp01 = _load("exp01", "01_offmanifold_phenomenon.py")   # for POS / NEG prompts

LAYERS = [3, 6, 9]
ALPHAS = [1.0, 2.0, 4.0, 8.0]
SEED = 0


def diffmean(layer):
    hp = resid_post(exp01.POS, layer, seq_len=32, batch=8)
    hn = resid_post(exp01.NEG, layer, seq_len=32, batch=8)
    return (hp.mean(0) - hn.mean(0)).astype(np.float32)


def run_layer(layer, eval_texts, train_texts):
    torch.manual_seed(SEED); np.random.seed(SEED)
    exp03.LAYER = layer                                   # retarget the shared recipe

    v = diffmean(layer)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)

    # clean Gaussian at this layer (fit set disjoint from train/eval)
    H = resid_post(fineweb_texts(400), layer, seq_len=128, batch=16)
    mu, cov, prec = exp03.gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(exp03.mahalanobis(Hs, mu, prec).mean())
    del H
    print(f"[L{layer}] |v|={vnorm:.2f} mean|h|={hnorm:.2f} clean D_M={dm_clean:.2f}", flush=True)

    # train corrector with the EXACT Exp 3 recipe (only LAYER changed)
    corr = exp03.Corrector().to(DEVICE)
    exp03.train_corrector(corr, vt, vhat_t, train_texts)

    clean_loss = exp03.lm_loss_fn(eval_texts, layer, lambda h: h)
    out = {"raw": {"dlm": [], "dm": []}, "learned": {"dlm": [], "dm": [], "retention": []}}
    for a in ALPHAS:
        def f_raw(h):  return h + a * vt
        def f_learn(h):
            hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
        Z = Hs + a * v[None, :]
        Zl = exp03.corrector_acts(corr, Hs, a, vt, vhat_t)
        out["raw"]["dlm"].append(exp03.lm_loss_fn(eval_texts, layer, f_raw) - clean_loss)
        out["raw"]["dm"].append(float(exp03.mahalanobis(Z, mu, prec).mean()))
        out["learned"]["dlm"].append(exp03.lm_loss_fn(eval_texts, layer, f_learn) - clean_loss)
        out["learned"]["dm"].append(float(exp03.mahalanobis(Zl, mu, prec).mean()))
        out["learned"]["retention"].append(float(((Zl - Hs) @ vhat).mean()))
        rec = 1.0 - out["learned"]["dlm"][-1] / out["raw"]["dlm"][-1]
        print(f"[L{layer}] a={a:>4} raw ΔLM={out['raw']['dlm'][-1]:+.3f} "
              f"learned={out['learned']['dlm'][-1]:+.3f} rec={rec*100:.0f}% "
              f"D_M raw={out['raw']['dm'][-1]:.1f} learn={out['learned']['dm'][-1]:.1f}", flush=True)

    del corr
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return dict(layer=layer, v_norm=vnorm, mean_h_norm=hnorm, dm_clean=dm_clean,
                clean_loss=clean_loss, results=out)


def main():
    model, _ = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]        # 300 docs (same split as Exp 3)
    eval_texts = all_texts[400:500]         # same held-out 100 docs

    layers = [run_layer(L, eval_texts, train_texts) for L in LAYERS]
    res = dict(alphas=ALPHAS, layers_run=LAYERS, seed=SEED,
               n_train_texts=len(train_texts), n_eval_texts=len(eval_texts),
               per_layer=layers)
    with open(os.path.join(RESULTS, "12_layer_robustness.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    colors = {3: "#e67e22", 6: "#2980b9", 9: "#8e44ad"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for Lr in layers:
        L = Lr["layer"]; r = Lr["results"]; c = colors[L]
        ax[0].plot(ALPHAS, r["raw"]["dlm"], "o--", color=c, alpha=0.6,
                   label=f"L{L} raw")
        ax[0].plot(ALPHAS, r["learned"]["dlm"], "o-", color=c,
                   label=f"L{L} corrected")
        rec = [1.0 - r["learned"]["dlm"][i] / r["raw"]["dlm"][i] for i in range(len(ALPHAS))]
        ax[1].plot(ALPHAS, [x * 100 for x in rec], "o-", color=c, label=f"block {L}")
        ax[2].plot(ALPHAS, r["raw"]["dm"], "o--", color=c, alpha=0.6, label=f"L{L} raw")
        ax[2].plot(ALPHAS, r["learned"]["dm"], "o-", color=c, label=f"L{L} corrected")
    ax[0].axhline(0, ls=":", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) LM damage: raw (dashed) vs corrected (solid)"); ax[0].legend(fontsize=7, ncol=3)
    ax[1].axhline(0, ls=":", color="k", lw=1); ax[1].axhline(100, ls=":", color="gray", lw=1)
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("fluency recovery (%)")
    ax[1].set_title("(b) recovery = 1 - ΔLM$_{corr}$/ΔLM$_{raw}$"); ax[1].legend(fontsize=8)
    ax[2].set_xlabel(r"steering strength $\alpha$"); ax[2].set_ylabel("mean Mahalanobis $D_M$")
    ax[2].set_title("(c) off-Gaussian distance (corrected > raw at every layer)")
    ax[2].legend(fontsize=7, ncol=3)
    fig.suptitle("ColdSteer replicates across layers: raw steering breaks the LM and the "
                 "LM-supervised corrector recovers it at blocks 3, 6, 9 (GPT-2 small)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "12_layer_robustness.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
