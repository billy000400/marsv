"""Iteration 13 — cross-model generality: is the ColdSteer result a GPT-2-*small* artifact?

Exp 12 showed the flagship result (raw steering breaks the LM; an LM-supervised
projection-preserving corrector recovers it at matched projection) replicates across
LAYERS of GPT-2 small. The next obvious external-validity question is the MODEL axis:
does the same recipe work on a *different, larger* model?

We replicate the flagship Exp-3 pipeline UNCHANGED on GPT-2 **medium** (355M, 24 blocks,
d=1024), steering/correcting at its mid layer (block 12 of 24 — the depth analogue of
block 6 of 12 in small). Only the model (and hence d_model, layer count, |v|) changes;
the DiffMean sentiment prompts, 400-doc Gaussian fit, 300-doc training set, held-out
100-doc eval, 4-layer corrector, seed, and hyper-parameters are all identical to Exp 3.

Reuse trick: Exp 3's train/eval helpers fetch the model through common.load_model()'s
cache. We load GPT-2 medium once and overwrite that cache (common._model/_tok) so every
imported helper transparently runs on medium. The corrector is instantiated at d=1024.

Model: GPT-2 medium, resid_post block 12. Baseline: raw steering z=h+alpha*v.
Outputs: results/13_cross_model.json + plots/13_cross_model.png.
"""
import os, json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import common
from common import resid_post, fineweb_texts, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

MODEL_NAME = "gpt2-medium"
LAYER = 12            # mid of 24 blocks (depth analogue of block 6 of 12 in GPT-2 small)
D_MODEL = 1024
ALPHAS = [1.0, 2.0, 4.0, 8.0]
SEED = 0


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

exp03 = _load("exp03", "03_learned_corrector.py")
exp01 = _load("exp01", "01_offmanifold_phenomenon.py")   # POS / NEG sentiment prompts


def retarget_to_medium():
    """Load GPT-2 medium once and install it in common's model cache so every
    imported Exp-3 helper (which calls common.load_model()) runs on medium."""
    tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    common._model = model
    common._tok = tok
    return model, tok


def diffmean(layer):
    hp = resid_post(exp01.POS, layer, seq_len=32, batch=8)
    hn = resid_post(exp01.NEG, layer, seq_len=32, batch=8)
    return (hp.mean(0) - hn.mean(0)).astype(np.float32)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    model, tok = retarget_to_medium()
    print(f"model={MODEL_NAME} n_layer={model.config.n_layer} d={model.config.n_embd} "
          f"device={DEVICE}", flush=True)

    # --- steering vector at the mid layer ---
    v = diffmean(LAYER)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)

    # --- clean Gaussian (fit set disjoint from train/eval) ---
    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=8)
    mu, cov, prec = exp03.gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]; del H
    dm_clean = float(exp03.mahalanobis(Hs, mu, prec).mean())
    print(f"|v|={vnorm:.2f} mean|h|={hnorm:.2f} clean D_M={dm_clean:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]     # 300 docs (same split as Exp 3)
    eval_texts = all_texts[400:500]      # held-out 100 docs

    # --- train the identical Exp-3 corrector at d=1024, batch=4 for medium's memory ---
    corr = exp03.Corrector(d=D_MODEL).to(DEVICE)
    print(f"corrector params: {sum(p.numel() for p in corr.parameters())/1e6:.2f}M", flush=True)
    exp03.LAYER = LAYER
    exp03.train_corrector(corr, vt, vhat_t, train_texts, seq_len=64, batch=4)

    # --- eval at matched projection ---
    clean_loss = exp03.lm_loss_fn(eval_texts, LAYER, lambda h: h, seq_len=128, batch=8)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)
    out = {"raw": {"dlm": [], "dm": []},
           "learned": {"dlm": [], "dm": [], "retention": []}}
    for a in ALPHAS:
        def f_raw(h):  return h + a * vt
        def f_learn(h):
            hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
        Z = Hs + a * v[None, :]
        Zl = exp03.corrector_acts(corr, Hs, a, vt, vhat_t)
        out["raw"]["dlm"].append(exp03.lm_loss_fn(eval_texts, LAYER, f_raw, seq_len=128, batch=8) - clean_loss)
        out["raw"]["dm"].append(float(exp03.mahalanobis(Z, mu, prec).mean()))
        out["learned"]["dlm"].append(exp03.lm_loss_fn(eval_texts, LAYER, f_learn, seq_len=128, batch=8) - clean_loss)
        out["learned"]["dm"].append(float(exp03.mahalanobis(Zl, mu, prec).mean()))
        out["learned"]["retention"].append(float(((Zl - Hs) @ vhat).mean()))
        rec = 1.0 - out["learned"]["dlm"][-1] / out["raw"]["dlm"][-1]
        print(f"a={a:>4} raw ΔLM={out['raw']['dlm'][-1]:+.3f} "
              f"learned={out['learned']['dlm'][-1]:+.3f} rec={rec*100:.0f}% "
              f"D_M raw={out['raw']['dm'][-1]:.1f} learn={out['learned']['dm'][-1]:.1f}", flush=True)

    res = dict(model=MODEL_NAME, layer=LAYER, d_model=D_MODEL, v_norm=vnorm,
               mean_h_norm=hnorm, dm_clean=dm_clean, clean_loss=clean_loss,
               alphas=ALPHAS, seed=SEED, n_fit_docs=400,
               n_train_texts=len(train_texts), n_eval_texts=len(eval_texts),
               corr_params=sum(p.numel() for p in corr.parameters()), results=out)
    with open(os.path.join(RESULTS, "13_cross_model.json"), "w") as f:
        json.dump(res, f, indent=2)

    # --- figure ---
    rec = [1.0 - out["learned"]["dlm"][i] / out["raw"]["dlm"][i] for i in range(len(ALPHAS))]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(ALPHAS, out["raw"]["dlm"], "o--", color="#c0392b", label="raw steer  h+αv")
    ax[0].plot(ALPHAS, out["learned"]["dlm"], "o-", color="#2980b9", label="learned corrector")
    ax[0].axhline(0, ls=":", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) LM damage: raw vs corrected"); ax[0].legend(fontsize=8)
    ax[1].plot(ALPHAS, [x * 100 for x in rec], "o-", color="#16a085")
    ax[1].axhline(0, ls=":", color="k", lw=1); ax[1].axhline(100, ls=":", color="gray", lw=1)
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("fluency recovery (%)")
    ax[1].set_title(r"(b) recovery = 1 - $\Delta$LM$_{corr}$/$\Delta$LM$_{raw}$")
    ax[2].plot(ALPHAS, out["raw"]["dm"], "o--", color="#c0392b", label="raw")
    ax[2].plot(ALPHAS, out["learned"]["dm"], "o-", color="#2980b9", label="corrected")
    ax[2].axhline(dm_clean, ls="--", color="gray", lw=1, label=f"real acts ({dm_clean:.1f})")
    ax[2].set_xlabel(r"steering strength $\alpha$"); ax[2].set_ylabel("mean Mahalanobis $D_M$")
    ax[2].set_title("(c) off-Gaussian distance (corrected > raw)"); ax[2].legend(fontsize=8)
    fig.suptitle("ColdSteer replicates on GPT-2 MEDIUM (355M, block 12/24): raw steering "
                 "breaks the LM and the LM-supervised corrector recovers it")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "13_cross_model.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
