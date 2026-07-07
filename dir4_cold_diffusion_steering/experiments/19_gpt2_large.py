"""Iteration 19 — model-scaling: does the ColdSteer result hold on GPT-2 LARGE?

Exp 13 replicated the flagship result on GPT-2 medium (355M). This adds the THIRD
model-scale point — GPT-2 **large** (774M, 36 blocks, d=1280) — so the model-robustness
axis spans 124M -> 355M -> 774M (a >6x parameter range).

We replicate the flagship Exp-3 pipeline UNCHANGED, steering/correcting at the mid layer
(block 18 of 36 — depth analogue of block 6 of 12 in small and block 12 of 24 in medium).
Only the model (and hence d_model, layer count, |v|) changes; the DiffMean sentiment
prompts, 400-doc Gaussian fit, 300-doc training set, held-out 100-doc eval, 4-layer
corrector, seed, and hyper-parameters are all identical to Exp 3.

Reuse trick (same as Exp 13): overwrite common's model cache with GPT-2 large so every
imported Exp-3 helper transparently runs on it. Batch sizes are cut for VRAM (this run is
one of 5 agents sharing a 24 GB card at fraction 0.18 ~ 4.3 GB; large's fp32 weights alone
are ~3.1 GB). HALVE the batch on OOM per BUDGET.md.

Model: GPT-2 large, resid_post block 18. Baseline: raw steering z=h+alpha*v.
Outputs: results/19_gpt2_large.json + plots/19_gpt2_large.png.
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

MODEL_NAME = "gpt2-large"
LAYER = 18            # mid of 36 blocks (depth analogue of block 6 of 12 in GPT-2 small)
D_MODEL = 1280
ALPHAS = [1.0, 2.0, 4.0, 8.0]
SEED = 0

# VRAM-conscious batch sizes for the 774M model on a ~4.3 GB share.
FIT_BATCH = 4        # resid_post forward (no grad)
TRAIN_BATCH = 2      # backprop through the frozen upper LM
EVAL_BATCH = 4       # lm_loss_fn (no grad, but full-vocab logits)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

exp03 = _load("exp03", "03_learned_corrector.py")
exp01 = _load("exp01", "01_offmanifold_phenomenon.py")   # POS / NEG sentiment prompts


def retarget():
    """Load GPT-2 large once and install it in common's model cache so every
    imported Exp-3 helper (which calls common.load_model()) runs on it."""
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
    model, tok = retarget()
    print(f"model={MODEL_NAME} n_layer={model.config.n_layer} d={model.config.n_embd} "
          f"device={DEVICE}", flush=True)

    # --- steering vector at the mid layer ---
    v = diffmean(LAYER)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)

    # --- clean Gaussian (fit set disjoint from train/eval) ---
    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=FIT_BATCH)
    mu, cov, prec = exp03.gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]; del H
    dm_clean = float(exp03.mahalanobis(Hs, mu, prec).mean())
    print(f"|v|={vnorm:.2f} mean|h|={hnorm:.2f} clean D_M={dm_clean:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]     # 300 docs (same split as Exp 3)
    eval_texts = all_texts[400:500]      # held-out 100 docs

    # --- train the identical Exp-3 corrector at d=1280 ---
    corr = exp03.Corrector(d=D_MODEL).to(DEVICE)
    print(f"corrector params: {sum(p.numel() for p in corr.parameters())/1e6:.2f}M", flush=True)
    exp03.LAYER = LAYER
    exp03.train_corrector(corr, vt, vhat_t, train_texts, seq_len=64, batch=TRAIN_BATCH)

    # --- eval at matched projection ---
    clean_loss = exp03.lm_loss_fn(eval_texts, LAYER, lambda h: h, seq_len=128, batch=EVAL_BATCH)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)
    out = {"raw": {"dlm": [], "dm": []},
           "learned": {"dlm": [], "dm": [], "retention": []}}
    for a in ALPHAS:
        def f_raw(h):  return h + a * vt
        def f_learn(h):
            hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
        Z = Hs + a * v[None, :]
        Zl = exp03.corrector_acts(corr, Hs, a, vt, vhat_t)
        out["raw"]["dlm"].append(exp03.lm_loss_fn(eval_texts, LAYER, f_raw, seq_len=128, batch=EVAL_BATCH) - clean_loss)
        out["raw"]["dm"].append(float(exp03.mahalanobis(Z, mu, prec).mean()))
        out["learned"]["dlm"].append(exp03.lm_loss_fn(eval_texts, LAYER, f_learn, seq_len=128, batch=EVAL_BATCH) - clean_loss)
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
    with open(os.path.join(RESULTS, "19_gpt2_large.json"), "w") as f:
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
    fig.suptitle("ColdSteer replicates on GPT-2 LARGE (774M, block 18/36): raw steering "
                 "breaks the LM and the LM-supervised corrector recovers it")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "19_gpt2_large.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
