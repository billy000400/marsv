"""Experiment 27 — Seed robustness on GPT-2 MEDIUM: an error bar on the cross-model recovery.

Exp 26 put a 5-seed confidence interval on the FLAGSHIP recovery (GPT-2 small,
block 6): 83.3 ± 2.0% @α=8. The cross-model number (Exp 13, GPT-2 medium, block
12/24) reported 89% @α=8 / 101% @α=4 from a SINGLE seed-0 run — so we cannot yet
say whether medium's apparently-higher recovery is a real model-scale effect or
just optimization noise. This is Next-step option (i): give the cross-model check
its own error bar.

This script re-runs the EXACT Exp-13 GPT-2-medium pipeline (same DiffMean sentiment
vector at block 12, same 400-doc Gaussian fit, 300-doc train / held-out 100-doc eval,
same 5.25M corrector at d=1024, same recipe, matched projection) at N seeds and reports
mean ± std of the fluency recovery
        recovery(α) = (ΔLM_raw(α) − ΔLM_learned(α)) / ΔLM_raw(α)
at each α. Raw steering has no learned parameters, so ΔLM_raw is seed-independent
(computed once); only the learned corrector varies with the seed.

seed 0 reproduces Exp 13 to the digit — a built-in reproducibility check.

Reuses the Exp-3 module (Corrector/make_hat/train_corrector/eval helpers) and the
Exp-13 medium-retarget trick verbatim.
Outputs: results/27_seed_robustness_medium.json + plots/27_seed_robustness_medium.png.
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

MODEL_NAME = "gpt2-medium"
LAYER = 12            # mid of 24 blocks (depth analogue of block 6 of 12 in GPT-2 small)
D_MODEL = 1024
SEEDS = [0, 1, 2, 3, 4]
ALPHAS = [1.0, 2.0, 4.0, 8.0]


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
    retarget_to_medium()
    exp03.LAYER = LAYER
    print(f"model={MODEL_NAME} layer={LAYER} d={D_MODEL} device={DEVICE}", flush=True)

    # --- steering vector at the mid layer (deterministic, seed-independent) ---
    v = diffmean(LAYER)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)

    # --- clean Gaussian fit + eval set — identical to Exp 13, deterministic across seeds ---
    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=8)
    mu, cov, prec = exp03.gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]; del H
    dm_clean = float(exp03.mahalanobis(Hs, mu, prec).mean())
    print(f"|v|={vnorm:.2f} mean|h|={hnorm:.2f} clean D_M={dm_clean:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]     # 300 docs (same split as Exp 13)
    eval_texts = all_texts[400:500]      # held-out 100 docs

    clean_loss = exp03.lm_loss_fn(eval_texts, LAYER, lambda h: h, seq_len=128, batch=8)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    # --- raw steering ΔLM (seed-independent) ---
    dlm_raw, dm_raw = [], []
    for a in ALPHAS:
        def f_raw(h, a=a): return h + a * vt
        dlm_raw.append(exp03.lm_loss_fn(eval_texts, LAYER, f_raw, seq_len=128, batch=8) - clean_loss)
        dm_raw.append(float(exp03.mahalanobis(Hs + a * v[None, :], mu, prec).mean()))
    print("ΔLM raw:", [f"{x:+.3f}" for x in dlm_raw], flush=True)

    # --- learned corrector, one run per seed ---
    per_seed = {}
    for seed in SEEDS:
        exp03.SEED = seed                      # train_corrector reads module-global SEED
        torch.manual_seed(seed); np.random.seed(seed)
        corr = exp03.Corrector(d=D_MODEL).to(DEVICE)
        print(f"\n=== seed {seed} — training "
              f"({sum(p.numel() for p in corr.parameters())/1e6:.2f}M) ===", flush=True)
        exp03.train_corrector(corr, vt, vhat_t, train_texts, seq_len=64, batch=4)

        dlm, dm, rec = [], [], []
        for a, draw in zip(ALPHAS, dlm_raw):
            def f_learn(h, a=a):
                hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
            dl = exp03.lm_loss_fn(eval_texts, LAYER, f_learn, seq_len=128, batch=8) - clean_loss
            Zl = exp03.corrector_acts(corr, Hs, a, vt, vhat_t)
            dlm.append(dl)
            dm.append(float(exp03.mahalanobis(Zl, mu, prec).mean()))
            rec.append((draw - dl) / draw)
        per_seed[seed] = {"dlm": dlm, "dm": dm, "recovery": rec}
        print(f"seed {seed} recovery:", [f"{r*100:.0f}%" for r in rec], flush=True)
        del corr
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- aggregate across seeds ---
    rec_arr = np.array([per_seed[s]["recovery"] for s in SEEDS])
    dlm_arr = np.array([per_seed[s]["dlm"] for s in SEEDS])
    dm_arr = np.array([per_seed[s]["dm"] for s in SEEDS])
    agg = {
        "recovery_mean": rec_arr.mean(0).tolist(),
        "recovery_std": rec_arr.std(0, ddof=1).tolist(),
        "dlm_mean": dlm_arr.mean(0).tolist(),
        "dlm_std": dlm_arr.std(0, ddof=1).tolist(),
        "dm_mean": dm_arr.mean(0).tolist(),
        "dm_std": dm_arr.std(0, ddof=1).tolist(),
    }
    print("\n=== aggregate ===", flush=True)
    for i, a in enumerate(ALPHAS):
        print(f"alpha={a:>4}  ΔLM raw={dlm_raw[i]:+.3f}  ΔLM learned={agg['dlm_mean'][i]:+.3f}"
              f"±{agg['dlm_std'][i]:.3f}  recovery={agg['recovery_mean'][i]*100:.1f}"
              f"±{agg['recovery_std'][i]*100:.1f}%", flush=True)

    res = dict(model=MODEL_NAME, layer=LAYER, d_model=D_MODEL, v_norm=vnorm,
               mean_h_norm=hnorm, dm_clean=dm_clean, clean_loss=clean_loss,
               seeds=SEEDS, alphas=ALPHAS, dlm_raw=dlm_raw, dm_raw=dm_raw,
               per_seed={str(s): per_seed[s] for s in SEEDS}, aggregate=agg,
               n_fit_docs=400, n_train_texts=len(train_texts), n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "27_seed_robustness_medium.json"), "w") as f:
        json.dump(res, f, indent=2)

    # --- figure ---
    rec_m = np.array(agg["recovery_mean"]) * 100
    rec_s = np.array(agg["recovery_std"]) * 100
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(ALPHAS, dlm_raw, "o-", color="#c0392b", label="raw steer  h+αv")
    dlm_m = np.array(agg["dlm_mean"]); dlm_sd = np.array(agg["dlm_std"])
    ax[0].plot(ALPHAS, dlm_m, "o-", color="#2980b9",
               label=f"learned corrector (mean of {len(SEEDS)} seeds)")
    ax[0].fill_between(ALPHAS, dlm_m - dlm_sd, dlm_m + dlm_sd, color="#2980b9", alpha=0.25)
    ax[0].axhline(0, ls="--", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) LM degradation (lower is better)"); ax[0].legend(fontsize=8)

    ax[1].errorbar(ALPHAS, rec_m, yerr=rec_s, fmt="o-", color="#2980b9", capsize=4,
                   label=f"medium: mean ± std ({len(SEEDS)} seeds)")
    for s in SEEDS:
        ax[1].scatter(ALPHAS, np.array(per_seed[s]["recovery"]) * 100,
                      color="#95a5a6", s=14, alpha=0.6, zorder=1)
    ax[1].axhline(100, ls=":", color="k", lw=1)
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("fluency recovery (%)")
    ax[1].set_title("(b) recovery of raw steering's ΔLM damage"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Seed robustness on GPT-2 MEDIUM (355M, block {LAYER}/24, {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "27_seed_robustness_medium.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
