"""Experiment 26 — Seed robustness: a confidence interval on the flagship recovery.

Every experiment above is a SINGLE training run at SEED=0. The flagship number
("84% recovery @α=8", Exp 3) therefore has no error bar, and CLAUDE.md rule 10
names *seed* as a control that a trustworthy metric must survive. Layer (Exp 12),
model-scale (13/19), architecture (21/24), prompt-family (15), direction (5),
steering-family (18), and strength (4) are all covered — SEED is not.

This script re-runs the EXACT flagship Exp-3 pipeline (same vector, data, recipe,
matched projection) at N seeds and reports mean ± std of the fluency recovery
        recovery(α) = (ΔLM_raw(α) − ΔLM_learned(α)) / ΔLM_raw(α)
at each α. Raw steering has no learned parameters, so ΔLM_raw is seed-independent
(computed once); only the learned corrector varies with the seed.

seed 0 reproduces Exp 3 to the digit — a built-in reproducibility check.

Reuses the Exp-3 module verbatim (Corrector, make_hat, train_corrector, eval helpers)
by importing it and overriding its SEED per run.
Outputs: results/26_seed_robustness.json + plots/26_seed_robustness.png.
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

# ---- import the flagship Exp-3 module (filename starts with a digit) ----
spec = importlib.util.spec_from_file_location(
    "exp03", os.path.join(HERE, "03_learned_corrector.py"))
exp03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp03)

LAYER = exp03.LAYER
SEEDS = [0, 1, 2, 3, 4]
ALPHAS = [1.0, 2.0, 4.0, 6.0, 8.0]


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, "layer:", LAYER, flush=True)

    d = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))
    v = d["v"].astype(np.float32)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)
    print(f"|v| = {vnorm:.2f}", flush=True)

    # clean Gaussian fit + eval set — identical to Exp 3, deterministic across seeds
    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=16)
    mu, cov, prec = exp03.gaussian_stats(H)
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(exp03.mahalanobis(Hs, mu, prec).mean())

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]

    clean_loss = exp03.lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}  clean D_M={dm_clean:.2f}", flush=True)

    # ---- raw steering ΔLM (seed-independent) ----
    dlm_raw = []
    for a in ALPHAS:
        def f_raw(h, a=a): return h + a * vt
        dlm_raw.append(exp03.lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss)
    print("ΔLM raw:", [f"{x:+.3f}" for x in dlm_raw], flush=True)

    # ---- learned corrector, one run per seed ----
    per_seed = {}    # seed -> {"dlm": [...], "dm": [...], "recovery": [...]}
    for seed in SEEDS:
        exp03.SEED = seed                      # train_corrector reads module-global SEED
        torch.manual_seed(seed); np.random.seed(seed)
        corr = exp03.Corrector().to(DEVICE)
        print(f"\n=== seed {seed} — training ({sum(p.numel() for p in corr.parameters())/1e6:.2f}M) ===",
              flush=True)
        exp03.train_corrector(corr, vt, vhat_t, train_texts)

        dlm, dm, rec = [], [], []
        for a, draw in zip(ALPHAS, dlm_raw):
            def f_learn(h, a=a):
                hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
            dl = exp03.lm_loss_fn(eval_texts, LAYER, f_learn) - clean_loss
            Zl = exp03.corrector_acts(corr, Hs, a, vt, vhat_t)
            dlm.append(dl)
            dm.append(float(exp03.mahalanobis(Zl, mu, prec).mean()))
            rec.append((draw - dl) / draw)
        per_seed[seed] = {"dlm": dlm, "dm": dm, "recovery": rec}
        print(f"seed {seed} recovery:", [f"{r*100:.0f}%" for r in rec], flush=True)
        del corr
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---- aggregate across seeds ----
    rec_arr = np.array([per_seed[s]["recovery"] for s in SEEDS])      # [n_seed, n_alpha]
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

    res = dict(layer=LAYER, v_norm=vnorm, dm_clean=dm_clean, clean_loss=clean_loss,
               seeds=SEEDS, alphas=ALPHAS, dlm_raw=dlm_raw,
               per_seed={str(s): per_seed[s] for s in SEEDS}, aggregate=agg,
               n_fit_tokens=int(len(H)), n_train_texts=len(train_texts),
               n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "26_seed_robustness.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    rec_m = np.array(agg["recovery_mean"]) * 100
    rec_s = np.array(agg["recovery_std"]) * 100
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) ΔLM raw vs learned (mean±std band)
    ax[0].plot(ALPHAS, dlm_raw, "o-", color="#c0392b", label="raw steer  h+αv")
    dlm_m = np.array(agg["dlm_mean"]); dlm_sd = np.array(agg["dlm_std"])
    ax[0].plot(ALPHAS, dlm_m, "o-", color="#2980b9",
               label=f"learned corrector (mean of {len(SEEDS)} seeds)")
    ax[0].fill_between(ALPHAS, dlm_m - dlm_sd, dlm_m + dlm_sd, color="#2980b9", alpha=0.25)
    ax[0].axhline(0, ls="--", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) LM degradation (lower is better)"); ax[0].legend(fontsize=8)

    # (b) recovery mean ± std, with per-seed scatter
    ax[1].errorbar(ALPHAS, rec_m, yerr=rec_s, fmt="o-", color="#2980b9", capsize=4,
                   label=f"mean ± std ({len(SEEDS)} seeds)")
    for s in SEEDS:
        ax[1].scatter(ALPHAS, np.array(per_seed[s]["recovery"]) * 100,
                      color="#95a5a6", s=14, alpha=0.6, zorder=1)
    ax[1].axhline(100, ls=":", color="k", lw=1)
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("fluency recovery (%)")
    ax[1].set_title("(b) recovery of raw steering's ΔLM damage"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Seed robustness of the flagship corrector (GPT-2 small, layer {LAYER}, "
                 f"{len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "26_seed_robustness.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
