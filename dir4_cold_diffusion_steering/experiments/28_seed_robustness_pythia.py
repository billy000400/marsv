"""Experiment 28 — Seed robustness on Pythia-410m / GPT-NeoX: an error bar on the
cross-ARCHITECTURE recovery.

Exp 26 gave the flagship (GPT-2 small, block 6) recovery a 5-seed CI (83.3 ± 2.0% @α=8)
and Exp 27 gave GPT-2 medium one (88.3 ± 2.2% @α=8) — but both are the GPT-2 family. The
architecture axis (Exp 21 Qwen3, Exp 24 Pythia/GPT-NeoX; the 81–94% architecture band) was
still SINGLE-seed, so we could not say whether Pythia's 81% @α=8 (below GPT-2's) is a real
architecture effect or optimization noise. This is Exp 27's own Next check: give the
cross-architecture check its own error bar.

This script re-runs the EXACT Exp-24 Pythia-410m pipeline (same DiffMean sentiment vector at
block 12/24, same 400-doc Gaussian fit, 300-doc train / held-out 100-doc eval, same 5.25M
corrector at d=1024, same recipe, matched projection) at N seeds and reports mean ± std of the
fluency recovery
        recovery(α) = (ΔLM_raw(α) − ΔLM_learned(α)) / ΔLM_raw(α)
at each α. Raw steering has no learned parameters, so ΔLM_raw is seed-independent (computed
once); only the learned corrector varies with the seed.

seed 0 reproduces Exp 24 to the digit — a built-in reproducibility check.

Reuses the Exp-24 module (load / resid_post / make_hat / lm_loss_fn / train_corrector /
corrector_acts) VERBATIM; only the per-seed loop is new. exp24.SEED is overridden per run.
Outputs: results/28_seed_robustness_pythia.json + plots/28_seed_robustness_pythia.png.
"""
import os, json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# exp24 sets the CUDA memory fraction (0.18) + num_threads at import.
exp24 = _load("exp24", "24_cross_arch_pythia.py")
exp03 = _load("exp03", "03_learned_corrector.py")
exp01 = _load("exp01", "01_offmanifold_phenomenon.py")
from projections import normalize_vector

LAYER = exp24.LAYER
D_MODEL = exp24.D_MODEL
ALPHAS = exp24.ALPHAS
SEEDS = [0, 1, 2, 3, 4]
DEVICE = exp24.DEVICE


def main():
    model, tok = exp24.load()
    print(f"model={exp24.MODEL_NAME} layers={model.config.num_hidden_layers} "
          f"d={model.config.hidden_size} device={DEVICE}", flush=True)

    # --- steering vector at the mid layer (deterministic, seed-independent) ---
    hp = exp24.resid_post(exp01.POS, LAYER, seq_len=32, batch=8)
    hn = exp24.resid_post(exp01.NEG, LAYER, seq_len=32, batch=8)
    v = (hp.mean(0) - hn.mean(0)).astype(np.float32)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)

    # --- clean Gaussian fit + eval set — identical to Exp 24, deterministic across seeds ---
    H = exp24.resid_post(exp24.fineweb_texts(400), LAYER, seq_len=128, batch=exp24.FIT_BATCH)
    mu, cov, prec = exp03.gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(10000, len(H)), replace=False)
    Hs = H[idx]; del H
    dm_clean = float(exp03.mahalanobis(Hs, mu, prec).mean())
    print(f"|v|={vnorm:.2f} mean|h|={hnorm:.2f} clean D_M={dm_clean:.2f}", flush=True)

    all_texts = exp24.fineweb_texts(800)
    train_texts = all_texts[500:800]      # 300 docs (same split as Exp 24)
    eval_texts = all_texts[400:500]       # held-out 100 docs

    clean_loss = exp24.lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    # --- raw steering ΔLM (seed-independent) ---
    dlm_raw, dm_raw = [], []
    for a in ALPHAS:
        def f_raw(h, a=a): return h + a * vt
        dlm_raw.append(exp24.lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss)
        dm_raw.append(float(exp03.mahalanobis(Hs + a * v[None, :], mu, prec).mean()))
    print("ΔLM raw:", [f"{x:+.3f}" for x in dlm_raw], flush=True)

    # --- learned corrector, one run per seed ---
    per_seed = {}
    for seed in SEEDS:
        exp24.SEED = seed                   # train_corrector reads module-global SEED
        torch.manual_seed(seed); np.random.seed(seed)
        corr = exp03.Corrector(d=D_MODEL).to(DEVICE)
        print(f"\n=== seed {seed} — training "
              f"({sum(p.numel() for p in corr.parameters())/1e6:.2f}M) ===", flush=True)
        exp24.train_corrector(corr, vt, vhat_t, train_texts)

        dlm, dm, rec = [], [], []
        for a, draw in zip(ALPHAS, dlm_raw):
            def f_learn(h, a=a):
                hat, _ = exp24.make_hat(corr, h, a, vt, vhat_t); return hat
            dl = exp24.lm_loss_fn(eval_texts, LAYER, f_learn) - clean_loss
            Zl = exp24.corrector_acts(corr, Hs, a, vt, vhat_t)
            dlm.append(dl)
            dm.append(float(exp03.mahalanobis(Zl, mu, prec).mean()))
            rec.append((draw - dl) / draw)
        per_seed[seed] = {"dlm": dlm, "dm": dm, "recovery": rec}
        print(f"seed {seed} recovery:", [f"{r*100:.0f}%" for r in rec], flush=True)
        del corr
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

    res = dict(model=exp24.MODEL_NAME,
               arch="gpt_neox (parallel residual / rotary / LayerNorm / dense MHA)",
               layer=LAYER, d_model=D_MODEL, v_norm=vnorm, mean_h_norm=hnorm,
               dm_clean=dm_clean, clean_loss=clean_loss, seeds=SEEDS, alphas=ALPHAS,
               dlm_raw=dlm_raw, dm_raw=dm_raw,
               per_seed={str(s): per_seed[s] for s in SEEDS}, aggregate=agg,
               n_fit_docs=400, n_train_texts=len(train_texts), n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "28_seed_robustness_pythia.json"), "w") as f:
        json.dump(res, f, indent=2)

    # --- figure ---
    rec_m = np.array(agg["recovery_mean"]) * 100
    rec_s = np.array(agg["recovery_std"]) * 100
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(ALPHAS, dlm_raw, "o-", color="#c0392b", label="raw steer  h+αv")
    dlm_m = np.array(agg["dlm_mean"]); dlm_sd = np.array(agg["dlm_std"])
    ax[0].plot(ALPHAS, dlm_m, "o-", color="#8e44ad",
               label=f"learned corrector (mean of {len(SEEDS)} seeds)")
    ax[0].fill_between(ALPHAS, dlm_m - dlm_sd, dlm_m + dlm_sd, color="#8e44ad", alpha=0.25)
    ax[0].axhline(0, ls="--", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) LM degradation (lower is better)"); ax[0].legend(fontsize=8)

    ax[1].errorbar(ALPHAS, rec_m, yerr=rec_s, fmt="o-", color="#8e44ad", capsize=4,
                   label=f"Pythia-410m: mean ± std ({len(SEEDS)} seeds)")
    for s in SEEDS:
        ax[1].scatter(ALPHAS, np.array(per_seed[s]["recovery"]) * 100,
                      color="#95a5a6", s=14, alpha=0.6, zorder=1)
    ax[1].axhline(100, ls=":", color="k", lw=1)
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("fluency recovery (%)")
    ax[1].set_title("(b) recovery of raw steering's ΔLM damage"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Seed robustness on Pythia-410m / GPT-NeoX (block {LAYER}/24, {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "28_seed_robustness_pythia.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
