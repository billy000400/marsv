"""Experiment 33 — Joint vector×seed resample: the TOTAL flagship uncertainty.

Two sampling axes have been controlled SEPARATELY:
  - Exp 26 varied only the OPTIMIZATION SEED (vector fixed):  recovery 83.3±2.0% @α=8
  - Exp 32 varied only the STEERING VECTOR   (seed fixed = 0): recovery 82.1±2.7% @α=8
Neither answers the practical question: if you rebuild `v` from a fresh sample of
sentences AND retrain the corrector from a fresh init, what is the COMBINED spread?
If the two noise sources are independent they should add in quadrature
(≈ sqrt(2.0² + 2.7²) ≈ 3.4 pp); a smaller/larger joint std would signal that they
partly cancel or compound.

This script is a controlled edit of Exp 32: it reuses the SAME sentence-resampling
RNG (RandomState(1234), same call sequence) so the bootstrap VECTORS are IDENTICAL
to Exp 32's — but it now ALSO floats the corrector seed per draw (seed = b, instead
of Exp 32's fixed seed 0). So Exp 33 differs from Exp 32 by exactly one thing: the
seed varies jointly with the vector. b=0 is the original full set at seed 0 — a
built-in reproduction of Exp 3 / Exp 32's b=0.

Reuses the Exp-3 module verbatim (Corrector, make_hat, train_corrector, eval
helpers) and the Exp-1 module's POS/NEG sentence lists.
Outputs: results/33_joint_vector_seed.json + plots/33_joint_vector_seed.png.
"""
import os, json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.cuda.set_per_process_memory_fraction(0.180)
torch.set_num_threads(1)

from common import resid_post, fineweb_texts, load_model, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp03 = _load("exp03", "03_learned_corrector.py")
exp01 = _load("exp01", "01_offmanifold_phenomenon.py")

LAYER = exp03.LAYER
ALPHAS = [1.0, 2.0, 4.0, 6.0, 8.0]
N_BOOT = 6          # b=0 is the un-resampled original (seed 0); b=1..5 joint resamples


def diffmean(pos, neg):
    """DiffMean over all tokens: mean(POS tokens) − mean(NEG tokens), raw units."""
    hp = resid_post(pos, LAYER, seq_len=32, batch=8)
    hn = resid_post(neg, LAYER, seq_len=32, batch=8)
    return (hp.mean(0) - hn.mean(0)).astype(np.float32)


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, "layer:", LAYER, flush=True)

    POS, NEG = exp01.POS, exp01.NEG
    print(f"|POS|={len(POS)}  |NEG|={len(NEG)}", flush=True)

    v_full = diffmean(POS, NEG)
    vhat_full = normalize_vector(v_full)

    # ---- clean Gaussian fit + eval/train split — identical to Exp 26/32 ----
    H = resid_post(fineweb_texts(800), LAYER, seq_len=128, batch=16)
    mu, cov, prec = exp03.gaussian_stats(H)
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(exp03.mahalanobis(Hs, mu, prec).mean())

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]
    clean_loss = exp03.lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}  clean D_M={dm_clean:.2f}", flush=True)

    # Same sentence-resampling RNG as Exp 32 -> IDENTICAL bootstrap vectors.
    rng = np.random.RandomState(1234)

    per_boot = {}
    for b in range(N_BOOT):
        if b == 0:
            pos_b, neg_b = POS, NEG            # original, un-resampled
        else:
            pos_b = [POS[i] for i in rng.randint(0, len(POS), size=len(POS))]
            neg_b = [NEG[i] for i in rng.randint(0, len(NEG), size=len(NEG))]
        v = diffmean(pos_b, neg_b)
        vnorm = float(np.linalg.norm(v))
        cos = float(np.dot(normalize_vector(v), vhat_full))
        vt = torch.tensor(v, device=DEVICE)
        vhat_t = torch.tensor(normalize_vector(v), device=DEVICE)

        # raw ΔLM for THIS vector (v changes raw steering too; seed-independent)
        dlm_raw = []
        for a in ALPHAS:
            def f_raw(h, a=a, vt=vt): return h + a * vt
            dlm_raw.append(exp03.lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss)

        # JOINT: seed floats with b (Exp 32 held it at 0). b=0 -> seed 0 (= Exp 3).
        seed_b = b
        exp03.SEED = seed_b
        torch.manual_seed(seed_b); np.random.seed(seed_b)
        corr = exp03.Corrector().to(DEVICE)
        print(f"\n=== boot {b} — seed={seed_b} |v|={vnorm:.2f} cos(v,v_full)={cos:.4f} — training ===",
              flush=True)
        exp03.train_corrector(corr, vt, vhat_t, train_texts)

        dlm, rec = [], []
        for a, draw in zip(ALPHAS, dlm_raw):
            def f_learn(h, a=a, vt=vt, vhat_t=vhat_t):
                hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
            dl = exp03.lm_loss_fn(eval_texts, LAYER, f_learn) - clean_loss
            dlm.append(dl)
            rec.append((draw - dl) / draw)
        per_boot[b] = dict(seed=seed_b, v_norm=vnorm, cos=cos, dlm_raw=dlm_raw,
                           dlm=dlm, recovery=rec)
        print(f"boot {b} raw ΔLM:", [f"{x:+.3f}" for x in dlm_raw], flush=True)
        print(f"boot {b} recovery:", [f"{r*100:.0f}%" for r in rec], flush=True)
        del corr
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---- aggregate over the JOINT resamples only (b>=1); b=0 is the reference ----
    boots = [b for b in range(N_BOOT) if b >= 1]
    rec_arr = np.array([per_boot[b]["recovery"] for b in boots])       # [n_boot, n_alpha]
    raw_arr = np.array([per_boot[b]["dlm_raw"] for b in boots])
    cos_arr = np.array([per_boot[b]["cos"] for b in boots])
    agg = dict(
        recovery_mean=rec_arr.mean(0).tolist(),
        recovery_std=rec_arr.std(0, ddof=1).tolist(),
        recovery_min=rec_arr.min(0).tolist(),
        recovery_max=rec_arr.max(0).tolist(),
        dlm_raw_mean=raw_arr.mean(0).tolist(),
        dlm_raw_std=raw_arr.std(0, ddof=1).tolist(),
        cos_mean=float(cos_arr.mean()), cos_min=float(cos_arr.min()),
    )
    print("\n=== aggregate over %d JOINT resamples (b=0 reference) ===" % len(boots), flush=True)
    print(f"cos(v_boot, v_full): mean={agg['cos_mean']:.4f} min={agg['cos_min']:.4f}", flush=True)
    for i, a in enumerate(ALPHAS):
        print(f"alpha={a:>4}  recovery={agg['recovery_mean'][i]*100:.1f}"
              f"±{agg['recovery_std'][i]*100:.1f}%  "
              f"[{agg['recovery_min'][i]*100:.0f},{agg['recovery_max'][i]*100:.0f}]  "
              f"(b=0 ref {per_boot[0]['recovery'][i]*100:.1f}%)", flush=True)

    res = dict(layer=LAYER, alphas=ALPHAS, n_boot=N_BOOT, joint_seed_and_vector=True,
               dm_clean=dm_clean, clean_loss=clean_loss, v_norm_full=float(np.linalg.norm(v_full)),
               per_boot={str(b): per_boot[b] for b in range(N_BOOT)}, aggregate=agg,
               n_train_texts=len(train_texts), n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "33_joint_vector_seed.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    rec_m = np.array(agg["recovery_mean"]) * 100
    rec_s = np.array(agg["recovery_std"]) * 100
    rec0 = np.array(per_boot[0]["recovery"]) * 100
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].errorbar(ALPHAS, rec_m, yerr=rec_s, fmt="o-", color="#16a085", capsize=4,
                   label=f"mean ± std ({len(boots)} joint vector×seed resamples)")
    for b in boots:
        ax[0].scatter(ALPHAS, np.array(per_boot[b]["recovery"]) * 100,
                      color="#95a5a6", s=14, alpha=0.6, zorder=1)
    ax[0].plot(ALPHAS, rec0, "s--", color="#c0392b", label="b=0 original set, seed 0 (Exp 3)")
    ax[0].axhline(100, ls=":", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel("fluency recovery (%)")
    ax[0].set_title("(a) recovery under JOINT vector×seed resampling"); ax[0].legend(fontsize=8)

    # compare the three uncertainty budgets at alpha=8 (index 4)
    labels = ["seed only\n(Exp 26)", "vector only\n(Exp 32)", "joint\n(Exp 33)"]
    stds = [2.0, 2.7, rec_s[4]]
    quad = (2.0**2 + 2.7**2) ** 0.5
    ax[1].bar(labels, stds, color=["#2980b9", "#8e44ad", "#16a085"], alpha=0.85)
    ax[1].axhline(quad, ls="--", color="k", lw=1.2,
                  label=f"independent quadrature = {quad:.1f} pp")
    ax[1].set_ylabel("recovery std @α=8 (pp)")
    ax[1].set_title("(b) uncertainty budget @α=8"); ax[1].legend(fontsize=8)

    fig.suptitle(f"Joint vector×seed resample of the flagship corrector "
                 f"(GPT-2 small, layer {LAYER})")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "33_joint_vector_seed.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
