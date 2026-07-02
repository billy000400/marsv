"""Experiment 15 — held-out PROMPT-FAMILY generalization.

Every prior experiment trains AND evaluates the corrector on FineWeb web text. The obvious
external-validity question left open (PLAN Next-step ii) is whether the corrector overfit to
that prompt distribution: does its fluency recovery survive on a genuinely DIFFERENT family of
prompts, whose clean activations sit in a different region of the residual stream?

Design. Train the flagship sentiment corrector EXACTLY as Experiment 3 (same vector, seed,
recipe, 300 FineWeb training docs). Then evaluate ΔLM recovery at matched projection α|v| on
three held-out prompt families of increasing distribution shift away from FineWeb:
    - fineweb   : held-out FineWeb docs (IN-DISTRIBUTION; reproduces Exp 3 — anchor).
    - markdown  : technical research prose (this project's own .md files; a different register).
    - code      : Python source (numpy / torch / transformers library files; non-NL, strongly OOD).
We quantify HOW OOD each family is by the mean Mahalanobis distance of its CLEAN activations
under the FineWeb Gaussian (fit in Exp 1/3), then ask whether recovery holds regardless.

Reuses Exp 3's Corrector / train_corrector / make_hat / lm_loss_fn / gaussian_stats /
mahalanobis / LAYER via import (no fork of the pipeline). Only the eval CORPORA change.
Model: GPT-2 small, resid_post block 6. Outputs: results/15_prompt_family.json + plot.
"""
import os, json, glob, importlib.util
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp03 = _load("exp03", "03_learned_corrector.py")
Corrector = exp03.Corrector
train_corrector = exp03.train_corrector
make_hat = exp03.make_hat
lm_loss_fn = exp03.lm_loss_fn
gaussian_stats = exp03.gaussian_stats
mahalanobis = exp03.mahalanobis
LAYER = exp03.LAYER

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


def chunk_text(paths, n_docs, chunk_chars=900, min_chars=500):
    """Read files, slice each into fixed-char windows, return up to n_docs non-trivial chunks."""
    docs = []
    for p in paths:
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for i in range(0, len(t), chunk_chars):
            c = t[i:i + chunk_chars]
            if len(c) >= min_chars and len(c.split()) >= 20:
                docs.append(c)
        if len(docs) >= n_docs * 3:
            break
    rng = np.random.RandomState(SEED)
    rng.shuffle(docs)
    return docs[:n_docs]


def build_families(n_docs=100):
    fam = {}
    # in-distribution: same held-out slice as Exp 3
    fam["fineweb"] = fineweb_texts(800)[400:500]

    # markdown: this project's technical research prose (exclude nothing — it's a distinct register)
    md_paths = sorted(glob.glob("/mars-vol/marsv/**/*.md", recursive=True))
    fam["markdown"] = chunk_text(md_paths, n_docs)

    # code: python source from installed scientific libraries (strongly OOD, non-natural-language)
    code_paths = []
    for lib in ["numpy", "torch", "transformers"]:
        try:
            m = __import__(lib)
            d = os.path.dirname(m.__file__)
            code_paths += sorted(glob.glob(os.path.join(d, "**", "*.py"), recursive=True))
        except Exception:
            pass
    rng = np.random.RandomState(SEED + 1)
    rng.shuffle(code_paths)
    fam["code"] = chunk_text(code_paths, n_docs)
    return fam


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

    # FineWeb Gaussian (Exp 1/3 fit set) for the activation-shift metric
    Hfit = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=16)
    mu, cov, prec = gaussian_stats(Hfit)

    families = build_families(n_docs=100)
    for k, docs in families.items():
        print(f"family {k}: {len(docs)} docs", flush=True)

    # how OOD is each family (clean-activation Mahalanobis under the FineWeb Gaussian)?
    shift = {}
    for k, docs in families.items():
        Hk = resid_post(docs, LAYER, seq_len=128, batch=16)
        idx = np.random.RandomState(0).choice(len(Hk), size=min(8000, len(Hk)), replace=False)
        shift[k] = float(mahalanobis(Hk[idx], mu, prec).mean())
        print(f"  clean D_M[{k}] = {shift[k]:.2f}  (mean|h|={np.linalg.norm(Hk,axis=1).mean():.1f})", flush=True)

    # ---- train the flagship corrector on FineWeb (Exp 3 recipe) ----
    train_texts = fineweb_texts(800)[500:800]
    corr = Corrector().to(DEVICE)
    print(f"corrector params: {sum(p.numel() for p in corr.parameters())/1e6:.2f}M", flush=True)
    train_corrector(corr, vt, vhat_t, train_texts)

    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]

    def raw_fn(a):
        return lambda h: h + a * vt

    def learned_fn(a):
        return lambda h: make_hat(corr, h, a, vt, vhat_t)[0]

    res = {"|v|": vnorm, "shift_DM": shift, "alphas": alphas, "families": {}}
    for k, docs in families.items():
        clean = lm_loss_fn(docs, LAYER, lambda h: h)
        rec = {"clean_loss": clean, "dlm_raw": [], "dlm_learned": [], "recovery": []}
        for a in alphas:
            lr = lm_loss_fn(docs, LAYER, raw_fn(a)) - clean
            ll = lm_loss_fn(docs, LAYER, learned_fn(a)) - clean
            recov = (1 - ll / lr) if abs(lr) > 1e-6 else float("nan")
            rec["dlm_raw"].append(lr); rec["dlm_learned"].append(ll); rec["recovery"].append(recov)
            print(f"[{k}] a={a:.0f} raw={lr:+.3f} learned={ll:+.3f} rec={recov*100:.0f}%", flush=True)
        res["families"][k] = rec

    os.makedirs(RESULTS, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    with open(os.path.join(RESULTS, "15_prompt_family.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    colors = {"fineweb": "#1f77b4", "markdown": "#2ca02c", "code": "#d62728"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for k in families:
        r = res["families"][k]
        c = colors[k]
        lab = f"{k} (D_M={shift[k]:.0f})"
        ax[0].plot(alphas, r["dlm_raw"], "--", color=c, alpha=.7)
        ax[0].plot(alphas, r["dlm_learned"], "-o", color=c, label=lab, ms=4)
        ax[1].plot(alphas, [x * 100 for x in r["recovery"]], "-o", color=c, label=lab, ms=4)
    ax[0].set_title("ΔLM vs α  (dashed=raw, solid=corrected)")
    ax[0].set_xlabel("steering strength α"); ax[0].set_ylabel("ΔLM (nats)"); ax[0].legend(fontsize=8)
    ax[0].axhline(0, color="k", lw=.5)
    ax[1].set_title("fluency recovery vs α, by prompt family")
    ax[1].set_xlabel("steering strength α"); ax[1].set_ylabel("recovery (%)"); ax[1].legend(fontsize=8)
    ax[1].axhline(100, color="k", lw=.4, ls=":"); ax[1].set_ylim(0, 110)
    # activation-shift bar
    ks = list(families.keys())
    ax[2].bar(ks, [shift[k] for k in ks], color=[colors[k] for k in ks])
    ax[2].axhline(shift["fineweb"], color="#1f77b4", ls=":", lw=1)
    ax[2].set_title("clean-activation shift under FineWeb Gaussian")
    ax[2].set_ylabel("mean Mahalanobis D_M")
    for i, k in enumerate(ks):
        ax[2].text(i, shift[k] + .3, f"{shift[k]:.1f}", ha="center", fontsize=9)
    fig.suptitle("Exp 15 — corrector trained on FineWeb, evaluated on held-out prompt families "
                 "(recovery @α=8: " +
                 ", ".join(f"{k} {res['families'][k]['recovery'][-1]*100:.0f}%" for k in ks) + ")",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PLOTS, "15_prompt_family.png"), dpi=110)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
