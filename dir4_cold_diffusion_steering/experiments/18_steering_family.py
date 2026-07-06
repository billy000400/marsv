"""Iteration 18 — Beyond hand-built DiffMean: does the ColdSteer recipe depend on
the steering-vector FAMILY? (acts on human feedback #3)

Every steering vector used so far (6 of them) is DiffMean over hand-written contrastive
sentences. The open critique: is the flagship result (raw steering breaks the LM; an
LM-supervised projection-preserving corrector recovers it) an artifact of the DiffMean
family and of hand-built prompts?

This experiment changes BOTH axes at once, on the SAME concept (sentiment):
  (A) DATA SOURCE — steering vectors are built from a REAL downloaded dataset (SST-2,
      thousands of movie-review sentences), not 20 hand-written sentences.
  (B) EXTRACTION FAMILY — we build three genuinely different steering-vector families
      from those activations, the three canonical linear-steering families:
        1. DiffMean   : v = mean(h+) - mean(h-)                (difference of means)
        2. LogReg     : weight of an L2 logistic probe pos-vs-neg (discriminative)
        3. PCA-contrast: top PC of centered pos-neg pair differences (RepE-style, unsupervised)
All three are sign-aligned to +sentiment and scaled to a COMMON norm (the DiffMean norm),
so the ONLY thing that differs across families is the DIRECTION. We then run the identical
flagship recipe (Exp 3) on each family at matched projection alpha|v| and ask whether the
corrector still recovers the fluency damage.

Reuses the Exp-3 pipeline verbatim (import). Model: GPT-2 small, resid_post block 6.
Outputs: results/18_steering_family.json + plots/18_steering_family.png.
"""
import os, json, importlib.util, urllib.request
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import resid_post, fineweb_texts, load_model, DATA, DEVICE
from projections import normalize_vector, cov_aligned_shift

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(PLOTS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)

# ---- import the flagship Exp-3 pipeline (module name starts with a digit) ----
spec = importlib.util.spec_from_file_location("exp03", os.path.join(HERE, "03_learned_corrector.py"))
exp03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp03)

LAYER = 6
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


# ---------- data: real SST-2 sentences ----------
def load_sst2(n_per_class=500):
    path = os.path.join(DATA, "sst2_train.tsv")
    if not os.path.exists(path):
        url = ("https://raw.githubusercontent.com/clairett/"
               "pytorch-sentiment-classification/master/data/SST2/train.tsv")
        print("downloading SST-2 ...", flush=True)
        urllib.request.urlretrieve(url, path)
    pos, neg = [], []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            text, lab = line.rsplit("\t", 1)
            text = text.strip()
            if len(text.split()) < 4:            # skip very short fragments
                continue
            if lab == "1" and len(pos) < n_per_class:
                pos.append(text)
            elif lab == "0" and len(neg) < n_per_class:
                neg.append(text)
            if len(pos) >= n_per_class and len(neg) >= n_per_class:
                break
    return pos, neg


@torch.no_grad()
def sent_acts(texts, layer, batch=32, seq_len=64):
    """Mean-pooled resid_post at `layer`, one vector per sentence -> [N, 768]."""
    model, tok = load_model()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                  max_length=seq_len, padding=True)
        ids = enc.input_ids.to(DEVICE); mask = enc.attention_mask.to(DEVICE)
        hs = model(ids, attention_mask=mask, output_hidden_states=True).hidden_states[layer + 1]
        m = mask.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, 0)


# ---------- three steering-vector families ----------
def dir_diffmean(Hp, Hn):
    return Hp.mean(0) - Hn.mean(0)


def dir_logreg(Hp, Hn, epochs=400, lr=0.05, wd=1e-2):
    """L2 logistic probe pos-vs-neg in per-dim-standardized space; map weight back to
    raw activation space (w_raw = w_std / std)."""
    X = np.vstack([Hp, Hn]).astype(np.float32)
    y = np.concatenate([np.ones(len(Hp)), np.zeros(len(Hn))]).astype(np.float32)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xs = torch.tensor((X - mu) / sd, device=DEVICE)
    yt = torch.tensor(y, device=DEVICE)
    w = torch.zeros(X.shape[1], device=DEVICE, requires_grad=True)
    b = torch.zeros(1, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logit = Xs @ w + b
        loss = lossf(logit, yt)
        loss.backward(); opt.step()
    w_std = w.detach().cpu().numpy()
    return (w_std / sd).astype(np.float32)          # direction in raw activation space


def dir_pca_contrast(Hp, Hn, seed=0):
    """RepE-style: top principal component of centered random pos-neg pair differences."""
    rng = np.random.RandomState(seed)
    n = min(len(Hp), len(Hn))
    perm = rng.permutation(len(Hn))[:n]
    D = Hp[:n] - Hn[perm]
    D = D - D.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[0].astype(np.float32)


def orient_scale(d, ref, target_norm):
    """Sign-align d to ref (positive-sentiment) and rescale to target_norm."""
    d = d.astype(np.float32)
    if float(d @ ref) < 0:
        d = -d
    return d / (np.linalg.norm(d) + 1e-8) * target_norm


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    # ---- build steering families from REAL data ----
    pos, neg = load_sst2(n_per_class=500)
    print(f"SST-2: {len(pos)} pos / {len(neg)} neg sentences", flush=True)
    Hp = sent_acts(pos, LAYER); Hn = sent_acts(neg, LAYER)

    v_dm_raw = dir_diffmean(Hp, Hn)
    natural_dm_norm = float(np.linalg.norm(v_dm_raw))    # DiffMean's own (sentence-pooled) norm
    base_norm = 11.0                                     # match the flagship |v| so all families
                                                         # break the LM comparably (direction is the
                                                         # only variable across families)
    ref = v_dm_raw                                        # +sentiment reference for signs
    fams = {
        "diffmean": orient_scale(v_dm_raw, ref, base_norm),
        "logreg":   orient_scale(dir_logreg(Hp, Hn), ref, base_norm),
        "pca":      orient_scale(dir_pca_contrast(Hp, Hn), ref, base_norm),
    }
    dm_unit = normalize_vector(fams["diffmean"])
    cos = {k: float(normalize_vector(v) @ dm_unit) for k, v in fams.items()}
    # cosine of SST-2 DiffMean to the hand-built DiffMean from Exp 1 (concept reproducibility)
    hand = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))["v"].astype(np.float32)
    cos_hand = float(dm_unit @ normalize_vector(hand))
    print("family |v| (matched):", {k: round(float(np.linalg.norm(v)), 2) for k, v in fams.items()}, flush=True)
    print("cos to SST2-DiffMean:", {k: round(v, 3) for k, v in cos.items()},
          "| cos(SST2-DM, hand-DM)=", round(cos_hand, 3), flush=True)

    # ---- shared Gaussian manifold + eval/train text (identical to Exp 3) ----
    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=16)
    mu, cov, prec = exp03.gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(exp03.mahalanobis(Hs, mu, prec).mean())
    print(f"clean D_M={dm_clean:.2f} mean|h|={hnorm:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]
    clean_loss = exp03.lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]
    out = {}
    for fam, v in fams.items():
        print(f"\n==== family: {fam} (|v|={np.linalg.norm(v):.2f}) ====", flush=True)
        vhat = normalize_vector(v)
        vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)

        torch.manual_seed(SEED); np.random.seed(SEED)
        corr = exp03.Corrector().to(DEVICE)
        exp03.train_corrector(corr, vt, vhat_t, train_texts)

        rec = {"raw_dlm": [], "learned_dlm": [], "dm_raw": [], "dm_learned": [],
               "retention": [], "recovery": []}
        for a in alphas:
            def f_raw(h):  return h + a * vt
            def f_learn(h):
                hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
            dlm_raw = exp03.lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss
            dlm_lrn = exp03.lm_loss_fn(eval_texts, LAYER, f_learn) - clean_loss
            Z = Hs + a * v[None, :]
            Zl = exp03.corrector_acts(corr, Hs, a, vt, vhat_t)
            rec["raw_dlm"].append(dlm_raw)
            rec["learned_dlm"].append(dlm_lrn)
            rec["dm_raw"].append(float(exp03.mahalanobis(Z, mu, prec).mean()))
            rec["dm_learned"].append(float(exp03.mahalanobis(Zl, mu, prec).mean()))
            rec["retention"].append(float(((Zl - Hs) @ vhat).mean()))
            rec["recovery"].append(1.0 - dlm_lrn / dlm_raw if abs(dlm_raw) > 1e-6 else float("nan"))
            print(f"  a={a:>4} ΔLM raw={dlm_raw:+.3f} learned={dlm_lrn:+.3f} "
                  f"rec={rec['recovery'][-1]*100:5.0f}%  D_M {rec['dm_raw'][-1]:.1f}/{rec['dm_learned'][-1]:.1f}",
                  flush=True)
        out[fam] = rec

    res = dict(layer=LAYER, base_norm=base_norm, natural_dm_norm=natural_dm_norm,
               mean_h_norm=hnorm, dm_clean=dm_clean,
               clean_loss=clean_loss, alphas=alphas, cos_to_diffmean=cos,
               cos_sst2dm_to_handdm=cos_hand, n_pos=len(pos), n_neg=len(neg),
               n_fit_tokens=int(len(H)), results=out)
    with open(os.path.join(RESULTS, "18_steering_family.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    C = {"diffmean": "#2980b9", "logreg": "#e67e22", "pca": "#27ae60"}
    Lbl = {"diffmean": f"DiffMean (cos 1.00)",
           "logreg": f"LogReg probe (cos {cos['logreg']:.2f})",
           "pca": f"PCA-contrast (cos {cos['pca']:.2f})"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for fam in fams:
        ax[0].plot(alphas, out[fam]["raw_dlm"], "o--", color=C[fam], alpha=0.55)
        ax[0].plot(alphas, out[fam]["learned_dlm"], "o-", color=C[fam], label=Lbl[fam])
    ax[0].axhline(0, ls=":", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) raw (dashed) vs corrected (solid)"); ax[0].legend(fontsize=8)
    for fam in fams:
        ax[1].plot(alphas, [100 * x for x in out[fam]["recovery"]], "o-", color=C[fam], label=Lbl[fam])
    ax[1].axhline(100, ls=":", color="k", lw=1)
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("fluency recovery (%)")
    ax[1].set_title("(b) recovery per family (higher = better)"); ax[1].set_ylim(0, 130); ax[1].legend(fontsize=8)
    for fam in fams:
        ax[2].plot(alphas, out[fam]["dm_raw"], "o--", color=C[fam], alpha=0.55)
        ax[2].plot(alphas, out[fam]["dm_learned"], "o-", color=C[fam], label=Lbl[fam])
    ax[2].axhline(dm_clean, ls=":", color="k", lw=1, label=f"real acts ({dm_clean:.0f})")
    ax[2].set_xlabel(r"steering strength $\alpha$"); ax[2].set_ylabel(r"Mahalanobis $D_M$")
    ax[2].set_title("(c) off-manifold: raw (dashed) vs corrected (solid)"); ax[2].legend(fontsize=8)
    fig.suptitle("ColdSteer recipe across three steering-vector families (real SST-2, matched norm/projection, GPT-2 small L6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "18_steering_family.png"), dpi=120)
    plt.close(fig)
    print("\nsaved figure + results", flush=True)


if __name__ == "__main__":
    main()
