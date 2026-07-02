"""Iteration 5 — GENERALIZATION to a HELD-OUT steering vector (S4b).

Failure Mode 4 in the proposal: "the method overfits to one vector." Exp 3/4 trained and
evaluated the corrector on a SINGLE steering direction (DiffMean sentiment). The corrector
r_θ(h, z, α) never receives v as an explicit input — it sees the direction only implicitly
through z = h + α v. So the sharpest overfit probe is: build a SECOND, semantically different
DiffMean direction (formality: formal ↔ informal), and ask whether the sentiment-trained
corrector, applied UNCHANGED, still recovers fluency at matched projection on the new direction.

Three methods, all evaluated on the NEW formality vector v2 at matched projection α|v2|:
  - raw        : z = h + α v2                          (baseline damage)
  - transfer   : corrector trained on SENTIMENT v1, applied to v2's z   (held-out-vector test)
  - native     : corrector trained on v2, applied to v2  (direction-specific oracle / ceiling)

If `transfer` recovers a large share of raw's ΔLM damage and tracks `native`, the correction
rule generalizes across directions rather than memorizing one v. Same model/layer/eval set as
Exp 3/4. Reuses Exp 3's Corrector / training / eval helpers verbatim (imported).
Outputs: results/05_heldout_vector.json + plots/05_heldout_vector.png.
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

# reuse Exp 3 code verbatim (DRY: identical Corrector / training / eval path)
_spec = importlib.util.spec_from_file_location(
    "exp03", os.path.join(HERE, "03_learned_corrector.py"))
exp03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp03)
Corrector = exp03.Corrector
train_corrector = exp03.train_corrector
corrector_acts = exp03.corrector_acts
gaussian_stats = exp03.gaussian_stats
mahalanobis = exp03.mahalanobis
lm_loss_fn = exp03.lm_loss_fn
make_hat = exp03.make_hat
LAYER = exp03.LAYER

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

# ---- formality contrastive pairs (a concept orthogonal in meaning to sentiment) ----
FORMAL = [
    "The committee shall convene to deliberate upon the proposed amendments.",
    "We hereby acknowledge receipt of your correspondence dated the fifth instant.",
    "It is imperative that all participants adhere to the established protocol.",
    "The findings indicate a statistically significant correlation between the variables.",
    "Please be advised that the aforementioned regulations remain in full effect.",
    "The board of directors approved the resolution following extensive deliberation.",
    "Kindly ensure that the requisite documentation is submitted prior to the deadline.",
    "The organization endeavors to uphold the highest standards of professional conduct.",
    "Subsequent to the review, the recommendations shall be implemented accordingly.",
    "We respectfully request your attendance at the forthcoming general assembly.",
    "The applicant must demonstrate compliance with all statutory requirements.",
    "This report constitutes a comprehensive analysis of the fiscal situation.",
    "The undersigned hereby certifies the accuracy of the enclosed statements.",
    "Attendees are kindly requested to observe the designated procedures at all times.",
    "The institution reserves the right to amend these terms at its sole discretion.",
    "A thorough examination of the evidence corroborates the initial hypothesis.",
    "The parties shall negotiate in good faith to resolve any outstanding disputes.",
    "Pursuant to the agreement, remuneration will be disbursed on a monthly basis.",
    "The methodology was rigorously validated against an independent control cohort.",
    "We wish to express our sincere appreciation for your continued patronage.",
]
INFORMAL = [
    "So yeah the group is gonna get together and hash out those changes.",
    "Hey, just wanted to say we got your message from the other day.",
    "Look, everybody's gotta stick to the plan, no exceptions okay.",
    "Turns out the numbers kinda line up, which is pretty neat.",
    "Heads up, those rules are still a thing so don't forget.",
    "The bosses gave the thumbs up after chatting about it forever.",
    "Make sure you send over the paperwork before it's too late alright.",
    "We try real hard to keep things classy around here you know.",
    "After we looked it over we're just gonna go ahead and do the stuff.",
    "Come hang out at the big meeting thing coming up, would be cool.",
    "You gotta show you actually followed the rules and whatnot.",
    "This thing basically breaks down where all the money went.",
    "I'm telling you straight up, all this info checks out for real.",
    "Just do the normal steps and you'll be totally fine, no worries.",
    "We can switch this stuff up whenever we feel like it, honestly.",
    "Yeah we dug into it and it pretty much matches what we figured.",
    "Let's just talk it out nicely and sort out whatever's bugging us.",
    "So the deal is you get paid every month, easy as that.",
    "We double-checked the whole thing against another batch, all good.",
    "Thanks a ton for sticking with us, you guys are the best.",
]


def diffmean_vector(pos, neg):
    hp = resid_post(pos, LAYER, seq_len=32, batch=8)
    hn = resid_post(neg, LAYER, seq_len=32, batch=8)
    return (hp.mean(0) - hn.mean(0)).astype(np.float32)


def train_on(v, train_texts):
    """Return a corrector trained (α~U(0.5,8)) to correct steering along `v`."""
    vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)
    corr = Corrector().to(DEVICE)
    train_corrector(corr, vt, vhat_t, train_texts)
    return corr


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    # sentiment vector v1 (reused from Exp 1), formality vector v2 (new, held-out)
    v1 = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))["v"].astype(np.float32)
    v2 = diffmean_vector(FORMAL, INFORMAL)
    np.save(os.path.join(DATA, "formality_vec_layer6.npy"), v2)
    v2n = float(np.linalg.norm(v2)); v2hat = normalize_vector(v2)
    cos12 = float(v1 @ v2 / (np.linalg.norm(v1) * v2n))
    v2t = torch.tensor(v2, device=DEVICE); v2hat_t = torch.tensor(v2hat, device=DEVICE)
    print(f"|v1|={np.linalg.norm(v1):.2f} |v2(formality)|={v2n:.2f} cos(v1,v2)={cos12:+.3f}",
          flush=True)

    # clean Gaussian + eval/train split — identical protocol to Exp 3/4
    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=16)
    mu, cov, prec = gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(mahalanobis(Hs, mu, prec).mean())
    print(f"clean D_M={dm_clean:.2f} mean|h|={hnorm:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]

    # transfer corrector: trained on SENTIMENT v1 (this reproduces Exp 3's model exactly)
    print("training TRANSFER corrector on sentiment v1 ...", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    corr_transfer = train_on(v1, train_texts)
    # native corrector: trained on the held-out FORMALITY v2 (direction-specific oracle)
    print("training NATIVE corrector on formality v2 ...", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    corr_native = train_on(v2, train_texts)

    # ---- eval on v2 at matched projection ----
    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]
    methods = ["raw", "transfer", "native"]
    out = {m: {"dm": [], "dlm": [], "retention": []} for m in methods}

    clean_loss = lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    for a in alphas:
        def f_raw(h):  return h + a * v2t
        def f_tr(h):
            hat, _ = make_hat(corr_transfer, h, a, v2t, v2hat_t); return hat
        def f_na(h):
            hat, _ = make_hat(corr_native, h, a, v2t, v2hat_t); return hat
        fns = {"raw": f_raw, "transfer": f_tr, "native": f_na}

        Z = Hs + a * v2[None, :]
        Zt = corrector_acts(corr_transfer, Hs, a, v2t, v2hat_t)
        Zn = corrector_acts(corr_native, Hs, a, v2t, v2hat_t)
        acts = {"raw": Z, "transfer": Zt, "native": Zn}
        for m in methods:
            out[m]["dm"].append(float(mahalanobis(acts[m], mu, prec).mean()))
            out[m]["retention"].append(float(((acts[m] - Hs) @ v2hat).mean()))
            out[m]["dlm"].append(lm_loss_fn(eval_texts, LAYER, fns[m]) - clean_loss)
        print(f"alpha={a:>4}  ΔLM raw={out['raw']['dlm'][-1]:+.3f} "
              f"transfer={out['transfer']['dlm'][-1]:+.3f} native={out['native']['dlm'][-1]:+.3f}",
              flush=True)

    res = dict(layer=LAYER, v1_norm=float(np.linalg.norm(v1)), v2_norm=v2n, cos_v1_v2=cos12,
               mean_h_norm=hnorm, dm_clean=dm_clean, clean_loss=clean_loss, alphas=alphas,
               methods=methods, results=out, n_train_texts=len(train_texts),
               n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "05_heldout_vector.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    C = {"raw": "#c0392b", "transfer": "#2980b9", "native": "#27ae60"}
    Lb = {"raw": "raw steer  h+αv₂",
          "transfer": "transfer corrector (trained on sentiment v₁)",
          "native": "native corrector (trained on v₂) — oracle"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in methods:
        ax[0].plot(alphas, out[m]["dlm"], "o-", color=C[m], label=Lb[m])
    ax[0].axhline(0, ls="--", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) LM degradation on held-out vector (lower is better)")
    ax[0].legend(fontsize=8, loc="upper left")
    for m in methods:
        ax[1].plot(alphas, out[m]["dm"], "o-", color=C[m], label=Lb[m])
    ax[1].axhline(dm_clean, ls="--", color="k", lw=1, label=f"real acts ({dm_clean:.1f})")
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("mean Mahalanobis $D_M$")
    ax[1].set_title("(b) off-manifold distance"); ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Held-out steering vector: the correction rule is direction-specific — the "
                 "sentiment-trained\ncorrector does NOT transfer, but retraining natively "
                 "recovers fluency (GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "05_heldout_vector.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
