"""Iteration 7 — DOES A DENSER BANK CLOSE THE HELD-OUT GAP? (S4c follow-up).

Exp 6 built ONE direction-conditional corrector r_θ(h, z, v̂, α) on a 3-vector bank
{sentiment, formality, concreteness} and found it PARTIALLY transfers to a held-out
direction (certainty): recovery 51% @α=1 → only 7% @α=8. The obvious question left open:

    If we make the BANK DENSER (more training directions), does the held-out transfer
    improve — especially at strong steering (α=8), where the 3-vector bank collapsed to 7%?

We hold `certainty` out as before and sweep the training-bank size ∈ {1, 3, 5} using NESTED
subsets of an ordered 5-direction pool:
    1: [sentiment]
    3: [sentiment, formality, concreteness]            (== Exp 6's bank)
    5: [+ politeness, + complexity]                    (two NEW DiffMean directions)
Every bank trains the SAME conditional architecture / recipe / seed / data as Exp 6; the
native oracle (retrained on certainty) is the per-direction ceiling. We report recovery on
held-out certainty vs bank size, and confirm the size-5 model still corrects all in-bank
directions.

Model: GPT-2 small, resid_post block 6. Same held-out 100 FineWeb docs as Exp 3/4/5/6.
Outputs: results/07_bank_scaling.json + plots/07_bank_scaling.png.
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

# reuse Exp 6 machinery verbatim (CondCorrector / make_hat_cond / train_cond / cond_acts)
_spec6 = importlib.util.spec_from_file_location("exp06", os.path.join(HERE, "06_conditional_bank.py"))
exp06 = importlib.util.module_from_spec(_spec6); _spec6.loader.exec_module(exp06)
CondCorrector = exp06.CondCorrector
make_hat_cond = exp06.make_hat_cond
train_cond = exp06.train_cond
cond_acts = exp06.cond_acts
diffmean_vector = exp06.diffmean_vector
# Exp 3 helpers (LM loss / Gaussian / Mahalanobis / layer)
exp03 = exp06.exp03
lm_loss_fn = exp03.lm_loss_fn
gaussian_stats = exp03.gaussian_stats
mahalanobis = exp03.mahalanobis
LAYER = exp03.LAYER

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

# ---- new concept: politeness (courteous/deferential ↔ blunt/rude) ----
POLITE = [
    "Would you mind terribly passing me the salt when you have a moment?",
    "I truly appreciate your help, and I'm sorry to trouble you again.",
    "Please forgive the interruption, but might I ask a quick question?",
    "Thank you so much for waiting; I'll be right with you shortly.",
    "If it isn't too much bother, could you review this at your leisure?",
    "I'd be very grateful if you could kindly consider my request.",
    "Excuse me, I hope you don't mind me stepping in for just a second.",
    "It would mean a great deal to me if you could spare some time.",
    "Many thanks for your patience; your understanding is deeply appreciated.",
    "May I gently suggest that we revisit this point together later?",
    "I'm terribly sorry for the confusion; please allow me to explain.",
    "Would it be at all possible to reschedule, if that suits you?",
    "Thank you kindly for your thoughtful and considerate response.",
    "I hope this message finds you well; I look forward to your reply.",
    "Please do let me know if there's anything I can do to help you.",
    "It's very kind of you to offer; I would be honored to accept.",
]
RUDE = [
    "Give me the salt now and stop wasting my time with this nonsense.",
    "I don't care about your excuses; just get the job done already.",
    "Move it, I need to get through and you're standing in the way.",
    "You're wrong, obviously, and I shouldn't have to explain why.",
    "Hurry up, nobody has all day to wait around for you to finish.",
    "That's a stupid question and you should have figured it out yourself.",
    "Whatever, I don't have time for this pointless conversation anymore.",
    "Just do what I said and quit complaining about every little thing.",
    "Your work is sloppy and frankly I'm not impressed at all.",
    "Stop talking, nobody here actually wants to hear your opinion.",
    "Deal with it yourself; that's not my problem to solve for you.",
    "You clearly have no idea what you're doing, so step aside.",
    "I already told you once, so pay attention this time around.",
    "Get out of my way, I have far more important things to do.",
    "That's the dumbest idea I've heard all week, honestly ridiculous.",
    "Figure it out on your own; I'm done wasting my breath on you.",
]

# ---- new concept: complexity (elaborate/nested ↔ plain/simple) ----
COMPLEX = [
    "The committee, having deliberated at length over the contested provisions, ultimately concluded that amendment was warranted.",
    "Notwithstanding the aforementioned constraints, the analysis, once suitably generalized, yields a broadly applicable framework.",
    "The phenomenon, insofar as it can be characterized, emerges from the interaction of several mutually reinforcing mechanisms.",
    "Although the hypothesis was initially plausible, subsequent scrutiny revealed inconsistencies that necessitated its revision.",
    "The architecture, comprising numerous interdependent modules, coordinates its subsystems through a hierarchical control scheme.",
    "Insofar as the evidence permits inference, the causal pathway appears mediated by multiple intervening latent variables.",
    "The negotiation, protracted by competing interests, culminated in a compromise satisfying none of the parties entirely.",
    "Given the intricate interdependencies among the components, isolating any single contributing factor proves exceedingly difficult.",
    "The theory, elegant in its abstraction yet cumbersome in application, resists straightforward empirical falsification.",
    "Having accounted for the confounding influences, the residual variance remains attributable to unspecified stochastic processes.",
    "The manuscript, dense with qualifications and subordinate clauses, demands considerable patience from its intended readership.",
    "The regulatory regime, layered across overlapping jurisdictions, generates ambiguities that litigants routinely exploit.",
    "The proof proceeds by a sequence of nested inductions, each contingent upon the lemmas established previously.",
    "The ecosystem's resilience derives from redundancies distributed across trophic levels and functional guilds.",
    "The instrument, calibrated against multiple reference standards, achieves precision at the cost of operational complexity.",
    "The doctrine, evolving through successive judicial reinterpretations, now encompasses considerations its framers never anticipated.",
]
SIMPLE = [
    "The dog ran fast. It was happy. It liked the park.",
    "I ate lunch. Then I went home. I was tired.",
    "The sky is blue. The sun is out. It is warm today.",
    "She has a cat. The cat is small. It sleeps a lot.",
    "We went to the store. We bought bread. We came back.",
    "He is tall. He plays ball. He is good at it.",
    "The car is red. It is new. Dad drives it.",
    "I like tea. I drink it hot. It tastes good.",
    "The book is long. I read a bit. I will read more.",
    "Rain fell all day. The street got wet. Then it stopped.",
    "My room is clean. The bed is made. The floor is clear.",
    "The kids play games. They laugh a lot. They have fun.",
    "The soup is hot. I blow on it. Then I eat.",
    "Birds sing at dawn. The air is cool. I go for a walk.",
    "The shop opens at nine. I get there early. I wait outside.",
    "The lamp is on. The room is bright. I can read now.",
]


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    # ---- directions: reuse saved sentiment/formality/concreteness/certainty; build 2 new ----
    v_sent = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))["v"].astype(np.float32)
    v_form = np.load(os.path.join(DATA, "formality_vec_layer6.npy")).astype(np.float32)
    v_conc = np.load(os.path.join(DATA, "concreteness_vec_layer6.npy")).astype(np.float32)
    v_cert = np.load(os.path.join(DATA, "certainty_vec_layer6.npy")).astype(np.float32)  # held-out
    v_pol = diffmean_vector(POLITE, RUDE)
    v_cplx = diffmean_vector(COMPLEX, SIMPLE)
    np.save(os.path.join(DATA, "politeness_vec_layer6.npy"), v_pol)
    np.save(os.path.join(DATA, "complexity_vec_layer6.npy"), v_cplx)

    vecs = {"sentiment": v_sent, "formality": v_form, "concreteness": v_conc,
            "politeness": v_pol, "complexity": v_cplx, "certainty": v_cert}
    pool = ["sentiment", "formality", "concreteness", "politeness", "complexity"]  # ordered pool
    heldout = "certainty"
    bank_sizes = [1, 3, 5]

    def T(v):
        vhat = normalize_vector(v)
        return (torch.tensor(v, device=DEVICE), torch.tensor(vhat, device=DEVICE), vhat)

    tens = {n: T(vecs[n]) for n in vecs}
    norms = {n: float(np.linalg.norm(vecs[n])) for n in vecs}
    # cosine of each pool direction to the held-out direction (does denser = closer coverage?)
    cos_to_heldout = {n: float(vecs[n] @ vecs[heldout] /
                               (np.linalg.norm(vecs[n]) * np.linalg.norm(vecs[heldout])))
                      for n in pool}
    print("norms:", {n: round(norms[n], 1) for n in vecs}, flush=True)
    print("cos(dir, certainty):", {k: round(v, 3) for k, v in cos_to_heldout.items()}, flush=True)

    # ---- clean Gaussian + eval split (identical protocol to Exp 3/4/5/6) ----
    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=16)
    mu, cov, prec = gaussian_stats(H)
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(mahalanobis(Hs, mu, prec).mean())
    print(f"clean D_M={dm_clean:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]
    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]
    clean_loss = lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    vt_h, vhat_th, vhat_h = tens[heldout]

    def eval_dlm_on_heldout(corr):
        dlm = []
        for a in alphas:
            def f(h):
                hat, _ = make_hat_cond(corr, h, a, vt_h, vhat_th); return hat
            dlm.append(lm_loss_fn(eval_texts, LAYER, f) - clean_loss)
        return dlm

    raw_dlm_h = []
    for a in alphas:
        def f_raw(h): return h + a * vt_h
        raw_dlm_h.append(lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss)
    print("held-out raw ΔLM:", [round(x, 3) for x in raw_dlm_h], flush=True)

    # ---- train ONE conditional corrector per bank size; eval transfer to held-out ----
    out = {"alphas": alphas, "raw_dlm": raw_dlm_h, "cos_to_heldout": cos_to_heldout,
           "norms": norms, "pool": pool, "heldout": heldout, "bank_sizes": bank_sizes,
           "bank": {}}
    corr_size5 = None
    for k in bank_sizes:
        bank_names = pool[:k]
        bank = [(tens[n][0], tens[n][1]) for n in bank_names]
        print(f"\n=== training bank size {k}: {bank_names} ===", flush=True)
        torch.manual_seed(SEED); np.random.seed(SEED)
        corr = CondCorrector().to(DEVICE)
        train_cond(corr, bank, train_texts)
        dlm = eval_dlm_on_heldout(corr)
        rec = [100.0 * (raw_dlm_h[i] - dlm[i]) / raw_dlm_h[i] for i in range(len(alphas))]
        out["bank"][str(k)] = {"names": bank_names, "heldout_dlm": dlm, "heldout_rec": rec}
        print(f"  size {k}: held-out ΔLM {[round(x,3) for x in dlm]}  "
              f"rec {[round(x,0) for x in rec]}", flush=True)
        if k == 5:
            corr_size5 = corr

    # ---- native oracle on the held-out direction (ceiling) ----
    print(f"\n=== training NATIVE oracle on held-out '{heldout}' ===", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    corr_native = CondCorrector().to(DEVICE)
    train_cond(corr_native, [(tens[heldout][0], tens[heldout][1])], train_texts)
    native_dlm = eval_dlm_on_heldout(corr_native)
    native_rec = [100.0 * (raw_dlm_h[i] - native_dlm[i]) / raw_dlm_h[i] for i in range(len(alphas))]
    out["native_dlm"] = native_dlm
    out["native_rec"] = native_rec
    print(f"  native held-out ΔLM {[round(x,3) for x in native_dlm]}  "
          f"rec {[round(x,0) for x in native_rec]}", flush=True)

    # ---- confirm the size-5 model still corrects ALL 5 in-bank directions (α=8) ----
    inbank8 = {}
    for n in pool:
        vt, vhat_t, _ = tens[n]
        def f_raw(h): return h + 8.0 * vt
        def f_bank(h):
            hat, _ = make_hat_cond(corr_size5, h, 8.0, vt, vhat_t); return hat
        r8 = lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss
        b8 = lm_loss_fn(eval_texts, LAYER, f_bank) - clean_loss
        inbank8[n] = {"raw": r8, "bank": b8, "rec": 100.0 * (r8 - b8) / r8}
        print(f"  in-bank(size5) {n:12} α=8 raw={r8:+.3f} bank={b8:+.3f} "
              f"rec={inbank8[n]['rec']:.0f}%", flush=True)
    out["inbank_alpha8_size5"] = inbank8

    with open(os.path.join(RESULTS, "07_bank_scaling.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # panel (a): held-out recovery vs α, one line per bank size + native oracle
    cmap = {1: "#f1c40f", 3: "#e67e22", 5: "#c0392b"}
    for k in bank_sizes:
        ax[0].plot(alphas, out["bank"][str(k)]["heldout_rec"], "o-", color=cmap[k],
                   label=f"bank size {k}")
    ax[0].plot(alphas, native_rec, "s--", color="#27ae60", label="native oracle")
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$")
    ax[0].set_ylabel("held-out fluency recovery (%)")
    ax[0].set_title(f"(a) transfer to held-out '{heldout}' vs bank size")
    ax[0].legend(fontsize=8, loc="upper right")
    # panel (b): held-out recovery at α=1 and α=8 vs bank size (the headline curve)
    rec_a1 = [out["bank"][str(k)]["heldout_rec"][0] for k in bank_sizes]
    rec_a8 = [out["bank"][str(k)]["heldout_rec"][-1] for k in bank_sizes]
    ax[1].plot(bank_sizes, rec_a1, "o-", color="#2980b9", label=r"$\alpha=1$ (weak)")
    ax[1].plot(bank_sizes, rec_a8, "o-", color="#c0392b", label=r"$\alpha=8$ (strong)")
    ax[1].axhline(native_rec[-1], ls="--", color="#27ae60",
                  label=r"native oracle $\alpha=8$")
    ax[1].set_xlabel("training-bank size (# directions)")
    ax[1].set_ylabel("held-out fluency recovery (%)")
    ax[1].set_xticks(bank_sizes)
    ax[1].set_title(f"(b) does a denser bank close the '{heldout}' gap?")
    ax[1].legend(fontsize=8, loc="best")
    for x, y in zip(bank_sizes, rec_a8):
        ax[1].text(x, y + 3, f"{y:.0f}%", ha="center", fontsize=9, color="#c0392b")
    fig.suptitle("Scaling the vector bank: held-out transfer vs number of training directions "
                 "(GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "07_bank_scaling.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
