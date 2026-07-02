"""Iteration 9 — DOES CURATING THE BANK TOWARD THE TARGET SUBSPACE CLOSE THE GAP? (S4c follow-up #3).

Exp 7 (more directions) and Exp 8 (more parameters) BOTH failed to close the held-out transfer
gap and concluded the open path is "curate the bank toward the target's SUBSPACE" — but neither
tested it. This experiment tests it directly, the clean controlled way:

    Hold BANK SIZE fixed at 3 and model CAPACITY fixed at hidden=1024 (5.25M params), and vary
    only WHICH 3 pool directions are in the bank, by their alignment to the held-out `certainty`:

      diffuse  = {sentiment, politeness, formality}     mean |cos to certainty| ≈ 0.38
      exp6     = {sentiment, formality, concreteness}    mean |cos to certainty| ≈ 0.54  (Exp 6/7 bank)
      curated  = {formality, concreteness, complexity}   mean |cos to certainty| ≈ 0.80

    All three banks are size 3, same 5.25M corrector, identical recipe / seed / data / eval.
    diffuse and curated share exactly ONE member (formality) and differ only in the other two —
    a controlled contrast. If SUBSPACE COVERAGE (not count, not params) is the binding constraint,
    held-out `certainty` transfer should rise monotonically with bank alignment:
    curated > exp6 > diffuse. The `exp6` point re-runs Exp 6/7's size-3 bank (reproducibility).

Model: GPT-2 small, resid_post block 6. Same held-out 100 FineWeb docs as Exp 3-8.
Outputs: results/09_curated_bank.json + plots/09_curated_bank.png.
"""
import os, json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import fineweb_texts, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")

# reuse Exp 6 machinery verbatim (CondCorrector / make_hat_cond / train_cond) + Exp 3 LM loss
_spec6 = importlib.util.spec_from_file_location("exp06", os.path.join(HERE, "06_conditional_bank.py"))
exp06 = importlib.util.module_from_spec(_spec6); _spec6.loader.exec_module(exp06)
CondCorrector = exp06.CondCorrector
make_hat_cond = exp06.make_hat_cond
train_cond = exp06.train_cond
exp03 = exp06.exp03
lm_loss_fn = exp03.lm_loss_fn
LAYER = exp03.LAYER

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

HELDOUT = "certainty"
BANKS = {
    "diffuse": ["sentiment", "politeness", "formality"],
    "exp6":    ["sentiment", "formality", "concreteness"],
    "curated": ["formality", "concreteness", "complexity"],
}


def main():
    print("device:", DEVICE, flush=True)

    names = ["sentiment", "formality", "concreteness", "certainty", "politeness", "complexity"]
    vecs = {}
    for n in names:
        if n == "sentiment":
            vecs[n] = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))["v"].astype(np.float32)
        else:
            vecs[n] = np.load(os.path.join(DATA, f"{n}_vec_layer6.npy")).astype(np.float32)

    def T(v):
        vhat = normalize_vector(v)
        return (torch.tensor(v, device=DEVICE), torch.tensor(vhat, device=DEVICE))
    tens = {n: T(vecs[n]) for n in names}
    norms = {n: float(np.linalg.norm(vecs[n])) for n in names}
    cos_to_h = {n: float(vecs[n] @ vecs[HELDOUT] /
                         (np.linalg.norm(vecs[n]) * np.linalg.norm(vecs[HELDOUT])))
                for n in names if n != HELDOUT}
    print("cos(dir, certainty):", {k: round(v, 3) for k, v in cos_to_h.items()}, flush=True)
    bank_align = {b: float(np.mean([abs(cos_to_h[n]) for n in mem])) for b, mem in BANKS.items()}
    print("bank mean |cos to certainty|:", {b: round(a, 3) for b, a in bank_align.items()}, flush=True)

    # ---- eval split identical to Exp 3-8 ----
    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]
    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]
    clean_loss = lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    vt_h, vhat_h = tens[HELDOUT]

    def eval_dlm_on(corr, vt, vhat_t):
        dlm = []
        for a in alphas:
            def f(h):
                hat, _ = make_hat_cond(corr, h, a, vt, vhat_t); return hat
            dlm.append(lm_loss_fn(eval_texts, LAYER, f) - clean_loss)
        return dlm

    # held-out RAW baseline (same for every bank)
    raw_dlm_h = []
    for a in alphas:
        def f_raw(h): return h + a * vt_h
        raw_dlm_h.append(lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss)
    print("held-out raw ΔLM:", [round(x, 3) for x in raw_dlm_h], flush=True)

    # in-bank RAW baseline @α=8 for every pool direction that appears in any bank
    used = sorted(set(sum(BANKS.values(), [])))
    raw8 = {}
    for n in used:
        vt, _ = tens[n]
        def fr(h): return h + 8.0 * vt
        raw8[n] = lm_loss_fn(eval_texts, LAYER, fr) - clean_loss

    out = {"alphas": alphas, "raw_dlm_heldout": raw_dlm_h, "cos_to_heldout": cos_to_h,
           "norms": norms, "heldout": HELDOUT, "banks": BANKS, "bank_align": bank_align,
           "raw8_inbank": raw8, "results": {}}

    for bname, members in BANKS.items():
        print(f"\n=== training bank '{bname}' = {members} (align {bank_align[bname]:.2f}) ===", flush=True)
        torch.manual_seed(SEED); np.random.seed(SEED)
        corr = CondCorrector(hidden=1024).to(DEVICE)
        nparam = sum(p.numel() for p in corr.parameters())
        bank = [tens[n] for n in members]
        train_cond(corr, bank, train_texts)
        # held-out transfer
        dlm_h = eval_dlm_on(corr, vt_h, vhat_h)
        rec_h = [100.0 * (raw_dlm_h[i] - dlm_h[i]) / raw_dlm_h[i] for i in range(len(alphas))]
        # in-bank per-direction recovery @α=8
        inbank8 = {}
        for n in members:
            vt, vhat_t = tens[n]
            def fb(h):
                hat, _ = make_hat_cond(corr, h, 8.0, vt, vhat_t); return hat
            b8 = lm_loss_fn(eval_texts, LAYER, fb) - clean_loss
            inbank8[n] = {"bank": b8, "rec": 100.0 * (raw8[n] - b8) / raw8[n]}
        out["results"][bname] = {
            "members": members, "align": bank_align[bname], "params_M": nparam / 1e6,
            "heldout_dlm": dlm_h, "heldout_rec": rec_h, "inbank_alpha8": inbank8}
        ib = ", ".join("%s:%.0f" % (n, inbank8[n]["rec"]) for n in members)
        print(f"  '{bname}': held-out rec {[round(x,0) for x in rec_h]}  in-bank@8 {{{ib}}}", flush=True)
        del corr; torch.cuda.empty_cache()

    # ---- native oracle on held-out (ceiling) ----
    print(f"\n=== training NATIVE oracle on held-out '{HELDOUT}' ===", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    corr_native = CondCorrector(hidden=1024).to(DEVICE)
    train_cond(corr_native, [tens[HELDOUT]], train_texts)
    native_dlm = eval_dlm_on(corr_native, vt_h, vhat_h)
    native_rec = [100.0 * (raw_dlm_h[i] - native_dlm[i]) / raw_dlm_h[i] for i in range(len(alphas))]
    out["native_dlm"] = native_dlm
    out["native_rec"] = native_rec
    print(f"  native held-out rec {[round(x,0) for x in native_rec]}", flush=True)
    del corr_native; torch.cuda.empty_cache()

    with open(os.path.join(RESULTS, "09_curated_bank.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = {"diffuse": "#95a5a6", "exp6": "#2980b9", "curated": "#c0392b"}
    order = ["diffuse", "exp6", "curated"]
    # panel (a): held-out recovery vs α, one line per bank + native oracle
    for b in order:
        r = out["results"][b]
        ax[0].plot(alphas, r["heldout_rec"], "o-", color=cmap[b],
                   label=f"{b} (|cos| {r['align']:.2f})")
    ax[0].plot(alphas, native_rec, "s--", color="#27ae60", label="native oracle (ceiling)")
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$")
    ax[0].set_ylabel("held-out fluency recovery (%)")
    ax[0].set_title(f"(a) transfer to held-out '{HELDOUT}' vs bank curation")
    ax[0].legend(fontsize=8, loc="upper right")
    # panel (b): held-out recovery vs bank alignment (α=1 and α=8)
    aligns = [out["results"][b]["align"] for b in order]
    rec1 = [out["results"][b]["heldout_rec"][0] for b in order]
    rec8 = [out["results"][b]["heldout_rec"][-1] for b in order]
    ax[1].plot(aligns, rec1, "o-", color="#8e44ad", label=r"held-out ($\alpha=1$)")
    ax[1].plot(aligns, rec8, "o-", color="#c0392b", label=r"held-out ($\alpha=8$)")
    ax[1].axhline(native_rec[-1], ls="--", color="#27ae60", label=r"native oracle ($\alpha=8$)")
    ax[1].set_xlabel(r"bank mean $|\cos|$ to held-out direction")
    ax[1].set_ylabel("held-out fluency recovery (%)")
    ax[1].set_title("(b) does curation toward the target subspace help?")
    ax[1].legend(fontsize=8, loc="best")
    for b, x in zip(order, aligns):
        ax[1].annotate(b, (x, out["results"][b]["heldout_rec"][-1]),
                       textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)
    fig.suptitle("Curating the bank toward the held-out subspace at FIXED size 3 & capacity 5.25M "
                 "(GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "09_curated_bank.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
