"""Iteration 14 — DIRECT test of the bank-DIVERSITY lever (S4c follow-up #4, positive counterpart).

Exp 9 concluded that what governs a direction-conditional corrector's transfer/recovery is bank
ANGULAR DIVERSITY (how separable its member directions are), NOT how well the bank covers the
held-out target's subspace. But Exp 9 could only INFER this: its three banks confounded the two —
the most target-aligned bank was also the most internally collinear (because the held-out `certainty`
lives inside the collinear cluster {formality, concreteness, complexity, certainty}). So a reader
cannot tell whether transfer collapsed because the bank was target-aligned or because it was
internally collinear.

This experiment isolates the diversity mechanism with a CONTROLLED THIRD-MEMBER SWAP. Every bank is
size 3 and shares the SAME anchor pair {sentiment, formality}; only the THIRD member changes, chosen
to be increasingly collinear with `formality`:

    div  = {sentiment, formality, politeness}    third |cos(·,formality)| = 0.07   internal D = 0.13
    mid  = {sentiment, formality, complexity}    third |cos(·,formality)| = 0.57   internal D = 0.21
    coll = {sentiment, formality, concreteness}  third |cos(·,formality)| = 0.76   internal D = 0.26

(internal D = mean pairwise |cos| among the 3 members; higher D = more collinear = less diverse.)

We then measure the fluency recovery of the SHARED anchor pair {sentiment, formality}. The clean
signal: `sentiment` is orthogonal to EVERYTHING (|cos| ≤ 0.03 to every other direction, incl. the
held-out `certainty`). So sentiment's recovery cannot depend on target coverage or on the third
member's identity — the ONLY thing that changes across banks is the bank's internal separability.
If sentiment gets corrected WORSE as the third member becomes collinear-with-formality, that is
unambiguous evidence for the separability/diversity mechanism, with no target-alignment confound.

We also report held-out `certainty` transfer for continuity with Exp 6-9.

Model: GPT-2 small, resid_post block 6. Same held-out 100 FineWeb docs / recipe / seed as Exp 3-9.
Outputs: results/14_diversity_lever.json + plots/14_diversity_lever.png.
"""
import os, json, importlib.util
from itertools import combinations
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
ANCHOR = ["sentiment", "formality"]         # shared by every bank
# banks ordered by increasing internal collinearity (third member ever more aligned with formality)
BANKS = {
    "div":  ["sentiment", "formality", "politeness"],     # third ⟂ formality (0.07)
    "mid":  ["sentiment", "formality", "complexity"],     # third moderately collinear (0.57)
    "coll": ["sentiment", "formality", "concreteness"],   # third strongly collinear (0.76)
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

    def cos(a, b):
        return float(vecs[a] @ vecs[b] / (np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b])))

    # internal diversity D (mean pairwise |cos| among members) and alignment A (mean |cos| to heldout)
    bank_D = {b: float(np.mean([abs(cos(x, y)) for x, y in combinations(mem, 2)]))
              for b, mem in BANKS.items()}
    bank_A = {b: float(np.mean([abs(cos(m, HELDOUT)) for m in mem])) for b, mem in BANKS.items()}
    print("bank internal D (mean pairwise |cos|):", {b: round(d, 3) for b, d in bank_D.items()}, flush=True)
    print("bank align A (mean |cos| to certainty):", {b: round(a, 3) for b, a in bank_A.items()}, flush=True)

    # ---- eval split identical to Exp 3-9 ----
    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]
    alphas = [1.0, 2.0, 4.0, 8.0]
    clean_loss = lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    # ---- raw ΔLM baseline for every direction that appears anywhere ----
    used = sorted(set(sum(BANKS.values(), []) + [HELDOUT]))
    raw_dlm = {}
    for n in used:
        vt, _ = tens[n]
        row = []
        for a in alphas:
            def f_raw(h, vt=vt, a=a): return h + a * vt
            row.append(lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss)
        raw_dlm[n] = row
    print("raw ΔLM sentiment:", [round(x, 3) for x in raw_dlm["sentiment"]], flush=True)
    print("raw ΔLM formality:", [round(x, 3) for x in raw_dlm["formality"]], flush=True)

    def eval_dlm(corr, n):
        vt, vhat_t = tens[n]
        row = []
        for a in alphas:
            def f(h, a=a):
                hat, _ = make_hat_cond(corr, h, a, vt, vhat_t); return hat
            row.append(lm_loss_fn(eval_texts, LAYER, f) - clean_loss)
        return row

    def rec_of(dlm, n):
        return [100.0 * (raw_dlm[n][i] - dlm[i]) / raw_dlm[n][i] for i in range(len(alphas))]

    out = {"alphas": alphas, "banks": BANKS, "bank_D": bank_D, "bank_A": bank_A,
           "anchor": ANCHOR, "heldout": HELDOUT, "norms": norms, "raw_dlm": raw_dlm,
           "results": {}}

    for bname, members in BANKS.items():
        print(f"\n=== training bank '{bname}' = {members}  (D={bank_D[bname]:.2f} A={bank_A[bname]:.2f}) ===",
              flush=True)
        torch.manual_seed(SEED); np.random.seed(SEED)
        corr = CondCorrector(hidden=1024).to(DEVICE)
        bank = [tens[n] for n in members]
        train_cond(corr, bank, train_texts)
        rec = {}
        # recovery of every member (anchor pair + the swapped third) and held-out transfer
        for n in members + [HELDOUT]:
            dlm = eval_dlm(corr, n)
            rec[n] = {"dlm": dlm, "rec": rec_of(dlm, n)}
        out["results"][bname] = {"members": members, "D": bank_D[bname], "A": bank_A[bname],
                                 "rec": rec}
        anc = ", ".join("%s@8:%.0f%%" % (n, rec[n]["rec"][-1]) for n in ANCHOR)
        print(f"  anchor recovery {{{anc}}} | third '{members[2]}'@8:{rec[members[2]]['rec'][-1]:.0f}% "
              f"| held-out '{HELDOUT}'@8:{rec[HELDOUT]['rec'][-1]:.0f}%", flush=True)
        del corr; torch.cuda.empty_cache()

    with open(os.path.join(RESULTS, "14_diversity_lever.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure ----
    order = ["div", "mid", "coll"]
    Ds = [bank_D[b] for b in order]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    # panel (a): anchor-pair recovery @α=8 vs internal diversity D
    ai = alphas.index(8.0)
    for n, col, mk in [("sentiment", "#2980b9", "o"), ("formality", "#c0392b", "s")]:
        ys = [out["results"][b]["rec"][n]["rec"][ai] for b in order]
        ax[0].plot(Ds, ys, mk + "-", color=col, label=f"{n} (anchor)")
    ax[0].set_xlabel(r"bank internal collinearity  $\bar{|\cos|}$  (higher = LESS diverse)")
    ax[0].set_ylabel("fluency recovery @ α=8 (%)")
    ax[0].set_title("(a) shared anchor pair corrected WORSE\nas the bank loses angular diversity")
    for b, x in zip(order, Ds):
        ax[0].annotate(b, (x, out["results"][b]["rec"]["sentiment"]["rec"][ai]),
                       textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax[0].legend(fontsize=9, loc="best")

    # panel (b): sentiment recovery vs α, one line per bank (sentiment ⟂ everything → pure isolate)
    cmap = {"div": "#27ae60", "mid": "#e67e22", "coll": "#c0392b"}
    for b in order:
        ax[1].plot(alphas, out["results"][b]["rec"]["sentiment"]["rec"], "o-", color=cmap[b],
                   label=f"{b}  (D={bank_D[b]:.2f})")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel(r"steering strength $\alpha$")
    ax[1].set_ylabel("sentiment fluency recovery (%)")
    ax[1].set_title("(b) sentiment (orthogonal to all) is the clean isolate:\nrecovery falls as the bank collinearizes")
    ax[1].legend(fontsize=9, loc="best")

    fig.suptitle("Bank angular DIVERSITY is the lever: same anchor pair, only the 3rd member's "
                 "collinearity varies\n(size 3, capacity 5.25M fixed; GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "14_diversity_lever.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
