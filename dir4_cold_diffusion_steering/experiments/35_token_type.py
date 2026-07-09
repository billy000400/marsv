"""Experiment 35 — TOKEN-TYPE breakdown of the flagship recovery (Exp 34's Next check).

Experiment 34 showed the flagship 84% recovery is flat across token POSITION (where in the
sequence a token sits). It left open the complementary metric-control question its own
"Next check" named: is the recovery uniform across token TYPE? Function words (the, of, a,
is) are high-frequency, low-information, and easy to predict; content words (nouns, verbs,
adjectives) carry the meaning and are where an off-manifold steer might do its real damage.
If the corrector only bought back fluency on cheap function words while leaving content-word
prediction broken, the pooled 84% would overstate what matters. This isolates that.

Method: train the EXACT flagship Exp-3 corrector (GPT-2 small, block 6, sentiment v, seed 0),
then evaluate next-token NLL on the SAME held-out 100 FineWeb docs, but accumulate the summed
NLL separately by the TYPE of the TARGET token (the token being predicted), for clean / raw /
learned at a in {4, 8}. Per type:
    excess_m = (Σ nll_m − Σ nll_clean) / count,   recovery = 1 − excess_learned / excess_raw.

Token-type classes (precomputed over the whole GPT-2 vocab from each token's decoded string):
  FUNCTION — word-initial (leading space) closed-class function word (a fixed stop-list).
  CONTENT  — word-initial alphabetic token that is NOT a function word (open-class content word).
  OTHER    — subword continuations, punctuation, digits (reported for completeness, not the contrast).

Reuses exp03 Corrector / train_corrector / make_hat / FuncPatcher / batched_ids verbatim (DRY);
the only new code is the vocab token-type map + a per-type NLL accumulator.
Outputs results/35_token_type.json + plots/35_token_type.png.
"""
import os, json, importlib, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exp03 = importlib.import_module("03_learned_corrector")
from common import fineweb_texts, load_model, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")
LAYER = exp03.LAYER
SEQ = 128
ALPHAS = [4.0, 8.0]
CLASSES = ["FUNCTION", "CONTENT", "OTHER"]

# Standard closed-class English function words (articles, prepositions, pronouns,
# conjunctions, auxiliaries, determiners, particles). Content = everything else alphabetic.
FUNCTION_WORDS = set("""
a an the this that these those my your his her its our their whose which what
i you he she it we they me him us them mine yours hers ours theirs myself yourself
himself herself itself ourselves themselves
of in on at by to from with for about into over under above below between among
through during before after since until against within without across behind beyond
and or but nor so yet for because although though while whereas if unless until
whether as than then once when where why how
is am are was were be been being do does did doing have has had having
can could shall should will would may might must ought need dare
not no nor none neither either both all any some each every few many much more most
several such only just even also too very quite rather almost
he's she's it's that's there's here's what's who's
here there where up down out off over on in
""".split())


def build_type_map(tok):
    """Vocab-length int array: 0=FUNCTION, 1=CONTENT, 2=OTHER, keyed by token id."""
    vocab = tok.get_vocab()  # token-string -> id
    n = max(vocab.values()) + 1
    types = np.full(n, 2, dtype=np.int8)  # default OTHER
    for tid in range(n):
        s = tok.decode([tid])
        if not s.startswith(" "):        # not word-initial -> subword/punct start
            continue
        core = s.strip().lower()
        if not core.isalpha():           # digits / punctuation / mixed
            continue
        types[tid] = 0 if core in FUNCTION_WORDS else 1
    return types


@torch.no_grad()
def per_type_nll(eval_texts, fn, type_map, seq_len=SEQ, batch=16):
    """Summed next-token NLL and token count, split by TARGET token type."""
    model, tok = load_model()
    tot = np.zeros(len(CLASSES)); cnt = np.zeros(len(CLASSES))
    tmap = torch.tensor(type_map.astype(np.int64), device=DEVICE)
    for ids, mask in exp03.batched_ids(tok, eval_texts, seq_len, batch):
        with exp03.FuncPatcher(model, LAYER, fn):
            logits = model(ids, attention_mask=mask).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]; m = mask[:, 1:].bool()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        cls = tmap[tgt]                                   # (B, T-1) class per target token
        for c in range(len(CLASSES)):
            sel = m & (cls == c)
            tot[c] += nll.masked_fill(~sel, 0.0).sum().item()
            cnt[c] += sel.sum().item()
    return tot, cnt


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    type_map = build_type_map(tok)
    print("vocab type counts (FUNCTION/CONTENT/OTHER):",
          [int((type_map == c).sum()) for c in range(3)], flush=True)

    d = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))
    v = d["v"].astype(np.float32)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)
    print(f"|v|={vnorm:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]      # identical to Exp 3
    eval_texts = all_texts[400:500]

    corr = exp03.Corrector().to(DEVICE)
    exp03.train_corrector(corr, vt, vhat_t, train_texts)

    clean_tot, clean_cnt = per_type_nll(eval_texts, lambda h: h, type_map)

    per_alpha = {}
    for a in ALPHAS:
        def f_raw(h, a=a):   return h + a * vt
        def f_learn(h, a=a):
            hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
        raw_tot, _ = per_type_nll(eval_texts, f_raw, type_map)
        lrn_tot, _ = per_type_nll(eval_texts, f_learn, type_map)

        e_raw = (raw_tot - clean_tot) / clean_cnt
        e_lrn = (lrn_tot - clean_tot) / clean_cnt
        rec = 1.0 - e_lrn / e_raw

        # pooled over FUNCTION+CONTENT+OTHER (reproduces Exp 3 / Exp 34)
        e_raw_all = (raw_tot.sum() - clean_tot.sum()) / clean_cnt.sum()
        e_lrn_all = (lrn_tot.sum() - clean_tot.sum()) / clean_cnt.sum()
        rec_all = 1.0 - e_lrn_all / e_raw_all

        per_alpha[a] = dict(
            classes=CLASSES, token_count=clean_cnt.tolist(),
            clean_nll=(clean_tot / clean_cnt).tolist(),
            excess_raw=e_raw.tolist(), excess_learned=e_lrn.tolist(),
            recovery=rec.tolist(),
            recovery_pooled=float(rec_all),
            excess_raw_pooled=float(e_raw_all),
            excess_learned_pooled=float(e_lrn_all),
        )
        print(f"alpha={a}: pooled rec={rec_all*100:.1f}%  "
              f"per-type rec%={dict(zip(CLASSES, np.round(rec*100,1).tolist()))}", flush=True)

    res = dict(layer=LAYER, v_norm=vnorm, seq_len=SEQ, alphas=ALPHAS,
               n_eval_texts=len(eval_texts), classes=CLASSES,
               token_count=clean_cnt.tolist(),
               per_alpha={str(a): per_alpha[a] for a in ALPHAS})
    with open(os.path.join(RESULTS, "35_token_type.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(CLASSES)); w = 0.35
    for i, (a, c) in enumerate([(4.0, "#27ae60"), (8.0, "#2980b9")]):
        pa = per_alpha[a]
        ax[0].bar(x + (i - 0.5) * w, np.array(pa["recovery"]) * 100, w,
                  color=c, label=fr"$\alpha={int(a)}$ (pooled {pa['recovery_pooled']*100:.0f}%)")
        ax[0].axhline(pa["recovery_pooled"] * 100, ls=":", color=c, lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(CLASSES)
    ax[0].set_ylabel("fluency recovery (%)"); ax[0].set_ylim(0, 110)
    ax[0].set_title("(a) recovery by target-token type"); ax[0].legend(fontsize=8)

    a8 = per_alpha[8.0]
    ax[1].bar(x - w / 2, a8["excess_raw"], w, color="#c0392b", label="raw  h+8v")
    ax[1].bar(x + w / 2, a8["excess_learned"], w, color="#2980b9", label="learned corrector")
    ax[1].axhline(0, ls="--", color="k", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(CLASSES)
    ax[1].set_ylabel("excess next-token NLL (nats) vs clean")
    ax[1].set_title(r"(b) per-type LM damage, $\alpha=8$"); ax[1].legend(fontsize=8)
    fig.suptitle("Token-type breakdown of the flagship recovery (GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "35_token_type.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
