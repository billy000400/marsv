"""Experiment 37 — decompose Exp 35's OTHER token class (Exp 35's own residual question).

Exp 35 split the flagship recovery by target-token TYPE and found the two LINGUISTIC
classes recover nearly equally (FUNCTION 73.9% / CONTENT 77.5% @a=8), while the pooled
84.3% sits ABOVE both because a third catch-all class, OTHER (subword continuations +
punctuation + digits), recovers ~100% and, carrying large excess NLL, pulls the
token-weighted pool up. That leaves one honest question: is that ~100% OTHER recovery
driven by trivial PUNCTUATION (a cheap-token effect that would make the pooled headline
look better than the linguistic reality), or does it also hold on SUBWORD word-continuation
pieces (which are genuine language)? This isolates it by splitting OTHER in two:

  SUBWORD — an OTHER token whose decoded string contains at least one alphabetic char
            (mid-word continuation pieces like "ing", "tion", non-word-initial alpha).
  PUNCT   — an OTHER token with NO alphabetic char (punctuation, digits, symbols, whitespace).

FUNCTION and CONTENT are exactly Exp 35's classes (reuses exp35.build_type_map verbatim),
so the pooled recovery over all four classes reproduces Exp 3 / 34 / 35 to the digit.

Reuses exp03 Corrector / train_corrector / make_hat / FuncPatcher / batched_ids verbatim (DRY)
and exp35.build_type_map + exp35.per-type machinery pattern; only new code is the OTHER split.
Outputs results/37_other_decomposition.json + plots/37_other_decomposition.png.
"""
import os, json, importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.cuda.set_per_process_memory_fraction(0.18)
torch.set_num_threads(1)

exp03 = importlib.import_module("03_learned_corrector")
exp35 = importlib.import_module("35_token_type")
from common import fineweb_texts, load_model, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")
LAYER = exp03.LAYER
SEQ = 128
ALPHAS = [4.0, 8.0]
CLASSES = ["FUNCTION", "CONTENT", "SUBWORD", "PUNCT"]


def build_type_map4(tok):
    """Vocab-length int array: 0=FUNCTION, 1=CONTENT, 2=SUBWORD, 3=PUNCT.

    FUNCTION/CONTENT are exactly exp35's (0/1); exp35's OTHER (2) is split into
    SUBWORD (has an alphabetic char) vs PUNCT (no alphabetic char)."""
    base = exp35.build_type_map(tok)          # 0=FUNCTION, 1=CONTENT, 2=OTHER
    types = base.astype(np.int64).copy()
    for tid in np.where(base == 2)[0]:
        s = tok.decode([int(tid)])
        has_alpha = any(ch.isalpha() for ch in s)
        types[tid] = 2 if has_alpha else 3    # SUBWORD if any letter else PUNCT
    return types


@torch.no_grad()
def per_type_nll(eval_texts, fn, type_map, seq_len=SEQ, batch=16):
    """Summed next-token NLL and token count, split by TARGET token class (4 classes)."""
    model, tok = load_model()
    tot = np.zeros(len(CLASSES)); cnt = np.zeros(len(CLASSES))
    tmap = torch.tensor(type_map, device=DEVICE)
    for ids, mask in exp03.batched_ids(tok, eval_texts, seq_len, batch):
        with exp03.FuncPatcher(model, LAYER, fn):
            logits = model(ids, attention_mask=mask).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]; m = mask[:, 1:].bool()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        cls = tmap[tgt]
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

    type_map = build_type_map4(tok)
    print("vocab class counts (FUNCTION/CONTENT/SUBWORD/PUNCT):",
          [int((type_map == c).sum()) for c in range(4)], flush=True)

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

        e_raw_all = (raw_tot.sum() - clean_tot.sum()) / clean_cnt.sum()
        e_lrn_all = (lrn_tot.sum() - clean_tot.sum()) / clean_cnt.sum()
        rec_all = 1.0 - e_lrn_all / e_raw_all

        # linguistic-only pool (FUNCTION+CONTENT, i.e. exclude SUBWORD+PUNCT)
        ling = np.array([0, 1])
        e_raw_ling = (raw_tot[ling].sum() - clean_tot[ling].sum()) / clean_cnt[ling].sum()
        e_lrn_ling = (lrn_tot[ling].sum() - clean_tot[ling].sum()) / clean_cnt[ling].sum()
        rec_ling = 1.0 - e_lrn_ling / e_raw_ling

        per_alpha[a] = dict(
            classes=CLASSES, token_count=clean_cnt.tolist(),
            clean_nll=(clean_tot / clean_cnt).tolist(),
            excess_raw=e_raw.tolist(), excess_learned=e_lrn.tolist(),
            recovery=rec.tolist(),
            recovery_pooled=float(rec_all),
            recovery_linguistic=float(rec_ling),
            excess_raw_pooled=float(e_raw_all),
            excess_learned_pooled=float(e_lrn_all),
        )
        print(f"alpha={a}: pooled={rec_all*100:.1f}%  linguistic={rec_ling*100:.1f}%  "
              f"per-class rec%={dict(zip(CLASSES, np.round(rec*100,1).tolist()))}", flush=True)

    res = dict(layer=LAYER, v_norm=vnorm, seq_len=SEQ, alphas=ALPHAS,
               n_eval_texts=len(eval_texts), classes=CLASSES,
               token_count=clean_cnt.tolist(),
               per_alpha={str(a): per_alpha[a] for a in ALPHAS})
    with open(os.path.join(RESULTS, "37_other_decomposition.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(CLASSES)); w = 0.35
    for i, (a, c) in enumerate([(4.0, "#27ae60"), (8.0, "#2980b9")]):
        pa = per_alpha[a]
        ax[0].bar(x + (i - 0.5) * w, np.array(pa["recovery"]) * 100, w,
                  color=c, label=fr"$\alpha={int(a)}$ (pooled {pa['recovery_pooled']*100:.0f}%)")
        ax[0].axhline(pa["recovery_pooled"] * 100, ls=":", color=c, lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(CLASSES, fontsize=8)
    ax[0].set_ylabel("fluency recovery (%)"); ax[0].set_ylim(0, 115)
    ax[0].set_title("(a) recovery by target-token class (OTHER split)"); ax[0].legend(fontsize=8)

    a8 = per_alpha[8.0]
    ax[1].bar(x - w / 2, a8["excess_raw"], w, color="#c0392b", label="raw  h+8v")
    ax[1].bar(x + w / 2, a8["excess_learned"], w, color="#2980b9", label="learned corrector")
    ax[1].axhline(0, ls="--", color="k", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(CLASSES, fontsize=8)
    ax[1].set_ylabel("excess next-token NLL (nats) vs clean")
    ax[1].set_title(r"(b) per-class LM damage, $\alpha=8$"); ax[1].legend(fontsize=8)
    fig.suptitle("OTHER-class decomposition of the flagship recovery (GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "37_other_decomposition.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
