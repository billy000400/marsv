"""Experiment 36 — CONTENT-word FREQUENCY breakdown (Exp 35's Next check).

Experiment 35 split the flagship recovery by token TYPE and found content words
(meaning-bearing, largest absolute raw damage) recover at least as well as cheap
function words. Its own "Next check" asked to refine the CONTENT class further. A
part-of-speech split (noun/verb/adjective) needs an in-context tagger that GPT-2's
word-piece tokens do not support reliably, so we take an OBJECTIVE, fully-controlled
cut instead: split CONTENT by how COMMON the target token is in the eval corpus.

Rare content tokens are the surprising, information-rich words where an off-manifold
steer is most likely to break prediction; common content tokens are easier. If the
pooled 84% recovery were carried by easy common content words while rare ones stayed
broken, that would matter. This isolates it.

Method: identical to Exp 35 (train the EXACT flagship Exp-3 corrector, GPT-2 small,
block 6, sentiment v, seed 0; accumulate summed next-token NLL split by TARGET-token
class on the held-out 100 FineWeb docs). We reuse Exp 35's FUNCTION / CONTENT / OTHER
map, then split CONTENT into CONTENT_COMMON / CONTENT_RARE by a corpus-frequency
threshold: for each content token id we count its occurrences as a TARGET over the eval
set, rank content tokens by count, and cut at the token-WEIGHTED median (so the two
buckets carry ~equal numbers of predicted tokens). Per class:
    excess_m = (Σ nll_m − Σ nll_clean) / count,  recovery = 1 − excess_learned / excess_raw.

Reuses exp35.build_type_map / exp35.per_type_nll machinery pattern + exp03 verbatim (DRY).
Outputs results/36_content_frequency.json + plots/36_content_frequency.png.
"""
import os, json, importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# FUNCTION, CONTENT split into COMMON/RARE, OTHER (OTHER kept for the pooled check)
CLASSES = ["FUNCTION", "CONTENT_COMMON", "CONTENT_RARE", "OTHER"]


@torch.no_grad()
def target_counts(eval_texts, seq_len=SEQ, batch=16):
    """Occurrence count of each token id used as a next-token TARGET over the eval set."""
    model, tok = load_model()
    n = max(tok.get_vocab().values()) + 1
    cnt = np.zeros(n, dtype=np.int64)
    for ids, mask in exp03.batched_ids(tok, eval_texts, seq_len, batch):
        tgt = ids[:, 1:][mask[:, 1:].bool()]
        ic = torch.bincount(tgt, minlength=n).cpu().numpy()
        cnt += ic
    return cnt


def refine_content_by_frequency(base_map, counts):
    """base_map: 0=FUNCTION 1=CONTENT 2=OTHER (Exp 35). Return a 4-class map:
    0=FUNCTION, 1=CONTENT_COMMON, 2=CONTENT_RARE, 3=OTHER, splitting CONTENT at the
    token-weighted median frequency so the two content buckets carry ~equal target counts."""
    is_content = base_map == 1
    c_ids = np.nonzero(is_content)[0]
    c_cnt = counts[c_ids]
    keep = c_cnt > 0                      # only content tokens that actually occur
    order = np.argsort(-c_cnt[keep])      # most frequent first
    ids_sorted = c_ids[keep][order]
    cnt_sorted = c_cnt[keep][order]
    cum = np.cumsum(cnt_sorted)
    half = cum[-1] / 2.0
    split = int(np.searchsorted(cum, half))  # first `split` (most frequent) = COMMON
    common_ids = set(ids_sorted[:split].tolist())

    m = np.full_like(base_map, 3)         # default OTHER
    m[base_map == 0] = 0                  # FUNCTION
    for tid in c_ids:
        if base_map[tid] != 1:
            continue
        if counts[tid] == 0:
            m[tid] = 3                     # unseen content token -> OTHER bucket (no targets anyway)
        else:
            m[tid] = 1 if tid in common_ids else 2
    thresh = int(cnt_sorted[split - 1]) if split > 0 else 0
    return m, thresh


@torch.no_grad()
def per_class_nll(eval_texts, fn, type_map, n_classes, seq_len=SEQ, batch=16):
    model, tok = load_model()
    tot = np.zeros(n_classes); cnt = np.zeros(n_classes)
    tmap = torch.tensor(type_map.astype(np.int64), device=DEVICE)
    for ids, mask in exp03.batched_ids(tok, eval_texts, seq_len, batch):
        with exp03.FuncPatcher(model, LAYER, fn):
            logits = model(ids, attention_mask=mask).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]; m = mask[:, 1:].bool()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        cls = tmap[tgt]
        for c in range(n_classes):
            sel = m & (cls == c)
            tot[c] += nll.masked_fill(~sel, 0.0).sum().item()
            cnt[c] += sel.sum().item()
    return tot, cnt


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]      # identical to Exp 3
    eval_texts = all_texts[400:500]

    base_map = exp35.build_type_map(tok)  # 0/1/2 FUNCTION/CONTENT/OTHER
    counts = target_counts(eval_texts)
    type_map, thresh = refine_content_by_frequency(base_map, counts)
    print("class target counts wanted ~equal for the two content buckets:", thresh, flush=True)

    d = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))
    v = d["v"].astype(np.float32)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)
    print(f"|v|={vnorm:.2f}", flush=True)

    corr = exp03.Corrector().to(DEVICE)
    exp03.train_corrector(corr, vt, vhat_t, train_texts)

    nC = len(CLASSES)
    clean_tot, clean_cnt = per_class_nll(eval_texts, lambda h: h, type_map, nC)
    print("class counts (FUNCTION/CC/CR/OTHER):", clean_cnt.tolist(), flush=True)

    per_alpha = {}
    for a in ALPHAS:
        def f_raw(h, a=a):   return h + a * vt
        def f_learn(h, a=a):
            hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
        raw_tot, _ = per_class_nll(eval_texts, f_raw, type_map, nC)
        lrn_tot, _ = per_class_nll(eval_texts, f_learn, type_map, nC)

        e_raw = (raw_tot - clean_tot) / clean_cnt
        e_lrn = (lrn_tot - clean_tot) / clean_cnt
        rec = 1.0 - e_lrn / e_raw

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
              f"per-class rec%={dict(zip(CLASSES, np.round(rec*100,1).tolist()))}", flush=True)

    res = dict(layer=LAYER, v_norm=vnorm, seq_len=SEQ, alphas=ALPHAS,
               n_eval_texts=len(eval_texts), classes=CLASSES,
               content_common_min_count=thresh,
               token_count=clean_cnt.tolist(),
               per_alpha={str(a): per_alpha[a] for a in ALPHAS})
    with open(os.path.join(RESULTS, "36_content_frequency.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = np.arange(len(CLASSES)); w = 0.35
    labels = ["FUNCTION", "CONTENT\ncommon", "CONTENT\nrare", "OTHER"]
    for i, (a, c) in enumerate([(4.0, "#27ae60"), (8.0, "#2980b9")]):
        pa = per_alpha[a]
        ax[0].bar(x + (i - 0.5) * w, np.array(pa["recovery"]) * 100, w,
                  color=c, label=fr"$\alpha={int(a)}$ (pooled {pa['recovery_pooled']*100:.0f}%)")
        ax[0].axhline(pa["recovery_pooled"] * 100, ls=":", color=c, lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("fluency recovery (%)"); ax[0].set_ylim(0, 120)
    ax[0].set_title("(a) recovery by content-word frequency"); ax[0].legend(fontsize=8)

    a8 = per_alpha[8.0]
    ax[1].bar(x - w / 2, a8["excess_raw"], w, color="#c0392b", label="raw  h+8v")
    ax[1].bar(x + w / 2, a8["excess_learned"], w, color="#2980b9", label="learned corrector")
    ax[1].axhline(0, ls="--", color="k", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("excess next-token NLL (nats) vs clean")
    ax[1].set_title(r"(b) per-class LM damage, $\alpha=8$"); ax[1].legend(fontsize=8)
    fig.suptitle("Content-word frequency breakdown of the flagship recovery (GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "36_content_frequency.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
