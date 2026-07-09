"""Experiment 34 — TOKEN-POSITION robustness of the flagship recovery.

Every recovery number in the study (the flagship 84% @a=8) pools next-token NLL over
ALL token positions in the held-out documents: recovery = 1 - Σ e_learned / Σ e_raw
summed over every position. CLAUDE.md rule 10 names "token" as one of the controls a
trustworthy metric must survive, and it is the one axis never isolated (seed / layer /
model / prompt / vector / eval-doc all were). This asks the token-position question
directly: is the 84% recovery UNIFORM across positions in the sequence, or is it an
artifact of pooling (e.g. the corrector helps only early tokens, or only late ones)?

Method: train the EXACT flagship Exp-3 corrector (GPT-2 small, block 6, sentiment v,
seed 0), then evaluate next-token NLL PER SOURCE POSITION j (0..126) on the same held-out
100 FineWeb docs (seq 128, right-padded so position j = distance from doc start), for
clean / raw / learned at a in {4, 8}. Per position/bucket:
    excess_m(j) = loss_m(j) - loss_clean(j),   recovery(j) = 1 - excess_learned(j)/excess_raw(j).
Reuses exp03 Corrector / train_corrector / make_hat / FuncPatcher / batched_ids verbatim
(DRY). Outputs results/34_token_position.json + plots/34_token_position.png.
"""
import os, json, importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exp03 = importlib.import_module("03_learned_corrector")
from common import resid_post, fineweb_texts, load_model, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")
LAYER = exp03.LAYER
SEQ = 128
ALPHAS = [4.0, 8.0]
NBINS = 8


@torch.no_grad()
def per_pos_nll(eval_texts, fn, seq_len=SEQ, batch=16):
    """Summed next-token NLL and token count per SOURCE position j (0..seq_len-2)."""
    model, tok = load_model()
    T = seq_len - 1
    tot = np.zeros(T); cnt = np.zeros(T)
    for ids, mask in exp03.batched_ids(tok, eval_texts, seq_len, batch):
        with exp03.FuncPatcher(model, LAYER, fn):
            logits = model(ids, attention_mask=mask).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]; m = mask[:, 1:].bool()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).masked_fill(~m, 0.0)
        tot += nll.sum(0).cpu().numpy()
        cnt += m.sum(0).cpu().numpy()
    return tot, cnt


def bucketize(tot, cnt, nbins=NBINS):
    """Group per-position sums into `nbins` contiguous position buckets -> (loss, mid_pos)."""
    T = len(tot)
    edges = np.linspace(0, T, nbins + 1).astype(int)
    loss, mids = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        c = cnt[a:b].sum()
        loss.append(tot[a:b].sum() / c if c > 0 else np.nan)
        mids.append((a + b) / 2.0)
    return np.array(loss), np.array(mids)


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

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

    # ---- per-position eval ----
    clean_tot, clean_cnt = per_pos_nll(eval_texts, lambda h: h)

    per_alpha = {}
    for a in ALPHAS:
        def f_raw(h, a=a):   return h + a * vt
        def f_learn(h, a=a):
            hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
        raw_tot, raw_cnt = per_pos_nll(eval_texts, f_raw)
        lrn_tot, lrn_cnt = per_pos_nll(eval_texts, f_learn)

        # aggregate (pooled over all positions) — reproduces Exp 3
        e_raw_all = (raw_tot.sum() - clean_tot.sum()) / clean_cnt.sum()
        e_lrn_all = (lrn_tot.sum() - clean_tot.sum()) / clean_cnt.sum()
        rec_all = 1.0 - e_lrn_all / e_raw_all

        # bucketed
        cl_b, mids = bucketize(clean_tot, clean_cnt)
        rw_b, _ = bucketize(raw_tot, raw_cnt)
        ln_b, _ = bucketize(lrn_tot, lrn_cnt)
        e_raw_b = rw_b - cl_b
        e_lrn_b = ln_b - cl_b
        rec_b = 1.0 - e_lrn_b / e_raw_b

        per_alpha[a] = dict(
            mids=mids.tolist(),
            excess_raw_bucket=e_raw_b.tolist(),
            excess_learned_bucket=e_lrn_b.tolist(),
            recovery_bucket=rec_b.tolist(),
            excess_raw_pooled=float(e_raw_all),
            excess_learned_pooled=float(e_lrn_all),
            recovery_pooled=float(rec_all),
        )
        print(f"alpha={a}: pooled recovery={rec_all*100:.1f}%  "
              f"bucket recovery%={np.round(rec_b*100,1).tolist()}", flush=True)

    res = dict(layer=LAYER, v_norm=vnorm, seq_len=SEQ, nbins=NBINS,
               alphas=ALPHAS, n_eval_texts=len(eval_texts),
               per_alpha={str(a): per_alpha[a] for a in ALPHAS})
    with open(os.path.join(RESULTS, "34_token_position.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    a8 = per_alpha[8.0]
    mids = np.array(a8["mids"])
    ax[0].plot(mids, a8["excess_raw_bucket"], "o-", color="#c0392b", label="raw  h+8v")
    ax[0].plot(mids, a8["excess_learned_bucket"], "o-", color="#2980b9",
               label="learned corrector")
    ax[0].axhline(0, ls="--", color="k", lw=1)
    ax[0].set_xlabel("token position in sequence")
    ax[0].set_ylabel(r"excess next-token NLL (nats) vs clean")
    ax[0].set_title(r"(a) per-position LM damage, $\alpha=8$")
    ax[0].legend(fontsize=8)
    for a, c in [(4.0, "#27ae60"), (8.0, "#2980b9")]:
        pa = per_alpha[a]
        ax[1].plot(pa["mids"], np.array(pa["recovery_bucket"]) * 100, "o-",
                   color=c, label=fr"$\alpha={int(a)}$ (pooled {pa['recovery_pooled']*100:.0f}%)")
        ax[1].axhline(pa["recovery_pooled"] * 100, ls=":", color=c, lw=1)
    ax[1].set_xlabel("token position in sequence")
    ax[1].set_ylabel("fluency recovery (%)")
    ax[1].set_title("(b) recovery vs token position")
    ax[1].set_ylim(0, 110); ax[1].legend(fontsize=8)
    fig.suptitle("Token-position robustness of the flagship recovery (GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "34_token_position.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
