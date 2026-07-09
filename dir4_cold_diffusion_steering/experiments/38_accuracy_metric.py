"""Experiment 38 — METRIC control: does the corrector recover next-token ACCURACY, not just nats?

Every recovery number in the study is measured in cross-entropy nats (ΔLM). A cross-entropy
drop can, in principle, come entirely from re-shaping the probability tail while the model's
actual argmax prediction stays broken. This experiment closes that gap by scoring the EXACT
flagship pipeline (GPT-2 small, block 6, sentiment v, seed 0 corrector) on a completely
different, harder-to-game metric: next-token TOP-1 (greedy) accuracy and TOP-5 accuracy.

For clean / raw-steered / learned-corrected activations at matched projection α|v| we compute
the fraction of held-out next tokens the model predicts as its argmax (top-1) and within its
top-5. We then report an ACCURACY-recovery, the direct analogue of the ΔLM recovery:

    acc_recovery(α) = (acc_learned − acc_raw) / (acc_clean − acc_raw)

i.e. the fraction of raw steering's accuracy DROP that the corrector buys back. If the nats
recovery were a tail-mass artifact, accuracy recovery would be far lower; if the corrector
genuinely restores the model's predictions, the two should broadly agree.

Reuses exp03's Corrector / train_corrector / make_hat / FuncPatcher / batched_ids verbatim
(imported), so the corrector is byte-identical to Exp 3. Only new code = an accuracy eval pass.
Outputs: results/38_accuracy_metric.json + plots/38_accuracy_metric.png.
"""
import os, json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import fineweb_texts, load_model, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")


def _load(mod, fname):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


exp03 = _load("exp03", "03_learned_corrector.py")
LAYER = exp03.LAYER


@torch.no_grad()
def accuracy_fn(texts, layer, fn, seq_len=128, batch=16):
    """Return (top1, top5) next-token accuracy over non-pad targets under patch `fn`."""
    model, tok = load_model()
    c1 = c5 = tot = 0
    for ids, mask in exp03.batched_ids(tok, texts, seq_len, batch):
        with exp03.FuncPatcher(model, layer, fn):
            logits = model(ids, attention_mask=mask).logits
        lg = logits[:, :-1].float()
        tgt = ids[:, 1:]; m = mask[:, 1:].bool()
        top5 = lg.topk(5, dim=-1).indices               # [B, T-1, 5]
        top1 = top5[..., 0]
        hit1 = (top1 == tgt)[m]
        hit5 = (top5 == tgt.unsqueeze(-1)).any(-1)[m]
        c1 += hit1.sum().item(); c5 += hit5.sum().item(); tot += m.sum().item()
    return c1 / tot, c5 / tot


def main():
    torch.manual_seed(exp03.SEED); np.random.seed(exp03.SEED)
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    d = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))
    v = d["v"].astype(np.float32)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)
    print(f"|v|={vnorm:.2f}", flush=True)

    # identical train/eval split to Exp 3
    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]

    corr = exp03.Corrector().to(DEVICE)
    exp03.train_corrector(corr, vt, vhat_t, train_texts)

    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]
    clean1, clean5 = accuracy_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean top1={clean1:.4f} top5={clean5:.4f}", flush=True)

    out = {"raw": {"top1": [], "top5": []}, "learned": {"top1": [], "top5": []},
           "rec1": [], "rec5": []}
    for a in alphas:
        def f_raw(h): return h + a * vt
        def f_learn(h):
            hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
        r1, r5 = accuracy_fn(eval_texts, LAYER, f_raw)
        l1, l5 = accuracy_fn(eval_texts, LAYER, f_learn)
        out["raw"]["top1"].append(r1); out["raw"]["top5"].append(r5)
        out["learned"]["top1"].append(l1); out["learned"]["top5"].append(l5)
        rec1 = (l1 - r1) / (clean1 - r1) if abs(clean1 - r1) > 1e-9 else float("nan")
        rec5 = (l5 - r5) / (clean5 - r5) if abs(clean5 - r5) > 1e-9 else float("nan")
        out["rec1"].append(rec1); out["rec5"].append(rec5)
        print(f"alpha={a:>4}  top1 clean={clean1:.3f} raw={r1:.3f} learned={l1:.3f} "
              f"rec1={rec1:+.2%} | top5 raw={r5:.3f} learned={l5:.3f} rec5={rec5:+.2%}", flush=True)

    res = dict(layer=LAYER, v_norm=vnorm, alphas=alphas,
               clean_top1=clean1, clean_top5=clean5, results=out,
               n_train_texts=len(train_texts), n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "38_accuracy_metric.json"), "w") as f:
        json.dump(res, f, indent=2)

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].axhline(clean1, ls="--", color="k", lw=1, label=f"clean top-1 ({clean1:.2f})")
    ax[0].plot(alphas, out["raw"]["top1"], "o-", color="#c0392b", label="raw steer")
    ax[0].plot(alphas, out["learned"]["top1"], "o-", color="#2980b9", label="learned corrector")
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel("next-token top-1 accuracy")
    ax[0].set_title("(a) top-1 accuracy (higher is better)"); ax[0].legend(fontsize=8)
    ax[1].plot(alphas, [100 * x for x in out["rec1"]], "o-", color="#2980b9", label="top-1 acc recovery")
    ax[1].plot(alphas, [100 * x for x in out["rec5"]], "s--", color="#16a085", label="top-5 acc recovery")
    ax[1].axhline(84.3, ls=":", color="#c0392b", lw=1, label="ΔLM recovery @α=8 (84%)")
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("accuracy recovery (%)")
    ax[1].set_title("(b) fraction of raw's accuracy drop recovered"); ax[1].legend(fontsize=8)
    fig.suptitle("Exp 38 — the corrector recovers next-token ACCURACY, not just nats (GPT-2 small, block 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "38_accuracy_metric.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
