"""Iteration 31 — EVAL-SET SAMPLING control: bootstrap the flagship recovery over
held-out documents.

Every prior confidence interval in this study (Exp 26–30) varies the OPTIMIZATION seed
— i.e. it bounds how much the *trained corrector* wanders with random init/data order.
None bounds the *other* source of noise: the 100 held-out eval documents are themselves a
finite sample, so the reported recovery could shift if we had drawn a different 100 docs.
This is exactly the sampling-axis control PLAN Next-step (i) names and CLAUDE.md rule 10
asks for.

Method: train the EXACT flagship Exp-3 corrector (GPT-2 small, resid_post block 6, seed 0),
then evaluate ΔLM PER DOCUMENT (summed NLL + token count per doc) for raw and learned at
each α. Bootstrap-resample the 100 docs with replacement B=2000 times; for each resample
compute the token-weighted aggregate recovery

    R = 1 - Σ_d (nll_learned_d - nll_clean_d) / Σ_d (nll_raw_d - nll_clean_d)

and report R's mean and 95% bootstrap CI at each α. Reuses exp03's Corrector /
train_corrector / make_hat verbatim (DRY); only the eval loop is new (per-doc losses).
Outputs: results/31_eval_bootstrap.json + plots/31_eval_bootstrap.png.
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib.util
HERE = os.path.dirname(__file__)


def _load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


exp03 = _load("exp03", "03_learned_corrector.py")
from common import resid_post, fineweb_texts, load_model, DATA, DEVICE  # noqa: E402
from projections import normalize_vector  # noqa: E402

LAYER = exp03.LAYER
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")


@torch.no_grad()
def per_doc_nll(texts, layer, fn, seq_len=128, batch=16):
    """Return (summed_nll[n_docs], n_tokens[n_docs]) under patch fn at `layer`."""
    model, tok = load_model()
    sums, counts = [], []
    for ids, mask in exp03.batched_ids(tok, texts, seq_len, batch):
        with exp03.FuncPatcher(model, layer, fn):
            logits = model(ids, attention_mask=mask).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]
        m = mask[:, 1:].bool()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        nll = nll * m
        sums.extend(nll.sum(1).cpu().numpy().tolist())
        counts.extend(m.sum(1).cpu().numpy().tolist())
    return np.array(sums, dtype=np.float64), np.array(counts, dtype=np.float64)


def main():
    torch.manual_seed(exp03.SEED)
    np.random.seed(exp03.SEED)

    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    d = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))
    v = d["v"].astype(np.float32)
    vnorm = float(np.linalg.norm(v))
    vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE)
    vhat_t = torch.tensor(vhat, device=DEVICE)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]     # exact flagship split
    eval_texts = all_texts[400:500]      # 100 held-out docs

    corr = exp03.Corrector().to(DEVICE)
    exp03.train_corrector(corr, vt, vhat_t, train_texts)

    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]

    # clean per-doc NLL (baseline)
    s_clean, n_tok = per_doc_nll(eval_texts, LAYER, lambda h: h)
    print(f"clean per-doc mean loss = {(s_clean.sum()/n_tok.sum()):.3f}", flush=True)

    rng = np.random.RandomState(exp03.SEED)
    B = 2000
    n = len(eval_texts)
    boot_idx = rng.randint(0, n, size=(B, n))

    rows = {}
    for a in alphas:
        def f_raw(h):
            return h + a * vt

        def f_learn(h):
            hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t)
            return hat

        s_raw, _ = per_doc_nll(eval_texts, LAYER, f_raw)
        s_learn, _ = per_doc_nll(eval_texts, LAYER, f_learn)

        # per-doc excess NLL over clean
        e_raw = s_raw - s_clean       # summed extra nats, raw
        e_learn = s_learn - s_clean   # summed extra nats, learned

        # point estimate: token-weighted aggregate recovery over the full 100 docs
        R_point = 1.0 - e_learn.sum() / e_raw.sum()

        # also the aggregate token-averaged ΔLM (matches Exp 3 headline)
        dlm_raw = e_raw.sum() / n_tok.sum()
        dlm_learn = e_learn.sum() / n_tok.sum()

        # bootstrap over documents
        R_boot = 1.0 - (e_learn[boot_idx].sum(1)) / (e_raw[boot_idx].sum(1))
        lo, hi = np.percentile(R_boot, [2.5, 97.5])
        sd = float(R_boot.std(ddof=1))

        rows[str(a)] = dict(
            recovery_point=float(R_point),
            recovery_boot_mean=float(R_boot.mean()),
            recovery_boot_sd=sd,
            recovery_ci95=[float(lo), float(hi)],
            dlm_raw=float(dlm_raw),
            dlm_learned=float(dlm_learn),
        )
        print(f"alpha={a:>4}  recovery={R_point*100:5.1f}%  "
              f"boot 95% CI [{lo*100:5.1f}, {hi*100:5.1f}]  sd={sd*100:.1f}pp  "
              f"ΔLM raw={dlm_raw:+.3f} learned={dlm_learn:+.3f}", flush=True)

    res = dict(model="gpt2", layer=LAYER, seed=exp03.SEED, v_norm=vnorm,
               n_eval_docs=n, n_bootstrap=B, alphas=alphas, rows=rows)
    with open(os.path.join(RESULTS, "31_eval_bootstrap.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure ----
    xs = alphas
    rec = [rows[str(a)]["recovery_point"] * 100 for a in xs]
    lo = [rows[str(a)]["recovery_ci95"][0] * 100 for a in xs]
    hi = [rows[str(a)]["recovery_ci95"][1] * 100 for a in xs]
    yerr = np.array([[r - l for r, l in zip(rec, lo)],
                     [h - r for h, r in zip(hi, rec)]])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.errorbar(xs, rec, yerr=yerr, fmt="o-", color="#2980b9", capsize=4,
                label="learned corrector (95% doc-bootstrap CI)")
    ax.axhline(100, ls="--", color="#27ae60", lw=1, label="full recovery (100%)")
    ax.axhline(0, ls="--", color="#c0392b", lw=1, label="no recovery (= raw)")
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.set_ylabel("ΔLM recovery vs raw (%)")
    ax.set_title("Exp 31 — flagship recovery, bootstrapped over held-out docs\n"
                 "(GPT-2 small, block 6, seed 0, B=2000)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "31_eval_bootstrap.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
