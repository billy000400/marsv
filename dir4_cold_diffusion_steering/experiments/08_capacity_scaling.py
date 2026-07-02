"""Iteration 8 — DOES MORE MODEL CAPACITY CLOSE THE HELD-OUT GAP? (S4c follow-up #2).

Exp 7 found that enlarging the vector BANK (3→5 directions) at FIXED model capacity does
NOT close the held-out transfer gap — it makes transfer WORSE — and attributed this to
CAPACITY INTERFERENCE between directions competing for the same 5.25M MLP. That is a causal
claim about *model capacity*, which Exp 7 never varied. This experiment tests it directly:

    Hold the BANK fixed at the size-5 set {sentiment, formality, concreteness, politeness,
    complexity} — the exact bank that gave Exp 7's WORST held-out transfer (3% @α=8) — and
    SCALE the corrector's WIDTH:  hidden ∈ {1024, 2048, 4096}  (5.25M, 14.7M, 46M params).
    Does held-out `certainty` transfer recover toward the size-3 peak / native oracle as
    capacity grows? And does per-direction IN-BANK recovery rise with capacity?

If capacity interference is the binding constraint (Exp 7's claim), a wider model on the
SAME 5 directions should correct each one better and transfer better to the held-out one.
Same recipe / seed / data / eval as Exp 6/7. The hidden=1024 point re-runs Exp 7's size-5
model and should reproduce it (built-in reproducibility check). Native oracle (retrained on
certainty, hidden=1024) is the per-direction ceiling.

Model: GPT-2 small, resid_post block 6. Same held-out 100 FineWeb docs as Exp 3/4/5/6/7.
Outputs: results/08_capacity_scaling.json + plots/08_capacity_scaling.png.
"""
import os, json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import resid_post, fineweb_texts, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")

# reuse Exp 6 machinery verbatim (CondCorrector / make_hat_cond / train_cond)
_spec6 = importlib.util.spec_from_file_location("exp06", os.path.join(HERE, "06_conditional_bank.py"))
exp06 = importlib.util.module_from_spec(_spec6); _spec6.loader.exec_module(exp06)
CondCorrector = exp06.CondCorrector
make_hat_cond = exp06.make_hat_cond
train_cond = exp06.train_cond
# Exp 3 helpers (LM loss / layer / clean loss)
exp03 = exp06.exp03
lm_loss_fn = exp03.lm_loss_fn
LAYER = exp03.LAYER

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

WIDTHS = [1024, 2048, 4096]     # capacity knob (5.25M / 14.7M / 46M params)


def main():
    print("device:", DEVICE, flush=True)

    # ---- reuse all 5 pool directions + held-out certainty (persisted by Exp 6/7) ----
    vecs = {n: np.load(os.path.join(DATA, f"{n}_vec_layer6.npy")).astype(np.float32)
            for n in ["formality", "concreteness", "certainty",
                      "politeness", "complexity"]}
    vecs["sentiment"] = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))["v"].astype(np.float32)
    pool = ["sentiment", "formality", "concreteness", "politeness", "complexity"]  # size-5 bank
    heldout = "certainty"

    def T(v):
        vhat = normalize_vector(v)
        return (torch.tensor(v, device=DEVICE), torch.tensor(vhat, device=DEVICE), vhat)
    tens = {n: T(vecs[n]) for n in vecs}
    norms = {n: float(np.linalg.norm(vecs[n])) for n in vecs}
    cos_to_heldout = {n: float(vecs[n] @ vecs[heldout] /
                               (np.linalg.norm(vecs[n]) * np.linalg.norm(vecs[heldout])))
                      for n in pool}
    print("cos(dir, certainty):", {k: round(v, 3) for k, v in cos_to_heldout.items()}, flush=True)

    # ---- eval split (identical protocol to Exp 3/4/5/6/7) ----
    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]
    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]
    clean_loss = lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    vt_h, vhat_th, _ = tens[heldout]

    def eval_dlm_on(corr, vt, vhat_t):
        dlm = []
        for a in alphas:
            def f(h):
                hat, _ = make_hat_cond(corr, h, a, vt, vhat_t); return hat
            dlm.append(lm_loss_fn(eval_texts, LAYER, f) - clean_loss)
        return dlm

    # held-out RAW baseline (same for every capacity)
    raw_dlm_h = []
    for a in alphas:
        def f_raw(h): return h + a * vt_h
        raw_dlm_h.append(lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss)
    print("held-out raw ΔLM:", [round(x, 3) for x in raw_dlm_h], flush=True)

    # in-bank RAW baselines @α=8 (same for every capacity)
    raw8 = {}
    for n in pool:
        vt, _, _ = tens[n]
        def fr(h): return h + 8.0 * vt
        raw8[n] = lm_loss_fn(eval_texts, LAYER, fr) - clean_loss

    bank = [(tens[n][0], tens[n][1]) for n in pool]
    out = {"alphas": alphas, "raw_dlm_heldout": raw_dlm_h, "cos_to_heldout": cos_to_heldout,
           "norms": norms, "pool": pool, "heldout": heldout, "widths": WIDTHS,
           "raw8_inbank": {n: raw8[n] for n in pool}, "capacity": {}}

    for hid in WIDTHS:
        nparam = None
        print(f"\n=== training size-5 bank, WIDTH hidden={hid} ===", flush=True)
        torch.manual_seed(SEED); np.random.seed(SEED)
        corr = CondCorrector(hidden=hid).to(DEVICE)
        nparam = sum(p.numel() for p in corr.parameters())
        print(f"  cond corrector params: {nparam/1e6:.2f}M", flush=True)
        train_cond(corr, bank, train_texts)
        # held-out transfer
        dlm_h = eval_dlm_on(corr, vt_h, vhat_th)
        rec_h = [100.0 * (raw_dlm_h[i] - dlm_h[i]) / raw_dlm_h[i] for i in range(len(alphas))]
        # in-bank per-direction recovery @α=8
        inbank8 = {}
        for n in pool:
            vt, vhat_t, _ = tens[n]
            def fb(h):
                hat, _ = make_hat_cond(corr, h, 8.0, vt, vhat_t); return hat
            b8 = lm_loss_fn(eval_texts, LAYER, fb) - clean_loss
            inbank8[n] = {"bank": b8, "rec": 100.0 * (raw8[n] - b8) / raw8[n]}
        out["capacity"][str(hid)] = {
            "params_M": nparam / 1e6,
            "heldout_dlm": dlm_h, "heldout_rec": rec_h, "inbank_alpha8": inbank8}
        ib = ", ".join("%s:%.0f" % (n, inbank8[n]["rec"]) for n in pool)
        print(f"  hid={hid}: held-out rec {[round(x,0) for x in rec_h]}  in-bank@8 rec {{{ib}}}",
              flush=True)
        del corr; torch.cuda.empty_cache()

    # ---- native oracle on held-out (ceiling), hidden=1024 (matches Exp 6/7) ----
    print(f"\n=== training NATIVE oracle on held-out '{heldout}' (hidden=1024) ===", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    corr_native = CondCorrector(hidden=1024).to(DEVICE)
    train_cond(corr_native, [(tens[heldout][0], tens[heldout][1])], train_texts)
    native_dlm = eval_dlm_on(corr_native, vt_h, vhat_th)
    native_rec = [100.0 * (raw_dlm_h[i] - native_dlm[i]) / raw_dlm_h[i] for i in range(len(alphas))]
    out["native_dlm"] = native_dlm
    out["native_rec"] = native_rec
    print(f"  native held-out rec {[round(x,0) for x in native_rec]}", flush=True)
    del corr_native; torch.cuda.empty_cache()

    with open(os.path.join(RESULTS, "08_capacity_scaling.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # panel (a): held-out recovery vs α, one line per capacity + native oracle
    cmap = {1024: "#f1c40f", 2048: "#e67e22", 4096: "#8e44ad"}
    for hid in WIDTHS:
        pm = out["capacity"][str(hid)]["params_M"]
        ax[0].plot(alphas, out["capacity"][str(hid)]["heldout_rec"], "o-", color=cmap[hid],
                   label=f"{pm:.0f}M params (hid {hid})")
    ax[0].plot(alphas, native_rec, "s--", color="#27ae60", label="native oracle (ceiling)")
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$")
    ax[0].set_ylabel("held-out fluency recovery (%)")
    ax[0].set_title(f"(a) transfer to held-out '{heldout}' vs model capacity")
    ax[0].legend(fontsize=8, loc="upper right")
    # panel (b): recovery vs capacity — held-out (α=8) and mean in-bank (α=8)
    pmids = [out["capacity"][str(hid)]["params_M"] for hid in WIDTHS]
    rec_h8 = [out["capacity"][str(hid)]["heldout_rec"][-1] for hid in WIDTHS]
    rec_inbank8 = [np.mean([out["capacity"][str(hid)]["inbank_alpha8"][n]["rec"] for n in pool])
                   for hid in WIDTHS]
    ax[1].plot(pmids, rec_inbank8, "o-", color="#2980b9", label=r"mean in-bank ($\alpha=8$)")
    ax[1].plot(pmids, rec_h8, "o-", color="#c0392b", label=r"held-out certainty ($\alpha=8$)")
    ax[1].axhline(native_rec[-1], ls="--", color="#27ae60", label=r"native oracle ($\alpha=8$)")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("corrector capacity (M params, log)")
    ax[1].set_ylabel(r"fluency recovery at $\alpha=8$ (%)")
    ax[1].set_xticks(pmids); ax[1].set_xticklabels([f"{p:.0f}" for p in pmids])
    ax[1].set_title("(b) does more capacity close the gap?")
    ax[1].legend(fontsize=8, loc="best")
    for x, y in zip(pmids, rec_h8):
        ax[1].text(x, y + 2, f"{y:.0f}%", ha="center", fontsize=9, color="#c0392b")
    for x, y in zip(pmids, rec_inbank8):
        ax[1].text(x, y + 2, f"{y:.0f}%", ha="center", fontsize=9, color="#2980b9")
    fig.suptitle("Scaling model capacity on a FIXED size-5 bank: held-out + in-bank recovery "
                 "(GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "08_capacity_scaling.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
