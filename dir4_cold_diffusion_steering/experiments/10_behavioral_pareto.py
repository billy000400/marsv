"""Iteration 10 — BEHAVIORAL test: does the corrected steer still steer the generated text?

Experiments 1-9 all score the corrector on ΔLM (fluency damage) at *matched projection*
along v. Matched projection is a proxy: it guarantees the injected edit along v is identical
at layer 6, but it does NOT prove the corrected activation, when used to actually GENERATE
text autoregressively, (a) still moves the produced text in the steered direction, or (b)
produces less-degenerate text than raw steering. Strong linear steering is known to collapse
generation into repetition/gibberish; the whole promise of ColdSteer is "same edit, better
text." This experiment tests that promise on real generation, not teacher-forced loss.

Setup (self-contained; reuses the Exp 3 flagship sentiment corrector):
  - Retrain the sentiment corrector r_θ with the EXACT Exp 3 recipe (imported).
  - Prompts: first PROMPT_LEN tokens of held-out FineWeb docs (disjoint from train/fit).
  - For each α and each regime {raw h+αv, corrected ĥ = z + P_{v⊥}r_θ}, greedily generate
    GEN_LEN continuation tokens with the steer applied at resid_post block 6 at EVERY position.
  - Two behavioral metrics, both measured on a CLEAN re-encode of the generated text:
      * sentiment EFFECT  B(α) = mean projection of continuation resid_post@6 onto v̂,
        reported as the shift B(α) − B(0) vs the unsteered greedy continuation.
      * DEGENERATION      distinct-2 = unique-bigram ratio of the generated continuation
        (lower = more repetitive/collapsed text).  Higher distinct-2 = more fluent/diverse.
  - The Pareto question: at a matched behavioral effect, does the corrector keep distinct-2
    higher (less degeneration) than raw steering?

Outputs: results/10_behavioral_pareto.json + plots/10_behavioral_pareto.png.
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

# import the Exp 3 flagship corrector + recipe (module name starts with a digit)
_spec = importlib.util.spec_from_file_location("exp03", os.path.join(HERE, "03_learned_corrector.py"))
exp03 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(exp03)
LAYER = exp03.LAYER

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

PROMPT_LEN = 12
GEN_LEN = 30


def prompt_ids(tok, texts, n_prompts):
    """First PROMPT_LEN tokens of each doc long enough; stack [n_prompts, PROMPT_LEN]."""
    kept = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=64).input_ids[0]
        if ids.numel() >= PROMPT_LEN:
            kept.append(ids[:PROMPT_LEN])
        if len(kept) >= n_prompts:
            break
    return torch.stack(kept).to(DEVICE)          # [N, PROMPT_LEN]


@torch.no_grad()
def generate(model, ids, fn):
    """Greedy generate GEN_LEN new tokens with resid_post@LAYER replaced by fn(h)."""
    with exp03.FuncPatcher(model, LAYER, fn):
        out = model.generate(ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=GEN_LEN, do_sample=False,
                             pad_token_id=model.config.eos_token_id)
    return out                                    # [N, PROMPT_LEN + GEN_LEN]


@torch.no_grad()
def sentiment_effect(model, full_ids, vhat_t):
    """Mean projection of CLEAN-re-encoded continuation activations onto v̂."""
    hs = model(full_ids, output_hidden_states=True).hidden_states
    h = hs[LAYER + 1][:, PROMPT_LEN:, :].float()  # continuation positions only
    return float((h @ vhat_t).mean().item())


def distinct2(gen_tokens):
    """Mean unique-bigram ratio over a batch of generated continuations (numpy [N, GEN_LEN])."""
    ratios = []
    for row in gen_tokens:
        bigrams = list(zip(row[:-1], row[1:]))
        if not bigrams:
            continue
        ratios.append(len(set(bigrams)) / len(bigrams))
    return float(np.mean(ratios))


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    d = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))
    v = d["v"].astype(np.float32)
    vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE)
    vhat_t = torch.tensor(vhat, device=DEVICE)
    print(f"|v| = {np.linalg.norm(v):.2f}", flush=True)

    # ---- train the sentiment corrector (identical Exp 3 recipe / data split) ----
    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    corr = exp03.Corrector().to(DEVICE)
    print(f"corrector params: {sum(p.numel() for p in corr.parameters())/1e6:.2f}M", flush=True)
    exp03.train_corrector(corr, vt, vhat_t, train_texts)

    # ---- prompts (held-out region, disjoint from train 500:800 and fit) ----
    ids = prompt_ids(tok, all_texts[400:500], n_prompts=48)
    print(f"{ids.shape[0]} prompts x {PROMPT_LEN} tokens, generating {GEN_LEN}", flush=True)

    # ---- baseline: unsteered greedy continuation ----
    clean_out = generate(model, ids, lambda h: h)
    B0 = sentiment_effect(model, clean_out, vhat_t)
    d2_0 = distinct2(clean_out[:, PROMPT_LEN:].cpu().numpy())
    print(f"clean: B0={B0:+.3f} distinct2={d2_0:.3f}", flush=True)

    alphas = [2.0, 4.0, 6.0, 8.0]
    res = {"raw": {"effect": [], "distinct2": []},
           "corrected": {"effect": [], "distinct2": []}}
    samples = {}

    for a in alphas:
        def f_raw(h):  return h + a * vt
        def f_corr(h):
            hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat

        for name, fn in [("raw", f_raw), ("corrected", f_corr)]:
            out = generate(model, ids, fn)
            B = sentiment_effect(model, out, vhat_t)
            d2 = distinct2(out[:, PROMPT_LEN:].cpu().numpy())
            res[name]["effect"].append(B - B0)     # sentiment shift vs unsteered
            res[name]["distinct2"].append(d2)
            if a in (4.0, 8.0):
                samples[f"{name}_a{int(a)}"] = tok.decode(out[0], skip_special_tokens=True)
        print(f"alpha={a:>4}  raw: eff={res['raw']['effect'][-1]:+.2f} d2={res['raw']['distinct2'][-1]:.3f}"
              f"  | corr: eff={res['corrected']['effect'][-1]:+.2f} d2={res['corrected']['distinct2'][-1]:.3f}",
              flush=True)

    out_json = dict(layer=LAYER, v_norm=float(np.linalg.norm(v)), prompt_len=PROMPT_LEN,
                    gen_len=GEN_LEN, n_prompts=int(ids.shape[0]), alphas=alphas,
                    B0=B0, distinct2_0=d2_0, results=res, samples=samples,
                    n_train_texts=len(train_texts))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "10_behavioral_pareto.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    # ---- figure ----
    C = {"raw": "#c0392b", "corrected": "#2980b9"}
    L = {"raw": "raw steer  h+αv", "corrected": "learned corrector  ĥ"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) behavioral effect vs alpha
    for m in ("raw", "corrected"):
        ax[0].plot(alphas, res[m]["effect"], "o-", color=C[m], label=L[m])
    ax[0].axhline(0, ls="--", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$")
    ax[0].set_ylabel(r"sentiment shift  $B(\alpha)-B(0)$")
    ax[0].set_title("(a) behavioral effect (higher = stronger steer)"); ax[0].legend(fontsize=8)

    # (b) degeneration vs alpha
    for m in ("raw", "corrected"):
        ax[1].plot(alphas, res[m]["distinct2"], "o-", color=C[m], label=L[m])
    ax[1].axhline(d2_0, ls="--", color="k", lw=1, label=f"unsteered ({d2_0:.2f})")
    ax[1].set_xlabel(r"steering strength $\alpha$")
    ax[1].set_ylabel("distinct-2 (unique-bigram ratio)")
    ax[1].set_title("(b) text degeneration (higher = more fluent)"); ax[1].legend(fontsize=8)

    # (c) Pareto: effect (x) vs fluency (y)
    for m in ("raw", "corrected"):
        ax[2].plot(res[m]["effect"], res[m]["distinct2"], "o-", color=C[m], label=L[m])
        for a, x, y in zip(alphas, res[m]["effect"], res[m]["distinct2"]):
            ax[2].annotate(f"{int(a)}", (x, y), fontsize=7, color=C[m])
    ax[2].set_xlabel(r"sentiment shift $B(\alpha)-B(0)$  (want $\rightarrow$)")
    ax[2].set_ylabel(r"distinct-2  (want $\uparrow$)")
    ax[2].set_title("(c) effect-vs-fluency Pareto (up-right dominates)"); ax[2].legend(fontsize=8)

    fig.suptitle("Behavioral test on generated text: corrected vs raw steering (GPT-2 small, layer 6, sentiment)")
    fig.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, "10_behavioral_pareto.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
