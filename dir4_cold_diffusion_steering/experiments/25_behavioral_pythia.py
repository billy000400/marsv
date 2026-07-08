"""Iteration 25 — BEHAVIORAL reality-check on Pythia-410m / GPT-NeoX: is Exp 24's 81%
fluency recovery bought by under-steering? (Exp 24's own Next check.)

Exp 24 replicated the flagship fluency result on a THIRD architecture family (Pythia-410m,
GPT-NeoX, parallel residual): raw steering breaks the LM (ΔLM@8 +3.10) and the identical
LM-supervised corrector recovers 81% of that damage at MATCHED PROJECTION. But Exp 10 (GPT-2)
and Exp 22 (Qwen3) both showed that matched layer projection is a proxy — the corrector's
correction is ⟂ v in activation space but NOT ⟂ the downstream sentiment readout, so its
fluency win comes partly at the cost of a weaker *behavioral* steer in generation. That caveat
has been confirmed on GPT-2 and Qwen3; this closes the third architecture.

This script runs the EXACT Exp 10/22 protocol on Pythia-410m, reusing the EXACT Exp 24
flagship corrector (checkpoint results/24_corr.pt trained on the Pythia block-12 sentiment
vector). Only the model/corrector differ from Exp 22; the vector build, prompt split, α grid,
and metrics are identical, so GPT-2 (Exp 10) / Qwen3 (Exp 22) / Pythia (this) are directly
comparable.

  - Prompts: first PROMPT_LEN tokens of held-out FineWeb docs [400:500] (disjoint from Exp 24
    train [500:800] and fit [0:400]).
  - For each α and each regime {raw h+αv, corrected ĥ = z + P_{v⊥}r_θ}, greedily generate
    GEN_LEN continuation tokens with the steer applied at resid_post block 12 at EVERY position.
  - Two behavioral metrics on a CLEAN re-encode of the generated text:
      * sentiment EFFECT  B(α)−B(0) = mean projection of continuation resid_post@12 onto v̂,
        relative to the unsteered greedy continuation (higher = more strongly steered).
      * DEGENERATION      distinct-2 = unique-bigram ratio of the generated tokens
        (lower = more repetitive/collapsed; higher = more fluent).

Outputs: results/25_behavioral_pythia.json + plots/25_behavioral_pythia.png.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")

# reuse the Exp 24 Pythia pipeline (load / gpt_neox hooks / make_hat / resid_post / fineweb_texts)
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
exp24 = _load("exp24", "24_cross_arch_pythia.py")
exp03 = _load("exp03", "03_learned_corrector.py")
exp01 = _load("exp01", "01_offmanifold_phenomenon.py")
from projections import normalize_vector

DEVICE = exp24.DEVICE
LAYER = exp24.LAYER          # 12
D_MODEL = exp24.D_MODEL      # 1024
SEED = 0
PROMPT_LEN = 12
GEN_LEN = 30
N_PROMPTS = 48
GEN_BATCH = 8
ALPHAS = [2.0, 4.0, 6.0, 8.0]


def prompt_ids(tok, texts, n_prompts):
    kept = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=64).input_ids[0]
        if ids.numel() >= PROMPT_LEN:
            kept.append(ids[:PROMPT_LEN])
        if len(kept) >= n_prompts:
            break
    return torch.stack(kept).to(DEVICE)      # [N, PROMPT_LEN]


@torch.no_grad()
def generate(model, tok, ids, steer):
    """Greedy generate GEN_LEN tokens with resid_post@LAYER replaced by steer(h). Batched."""
    outs = []
    for i in range(0, len(ids), GEN_BATCH):
        chunk = ids[i:i + GEN_BATCH]
        with exp24.FuncPatcher(model, LAYER, lambda h: steer(h).to(h.dtype)):
            o = model.generate(chunk, attention_mask=torch.ones_like(chunk),
                               max_new_tokens=GEN_LEN, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        outs.append(o)
        torch.cuda.empty_cache()
    return torch.cat(outs, 0)                 # [N, PROMPT_LEN + GEN_LEN]


@torch.no_grad()
def sentiment_effect(model, full_ids, vhat_t):
    """Mean projection of CLEAN-re-encoded continuation activations onto v̂. Batched."""
    vals = []
    for i in range(0, len(full_ids), GEN_BATCH):
        chunk = full_ids[i:i + GEN_BATCH]
        hs = model(chunk, output_hidden_states=True).hidden_states
        h = hs[LAYER + 1][:, PROMPT_LEN:, :].float()   # continuation positions only
        vals.append((h @ vhat_t).reshape(-1).cpu())
        torch.cuda.empty_cache()
    return float(torch.cat(vals).mean().item())


def distinct2(gen_tokens):
    ratios = []
    for row in gen_tokens:
        bigrams = list(zip(row[:-1], row[1:]))
        if not bigrams:
            continue
        ratios.append(len(set(bigrams)) / len(bigrams))
    return float(np.mean(ratios))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    model, tok = exp24.load()
    print(f"model={exp24.MODEL_NAME} layer={LAYER} d={D_MODEL} device={DEVICE}", flush=True)

    # rebuild the EXACT Exp 24 block-12 sentiment vector (deterministic; matches the checkpoint)
    hp = exp24.resid_post(exp01.POS, LAYER, seq_len=32, batch=8)
    hn = exp24.resid_post(exp01.NEG, LAYER, seq_len=32, batch=8)
    v = (hp.mean(0) - hn.mean(0)).astype(np.float32)
    vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)
    print(f"|v| = {np.linalg.norm(v):.2f}", flush=True)

    # load the EXACT Exp 24 flagship corrector (no retraining)
    corr = exp03.Corrector(d=D_MODEL).to(DEVICE)
    ckpt = os.path.join(RESULTS, "24_corr.pt")
    corr.load_state_dict(torch.load(ckpt, map_location=DEVICE)); corr.eval()
    print(f"loaded Exp 24 corrector ({sum(p.numel() for p in corr.parameters())/1e6:.2f}M params)", flush=True)

    all_texts = exp24.fineweb_texts(800)
    ids = prompt_ids(tok, all_texts[400:500], N_PROMPTS)
    print(f"{ids.shape[0]} prompts x {PROMPT_LEN} tokens, generating {GEN_LEN}", flush=True)

    # unsteered baseline
    clean_out = generate(model, tok, ids, lambda h: h)
    B0 = sentiment_effect(model, clean_out, vhat_t)
    d2_0 = distinct2(clean_out[:, PROMPT_LEN:].cpu().numpy())
    print(f"clean: B0={B0:+.3f} distinct2={d2_0:.3f}", flush=True)

    res = {"raw": {"effect": [], "distinct2": []},
           "corrected": {"effect": [], "distinct2": []}}
    samples = {}
    for a in ALPHAS:
        def f_raw(h):  return h.float() + a * vt
        def f_corr(h):
            hat, _ = exp24.make_hat(corr, h, a, vt, vhat_t); return hat
        for name, fn in [("raw", f_raw), ("corrected", f_corr)]:
            out = generate(model, tok, ids, fn)
            B = sentiment_effect(model, out, vhat_t)
            d2 = distinct2(out[:, PROMPT_LEN:].cpu().numpy())
            res[name]["effect"].append(B - B0)
            res[name]["distinct2"].append(d2)
            if a in (4.0, 8.0):
                samples[f"{name}_a{int(a)}"] = tok.decode(out[0], skip_special_tokens=True)
        print(f"alpha={a:>4}  raw: eff={res['raw']['effect'][-1]:+.2f} d2={res['raw']['distinct2'][-1]:.3f}"
              f"  | corr: eff={res['corrected']['effect'][-1]:+.2f} d2={res['corrected']['distinct2'][-1]:.3f}",
              flush=True)

    out_json = dict(model=exp24.MODEL_NAME, arch="gpt_neox (parallel residual)", layer=LAYER,
                    d_model=D_MODEL, v_norm=float(np.linalg.norm(v)), prompt_len=PROMPT_LEN,
                    gen_len=GEN_LEN, n_prompts=int(ids.shape[0]), alphas=ALPHAS, B0=B0,
                    distinct2_0=d2_0, seed=SEED, results=res, samples=samples)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "25_behavioral_pythia.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    # ---- figure (same 3 panels as Exp 10/22) ----
    C = {"raw": "#c0392b", "corrected": "#2980b9"}
    L = {"raw": "raw steer  h+αv", "corrected": "learned corrector  ĥ"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for m in ("raw", "corrected"):
        ax[0].plot(ALPHAS, res[m]["effect"], "o-", color=C[m], label=L[m])
    ax[0].axhline(0, ls="--", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$")
    ax[0].set_ylabel(r"sentiment shift  $B(\alpha)-B(0)$")
    ax[0].set_title("(a) behavioral effect (higher = stronger steer)"); ax[0].legend(fontsize=8)
    for m in ("raw", "corrected"):
        ax[1].plot(ALPHAS, res[m]["distinct2"], "o-", color=C[m], label=L[m])
    ax[1].axhline(d2_0, ls="--", color="k", lw=1, label=f"unsteered ({d2_0:.2f})")
    ax[1].set_xlabel(r"steering strength $\alpha$")
    ax[1].set_ylabel("distinct-2 (unique-bigram ratio)")
    ax[1].set_title("(b) text degeneration (higher = more fluent)"); ax[1].legend(fontsize=8)
    for m in ("raw", "corrected"):
        ax[2].plot(res[m]["effect"], res[m]["distinct2"], "o-", color=C[m], label=L[m])
        for a, x, y in zip(ALPHAS, res[m]["effect"], res[m]["distinct2"]):
            ax[2].annotate(f"{int(a)}", (x, y), fontsize=7, color=C[m])
    ax[2].set_xlabel(r"sentiment shift $B(\alpha)-B(0)$  (want $\rightarrow$)")
    ax[2].set_ylabel(r"distinct-2  (want $\uparrow$)")
    ax[2].set_title("(c) effect-vs-fluency Pareto (up-right dominates)"); ax[2].legend(fontsize=8)
    fig.suptitle("Behavioral test on generated text: corrected vs raw steering "
                 "(Pythia-410m / GPT-NeoX, block 12, sentiment) — is Exp 24's 81% recovery bought by under-steering?")
    fig.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, "25_behavioral_pythia.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
