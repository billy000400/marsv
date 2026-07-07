"""Iteration 23 — does the GPT-2 behavioral-preservation FIX (Exp 11) transfer across the
architecture boundary to Qwen3-1.7B?

Chain so far on the behavioral axis:
  * Exp 10 (GPT-2): the flagship corrector's matched-projection fluency win is partly bought
    by UNDER-STEERING in generation (correction ⟂ v in activation space but NOT ⟂ the
    downstream sentiment readout).
  * Exp 11 (GPT-2): adding one term that pushes the corrector's DOWNSTREAM sentiment readout
    (at a later layer L2) toward RAW steering's readout recovers 2-6× more generated effect
    while staying fluent — pushes the effect-fluency Pareto out (ceiling ≈+1.3).
  * Exp 22 (Qwen3): the Exp 10 under-steering caveat REPLICATES on Qwen3, and named the Exp 11
    term as "the indicated fix if strong behavioral steering is required on Qwen3" — but never
    tested it. This experiment tests exactly that.

We reuse the EXACT Exp 21 Qwen3 pipeline (load / hooks / make_hat / resid_post / Corrector) and
the EXACT Exp 22 generation protocol (48 prompts × 30 greedy tokens; effect B(α)−B(0) + distinct-2).
We add the Exp 11 behavioral term at a downstream Qwen3 layer L2=27 (last decoder block, the
analogue of GPT-2's block 11 that feeds the final norm + head) and train a family λ_b ∈ {0,10,40}.

Efficiency + faithfulness: λ_b=0 is EXACTLY the Exp 21/22 corrector, so we LOAD the existing
checkpoint results/21_corr.pt for it (a clean reproducibility anchor that must match Exp 22 to the
digit) and only train λ_b∈{10,40} fresh with the same 6-epoch recipe/seed/data as Exp 21.

Behavioral term (teacher-forced, per batch, α ~ U(0.5,8)):
  ŵ = unit DiffMean sentiment direction at Qwen3 block L2. Under raw z=h+αv the upper LM carries a
  downstream projection p_raw = ⟨resid@L2, ŵ⟩; the corrected ĥ carries p_corr. Add
      L_behav = mean_tokens( ((p_corr − p_raw)/100)^2 )
  Total loss = LM_CE + λ_near·‖P_{v⊥}r‖²/100² + λ_b·L_behav.

Outputs: results/23_behavioral_qwen_fix.json + plots/23_behavioral_qwen_fix.png.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


exp21 = _load("exp21", "21_cross_arch.py")   # Qwen3 load / FuncPatcher / make_hat / resid_post / batched_ids
exp22 = _load("exp22", "22_behavioral_qwen.py")  # prompt_ids / generate / sentiment_effect / distinct2
exp03 = _load("exp03", "03_learned_corrector.py")  # Corrector
exp01 = _load("exp01", "01_offmanifold_phenomenon.py")  # POS / NEG sentiment prompts
from projections import normalize_vector

DEVICE = exp21.DEVICE
LAYER = exp21.LAYER          # 14 — steer / correct here
L2 = 27                      # downstream readout: last Qwen3 decoder block (hidden_states[28])
D_MODEL = exp21.D_MODEL      # 2048
SEED = 0
LAM_B = [0.0, 10.0, 40.0]    # behavioral weights (0 = Exp 21/22 corrector, loaded from checkpoint)
ALPHAS = exp22.ALPHAS        # [2,4,6,8]


class Capture:
    """Forward hook that stores resid_post of Qwen3 decoder block `layer` from the current pass."""
    def __init__(self, model, layer):
        self.block = model.model.layers[layer]; self.store = {}; self.handle = None
    def _hook(self, m, i, o):
        self.store["h"] = o[0] if isinstance(o, tuple) else o
    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook); return self
    def __exit__(self, *a):
        self.handle.remove()


def train_behav(corr, vt, vhat_t, w2hat_t, train_texts, lam_b,
                epochs=6, lr=1e-3, lam_near=0.05, amin=0.5, amax=8.0):
    """Exp 21 recipe (LM_CE + λ_near) + Exp 11 downstream-readout preservation term (weight lam_b)."""
    model, tok = exp21.load()
    opt = torch.optim.Adam(corr.parameters(), lr=lr)
    rng = np.random.RandomState(SEED)
    corr.train(); step = 0
    for ep in range(epochs):
        order = rng.permutation(len(train_texts))
        texts = [train_texts[i] for i in order]
        for ids, mask in exp21.batched_ids(tok, texts, exp21.TRAIN_SEQ, exp21.TRAIN_BATCH):
            alpha = float(rng.uniform(amin, amax))
            m = mask.bool()

            # --- raw downstream target (no grad) ---
            with torch.no_grad(), exp21.FuncPatcher(model, LAYER, lambda h: (h.float() + alpha * vt).to(h.dtype)), \
                 Capture(model, L2) as capR:
                model(ids, attention_mask=mask)
                p_raw = (capR.store["h"].float() @ w2hat_t)      # [B, T]

            # --- corrected pass (grad); L2 capture rides the SAME forward that yields logits ---
            store = {}
            def fn(h):
                hat, r_perp = exp21.make_hat(corr, h.detach(), alpha, vt, vhat_t)
                store["rp"] = r_perp
                return hat.to(h.dtype)
            with exp21.FuncPatcher(model, LAYER, fn), Capture(model, L2) as capC:
                logits = model(ids, attention_mask=mask).logits
                p_corr = (capC.store["h"].float() @ w2hat_t)     # [B, T]

            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tgt = ids[:, 1:]; mm = mask[:, 1:].bool()
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lm = nll[mm].mean()
            near = (store["rp"] / 100.0).pow(2).sum(-1).mean()
            behav = (((p_corr - p_raw) / 100.0) ** 2)[m].mean()
            loss = lm + lam_near * near + lam_b * behav
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  lam_b={lam_b:>4} ep{ep} step{step:4d} a={alpha:4.1f} "
                      f"LM={lm.item():.3f} behav={behav.item():.4f} "
                      f"p_raw={p_raw[m].mean().item():+.1f} p_corr={p_corr[m].mean().item():+.1f}",
                      flush=True)
    corr.eval(); return corr


@torch.no_grad()
def eval_generation(corr, ids, vt, vhat_t, B0, alphas):
    """Exp 22 generation protocol for one corrector: effect B(α)-B0 and distinct-2."""
    model, tok = exp21.load()
    eff, d2 = [], []
    for a in alphas:
        def f_corr(h):
            hat, _ = exp21.make_hat(corr, h, a, vt, vhat_t); return hat
        out = exp22.generate(model, tok, ids, f_corr)
        eff.append(exp22.sentiment_effect(model, out, vhat_t) - B0)
        d2.append(exp22.distinct2(out[:, exp22.PROMPT_LEN:].cpu().numpy()))
    return eff, d2


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    model, tok = exp21.load()
    print(f"model={exp21.MODEL_NAME} steer L{LAYER} readout L{L2} d={D_MODEL} device={DEVICE}", flush=True)

    # rebuild the EXACT Exp 21 block-14 sentiment vector (deterministic; matches the checkpoint)
    hp = exp21.resid_post(exp01.POS, LAYER, seq_len=32, batch=8)
    hn = exp21.resid_post(exp01.NEG, LAYER, seq_len=32, batch=8)
    v = (hp.mean(0) - hn.mean(0)).astype(np.float32)
    vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)

    # downstream sentiment readout direction at layer L2 (DiffMean on the same POS/NEG prompts)
    hp2 = exp21.resid_post(exp01.POS, L2, seq_len=32, batch=8)
    hn2 = exp21.resid_post(exp01.NEG, L2, seq_len=32, batch=8)
    w2 = (hp2.mean(0) - hn2.mean(0)).astype(np.float32)
    w2hat_t = torch.tensor(normalize_vector(w2), device=DEVICE)
    print(f"|v|(L{LAYER})={np.linalg.norm(v):.2f}  |w|(L{L2})={np.linalg.norm(w2):.2f}", flush=True)

    # data splits identical to Exp 21/22
    all_texts = exp21.fineweb_texts(800)
    train_texts = all_texts[500:800]
    ids = exp22.prompt_ids(tok, all_texts[400:500], exp22.N_PROMPTS)
    print(f"{ids.shape[0]} prompts x {exp22.PROMPT_LEN} tokens, gen {exp22.GEN_LEN}", flush=True)

    # unsteered baseline (shared) — reused for every corrector and raw
    clean_out = exp22.generate(model, tok, ids, lambda h: h)
    B0 = exp22.sentiment_effect(model, clean_out, vhat_t)
    d2_0 = exp22.distinct2(clean_out[:, exp22.PROMPT_LEN:].cpu().numpy())
    print(f"clean: B0={B0:+.3f} distinct2={d2_0:.3f}", flush=True)

    # raw reference (no corrector) — identical for every λ_b
    raw_eff, raw_d2 = [], []
    for a in ALPHAS:
        out = exp22.generate(model, tok, ids, (lambda av: (lambda h: h.float() + av * vt))(a))
        raw_eff.append(exp22.sentiment_effect(model, out, vhat_t) - B0)
        raw_d2.append(exp22.distinct2(out[:, exp22.PROMPT_LEN:].cpu().numpy()))
    print(f"raw   effect={['%+.2f'%e for e in raw_eff]} d2={['%.2f'%x for x in raw_d2]}", flush=True)

    correctors = {}
    for lam_b in LAM_B:
        corr = exp03.Corrector(d=D_MODEL).to(DEVICE)
        ckpt = os.path.join(RESULTS, f"23_corr_lamb{lam_b:g}.pt")
        if lam_b == 0.0:
            corr.load_state_dict(torch.load(os.path.join(RESULTS, "21_corr.pt"), map_location=DEVICE))
            corr.eval()
            print("lam_b=0: loaded Exp 21 checkpoint (= Exp 22 corrector, reproducibility anchor)", flush=True)
        elif os.path.exists(ckpt):
            corr.load_state_dict(torch.load(ckpt, map_location=DEVICE)); corr.eval()
            print(f"lam_b={lam_b:g}: loaded checkpoint (skipped training)", flush=True)
        else:
            torch.manual_seed(SEED); np.random.seed(SEED)
            train_behav(corr, vt, vhat_t, w2hat_t, train_texts, lam_b)
            torch.save(corr.state_dict(), ckpt)
            print(f"lam_b={lam_b:g}: saved checkpoint", flush=True)
        torch.cuda.empty_cache()
        eff, d2 = eval_generation(corr, ids, vt, vhat_t, B0, ALPHAS)
        correctors[f"{lam_b:g}"] = {"lam_b": lam_b, "effect": eff, "distinct2": d2}
        print(f"lam_b={lam_b:>4} effect={['%+.2f'%e for e in eff]} d2={['%.2f'%x for x in d2]}", flush=True)
        torch.cuda.empty_cache()

    out_json = dict(model=exp21.MODEL_NAME, layer=LAYER, l2=L2, d_model=D_MODEL,
                    v_norm=float(np.linalg.norm(v)), w2_norm=float(np.linalg.norm(w2)),
                    prompt_len=exp22.PROMPT_LEN, gen_len=exp22.GEN_LEN, n_prompts=int(ids.shape[0]),
                    alphas=ALPHAS, B0=B0, distinct2_0=d2_0, lam_b=LAM_B, seed=SEED,
                    raw={"effect": raw_eff, "distinct2": raw_d2}, correctors=correctors,
                    n_train_texts=len(train_texts))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "23_behavioral_qwen_fix.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    # ---- figure (same 3 panels as Exp 11) ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    blues = ["#8ec6f0", "#3b8ed0", "#0b3d66"]   # light->dark = increasing λ_b
    ax[0].plot(ALPHAS, raw_eff, "s--", color="#c0392b", label="raw steer h+αv")
    ax[1].plot(ALPHAS, raw_d2, "s--", color="#c0392b", label="raw steer h+αv")
    ax[2].plot(raw_eff, raw_d2, "s--", color="#c0392b", label="raw steer h+αv")
    for a, x, y in zip(ALPHAS, raw_eff, raw_d2):
        ax[2].annotate(f"{int(a)}", (x, y), fontsize=7, color="#c0392b")
    for j, lam_b in enumerate(LAM_B):
        c = correctors[f"{lam_b:g}"]
        lab = f"corrector λ_b={lam_b:g}" + (" (Exp 22)" if lam_b == 0 else "")
        ax[0].plot(ALPHAS, c["effect"], "o-", color=blues[j], label=lab)
        ax[1].plot(ALPHAS, c["distinct2"], "o-", color=blues[j], label=lab)
        ax[2].plot(c["effect"], c["distinct2"], "o-", color=blues[j], label=lab)
        for a, x, y in zip(ALPHAS, c["effect"], c["distinct2"]):
            ax[2].annotate(f"{int(a)}", (x, y), fontsize=7, color=blues[j])
    ax[0].axhline(0, ls=":", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"sentiment shift $B(\alpha)-B(0)$")
    ax[0].set_title("(a) behavioral effect (higher = stronger steer)"); ax[0].legend(fontsize=7)
    ax[1].axhline(d2_0, ls=":", color="k", lw=1, label=f"unsteered ({d2_0:.2f})")
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("distinct-2 (unique-bigram ratio)")
    ax[1].set_title("(b) text degeneration (higher = more fluent)"); ax[1].legend(fontsize=7)
    ax[2].set_xlabel(r"sentiment shift $B(\alpha)-B(0)$  (want $\rightarrow$)")
    ax[2].set_ylabel(r"distinct-2  (want $\uparrow$)")
    ax[2].set_title("(c) effect-vs-fluency Pareto (up-right dominates)"); ax[2].legend(fontsize=7)
    fig.suptitle(f"Does the GPT-2 behavioral fix (Exp 11) transfer? Behavioral-preservation term on "
                 f"Qwen3-1.7B (steer L{LAYER}, readout L{L2})")
    fig.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, "23_behavioral_qwen_fix.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
