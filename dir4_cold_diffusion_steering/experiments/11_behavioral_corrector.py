"""Iteration 11 — can a BEHAVIORAL-preservation term push Exp 10's effect-fluency Pareto out?

Exp 10 found the flagship sentiment corrector's fluency win is NOT free: trained only on
the downstream LM loss at matched layer-6 projection, the correction (orthogonal to v at
layer 6) is NOT orthogonal to the downstream sentiment READOUT, so minimizing LM loss drives
it to near-normal, only-weakly-steered generations (behavioral effect ~1/6 of raw). The
matched layer-6 projection did not translate into matched behavioral steering.

This experiment asks: if we ADD an explicit term that preserves the DOWNSTREAM sentiment
readout (measured at a later layer, L2, inside the same teacher-forced forward pass) toward
what RAW steering achieves, can we recover more of the behavioral effect while keeping
generation more fluent than raw — i.e. push the effect-vs-fluency Pareto frontier of Exp 10
outward? We train a family of correctors with behavioral-weight λ_b ∈ {0, ...} (λ_b=0 is the
Exp 10 corrector) and score every one on the SAME generation protocol as Exp 10.

Behavioral term (teacher-forced, per batch, steering strength α ~ U(0.5,8)):
  Let ŵ = unit DiffMean sentiment direction at downstream layer L2. Under raw steering
  z=h+αv, the upper LM carries a downstream sentiment projection p_raw = ⟨resid@L2, ŵ⟩; the
  corrected ĥ = z+P_{v⊥}r_θ carries p_corr. We add
      L_behav = mean_tokens( ((p_corr − p_raw)/100)^2 )
  so the corrector is pushed to reproduce raw's downstream sentiment content, not just its
  layer-6 projection. Total loss = LM_CE + λ_near·‖P_{v⊥}r‖²/100² + λ_b·L_behav.

Outputs: results/11_behavioral_corrector.json + plots/11_behavioral_corrector.png.
"""
import os, json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import fineweb_texts, load_model, resid_post, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


exp01 = _load("exp01", "01_offmanifold_phenomenon.py")   # POS / NEG sentiment prompts
exp03 = _load("exp03", "03_learned_corrector.py")        # Corrector, make_hat, FuncPatcher, batched_ids
exp10 = _load("exp10", "10_behavioral_pareto.py")        # prompt_ids, generate, sentiment_effect, distinct2

LAYER = exp03.LAYER          # 6 — steer / correct here
L2 = 11                      # downstream readout layer (resid_post block 11 = hidden_states[12], feeds the head)

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

LAM_B = [0.0, 10.0, 40.0]    # behavioral-preservation weights (0 = Exp 10 corrector)


class Capture:
    """Forward hook that stores resid_post of block `layer` from the current pass."""
    def __init__(self, model, layer):
        self.block = model.transformer.h[layer]; self.store = {}; self.handle = None
    def _hook(self, m, i, o):
        self.store["h"] = o[0] if isinstance(o, tuple) else o
    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook); return self
    def __exit__(self, *a):
        self.handle.remove()


def train_behav(corr, vt, vhat_t, w2hat_t, train_texts, lam_b,
                seq_len=64, batch=8, epochs=6, lr=1e-3, lam_near=0.05, amin=0.5, amax=8.0):
    """Exp 3 recipe + a downstream-readout preservation term (weight lam_b)."""
    model, tok = load_model()
    opt = torch.optim.Adam(corr.parameters(), lr=lr)
    rng = np.random.RandomState(SEED)
    corr.train()
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(train_texts))
        texts = [train_texts[i] for i in order]
        for ids, mask in exp03.batched_ids(tok, texts, seq_len, batch):
            alpha = float(rng.uniform(amin, amax))
            m = mask.bool()

            # --- raw downstream target (no grad) ---
            with torch.no_grad(), exp03.FuncPatcher(model, LAYER, lambda h: h + alpha * vt), \
                 Capture(model, L2) as capR:
                model(ids, attention_mask=mask)
                p_raw = (capR.store["h"] @ w2hat_t)      # [B, T]

            # --- corrected pass (grad) ---
            store = {}
            def fn(h):
                h = h.detach()
                hat, r_perp = exp03.make_hat(corr, h, alpha, vt, vhat_t)
                store["rp"] = r_perp
                return hat
            with exp03.FuncPatcher(model, LAYER, fn), Capture(model, L2) as capC:
                logits = model(ids, attention_mask=mask).logits
                p_corr = (capC.store["h"] @ w2hat_t)     # [B, T]

            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tgt = ids[:, 1:]; mm = mask[:, 1:].bool()
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lm = nll[mm].mean()
            near = (store["rp"] / 100.0).pow(2).sum(-1).mean()
            behav = (((p_corr - p_raw) / 100.0) ** 2)[m].mean()
            loss = lm + lam_near * near + lam_b * behav
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 40 == 0:
                print(f"  lam_b={lam_b:>4} ep{ep} step{step:4d} a={alpha:4.1f} "
                      f"LM={lm.item():.3f} behav={behav.item():.3f} "
                      f"p_raw={p_raw[m].mean().item():+.1f} p_corr={p_corr[m].mean().item():+.1f}",
                      flush=True)
    corr.eval()
    return corr


@torch.no_grad()
def eval_generation(model, tok, corr, ids, vt, vhat_t, B0, alphas):
    """Exp 10 generation protocol for one corrector: effect B(α)-B0 and distinct-2."""
    eff, d2 = [], []
    for a in alphas:
        def f_corr(h):
            hat, _ = exp03.make_hat(corr, h, a, vt, vhat_t); return hat
        out = exp10.generate(model, ids, f_corr)
        eff.append(exp10.sentiment_effect(model, out, vhat_t) - B0)
        d2.append(exp10.distinct2(out[:, exp10.PROMPT_LEN:].cpu().numpy()))
    return eff, d2


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

    # downstream sentiment readout direction at layer L2 (DiffMean on the same POS/NEG prompts)
    hp = resid_post(exp01.POS, L2, seq_len=32, batch=8)
    hn = resid_post(exp01.NEG, L2, seq_len=32, batch=8)
    w2 = (hp.mean(0) - hn.mean(0)).astype(np.float32)
    w2hat_t = torch.tensor(normalize_vector(w2), device=DEVICE)
    print(f"|v|(L6)={np.linalg.norm(v):.2f}  |w|(L{L2})={np.linalg.norm(w2):.2f}", flush=True)

    # data splits identical to Exp 10
    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    ids = exp10.prompt_ids(tok, all_texts[400:500], n_prompts=48)
    print(f"{ids.shape[0]} prompts x {exp10.PROMPT_LEN} tokens, gen {exp10.GEN_LEN}", flush=True)

    # baseline (unsteered greedy continuation) — reused for all correctors
    clean_out = exp10.generate(model, ids, lambda h: h)
    B0 = exp10.sentiment_effect(model, clean_out, vhat_t)
    d2_0 = exp10.distinct2(clean_out[:, exp10.PROMPT_LEN:].cpu().numpy())
    print(f"clean: B0={B0:+.3f} distinct2={d2_0:.3f}", flush=True)

    alphas = [2.0, 4.0, 6.0, 8.0]

    # raw reference (no corrector) — identical for every λ_b
    raw_eff, raw_d2 = [], []
    for a in alphas:
        out = exp10.generate(model, ids, (lambda av: (lambda h: h + av * vt))(a))
        raw_eff.append(exp10.sentiment_effect(model, out, vhat_t) - B0)
        raw_d2.append(exp10.distinct2(out[:, exp10.PROMPT_LEN:].cpu().numpy()))
    print(f"raw   effect={['%+.2f'%e for e in raw_eff]} d2={['%.2f'%x for x in raw_d2]}", flush=True)

    correctors = {}
    for lam_b in LAM_B:
        torch.manual_seed(SEED); np.random.seed(SEED)
        corr = exp03.Corrector().to(DEVICE)
        train_behav(corr, vt, vhat_t, w2hat_t, train_texts, lam_b)
        eff, d2 = eval_generation(model, tok, corr, ids, vt, vhat_t, B0, alphas)
        correctors[f"{lam_b:g}"] = {"lam_b": lam_b, "effect": eff, "distinct2": d2}
        print(f"lam_b={lam_b:>4} effect={['%+.2f'%e for e in eff]} d2={['%.2f'%x for x in d2]}", flush=True)

    out_json = dict(layer=LAYER, l2=L2, v_norm=float(np.linalg.norm(v)),
                    w2_norm=float(np.linalg.norm(w2)), prompt_len=exp10.PROMPT_LEN,
                    gen_len=exp10.GEN_LEN, n_prompts=int(ids.shape[0]), alphas=alphas,
                    B0=B0, distinct2_0=d2_0, lam_b=LAM_B,
                    raw={"effect": raw_eff, "distinct2": raw_d2}, correctors=correctors,
                    n_train_texts=len(train_texts))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "11_behavioral_corrector.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    blues = ["#8ec6f0", "#3b8ed0", "#0b3d66"]   # light->dark = increasing λ_b
    ax[0].plot(alphas, raw_eff, "s--", color="#c0392b", label="raw steer h+αv")
    ax[1].plot(alphas, raw_d2, "s--", color="#c0392b", label="raw steer h+αv")
    ax[2].plot(raw_eff, raw_d2, "s--", color="#c0392b", label="raw steer h+αv")
    for a, x, y in zip(alphas, raw_eff, raw_d2):
        ax[2].annotate(f"{int(a)}", (x, y), fontsize=7, color="#c0392b")
    for j, lam_b in enumerate(LAM_B):
        c = correctors[f"{lam_b:g}"]
        lab = f"corrector λ_b={lam_b:g}" + (" (Exp 10)" if lam_b == 0 else "")
        ax[0].plot(alphas, c["effect"], "o-", color=blues[j], label=lab)
        ax[1].plot(alphas, c["distinct2"], "o-", color=blues[j], label=lab)
        ax[2].plot(c["effect"], c["distinct2"], "o-", color=blues[j], label=lab)
        for a, x, y in zip(alphas, c["effect"], c["distinct2"]):
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
    fig.suptitle(f"Behavioral-preservation term: pushing the effect-fluency Pareto (GPT-2 small, L6 steer, L{L2} readout)")
    fig.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, "11_behavioral_corrector.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
