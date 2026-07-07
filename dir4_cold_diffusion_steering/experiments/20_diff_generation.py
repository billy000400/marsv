"""Iteration 20 — break the Exp 11 ceiling by supervising the behavioral readout
THROUGH differentiable (autoregressive) generation, not teacher-forced text.

Exp 10 found the flagship sentiment corrector under-steers in generation; Exp 11 added a
behavioral term matching the corrector's DOWNSTREAM sentiment readout to raw steering's — but
measured that readout on a TEACHER-FORCED pass over ground-truth FineWeb tokens. Its ceiling
(effect never past ~+1.3) was diagnosed as a proxy gap: matching a *teacher-forced* readout
only partially transfers to *autoregressive* generation.

This experiment closes that gap directly: it supervises the readout on the corrector's OWN
generated continuation via a DIFFERENTIABLE soft-token rollout. At each of K steps we forward
the model on the current sequence (steer applied at LAYER at every position), read the
downstream sentiment projection at L2 for the just-produced position, then feed the
softmax-weighted expected token embedding (probs @ Wte) back as the next input — a fully
differentiable autoregressive rollout. We push the corrected rollout's readout toward the
readout of raw steering's OWN rollout, backpropping through the K-step unroll into r_θ.

  L_gen = mean_over_K( ((p_corr^gen − p_raw^gen)/100)^2 )

Total loss = LM_CE(teacher-forced, Exp 3) + λ_near·‖P_{v⊥}r‖²/100² + λ_g·L_gen.
λ_g = 0 is exactly the Exp 3 / Exp 10-11 base corrector (built-in reproducibility check).
Every corrector is scored on the IDENTICAL Exp 10/11 generation protocol (hard greedy, 48
prompts, 30 tokens; sentiment effect B(α)−B(0) and distinct-2 on a clean re-encode).

Outputs: results/20_diff_generation.json + plots/20_diff_generation.png.
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


exp01 = _load("exp01", "01_offmanifold_phenomenon.py")
exp03 = _load("exp03", "03_learned_corrector.py")
exp10 = _load("exp10", "10_behavioral_pareto.py")
exp11 = _load("exp11", "11_behavioral_corrector.py")

LAYER = exp03.LAYER          # 6 — steer / correct here
L2 = exp11.L2                # 11 — downstream readout layer

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

LAM_G = [0.0, 40.0, 160.0]   # differentiable-generation behavioral weights (0 = Exp 10/11 base)
GEN_B = 4                    # rollout batch (per-agent VRAM share); halved on OOM
GEN_P = 8                    # rollout prompt length (tokens)
K = 8                        # rollout steps (differentiable)
TEMP = 1.0                   # softmax temperature for the soft-token feedback


def soft_rollout(model, wte, prompt_embeds, steer_fn, w2hat_t, k=K, temp=TEMP):
    """Differentiable autoregressive rollout with soft-token feedback.
    Returns [B, k] downstream sentiment readouts of the k produced positions."""
    cur = prompt_embeds                              # [B, P, d]
    reads = []
    for _ in range(k):
        with exp03.FuncPatcher(model, LAYER, steer_fn), exp11.Capture(model, L2) as cap:
            logits = model(inputs_embeds=cur).logits[:, -1, :]     # [B, V]
        reads.append(cap.store["h"][:, -1, :] @ w2hat_t)           # [B]
        probs = torch.softmax(logits / temp, dim=-1)               # [B, V]
        nxt = probs @ wte                                          # [B, d] expected embedding
        cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
    return torch.stack(reads, dim=1)                               # [B, k]


def train_diffgen(corr, vt, vhat_t, w2hat_t, train_texts, lam_g,
                  seq_len=64, batch=8, epochs=6, lr=1e-3, lam_near=0.05, amin=0.5, amax=8.0):
    """Exp 3 recipe (teacher-forced LM CE) + a differentiable-generation behavioral term."""
    global GEN_B
    model, tok = load_model()
    wte = model.transformer.wte.weight                # [V, d], frozen
    opt = torch.optim.Adam(corr.parameters(), lr=lr)
    rng = np.random.RandomState(SEED)
    corr.train()
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(train_texts))
        texts = [train_texts[i] for i in order]
        for ids, mask in exp03.batched_ids(tok, texts, seq_len, batch):
            alpha = float(rng.uniform(amin, amax))
            opt.zero_grad()

            # --- teacher-forced LM CE (identical Exp 3) ---
            store = {}
            def fn_tf(h):
                h = h.detach()
                hat, rp = exp03.make_hat(corr, h, alpha, vt, vhat_t)
                store["rp"] = rp; return hat
            with exp03.FuncPatcher(model, LAYER, fn_tf):
                logits = model(ids, attention_mask=mask).logits
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tgt = ids[:, 1:]; mm = mask[:, 1:].bool()
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lm = nll[mm].mean()
            near = (store["rp"] / 100.0).pow(2).sum(-1).mean()
            (lm + lam_near * near).backward()          # frees the TF graph before the gen graph

            l_gen = 0.0
            if lam_g > 0:
                gb = min(GEN_B, ids.shape[0])
                pe = wte[ids[:gb, :GEN_P]]             # [gb, P, d] (frozen embeddings)
                try:
                    with torch.no_grad():
                        p_raw = soft_rollout(model, wte, pe,
                                             (lambda av: (lambda h: h + av * vt))(alpha), w2hat_t)
                    def fn_gen(h):
                        hat, _ = exp03.make_hat(corr, h, alpha, vt, vhat_t); return hat
                    p_corr = soft_rollout(model, wte, pe, fn_gen, w2hat_t)
                    lg = ((p_corr - p_raw) / 100.0).pow(2).mean()
                    (lam_g * lg).backward()
                    l_gen = lg.item()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); GEN_B = max(1, GEN_B // 2)
                    print(f"  [OOM] halved GEN_B -> {GEN_B}; skipping gen term this step", flush=True)

            opt.step()
            step += 1
            if step % 40 == 0:
                print(f"  lam_g={lam_g:>5} ep{ep} step{step:4d} a={alpha:4.1f} "
                      f"LM={lm.item():.3f} L_gen={l_gen:.4f}", flush=True)
    corr.eval()
    return corr


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

    # downstream sentiment readout direction at L2 (same as Exp 11)
    hp = resid_post(exp01.POS, L2, seq_len=32, batch=8)
    hn = resid_post(exp01.NEG, L2, seq_len=32, batch=8)
    w2 = (hp.mean(0) - hn.mean(0)).astype(np.float32)
    w2hat_t = torch.tensor(normalize_vector(w2), device=DEVICE)
    print(f"|v|(L6)={np.linalg.norm(v):.2f}  |w|(L{L2})={np.linalg.norm(w2):.2f}", flush=True)

    # identical data splits & prompts to Exp 10/11
    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    ids = exp10.prompt_ids(tok, all_texts[400:500], n_prompts=48)
    print(f"{ids.shape[0]} prompts x {exp10.PROMPT_LEN} tokens, gen {exp10.GEN_LEN}", flush=True)

    clean_out = exp10.generate(model, ids, lambda h: h)
    B0 = exp10.sentiment_effect(model, clean_out, vhat_t)
    d2_0 = exp10.distinct2(clean_out[:, exp10.PROMPT_LEN:].cpu().numpy())
    print(f"clean: B0={B0:+.3f} distinct2={d2_0:.3f}", flush=True)

    alphas = [2.0, 4.0, 6.0, 8.0]

    raw_eff, raw_d2 = [], []
    for a in alphas:
        out = exp10.generate(model, ids, (lambda av: (lambda h: h + av * vt))(a))
        raw_eff.append(exp10.sentiment_effect(model, out, vhat_t) - B0)
        raw_d2.append(exp10.distinct2(out[:, exp10.PROMPT_LEN:].cpu().numpy()))
    print(f"raw   effect={['%+.2f'%e for e in raw_eff]} d2={['%.2f'%x for x in raw_d2]}", flush=True)

    correctors, samples = {}, {}
    for lam_g in LAM_G:
        torch.manual_seed(SEED); np.random.seed(SEED)
        corr = exp03.Corrector().to(DEVICE)
        train_diffgen(corr, vt, vhat_t, w2hat_t, train_texts, lam_g)
        eff, d2 = exp11.eval_generation(model, tok, corr, ids, vt, vhat_t, B0, alphas)
        correctors[f"{lam_g:g}"] = {"lam_g": lam_g, "effect": eff, "distinct2": d2}
        # one sample continuation at alpha=8
        def f8(h):
            hat, _ = exp03.make_hat(corr, h, 8.0, vt, vhat_t); return hat
        out8 = exp10.generate(model, ids, f8)
        samples[f"lam_g{lam_g:g}_a8"] = tok.decode(out8[0], skip_special_tokens=True)
        print(f"lam_g={lam_g:>5} effect={['%+.2f'%e for e in eff]} d2={['%.2f'%x for x in d2]}", flush=True)

    out_json = dict(layer=LAYER, l2=L2, v_norm=float(np.linalg.norm(v)),
                    w2_norm=float(np.linalg.norm(w2)), prompt_len=exp10.PROMPT_LEN,
                    gen_len=exp10.GEN_LEN, n_prompts=int(ids.shape[0]), alphas=alphas,
                    B0=B0, distinct2_0=d2_0, lam_g=LAM_G, gen_prompt_len=GEN_P, rollout_k=K,
                    gen_batch=GEN_B, raw={"effect": raw_eff, "distinct2": raw_d2},
                    correctors=correctors, samples=samples, n_train_texts=len(train_texts))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "20_diff_generation.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    greens = ["#9bd6a3", "#2fa14e", "#0b5323"]   # light->dark = increasing λ_g
    ax[0].plot(alphas, raw_eff, "s--", color="#c0392b", label="raw steer h+αv")
    ax[1].plot(alphas, raw_d2, "s--", color="#c0392b", label="raw steer h+αv")
    ax[2].plot(raw_eff, raw_d2, "s--", color="#c0392b", label="raw steer h+αv")
    for a, x, y in zip(alphas, raw_eff, raw_d2):
        ax[2].annotate(f"{int(a)}", (x, y), fontsize=7, color="#c0392b")
    for j, lam_g in enumerate(LAM_G):
        c = correctors[f"{lam_g:g}"]
        lab = f"gen-supervised λ_g={lam_g:g}" + (" (=Exp 10 base)" if lam_g == 0 else "")
        ax[0].plot(alphas, c["effect"], "o-", color=greens[j], label=lab)
        ax[1].plot(alphas, c["distinct2"], "o-", color=greens[j], label=lab)
        ax[2].plot(c["effect"], c["distinct2"], "o-", color=greens[j], label=lab)
        for a, x, y in zip(alphas, c["effect"], c["distinct2"]):
            ax[2].annotate(f"{int(a)}", (x, y), fontsize=7, color=greens[j])
    ax[0].axhline(0, ls=":", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"sentiment shift $B(\alpha)-B(0)$")
    ax[0].set_title("(a) behavioral effect (higher = stronger steer)"); ax[0].legend(fontsize=7)
    ax[1].axhline(d2_0, ls=":", color="k", lw=1, label=f"unsteered ({d2_0:.2f})")
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("distinct-2 (unique-bigram ratio)")
    ax[1].set_title("(b) text degeneration (higher = more fluent)"); ax[1].legend(fontsize=7)
    ax[2].set_xlabel(r"sentiment shift $B(\alpha)-B(0)$  (want $\rightarrow$)")
    ax[2].set_ylabel(r"distinct-2  (want $\uparrow$)")
    ax[2].set_title("(c) effect-vs-fluency Pareto (up-right dominates)"); ax[2].legend(fontsize=7)
    fig.suptitle(f"Differentiable-generation behavioral supervision (GPT-2 small, L6 steer, L{L2} readout, K={K} rollout)")
    fig.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, "20_diff_generation.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
