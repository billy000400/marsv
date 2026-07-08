"""Iteration 24 — SECOND non-GPT-2 architecture: turn the architecture axis into a SWEEP.

Exp 21 crossed the GPT-2 boundary ONCE (to Qwen3-1.7B). One point off the family is a weak
"architecture-robust" claim. This script adds a THIRD, structurally-distinct architecture so
the axis becomes a genuine sweep of three families:

  * GPT-2 (Exp 3/13/19): learned positional embeddings, LayerNorm, GELU MLP, dense MHA,
    SERIAL residual (attention then MLP).
  * Qwen3-1.7B (Exp 21): RMSNorm, rotary positions, SwiGLU MLP, grouped-query attention.
  * Pythia-410m == GPT-NeoX (THIS script): rotary positions + LayerNorm + dense MHA + GELU,
    but a PARALLEL residual (attention and MLP computed from the same input and summed) and
    untied input/output embeddings. It shares rotary with Qwen3 and LayerNorm/GELU/dense-MHA
    with GPT-2, yet its parallel-residual block is different from BOTH — a clean third point.

We replicate the flagship Exp-3 pipeline UNCHANGED (same DiffMean sentiment prompts, 400-doc
Gaussian fit, 300-doc training set, held-out 100-doc eval, 4-layer corrector, seed, objective,
alpha ~ U(0.5,8)); only the model changes. Steer & correct at the mid layer, block 12 of 24
(depth analogue of block 6 of 12 in GPT-2 small). Pythia-410m is small (d=1024, ~800MB) so it
loads in fp32 within the ~4.3 GB VRAM share and its 50304-token vocab is no eval bottleneck.

Model: EleutherAI/pythia-410m, resid_post block 12. Baseline: raw steering z=h+alpha*v.
Outputs: results/24_cross_arch_pythia.json + plots/24_cross_arch_pythia.png.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")
FINEWEB = os.path.join(HERE, "..", "..", "dir3_manifold", "data", "fineweb_texts.json")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

DEVICE = "cuda"
torch.set_num_threads(1)
torch.cuda.set_per_process_memory_fraction(0.18)

MODEL_NAME = "EleutherAI/pythia-410m"
LAYER = 12           # mid of 24 blocks
D_MODEL = 1024
ALPHAS = [1.0, 2.0, 4.0, 8.0]
SEED = 0
FIT_BATCH = 8
TRAIN_BATCH = 4
EVAL_BATCH = 4
TRAIN_SEQ = 48
EVAL_SEQ = 128


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


_model = None; _tok = None
def load():
    global _model, _tok
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float32).to(DEVICE).eval()
        for p in _model.parameters():
            p.requires_grad_(False)
    return _model, _tok


def fineweb_texts(n):
    with open(FINEWEB) as f:
        return json.load(f)[:n]


# ---- hook that swaps resid_post at LAYER (GPT-NeoX decoder layer output) ----
class FuncPatcher:
    def __init__(self, model, layer, fn):
        self.block = model.gpt_neox.layers[layer]; self.fn = fn; self.handle = None
    def _hook(self, module, inp, out):
        if isinstance(out, tuple):
            return (self.fn(out[0]),) + out[1:]
        return self.fn(out)
    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook); return self
    def __exit__(self, *a):
        self.handle.remove()


def batched_ids(tok, texts, seq_len, batch):
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                  max_length=seq_len, padding="max_length")
        yield enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)


@torch.no_grad()
def resid_post(texts, layer, seq_len=128, batch=4):
    model, tok = load()
    out = []
    for ids, mask in batched_ids(tok, texts, seq_len, batch):
        hs = model(ids, attention_mask=mask, output_hidden_states=True).hidden_states
        h = hs[layer + 1]
        out.append(h[mask.bool()].float().cpu().numpy())
    return np.concatenate(out, 0)


def make_hat(corr, h, alpha, vt, vhat_t):
    """h -> ĥ = z + P_{v⊥} r_θ(h,z,α)."""
    h = h.float()
    z = h + alpha * vt
    r = corr(h, z, alpha)
    r_perp = r - (r @ vhat_t).unsqueeze(-1) * vhat_t
    return z + r_perp, r_perp


@torch.no_grad()
def lm_loss_fn(texts, layer, fn, seq_len=EVAL_SEQ, batch=EVAL_BATCH):
    model, tok = load()
    total, count = 0.0, 0
    for ids, mask in batched_ids(tok, texts, seq_len, batch):
        with FuncPatcher(model, layer, lambda h: fn(h).to(h.dtype)):
            logits = model(ids, attention_mask=mask).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]; m = mask[:, 1:].bool()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total += nll[m].sum().item(); count += m.sum().item()
    return total / count


def train_corrector(corr, vt, vhat_t, train_texts, epochs=6, lr=1e-3, lam_near=0.05,
                    amin=0.5, amax=8.0):
    model, tok = load()
    opt = torch.optim.Adam(corr.parameters(), lr=lr)
    rng = np.random.RandomState(SEED)
    corr.train(); step = 0
    for ep in range(epochs):
        order = rng.permutation(len(train_texts))
        texts = [train_texts[i] for i in order]
        for ids, mask in batched_ids(tok, texts, TRAIN_SEQ, TRAIN_BATCH):
            alpha = float(rng.uniform(amin, amax))
            store = {}
            def fn(h):
                hat, r_perp = make_hat(corr, h.detach(), alpha, vt, vhat_t)
                store["rp"] = r_perp
                return hat.to(h.dtype)
            with FuncPatcher(model, LAYER, fn):
                logits = model(ids, attention_mask=mask).logits
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tgt = ids[:, 1:]; m = mask[:, 1:].bool()
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lm = nll[m].mean()
            near = (store["rp"] / 100.0).pow(2).sum(-1).mean()
            loss = lm + lam_near * near
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  ep{ep} step{step:4d} a={alpha:4.1f} LM={lm.item():.3f} "
                      f"near={near.item():.3f}", flush=True)
    corr.eval(); return corr


@torch.no_grad()
def corrector_acts(corr, Hs, alpha, vt, vhat_t, batch=4096):
    out = []
    for i in range(0, len(Hs), batch):
        h = torch.tensor(Hs[i:i + batch], device=DEVICE)
        hat, _ = make_hat(corr, h, alpha, vt, vhat_t)
        out.append(hat.float().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    exp03 = _load("exp03", "03_learned_corrector.py")
    exp01 = _load("exp01", "01_offmanifold_phenomenon.py")
    from projections import normalize_vector

    torch.manual_seed(SEED); np.random.seed(SEED)
    model, tok = load()
    print(f"model={MODEL_NAME} layers={model.config.num_hidden_layers} "
          f"d={model.config.hidden_size} device={DEVICE}", flush=True)

    hp = resid_post(exp01.POS, LAYER, seq_len=32, batch=8)
    hn = resid_post(exp01.NEG, LAYER, seq_len=32, batch=8)
    v = (hp.mean(0) - hn.mean(0)).astype(np.float32)
    vnorm = float(np.linalg.norm(v)); vhat = normalize_vector(v)
    vt = torch.tensor(v, device=DEVICE); vhat_t = torch.tensor(vhat, device=DEVICE)

    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=FIT_BATCH)
    mu, cov, prec = exp03.gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(10000, len(H)), replace=False)
    Hs = H[idx]; del H
    dm_clean = float(exp03.mahalanobis(Hs, mu, prec).mean())
    print(f"|v|={vnorm:.2f} mean|h|={hnorm:.2f} clean D_M={dm_clean:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]

    corr = exp03.Corrector(d=D_MODEL).to(DEVICE)
    print(f"corrector params: {sum(p.numel() for p in corr.parameters())/1e6:.2f}M", flush=True)
    ckpt = os.path.join(RESULTS, "24_corr.pt")
    if os.path.exists(ckpt):
        corr.load_state_dict(torch.load(ckpt, map_location=DEVICE)); corr.eval()
        print("loaded corrector checkpoint (skipped training)", flush=True)
    else:
        train_corrector(corr, vt, vhat_t, train_texts)
        torch.save(corr.state_dict(), ckpt)
        print("saved corrector checkpoint", flush=True)
    torch.cuda.empty_cache()

    clean_loss = lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)
    out = {"raw": {"dlm": [], "dm": []},
           "learned": {"dlm": [], "dm": [], "retention": []}}
    for a in ALPHAS:
        def f_raw(h):  return h + a * vt
        def f_learn(h):
            hat, _ = make_hat(corr, h, a, vt, vhat_t); return hat
        Z = Hs + a * v[None, :]
        Zl = corrector_acts(corr, Hs, a, vt, vhat_t)
        out["raw"]["dlm"].append(lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss)
        out["raw"]["dm"].append(float(exp03.mahalanobis(Z, mu, prec).mean()))
        out["learned"]["dlm"].append(lm_loss_fn(eval_texts, LAYER, f_learn) - clean_loss)
        out["learned"]["dm"].append(float(exp03.mahalanobis(Zl, mu, prec).mean()))
        out["learned"]["retention"].append(float(((Zl - Hs) @ vhat).mean()))
        rec = 1.0 - out["learned"]["dlm"][-1] / out["raw"]["dlm"][-1]
        print(f"a={a:>4} raw ΔLM={out['raw']['dlm'][-1]:+.3f} "
              f"learned={out['learned']['dlm'][-1]:+.3f} rec={rec*100:.0f}% "
              f"D_M raw={out['raw']['dm'][-1]:.1f} learn={out['learned']['dm'][-1]:.1f}", flush=True)

    res = dict(model=MODEL_NAME, arch="gpt_neox (parallel residual / rotary / LayerNorm / dense MHA)",
               layer=LAYER, d_model=D_MODEL, v_norm=vnorm,
               mean_h_norm=hnorm, dm_clean=dm_clean, clean_loss=clean_loss,
               alphas=ALPHAS, seed=SEED, n_fit_docs=400,
               n_train_texts=len(train_texts), n_eval_texts=len(eval_texts),
               corr_params=sum(p.numel() for p in corr.parameters()), results=out)
    with open(os.path.join(RESULTS, "24_cross_arch_pythia.json"), "w") as f:
        json.dump(res, f, indent=2)

    rec = [1.0 - out["learned"]["dlm"][i] / out["raw"]["dlm"][i] for i in range(len(ALPHAS))]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(ALPHAS, out["raw"]["dlm"], "o--", color="#c0392b", label="raw steer  h+αv")
    ax[0].plot(ALPHAS, out["learned"]["dlm"], "o-", color="#2980b9", label="learned corrector")
    ax[0].axhline(0, ls=":", color="k", lw=1)
    ax[0].set_xlabel(r"steering strength $\alpha$"); ax[0].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[0].set_title("(a) LM damage: raw vs corrected"); ax[0].legend(fontsize=8)
    ax[1].plot(ALPHAS, [x * 100 for x in rec], "o-", color="#16a085")
    ax[1].axhline(0, ls=":", color="k", lw=1); ax[1].axhline(100, ls=":", color="gray", lw=1)
    ax[1].set_xlabel(r"steering strength $\alpha$"); ax[1].set_ylabel("fluency recovery (%)")
    ax[1].set_title(r"(b) recovery = 1 - $\Delta$LM$_{corr}$/$\Delta$LM$_{raw}$")
    ax[2].plot(ALPHAS, out["raw"]["dm"], "o--", color="#c0392b", label="raw")
    ax[2].plot(ALPHAS, out["learned"]["dm"], "o-", color="#2980b9", label="corrected")
    ax[2].axhline(dm_clean, ls="--", color="gray", lw=1, label=f"real acts ({dm_clean:.1f})")
    ax[2].set_xlabel(r"steering strength $\alpha$"); ax[2].set_ylabel("mean Mahalanobis $D_M$")
    ax[2].set_title("(c) off-Gaussian distance (corrected > raw)"); ax[2].legend(fontsize=8)
    fig.suptitle("ColdSteer replicates on Pythia-410m / GPT-NeoX (24 blocks, block 12/24): a "
                 "THIRD architecture (parallel residual / rotary / LayerNorm / dense MHA)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "24_cross_arch_pythia.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
