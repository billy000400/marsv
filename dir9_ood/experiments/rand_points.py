"""Direction #9 — operator request 2026-07-09 (`human_feedback_07082204.md`):
"try: GPT-2 XL, and OOD detection with randomly sampled points in residual streams".

Two additions to the existing negative-result study, both forward-only (no autograd):

1. NEW DETECTOR `rand-points`: "OOD detection with randomly sampled points in the residual
   stream". For each input, take the last-token activation h at a measurement point, then sample
   K RANDOM POINTS h_k = h + sigma * z_k  (z_k ~ N(0, I), sigma = rel_sigma * ||h||, so the cloud
   scales with the local activation magnitude). Continue the forward pass from each random point to
   get K next-token distributions p_1..p_K. Score the input by how much those outputs DISPERSE:

       rand-points-disp(x) = (1/K) * sum_k KL( p_k || p_bar ),   p_bar = (1/K) sum_k p_k

   This is the epistemic-uncertainty / "plateau-width" term (a.k.a. BALD mutual-information): on a
   flat plateau (in-distribution) the random points map to nearly the same output, so dispersion is
   low; off-plateau (OOD) the outputs scatter, so dispersion is high. Higher => more OOD. We also
   report `rand-points-ent` = total predictive entropy H(p_bar) as a secondary summary.

   This differs from the existing `plateau-perturbation` (mean KL of each perturbed output vs the
   CLEAN output, at fixed unit-direction magnitude eps=6): rand-points measures the SPREAD AMONG the
   randomly sampled outputs themselves (a Monte-Carlo epistemic estimate), not distance-from-clean.

2. SCALING CHECK: run the same detector + reference baselines (MSP, Mahalanobis) on a LARGER GPT-2.
   GPT-2 XL is not in the offline HF cache; the largest cached is gpt2-large (774M), used here. Does
   the negative result (plateau-style signals lose to baselines) hold at ~6x the parameters?

Canonical split identical to plateau_v2.py: randperm(seed=7), fit=perm[:NFIT], test=perm[NFIT:NFIT+N].
All GPT-2 sizes share the same BPE tokenizer, so token ids / OOD sets are identical across models.
Output: results/auroc_randpoints.csv  [model, ood_set, method, measurement_point, auroc].
"""
import os, sys, time, csv
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import plateau_score as ps

torch.set_num_threads(1)
torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.180)   # BUDGET: ~4.3 GB, 1 of 5 agents

HERE = os.path.dirname(__file__)


def load_model(name):
    """Load a GPT-2 variant in fp16 (forward-only) and register it into plateau_score so its data
    helpers reuse the same tokenizer (all GPT-2 sizes share the gpt2 BPE vocab)."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(name)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=dtype).eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    ps._MODEL, ps._TOK = model, tok
    return model, tok


def points_for(model):
    """Measurement points at fixed FRACTIONAL depth so they're comparable across model sizes:
    input embeddings + residual stream at ~1/4, 1/2, 3/4 depth."""
    depth = len(model.transformer.h)
    layers = sorted({int(depth * f) for f in (0.25, 0.5, 0.75)})
    return ["input"] + [f"resid{L}" for L in layers]


def _module(model, point):
    if point == "input":
        return model.transformer.drop
    return model.transformer.h[int(point[5:])]


def _hs_index(point):
    return 0 if point == "input" else int(point[5:]) + 1


def _batches(ids, bs):
    for i in range(0, ids.shape[0], bs):
        yield ids[i:i + bs].to(DEVICE)


@torch.no_grad()
def collect_acts_msp(model, ids, points, seq_len, bs):
    """One batched clean forward: last-token activations at every point + MSP. ({point:[N,d]}, msp[N])."""
    pos = seq_len - 1
    acts = {p: [] for p in points}
    msp = []
    idx = {p: _hs_index(p) for p in points}
    for b in _batches(ids, bs):
        out = model(b, output_hidden_states=True)
        hs = out.hidden_states
        for p in points:
            acts[p].append(hs[idx[p]][:, pos].float().cpu())
        msp.append((1.0 - torch.softmax(out.logits[:, pos].float(), -1).max(-1).values).cpu())
    return {p: torch.cat(acts[p]).numpy() for p in points}, torch.cat(msp).numpy()


@torch.no_grad()
def randpoints_score(model, ids, point, seq_len, K=16, rel_sigma=0.1, bs=8):
    """Randomly-sampled-points detector at `point`. Returns (disp[N], ent[N]); both higher => OOD."""
    pos = seq_len - 1
    hdim = model.config.n_embd
    g = torch.Generator(device=DEVICE).manual_seed(2024)
    inj = {"delta": None, "cap": None}

    def hook(mod, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if inj["cap"] is not None:                       # clean pass: record last-token norm
            inj["cap"] = h[:, pos].float().norm(dim=-1)
        if inj["delta"] is None:
            return o
        h = h.clone()
        h[:, pos] = h[:, pos] + inj["delta"]
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    handle = _module(model, point).register_forward_hook(hook)
    disp_all, ent_all = [], []
    try:
        for b in _batches(ids, bs):
            B = b.shape[0]
            inj["delta"], inj["cap"] = None, torch.zeros(B, device=DEVICE)
            model(b)                                      # clean fwd only to grab activation norm
            r = inj["cap"]; inj["cap"] = None
            sigma = (rel_sigma * r).clamp_min(1e-6)       # [B] per-sample noise scale
            logps = []
            for _ in range(K):
                z = torch.randn(B, hdim, generator=g, device=DEVICE)
                inj["delta"] = sigma[:, None] * z
                logp = torch.log_softmax(model(b).logits[:, pos].float(), -1)   # [B,V]
                logps.append(logp)
                inj["delta"] = None
            logps = torch.stack(logps)                    # [K,B,V]
            ps_ = logps.exp()
            p_bar = ps_.mean(0)                           # [B,V]
            logp_bar = p_bar.clamp_min(1e-12).log()
            disp = (ps_ * (logps - logp_bar[None])).sum(-1).mean(0)            # [B] mean_k KL(p_k||p_bar)
            ent = -(p_bar * logp_bar).sum(-1)             # [B] H(p_bar)
            disp_all.append(disp.cpu()); ent_all.append(ent.cpu())
    finally:
        handle.remove()
    return torch.cat(disp_all).numpy(), torch.cat(ent_all).numpy()


def naive_mahalanobis(fit, test, eps=1e-3):
    mu = fit.mean(0); X = fit - mu
    cov = (X.T @ X) / X.shape[0] + eps * np.eye(X.shape[1])
    P = np.linalg.inv(cov); D = test - mu
    return np.einsum("nd,dk,nk->n", D, P, D)


def run_model(name, N, SEQ, NFIT, K, rel_sigma, bs, t0):
    model, tok = load_model(name)
    points = points_for(model)
    print(f"[{name}] DEVICE={DEVICE} depth={len(model.transformer.h)} points={points} "
          f"N={N} SEQ={SEQ} NFIT={NFIT} K={K} rel_sigma={rel_sigma} bs={bs}", flush=True)
    if DEVICE == "cuda":
        print("  GPU:", torch.cuda.get_device_name(0), flush=True)

    texts = [l.rstrip("\n") for l in open(os.path.join(HERE, "..", "data", "fineweb_sample.txt"))]
    all_ids = ps.encode_fixed(texts, seq_len=SEQ)
    g_split = torch.Generator().manual_seed(7)
    perm = torch.randperm(all_ids.shape[0], generator=g_split)
    id_fit = all_ids[perm[:NFIT]]
    id_test = all_ids[perm[NFIT:NFIT + N]]
    OOD = {k: ps.make_ood(id_test, k, seed=0) for k in ("random", "shuffled", "code")}
    print(f"  id_test {tuple(id_test.shape)} id_fit {tuple(id_fit.shape)} "
          + " ".join(f"{k}{tuple(v.shape)}" for k, v in OOD.items()), flush=True)

    fit_acts, _ = collect_acts_msp(model, id_fit, points, SEQ, bs)

    def score_set(ids, tag):
        acts, msp = collect_acts_msp(model, ids, points, SEQ, bs)
        disp, ent = {}, {}
        for p in points:
            disp[p], ent[p] = randpoints_score(model, ids, p, SEQ, K=K, rel_sigma=rel_sigma, bs=bs)
        print(f"  [{name}] scored {tag} {time.time()-t0:.0f}s", flush=True)
        return {"acts": acts, "msp": msp, "disp": disp, "ent": ent}

    S_id = score_set(id_test, "id_test")
    S_ood = {k: score_set(v, k) for k, v in OOD.items()}

    rows = []

    def addrow(ood, method, point, sid, sood):
        a = ps.auroc(np.concatenate([sid, sood]),
                     np.concatenate([np.zeros(len(sid)), np.ones(len(sood))]))
        rows.append([name, ood, method, point, round(float(a), 4)])

    for k in OOD:
        addrow(k, "baseline-MSP", "n/a", S_id["msp"], S_ood[k]["msp"])
        for p in points:
            addrow(k, "rand-points-disp", p, S_id["disp"][p], S_ood[k]["disp"][p])
            addrow(k, "rand-points-ent", p, S_id["ent"][p], S_ood[k]["ent"][p])
            addrow(k, "baseline-mahalanobis", p,
                   naive_mahalanobis(fit_acts[p], S_id["acts"][p]),
                   naive_mahalanobis(fit_acts[p], S_ood[k]["acts"][p]))
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    t0 = time.time()
    # small run: full N; large run: trimmed for the shared VRAM/time budget
    CFG = {"gpt2":       dict(N=200, SEQ=64, NFIT=1000, K=16, rel_sigma=0.1, bs=8),
           "gpt2-large": dict(N=150, SEQ=64, NFIT=600,  K=8,  rel_sigma=0.1, bs=4)}
    which = sys.argv[1:] or ["gpt2", "gpt2-large"]
    all_rows = []
    for name in which:
        all_rows += run_model(name, t0=t0, **CFG[name])
    out = os.path.join(HERE, "..", "results", "auroc_randpoints.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ood_set", "method", "measurement_point", "auroc"])
        w.writerows(all_rows)
    print(f"\nwrote {out} ({len(all_rows)} rows) in {time.time()-t0:.0f}s", flush=True)
    for r in sorted(all_rows, key=lambda x: (x[0], x[1], -x[4])):
        print(r)
