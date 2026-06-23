"""Direction #9 v2 — GPU rerun addressing the operator review (CODEX_REVIEW.md + human_feedback.md).

Fixes vs iter-1:
  * RUNS ON GPU. The iter-1 "V100/CPU-only" story was FALSE: this box has a working NVIDIA A10
    (sm_86, CUDA 13.2). We cap VRAM to BUDGET's 0.45 fraction and 2 CPU threads.
  * GENUINE Jacobian-Frobenius plateau metric `plateau-jacFrob`: Hutchinson estimate of
    ||d logp(next token)/d h||_F at the measurement point (k random Gaussian output directions).
    This measures local flatness of the OUTPUT DISTRIBUTION map — label-free, not argmax-confidence.
  * The iter-1 metric (grad-norm of the model's own argmax NLL) is kept but RENAMED honestly to
    `selfNLL-grad` (it is MSP-adjacent, per the review), for transparency / comparison.
  * cupbearer baselines (human feedback): cupbearer's ACTUAL detector math, vendored verbatim in
    cupbearer_helpers.py: `cup-RMD` (relative Mahalanobis) and `cup-QUE` (Quantum-Entropy/SPECTRE).
    The full package is NOT installed (pins numpy<2 vs shared numpy 2.3.3 + heavy lightning/tv deps).
  * Larger N and a much larger covariance-fit set (addresses "Mahalanobis underpowered").

All scores oriented so HIGHER == more OOD. AUROC = pure-numpy Mann-Whitney (from plateau_score).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import plateau_score as ps           # reuse encode_fixed / make_ood / load_code_seqs / auroc
import cupbearer_helpers as cup       # vendored cupbearer detector math

torch.set_num_threads(2)
torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.45)
POINTS = ["input", "resid3", "resid6", "resid9"]


def load_model():
    """Load GPT-2 small on DEVICE and register it into plateau_score so its data helpers
    (encode_fixed/make_ood/load_code_seqs) reuse the same tokenizer without a 2nd load."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    ps._MODEL, ps._TOK = model, tok   # so ps.encode_fixed etc. don't load a 2nd (CPU) model
    return model, tok


def _module(model, point):
    if point == "input":
        return model.transformer.drop
    if point.startswith("resid"):
        return model.transformer.h[int(point[5:])]
    raise ValueError(point)


def _hs_index(point):
    return 0 if point == "input" else int(point[5:]) + 1


def _batches(ids, bs):
    for i in range(0, ids.shape[0], bs):
        yield ids[i:i + bs].to(DEVICE)


@torch.no_grad()
def collect_all(ids, seq_len, bs=32):
    """One batched forward: last-token activations at every point + MSP. Returns ({point:[N,d]}, msp[N])."""
    model, _ = load_model()
    pos = seq_len - 1
    acts = {p: [] for p in POINTS}
    msp = []
    idx = {p: _hs_index(p) for p in POINTS}
    for b in _batches(ids, bs):
        out = model(b, output_hidden_states=True)
        hs = out.hidden_states
        for p in POINTS:
            acts[p].append(hs[idx[p]][:, pos].float().cpu())
        msp.append((1.0 - torch.softmax(out.logits[:, pos], -1).max(-1).values).cpu())
    return ({p: torch.cat(acts[p]).numpy() for p in POINTS},
            torch.cat(msp).numpy())


@torch.no_grad()
def perturbation_score(ids, point, seq_len, n_dirs=16, eps=6.0, bs=8):
    """Mean next-token KL(clean||perturbed) over n_dirs unit dirs at `point`, last token. Higher=>OOD."""
    model, _ = load_model()
    pos = seq_len - 1
    hdim = model.config.n_embd
    g = torch.Generator(device=DEVICE).manual_seed(1234)
    inj = {"delta": None}

    def hook(mod, i, o):
        if inj["delta"] is None:
            return o
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[:, pos] = h[:, pos] + inj["delta"]
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    handle = _module(model, point).register_forward_hook(hook)
    scores = []
    try:
        for b in _batches(ids, bs):
            inj["delta"] = None
            clean = torch.log_softmax(model(b).logits[:, pos], -1)          # [B,V]
            p_clean = clean.exp()
            B = b.shape[0]
            kl = torch.zeros(B, device=DEVICE)
            for _ in range(n_dirs):
                d = torch.randn(B, hdim, generator=g, device=DEVICE)
                d = d / d.norm(dim=-1, keepdim=True)
                inj["delta"] = eps * d
                pert = torch.log_softmax(model(b).logits[:, pos], -1)
                kl += (p_clean * (clean - pert)).sum(-1)
                inj["delta"] = None
            scores.append((kl / n_dirs).cpu())
    finally:
        handle.remove()
    return torch.cat(scores).numpy()


def _grad_at_point(model, b, point, seq_len, build_loss, k=1):
    """Run forward with `point` activation as a leaf; backprop build_loss(logp) k times with fresh
    random output dirs; return summed per-sample ||grad||^2 over the k draws. [B] tensor."""
    pos = seq_len - 1
    cap = {}

    def hook(mod, i, o):
        h = (o[0] if isinstance(o, tuple) else o).detach().requires_grad_(True)
        cap["h"] = h
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    handle = _module(model, point).register_forward_hook(hook)
    try:
        acc = torch.zeros(b.shape[0], device=DEVICE)
        for _ in range(k):
            model.zero_grad(set_to_none=True)
            if cap.get("h") is not None and cap["h"].grad is not None:
                cap["h"].grad = None
            logits = model(b).logits[:, pos]
            logp = torch.log_softmax(logits, -1)
            loss = build_loss(logp, logits)           # scalar (summed over batch)
            loss.backward()
            g = cap["h"].grad[:, pos]                  # [B,d] per-sample grad
            acc = acc + (g * g).sum(-1)
    finally:
        handle.remove()
    return acc


def jacfrob_score(ids, point, seq_len, k=4, bs=8):
    """GENUINE plateau metric: Hutchinson estimate of ||d logp/d h||_F at last token. Higher=>OOD.

    For Gaussian v in R^V, E_v ||J^T v||^2 = ||J||_F^2. We average k draws and take sqrt."""
    model, _ = load_model()
    g = torch.Generator(device=DEVICE).manual_seed(7)
    out = []
    for b in _batches(ids, bs):
        B = b.shape[0]
        def make_loss(logp, logits, _B=B):
            v = torch.randn(_B, logp.shape[-1], generator=g, device=DEVICE)
            return (logp * v).sum()
        acc = _grad_at_point(model, b, point, seq_len, make_loss, k=k)
        out.append(torch.sqrt(acc / k).detach().cpu())
    return torch.cat(out).numpy()


def selfnll_grad_score(ids, point, seq_len, bs=8):
    """Iter-1's metric, RENAMED honestly: ||d(-logp[argmax])/d h|| (MSP-adjacent). Higher=>OOD."""
    model, _ = load_model()
    out = []
    for b in _batches(ids, bs):
        def make_loss(logp, logits):
            tgt = logits.argmax(-1)
            return (-logp.gather(-1, tgt[:, None])).sum()
        acc = _grad_at_point(model, b, point, seq_len, make_loss, k=1)
        out.append(torch.sqrt(acc).detach().cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------- baselines
def naive_mahalanobis(fit, test, eps=1e-3):
    mu = fit.mean(0); X = fit - mu
    cov = (X.T @ X) / X.shape[0] + eps * np.eye(X.shape[1])
    P = np.linalg.inv(cov); D = test - mu
    return np.einsum("nd,dk,nk->n", D, P, D)


def cup_rmd(fit_np, test_np, rcond=1e-5):
    """cupbearer relative-Mahalanobis distance (helpers.mahalanobis with inv_diag_covariance)."""
    fit = torch.from_numpy(fit_np).double(); test = torch.from_numpy(test_np).double()
    mean = fit.mean(0)
    X = fit - mean
    cov = X.T @ X / (fit.shape[0] - 1)
    inv_cov = torch.linalg.pinv(cov, rcond=rcond)
    inv_diag = 1.0 / (cov.diagonal() + 1e-6)
    return cup.mahalanobis(test, mean, inv_cov, inv_diag).numpy()


def cup_que(fit_np, test_np, rcond=1e-5, alpha=4.0):
    """cupbearer Quantum-Entropy (SPECTRE-like): whiten by trusted(ID) cov, score with untrusted
    (the set being screened) cov. Transductive: each scored set uses its own untrusted covariance."""
    fit = torch.from_numpy(fit_np).double(); test = torch.from_numpy(test_np).double()
    mean = fit.mean(0); Xf = fit - mean
    cov_t = Xf.T @ Xf / (fit.shape[0] - 1)
    eigs = torch.linalg.eigh(cov_t)
    vals_rsqrt = eigs.eigenvalues.clamp_min(0).rsqrt()
    vals_rsqrt[eigs.eigenvalues < rcond * eigs.eigenvalues.max()] = 0
    W = eigs.eigenvectors * vals_rsqrt.unsqueeze(0)           # whitening matrix
    whitened = (test - mean) @ W
    Xu = whitened - whitened.mean(0)
    cov_u = Xu.T @ Xu / (whitened.shape[0] - 1)
    norm = torch.linalg.eigvalsh(cov_u).max()
    return cup.quantum_entropy(whitened, cov_u, norm, alpha=alpha).numpy()


if __name__ == "__main__":
    t0 = time.time()
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    NDIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    KHUT = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    NFIT = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    EPS = 6.0
    HERE = os.path.dirname(__file__)
    load_model()
    print(f"DEVICE={DEVICE} N={N} SEQ={SEQ} NDIRS={NDIRS} KHUT={KHUT} NFIT={NFIT}", flush=True)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    texts = [l.rstrip("\n") for l in open(os.path.join(HERE, "..", "data", "fineweb_sample.txt"))]
    all_ids = ps.encode_fixed(texts, seq_len=SEQ)
    print(f"encoded {all_ids.shape[0]} ID seqs at seq_len={SEQ}", flush=True)
    # CANONICAL SPLIT (Codex 2026-06-22 High finding): match extract_acts.py EXACTLY —
    # randperm(seed=7), fit=perm[:NFIT], test=perm[NFIT:NFIT+N] — so the plateau / standard-baseline
    # AUROCs computed here are on the SAME ID split as the real-cupbearer acts in results/acts/
    # (which extract_acts.py produced). This makes auroc_table.csv and auroc_cupbearer.csv strictly
    # apples-to-apples. Save the split indices to results/split/ for provenance/reproducibility.
    g_split = torch.Generator().manual_seed(7)
    perm = torch.randperm(all_ids.shape[0], generator=g_split)
    fit_idx = perm[:NFIT]
    test_idx = perm[NFIT:NFIT + N]
    id_fit = all_ids[fit_idx]                          # large fit set => well-conditioned covariance
    id_test = all_ids[test_idx]
    os.makedirs(os.path.join(HERE, "..", "results", "split"), exist_ok=True)
    np.savez(os.path.join(HERE, "..", "results", "split", "canonical_split.npz"),
             perm=perm.numpy(), fit_idx=fit_idx.numpy(), test_idx=test_idx.numpy(),
             seq_len=SEQ, nfit=NFIT, n=N, seed=7)
    OOD = {"random": ps.make_ood(id_test, "random", seed=0),
           "shuffled": ps.make_ood(id_test, "shuffled", seed=0),
           "code": ps.make_ood(id_test, "code", seed=0)}
    print(f"id_test {id_test.shape} id_fit {id_fit.shape} | ood "
          + " ".join(f"{k}{tuple(v.shape)}" for k, v in OOD.items()), flush=True)

    def score_all(ids, tag):
        a, msp = collect_all(ids, SEQ)
        d = {"acts": a, "baseline-MSP": {None: msp},
             "baseline-L2norm": {p: np.linalg.norm(a[p], axis=1) for p in POINTS},
             "plateau-jacFrob": {p: jacfrob_score(ids, p, SEQ, k=KHUT) for p in POINTS},
             "selfNLL-grad": {p: selfnll_grad_score(ids, p, SEQ) for p in POINTS},
             "plateau-perturbation": {p: perturbation_score(ids, p, SEQ, n_dirs=NDIRS, eps=EPS) for p in POINTS}}
        print(f"  scored {tag} {time.time()-t0:.0f}s", flush=True)
        return d

    fit_acts, _ = collect_all(id_fit, SEQ)
    print(f"fit acts collected {time.time()-t0:.0f}s", flush=True)
    S_id = score_all(id_test, "id_test")
    S_ood = {k: score_all(v, k) for k, v in OOD.items()}

    POINTED = ["plateau-perturbation", "plateau-jacFrob", "selfNLL-grad", "baseline-L2norm"]
    rows = []
    dump = {}

    def addrow(ood, method, point, sid, sood):
        a = ps.auroc(np.concatenate([sid, sood]),
                     np.concatenate([np.zeros(len(sid)), np.ones(len(sood))]))
        rows.append(["fineweb-vs", ood, method, point, round(float(a), 4)])

    for k in OOD:
        addrow(k, "baseline-MSP", "n/a", S_id["baseline-MSP"][None], S_ood[k]["baseline-MSP"][None])
        for p in POINTS:
            # naive (iter-1) Mahalanobis + cupbearer RMD + cupbearer QUE
            addrow(k, "baseline-mahalanobis", p,
                   naive_mahalanobis(fit_acts[p], S_id["acts"][p]),
                   naive_mahalanobis(fit_acts[p], S_ood[k]["acts"][p]))
            addrow(k, "cup-RMD", p,
                   cup_rmd(fit_acts[p], S_id["acts"][p]),
                   cup_rmd(fit_acts[p], S_ood[k]["acts"][p]))
            addrow(k, "cup-QUE", p,
                   cup_que(fit_acts[p], S_id["acts"][p]),
                   cup_que(fit_acts[p], S_ood[k]["acts"][p]))
            for m in POINTED:
                addrow(k, m, p, S_id[m][p], S_ood[k][m][p])

    # dump raw scores for plotting
    def add_dump(setname, S):
        for m in POINTED:
            for p in POINTS:
                dump[f"{setname}|{m}|{p}"] = S[m][p]
        dump[f"{setname}|baseline-MSP|n/a"] = S["baseline-MSP"][None]
    add_dump("id", S_id)
    for k in OOD:
        add_dump(k, S_ood[k])
    for p in POINTS:
        dump[f"id|cup-RMD|{p}"] = cup_rmd(fit_acts[p], S_id["acts"][p])
        dump[f"id|cup-QUE|{p}"] = cup_que(fit_acts[p], S_id["acts"][p])
        dump[f"id|baseline-mahalanobis|{p}"] = naive_mahalanobis(fit_acts[p], S_id["acts"][p])
        for k in OOD:
            dump[f"{k}|cup-RMD|{p}"] = cup_rmd(fit_acts[p], S_ood[k]["acts"][p])
            dump[f"{k}|cup-QUE|{p}"] = cup_que(fit_acts[p], S_ood[k]["acts"][p])
            dump[f"{k}|baseline-mahalanobis|{p}"] = naive_mahalanobis(fit_acts[p], S_ood[k]["acts"][p])

    import csv
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    csv_path = os.path.join(HERE, "..", "results", "auroc_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "ood_set", "method", "measurement_point", "auroc"])
        w.writerows(rows)
    np.savez(os.path.join(HERE, "..", "results", "scores_full.npz"), **dump)
    print(f"\nwrote {csv_path} ({len(rows)} rows) in {time.time()-t0:.0f}s", flush=True)
    for r in sorted(rows, key=lambda x: (x[1], -x[4])):
        print(r)
