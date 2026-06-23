"""Direction #9 — Plateau-ness as an OOD detector.

Measurement-point-agnostic plateau / robustness scores for GPT-2 small, plus baselines.

A "measurement point" is WHERE in the network we read/perturb the activation:
  - "input"   : the post-embedding residual stream (transformer.drop output == hidden_states[0])
  - "residB"  : the residual stream right after transformer block index B (hidden_states[B+1])
We measure at the LAST token position of a fixed-length sequence (no padding).

Scores (all oriented so HIGHER == more OOD):
  - perturbation : mean next-token KL after adding eps*unit-direction at the point (sharper = OOD)
  - jacobian     : L2 norm of d(next-token NLL)/d(activation at the point)   (steeper = OOD)
  - l2norm       : L2 norm of the activation at the point                    (baseline)
  - mahalanobis  : Mahalanobis distance to a Gaussian fit on ID activations  (baseline)
  - msp          : 1 - max softmax prob of the next-token distribution       (baseline; needs no point)

Everything runs on CPU (the cluster V100 has no CUDA kernels for this torch build).
"""
import os, sys
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

torch.set_num_threads(2)
torch.manual_seed(0)
DEVICE = "cpu"

_MODEL = None
_TOK = None


def load_model():
    global _MODEL, _TOK
    if _MODEL is None:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _TOK = GPT2TokenizerFast.from_pretrained("gpt2")
        _MODEL = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
        for p in _MODEL.parameters():
            p.requires_grad_(False)
    return _MODEL, _TOK


# ---------------------------------------------------------------- data helpers
def encode_fixed(texts, seq_len=64):
    """Encode each text to exactly seq_len tokens (drop texts that are too short)."""
    _, tok = load_model()
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"][0]
        if ids.shape[0] == seq_len:
            out.append(ids)
    return torch.stack(out) if out else torch.empty(0, seq_len, dtype=torch.long)


def load_code_seqs(n, seq_len, seed=0):
    """Real domain-shift OOD: Python *source code* from offline site-packages, encoded into
    contiguous seq_len-token chunks. A genuine domain shift (not random/shuffled) that GPT-2
    has seen a little of, so MSP need not trivially catch it. Returns [n, seq_len] long tensor."""
    import glob
    _, tok = load_model()
    roots = ["/opt/conda/lib/python3*/site-packages/numpy",
             "/opt/conda/lib/python3*/site-packages/torch/nn",
             "/opt/conda/lib/python3*/site-packages/torch/optim"]
    files = []
    for r in roots:
        for d in glob.glob(r):
            files += glob.glob(os.path.join(d, "**", "*.py"), recursive=True)
    files = sorted(files)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(files), generator=g).tolist()
    seqs = []
    for fi in perm:
        try:
            txt = open(files[fi], encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if len(txt) < 200:
            continue
        ids = tok(txt, return_tensors="pt")["input_ids"][0]
        # carve non-overlapping seq_len chunks, take at most a few per file for diversity
        nchunks = min(ids.shape[0] // seq_len, 4)
        for c in range(nchunks):
            seqs.append(ids[c * seq_len:(c + 1) * seq_len])
            if len(seqs) >= n:
                return torch.stack(seqs)
    return torch.stack(seqs) if seqs else torch.empty(0, seq_len, dtype=torch.long)


def make_ood(id_ids, kind, seed=0):
    """Build an OOD set from in-distribution token id tensors [N, L]."""
    g = torch.Generator().manual_seed(seed)
    _, tok = load_model()
    V = tok.vocab_size
    if kind == "random":
        return torch.randint(0, V, id_ids.shape, generator=g)
    if kind == "shuffled":
        out = id_ids.clone()
        for i in range(out.shape[0]):
            perm = torch.randperm(out.shape[1], generator=g)
            out[i] = out[i][perm]
        return out
    if kind == "code":
        return load_code_seqs(id_ids.shape[0], id_ids.shape[1], seed=seed)
    raise ValueError(kind)


# ---------------------------------------------------------------- measurement-point plumbing
def _point_module(model, point):
    if point == "input":
        return model.transformer.drop
    if point.startswith("resid"):
        b = int(point[5:])
        return model.transformer.h[b]
    raise ValueError(point)


class _Injector:
    """Forward hook that adds `delta` [batch,d] at position `pos` of the module output."""
    def __init__(self):
        self.delta = None
        self.pos = -1

    def __call__(self, module, inp, out):
        if self.delta is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone()
            h[:, self.pos, :] = h[:, self.pos, :] + self.delta
            return (h,) + tuple(out[1:])
        h = out.clone()
        h[:, self.pos, :] = h[:, self.pos, :] + self.delta
        return h


# ---------------------------------------------------------------- scores
@torch.no_grad()
def perturbation_score(ids, point, n_dirs=16, eps=6.0, seq_len=64, verbose=False):
    """Mean next-token KL(clean||perturbed) over n_dirs random unit dirs at `point`, last token.

    `eps` is the perturbation magnitude in activation space.  Higher KL => sharper => more OOD.
    """
    model, _ = load_model()
    inj = _Injector()
    handle = _point_module(model, point).register_forward_hook(inj)
    pos = seq_len - 1
    g = torch.Generator().manual_seed(1234)
    scores = np.zeros(ids.shape[0], dtype=np.float64)
    try:
        for i in range(ids.shape[0]):
            seq = ids[i : i + 1]
            # clean last-token distribution
            inj.delta = None
            clean_logits = model(seq).logits[0, pos]
            clean_logp = torch.log_softmax(clean_logits, -1)
            # batch of n_dirs perturbed copies
            d = clean_logits.shape[-1]
            hdim = model.config.n_embd
            dirs = torch.randn(n_dirs, hdim, generator=g)
            dirs = dirs / dirs.norm(dim=-1, keepdim=True)
            inj.delta = eps * dirs
            inj.pos = pos
            batch = seq.expand(n_dirs, -1)
            plogits = model(batch).logits[:, pos]
            plogp = torch.log_softmax(plogits, -1)
            p_clean = clean_logp.exp().unsqueeze(0)  # [1,V]
            kl = (p_clean * (clean_logp.unsqueeze(0) - plogp)).sum(-1)  # KL(clean||pert)
            scores[i] = kl.mean().item()
            if verbose and i % 25 == 0:
                print(f"  perturb {point} {i}/{ids.shape[0]}", flush=True)
    finally:
        handle.remove()
        inj.delta = None
    return scores


def jacobian_score(ids, point, seq_len=64, verbose=False):
    """L2 norm of d(next-token NLL)/d(activation at `point`) at the last token. Higher => OOD."""
    model, _ = load_model()
    pos = seq_len - 1
    captured = {}

    def grab(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.retain_grad()
        captured["h"] = h
        return out

    handle = _point_module(model, point).register_forward_hook(grab)
    scores = np.zeros(ids.shape[0], dtype=np.float64)
    try:
        for i in range(ids.shape[0]):
            seq = ids[i : i + 1]
            model.zero_grad(set_to_none=True)
            logits = model(seq).logits
            logp = torch.log_softmax(logits[0, pos], -1)
            # NLL of the model's own argmax next token (label-free, deterministic)
            tgt = logits[0, pos].argmax()
            loss = -logp[tgt]
            loss.backward()
            grad = captured["h"].grad[0, pos]
            scores[i] = grad.norm().item()
            if verbose and i % 25 == 0:
                print(f"  jac {point} {i}/{ids.shape[0]}", flush=True)
    finally:
        handle.remove()
    return scores


@torch.no_grad()
def collect_activations(ids, point, seq_len=64):
    """Return [N, d] last-token activations at `point` (for l2norm / mahalanobis)."""
    model, _ = load_model()
    pos = seq_len - 1
    store = {}

    def grab(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        store["h"] = h[:, pos, :].detach().clone()

    handle = _point_module(model, point).register_forward_hook(grab)
    acts = []
    try:
        for i in range(ids.shape[0]):
            model(ids[i : i + 1])
            acts.append(store["h"][0].numpy())
    finally:
        handle.remove()
    return np.stack(acts)


def _hs_index(point):
    """hidden_states index for a measurement point (consistent with the perturbation hooks)."""
    if point == "input":
        return 0
    if point.startswith("resid"):
        return int(point[5:]) + 1  # residB == output of block B == hidden_states[B+1]
    raise ValueError(point)


@torch.no_grad()
def collect_all(ids, points, seq_len=64):
    """ONE forward per seq: last-token activations at every point + MSP. Cheap shared pass.

    Returns (acts: {point: [N,d]}, msp: [N]).
    """
    model, _ = load_model()
    pos = seq_len - 1
    acts = {p: [] for p in points}
    msp = np.zeros(ids.shape[0])
    idx = {p: _hs_index(p) for p in points}
    for i in range(ids.shape[0]):
        out = model(ids[i : i + 1], output_hidden_states=True)
        hs = out.hidden_states
        for p in points:
            acts[p].append(hs[idx[p]][0, pos].numpy())
        msp[i] = 1.0 - torch.softmax(out.logits[0, pos], -1).max().item()
    return {p: np.stack(v) for p, v in acts.items()}, msp


def jacobian_all(ids, points, seq_len=64, verbose=False):
    """L2 grad-norm of next-token NLL w.r.t. activation at each point. One fwd+bwd PER point
    (detaching to make the activation a leaf cuts the path to earlier points, so points can't
    share a backward). Returns {point: [N]}. Higher => steeper => more OOD."""
    model, _ = load_model()
    pos = seq_len - 1
    out_scores = {p: np.zeros(ids.shape[0]) for p in points}
    for p in points:
        captured = {}

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.detach().requires_grad_(True)
            captured["h"] = h
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        handle = _point_module(model, p).register_forward_hook(hook)
        try:
            for i in range(ids.shape[0]):
                logits = model(ids[i : i + 1]).logits[0, pos]
                logp = torch.log_softmax(logits, -1)
                tgt = logits.argmax()
                (-logp[tgt]).backward()
                out_scores[p][i] = captured["h"].grad[0, pos].norm().item()
                if verbose and i % 25 == 0:
                    print(f"  jac {p} {i}/{ids.shape[0]}", flush=True)
        finally:
            handle.remove()
    return out_scores


@torch.no_grad()
def msp_score(ids, seq_len=64):
    """1 - max softmax prob of the next-token distribution at last token. Higher => OOD."""
    model, _ = load_model()
    pos = seq_len - 1
    out = np.zeros(ids.shape[0])
    for i in range(ids.shape[0]):
        logits = model(ids[i : i + 1]).logits[0, pos]
        out[i] = 1.0 - torch.softmax(logits, -1).max().item()
    return out


def mahalanobis(fit_acts, test_acts, eps=1e-3):
    mu = fit_acts.mean(0)
    X = fit_acts - mu
    cov = (X.T @ X) / X.shape[0] + eps * np.eye(X.shape[1])
    P = np.linalg.inv(cov)
    D = test_acts - mu
    return np.einsum("nd,dk,nk->n", D, P, D)


# ---------------------------------------------------------------- metric
def auroc(scores, labels):
    """AUROC with label==1 the positive (OOD) class. Pure numpy."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(scores)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = (csum - (counts - 1) / 2.0)
    ranks = avg[inv]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


if __name__ == "__main__":
    import time
    t0 = time.time()
    texts = [l.rstrip("\n") for l in open(os.path.join(os.path.dirname(__file__), "..", "data", "fineweb_sample.txt"))]
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    id_ids = encode_fixed(texts[:600], seq_len=64)[:N]
    ood_ids = make_ood(id_ids, "random", seed=0)
    print(f"ID {id_ids.shape} OOD {ood_ids.shape}")
    point = "resid6"
    s_id = perturbation_score(id_ids, point, n_dirs=16, eps=6.0, verbose=True)
    s_ood = perturbation_score(ood_ids, point, n_dirs=16, eps=6.0, verbose=True)
    scores = np.concatenate([s_id, s_ood])
    labels = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))])
    print(f"perturbation@{point} AUROC(FineWeb vs random) = {auroc(scores, labels):.4f}")
    print(f"  ID mean KL {s_id.mean():.4f}  OOD mean KL {s_ood.mean():.4f}")
    print(f"elapsed {time.time()-t0:.1f}s")
