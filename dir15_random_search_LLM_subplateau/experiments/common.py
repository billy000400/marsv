"""Shared machinery for the random-pair sub-plateau screen (direction 15).

GPT-2 Large, final-position `resid_post` patching at an early block, `slerp_rescale`
interpolation between two natural contexts' activations, and the frozen A|C|B detector.

Everything here (window length, alpha grid, layers, detector thresholds, score) is frozen
BEFORE any interpolation curve was inspected; see PLAN.md and JOURNAL.md.
"""
import os

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
PLOTS = os.path.join(HERE, "..", "plots")

MODEL = "gpt2-large"
REVISION = "main"
CTX_LEN = 32                 # tokens per context window
LAYERS = (0, 2, 4, 6)        # preregistered early blocks (resid_post)
N_ALPHA = 50                 # interpolation grid points, endpoints included
ALPHAS = np.linspace(0.0, 1.0, N_ALPHA)
SEED = 0

# frozen detector thresholds
MIN_RUN = 3                  # C must be top-1 for >= 3 consecutive alphas
JSD_FLOOR = 0.005            # bits; both C-run boundaries must be real distribution changes
EPS_THETA = 1e-4


# ------------------------------------------------------------------ interpolation (dir14 copy)
def slerp_rescale(hA, hB, ts):
    """Slerp of unit directions, linear interpolation of L2 norms. hA,hB:[C], ts:[K] -> [K,C]."""
    nA, nB = torch.linalg.norm(hA), torch.linalg.norm(hB)
    uA, uB = hA / nA, hB / nB
    cos = torch.clamp(torch.dot(uA, uB), -1.0, 1.0)
    theta = torch.arccos(cos)
    ts = ts[:, None]
    if theta < EPS_THETA:
        u = (1 - ts) * uA[None] + ts * uB[None]
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True)
    else:
        s = torch.sin(theta)
        u = (torch.sin((1 - ts) * theta) / s) * uA[None] + (torch.sin(ts * theta) / s) * uB[None]
    return ((1 - ts) * nA + ts * nB) * u


def lerp(hA, hB, ts):
    """Plain linear interpolation (secondary geometry control)."""
    ts = ts[:, None]
    return (1 - ts) * hA[None] + ts * hB[None]


# --------------------------------------------------------------------------------- the model
class Runner:
    """GPT-2 Large with a final-position resid_post patch hook and resid_post recorders."""

    def __init__(self, device="cuda", mem_frac=0.225, threads=2):
        torch.set_num_threads(threads)
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(mem_frac)
        self.device = device
        self.tok = GPT2TokenizerFast.from_pretrained(MODEL, revision=REVISION)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL, revision=REVISION)
        self.model = self.model.to(device=device, dtype=torch.float32).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_layer = self.model.config.n_layer
        self._rec, self._patch = {}, None
        for l, blk in enumerate(self.model.transformer.h):
            blk.register_forward_hook(self._mk_block(l))

    def _mk_block(self, l):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self._patch is not None and self._patch[0] == l:
                h = h.clone()
                h[:, -1, :] = self._patch[1]
                if l in self._rec_layers:
                    self._rec[l] = h[:, -1, :].detach()
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            if l in self._rec_layers:
                self._rec[l] = h[:, -1, :].detach()
            return None
        return fn

    _rec_layers = set(LAYERS)

    @torch.no_grad()
    def forward(self, ids, patch=None, rec_layers=None):
        """ids:[B,T] long. patch:(layer, H[B,C]) or None.
        Returns (dict layer->[B,C] final-position resid_post, final-position logits [B,V])."""
        self._rec_layers = set(LAYERS if rec_layers is None else rec_layers)
        self._rec, self._patch = {}, patch
        h = self.model.transformer(input_ids=ids.to(self.device)).last_hidden_state
        self._patch = None
        logits = self.model.lm_head(h[:, -1, :])
        rec, self._rec = self._rec, {}
        return rec, logits

    @torch.no_grad()
    def forward_embeds(self, ids, last_embed=None):
        """Unhooked forward from token embeddings; `last_embed` [B,C] replaces the final position's
        wte vector (positional embeddings are still added by the model). Returns logits [B,V]."""
        self._rec_layers, self._rec, self._patch = set(), {}, None
        e = self.model.transformer.wte(ids.to(self.device))
        if last_embed is not None:
            e = torch.cat([e[:, :-1, :], last_embed[:, None, :].to(e.dtype)], dim=1)
        h = self.model.transformer(inputs_embeds=e).last_hidden_state
        return self.model.lm_head(h[:, -1, :])

    @torch.no_grad()
    def generate(self, ids, patch=None, n_new=20, temperature=0.0, seed=0):
        """Continue `ids` [1,T]. The patch (if any) is applied on the FIRST step only; the KV cache
        then carries the patched state forward, so later steps are ordinary decoding."""
        self._rec_layers, self._rec, self._patch = set(), {}, patch
        out = self.model.transformer(input_ids=ids.to(self.device), use_cache=True)
        self._patch = None
        past, h = out.past_key_values, out.last_hidden_state[:, -1:, :]
        g = torch.Generator(device=self.device).manual_seed(seed)
        toks = []
        for _ in range(n_new):
            logits = self.model.lm_head(h)[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                p = torch.softmax(logits / temperature, -1)
                nxt = torch.multinomial(p, 1, generator=g)
            toks.append(int(nxt))
            out = self.model.transformer(input_ids=nxt, past_key_values=past, use_cache=True)
            past, h = out.past_key_values, out.last_hidden_state
        return toks


# ----------------------------------------------------------------------------------- corpus
def build_windows(tok, n_windows, ctx_len=CTX_LEN):
    """Non-overlapping `ctx_len`-token windows from the wikitext-103 validation split."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    text, n_docs = [], 0
    for row in ds:
        t = row["text"]
        if t.strip():
            text.append(t)
            n_docs += 1
        if sum(len(x) for x in text) > 40 * ctx_len * n_windows:
            break
    ids = tok("".join(text))["input_ids"]
    n = min(n_windows, len(ids) // ctx_len)
    W = np.array(ids[: n * ctx_len], dtype=np.int64).reshape(n, ctx_len)
    return W, {"dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1",
               "split": "validation", "n_docs_used": n_docs, "n_tokens": int(n * ctx_len)}


# -------------------------------------------------------------------- path summaries + rule
def path_summary(probs):
    """probs:[K,V] torch tensor on GPU -> dict of frozen per-path summaries (numpy).

    Retains the top-1 trajectory, predictive entropy, adjacent-alpha Jensen-Shannon divergence
    (base 2, bits) and the full probability trajectory of every token that is top-1 somewhere.
    """
    top_p, top_i = probs.max(dim=-1)
    ent = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(-1)
    p, q = probs[:-1], probs[1:]
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log2(a.clamp_min(1e-12)) - torch.log2(b.clamp_min(1e-12)))).sum(-1)
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    toks = torch.unique(top_i)
    traj = probs[:, toks]                                  # [K, n_tok]
    return {"top1": top_i.cpu().numpy().astype(np.int64),
            "top1_p": top_p.float().cpu().numpy(),
            "entropy": ent.float().cpu().numpy(),
            "jsd": jsd.float().cpu().numpy(),
            "tok_ids": toks.cpu().numpy().astype(np.int64),
            "tok_probs": traj.float().cpu().numpy()}       # [K, n_tok]


def rle(seq):
    """Run-length encoding -> list of (value, start, end) inclusive index ranges."""
    out, s = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[s]:
            out.append((int(seq[s]), s, i - 1))
            s = i
    return out


def detect(summ, min_run=MIN_RUN, jsd_floor=JSD_FLOOR, alphas=ALPHAS):
    """Frozen A|C|B detector. Returns dict with `is_candidate` and every rule component."""
    top1, jsd = summ["top1"], summ["jsd"]
    A, B = int(top1[0]), int(top1[-1])
    runs = rle(top1)
    out = {"A": A, "B": B, "n_runs": len(runs), "eligible": A != B,
           "is_candidate": False, "clean": False, "C": -1,
           "width_C": np.nan, "margin_min": np.nan, "sep": np.nan, "score": np.nan,
           "k_in": -1, "k_out": -1, "jsd_in": np.nan, "jsd_out": np.nan,
           "run_len": 0, "n_transient": 0}
    if A == B:
        return out
    if runs[0][0] != A or runs[-1][0] != B:
        return out
    pidx = {int(t): j for j, t in enumerate(summ["tok_ids"])}
    pA, pB = summ["tok_probs"][:, pidx[A]], summ["tok_probs"][:, pidx[B]]
    best = None
    for (c, s, e) in runs[1:-1]:
        if c == A or c == B or (e - s + 1) < min_run:
            continue
        pC = summ["tok_probs"][:, pidx[c]]
        marg = (pC[s:e + 1] - np.maximum(pA[s:e + 1], pB[s:e + 1])).min()
        if marg <= 0:
            continue
        j_in, j_out = float(jsd[s - 1]), float(jsd[e])     # jsd[k] is between alpha k and k+1
        if j_in < jsd_floor or j_out < jsd_floor:
            continue
        # A-run before C-run before B-run in order (guaranteed by runs[0]/runs[-1] and position)
        width = (e - s + 1) / len(top1)
        sep = float(0.5 * (alphas[e] + alphas[e + 1]) - 0.5 * (alphas[s - 1] + alphas[s]))
        cand = {"C": int(c), "k_in": s, "k_out": e, "run_len": e - s + 1,
                "width_C": width, "margin_min": float(marg), "sep": sep,
                "score": float(marg) * sep, "jsd_in": j_in, "jsd_out": j_out}
        if best is None or cand["score"] > best["score"]:
            best = cand
    if best is None:
        return out
    out.update(best)
    out["is_candidate"] = True
    out["clean"] = len(runs) == 3
    out["n_transient"] = len(runs) - 3
    return out


def wilson(k, n, z=1.96):
    """Wilson binomial confidence interval."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ------------------------------------------------------------------------------- self-tests
def _self_test():
    hA, hB = torch.tensor([2.0, 0.0]), torch.tensor([0.0, 4.0])
    H = slerp_rescale(hA, hB, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(H[0], hA, atol=1e-6) and torch.allclose(H[2], hB, atol=1e-6)
    assert abs(float(torch.linalg.norm(H[1])) - 3.0) < 1e-5

    K, V = N_ALPHA, 8
    # synthetic A|C|B: token 1 -> token 5 -> token 2, C clearly dominant in the middle
    P = np.full((K, V), 0.01)
    P[:20, 1] = 0.9
    P[20:30, 5] = 0.9
    P[30:, 2] = 0.9
    P = P / P.sum(1, keepdims=True)
    s = path_summary(torch.tensor(P))
    r = detect(s)
    assert r["is_candidate"] and r["clean"] and r["C"] == 5 and r["run_len"] == 10, r
    # single A->B crossing must not be a candidate
    P2 = np.full((K, V), 0.01)
    P2[:25, 1] = 0.9
    P2[25:, 2] = 0.9
    P2 = P2 / P2.sum(1, keepdims=True)
    r2 = detect(path_summary(torch.tensor(P2)))
    assert not r2["is_candidate"] and r2["eligible"], r2
    # 2-point C run fails the persistence threshold
    P3 = P.copy()
    P3[20:28, 5] = 0.01
    P3[20:28, 1] = 0.9
    P3 = P3 / P3.sum(1, keepdims=True)
    r3 = detect(path_summary(torch.tensor(P3)))
    assert not r3["is_candidate"], r3
    assert rle([1, 1, 2, 2, 2, 3]) == [(1, 0, 1), (2, 2, 4), (3, 5, 5)]
    lo, hi = wilson(5, 100)
    assert lo < 0.05 < hi
    print("common._self_test OK")


if __name__ == "__main__":
    _self_test()
