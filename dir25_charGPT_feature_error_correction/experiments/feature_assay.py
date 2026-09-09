"""Shared machinery: load the char GPT, build feature directions, perturb one activation.

Intervention point (fixed by PLAN): the residual stream after block 0, at the last character
position of a 128-character context. The perturbed activation keeps the clean activation's length.
Readout: the next-character distribution at that same position.
"""
import os, sys, hashlib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "..", "dir13_plateau_on_grok_gpt", "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig
from collect_examples import load, line_spans, index_maps, collect, CTX

BLOCK = 0            # intervention block (dir13's interpolation block)
CKPT = os.path.join(ROOT, "results", "checkpoints_char", "ckpt_final.pt")


def load_model(device="cuda"):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    m = GPT(cfg)
    m.load_state_dict(ck["model"])
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck["stoi"], ck["step"]


def contexts(text, idxs, stoi):
    """[N,128] token ids for the windows ending at each index."""
    return torch.tensor([[stoi[c] for c in text[i - CTX + 1:i + 1]] for i in idxs], dtype=torch.long)


@torch.no_grad()
def resid_and_probs(model, ids, device, bs=32):
    """Post-block-BLOCK residual at the last position, and the clean next-char distribution."""
    H, P = [], []
    for k in range(0, len(ids), bs):
        res, lg = model.residuals_and_logits(ids[k:k + bs].to(device))
        H.append(res[BLOCK][:, -1, :].float().cpu())
        P.append(torch.softmax(lg[:, -1, :].float(), -1).cpu())
    return torch.cat(H), torch.cat(P)


@torch.no_grad()
def probs_from_resid(model, ids, h_new, device, bs=32):
    """Re-run from block BLOCK+1 with the last-position residual replaced by h_new [N,C]."""
    out = []
    for k in range(0, len(ids), bs):
        res, _ = model.residuals_and_logits(ids[k:k + bs].to(device))
        x = res[BLOCK].clone()
        x[:, -1, :] = h_new[k:k + bs].to(device).to(x.dtype)
        _, lg = model.forward_from_block(x, BLOCK + 1)
        out.append(torch.softmax(lg[:, -1, :].float(), -1).cpu())
    return torch.cat(out)


def perturb(h, v, alpha):
    """Move h toward unit direction v by relative size alpha, keeping ||h|| fixed.
    h: [N,C], v: [C] or [N,C], alpha: scalar. Returns [N,C]."""
    nrm = h.norm(dim=-1, keepdim=True)
    u = h / nrm
    w = u + alpha * (v if v.dim() == 2 else v[None, :])
    return nrm * w / w.norm(dim=-1, keepdim=True)


def jsd(P, Q, eps=1e-12):
    """Base-2 Jensen-Shannon divergence, rows of P against rows (or one row) of Q."""
    P, Q = np.asarray(P, dtype=np.float64), np.asarray(Q, dtype=np.float64)
    M = 0.5 * (P + Q)
    def kl(a, b):
        return np.sum(np.where(a > eps, a * np.log2(np.maximum(a, eps) / np.maximum(b, eps)), 0.0), -1)
    return np.clip(0.5 * kl(P, M) + 0.5 * kl(Q, M), 0.0, 1.0)


def char_mask(stoi, pred):
    m = np.zeros(len(stoi))
    for c, i in stoi.items():
        m[i] = float(pred(c))
    return m


def matched_pairs(text, pos, neg, n, rng):
    """n pairs sharing a final character, spread over as many distinct characters as possible."""
    from collections import defaultdict
    P, N = defaultdict(list), defaultdict(list)
    for i in pos:
        P[text[i]].append(i)
    for i in neg:
        N[text[i]].append(i)
    chars = sorted(set(P) & set(N), key=lambda c: -min(len(P[c]), len(N[c])))
    for c in chars:
        rng.shuffle(P[c]); rng.shuffle(N[c])
    pairs, r = [], 0
    while len(pairs) < n:
        added = False
        for c in chars:
            if r < min(len(P[c]), len(N[c])) and len(pairs) < n:
                pairs.append((P[c][r], N[c][r])); added = True
        if not added:
            break
        r += 1
    assert len(pairs) == n, f"only {len(pairs)} matched pairs available"
    return pairs


def feature_positions(split="train"):
    """Candidate positions for both features in one corpus split."""
    text = load()
    spans = line_spans(text)
    lid = index_maps(text, spans)
    n = int(0.9 * len(text))
    lo, hi = (0, n) if split == "train" else (n, len(text))
    return text, collect(text, lo, hi, spans, lid)


@torch.no_grad()
def sweep_from_base(model, base, H, device, bs=64):
    """Probabilities for many replacements of one context's last-position residual.
    base: [T,C] post-block-BLOCK residual of a single context; H: [K,C]. Returns [K,V] numpy."""
    out = []
    for k in range(0, len(H), bs):
        x = base[None].repeat(min(bs, len(H) - k), 1, 1).clone()
        x[:, -1, :] = H[k:k + bs].to(device).to(x.dtype)
        _, lg = model.forward_from_block(x, BLOCK + 1)
        out.append(torch.softmax(lg[:, -1, :].float(), -1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def base_resid(model, ids, device):
    """[T,C] post-block-BLOCK residual for a single context [1,T]."""
    res, lg = model.residuals_and_logits(ids.to(device))
    return res[BLOCK][0], torch.softmax(lg[0, -1].float(), -1).cpu().numpy()
