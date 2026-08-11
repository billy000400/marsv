"""Build top-heavy and tail-heavy embedding edits instead of drawing them, and re-measure width.

The mode split showed that a large random embedding edit disturbs a token's output in a tail-weighted
way (top-32 successors hold 0.71 of the mass but absorb 0.39 of the divergence), and that the most
top-heavy and most tail-heavy of 24 random draws both destroy the token ordering. That causal half was
underpowered: random draws span only S = 0.36-0.56 in top-mass share, so the two arms were barely
different.

Here the two directions are CONSTRUCTED. For small displacements the per-successor JSD contribution is
    term_k ~ p_k * (dlogit_k - <dlogit>_p)^2,
so within the span of m probe directions the top-mass share S is a Rayleigh quotient c'Ac / c'Bc in the
mixing coefficients c, with A the top-K block and B the whole vocabulary. Solving the 24x24 generalised
eigenproblem gives the S-maximising and S-minimising combinations directly. Both are then rescaled to
the same 0.4 bits of output movement and anchor width is re-measured.

A top-heavy edit that keeps the token ordering where a tail-heavy one destroys it places the width
trait in the tail of the next-token distribution; both destroying it says the trait belongs to the
token's whole output map. Writes results/mode_construct.json.
"""
import json
import os

import numpy as np
import scipy.linalg as sla
import torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR
from basin_probe import MODEL, REVISION, Patcher
from common import D18, RESULTS
from embed_intervene import measure
from embed_intervene2 import out_logp
from mode_split import NORM, TOPK, TARGET, decompose

M_PROBE = 24        # random directions spanning the subspace the construction searches
PROBE_NORM = 0.6    # displacement at which the probe responses are read (small enough to stay linear)

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


@torch.inference_mode()
def responses(model, tok, W, i, base_row, base_logp, U):
    """Whitened, p-weighted centred logit responses g[j] to each probe direction, per frame."""
    G = []
    for j in range(len(U)):
        W.data[i] = base_row + torch.tensor(PROBE_NORM * U[j], dtype=W.dtype, device=W.device)
        G.append(out_logp(model, tok, i) - base_logp)
    W.data[i] = base_row
    G = torch.stack(G)                                    # (m, frames, V)
    p = base_logp.exp()                                   # (frames, V)
    G = G - (G * p).sum(-1, keepdim=True)                 # centre under p
    return G * p.sqrt()


def construct(G, top_idx):
    """Coefficients that maximise and minimise the top-K share of the induced divergence."""
    m = G.shape[0]
    mask = torch.zeros_like(G[0], dtype=torch.bool).scatter_(-1, top_idx, True)
    F = G.reshape(m, -1).double()
    T = (G * mask).reshape(m, -1).double()
    B = (F @ F.T).cpu().numpy()
    A = (T @ T.T).cpu().numpy()
    B += np.eye(m) * 1e-8 * np.trace(B) / m
    vals, vecs = sla.eigh(A, B)
    return vecs[:, -1], vecs[:, 0], float(vals[-1]), float(vals[0])


@torch.inference_mode()
def calibrate(model, tok, W, i, base_row, base_logp, d, top_idx):
    """Norm along `d` that moves the token's output by TARGET bits (log-log interpolation)."""
    cs, bits = [], []
    for c in NORM * 1.45 ** np.arange(-5, 5):
        W.data[i] = base_row + torch.tensor(c * d, dtype=W.dtype, device=W.device)
        b = decompose(base_logp, out_logp(model, tok, i), top_idx)[0]
        cs.append(float(c))
        bits.append(max(b, 1e-9))
        if b > 1.3 * TARGET:
            break
    W.data[i] = base_row
    o = np.argsort(bits)
    return float(np.exp(np.interp(np.log(TARGET), np.log(bits)[o], np.log(cs)[o])))


def main():
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    used = {p["a"] for p in man} | {p["b_tok"] for p in man}
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    W = model.gpt_neox.embed_in.weight
    dim = W.shape[1]

    prev = json.load(open(f"{RESULTS}/mode_split.json"))["tokens"]
    patcher = Patcher(model)
    rows = {}
    for k, s in enumerate(sorted(prev, key=lambda x: prev[x]["base_w"])):
        i = prev[s]["token_id"]
        base_row = W.data[i].clone()
        base_logp = out_logp(model, tok, i)
        top_idx = base_logp.topk(TOPK, dim=-1).indices
        base_w, _ = measure(model, patcher, tok, anchors, i)

        rng = np.random.default_rng(400 + k)
        U = rng.normal(size=(M_PROBE, dim))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        G = responses(model, tok, W, i, base_row, base_logp, U)
        c_top, c_tail, s_hi, s_lo = construct(G, top_idx)

        r = dict(token_id=i, base_w=base_w, base_w_split=prev[s]["base_w"],
                 predicted_share=[s_hi, s_lo], edits={})
        for tag, c in (("top_heavy", c_top), ("tail_heavy", c_tail)):
            d = c @ U
            d /= np.linalg.norm(d)
            cn = calibrate(model, tok, W, i, base_row, base_logp, d, top_idx)
            W.data[i] = base_row + torch.tensor(cn * d, dtype=W.dtype, device=W.device)
            got, sh = decompose(base_logp, out_logp(model, tok, i), top_idx)
            w_new, _ = measure(model, patcher, tok, anchors, i)
            W.data[i] = base_row
            r["edits"][tag] = dict(step_norm=cn, bits=got, top_share=sh, w=w_new, dw=w_new - base_w)
        rows[s] = r
        e = r["edits"]
        print(f"[{k + 1}/{len(prev)}] {s!r} w {base_w:.3f} | predicted S {s_hi:.2f}/{s_lo:.2f} | "
              + " ".join(f"{t} {e[t]['bits']:.3f}b S {e[t]['top_share']:.3f} dw {e[t]['dw']:+.3f}"
                         for t in e), flush=True)
        json.dump(dict(partial=True, tokens=rows), open(f"{RESULTS}/mode_construct.json", "w"), indent=1)
    patcher.close()

    ks = list(rows)
    base = np.array([rows[s]["base_w"] for s in ks])
    summ = dict(edits={})
    for tag in ("top_heavy", "tail_heavy"):
        w = np.array([rows[s]["edits"][tag]["w"] for s in ks])
        summ["edits"][tag] = dict(
            bits=float(np.median([rows[s]["edits"][tag]["bits"] for s in ks])),
            top_share=float(np.mean([rows[s]["edits"][tag]["top_share"] for s in ks])),
            top_share_sd=float(np.std([rows[s]["edits"][tag]["top_share"] for s in ks], ddof=1)),
            step_norm=float(np.median([rows[s]["edits"][tag]["step_norm"] for s in ks])),
            dw=float(np.mean([rows[s]["edits"][tag]["dw"] for s in ks])),
            w=float(w.mean()), w_sd=float(w.std(ddof=1)),
            rho_base=[float(x) for x in spearmanr(base, w)[:2]])
    dt = np.array([rows[s]["edits"]["top_heavy"]["dw"] for s in ks])
    db = np.array([rows[s]["edits"]["tail_heavy"]["dw"] for s in ks])
    ds = np.array([rows[s]["edits"]["top_heavy"]["top_share"] -
                   rows[s]["edits"]["tail_heavy"]["top_share"] for s in ks])
    summ["paired_p_dw"] = float(wilcoxon(dt, db)[1])
    summ["paired_p_share"] = float(wilcoxon(ds)[1])
    res = dict(n_tokens=len(rows), m_probe=M_PROBE, probe_norm=PROBE_NORM, topk=TOPK,
               target_bits=TARGET, base=dict(w=float(base.mean()), w_sd=float(base.std(ddof=1))),
               tokens=rows, summary=summ)
    json.dump(res, open(os.path.join(RESULTS, "mode_construct.json"), "w"), indent=1)

    print(f"\nbefore any edit: w {res['base']['w']:.3f} +- {res['base']['w_sd']:.3f}")
    for tag in ("top_heavy", "tail_heavy"):
        o = summ["edits"][tag]
        print(f"{tag:11s}: {o['bits']:.3f} bits, S = {o['top_share']:.3f}+-{o['top_share_sd']:.3f}, "
              f"step {o['step_norm']:.2f}, w {o['w']:.3f}+-{o['w_sd']:.3f} (dw {o['dw']:+.3f}), "
              f"rho(before, after) = {o['rho_base'][0]:+.2f} (p={o['rho_base'][1]:.2g})")
    print(f"paired S difference p = {summ['paired_p_share']:.4f}; "
          f"paired dw difference p = {summ['paired_p_dw']:.4f}")


if __name__ == "__main__":
    main()
