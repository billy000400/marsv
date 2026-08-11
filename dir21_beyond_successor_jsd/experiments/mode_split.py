"""Which part of a token's output behaviour carries the width trait: its top successors or its tail?

The displacement ladder showed that at a fixed embedding displacement a behaviourally LOUD edit
destroys the token ordering the screen depends on while a quiet one preserves it. That localises the
trait in output behaviour but not in which part of the output. Here every edit's output change is
decomposed by successor token into the part that lands on the token's top-K successors (the high-mass
continuations corpus successor JSD is built from) and the part that lands on the tail. Among 24 random
directions at the ladder's top rung we pick the most top-heavy and the most tail-heavy, rescale each
until both move the output by the SAME number of bits, and re-measure anchor width.

Top-heavy edits doing the damage would tie the per-token trait back to corpus successor JSD; tail-heavy
edits doing it would say the embedding probe reads something corpus statistics cannot see.
Writes results/mode_split.json.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR
from basin_probe import MODEL, REVISION, Patcher
from common import D18, RESULTS
from embed_intervene import measure
from embed_intervene2 import out_logp

NORM = 1.8          # the ladder rung at which the loud direction erased the token ordering
N_DIR = 24          # random directions whose output change is decomposed
TOPK = 32           # "top successors" = the token's TOPK most likely continuations before the edit
TARGET = 0.4        # bits of output movement both selected directions are rescaled to produce

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def jsd_terms(logp, logq):
    """Per-successor contributions to the base-2 JSD, one row per frame."""
    p, q = logp.exp(), logq.exp()
    lm = (0.5 * (p + q)).clamp_min(1e-30).log()
    return (0.5 * (p * (logp - lm) + q * (logq - lm))) / np.log(2)


def decompose(base_logp, new_logp, top_idx):
    """Total JSD in bits and the share of it carried by the token's top-K successors."""
    terms = jsd_terms(base_logp, new_logp)
    tot = terms.sum(-1)
    top = terms.gather(-1, top_idx).sum(-1)
    return float(tot.mean()), float((top / tot.clamp_min(1e-12)).mean())


@torch.inference_mode()
def calibrate(model, tok, W, i, base_row, base_logp, d, top_idx):
    """Norm along direction `d` that moves the token's output by TARGET bits (log-log interpolation)."""
    cs, bits = [], []
    for c in NORM * 1.45 ** np.arange(-4, 4):
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

    prev = json.load(open(f"{RESULTS}/intervene2.json"))["tokens"]
    patcher = Patcher(model)
    rows = {}
    for k, s in enumerate(sorted(prev, key=lambda x: prev[x]["base_w"])):
        i = prev[s]["token_id"]
        base_row = W.data[i].clone()
        base_logp = out_logp(model, tok, i)
        top_idx = base_logp.topk(TOPK, dim=-1).indices
        top_mass = float(base_logp.exp().gather(-1, top_idx).sum(-1).mean())
        base_w, _ = measure(model, patcher, tok, anchors, i)

        rng = np.random.default_rng(200 + k)
        U = rng.normal(size=(N_DIR, dim))
        U /= np.linalg.norm(U, axis=1, keepdims=True)

        bits = np.empty(N_DIR)
        share = np.empty(N_DIR)
        for j in range(N_DIR):
            W.data[i] = base_row + torch.tensor(NORM * U[j], dtype=W.dtype, device=W.device)
            bits[j], share[j] = decompose(base_logp, out_logp(model, tok, i), top_idx)
        W.data[i] = base_row

        r = dict(token_id=i, base_w=base_w, top_mass=top_mass,
                 scan=dict(bits=bits.tolist(), top_share=share.tolist(),
                           loud=dict(bits=float(bits.max()), top_share=float(share[bits.argmax()])),
                           rho_bits_share=[float(x) for x in spearmanr(bits, share)[:2]]),
                 edits={})
        for tag, j in (("top_heavy", int(share.argmax())), ("tail_heavy", int(share.argmin()))):
            c = calibrate(model, tok, W, i, base_row, base_logp, U[j], top_idx)
            W.data[i] = base_row + torch.tensor(c * U[j], dtype=W.dtype, device=W.device)
            got, sh = decompose(base_logp, out_logp(model, tok, i), top_idx)
            w_new, _ = measure(model, patcher, tok, anchors, i)
            W.data[i] = base_row
            r["edits"][tag] = dict(step_norm=c, bits=got, top_share=sh, w=w_new, dw=w_new - base_w)
        rows[s] = r
        e = r["edits"]
        print(f"[{k + 1}/{len(prev)}] {s!r} w {base_w:.3f} | top-mass {top_mass:.3f} | "
              f"loud share {r['scan']['loud']['top_share']:.3f} | "
              + " ".join(f"{t} {e[t]['bits']:.3f}b share {e[t]['top_share']:.3f} "
                         f"dw {e[t]['dw']:+.3f}" for t in e), flush=True)
        json.dump(dict(partial=True, tokens=rows), open(f"{RESULTS}/mode_split.json", "w"), indent=1)
    patcher.close()

    ks = list(rows)
    base = np.array([rows[s]["base_w"] for s in ks])
    summ = dict(top_mass=float(np.mean([rows[s]["top_mass"] for s in ks])),
                loud_share=float(np.mean([rows[s]["scan"]["loud"]["top_share"] for s in ks])),
                scan_share=[float(np.median([np.min(rows[s]["scan"]["top_share"]) for s in ks])),
                            float(np.median([np.median(rows[s]["scan"]["top_share"]) for s in ks])),
                            float(np.median([np.max(rows[s]["scan"]["top_share"]) for s in ks]))],
                rho_bits_share=float(np.median([rows[s]["scan"]["rho_bits_share"][0] for s in ks])),
                edits={})
    for tag in ("top_heavy", "tail_heavy"):
        w = np.array([rows[s]["edits"][tag]["w"] for s in ks])
        summ["edits"][tag] = dict(
            bits=float(np.median([rows[s]["edits"][tag]["bits"] for s in ks])),
            top_share=float(np.mean([rows[s]["edits"][tag]["top_share"] for s in ks])),
            step_norm=float(np.median([rows[s]["edits"][tag]["step_norm"] for s in ks])),
            dw=float(np.mean([rows[s]["edits"][tag]["dw"] for s in ks])),
            w=float(w.mean()), w_sd=float(w.std(ddof=1)),
            rho_base=[float(x) for x in spearmanr(base, w)[:2]])
    dt = np.array([rows[s]["edits"]["top_heavy"]["dw"] for s in ks])
    db = np.array([rows[s]["edits"]["tail_heavy"]["dw"] for s in ks])
    summ["paired_p"] = float(wilcoxon(dt, db)[1])
    res = dict(n_tokens=len(rows), n_dir=N_DIR, norm=NORM, topk=TOPK, target_bits=TARGET,
               base=dict(w=float(base.mean()), w_sd=float(base.std(ddof=1))),
               tokens=rows, summary=summ)
    json.dump(res, open(os.path.join(RESULTS, "mode_split.json"), "w"), indent=1)

    print(f"\nbase mass in top-{TOPK}: {summ['top_mass']:.3f}; loud direction puts "
          f"{summ['loud_share']:.3f} of its JSD there")
    print(f"top-share across 24 random directions (min/med/max, median token): "
          + "/".join(f"{x:.3f}" for x in summ["scan_share"])
          + f"; rho(bits, top-share) = {summ['rho_bits_share']:+.2f}")
    print(f"before any edit: w {res['base']['w']:.3f} +- {res['base']['w_sd']:.3f}")
    for tag in ("top_heavy", "tail_heavy"):
        o = summ["edits"][tag]
        print(f"{tag:11s}: {o['bits']:.3f} bits, top-share {o['top_share']:.3f}, "
              f"step {o['step_norm']:.2f}, w {o['w']:.3f}+-{o['w_sd']:.3f} (dw {o['dw']:+.3f}), "
              f"rho(before, after) = {o['rho_base'][0]:+.2f} (p={o['rho_base'][1]:.2g})")
    print(f"top-heavy vs tail-heavy dw, paired Wilcoxon p = {summ['paired_p']:.3f}")


if __name__ == "__main__":
    main()
