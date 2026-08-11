"""Is the width collapse driven by how far the embedding moves, or by how much the model's output moves?

The behaviour-calibrated intervention showed that any edit the model can feel destroys a token's width
trait. It could not separate two readings: the trait is tied to the embedding's exact location (so any
large displacement erases it), or the trait is a function of the behaviour the embedding induces (so
only displacements that change behaviour erase it).

Here we hold the DISPLACEMENT NORM fixed and vary how loudly the model responds. For each token we
probe 48 random directions in the linear regime, take the SVD of the logit responses, and construct a
quiet direction (smallest singular value) and a loud one (largest); we then step by the same norm the
0.2-bit edit used along the quiet, the loud and a plain random direction, and re-measure anchor width.
Writes results/quiet.json.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR
from basin_probe import MODEL, REVISION, FRAMES, Patcher, jsd_bits
from common import D18, RESULTS
from embed_intervene import measure
from embed_intervene2 import out_logp

N_DIR = 48        # random directions probed in the linear regime to build the response matrix
EPS = 0.05        # probe step norm: the size the first intervention used, known to be near-linear

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


@torch.inference_mode()
def logits_flat(model, tok, token_id):
    """The token's final-position logits in the three frames, flattened into one vector."""
    out = []
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        ids = torch.cat([pre, torch.tensor([[token_id]], device=pre.device)], 1)
        out.append(model(ids).logits[0, -1].float())
    return torch.cat(out).cpu().numpy()


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    used = set(ids_by_str.values())
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    W = model.gpt_neox.embed_in.weight
    dim = W.shape[1]

    prev = json.load(open(f"{RESULTS}/intervene2.json"))["tokens"]
    rng = np.random.default_rng(23)
    patcher = Patcher(model)
    rows = {}
    for k, s in enumerate(sorted(prev, key=lambda x: prev[x]["base_w"])):
        i = prev[s]["token_id"]
        # the displacement every direction gets: the norm this token's 0.2-bit random edit needed
        c = [x["step_norm"] for x in prev[s]["random"]
             if x["bits_target"] == 0.2 and x["sign"] > 0][0]
        base_row = W.data[i].clone()
        base_logp = out_logp(model, tok, i)
        base_logits = logits_flat(model, tok, i)
        base_w, _ = measure(model, patcher, tok, anchors, i)

        U = rng.normal(size=(N_DIR, dim))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        A = np.empty((N_DIR, len(base_logits)), dtype=np.float32)
        for j in range(N_DIR):
            W.data[i] = base_row + torch.tensor(EPS * U[j], dtype=W.dtype, device=W.device)
            A[j] = logits_flat(model, tok, i) - base_logits
        W.data[i] = base_row
        # combinations of the probed directions, ordered by how loudly the logits respond
        Uc, S, _ = np.linalg.svd(A, full_matrices=False)
        quiet = Uc[:, -1] @ U
        loud = Uc[:, 0] @ U

        r = dict(token_id=i, base_w=base_w, step_norm=float(c),
                 sing_ratio=float(S[0] / S[-1]), dirs={})
        for tag, d in (("quiet", quiet), ("loud", loud), ("random", U[0])):
            d = d / np.linalg.norm(d)
            W.data[i] = base_row + torch.tensor(c * d, dtype=W.dtype, device=W.device)
            w_new, logp_new = measure(model, patcher, tok, anchors, i)
            r["dirs"][tag] = dict(bits=float(jsd_bits(base_logp, logp_new).mean()),
                                  w=w_new, dw=w_new - base_w)
            W.data[i] = base_row
        rows[s] = r
        json.dump(dict(partial=True, tokens=rows), open(f"{RESULTS}/quiet.json", "w"), indent=1)
        print(f"[{k + 1}/{len(prev)}] {s!r} base w {base_w:.3f} step {c:.2f} | "
              + " ".join(f"{t} {r['dirs'][t]['bits']:.3f}b->{r['dirs'][t]['dw']:+.3f}"
                         for t in ("quiet", "loud", "random")), flush=True)
    patcher.close()

    bits = np.array([rows[s]["dirs"][t]["bits"] for s in rows for t in ("quiet", "loud", "random")])
    dw = np.array([rows[s]["dirs"][t]["dw"] for s in rows for t in ("quiet", "loud", "random")])
    res = dict(n_tokens=len(rows), n_dir=N_DIR, eps=EPS, tokens=rows,
               rho_bits_dw=[float(x) for x in spearmanr(bits, dw)[:2]],
               by_dir={t: dict(
                   bits=float(np.median([rows[s]["dirs"][t]["bits"] for s in rows])),
                   dw=float(np.mean([rows[s]["dirs"][t]["dw"] for s in rows])),
                   w=float(np.mean([rows[s]["dirs"][t]["w"] for s in rows])),
                   w_sd=float(np.std([rows[s]["dirs"][t]["w"] for s in rows], ddof=1)),
                   rho_base=[float(x) for x in spearmanr(
                       [rows[s]["base_w"] for s in rows],
                       [rows[s]["dirs"][t]["w"] for s in rows])[:2]])
                   for t in ("quiet", "loud", "random")},
               base=dict(w=float(np.mean([rows[s]["base_w"] for s in rows])),
                         w_sd=float(np.std([rows[s]["base_w"] for s in rows], ddof=1))),
               step_norm=float(np.median([rows[s]["step_norm"] for s in rows])))
    json.dump(res, open(os.path.join(RESULTS, "quiet.json"), "w"), indent=1)

    print(f"\nsame displacement norm (median {res['step_norm']:.2f}) for all three directions")
    print(f"before any edit: mean w {res['base']['w']:.3f}, sd across tokens {res['base']['w_sd']:.3f}")
    for t in ("quiet", "loud", "random"):
        o = res["by_dir"][t]
        print(f"  {t:7s}: output shift {o['bits']:.4f} bits, mean dw {o['dw']:+.4f}, "
              f"mean w {o['w']:.3f}, sd across tokens {o['w_sd']:.3f}, "
              f"rho(base w, w after) {o['rho_base'][0]:+.3f} (p={o['rho_base'][1]:.1e})")
    print(f"rho(output shift, dw) over all {len(bits)} edits: {res['rho_bits_dw'][0]:+.3f} "
          f"(p={res['rho_bits_dw'][1]:.1e})")


if __name__ == "__main__":
    main()
