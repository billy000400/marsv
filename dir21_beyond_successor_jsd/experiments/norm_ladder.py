"""Does the width collapse start when the embedding moves, or when the model's output moves?

The fixed-displacement test held the displacement norm at 1.84 and could not build a behaviourally
quiet direction there: the linear response measured at a step of 0.05 does not survive that far. Here
the quiet and loud directions are rebuilt AT EACH RUNG of a displacement ladder, by measuring what 24
random directions actually do to the token's output at that norm and taking the argmin and argmax --
so "quiet" is quiet by measurement, not by extrapolation.

If width survives along the quiet direction to norms where the loud direction has already collapsed
it, the trait is a function of the behaviour the embedding induces. If the two curves stay together,
the trait is tied to the embedding's location and the vocabulary-wide lookup reads geometry.
Writes results/ladder.json.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR
from basin_probe import MODEL, REVISION, Patcher, jsd_bits
from common import D18, RESULTS
from embed_intervene import measure
from embed_intervene2 import out_logp

NORMS = [0.15, 0.4, 0.9, 1.8]     # displacement norms, against a median embedding-row norm of 0.98
N_DIR = 24                        # random directions measured at each rung

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


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
        base_w, _ = measure(model, patcher, tok, anchors, i)

        rng = np.random.default_rng(100 + k)
        U = rng.normal(size=(N_DIR, dim))
        U /= np.linalg.norm(U, axis=1, keepdims=True)

        r = dict(token_id=i, base_w=base_w, rungs={})
        for c in NORMS:
            bits = np.empty(N_DIR)
            for j in range(N_DIR):
                W.data[i] = base_row + torch.tensor(c * U[j], dtype=W.dtype, device=W.device)
                bits[j] = float(jsd_bits(base_logp, out_logp(model, tok, i)).mean())
            W.data[i] = base_row
            picks = (("quiet", int(bits.argmin())), ("loud", int(bits.argmax())), ("random", 0))
            rung = {}
            for tag, j in picks:
                W.data[i] = base_row + torch.tensor(c * U[j], dtype=W.dtype, device=W.device)
                w_new, _ = measure(model, patcher, tok, anchors, i)
                W.data[i] = base_row
                rung[tag] = dict(bits=float(bits[j]), w=w_new, dw=w_new - base_w)
            rung["bits_spread"] = [float(bits.min()), float(np.median(bits)), float(bits.max())]
            r["rungs"][f"{c:g}"] = rung
            print(f"  norm {c:4.2f}: " + " ".join(
                f"{t} {rung[t]['bits']:.3f}b->{rung[t]['dw']:+.3f}"
                for t in ("quiet", "loud", "random")), flush=True)
        rows[s] = r
        json.dump(dict(partial=True, norms=NORMS, tokens=rows),
                  open(f"{RESULTS}/ladder.json", "w"), indent=1)
        print(f"[{k + 1}/{len(prev)}] {s!r} base w {base_w:.3f} done", flush=True)
    patcher.close()

    base = np.array([rows[s]["base_w"] for s in rows])
    by_norm = {}
    for c in NORMS:
        key = f"{c:g}"
        g = {t: np.array([rows[s]["rungs"][key][t]["dw"] for s in rows])
             for t in ("quiet", "loud", "random")}
        wf = {t: np.array([rows[s]["rungs"][key][t]["w"] for s in rows])
              for t in ("quiet", "loud", "random")}
        bt = {t: np.array([rows[s]["rungs"][key][t]["bits"] for s in rows])
              for t in ("quiet", "loud", "random")}
        pooled_b = np.concatenate([bt[t] for t in bt])
        pooled_d = np.concatenate([g[t] for t in g])
        by_norm[key] = dict(
            bits={t: float(np.median(bt[t])) for t in bt},
            dw={t: float(g[t].mean()) for t in g},
            w={t: float(wf[t].mean()) for t in wf},
            w_sd={t: float(wf[t].std(ddof=1)) for t in wf},
            rho_base={t: [float(x) for x in spearmanr(base, wf[t])[:2]] for t in wf},
            quiet_vs_loud_p=float(wilcoxon(g["quiet"], g["loud"])[1]),
            rho_bits_dw=[float(x) for x in spearmanr(pooled_b, pooled_d)[:2]])

    res = dict(n_tokens=len(rows), n_dir=N_DIR, norms=NORMS, tokens=rows, by_norm=by_norm,
               base=dict(w=float(base.mean()), w_sd=float(base.std(ddof=1))))
    json.dump(res, open(os.path.join(RESULTS, "ladder.json"), "w"), indent=1)

    print(f"\nbefore any edit: mean w {res['base']['w']:.3f}, sd across tokens {res['base']['w_sd']:.3f}")
    for c in NORMS:
        o = by_norm[f"{c:g}"]
        print(f"norm {c:4.2f}: " + "  ".join(
            f"{t} {o['bits'][t]:.3f}b w {o['w'][t]:.3f}+-{o['w_sd'][t]:.3f}"
            for t in ("quiet", "loud", "random"))
            + f" | quiet-vs-loud p={o['quiet_vs_loud_p']:.2f}"
              f" rho(bits,dw)={o['rho_bits_dw'][0]:+.2f}")


if __name__ == "__main__":
    main()
