"""Is the width-specific component readable anywhere early in Pythia-1.4B, now that we know it moves?

Pattern 41 (iteration 17) split what the static embedding row W_E[u] holds into two parts: the curve's
SHAPE (edge drift E) and the part of its WIDTH that shape does not already explain. Shape came out at
+0.783 and the width residual at +0.072 (permutation p = 0.255) -- chance. Pattern 43 (iteration 18)
then showed that the width-specific component is nevertheless carried by the block-0 MLP output m_u:
writing a donor's m_u into a recipient moves the recipient's landing width at +0.796 with the donor's
shape held constant.

A failed linear readout bounds the probe, not the vector, so those two results do not conflict. But
they leave one cheap question open: is the width-specific component readable ONE BLOCK LATER, from the
vector that demonstrably transports it? This refits pattern 41's four probes with the embedding row
replaced by

  mlp_out       m_u -- the block-0 MLP's final-position output (the transplanted vector), and
  resid_block0  x_u -- the full post-block-0 residual state at the same position,

on the same 123 endpoint tokens, the same four targets, the same 80/43 splits and the same
50-permutation nulls. The embedding block is refit alongside as a tie-check: it must reproduce pattern
41's numbers exactly, which makes the three blocks comparable rather than merely similar.

Costs 123 tokens x 3 frames of single-token forward passes; no interpolation curves are run.

Writes results/early_shape.json.
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from basin_probe import MODEL, REVISION, FRAMES, Patcher
from common import D18, RESULTS
from edgedrift_analysis import rel, widths
from gpt2_shape_probe import EDGE_CUT, rel_ci, rel_resid, resid, run
from mlp_read import MLPOut, states

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)

TARGETS = ["shape", "width", "width_given_shape", "shape_given_width"]


def features(names):
    """m_u, x_u and W_E[u] for the 123 endpoint tokens, averaged over the three frames."""
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    endpoints = sorted(ids_by_str.items())
    assert [s for s, _ in endpoints] == list(names)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher, mlp = Patcher(model), MLPOut(model)
    M, X = [], []
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        m_f, x_f = [], []
        for s, i in endpoints:
            m, x, _ = states(model, mlp, patcher, pre, i)
            m_f.append(m.cpu().numpy())
            x_f.append(x.cpu().numpy())
        M.append(np.stack(m_f))
        X.append(np.stack(x_f))
    E = model.gpt_neox.embed_in.weight.detach().float().cpu().numpy()
    emb = np.array([E[ids_by_str[s]] for s in names])
    del model
    torch.cuda.empty_cache()
    return {"mlp_out": np.stack(M).mean(0), "resid_block0": np.stack(X).mean(0), "embedding": emb}


def paired_tags(res, name, a, b):
    """Same 50 splits, same target, two feature sets: is `a` above `b` split by split?"""
    d = np.array(res[a][name]["rho_per_split"]) - np.array(res[b][name]["rho_per_split"])
    res.setdefault("paired_across_features", {})[f"{name}:{a}_minus_{b}"] = dict(
        mean=float(d.mean()), sd=float(d.std()), frac_a_higher=float((d > 0).mean()),
        p=float(wilcoxon(d).pvalue))
    print(f"paired {name:17s} {a} - {b}: {d.mean():+.3f} +- {d.std():.3f}, {a} higher in "
          f"{(d > 0).mean():.0%} of 50 splits (Wilcoxon p {wilcoxon(d).pvalue:.3g})", flush=True)


def main():
    row = json.load(open(f"{RESULTS}/edgedrift.json"))["rows"]["1.4b_block0"]
    names = row["tokens"]
    raw = json.load(open(f"{RESULTS}/xwidth_1.4b.json"))["raw"]

    e_curves = np.array(row["edge_curves"], float).reshape(len(names), -1)
    w_curves = widths(raw, names)
    w_pl = np.where(e_curves <= EDGE_CUT, w_curves, np.nan)
    y = {"shape": np.median(e_curves, 1), "width": np.nanmedian(w_pl, 1)}
    y["width_given_shape"] = resid(y["width"], y["shape"])
    y["shape_given_width"] = resid(y["shape"], y["width"])

    n = len(names)
    ceil = {
        "shape": (rel(e_curves)[0], rel_ci(lambda i: rel(e_curves[i])[0], n)),
        "width": (rel(w_pl)[0], rel_ci(lambda i: rel(w_pl[i])[0], n)),
        "width_given_shape": (rel_resid(w_pl, e_curves)[0],
                              rel_ci(lambda i: rel_resid(w_pl[i], e_curves[i])[0], n)),
        "shape_given_width": (rel_resid(e_curves, w_pl)[0],
                              rel_ci(lambda i: rel_resid(e_curves[i], w_pl[i])[0], n)),
    }

    res = {"model": MODEL, "revision": REVISION, "frames": FRAMES, "tokens": names,
           "edge_cut": EDGE_CUT, "n_tokens": n,
           "rho_shape_width": [float(x) for x in spearmanr(y["shape"], y["width"])[:2]]}
    print(f"shape and width rank the {n} tokens at rho {res['rho_shape_width'][0]:+.3f}", flush=True)

    feats = features(names)
    for tag, F in feats.items():
        res[tag] = {"n_features": int(F.shape[1])}
        print(f"\n--- {tag} ({F.shape[1]} features) ---", flush=True)
        for name in TARGETS:
            run(tag, name, F, y[name], ceil[name][0], ceil[name][1], res)

    print()
    for name in TARGETS:
        paired_tags(res, name, "mlp_out", "embedding")
        paired_tags(res, name, "resid_block0", "embedding")

    json.dump(res, open(f"{RESULTS}/early_shape.json", "w"), indent=1)
    print(f"\nwrote {RESULTS}/early_shape.json")


if __name__ == "__main__":
    main()
