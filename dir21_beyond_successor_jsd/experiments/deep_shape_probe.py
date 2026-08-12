"""Does the width-specific component become linearly readable deeper in Pythia-1.4B?

Iteration 19 (pattern 44) refit pattern 41's four probes from three EARLY sites -- the static
embedding row W_E[u], the block-0 MLP output m_u, and the post-block-0 residual state x_u. The part of
the width that curve shape does not already explain came out at chance at all three (+0.072, +0.084,
+0.115; permutation p 0.22-0.31) against a ceiling of 0.630 the target's reliability does allow. That
bounds the three earliest sites only. The layer sweep (pattern 9) shows the median width climbing
0.553 -> 0.800 across blocks while the across-token spread collapses fivefold and the ordering
survives at +0.72, so the width is sharpened downstream of block 0, and the component could be made
explicit anywhere along the way.

This refits the same four probes from the residual state at blocks 6, 12 and 18 -- same 123 endpoint
tokens, same four targets and ceilings, same 80/43 splits, same 50-permutation nulls. Block 0 is refit
alongside as a tie-check: taken from hidden_states this time rather than from the forward hook, it must
reproduce pattern 44's four numbers, which is what makes the four panels a comparison of DEPTH rather
than of runs.

Above chance at some depth -> the component is made explicit downstream and a mid-network probe is the
screen. At chance everywhere -> the negative generalises from the earliest sites to the residual stream
at any depth.

Costs 123 tokens x 3 frames of single-token forward passes with hidden states retained; no
interpolation curves are run.

Writes results/deep_shape.json.
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from basin_probe import MODEL, REVISION, FRAMES
from common import D18, RESULTS
from edgedrift_analysis import rel, widths
from gpt2_shape_probe import EDGE_CUT, rel_ci, rel_resid, resid, run

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)

TARGETS = ["shape", "width", "width_given_shape", "shape_given_width"]
BLOCKS = [0, 6, 12, 18]


@torch.inference_mode()
def features(names):
    """Post-block residual state at the final position, for each block, averaged over the frames."""
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
    per_frame = {b: [] for b in BLOCKS}
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        rows = {b: [] for b in BLOCKS}
        for _, i in endpoints:
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            hs = model(ids, output_hidden_states=True).hidden_states
            for b in BLOCKS:                      # hidden_states[b + 1] = output of block b
                rows[b].append(hs[b + 1][0, -1].float().cpu().numpy())
        for b in BLOCKS:
            per_frame[b].append(np.stack(rows[b]))
    del model
    torch.cuda.empty_cache()
    return {f"block{b}": np.stack(per_frame[b]).mean(0) for b in BLOCKS}


def paired_tags(res, name, a, b):
    """Same 50 splits, same target, two depths: is `a` above `b` split by split?"""
    d = np.array(res[a][name]["rho_per_split"]) - np.array(res[b][name]["rho_per_split"])
    res.setdefault("paired_across_blocks", {})[f"{name}:{a}_minus_{b}"] = dict(
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
           "edge_cut": EDGE_CUT, "n_tokens": n, "blocks": BLOCKS,
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
        for b in BLOCKS[1:]:
            paired_tags(res, name, f"block{b}", "block0")

    # tie-check against pattern 44's post-block-0 probes, which used the forward hook, not
    # hidden_states: the same site reached two ways must give the same four numbers.
    prev = json.load(open(f"{RESULTS}/early_shape.json"))["resid_block0"]
    res["tie_check_vs_early_shape"] = {
        name: dict(now=res["block0"][name]["rho_mean"], iteration_19=prev[name]["rho_mean"],
                   diff=res["block0"][name]["rho_mean"] - prev[name]["rho_mean"])
        for name in TARGETS}
    print("\ntie-check, block 0 now vs pattern 44:")
    for name, d in res["tie_check_vs_early_shape"].items():
        print(f"  {name:17s} {d['now']:+.3f} vs {d['iteration_19']:+.3f}  (diff {d['diff']:+.4f})")

    json.dump(res, open(f"{RESULTS}/deep_shape.json", "w"), indent=1)
    print(f"\nwrote {RESULTS}/deep_shape.json")


if __name__ == "__main__":
    main()
