"""Refit the one open cell on 250 tokens instead of 123.

Pattern 46 left exactly one cell unresolved: RBF kernel ridge on the residual stream after block 6,
reading the part of the crossing width that curve shape does not already explain. It rose at every
training size and reached permutation p = 0.078 at n_train = 100 -- consistent with an emerging signal
and equally consistent with the best of 48 cells wandering upwards. Pattern 47 priced what would
settle it: the permutation null shrinks as 0.572/sqrt(n_test), so a true rho of +0.15 needs a test half
of about 58 tokens, which 123 tokens cannot supply at any training size.

pool_widths.py measured 127 further tokens the same way. This run refits the open cell on the enlarged
pool, in three samples that answer three different questions:

  old   the original 123, n_train = 80 -- a tie-check that this code reproduces the published cell;
  new   the 127 new tokens alone, n_train = 80 -- does the negative hold on tokens never fitted before?
  all   the combined 250, n_train = 80 and 125 -- the powered test: test halves of 170 and 125, both
        far past the 58 tokens the pricing formula asks for.

Ridge is carried alongside kernel ridge at each sample so the linear reference moves with the pool.
Everything else -- features, targets, ceilings, the 50 splits, the 50 permutation seeds, the block-6
site, EDGE_CUT -- is what patterns 44-47 used.

Writes results/pool.json.
"""
import json
import os
from collections import Counter

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from basin_probe import MODEL, REVISION, FRAMES
from common import RESULTS
from curve_probe import ridge_maps, run
from edgedrift_analysis import rel
from gpt2_shape_probe import EDGE_CUT, rel_ci, rel_resid, resid
from readout_probe import TARGETS, krr_maps, split_plan

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)

SITE = "block6"
BLOCK = 6
PRICE_C = 0.572          # null_sd = c / sqrt(n_test), fitted on pattern 47's 48 cells


@torch.inference_mode()
def features_block6(items):
    """Post-block-6 residual state at the final position, averaged over the three frames.

    Same site and same averaging as deep_shape_probe.features, which is hard-wired to the original
    123 tokens; this takes an explicit (string, id) list so the new tokens can use it too. The
    tie-check in main() confirms the two agree on the tokens they share.
    """
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    per_frame = []
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        rows = []
        for _, i in items:
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            hs = model(ids, output_hidden_states=True).hidden_states
            rows.append(hs[BLOCK + 1][0, -1].float().cpu().numpy())
        per_frame.append(np.stack(rows))
    del model
    torch.cuda.empty_cache()
    return np.stack(per_frame).mean(0)


def curves():
    """Per-curve edge drift and envelope width for the 123 measured tokens and the 127 new ones."""
    row = json.load(open(f"{RESULTS}/edgedrift.json"))["rows"]["1.4b_block0"]
    raw = json.load(open(f"{RESULTS}/xwidth_1.4b.json"))["raw"]
    old = row["tokens"]
    E = {s: np.array(e, float) for s, e in
         zip(old, np.array(row["edge_curves"], float).reshape(len(old), -1))}
    W = {s: np.array(raw[s]["w_env"], float) for s in old}
    ids = {}
    new = json.load(open(f"{RESULTS}/pool_widths.json"))
    for s, r in new["tokens"].items():
        E[s], W[s], ids[s] = np.array(r["edge"], float), np.array(r["w_env"], float), r["token_id"]
    return old, sorted(new["tokens"]), E, W, ids, new


def targets(names, E, W):
    """The four targets and their split-half ceilings, exactly as readout_probe builds them."""
    e = np.array([E[s] for s in names])
    w = np.where(e <= EDGE_CUT, np.array([W[s] for s in names]), np.nan)
    y = {"shape": np.median(e, 1), "width": np.nanmedian(w, 1)}
    y["width_given_shape"] = resid(y["width"], y["shape"])
    y["shape_given_width"] = resid(y["shape"], y["width"])
    n = len(names)
    ceil = {"shape": (rel(e)[0], rel_ci(lambda i: rel(e[i])[0], n)),
            "width": (rel(w)[0], rel_ci(lambda i: rel(w[i])[0], n)),
            "width_given_shape": (rel_resid(w, e)[0], rel_ci(lambda i: rel_resid(w[i], e[i])[0], n)),
            "shape_given_width": (rel_resid(e, w)[0], rel_ci(lambda i: rel_resid(e[i], w[i])[0], n))}
    return y, ceil, e, w


def fit(res, tag, names, idx, F, E, W, sizes):
    """Every training size for one sample: both readouts, four targets, 50 splits, 50 permutations."""
    y, ceil, e, w = targets(names, E, W)
    res[tag] = {"n_tokens": len(names), "tokens": names,
                "frac_curves_kept": float(np.mean(e <= EDGE_CUT)),
                "reliability": {k: v[0] for k, v in ceil.items()},
                "target_median": {k: float(np.median(v)) for k, v in y.items() if k in ("shape", "width")}}
    Fs = F[idx]
    for size in sizes:
        plan = split_plan(len(names), n_train=size)
        maps = {"ridge": [ridge_maps(Fs, *s) for s in plan], "krr": [krr_maps(Fs, *s) for s in plan]}
        res[tag][str(size)] = {"n_test": len(names) - size, "ridge": {}, "krr": {}}
        for name in TARGETS:
            for readout in ("ridge", "krr"):
                r = run(plan, maps[readout], y[name], ceil[name])
                res[tag][str(size)][readout][name] = r
                print(f"[{tag:4s} n={size:3d} test={len(names)-size:3d} {readout:5s}] {name:17s} "
                      f"rho {r['rho_mean']:+.3f} +- {r['rho_sd']:.3f} (ceiling {r['ceiling']:.3f} -> "
                      f"{r['disattenuated']:+.3f}); null {r['null_mean']:+.3f} +- {r['null_sd']:.3f}"
                      f" (2sd band {2 * r['null_sd']:.3f}), permutation p {r['perm_p']:.3f}",
                      flush=True)
    return y


def main():
    old, new, E, W, new_ids, meta = curves()
    # a token whose 18 curves are all above EDGE_CUT has no width left to model; the original 123 have
    # none of these, so this only ever drops new tokens, and how many is worth recording.
    dropped = [s for s in old + new if not np.any(E[s] <= EDGE_CUT)]
    old = [s for s in old if s not in dropped]
    new = [s for s in new if s not in dropped]
    names = sorted(old + new)
    idx = {s: k for k, s in enumerate(names)}
    print(f"{len(old)} original + {len(new)} new = {len(names)} tokens "
          f"({len(dropped)} dropped for having no curve at or below edge {EDGE_CUT})", flush=True)

    cache = f"{RESULTS}/pool_features.npz"
    if os.path.exists(cache):
        F = np.load(cache)["block6"]
    else:
        ids = dict(new_ids)
        row = json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"]
        ids.update({s: r["token_id"] for s, r in row.items()})
        F = features_block6([(s, ids[s]) for s in names])
        np.savez(cache, block6=F)
    assert F.shape[0] == len(names)

    # tie-check: the block-6 features of the original tokens must match the published cache, or the
    # new tokens are not sitting in the same feature space as the old ones.
    ref = np.load(f"{RESULTS}/curve_features.npz")["block6"]
    keep = [idx[s] for s in old]
    fdiff = float(np.abs(F[keep] - ref).max())
    print(f"feature tie-check over {len(old)} shared tokens: largest absolute difference {fdiff:.2e}",
          flush=True)

    res = {"site": SITE, "block": BLOCK, "edge_cut": EDGE_CUT, "targets": TARGETS,
           "n_split": 50, "n_perm": 50, "price_c": PRICE_C, "dropped_tokens": dropped,
           "feature_tie_check_max_abs_diff": fdiff, "n_features": int(F.shape[1]),
           "rank_used": meta["rank_used"], "rank_new": meta["rank_new"],
           "logprob_used": meta["logprob_used"], "logprob_new": meta["logprob_new"]}

    fit(res, "old", old, [idx[s] for s in old], F, E, W, [80])
    fit(res, "new", new, [idx[s] for s in new], F, E, W, [80])
    y_all = fit(res, "all", names, [idx[s] for s in names], F, E, W, [80, 125])

    # do the new tokens look like the old ones on the two measured targets?
    res["target_by_token"] = {k: {s: float(v) for s, v in zip(names, y_all[k])}
                              for k in ("shape", "width")}
    io, iN = [names.index(s) for s in old], [names.index(s) for s in new]
    res["sample_comparison"] = {
        k: dict(old_median=float(np.median(y_all[k][io])), new_median=float(np.median(y_all[k][iN])),
                old_iqr=[float(q) for q in np.percentile(y_all[k][io], [25, 75])],
                new_iqr=[float(q) for q in np.percentile(y_all[k][iN], [25, 75])],
                mannwhitney_p=float(__import__("scipy.stats", fromlist=["mannwhitneyu"])
                                    .mannwhitneyu(y_all[k][io], y_all[k][iN]).pvalue))
        for k in ("shape", "width")}

    # Pooling two samples that differ on the target lets a probe score by telling the samples apart
    # rather than by ranking tokens within either. This control removes that route: the residual
    # targets are formed inside each sample separately and z-scored there, so the two groups have the
    # same mean by construction and a group indicator carries no information about the target.
    yo, _, _, _ = targets(old, E, W)
    yn, _, _, _ = targets(new, E, W)
    pos = {s: k for k, s in enumerate(names)}
    res["group_control"] = {"n_train": 125, "ridge": {}, "krr": {}}
    plan = split_plan(len(names), n_train=125)
    maps = {"ridge": [ridge_maps(F, *s) for s in plan], "krr": [krr_maps(F, *s) for s in plan]}
    for name in TARGETS:
        yg = np.empty(len(names))
        for grp, yy in ((old, yo), (new, yn)):
            v = yy[name]
            v = (v - v.mean()) / v.std()
            for s, x in zip(grp, v):
                yg[pos[s]] = x
        for readout in ("ridge", "krr"):
            r = run(plan, maps[readout], yg, (0.0, [None, None]))
            for k in ("reliability", "reliability_ci", "ceiling", "disattenuated"):
                r.pop(k)                       # the z-scored within-sample target has no ceiling here
            res["group_control"][readout][name] = r
            print(f"[within-sample control n=125 test=125 {readout:5s}] {name:17s} "
                  f"rho {r['rho_mean']:+.3f} +- {r['rho_sd']:.3f}; null {r['null_mean']:+.3f} +- "
                  f"{r['null_sd']:.3f} (2sd band {2 * r['null_sd']:.3f}), p {r['perm_p']:.3f}",
                  flush=True)

    # tie-check: old at n_train = 80 must reproduce the published pattern-46 cell.
    pub = json.load(open(f"{RESULTS}/readout.json"))[SITE]
    res["tie_check_published"] = {
        f"{readout}:{name}": dict(now=res["old"]["80"][readout][name]["rho_mean"],
                                  published=pub[readout][name]["rho_mean"],
                                  diff=res["old"]["80"][readout][name]["rho_mean"]
                                  - pub[readout][name]["rho_mean"])
        for readout in ("ridge", "krr") for name in TARGETS}
    worst = max(abs(d["diff"]) for d in res["tie_check_published"].values())
    print(f"\ntie-check vs pattern 46 at block6, n_train=80: largest |rho diff| {worst:.4f} over "
          f"{len(res['tie_check_published'])} probes", flush=True)

    # out-of-sample check of pattern 47's pricing formula null_sd = c / sqrt(n_test)
    res["price_check"] = [
        dict(sample=t, n_train=int(s), n_test=res[t][s]["n_test"], readout=r,
             null_sd=res[t][s][r]["width_given_shape"]["null_sd"],
             predicted=PRICE_C / np.sqrt(res[t][s]["n_test"]))
        for t in ("old", "new", "all") for s in res[t] if s.isdigit() for r in ("ridge", "krr")]
    for p in res["price_check"]:
        print(f"price check [{p['sample']:4s} n_test={p['n_test']:3d} {p['readout']:5s}] null sd "
              f"{p['null_sd']:.4f} vs predicted {p['predicted']:.4f}", flush=True)

    json.dump(res, open(f"{RESULTS}/pool.json", "w"), indent=1)
    print(f"\nwrote {RESULTS}/pool.json")


if __name__ == "__main__":
    main()
