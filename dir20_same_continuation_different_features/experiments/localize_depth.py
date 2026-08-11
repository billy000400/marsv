"""S11 control: does the fixed head set act downstream, or only by reshaping the two endpoints?

The patch replaces the final token's resid_post *after block 0*, so a block-0 head's contribution
enters the sweep only through the two endpoint activations that are interpolated - not through the
downstream computation. Most selected heads sit in block 0, so the fixed-set result of
localize_heads.py could be an endpoint-geometry effect rather than a processing effect.

This script repeats the held-out fixed-set ablation in gpt2-large with block 0 excluded from the
frequency ranking, so every ablated head is genuinely downstream of the patch. Writes
results/localize_depth.json and regenerates plots/localization.png with the extra condition.
"""
import json
import os

import numpy as np
import torch

from ablate_heads import Ablation, N_MEAN, head_means
from analyze import tv_width
from analyze_bank import cluster_boot
from circuit_features import probe
from ablate_heads import select_heads
from common import RESULTS, load
from localize_heads import DOSE, MODELS, figure, run_condition
from mine_lowjsd import N_PREFIX, SEED, get_prefixes
from scipy.stats import wilcoxon

MKEY = "gpt2-large"


def main():
    alphas = np.linspace(0, 1, 101)
    rng_boot = np.random.default_rng(0)
    loc = json.load(open(os.path.join(RESULTS, "localize_heads.json")))
    abl = json.load(open(os.path.join(RESULTS, "ablate_heads.json")))
    small = json.load(open(os.path.join(RESULTS, f"ablate_gpt2-small.json")))
    base_rows = {r["key"]: r for r in loc[MKEY]["rows"]}

    rows = json.load(open(os.path.join(RESULTS, f"lowjsd_{MKEY}.json")))
    tok, m = load(MKEY)
    prefixes = get_prefixes(tok, np.random.default_rng(SEED), N_PREFIX)
    nh, n_block = m.config.n_head, len(m.transformer.h)
    dh = m.config.n_embd // nh
    k = max(4, int(round(DOSE * nh * n_block)))

    sets, prefix_of = [], []
    for n, r in enumerate(rows):
        pre = prefixes[r["prefix_idx"]]
        pa, pb = probe(m, pre + [r["id_a"]]), probe(m, pre + [r["id_b"]])
        d_idx, _, _, _ = select_heads(pa["contrib"], pb["contrib"], k)
        sets.append({b * nh + h for b, h in d_idx})
        prefix_of.append(r["prefix_idx"])
        if n % 100 == 0:
            print(f"    probe {n}/{len(rows)}", flush=True)

    seen, mean_prompts = set(), []
    for r in rows:
        if r["prefix_idx"] not in seen and len(mean_prompts) < N_MEAN:
            seen.add(r["prefix_idx"])
            mean_prompts.append(prefixes[r["prefix_idx"]] + [r["id_a"]])
    means = head_means(m, mean_prompts)

    folds = {0: [i for i, p in enumerate(prefix_of) if p % 2 == 0],
             1: [i for i, p in enumerate(prefix_of) if p % 2 == 1]}
    recs, gsets = [], {}
    with Ablation(m, means, dh) as ab:
        for f in (0, 1):
            freq = np.zeros(nh * n_block)
            for i in folds[1 - f]:
                for h in sets[i]:
                    freq[h] += 1
            freq[:nh] = -1                      # block 0 is upstream of the patch: exclude it
            G = [int(i) for i in np.argsort(-freq)[:k]]
            gsets[f] = G
            ab.set([(g // nh, g % nh) for g in G])
            for n, i in enumerate(folds[f]):
                r = rows[i]
                pre = prefixes[r["prefix_idx"]]
                d = run_condition(m, pre + [r["id_a"]], pre + [r["id_b"]], alphas)
                recs.append(dict(key=r["key"], prefix_idx=r["prefix_idx"], fold=f,
                                 wtv_noL0=tv_width(alphas, d),
                                 err=float(max(abs(d[0]), abs(d[-1] - 1)))))
                if n % 100 == 0:
                    print(f"    fold {f} pair {n}/{len(folds[f])}", flush=True)
            ab.remove()
    del m
    torch.cuda.empty_cache()

    joined = [dict(r, **base_rows[r["key"]]) for r in recs if r["key"] in base_rows]
    cl = {}
    for r in joined:
        cl.setdefault(r["prefix_idx"], []).append(r)
    med = lambda c: float(np.median([r[c] for r in joined]))
    stats = dict(
        n=len(joined), n_prefix=len(cl), k=k,
        median_wtv={c: med(c) for c in ("base", "ctrl", "diff", "wtv_glob", "wtv_noL0")},
        delta_vs_base=float(np.median([r["wtv_noL0"] - r["base"] for r in joined])),
        ci_vs_base=list(cluster_boot(
            cl, lambda rr: float(np.median([q["wtv_noL0"] - q["base"] for q in rr])), rng_boot)),
        p_vs_ctrl=float(wilcoxon([r["wtv_noL0"] for r in joined],
                                 [r["ctrl"] for r in joined])[1]),
        block0_share_of_selected=float(np.mean([h < nh for s in sets for h in s])),
        max_endpoint_err=float(max(r["err"] for r in joined)),
        global_sets={str(f): [[g // nh, g % nh] for g in G] for f, G in gsets.items()})
    stats["recovery"] = stats["delta_vs_base"] / float(
        np.median([r["diff"] - r["base"] for r in joined]))
    print(json.dumps({kk: vv for kk, vv in stats.items() if kk != "global_sets"}, indent=1))
    with open(os.path.join(RESULTS, "localize_depth.json"), "w") as f:
        json.dump(dict(stats=stats, rows=joined), f, indent=1)

    figure(loc, abl, small, extra={MKEY: stats["median_wtv"]["wtv_noL0"]})
    return stats


if __name__ == "__main__":
    main()
