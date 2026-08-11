"""Where the differentially-engaged heads sit, in the third model too.

localize_depth.py showed that gpt2-large's intervention effect is carried by heads in block 0 -
upstream of the patch, so they act on the interpolated vector itself rather than on the computation
below it. gpt2-large selects 16.7% of its heads there and gpt2-medium 0.0%. This adds gpt2-small,
whose dose sweep is in results/ablate_gpt2-small.json, so the share can be read against the three
models' effect sizes. No sweeps: two clean forward passes per pair.

Appends to results/localize_heads.json under "gpt2-small".
"""
import json
import os

import numpy as np

from ablate_heads import select_heads
from circuit_features import probe
from common import RESULTS, load
from localize_heads import DOSE, recurrence
from mine_lowjsd import N_PREFIX, SEED, get_prefixes

MKEY = "gpt2-small"


def main():
    rows = json.load(open(os.path.join(RESULTS, f"lowjsd_{MKEY}.json")))
    tok, m = load(MKEY)
    prefixes = get_prefixes(tok, np.random.default_rng(SEED), N_PREFIX)
    nh, n_block = m.config.n_head, len(m.transformer.h)
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

    mag_sets = sets   # magnitude sets are not needed here; recurrence() only reads "diff" and "mag"
    rec = recurrence(dict(diff=sets, mag=mag_sets), prefix_of, k, nh * n_block,
                     np.random.default_rng(0))
    rec["layer_mass"] = np.bincount([h // nh for s in sets for h in s],
                                    minlength=n_block).astype(float).tolist()
    rec["n_block"], rec["nh"], rec["k"], rec["n_heads"] = n_block, nh, k, nh * n_block
    rec["block0_share"] = float(np.mean([h < nh for s in sets for h in s]))
    del rec["jaccard"]["mag_within"], rec["jaccard"]["mag_across"]   # duplicate of diff here

    path = os.path.join(RESULTS, "localize_heads.json")
    out = json.load(open(path))
    out[MKEY] = dict(recurrence=rec)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({kk: vv for kk, vv in rec.items() if kk != "freq"}, indent=1))
    return rec


if __name__ == "__main__":
    main()
