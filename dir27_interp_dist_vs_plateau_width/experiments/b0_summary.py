"""Block-0 rerun: the same descriptive numbers for every model at both intervention sites, from the sweep shards.
Writes results/block0_resid_post/summary_all.json. Run with DIR27_SITE=block0_resid_post."""
import glob
import json
import os

import numpy as np

from common import RESULTS, ROOT, SITE

assert SITE == "block0_resid_post"
BACKSTEP = 0.05
OLD = os.path.join(ROOT, "results")
DIRS = {"gpt2-large": "sweep", "gpt2-xl": "gen/gpt2-xl", "pythia-1.4b": "gen/pythia-1.4b",
        "llama-3.1-8b": "gen/llama-3.1-8b", "qwen3-8b-base": "gen/qwen3-8b-base"}


def stats(d):
    sh = [np.load(p) for p in sorted(glob.glob(os.path.join(d, "shard_*.npz")))]
    if not sh:
        return None
    D = np.concatenate([s["D"] for s in sh])
    st = np.concatenate([s["stats"] for s in sh])
    Wm, Wf = st[:, 2], st[:, 8]
    fm = (st[:, [3, 4]] > 1).any(1) | (st[:, 5] > BACKSTEP)
    ff = (st[:, [9, 10]] > 1).any(1) | (st[:, 11] > BACKSTEP)
    q = np.percentile(D, [0, 20, 40, 60, 80, 100])
    bins = []
    for i in range(5):
        s = (D >= q[i]) & (D <= q[i + 1])
        bins.append(dict(D_range=[float(q[i]), float(q[i + 1])], n=int(s.sum()),
                         W_mid=[float(x) for x in np.percentile(Wm[s], [25, 50, 75])],
                         W_final=[float(x) for x in np.percentile(Wf[s], [25, 50, 75])]))
    hist, edges = np.histogram(Wf, bins=200)
    k = int(hist.argmax())
    return dict(n=int(len(D)), D_pct_0_5_50_95_100=[float(x) for x in np.percentile(D, [0, 5, 50, 95, 100])],
                W_mid_pct_5_25_50_75_95=[float(x) for x in np.percentile(Wm, [5, 25, 50, 75, 95])],
                W_final_pct_5_25_50_75_95=[float(x) for x in np.percentile(Wf, [5, 25, 50, 75, 95])],
                W_final_max=float(Wf.max()), frac_final_narrower=float((Wf < Wm).mean()),
                flagged_mid=int(fm.sum()), flagged_final=int(ff.sum()), flagged_either=int((fm | ff).sum()),
                hist_peak_W=[float(edges[k]), float(edges[k + 1])], hist_peak_n=int(hist[k]), D_quintiles=bins)


out = {k: dict(embed=stats(os.path.join(OLD, d)), block0_resid_post=stats(os.path.join(RESULTS, d)))
       for k, d in DIRS.items()}
json.dump(out, open(os.path.join(RESULTS, "summary_all.json"), "w"), indent=1)
for k, v in out.items():
    for site, s in v.items():
        if s:
            print(k, site, "n", s["n"], "D", np.round(s["D_pct_0_5_50_95_100"], 2), "Wm", np.round(s["W_mid_pct_5_25_50_75_95"], 3),
                  "Wf", np.round(s["W_final_pct_5_25_50_75_95"], 3), "narrower", round(s["frac_final_narrower"], 3),
                  "flag m/f/e", s["flagged_mid"], s["flagged_final"], s["flagged_either"])
            print("   quintile median Wm", [round(b["W_mid"][1], 3) for b in s["D_quintiles"]],
                  "Wf", [round(b["W_final"][1], 3) for b in s["D_quintiles"]])
