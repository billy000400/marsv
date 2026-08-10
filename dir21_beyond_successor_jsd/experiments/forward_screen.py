"""Forward screen: predict the width of pairs of UNSEEN tokens before running them.

Takes 40 tokens that appear in none of the 1,000 bank pairs, measures only their anchor widths, and
predicts the width of every pair among them using a slope and intercept fitted entirely on the bank.
Nothing is fitted on the new tokens. Writes results/forward.json.
"""
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

sys.path.append("/workspace/marsv_agent_haoyang/dir18_continuation_jsd_plateau/experiments")
import curve_metrics

from anchor_width import GRID, N_ANCHOR, run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint
from common import D18, RESULTS, load

N_NEW = 40
GATE = 0.2

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def bank_fit():
    """Slope and intercept of pair width on the sum of the two tokens' anchor widths, fitted on the
    929 gated bank pairs only."""
    t, _, _ = load()
    m = t["out_jsd_min"] >= GATE
    aw = {s: float(np.nanmedian(v["w"]))
          for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    s = (np.array([aw[x] for x in t["a_str"]]) + np.array([aw[x] for x in t["b_str"]]))[m]
    X = np.column_stack([np.ones(m.sum()), s])
    beta = np.linalg.lstsq(X, t["w"][m], rcond=None)[0]
    return [float(b) for b in beta]


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    used = set()
    for p in man:
        used |= {p["a"], p["b_tok"]}
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]
    rest = [i for i in pool if i not in set(anchors)]
    new = rest[:: max(1, len(rest) // N_NEW)][:N_NEW]

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher = Patcher(model)
    strs = [tok.convert_ids_to_tokens(i).replace("Ġ", " ") for i in new]

    # 1. per-token anchor widths, and 2. the true width of every pair among the new tokens
    aw = {s: [] for s in strs}
    pairs = list(itertools.combinations(range(len(new)), 2))
    pw = {f"{i}_{j}": dict(w=[], out_jsd=[]) for i, j in pairs}
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        anc = [endpoint(model, patcher, torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
               for a in anchors]
        ep = []
        for i in new:
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            ep.append((ids,) + endpoint(model, patcher, ids))
        for k, (ids, x, z) in enumerate(ep):
            for xb, zb in anc:
                aw[strs[k]].append(run_pair(model, patcher, ids, x, z, xb, zb)[0])
        print(f"anchor widths done for frame {frame!r}", flush=True)
        for n, (i, j) in enumerate(pairs):
            ids, x, z = ep[i]
            _, xb, zb = ep[j]
            w, oj, _ = run_pair(model, patcher, ids, x, z, xb, zb)
            pw[f"{i}_{j}"]["w"].append(w)
            pw[f"{i}_{j}"]["out_jsd"].append(oj)
            if n % 200 == 0:
                print(f"  pair {n}/{len(pairs)}", flush=True)

    # 3. score the forward prediction: no parameter is fitted on these tokens
    b0, b1 = bank_fit()
    what = {s: float(np.nanmedian(v)) for s, v in aw.items()}
    rows = []
    for i, j in pairs:
        d = pw[f"{i}_{j}"]
        w = float(np.nanmedian(d["w"]))
        rows.append(dict(a=strs[i], b=strs[j], w=w,
                         n_valid=int(np.sum(~np.isnan(d["w"]))),
                         out_jsd_med=float(np.median(d["out_jsd"])),
                         out_jsd_min=float(np.min(d["out_jsd"])),
                         pred=b0 + b1 * (what[strs[i]] + what[strs[j]])))
    ok = [r for r in rows if not np.isnan(r["w"]) and r["n_valid"] >= 2
          and r["out_jsd_min"] >= GATE]
    y = np.array([r["w"] for r in ok])
    p = np.array([r["pred"] for r in ok])
    o = np.array([r["out_jsd_med"] for r in ok])
    r2 = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    out = dict(n_new_tokens=len(new), new_tokens=strs, anchors=[tok.convert_ids_to_tokens(a)
                                                               for a in anchors],
               bank_fit=dict(intercept=b0, slope=b1),
               n_pairs=len(rows), n_scored=len(ok),
               anchor_width_pct=[float(x) for x in np.percentile(list(what.values()), [5, 50, 95])],
               w_pct=[float(x) for x in np.percentile(y, [5, 50, 95])],
               r2_forward=r2,
               rho_forward=[float(x) for x in spearmanr(p, y)[:2]],
               rho_out_jsd=[float(x) for x in spearmanr(o, y)[:2]],
               mean_abs_err=float(np.abs(y - p).mean()),
               anchor_width=what, rows=rows)
    json.dump(out, open(os.path.join(RESULTS, "forward.json"), "w"), indent=1)
    print(f"scored {len(ok)}/{len(rows)} pairs   forward R2 = {r2:+.3f}   "
          f"rho = {out['rho_forward'][0]:+.3f}   MAE = {out['mean_abs_err']:.3f}")


if __name__ == "__main__":
    main()
