"""Where is the per-token width trait established? Anchor widths measured at later blocks.

Repeats the anchor-width measurement with the interpolation site moved from block 0 to blocks 6, 12
and 18, using the same six anchors, and asks whether the ranking of tokens is already fixed at block 0
or is reordered downstream. Writes results/layers.json.
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

sys.path.append("/workspace/marsv_agent_haoyang/dir18_continuation_jsd_plateau/experiments")
import curve_metrics

from anchor_width import N_ANCHOR, run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint
from common import D18, RESULTS, load
from explore1 import cv_r2

LAYERS = [6, 12, 18]
GATE = 0.2

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    endpoints = sorted(ids_by_str.items())
    used = set(ids_by_str.values())
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]
    assert len(anchors) == N_ANCHOR

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()

    widths = {}
    for L in LAYERS:
        patcher = Patcher(model, layer=L)
        rows = {s: [] for s, _ in endpoints}
        for frame in FRAMES:
            pre = tok(frame, return_tensors="pt").input_ids.cuda()
            anc = [endpoint(model, patcher,
                            torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
                   for a in anchors]
            for k, (s, i) in enumerate(endpoints):
                ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
                x, z = endpoint(model, patcher, ids)
                for xb, zb in anc:
                    rows[s].append(run_pair(model, patcher, ids, x, z, xb, zb)[0])
                if k % 60 == 0:
                    print(f"block {L} frame {frame!r} token {k}/{len(endpoints)}", flush=True)
        patcher.close()
        widths[str(L)] = {s: float(np.nanmedian(v)) for s, v in rows.items()}
        widths[f"{L}_valid"] = float(np.mean(~np.isnan(
            np.array([rows[s] for s, _ in endpoints]))))
        print(f"block {L}: valid curves {widths[f'{L}_valid']:.3f}", flush=True)

    # compare each layer's ranking with block 0 and with the fitted token effects
    w0 = {s: float(np.nanmedian(v["w"]))
          for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    tokfx = json.load(open(f"{RESULTS}/explore1.json"))["token_effects"]
    a = np.array(tokfx["effect"])
    names = tokfx["tokens"]
    t, _, _ = load()
    m = t["out_jsd_min"] >= GATE
    y = t["w"][m]
    one = np.ones((len(y), 1))
    col = lambda d: (np.array([d[x] for x in t["a_str"]])[m]
                     + np.array([d[x] for x in t["b_str"]])[m])[:, None]

    W0 = np.array([w0[s] for s in names])
    summary = {"0": dict(rho_vs_block0=[1.0, 0.0],
                         rho_vs_token_effect=[float(x) for x in spearmanr(W0, a)[:2]],
                         cv_r2=cv_r2(np.hstack([one, col(w0)]), y)[0],
                         median=float(np.median(W0)),
                         iqr=float(np.subtract(*np.percentile(W0, [75, 25]))))}
    for L in LAYERS:
        WL = np.array([widths[str(L)][s] for s in names])
        ok = ~np.isnan(WL)
        summary[str(L)] = dict(
            n=int(ok.sum()),
            rho_vs_block0=[float(x) for x in spearmanr(WL[ok], W0[ok])[:2]],
            rho_vs_token_effect=[float(x) for x in spearmanr(WL[ok], a[ok])[:2]],
            cv_r2=cv_r2(np.hstack([one, col(widths[str(L)])]), y)[0],
            median=float(np.nanmedian(WL)),
            iqr=float(np.subtract(*np.nanpercentile(WL, [75, 25]))),
            valid_rate=widths[f"{L}_valid"])
    json.dump(dict(layers=[0] + LAYERS, anchors=[tok.convert_ids_to_tokens(x) for x in anchors],
                   summary=summary, anchor_width={"0": w0,
                                                  **{str(L): widths[str(L)] for L in LAYERS}}),
              open(os.path.join(RESULTS, "layers.json"), "w"), indent=1)
    for L, v in summary.items():
        print(f"block {L:>2}: rho vs block0 {v['rho_vs_block0'][0]:+.3f}  "
              f"rho vs a_u {v['rho_vs_token_effect'][0]:+.3f}  CV-R2 {v['cv_r2']:+.3f}  "
              f"median w {v['median']:.3f}  IQR {v['iqr']:.3f}")


if __name__ == "__main__":
    main()
