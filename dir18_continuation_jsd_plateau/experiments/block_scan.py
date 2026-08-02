"""S3 control: repeat the assay on 10 frozen pairs while patching at blocks 0, 6, 12, 18, 23.

If the sharp transition is produced by downstream computation, it should generally weaken as fewer
blocks remain after the patch. This is a sanity control, not a layer sweep.
"""
import json
import os

import numpy as np
import torch

from assay import GRID, run_pair
from common import DATA, RESULTS, load

BLOCKS = [0, 6, 12, 18, 23]
N_EACH = 5  # 5 lowest-JSD_B + 5 highest-JSD_B pairs

if __name__ == "__main__":
    man = json.load(open(os.path.join(RESULTS, "pair_manifest.json")))
    valid = torch.tensor(np.load(os.path.join(DATA, "reliability_bank.npz"))["valid"],
                         device="cuda", dtype=torch.long)
    order = np.argsort([p["jsd_B"] for p in man["pairs"]])
    sel = [int(i) for i in list(order[:N_EACH]) + list(order[-N_EACH:])]

    tok, m = load()
    c = man["contexts"][0]
    ctx = tok(c)["input_ids"]
    curves = np.full((len(sel), len(BLOCKS), len(GRID)), np.nan, dtype=np.float32)
    rows = []
    for n, pi in enumerate(sel):
        p = man["pairs"][pi]
        ia = torch.tensor([ctx + [p["a"]]], device="cuda")
        ib = torch.tensor([ctx + [p["b_tok"]]], device="cuda")
        ws = []
        for bi, L in enumerate(BLOCKS):
            r = run_pair(m, ia, ib, layer=L, valid=valid)
            curves[n, bi] = r["d"]
            ws.append(r["w"])
        rows.append(dict(pair_idx=pi, group="low" if n < N_EACH else "high",
                         jsd_B=p["jsd_B"], a_str=p["a_str"], b_str=p["b_str"],
                         w_by_block={str(L): ws[i] for i, L in enumerate(BLOCKS)}))
        print(f"{p['a_str']!r}/{p['b_str']!r} w={np.round(ws,3)}", flush=True)

    out = dict(blocks=BLOCKS, context=c, n_pairs=len(sel), rows=rows)
    json.dump(out, open(os.path.join(RESULTS, "block_scan.json"), "w"), indent=2)
    np.save(os.path.join(RESULTS, "curves_block_scan.npy"), curves)
    for i, L in enumerate(BLOCKS):
        w = np.array([r["w_by_block"][str(L)] for r in rows], dtype=float)
        print(f"block {L}: median w = {np.nanmedian(w):.3f} (valid {np.isfinite(w).sum()}/{len(w)})")
