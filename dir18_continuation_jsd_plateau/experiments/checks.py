"""Required validity checks that are not already recorded by run_assay.py.

  * reversal: swapping the roles of A and B must leave w unchanged to grid precision;
  * prefix identity: within a pair the two prompts must share every token and every block-0 prefix
    residual, so the only thing the interpolation varies is the endpoint state.
"""
import json
import os

import numpy as np
import torch

from assay import run_pair
from common import DATA, RESULTS, load

N_REV = 20

if __name__ == "__main__":
    man = json.load(open(os.path.join(RESULTS, "pair_manifest.json")))
    valid = torch.tensor(np.load(os.path.join(DATA, "reliability_bank.npz"))["valid"],
                         device="cuda", dtype=torch.long)
    tok, m = load()
    c = man["contexts"][0]
    ctx = tok(c)["input_ids"]
    rng = np.random.default_rng(0)
    sel = sorted(rng.choice(len(man["pairs"]), N_REV, replace=False).tolist())

    dw, prefix_max = [], 0.0
    for pi in sel:
        p = man["pairs"][pi]
        ia = torch.tensor([ctx + [p["a"]]], device="cuda")
        ib = torch.tensor([ctx + [p["b_tok"]]], device="cuda")
        assert (ia[0, :-1] == ib[0, :-1]).all(), "prompts must share their prefix tokens"
        f = run_pair(m, ia, ib, layer=0, valid=valid)
        r = run_pair(m, ib, ia, layer=0, valid=valid)
        dw.append(abs(f["w"] - r["w"]))

    # prefix residual identity, measured directly on the hidden states of block 0
    with torch.inference_mode():
        outs = []
        for pi in sel[:5]:
            p = man["pairs"][pi]
            ia = torch.tensor([ctx + [p["a"]]], device="cuda")
            ib = torch.tensor([ctx + [p["b_tok"]]], device="cuda")
            h = [m(x, output_hidden_states=True).hidden_states[1][0, :-1] for x in (ia, ib)]
            outs.append(float((h[0] - h[1]).abs().max()))
    prefix_max = max(outs)

    out = dict(n_reversal_pairs=N_REV, grid_spacing=1 / 49,
               max_abs_w_change_on_reversal=float(np.max(dw)),
               median_abs_w_change_on_reversal=float(np.median(dw)),
               max_prefix_block0_residual_diff=prefix_max,
               context=c)
    json.dump(out, open(os.path.join(RESULTS, "checks.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
