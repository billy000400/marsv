"""Sensitivity check: does it matter which corpus split supplies the predictor?

The primary analysis uses the HOLDOUT split's JSD. The SELECTION split's JSD defined the strata and
the pair bank, so using it as the predictor risks selection-on-noise -- but the two estimates agree
almost perfectly, so the choice changes nothing. Writes results/split_sensitivity.json.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS

if __name__ == "__main__":
    man = json.load(open(os.path.join(RESULTS, "pair_manifest_top256.json")))
    out = {}
    for tag in ["step143000_t256", "step0_t256", "step143000_410m_t256"]:
        rows = json.load(open(os.path.join(RESULTS, f"qc_{tag}.json")))["rows"]
        w = np.array([r["w"] for r in rows], dtype=float)
        j_sel = np.array([man["pairs"][r["pair_idx"]]["jsd_A"] for r in rows])
        j_hold = np.array([man["pairs"][r["pair_idx"]]["jsd_B"] for r in rows])
        m = np.isfinite(w)
        out[tag] = dict(n=int(m.sum()),
                        rho_selection=float(spearmanr(j_sel[m], w[m]).statistic),
                        p_selection=float(spearmanr(j_sel[m], w[m]).pvalue),
                        rho_holdout=float(spearmanr(j_hold[m], w[m]).statistic),
                        p_holdout=float(spearmanr(j_hold[m], w[m]).pvalue),
                        rho_sel_hold=float(spearmanr(j_sel[m], j_hold[m]).statistic))
        print(tag, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[tag].items()})
    json.dump(out, open(os.path.join(RESULTS, "split_sensitivity.json"), "w"), indent=2)
