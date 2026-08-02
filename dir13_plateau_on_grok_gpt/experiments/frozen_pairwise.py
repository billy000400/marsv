"""Paired per-pair width comparisons BETWEEN frozen runs.

frozen_assay.py already stores each condition's paired shift against the three reference conditions,
but the count-vs-depth reading compares frozen runs to EACH OTHER (frozen_deep vs frozen_mirror vs
frozen_two). The per-pair widths are in results/frozen_assay_raw.npz as "<cond>_w" over the same fixed
150 pairs, so the paired median shift and a Wilcoxon signed-rank test are a few lines on top of it.

Writes results/frozen_pairwise.json.
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PAIRS = [("frozen_two_last", "frozen_deep_last"),
         ("frozen_two_last", "frozen_mirror_last"),
         ("frozen_mirror_last", "frozen_deep_last"),
         ("frozen_late_last", "frozen_early_last"),
         ("frozen_two_last", "ref_init"),
         ("frozen_two_last", "ref_trained"),
         ("frozen_two_last", "ref_matched_step"),
         ("narrow192_matched", "ref_matched_step"),
         ("narrow192_matched", "frozen_early_matched"),
         ("narrow192_matched", "frozen_late_matched"),
         ("narrow192_matched", "ref_trained")]

raw = np.load(os.path.join(RES, "frozen_assay_raw.npz"))
out = {}
for a, b in PAIRS:
    ka, kb = f"{a}_w", f"{b}_w"
    if ka not in raw or kb not in raw:
        print(f"[skip] {ka} or {kb} missing")
        continue
    wa, wb = raw[ka], raw[kb]
    ok = ~(np.isnan(wa) | np.isnan(wb))
    d = wa[ok] - wb[ok]
    out[f"{a}_minus_{b}"] = {
        "n": int(ok.sum()),
        "median_dw": round(float(np.median(d)), 4),
        "frac_increased": round(float((d > 0).mean()), 4),
        "wilcoxon_p": float(wilcoxon(d).pvalue),
        "median_w_a": round(float(np.nanmedian(wa)), 4),
        "median_w_b": round(float(np.nanmedian(wb)), 4)}

print(json.dumps(out, indent=2))
with open(os.path.join(RES, "frozen_pairwise.json"), "w") as f:
    json.dump(out, f, indent=1)
