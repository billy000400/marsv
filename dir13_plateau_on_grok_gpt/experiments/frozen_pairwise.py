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
from scipy.stats import mannwhitneyu, wilcoxon

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
         ("narrow192_matched", "ref_trained"),
         ("narrow192_last", "ref_trained"),
         ("narrow192_last", "frozen_early_last"),
         ("narrow192_last", "frozen_late_last"),
         ("narrow192_last", "narrow192_matched"),
         # second seeds: the across-seed spread of each measure, and the same comparisons re-run
         # from the second seed so no reported gap rests on one initialization
         ("narrow192_s2_matched", "narrow192_matched"),
         ("narrow192_s2_matched", "ref_matched_step"),
         ("narrow192_s2_matched", "frozen_early_matched"),
         ("narrow192_s2_matched", "frozen_late_matched"),
         ("frozen_early_s2_matched", "frozen_early_matched"),
         ("frozen_early_s2_matched", "ref_matched_step"),
         ("frozen_early_s2_matched", "narrow192_matched"),
         ("frozen_early_s2_matched", "narrow192_s2_matched"),
         ("frozen_early_s2_last", "frozen_early_last"),
         ("frozen_early_s2_last", "ref_trained"),
         ("frozen_early_s2_last", "narrow192_last"),
         ("frozen_early_s2_last", "frozen_deep_last"),
         # second seed of frozen 1-7: the position sub-claim (5 trainable blocks by the readout vs
         # 5 at the bottom) rests on frozen_deep vs frozen_mirror, one seed each until now
         ("frozen_deep_s2_matched", "frozen_deep_matched"),
         ("frozen_deep_s2_matched", "frozen_mirror_matched"),
         ("frozen_deep_s2_matched", "frozen_early_matched"),
         ("frozen_deep_s2_matched", "ref_matched_step"),
         ("frozen_deep_s2_last", "frozen_deep_last"),
         ("frozen_deep_s2_last", "frozen_mirror_last"),
         ("frozen_deep_s2_last", "frozen_early_last"),
         ("frozen_deep_s2_last", "ref_trained"),
         # middle-of-stack five trainable blocks (freeze 0-3 and 9-11): the third position at the same
         # trainable-block count, which turns the position term into an ordered three-point comparison
         ("frozen_mid_matched", "frozen_deep_matched"),
         ("frozen_mid_matched", "frozen_deep_s2_matched"),
         ("frozen_mid_matched", "frozen_mirror_matched"),
         ("frozen_mid_matched", "ref_matched_step"),
         ("frozen_mid_last", "frozen_deep_last"),
         ("frozen_mid_last", "frozen_mirror_last"),
         ("frozen_mid_last", "ref_trained")]

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

# Across-RUN test of the depth step. The paired tests above compare two runs pair-by-pair, which the
# second seeds show is partly a seed effect; this asks the coarser question the depth claim actually
# needs -- do the three runs with all 12 blocks trainable separate from the three with 8, treating each
# RUN's median width as one observation? With 3 vs 3 the smallest attainable one-sided p is 1/20.
GROUPS = {"trainable_12": ["ref_matched_step", "narrow192_matched", "narrow192_s2_matched"],
          "trainable_8": ["frozen_early_matched", "frozen_early_s2_matched", "frozen_late_matched"]}
C = json.load(open(os.path.join(RES, "frozen_assay_summary.json")))["summary"]["conditions"]
if all(k in C for g in GROUPS.values() for k in g):
    a = [C[k]["median_w"] for k in GROUPS["trainable_12"]]
    b = [C[k]["median_w"] for k in GROUPS["trainable_8"]]
    out["depth_step_12_vs_8_blocks"] = {
        "trainable_12_medians": a, "trainable_8_medians": b,
        "max_of_12": max(a), "min_of_8": min(b), "groups_disjoint": bool(max(a) < min(b)),
        "mannwhitney_one_sided_p": float(mannwhitneyu(a, b, alternative="less").pvalue),
        "within_condition_seed_spread": {
            "narrow192": round(abs(C["narrow192_matched"]["median_w"]
                                   - C["narrow192_s2_matched"]["median_w"]), 4),
            "frozen_early": round(abs(C["frozen_early_matched"]["median_w"]
                                      - C["frozen_early_s2_matched"]["median_w"]), 4)}}

# The position sub-claim: 5 trainable blocks next to the readout (frozen 1-7) against 5 at the bottom
# (frozen 5-11). Both conditions freeze the same 58.0% of parameters, so only position differs. With
# two seeds on the frozen_deep side the question is whether BOTH of them sit below frozen_mirror by
# more than the frozen_deep seed spread.
if "frozen_deep_s2_matched" in C:
    for which in ("matched", "last"):
        d1 = C[f"frozen_deep_{which}"]["median_w"]
        d2 = C[f"frozen_deep_s2_{which}"]["median_w"]
        m = C[f"frozen_mirror_{which}"]["median_w"]
        out[f"position_contrast_{which}"] = {
            "frozen_deep_medians": [d1, d2], "frozen_mirror_median": m,
            "frozen_deep_seed_spread": round(abs(d1 - d2), 4),
            "both_deep_seeds_below_mirror": bool(max(d1, d2) < m),
            "gap_worst_deep_seed_to_mirror": round(m - max(d1, d2), 4)}
        mid = C.get(f"frozen_mid_{which}")
        if mid is not None:  # the third position at five trainable blocks
            out[f"position_contrast_{which}"]["frozen_mid_median"] = mid["median_w"]
            out[f"position_contrast_{which}"]["mid_between_deep_and_mirror"] = \
                bool(max(d1, d2) < mid["median_w"] < m)

print(json.dumps(out, indent=2))
with open(os.path.join(RES, "frozen_pairwise.json"), "w") as f:
    json.dump(out, f, indent=1)
