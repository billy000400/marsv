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
         ("frozen_mid_last", "ref_trained"),
         # three-block mid-stack window (freeze 0-4 and 8-11): does mid-stack position still beat five
         # blocks at either end once the window shrinks below five?
         ("frozen_mid3_matched", "frozen_mid_matched"),
         ("frozen_mid3_matched", "frozen_deep_matched"),
         ("frozen_mid3_matched", "frozen_deep_s2_matched"),
         ("frozen_mid3_matched", "frozen_mirror_matched"),
         ("frozen_mid3_matched", "ref_matched_step"),
         ("frozen_mid3_last", "frozen_mid_last"),
         ("frozen_mid3_last", "frozen_deep_last"),
         ("frozen_mid3_last", "frozen_mirror_last"),
         ("frozen_mid3_last", "ref_trained"),
         # off-centre five-block window (freeze 0-1 and 7-11, trainable 2-6): the fourth position at
         # five trainable blocks, between frozen-mid's centre and frozen-mirror's bottom
         ("frozen_mid_off_matched", "frozen_mid_matched"),
         ("frozen_mid_off_matched", "frozen_deep_matched"),
         ("frozen_mid_off_matched", "frozen_deep_s2_matched"),
         ("frozen_mid_off_matched", "frozen_mirror_matched"),
         ("frozen_mid_off_matched", "ref_matched_step"),
         ("frozen_mid_off_last", "frozen_mid_last"),
         ("frozen_mid_off_last", "frozen_deep_last"),
         ("frozen_mid_off_last", "frozen_mirror_last"),
         ("frozen_mid_off_last", "ref_trained"),
         # bottom-shifted five-block window (freeze 0 and 6-11, trainable 1-5): the fifth position at
         # five trainable blocks and the pre-registered test of the interior/end split -- its usable
         # window touches block 1, so the split predicts it lands with the end-touching runs (> 0.47)
         # rather than with the two interior five-block windows (0.365)
         ("frozen_mid_low_matched", "frozen_mid_matched"),
         ("frozen_mid_low_matched", "frozen_mid_off_matched"),
         ("frozen_mid_low_matched", "frozen_deep_matched"),
         ("frozen_mid_low_matched", "frozen_deep_s2_matched"),
         ("frozen_mid_low_matched", "frozen_mirror_matched"),
         ("frozen_mid_low_matched", "frozen_early_matched"),
         ("frozen_mid_low_matched", "ref_matched_step"),
         ("frozen_mid_low_last", "frozen_mid_last"),
         ("frozen_mid_low_last", "frozen_mid_off_last"),
         ("frozen_mid_low_last", "frozen_deep_last"),
         ("frozen_mid_low_last", "frozen_mirror_last"),
         ("frozen_mid_low_last", "ref_trained"),
         # upper-stack five-block window (freeze 0-5 and 11, trainable 6-10): the pre-registered test of
         # the block-5 coverage description -- its usable window excludes block 5 and touches neither
         # end, so coverage predicts >= 0.55 while the refuted interior/end split would have called it
         # sharp (~0.365)
         ("frozen_high_matched", "frozen_mid_matched"),
         ("frozen_high_matched", "frozen_mid_off_matched"),
         ("frozen_high_matched", "frozen_mid_low_matched"),
         ("frozen_high_matched", "frozen_deep_matched"),
         ("frozen_high_matched", "frozen_mirror_matched"),
         ("frozen_high_matched", "frozen_late_matched"),
         ("frozen_high_matched", "ref_matched_step"),
         ("frozen_high_last", "frozen_mid_last"),
         ("frozen_high_last", "frozen_mid_low_last"),
         ("frozen_high_last", "frozen_mirror_last"),
         ("frozen_high_last", "ref_trained")]

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
        off = C.get(f"frozen_mid_off_{which}")
        if off is not None and mid is not None:  # the fourth position at five trainable blocks
            out[f"position_contrast_{which}"]["frozen_mid_off_median"] = off["median_w"]
            out[f"position_contrast_{which}"]["off_between_mid_and_deep"] = \
                bool(mid["median_w"] < off["median_w"] < min(d1, d2))
        mid3 = C.get(f"frozen_mid3_{which}")
        if mid3 is not None:  # three-block mid-stack window: does position still beat the count?
            out[f"position_contrast_{which}"]["frozen_mid3_median"] = mid3["median_w"]
            out[f"position_contrast_{which}"]["mid3_below_both_five_block_ends"] = \
                bool(mid3["median_w"] < min(d1, d2, m))

# The interior/end split, stated after S19 and tested by S20. Order the frozen runs by their USABLE
# trainable window -- trainable blocks intersected with 1-11, since patching at block 0 overwrites
# block 0's output. Windows strictly inside 1-11 are "interior"; windows containing block 1 or block 11
# are "end-touching". The split says the two groups do not overlap. frozen_mid_low (trainable 1-5)
# was run to test it: its window touches block 1, so the split predicts it joins the end-touching
# group. Reported as a group separation, not a p-value: one run per condition.
INTERIOR = {"frozen_mid_matched": "4-8", "frozen_mid_off_matched": "2-6",
            "frozen_mid3_matched": "5-7"}
END_TOUCH = {"frozen_mid_low_matched": "1-5", "frozen_early_matched": "5-11",
             "frozen_early_s2_matched": "5-11", "frozen_late_matched": "1-7",
             "frozen_deep_matched": "8-11", "frozen_deep_s2_matched": "8-11",
             "frozen_mirror_matched": "1-4", "frozen_two_matched": "11"}
have_i = {k: C[k]["median_w"] for k in INTERIOR if k in C}
have_e = {k: C[k]["median_w"] for k in END_TOUCH if k in C}
if have_i and have_e:
    out["interior_end_split"] = {
        "interior_windows": {k: [INTERIOR[k], v] for k, v in have_i.items()},
        "end_touching_windows": {k: [END_TOUCH[k], v] for k, v in have_e.items()},
        "max_interior": max(have_i.values()), "min_end_touching": min(have_e.values()),
        "groups_disjoint": bool(max(have_i.values()) < min(have_e.values())),
        "gap": round(min(have_e.values()) - max(have_i.values()), 4),
        "frozen_mid_low_median": have_e.get("frozen_mid_low_matched"),
        "mid_low_joins_end_group": (None if "frozen_mid_low_matched" not in have_e else
                                    bool(have_e["frozen_mid_low_matched"] > max(have_i.values())))}

print(json.dumps(out, indent=2))
with open(os.path.join(RES, "frozen_pairwise.json"), "w") as f:
    json.dump(out, f, indent=1)
