"""Print every number the S17 (frozen_mid) curation needs, once the scoring chain has run."""
import json

S = json.load(open("results/frozen_assay_summary.json"))["summary"]
P = json.load(open("results/frozen_pairwise.json"))
C = S["conditions"]

for k in ("frozen_mid_matched", "frozen_mid_last", "frozen_deep_matched", "frozen_deep_s2_matched",
          "frozen_mirror_matched", "frozen_deep_last", "frozen_deep_s2_last", "frozen_mirror_last",
          "ref_matched_step", "ref_trained"):
    c = C.get(k)
    if c is None:
        print(f"{k}: MISSING")
        continue
    print(f"\n== {k} step={c.get('step')} val={c.get('val_acc')} frozen={c.get('frozen_blocks')}")
    for f in ("median_w", "iqr_w", "strict_frac", "median_t_star", "frac_endpoints_differ",
              "median_n_argmax", "median_abs_tstar_minus_tflip", "partial_rho_w_vs_max_p_given_sep",
              "depth_median_w", "n_trainable_params"):
        if f in c:
            print(f"   {f}: {c[f]}")

print("\n== pairwise (frozen_mid rows) ==")
for k, v in P.items():
    if "frozen_mid" in k or k.startswith("position_contrast"):
        print(f"{k}: {json.dumps(v)}")
