"""Print the Experiment 9 comparison table (three GPT-2 models x two patch sites) as markdown."""
import json
import os

from common import RESULTS
from depth_gap import MODELS, block0_stats

out = json.load(open(os.path.join(RESULTS, "depth_gap.json")))
b0 = block0_stats(out)
fmt = lambda p: f"{p:.3g}" if p >= 1e-3 else f"{p:.0e}"

print("| Model | patch site | $f$ | $n$ | median $w_{TV}$: none | control | fixed set | $\\Delta$ | "
      "95% CI | $p$ | $\\hat\\Delta$ | fixed-set heads at or below the patch |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for mk in MODELS:
    for site, s, L, f in (("block 0", b0.get(mk), 0, 1.0),
                          ("mid", out[mk]["mid"]["stats"], out[mk]["mid_block"], out[mk]["f_mid"])):
        if s is None:
            continue
        w = s["median_wtv"]
        hd = s["median_delta"] / (0.5 - w["ctrl"])
        label = f"block {L}" if site == "mid" else site
        print(f"| {mk} | {label} | {f:.3f} | {s['n']} | {w['base']:.3f} | {w['ctrl']:.3f} | "
              f"{w['glob']:.3f} | ${s['median_delta']:+.3f}$ | "
              f"$[{s['ci_delta'][0]:+.3f}, {s['ci_delta'][1]:+.3f}]$ | ${fmt(s['wilcoxon_p'])}$ | "
              f"{hd:.1%} | {s.get('below_share', float('nan')):.0%} |")
print()
for mk in MODELS:
    print(mk, "k =", out[mk]["k"], "of", out[mk]["n_heads"], "| mid sets:", out[mk]["mid"]["sets"],
          "| mean overlap", round(out[mk]["mid"]["stats"]["mean_overlap"], 3),
          "| frac>ctrl", round(out[mk]["mid"]["stats"]["frac_glob_gt_ctrl"], 3),
          "| max endpoint err", f"{out[mk]['mid']['stats']['max_endpoint_err']:.1e}")
