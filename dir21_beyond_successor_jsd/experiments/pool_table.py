"""Print the 250-token refit as the markdown rows the deliverables use. Reads results/pool.json."""
import json

import numpy as np

from common import RESULTS

d = json.load(open(f"{RESULTS}/pool.json"))
CFG = [("old", "80"), ("new", "80"), ("all", "80"), ("all", "125")]
HEAD = {("old", "80"): "original 123", ("new", "80"): "new 127", ("all", "80"): "all 250",
        ("all", "125"): "all 250"}

print(f"tokens: {d['old']['n_tokens']} original + {d['new']['n_tokens']} new = {d['all']['n_tokens']}"
      f"; dropped {len(d['dropped_tokens'])}; features {d['n_features']}; "
      f"feature tie-check {d['feature_tie_check_max_abs_diff']:.2e}")
print("published tie-check (block 6, n_train 80): largest |rho diff| "
      f"{max(abs(v['diff']) for v in d['tie_check_published'].values()):.4f}")

for key in d["targets"]:
    print(f"\n### {key}")
    print("| readout | " + " | ".join(f"{HEAD[c]}, train {c[1]}, test {d[c[0]][c[1]]['n_test']}"
                                      for c in CFG) + " |")
    print("|---" * (len(CFG) + 1) + "|")
    for r in ["ridge", "krr"]:
        cells = []
        for t, s in CFG:
            v = d[t][s][r][key]
            cells.append(f"${v['rho_mean']:+.3f}$ ($p = {v['perm_p']:.2f}$)")
        print(f"| {r} | " + " | ".join(cells) + " |")
    print("| null, $\\pm$2 s.d. | " + " | ".join(
        f"$\\pm{2 * d[t][s]['krr'][key]['null_sd']:.3f}$" for t, s in CFG) + " |")
    print("| ceiling | " + " | ".join(f"{d[t][s]['krr'][key]['ceiling']:.3f}" for t, s in CFG) + " |")

print("\nwidth_given_shape, per configuration: rho, null mean +- 2 s.d., cleared?")
for t, s in CFG:
    for r in ["ridge", "krr"]:
        v = d[t][s][r]["width_given_shape"]
        bar = v["null_mean"] + 2 * v["null_sd"]
        print(f"  {t:4s} train {s:>3s} test {d[t][s]['n_test']:3d} {r:5s}: rho {v['rho_mean']:+.3f} "
              f"vs bar {bar:+.3f} -> {'ABOVE' if v['rho_mean'] > bar else 'inside'}; "
              f"p {v['perm_p']:.3f}; {v['rho_mean'] / v['ceiling']:+.2f} of ceiling")

print("\ndetectable effect: rho that would sit 2 s.d. clear of the null, per configuration (krr)")
for t, s in CFG:
    v = d[t][s]["krr"]["width_given_shape"]
    print(f"  {t:4s} test {d[t][s]['n_test']:3d}: {v['null_mean'] + 2 * v['null_sd']:+.3f}")

print("\nsample comparison, original vs new tokens")
for k, v in d["sample_comparison"].items():
    print(f"  {k:6s} median {v['old_median']:+.3f} vs {v['new_median']:+.3f}; IQR "
          f"[{v['old_iqr'][0]:+.3f}, {v['old_iqr'][1]:+.3f}] vs [{v['new_iqr'][0]:+.3f}, "
          f"{v['new_iqr'][1]:+.3f}]; Mann-Whitney p {v['mannwhitney_p']:.3f}")
print(f"  rank of the original tokens {min(d['rank_used'])}-{max(d['rank_used'])}, "
      f"of the new tokens {min(d['rank_new'].values())}-{max(d['rank_new'].values())}")
print(f"  curves kept at edge <= {d['edge_cut']}: original "
      f"{d['old']['frac_curves_kept']:.3f}, new {d['new']['frac_curves_kept']:.3f}")

print("\nprice check: null s.d. vs 0.572/sqrt(n_test)")
err = []
for p in d["price_check"]:
    err.append(p["null_sd"] - p["predicted"])
    print(f"  {p['sample']:4s} test {p['n_test']:3d} {p['readout']:5s}: {p['null_sd']:.4f} vs "
          f"{p['predicted']:.4f} ({p['null_sd'] / p['predicted']:.2f}x)")
print(f"  largest absolute error {np.abs(err).max():.4f}")
