"""Print the learning curve as the markdown rows the deliverables use. Reads results/curve.json."""
import json

import numpy as np

from common import RESULTS

d = json.load(open(f"{RESULTS}/curve.json"))
SITES, SIZES = d["sites"], d["train_sizes"]
LAB = {"embedding": "$W_E[u]$", "mlp_out": "$m_u$", "resid_block0": "block 0", "block6": "block 6",
       "block12": "block 12", "block18": "block 18"}

for key in d["targets"]:
    print(f"\n### {key}  (ceiling {d[SITES[0]][str(SIZES[0])]['ridge'][key]['ceiling']:.3f})")
    print("| readout, training tokens | " + " | ".join(LAB[s] for s in SITES) + " |")
    print("|---" * (len(SITES) + 1) + "|")
    for r in ["ridge", "krr"]:
        for n in SIZES:
            cells = []
            for s in SITES:
                v = d[s][str(n)][r][key]
                cells.append(f"${v['rho_mean']:+.3f}$" + (f" ($p = {v['perm_p']:.2f}$)"
                                                          if key.startswith("width_given") else ""))
            print(f"| {r}, {n} | " + " | ".join(cells) + " |")

print("\nwidth_given_shape: best cell at each training size, over 6 sites x 2 readouts")
for n in SIZES:
    cells = [(s, r, d[s][str(n)][r]["width_given_shape"]) for s in SITES for r in ["ridge", "krr"]]
    b = max(cells, key=lambda t: t[2]["rho_mean"])
    sm = min(cells, key=lambda t: t[2]["perm_p"])
    print(f"  n={n:3d}: best {b[0]} {b[1]} rho {b[2]['rho_mean']:+.3f} (p {b[2]['perm_p']:.3f}, "
          f"{b[2]['rho_mean'] / b[2]['ceiling']:.2f} of ceiling); smallest p {sm[0]} {sm[1]} "
          f"p {sm[2]['perm_p']:.3f} at rho {sm[2]['rho_mean']:+.3f}; "
          f"cells above null_mean+2sd: "
          f"{sum(c[2]['rho_mean'] > c[2]['null_mean'] + 2 * c[2]['null_sd'] for c in cells)}/12")

print("\nslope of mean rho from 30 to 100 training tokens (mean over the 6 sites)")
for key in d["targets"]:
    for r in ["ridge", "krr"]:
        v = [[d[s][str(n)][r][key]["rho_mean"] for s in SITES] for n in SIZES]
        print(f"  {key:17s} {r:5s} " + "  ".join(f"n={n}: {np.mean(x):+.3f}"
                                                 for n, x in zip(SIZES, v))
              + f"   change {np.mean(v[-1]) - np.mean(v[0]):+.3f}")

print("\nnull width (2 s.d.) on width_given_shape, mean over sites and readouts, by size")
for n in SIZES:
    sd = [d[s][str(n)][r]["width_given_shape"]["null_sd"] for s in SITES for r in ["ridge", "krr"]]
    print(f"  n={n:3d}: 2 s.d. = {2 * np.mean(sd):.3f}")

print("\nwidth_given_shape cells rising monotonically with training size, out of 12 site x readout")
mono = [(s, r) for s in SITES for r in ["ridge", "krr"]
        if all(d[s][str(b)][r]["width_given_shape"]["rho_mean"]
               > d[s][str(a)][r]["width_given_shape"]["rho_mean"]
               for a, b in zip(SIZES, SIZES[1:]))]
print(f"  {len(mono)}/12: {mono}")

# How big a token pool would settle it? The null's spread is set by the size of the TEST half, and
# with 123 tokens every extra training token costs one. Fitting null_sd = c / sqrt(n_test) over the 48
# width-residual cells turns the curve into the pool size at which a given true rho clears its null.
nt = np.array([d["n_tokens"] - n for n in SIZES], float)
sd = np.array([np.mean([d[s][str(n)][r]["width_given_shape"]["null_sd"]
                        for s in SITES for r in ["ridge", "krr"]]) for n in SIZES])
c = float(np.mean(sd * np.sqrt(nt)))
print(f"\nnull_sd ~ c / sqrt(n_test) on the width residual: c = {c:.3f} "
      f"(fitted {sd.round(3).tolist()} at n_test {nt.astype(int).tolist()})")
for rho in (0.15, 0.20):
    need_test = (2 * c / rho) ** 2
    print(f"  a true rho of {rho:.2f} sits 2 null s.d. clear with a test half of {need_test:.0f} "
          f"tokens -> a pool of about {round(need_test * 3 / 50) * 50:.0f} measured tokens at the "
          f"same 2:1 train:test ratio")

print("\ntie-check at n_train=80 vs patterns 44/45:")
t = d["tie_check_ridge_n80"]
w = max(t.items(), key=lambda kv: abs(kv[1]["diff"]))
print(f"  largest |rho diff| {w[0]}: now {w[1]['now']:+.4f} vs published {w[1]['published']:+.4f} "
      f"(diff {w[1]['diff']:+.4f}); largest |null draw diff| "
      f"{max(v['null_diff'] for v in t.values()):.4f} over {len(t)} probes")

print("\nmodal ridge penalty on width_given_shape, by size:")
for n in SIZES:
    print(f"  n={n:3d}: " + ", ".join(f"{s} {d[s][str(n)]['ridge']['width_given_shape']['hyper']}"
                                      for s in SITES))
