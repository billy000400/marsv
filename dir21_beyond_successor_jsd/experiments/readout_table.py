"""Print the readout comparison as the markdown rows the deliverables use. Reads results/readout.json."""
import json

from common import RESULTS

d = json.load(open(f"{RESULTS}/readout.json"))
SITES, LAB = d["sites"], {"embedding": "$W_E[u]$", "mlp_out": "$m_u$", "resid_block0": "block 0",
                          "block6": "block 6", "block12": "block 12", "block18": "block 18"}

for key in ["shape", "width", "width_given_shape", "shape_given_width"]:
    print(f"\n### {key}  (ceiling {d[SITES[0]]['ridge'][key]['ceiling']:.3f})")
    print("| readout | " + " | ".join(LAB[s] for s in SITES) + " |")
    print("|---" * (len(SITES) + 1) + "|")
    for r in ["ridge", "krr", "knn"]:
        cells = []
        for s in SITES:
            v = d[s][r][key]
            cells.append(f"${v['rho_mean']:+.3f}$" + (f" ($p = {v['perm_p']:.2f}$)"
                                                     if key.startswith("width_given") else ""))
        print(f"| {r} | " + " | ".join(cells) + " |")

print("\nbest width_given_shape over all 18 site x readout cells:")
best = max(((s, r, d[s][r]["width_given_shape"]) for s in SITES for r in ["ridge", "krr", "knn"]),
           key=lambda t: t[2]["rho_mean"])
print(f"  {best[0]} {best[1]}: rho {best[2]['rho_mean']:+.3f}, p {best[2]['perm_p']:.3f}, "
      f"{best[2]['rho_mean'] / best[2]['ceiling']:+.3f} of ceiling")
print("smallest permutation p over those 18 cells:")
sm = min(((s, r, d[s][r]["width_given_shape"]) for s in SITES for r in ["ridge", "krr", "knn"]),
         key=lambda t: t[2]["perm_p"])
print(f"  {sm[0]} {sm[1]}: p {sm[2]['perm_p']:.3f}, rho {sm[2]['rho_mean']:+.3f}")

print("\npaired krr - ridge / knn - ridge on width_given_shape:")
for s in SITES:
    for r in ["krr", "knn"]:
        p = d["paired_readouts"][f"{s}:width_given_shape:{r}_minus_ridge"]
        print(f"  {s:12s} {r}: {p['mean']:+.3f} ({p['frac_a_higher']:.0%} of splits, p {p['p']:.2g})")

print("\ntie-check ridge vs patterns 44/45, largest |diff|:")
w = max(d["tie_check_ridge"].items(), key=lambda kv: abs(kv[1]["diff"]))
print(f"  {w[0]}: now {w[1]['now']:+.4f} vs published {w[1]['published']:+.4f} "
      f"(diff {w[1]['diff']:+.4f})")

print("\nmodal hyperparameters on width_given_shape:")
for s in SITES:
    print(f"  {s:12s} krr {d[s]['krr']['width_given_shape']['hyper']:22s} "
          f"knn k={d[s]['knn']['width_given_shape']['hyper']}")
