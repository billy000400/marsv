"""Feedback #5: emit the full 60-pair bank as a markdown table for the report appendix.

Reads the frozen manifest and the two 1.4B assay runs and writes results/bank_table.md.
"""
import json
import os

from common import RESULTS

if __name__ == "__main__":
    man = json.load(open(os.path.join(RESULTS, "pair_manifest_top256.json")))
    tr = {r["pair_idx"]: r for r in
          json.load(open(os.path.join(RESULTS, "assay_step143000_t256.json")))["rows"]}
    un = {r["pair_idx"]: r for r in
          json.load(open(os.path.join(RESULTS, "assay_step0_t256.json")))["rows"]}
    calib = set(man["calibration_idx"])

    lines = ["| # | Q | endpoint $u$ | endpoint $v$ | count $u$ | count $v$ | "
             "$\\widehat J_{\\mathrm{sel}}$ | $\\widehat J_{\\mathrm{hold}}$ | "
             "$w$ trained | $w$ step 0 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, p in enumerate(man["pairs"]):
        star = "*" if k in calib else ""
        lines.append(
            f"| {k + 1}{star} | Q{p['bin'] + 1} | `{p['a_str']}` | `{p['b_str']}` | "
            f"{p['count_a']:,} | {p['count_b']:,} | {p['jsd_A']:.3f} | {p['jsd_B']:.3f} | "
            f"{tr[k]['w']:.3f} | {un[k]['w']:.3f} |")
    table = "\n".join(lines)
    open(os.path.join(RESULTS, "bank_table.md"), "w").write(table + "\n")
    print(table)
    print(f"\n{len(man['pairs'])} pairs, calibration subset {sorted(calib)}")
    print("bin edges (J_sel):", [round(x, 4) for x in man["jsd_A_bin_edges"]])
    print("eligible endpoints", man["n_eligible_endpoints"],
          "candidate pairs", man["n_candidate_pairs"])
