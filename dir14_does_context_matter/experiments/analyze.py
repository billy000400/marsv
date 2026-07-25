"""S5 — headline tables, threshold sensitivity, layer dependence.

Writes results/headline.json:
  exp1     : per-pair final-logit width/location/plateau at every interpolation layer
  exp2     : per-context final-logit summaries, per interpolation layer, both endpoint pairs
  classes  : per-class median/min/max width at the headline interpolation layer (block 0)
  sens     : plateau-rule threshold sensitivity (fraction of conditions called a plateau)
"""
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from assay import summarize
from sweep import RES, load

HEADLINE_L = 0          # interpolation block used for the compact headline (deepest downstream path)
CLASSES = ["none", "random", "unrelated", "relevant"]
PAIRS = ["big->large", "big->in"]


def class_of(cid):
    return "".join(ch for ch in cid if not ch.isdigit())


def main():
    c1, m1 = load("exp1")
    c2, m2 = load("exp2")
    ts = np.array(m1["ts"])
    n_layer = m1["n_layer"]
    ctx = m2["contexts"]

    exp1 = {p: {L: m1["summaries"][f"{p}|L{L}|logits"] for L in range(n_layer)} for p in PAIRS}

    exp2 = {}
    for cid in ctx:
        for p in PAIRS:
            exp2[f"{cid}|{p}"] = {L: m2["summaries"][f"{cid}|{p}|L{L}|logits"]
                                  for L in range(n_layer)}

    classes = {}
    for p in PAIRS:
        for cl in CLASSES:
            ws = [exp2[f"{cid}|{p}"][HEADLINE_L]["width"] for cid in ctx if class_of(cid) == cl]
            pl = [exp2[f"{cid}|{p}"][HEADLINE_L]["plateau"] for cid in ctx if class_of(cid) == cl]
            lo = [exp2[f"{cid}|{p}"][HEADLINE_L]["location"] for cid in ctx if class_of(cid) == cl]
            classes[f"{p}|{cl}"] = {"n": len(ws), "widths": ws, "locations": lo,
                                    "median_width": float(np.median(ws)),
                                    "min_width": float(np.min(ws)), "max_width": float(np.max(ws)),
                                    "median_location": float(np.median(lo)),
                                    "n_plateau": int(sum(pl))}

    # threshold sensitivity of the plateau rule, on the exp2 headline curves
    sens = {}
    for near in (0.05, 0.10, 0.15):
        for mrf in (0.15, 0.20, 0.30):
            for p in PAIRS:
                hits = {cl: 0 for cl in CLASSES}
                tot = {cl: 0 for cl in CLASSES}
                for cid in ctx:
                    d = c2[f"{cid}|{p}|L{HEADLINE_L}|logits"]
                    s = summarize(ts, d, near=near, min_run_frac=mrf)
                    cl = class_of(cid)
                    tot[cl] += 1
                    hits[cl] += int(s["plateau"])
                sens[f"near={near}|run={mrf}|{p}"] = {cl: [hits[cl], tot[cl]] for cl in CLASSES}

    # exp1 sensitivity too (reference context, both pairs)
    sens_exp1 = {}
    for near in (0.05, 0.10, 0.15):
        for mrf in (0.15, 0.20, 0.30):
            for p in PAIRS:
                s = summarize(ts, c1[f"{p}|L{HEADLINE_L}|logits"], near=near, min_run_frac=mrf)
                sens_exp1[f"near={near}|run={mrf}|{p}"] = [s["plateau"], s["width"]]

    # Is the class ordering explained by endpoint geometry rather than by context?
    # cos(h_A,h_B) and the norm ratio at the patched block are the two geometric covariates.
    geo = {}
    for cid in ctx:
        for p in PAIRS:
            ck = m2["checks"][f"{cid}|{p}"]
            nA, nB = ck["norms"][str(HEADLINE_L)]
            geo[f"{cid}|{p}"] = {"cos_AB": ck["cos_AB"][str(HEADLINE_L)],
                                 "norm_ratio": nB / nA,
                                 "width": exp2[f"{cid}|{p}"][HEADLINE_L]["width"],
                                 "class": class_of(cid)}
    corr = {}
    for p in PAIRS:
        rows = [v for k, v in geo.items() if k.endswith(p)]
        w = np.array([r["width"] for r in rows])
        for cov in ("cos_AB", "norm_ratio"):
            r = spearmanr(np.array([x[cov] for x in rows]), w)
            corr[f"{p}|{cov}"] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(rows)}

    # exact rank-sum tests between context classes (n=4 each; `none` has n=1 and is excluded)
    tests = {}
    for p in PAIRS:
        for a, b in (("relevant", "unrelated"), ("relevant", "random"), ("unrelated", "random")):
            wa = classes[f"{p}|{a}"]["widths"]
            wb = classes[f"{p}|{b}"]["widths"]
            r = mannwhitneyu(wa, wb, alternative="two-sided", method="exact")
            tests[f"{p}|{a}_vs_{b}"] = {"U": float(r.statistic), "p": float(r.pvalue),
                                        "median_a": float(np.median(wa)),
                                        "median_b": float(np.median(wb))}

    out = {"headline_layer": HEADLINE_L, "exp1": exp1, "exp2": exp2, "classes": classes,
           "geometry": geo, "geometry_corr": corr, "class_tests": tests,
           "sens_exp2": sens, "sens_exp1": sens_exp1, "contexts": ctx,
           "checks": {"exp1": m1["worst_checks"], "exp2": m2["worst_checks"],
                      "rerun_maxabs_diff": m1["rerun_maxabs_diff"],
                      "rerun_layer": m1["rerun_layer"]},
           "env": {k: m1[k] for k in ("model", "revision", "torch", "dtype", "device")}}
    with open(os.path.join(RES, "headline.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"headline interpolation block {HEADLINE_L}, final logits")
    for p in PAIRS:
        print(f"\n== {p}")
        print(f"  reference 'The house was': w={exp1[p][HEADLINE_L]['width']:.3f} "
              f"loc={exp1[p][HEADLINE_L]['location']:.3f} "
              f"plateau={exp1[p][HEADLINE_L]['plateau']}")
        for cl in CLASSES:
            k = classes[f"{p}|{cl}"]
            print(f"  {cl:<10} n={k['n']} width median={k['median_width']:.3f} "
                  f"[{k['min_width']:.3f},{k['max_width']:.3f}] plateau {k['n_plateau']}/{k['n']} "
                  f"loc median={k['median_location']:.3f}")
    print("\nclass comparisons (exact rank-sum on transition width, block 0 logits):")
    for k, v in tests.items():
        print(f"  {k:<34} U={v['U']:.1f} p={v['p']:.4f} "
              f"medians {v['median_a']:.3f} vs {v['median_b']:.3f}")
    print("geometry controls (Spearman of width vs endpoint geometry across the 13 contexts):")
    for k, v in corr.items():
        print(f"  {k:<28} rho={v['rho']:+.3f} p={v['p']:.3f} n={v['n']}")
    print("\nwrote", os.path.join(RES, "headline.json"))


if __name__ == "__main__":
    main()
