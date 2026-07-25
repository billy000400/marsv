"""Confirmatory analysis of bank 2 (8 prefixes per class) and the bank-1 + bank-2 pooled test."""
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

from sweep import RES, load_meta

CLASSES = ["random", "unrelated", "relevant"]
PAIRS = ["big->large", "big->in"]
L0 = 0


def widths(meta, cls, pair, L=L0):
    return [meta["summaries"][f"{cid}|{pair}|L{L}|logits"]["width"]
            for cid, c in meta["contexts"].items() if c["class"] == cls]


def main():
    m1 = load_meta("exp2")
    m2 = load_meta("bank2")
    out = {"L": L0, "banks": {}, "tests": {}, "pooled_tests": {}}

    for name, meta in (("bank1", m1), ("bank2", m2)):
        for p in PAIRS:
            for cl in CLASSES:
                w = widths(meta, cl, p)
                out["banks"][f"{name}|{p}|{cl}"] = {
                    "n": len(w), "widths": w, "median": float(np.median(w)),
                    "min": float(np.min(w)), "max": float(np.max(w))}

    for name, meta in (("bank1", m1), ("bank2", m2)):
        for p in PAIRS:
            for a, b in (("relevant", "unrelated"), ("relevant", "random"),
                         ("unrelated", "random")):
                r = mannwhitneyu(widths(meta, a, p), widths(meta, b, p),
                                 alternative="two-sided", method="exact")
                out["tests"][f"{name}|{p}|{a}_vs_{b}"] = {"U": float(r.statistic),
                                                          "p": float(r.pvalue)}
    # pooled: natural language (relevant + unrelated) vs random tokens, both banks
    for p in PAIRS:
        nat = (widths(m1, "relevant", p) + widths(m1, "unrelated", p)
               + widths(m2, "relevant", p) + widths(m2, "unrelated", p))
        rnd = widths(m1, "random", p) + widths(m2, "random", p)
        r = mannwhitneyu(nat, rnd, alternative="two-sided", method="exact")
        out["pooled_tests"][f"{p}|natural_vs_random"] = {
            "n_nat": len(nat), "n_rnd": len(rnd), "U": float(r.statistic), "p": float(r.pvalue),
            "median_nat": float(np.median(nat)), "median_rnd": float(np.median(rnd))}
        rel = widths(m1, "relevant", p) + widths(m2, "relevant", p)
        unr = widths(m1, "unrelated", p) + widths(m2, "unrelated", p)
        r2 = mannwhitneyu(rel, unr, alternative="two-sided", method="exact")
        out["pooled_tests"][f"{p}|relevant_vs_unrelated"] = {
            "n_rel": len(rel), "n_unr": len(unr), "U": float(r2.statistic), "p": float(r2.pvalue),
            "median_rel": float(np.median(rel)), "median_unr": float(np.median(unr))}

    with open(os.path.join(RES, "bank2_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    for p in PAIRS:
        print(f"\n=== {p} (final logits, patch at block {L0})")
        for name in ("bank1", "bank2"):
            for cl in CLASSES:
                k = out["banks"][f"{name}|{p}|{cl}"]
                print(f"  {name} {cl:<10} n={k['n']} median={k['median']:.3f} "
                      f"[{k['min']:.3f}, {k['max']:.3f}]")
        for k, v in out["tests"].items():
            if f"|{p}|" in k:
                print(f"  {k:<44} U={v['U']:.1f} p={v['p']:.4g}")
        for k, v in out["pooled_tests"].items():
            if k.startswith(p):
                print(f"  POOLED {k:<40} p={v['p']:.4g} "
                      f"({[x for x in v.items() if x[0].startswith('median')]})")
    print("\nwrote", os.path.join(RES, "bank2_analysis.json"))


if __name__ == "__main__":
    main()
