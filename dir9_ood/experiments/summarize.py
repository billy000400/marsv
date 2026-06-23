"""Read results/auroc_table.csv and print the comparisons needed for RESULTS.md / REPORT.md."""
import os, csv
from collections import defaultdict

HERE = os.path.dirname(__file__)
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "results", "auroc_table.csv"))))
for r in rows:
    r["auroc"] = float(r["auroc"])
ood_sets = sorted({r["ood_set"] for r in rows})
PLATEAU = ["plateau-perturbation", "plateau-jacobian"]
BASELINES = ["baseline-L2norm", "baseline-mahalanobis", "baseline-MSP"]
POINTS = ["input", "resid3", "resid6", "resid9"]

print("=== AUROC pivot (rows=method@point, cols=ood_set) ===")
methods_points = []
for m in PLATEAU + ["baseline-L2norm", "baseline-mahalanobis"]:
    for p in POINTS:
        methods_points.append((m, p))
methods_points.append(("baseline-MSP", "n/a"))
hdr = f"{'method@point':32s} " + " ".join(f"{o:>10s}" for o in ood_sets)
print(hdr)
val = {(r["method"], r["measurement_point"], r["ood_set"]): r["auroc"] for r in rows}
for (m, p) in methods_points:
    cells = " ".join(f"{val.get((m,p,o), float('nan')):10.3f}" for o in ood_sets)
    print(f"{m+'@'+p:32s} {cells}")

print("\n=== Verdict inputs per OOD set ===")
for o in ood_sets:
    print(f"\n-- {o} --")
    best_plateau = max(((m, p, val[(m, p, o)]) for m in PLATEAU for p in POINTS if (m, p, o) in val), key=lambda x: x[2])
    best_base = max(((m, p, val.get((m, p, o), val.get((m, "n/a", o)))) for m in BASELINES
                     for p in (POINTS if m != "baseline-MSP" else ["n/a"]) if (m, p, o) in val), key=lambda x: x[2])
    print(f"  best plateau : {best_plateau[0]}@{best_plateau[1]} = {best_plateau[2]:.3f}")
    print(f"  best baseline: {best_base[0]}@{best_base[1]} = {best_base[2]:.3f}")
    print(f"  PLATEAU BEATS BASELINES? {'YES' if best_plateau[2] > best_base[2] else 'NO'}")
    # residual vs input-space for each plateau variant
    for m in PLATEAU:
        inp = val.get((m, "input", o))
        best_resid = max((val[(m, p, o)] for p in ["resid3", "resid6", "resid9"] if (m, p, o) in val), default=float("nan"))
        if inp is not None:
            print(f"  {m}: input={inp:.3f}  best-resid={best_resid:.3f}  "
                  f"-> residual {'BEATS' if best_resid > inp else 'does NOT beat'} input-space")
