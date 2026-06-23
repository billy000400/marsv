"""Compare the REAL cupbearer package detectors (cupenv, results/auroc_cupbearer.csv) against
iter-2's vendored cupbearer math (results/auroc_table.csv) and the other baselines. Confirms the
vendored cup-RMD/cup-QUE matched the genuine package, and reports the two NEW detectors that only
the real package provides (cup-maha as a cupbearer object, cup-spectral). Pure stdlib; base env."""
import csv, os

HERE = os.path.dirname(__file__)
REAL = os.path.join(HERE, "..", "results", "auroc_cupbearer.csv")
VEND = os.path.join(HERE, "..", "results", "auroc_table.csv")
OUT = os.path.join(HERE, "..", "results", "cup_real_vs_vendored.csv")


def load(path, method_col, point_col, auroc_col, ood_col):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(r[ood_col], r[method_col], r[point_col])] = float(r[auroc_col])
    return rows


def main():
    real = load(REAL, "method", "measurement_point", "auroc", "ood_set")
    vend = load(VEND, "method", "measurement_point", "auroc", "ood_set")
    # map vendored names -> real names
    name_map = {"cup-RMD": "cup-RMD", "cup-QUE": "cup-QUE"}
    print("=== REAL cupbearer package AUROC (results/auroc_cupbearer.csv) ===")
    methods = sorted({m for (_, m, _) in real})
    points = ["input", "resid3", "resid6", "resid9"]
    oods = ["random", "shuffled", "code"]
    for m in methods:
        for p in points:
            cells = []
            for o in oods:
                v = real.get((o, m, p))
                cells.append(f"{o}={v:.3f}" if v is not None else f"{o}=NA")
            print(f"  {m}@{p}: " + " ".join(cells))
    print("\n=== vendored vs real (should match within ~0.03 if vendored math was faithful) ===")
    out_rows = [("ood_set", "method", "point", "real", "vendored", "abs_diff")]
    diffs = []
    for (o, vm, p), va in sorted(vend.items()):
        if vm in name_map:
            ra = real.get((o, name_map[vm], p))
            if ra is None:
                continue
            d = abs(ra - va)
            diffs.append(d)
            out_rows.append((o, vm, p, f"{ra:.4f}", f"{va:.4f}", f"{d:.4f}"))
            print(f"  {vm}@{p} {o}: real={ra:.3f} vendored={va:.3f} |d|={d:.3f}")
    if diffs:
        print(f"\nmax|diff|={max(diffs):.3f}  mean|diff|={sum(diffs)/len(diffs):.3f}  (n={len(diffs)})")
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)
    print("WROTE", OUT)
    # Best detector per OOD set across the REAL package methods
    print("\n=== best REAL-cupbearer detector per OOD set ===")
    for o in oods:
        best = max(((m, p, real[(o, m, p)]) for (oo, m, p) in real if oo == o),
                   key=lambda t: t[2])
        print(f"  {o}: {best[0]}@{best[1]} = {best[2]:.3f}")


if __name__ == "__main__":
    main()
