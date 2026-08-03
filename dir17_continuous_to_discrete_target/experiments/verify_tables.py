"""Re-derive every number in the RESULTS.md tables from results/*.json and diff.

Guards against transcription drift between the stored analysis and the curated
deliverable. Prints one line per mismatch; exit 1 if any.

Usage: python experiments/verify_tables.py
"""
import json
import re
import sys

ROOT = __file__.rsplit("/experiments/", 1)[0]
KS = [0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320]


def load(name):
    with open(f"{ROOT}/results/{name}") as f:
        return json.load(f)


def md_tables(path):
    """Every markdown table as a list of cell-lists (header + body rows)."""
    with open(path) as f:
        lines = f.read().split("\n")
    tables, cur = [], []
    for line in lines:
        if line.strip().startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


def num(cell):
    """Leading number of a cell like '1.094 ± 0.010' or '$0.5$'; None if absent."""
    m = re.search(r"-?\d+\.?\d*", cell.replace("$", ""))
    return float(m.group(0)) if m else None


def cmp_col(name, published, expected, tol):
    bad = []
    for k, p, e in zip(KS, published, expected):
        if p is None or abs(p - e) > tol:
            bad.append(f"  {name} k={k}: published {p} vs results {e:.4f}")
    return bad


# Which stored quantity each published column holds, per table content.
# ("main"/"n10k", json key) or ("lam", grid, layer key) for the Lambda table.
GAMMA = [("main", "target_G"), ("main", "pred_G"),
         ("main", "G1"), ("main", "G2"), ("main", "G3")]
PHI = [("main", "target_P"), ("main", "pred_P"),
       ("main", "P1"), ("main", "P2"), ("main", "P3")]
GRID10K = [("n10k", "target_G"), ("n10k", "pred_G"), ("n10k", "G3"),
           ("n10k", "pred_P"), ("n10k", "P3"), ("n10k", "r2")]
LAMBDA = [("lam", "main", "target"), ("lam", "main", "out"), ("lam", "main", "L3"),
          ("lam", "n10k", "out"), ("lam", "n10k", "L3")]
DIAG = [("main", "r2"), ("main", "val"), ("main", "rho"), ("main", "val_min_ep")]

# Table order differs between the two deliverables; list it explicitly.
LAYOUT = {"RESULTS.md": [GAMMA, PHI, DIAG, GRID10K, LAMBDA],
          "REPORT.md": [GAMMA, PHI, GRID10K, LAMBDA, DIAG]}


def main():
    rows = {"main": {r["k"]: r for r in load("summary_table.json")},
            "n10k": {r["k"]: r for r in load("summary_table_n10k.json")}}
    zoom = {"main": load("zoom_main.json"), "n10k": load("zoom_n10k.json")}

    def lam(grid, part, k):
        """Alignment-free concentration at w=0.0025, averaged over seeds."""
        z = zoom[grid]
        if part == "target":
            return z["target"][f"k{k:g}"]["lambda_0.0025"]
        vals = [z["models"][f"k{k:g}|s{s}"][part]["lambda_0.0025"] for s in z["seeds"]]
        return sum(vals) / len(vals)

    fails, n_rows = [], 0
    for path, spec in LAYOUT.items():
        body = [t[2:] for t in md_tables(f"{ROOT}/{path}")]  # drop header + separator
        if len(body) != len(spec):
            fails.append(f"  {path}: {len(body)} tables, expected {len(spec)}")
            continue
        for tnum, (tab, cols) in enumerate(zip(body, spec), start=1):
            n_rows += len(tab)
            for col, src in enumerate(cols, start=1):
                pub = [num(r[col]) for r in tab]
                if src[0] == "lam":
                    exp = [lam(src[1], src[2], k) for k in KS]
                    tol = 6e-3 if src[2] == "target" else 6e-4
                else:
                    exp = [rows[src[0]][k][src[1]] for k in KS]
                    # epochs are integers; target/output columns are published to 2dp
                    tol = (0.5 if src[1] == "val_min_ep"
                           else 6e-3 if src[1].startswith(("target", "pred_G"))
                           else 6e-4)
                fails += cmp_col(f"{path} T{tnum} col{col}={src[-1]}", pub, exp, tol)

    if fails:
        print(f"{len(fails)} MISMATCH(ES):")
        print("\n".join(fails))
        return 1
    print(f"all {n_rows} published table rows match results/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
