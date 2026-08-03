"""Re-score every saved raw d(t) curve set with the strict validity criteria (curve_metrics.py).

Reads results/curves_<tag>.npy + results/assay_<tag>.json, writes:
  results/qc_<tag>.json  -- per-context validity flags, per-pair w / edge drift, invalid rate by bin
  results/curves_<tag>.csv.gz  -- every raw curve, one row per (pair, context, t), for auditing

Pair-level rule: a pair's outcome is the median w over its VALID contexts; a pair with fewer than
two valid contexts is itself invalid and is excluded from the correlations (but still plotted).
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

from common import RESULTS
from curve_metrics import E_LINEAR, MONO_TOL, metrics

MIN_VALID_CTX = 2


def rescore(tag, write_csv=True):
    A = json.load(open(os.path.join(RESULTS, f"assay_{tag}.json")))
    curves = np.load(os.path.join(RESULTS, f"curves_{tag}.npy"))
    grid = np.array(A["grid"], dtype=float)
    rows = []
    for n, r in enumerate(A["rows"]):
        per = [metrics(curves[n, ci], grid) for ci in range(curves.shape[1])]
        ws = np.array([m["w"] for m in per], dtype=float)
        nv = int(sum(m["valid"] for m in per))
        rows.append(dict(
            pair_idx=r["pair_idx"], bin=r["bin"], jsd_A=r["jsd_A"], jsd_B=r["jsd_B"],
            a_str=r["a_str"], b_str=r["b_str"], out_jsd_med=r["out_jsd_med"],
            cos0=r["cos0"], dist0=r["dist0"],
            n_valid_ctx=nv, valid=bool(nv >= MIN_VALID_CTX),
            w=float(np.nanmedian(ws)) if nv >= MIN_VALID_CTX else float("nan"),
            edge_drift=float(np.median([m["edge_drift"] for m in per])),
            max_backslide=float(max(m["backslide"] for m in per)),
            fail_span=int(sum(not m["span"] for m in per)),
            fail_mono=int(sum(not m["mono"] for m in per)),
            fail_single=int(sum(not m["single"] for m in per)),
            ctx=per))
    ctx_valid = np.array([[m["valid"] for m in r["ctx"]] for r in rows])
    pair_valid = np.array([r["valid"] for r in rows])
    b = np.array([r["bin"] for r in rows])
    w = np.array([r["w"] for r in rows], dtype=float)
    ed = np.array([r["edge_drift"] for r in rows], dtype=float)
    out = dict(
        tag=tag, revision=A["revision"], model=A["model"], n_pairs=len(rows),
        mono_tol=MONO_TOL, min_valid_ctx=MIN_VALID_CTX, edge_drift_linear_reference=E_LINEAR,
        valid_curve_rate_context=float(ctx_valid.mean()),
        valid_pair_rate=float(pair_valid.mean()),
        invalid_pair_rate_by_bin={str(q): float(1 - pair_valid[b == q].mean()) for q in range(5)},
        invalid_ctx_rate_by_bin={str(q): float(1 - ctx_valid[b == q].mean()) for q in range(5)},
        fail_counts_context=dict(
            span=int(sum(r["fail_span"] for r in rows)),
            mono=int(sum(r["fail_mono"] for r in rows)),
            single=int(sum(r["fail_single"] for r in rows))),
        max_backslide_overall=float(max(r["max_backslide"] for r in rows)),
        median_w=float(np.nanmedian(w)), iqr_w=float(np.nanpercentile(w, 75) - np.nanpercentile(w, 25)),
        median_edge_drift=float(np.nanmedian(ed)), rows=rows)
    json.dump(out, open(os.path.join(RESULTS, f"qc_{tag}.json"), "w"), indent=2)

    if write_csv:  # full raw-curve export for auditing
        p = os.path.join(RESULTS, f"curves_{tag}.csv.gz")
        with gzip.open(p, "wt") as f:
            f.write("pair_idx,a,b,bin,jsd_B,context_idx,t,d\n")
            for n, r in enumerate(rows):
                for ci in range(curves.shape[1]):
                    for ti, t in enumerate(grid):
                        f.write(f"{r['pair_idx']},{r['a_str'].strip()},{r['b_str'].strip()},"
                                f"{r['bin']},{r['jsd_B']:.6f},{ci},{t:.6f},"
                                f"{curves[n, ci, ti]:.6f}\n")
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or [os.path.basename(p)[6:-5] for p in
                            sorted(glob.glob(os.path.join(RESULTS, "assay_*.json")))]
    for t in tags:
        if not os.path.exists(os.path.join(RESULTS, f"curves_{t}.npy")):
            continue
        o = rescore(t)
        print(f"{t:24s} valid ctx {o['valid_curve_rate_context']:.3f}  valid pairs "
              f"{o['valid_pair_rate']:.3f}  fails(span/mono/single) "
              f"{o['fail_counts_context']['span']}/{o['fail_counts_context']['mono']}/"
              f"{o['fail_counts_context']['single']}  max backslide {o['max_backslide_overall']:.4f}"
              f"  median w {o['median_w']:.3f}  median edge drift {o['median_edge_drift']:.3f}")
