"""Load dir18's 1,000-pair artifacts into one table. dir18 is read-only source material."""
import json
import os

import numpy as np

D18 = "/workspace/marsv_agent_haoyang/dir18_continuation_jsd_plateau/results"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
PLOTS = os.path.join(HERE, "plots")

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def curve_shape(d, grid):
    """Midpoint, edge drift and max slope of one d(t) curve."""
    m = float(np.interp(0.5, d, grid)) if np.all(np.diff(d) >= -0.02) else float("nan")
    lo = grid <= 0.2
    hi = grid >= 0.8
    E = float((d[lo] - d[0]).mean() + (d[-1] - d[hi]).mean())
    slope = float(np.max(np.diff(d) / np.diff(grid)))
    return m, E, slope


def load(step="step143000"):
    """One dict of parallel arrays over the 1,000 pairs, plus the raw curves."""
    assay = json.load(open(f"{D18}/assay_large_{step}.json"))
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    curves = np.load(f"{D18}/curves_large_{step}.npy")          # (1000, 3, 50)
    grid = np.asarray(assay["grid"])
    rows = assay["rows"]
    assert len(rows) == len(man) == curves.shape[0]
    for r, p in zip(rows, man):
        assert r["a_str"] == p["a_str"] and r["b_str"] == p["b_str"]

    t = {k: np.array([r[k] for r in rows]) for k in
         ("jsd_B", "w", "out_jsd_med", "bin")}
    t["a_str"] = np.array([r["a_str"] for r in rows])
    t["b_str"] = np.array([r["b_str"] for r in rows])
    t["w_ctx"] = np.array([r["w_ctx"] for r in rows], dtype=float)
    t["out_jsd_ctx"] = np.array([r["out_jsd"] for r in rows], dtype=float)
    t["cos0"] = np.array([np.median(r["cos0"]) for r in rows])
    t["dist0"] = np.array([np.median(r["dist0"]) for r in rows])
    for k in ("count_a", "count_b", "surp_a", "surp_b", "ent_a", "ent_b"):
        t[k] = np.array([p[k] for p in man], dtype=float)

    shp = np.array([[curve_shape(curves[i, c], grid) for c in range(3)]
                    for i in range(len(rows))])                  # (1000, 3, 3)
    t["mid"] = np.nanmedian(shp[:, :, 0], axis=1)
    t["edge"] = np.median(shp[:, :, 1], axis=1)
    t["slope"] = np.median(shp[:, :, 2], axis=1)
    t["w_spread"] = t["w_ctx"].max(1) - t["w_ctx"].min(1)
    t["out_jsd_min"] = t["out_jsd_ctx"].min(1)
    return t, curves, grid


def token_index(t):
    """Map the 123 endpoint tokens to 0..122 and give each pair's two token indices."""
    toks = sorted(set(t["a_str"]) | set(t["b_str"]))
    idx = {s: i for i, s in enumerate(toks)}
    ia = np.array([idx[s] for s in t["a_str"]])
    ib = np.array([idx[s] for s in t["b_str"]])
    return toks, ia, ib
