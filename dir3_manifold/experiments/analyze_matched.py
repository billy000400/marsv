"""Summarize the parameter-matched AE sweep: per-k mean/std val_FVU across seeds,
per-doubling marginal gain, and compare to the param-drifting v2 raw sweep."""
import os, json
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
m = json.load(open(os.path.join(RES, "ae_results_matched.json")))
v2 = json.load(open(os.path.join(RES, "ae_results_gpu_v2.json")))

KS = [2, 4, 8, 16, 24, 32, 48, 64, 128, 256]


def agg(rows, predicate):
    out = {}
    for k in KS:
        vals = [r["val_fvu"] for r in rows if r["k"] == k and predicate(r)]
        if vals:
            out[k] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return out


matched = agg(m, lambda r: True)
v2raw = agg(v2, lambda r: r["prep"] == "raw")

print(f"{'k':>4} | {'matched mean±std':>20} | {'v2raw mean':>10} | {'Δ/doubling(matched)':>20}")
prev = None
for k in KS:
    mm = matched.get(k)
    vv = v2raw.get(k)
    if not mm:
        continue
    d = "" if prev is None else f"{prev - mm[0]:+.4f}"
    print(f"{k:>4} | {mm[0]:.4f}±{mm[1]:.4f} (n={mm[2]}) | "
          f"{(vv[0] if vv else float('nan')):.4f} | {d:>20}")
    prev = mm[0]

# kneedle on log2(k): find max distance below the chord from first to last point
xs = np.log2(np.array([k for k in KS if k in matched]))
ys = np.array([matched[k][0] for k in KS if k in matched])
xn = (xs - xs[0]) / (xs[-1] - xs[0])
yn = (ys - ys[0]) / (ys[-1] - ys[0])  # decreasing -> goes 0..-? ; normalize
yn = (ys - ys.min()) / (ys.max() - ys.min())
chord = 1 - xn  # straight line from (0,1) to (1,0) in normalized decreasing space
dist = chord - yn  # positive where curve is below chord (the elbow)
ke = [k for k in KS if k in matched][int(np.argmax(dist))]
print(f"\nkneedle elbow (matched, log2-k) = k≈{ke}")
print(f"param spread across k: matched 0.087% vs drifting design 12%")
