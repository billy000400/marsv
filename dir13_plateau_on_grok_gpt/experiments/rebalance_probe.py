"""Readout-rebalancing intervention: does the plateau boundary FOLLOW the decision boundary?

PLAN's named follow-up. Experiment 5 left two live accounts of the character basins:

  (decision)     a plateau is the set of residual states that decode to the same next-character
                 prediction, so the d(t) transition should sit at the readout's decision boundary;
  (plausibility) the transition location/width is really driven by how likely the two endpoint
                 characters are, which correlates with the decision boundary anyway (rho = -0.59).

The intervention separates them WITHOUT touching blocks 1-11: we add a constant to one row of the
unembedding output (an additive logit bias), which moves the decision boundary along the path while
leaving every residual-stream activation identical.

For each pair we take the two endpoint predictions a* = argmax at t=0, b* = argmax at t=1, and the
readout gap g(t) = z_{a*}(t) - z_{b*}(t).  g(0) > 0, g(1) < 0, and the decision boundary is the
crossing t_gap where g = 0.  Biasing a* by -c gives g_c(t) = g(t) - c and moves the boundary.
Two bias settings, both fixed in advance:
  c_eq   = (g(0)+g(1))/2   -- "equalised readout": the two endpoint predictions score symmetrically;
  c_half = g(0.5)          -- the bias that puts the decision boundary exactly at the path midpoint.

Measured per pair: w and t* of d(t) (unchanged or not), t_gap under each bias, |t*-t_gap|, and the
bias size in nats relative to the endpoint gap span |g(0)-g(1)|.

Raw -> results/rebalance_raw.npz, stats -> results/rebalance_summary.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig
from matthew_assay import run_pair, transition_width, is_plateau, pava_isotonic, iso_crossing, self_test
from allpairs_sweep import load_vocab, load_model, CONTEXT, N_T, FINAL_STEP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
BLOCK = 0


def cross(ts, y):
    """First t where the strictly decreasing-through-zero function y crosses 0 (linear interp)."""
    s = np.nonzero((y[:-1] > 0) & (y[1:] <= 0))[0]
    if not len(s):
        return np.nan
    i = s[0]
    y0, y1 = y[i], y[i + 1]
    return float(ts[i] + (ts[i + 1] - ts[i]) * y0 / (y0 - y1))


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    self_test()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    V = len(chars)
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    pairs = list(itertools.combinations(range(V), 2))
    model = load_model(FINAL_STEP, device)

    rows, d_max_dev, skipped = [], 0.0, 0
    t0 = time.time()
    for n, (ia, ib) in enumerate(pairs):
        r = run_pair(model, seqs[ia], seqs[ib], BLOCK, ts, device, return_logits=True)
        z = r["logits"]                                   # [K,V] logits along the path
        am = r["argmax"]
        a_s, b_s = int(am[0]), int(am[-1])
        if a_s == b_s:                                    # no prediction change along the path
            skipped += 1
            continue
        d = r["d_logit"]
        ok, w, t_lo, t_hi, dev = is_plateau(ts, d)
        t_star = iso_crossing(ts, pava_isotonic(d), 0.5)

        g = z[:, a_s].astype(np.float64) - z[:, b_s].astype(np.float64)
        t_gap = cross(ts, g)
        c_eq = 0.5 * (g[0] + g[-1])
        c_half = float(np.interp(0.5, ts, g))
        t_gap_eq = cross(ts, g - c_eq)
        t_gap_half = cross(ts, g - c_half)

        # d(t) is computed from differences of logit vectors, so an additive bias vector cancels
        # exactly. Verify numerically on the equalised readout rather than asserting it.
        zb = z.astype(np.float64).copy()
        zb[:, a_s] -= c_eq
        dA = np.linalg.norm(zb - zb[0], axis=-1)
        dB = np.linalg.norm(zb - zb[-1], axis=-1)
        d_reb = dA / (dA + dB)
        d_max_dev = max(d_max_dev, float(np.nanmax(np.abs(d_reb - d))))

        rows.append({"A": chars[ia], "B": chars[ib], "ia": ia, "ib": ib,
                     "w": float(w), "t_star": float(t_star), "plateau": bool(ok),
                     "t_gap": t_gap, "t_gap_eq": t_gap_eq, "t_gap_half": t_gap_half,
                     "c_eq": float(c_eq), "c_half": float(c_half),
                     "gap_span": float(g[0] - g[-1]),
                     "g0": float(g[0]), "g1": float(g[-1])})
        if (n + 1) % 500 == 0:
            print(f"  {n+1}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    w, t_star = col("w"), col("t_star")
    t_gap, t_gap_eq, t_gap_half = col("t_gap"), col("t_gap_eq"), col("t_gap_half")
    ok = np.isfinite(t_gap) & np.isfinite(t_gap_eq) & np.isfinite(t_star)
    okh = ok & np.isfinite(t_gap_half)

    def med(x):
        return round(float(np.median(x)), 4)

    summary = {
        "context": CONTEXT, "step": FINAL_STEP, "block": BLOCK, "n_t": N_T,
        "n_pairs_total": len(pairs), "n_pairs_used": len(rows),
        "n_skipped_same_prediction": skipped,
        "d_invariance_max_abs_dev": float(d_max_dev),
        "median_w": med(w),
        "median_t_star": med(t_star),
        "median_t_gap": med(t_gap[ok]),
        "median_abs_tstar_minus_tgap": med(np.abs(t_star - t_gap)[ok]),
        "equalised": {
            "median_t_gap": med(t_gap_eq[ok]),
            "median_shift": med(np.abs(t_gap_eq - t_gap)[ok]),
            "frac_moved_gt_0.05": round(float(np.mean(np.abs(t_gap_eq - t_gap)[ok] > 0.05)), 4),
            "median_abs_tstar_minus_tgap": med(np.abs(t_star - t_gap_eq)[ok]),
            "median_bias_nats": med(np.abs(col("c_eq"))[ok]),
            "median_bias_over_gapspan": med((np.abs(col("c_eq")) / col("gap_span"))[ok]),
        },
        "midpoint_forced": {
            "n": int(okh.sum()),
            "median_t_gap": med(t_gap_half[okh]),
            "median_shift": med(np.abs(t_gap_half - t_gap)[okh]),
            "median_abs_tstar_minus_tgap": med(np.abs(t_star - t_gap_half)[okh]),
            "median_bias_nats": med(np.abs(col("c_half"))[okh]),
            "median_bias_over_gapspan": med((np.abs(col("c_half")) / col("gap_span"))[okh]),
        },
        "median_gap_span_nats": med(col("gap_span")),
    }
    print(json.dumps(summary, indent=2), flush=True)

    with open(os.path.join(RES, "rebalance_summary.json"), "w") as f:
        json.dump({"summary": summary, "pairs": rows}, f, indent=1)
    np.savez_compressed(os.path.join(RES, "rebalance_raw.npz"), ts=ts,
                        w=w, t_star=t_star, t_gap=t_gap, t_gap_eq=t_gap_eq,
                        t_gap_half=t_gap_half, c_eq=col("c_eq"), c_half=col("c_half"),
                        gap_span=col("gap_span"))
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
