"""Do the three onsets survive a different definition of "transition width"?

Every headline number in this report is computed from one metric,

    w = t(d = 0.9) - t(d = 0.1),

and the 10%/90% levels are a convention inherited from the upstream work, not something the data
chose. A reader is entitled to ask whether the ~60x separation between the divergence-ordering onset
and the plateau-shape onset is a property of the model or a property of that convention -- for
instance, a wider band (5%/95%) weights the flat ends more, a narrower one (30%/70%) weights only the
steep middle, and the two could in principle place the onsets differently.

So we recompute the whole trajectory at five levels a in {0.10, 0.15, 0.20, 0.25, 0.30}, with

    w_a = t(d = 1 - a) - t(d = a),      straight-line reference  1 - 2a,

and re-run both prespecified onset rules on each. Curve VALIDITY is left at the original 0.1/0.9
definition on purpose, so the set of curves entering every trajectory is identical and only the
width definition changes. Levels below 0.10 are not used: V1 only guarantees d(0) <= 0.1 and
d(1) >= 0.9, so a 5% level need not be attained at all.

CPU only; reads the saved curves. Writes results/threshold_robustness.json.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

import curve_metrics
from analyze import GRID, load_checkpoints
from common import RESULTS

B = 4000
RNG = np.random.default_rng(21)
LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30]
E_LINEAR = curve_metrics.E_LINEAR


def sp(x, y):
    """Spearman correlation, without scipy's per-call overhead (this is run ~400,000 times)."""
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def width_at(d, a):
    """t(d = 1-a) - t(d = a) for one raw curve, by linear interpolation between grid points."""
    lo = curve_metrics._first_up(d, GRID, a)
    hi = curve_metrics._first_up(d, GRID, 1 - a)
    return float(hi - lo) if (lo is not None and hi is not None and hi >= lo) else np.nan


def bracket_ordering(lo, hi):
    """Prespecified rule: opens after the last checkpoint whose simultaneous band contains zero,
    closes at the first of TWO CONSECUTIVE checkpoints whose band lies entirely below zero."""
    n = len(hi)
    for k in range(n - 1):
        if hi[k] < 0 and hi[k + 1] < 0:
            prev = [j for j in range(k) if lo[j] <= 0 <= hi[j]]
            return (prev[-1] if prev else None), k
    return None, None


def bracket_shape(med_hi, ref, e_hi):
    """Prespecified rule: two consecutive checkpoints whose median-width band lies below the
    straight-line reference AND whose edge-drift band lies below its own reference."""
    n = len(med_hi)
    for k in range(n - 1):
        if (med_hi[k] < ref and med_hi[k + 1] < ref
                and e_hi[k] < E_LINEAR and e_hi[k + 1] < E_LINEAR):
            return (k - 1 if k > 0 else None), k
    return None, None


def main():
    cps = load_checkpoints()
    steps = [c["step"] for c in cps]
    J = np.array([r["jsd_B"] for r in cps[0]["rows"]], float)   # held-out-split corpus JSD
    n_ck = len(cps)

    # per-level, per-checkpoint per-pair width (median over the three carrier sentences)
    W = {}
    for a in LEVELS:
        M = np.full((n_ck, len(J)), np.nan)
        for k, c in enumerate(cps):
            cur = c["curves"]
            for i in range(cur.shape[0]):
                M[k, i] = np.nanmedian([width_at(cur[i, j], a) for j in range(cur.shape[1])])
        W[a] = M
        print(f"level {a:.2f}: {np.isnan(M).sum()} NaN widths of {M.size}")

    # edge drift does not depend on the level; compute it once
    E = np.full((n_ck, len(J)), np.nan)
    for k, c in enumerate(cps):
        cur = c["curves"]
        for i in range(cur.shape[0]):
            E[k, i] = np.nanmedian([curve_metrics.edge_drift(cur[i, j], GRID)
                                    for j in range(cur.shape[1])])

    idx = RNG.integers(0, len(J), size=(B, len(J)))
    out = {"steps": steps, "levels": LEVELS, "n_boot": B, "E_linear": E_LINEAR, "by_level": {}}

    # the same resampled pairs at every checkpoint, so the trajectory bootstrap stays paired
    e_med_b = np.array([[np.median(E[k][i]) for k in range(n_ck)] for i in idx])
    e_med = np.array([np.median(E[k]) for k in range(n_ck)])
    e_hi = e_med + np.percentile(np.max(np.abs(e_med_b - e_med[None, :]), axis=1), 95)

    for a in LEVELS:
        M, ref = W[a], 1 - 2 * a
        rho = np.array([sp(J, M[k]) for k in range(n_ck)])
        med = np.array([np.median(M[k]) for k in range(n_ck)])
        rho_b = np.array([[sp(J[i], M[k][i]) for k in range(n_ck)] for i in idx])
        med_b = np.array([[np.median(M[k][i]) for k in range(n_ck)] for i in idx])
        c_rho = np.percentile(np.max(np.abs(rho_b - rho[None, :]), axis=1), 95)
        c_med = np.percentile(np.max(np.abs(med_b - med[None, :]), axis=1), 95)

        ia, ib = bracket_ordering(rho - c_rho, rho + c_rho)
        ja, jb = bracket_shape(med + c_med, ref, e_hi)
        k8, k32 = steps.index(8), steps.index(32)
        k512, k1000 = steps.index(512), steps.index(1000)
        out["by_level"][f"{a:.2f}"] = dict(
            straight_line_ref=ref,
            rho=[float(x) for x in rho], rho_band=float(c_rho),
            median_w=[float(x) for x in med], median_w_band=float(c_med),
            ordering_after=None if ia is None else steps[ia],
            ordering_by=None if ib is None else steps[ib],
            shape_after=None if ja is None else steps[ja],
            shape_by=None if jb is None else steps[jb],
            rho_32=float(rho[k32]),
            rho_dw_8_32=sp(J, M[k32] - M[k8]),
            rho_dw_512_1000=sp(J, M[k1000] - M[k512]),
            median_dw_8_32=float(np.median(M[k32] - M[k8])),
            median_dw_512_1000=float(np.median(M[k1000] - M[k512])),
        )
        r = out["by_level"][f"{a:.2f}"]
        print(f"level {a:.2f} (line ref {ref:.1f}): ordering {r['ordering_after']}->"
              f"{r['ordering_by']}, shape {r['shape_after']}->{r['shape_by']}, "
              f"rho_32 {rho[k32]:+.3f}, rho_dw(8->32) {r['rho_dw_8_32']:+.3f}, "
              f"rho_dw(512->1000) {r['rho_dw_512_1000']:+.3f}")

    out["edge_drift_median"] = [float(x) for x in e_med]
    json.dump(out, open(os.path.join(RESULTS, "threshold_robustness.json"), "w"), indent=2)
    print("wrote results/threshold_robustness.json")


if __name__ == "__main__":
    main()
