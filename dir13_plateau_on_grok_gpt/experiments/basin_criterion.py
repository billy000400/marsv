"""Redefine and validate the per-character basin criterion (operator feedback #6).

The old basin fraction used the indicator [t_lo >= 0.10 (as A) or t_hi <= 0.90 (as B)]. The straight
line d(t)=t gives exactly t_lo=0.10 and t_hi=0.90, so the null curve PASSES: the statistic could not
fail. Here we replace it with a rest ratio measured against the straight line, and report the pass
rate of both criteria on a family of null curves (exact line, noisy line, untrained network,
block-11 patch) alongside the trained network.

Reads results/allpairs_raw.npz + results/allpairs_summary.json; writes results/basin_criterion.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matthew_assay import pava_isotonic, iso_crossing  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DELTAS = [0.05, 0.10, 0.20]
KAPPAS = np.round(np.arange(1.0, 5.01, 0.25), 2)
DELTA0, KAPPA0 = 0.10, 2.0


def rest_lengths(ts, d, delta):
    """(r_A, r_B): how much of the path stays within `delta` of each endpoint's output, on the
    isotonic copy. Under d(t)=t both equal delta exactly."""
    iso = pava_isotonic(d)
    t_lo = iso_crossing(ts, iso, delta)
    t_hi = iso_crossing(ts, iso, 1.0 - delta)
    r_A = t_lo if np.isfinite(t_lo) else 1.0
    r_B = 1.0 - t_hi if np.isfinite(t_hi) else 1.0
    return float(r_A), float(r_B)


def rest_table(curves, ts):
    """(n,2,len(DELTAS)) array of rest lengths: [pair, endpoint(A/B), delta]."""
    out = np.zeros((len(curves), 2, len(DELTAS)))
    for i, d in enumerate(curves):
        for k, delta in enumerate(DELTAS):
            out[i, 0, k], out[i, 1, k] = rest_lengths(ts, d, delta)
    return out


def pass_rate_vs_kappa(rest_d0):
    """Fraction of endpoints (both ends of every curve) passing R >= kappa at DELTA0."""
    R = rest_d0 / DELTA0
    return [float(np.mean(R >= k)) for k in KAPPAS]


def main():
    z = np.load(os.path.join(RES, "allpairs_raw.npz"))
    summ = json.load(open(os.path.join(RES, "allpairs_summary.json")))
    ts = z["ts"]
    chars = summ["chars"]
    V = len(chars)
    pairs = [(p["i"], p["j"]) for p in summ["final_block0"]]

    trained = np.stack([z[f"final|L0|d|{i}_{j}"] for i, j in pairs]).astype(float)
    init = np.stack([z[f"init|L0|d|{i}_{j}"] for i, j in pairs]).astype(float)

    rng = np.random.default_rng(20260812)
    line = np.tile(ts, (1, 1))
    noisy = {}
    for sigma in (0.01, 0.02, 0.05):
        y = ts[None, :] + rng.normal(0, sigma, size=(2000, len(ts)))
        y[:, 0], y[:, -1] = 0.0, 1.0                      # d(0)=0, d(1)=1 by construction
        noisy[sigma] = np.clip(y, 0.0, 1.0)

    R_tr = rest_table(trained, ts)
    R_in = rest_table(init, ts)
    R_line = rest_table(line, ts)
    R_noisy = {s: rest_table(v, ts) for s, v in noisy.items()}

    # Block-11 patch: near-linear readout reference. Only t_lo/t_hi (delta=0.10) were stored.
    dep11 = summ["depth"]["11"]
    r11 = np.array([[p["t_lo"], 1.0 - p["t_hi"]] for p in dep11 if p["t_lo"] is not None])

    k0 = DELTAS.index(DELTA0)
    groups = {
        "trained (block 0, step 30000)": R_tr[:, :, k0].ravel(),
        "untrained (step 0)": R_in[:, :, k0].ravel(),
        "block-11 patch (trained)": r11.ravel(),
        "line + noise sigma=0.01": R_noisy[0.01][:, :, k0].ravel(),
        "line + noise sigma=0.02": R_noisy[0.02][:, :, k0].ravel(),
        "line + noise sigma=0.05": R_noisy[0.05][:, :, k0].ravel(),
        "exact line d(t)=t": R_line[:, :, k0].ravel(),
    }

    out = {
        "definition": {
            "delta": DELTA0, "kappa": KAPPA0, "deltas_checked": DELTAS,
            "rest_ratio": "R = r(delta)/delta, r = path fraction within delta of the endpoint output",
            "null_value_of_R": 1.0,
            "old_criterion": "t_lo >= 0.10 (as A) or t_hi <= 0.90 (as B)",
            "new_criterion": "R(0.10) >= 2  i.e. t_lo >= 0.20 (as A), t_hi <= 0.80 (as B)",
        },
        "kappas": [float(k) for k in KAPPAS],
        "pass_rate_vs_kappa": {name: pass_rate_vs_kappa(v) for name, v in groups.items()},
        "endpoint_pass_rate": {
            name: {
                "old": float(np.mean(v >= 0.10)),
                "new": float(np.mean(v >= DELTA0 * KAPPA0)),
                "median_R": float(np.median(v / DELTA0)),
                "n": int(v.size),
            } for name, v in groups.items()
        },
    }

    # delta-robustness of the new criterion (kappa fixed at 2)
    out["delta_robustness"] = {
        f"delta={d}": {
            "trained_new": float(np.mean(R_tr[:, :, k] >= d * KAPPA0)),
            "untrained_new": float(np.mean(R_in[:, :, k] >= d * KAPPA0)),
            "trained_old_analogue": float(np.mean(R_tr[:, :, k] >= d)),
            "untrained_old_analogue": float(np.mean(R_in[:, :, k] >= d)),
        } for k, d in enumerate(DELTAS)
    }

    # per-character basin fraction under both criteria
    per_char = []
    for c in range(V):
        rows, ends = [], []
        for n, (i, j) in enumerate(pairs):
            if i == c:
                rows.append(n); ends.append(0)
            elif j == c:
                rows.append(n); ends.append(1)
        rows, ends = np.array(rows), np.array(ends)
        r_tr = R_tr[rows, ends, k0]
        r_in = R_in[rows, ends, k0]
        per_char.append({
            "char": chars[c], "idx": c, "n_partners": int(len(rows)),
            "phi_old": float(np.mean(r_tr >= 0.10)),
            "phi_new": float(np.mean(r_tr >= DELTA0 * KAPPA0)),
            "phi_new_untrained": float(np.mean(r_in >= DELTA0 * KAPPA0)),
            "median_R": float(np.median(r_tr / DELTA0)),
        })
    out["per_char"] = per_char

    pn = np.array([p["phi_new"] for p in per_char])
    po = np.array([p["phi_old"] for p in per_char])
    pu = np.array([p["phi_new_untrained"] for p in per_char])
    out["per_char_summary"] = {
        "phi_new_min": float(pn.min()), "phi_new_median": float(np.median(pn)),
        "phi_new_max": float(pn.max()), "phi_new_mean": float(pn.mean()),
        "n_char_phi_new_ge_0.5": int((pn >= 0.5).sum()),
        "n_char_phi_new_ge_0.9": int((pn >= 0.9).sum()),
        "n_char_phi_new_eq_1": int((pn >= 0.999).sum()),
        "n_char_phi_new_le_untrained": int((pn <= pu).sum()),
        "phi_old_min": float(po.min()), "phi_old_median": float(np.median(po)),
        "phi_untrained_median": float(np.median(pu)), "phi_untrained_max": float(pu.max()),
        "weakest": [per_char[i]["char"] for i in np.argsort(pn)[:6]],
        "strongest": [per_char[i]["char"] for i in np.argsort(-pn)[:6]],
    }

    # does the sharpened criterion carry signal? Rank-correlate phi_new with training frequency.
    from scipy import stats
    counts = json.load(open(os.path.join(RES, "char_freq.json")))["train_counts"]
    freq = np.array([counts[p["char"]] for p in per_char], dtype=float)
    rho, pval = stats.spearmanr(freq, pn)
    out["per_char_summary"]["spearman_phi_new_vs_train_count"] = [float(rho), float(pval)]
    out["per_char_summary"]["spearman_phi_old_vs_train_count"] = [float(v) for v in stats.spearmanr(freq, po)]
    for p in per_char:
        p["train_count"] = int(counts[p["char"]])

    json.dump(out, open(os.path.join(RES, "basin_criterion.json"), "w"), indent=1)
    print(json.dumps(out["endpoint_pass_rate"], indent=1))
    print(json.dumps(out["delta_robustness"], indent=1))
    print(json.dumps(out["per_char_summary"], indent=1))


if __name__ == "__main__":
    main()
