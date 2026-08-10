"""Do the per-token width effects transfer to partners outside the pair bank?

Combines the two probes (anchor widths, basin radii) with the token effects fitted inside the
1,000-pair bank. Writes results/transfer.json.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, load
from explore1 import cv_r2

GATE = 0.2


def partial(x, y, z):
    """Spearman correlation of x and y after linearly removing z from both (rank scale)."""
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    X, Y, Z = rk(x), rk(y), np.column_stack([np.ones(len(z)), rk(z)])
    rx = X - Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    ry = Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]
    return [float(v) for v in spearmanr(rx, ry)[:2]]


def main():
    t, _, _ = load()
    m = t["out_jsd_min"] >= GATE
    tokfx = json.load(open(f"{RESULTS}/explore1.json"))["token_effects"]
    a = np.array(tokfx["effect"])
    names = tokfx["tokens"]
    AW = json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"]
    BP = json.load(open(f"{RESULTS}/basin.json"))["tokens"]

    anchor_w = {s: float(np.nanmedian(AW[s]["w"])) for s in AW}
    radius = {s: float(np.nanmedian(BP[s]["radius_anchor"]["0.2"])) for s in BP}
    rad_rand = {s: float(np.nanmedian(BP[s]["radius_random"]["0.1"])) for s in BP}
    entropy = {s: float(np.median(BP[s]["out_entropy"])) for s in BP}
    lognorm = {s: float(np.median(BP[s]["logit_norm"])) for s in BP}

    v = {k: np.array([d[s] for s in names]) for k, d in
         (("anchor_width", anchor_w), ("basin_radius_anchor", radius),
          ("basin_radius_random", rad_rand), ("output_entropy", entropy),
          ("logit_norm", lognorm))}
    tok_corr = {k: [float(x) for x in spearmanr(x, a)[:2]] for k, x in v.items()}
    tok_partial = {
        "anchor_width_given_entropy": partial(v["anchor_width"], a, v["output_entropy"]),
        "basin_radius_given_entropy": partial(v["basin_radius_anchor"], a, v["output_entropy"]),
    }

    # pair level: replace the 123 fitted token effects by two measured numbers
    y = t["w"][m]
    col = lambda d: (np.array([d[x] for x in t["a_str"]])[m]
                     + np.array([d[x] for x in t["b_str"]])[m])[:, None]
    one = np.ones((len(y), 1))
    J = t["jsd_B"][m][:, None]
    S, R, E = col(anchor_w), col(radius), col(entropy)
    models = {
        "anchor_sum": np.hstack([one, S]),
        "anchor_sum_plus_jsd": np.hstack([one, S, J]),
        "basin_radius_sum_plus_jsd": np.hstack([one, R, J]),
        "output_entropy_sum_plus_jsd": np.hstack([one, E, J]),
        "anchor_sum_plus_radius_plus_jsd": np.hstack([one, S, R, J]),
    }
    cv = {k: cv_r2(X, y)[0] for k, X in models.items()}

    out = dict(n_tokens=len(names), n_pairs=int(m.sum()),
               anchors=json.load(open(f"{RESULTS}/anchor_width.json"))["anchors"],
               anchor_width_valid_rate=float(np.mean(
                   ~np.isnan(np.array([AW[s]["w"] for s in names])))),
               token_corr=tok_corr, token_partial=tok_partial, cv_r2=cv,
               rho_anchor_sum_pair_w=[float(x) for x in spearmanr(S[:, 0], y)[:2]],
               anchor_width_pct=[float(x) for x in
                                 np.percentile([anchor_w[s] for s in names], [5, 50, 95])])
    json.dump(out, open(os.path.join(RESULTS, "transfer.json"), "w"), indent=1)
    for k, x in tok_corr.items():
        print(f"  rho(a_u, {k:22s}) = {x[0]:+.3f}  p={x[1]:.1e}")
    for k, x in tok_partial.items():
        print(f"  partial {k:34s} = {x[0]:+.3f}  p={x[1]:.1e}")
    for k, x in cv.items():
        print(f"  CV-R2 {k:32s} {x:+.3f}")


if __name__ == "__main__":
    main()
