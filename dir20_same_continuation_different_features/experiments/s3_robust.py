"""Post-hoc robustness of the S3 effect to the residual confound imbalance.

The matched bank balances successor JSD, block-0 log norm ratio and mean surprisal almost
exactly, but leaves the high-F member with a slightly larger final-logit distance and block-0
angle (standardized mean differences +0.23 and +0.25). Both point the same way as the
hypothesis, so this script asks whether the paired effect survives when that residual is
removed: (a) inside the subsets where the high-F member is NOT favoured on each confound,
and (b) as the intercept of a linear model of Delta w on the paired confound differences.

Declared post-hoc: the primary S3 result is unchanged by anything here.
Writes results/s3_robust.json.
"""
import json
import os

import numpy as np

from common import RESULTS

CONF = ["jsd", "logit_dist", "angle0", "lognorm_ratio", "mean_surprisal"]
SEED = 31
N_BOOT = 10000


def boot_ci(x, rng):
    n = len(x)
    b = np.array([np.median(x[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    s3 = json.load(open(os.path.join(RESULTS, "matched_metrics.json")))
    rows = s3["contrasts"]
    rng = np.random.default_rng(SEED)
    dw = np.array([r["dw"] for r in rows])
    dc = {c: np.array([r[f"{c}_high"] - r[f"{c}_low"] for r in rows]) for c in CONF}

    out = {"full": dict(n=len(dw), median=float(np.median(dw)), ci95=boot_ci(dw, rng),
                        frac=float((dw < 0).mean()))}

    subsets = {
        "logit_dist_not_favoured": dc["logit_dist"] <= 0,
        "angle0_not_favoured": dc["angle0"] <= 0,
        "both_not_favoured": (dc["logit_dist"] <= 0) & (dc["angle0"] <= 0),
    }
    for k, m in subsets.items():
        x = dw[m]
        out[k] = dict(n=int(m.sum()), median=float(np.median(x)), ci95=boot_ci(x, rng),
                      frac=float((x < 0).mean()))

    # Covariate adjustment: Delta w ~ 1 + paired confound differences. The intercept is the
    # predicted effect for a contrast whose confounds are matched exactly.
    X = np.column_stack([np.ones(len(dw))] + [dc[c] for c in CONF])
    beta, *_ = np.linalg.lstsq(X, dw, rcond=None)
    resid = dw - X @ beta
    dof = len(dw) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    out["adjusted"] = dict(
        intercept=float(beta[0]), se=float(se[0]),
        ci95=[float(beta[0] - 1.96 * se[0]), float(beta[0] + 1.96 * se[0])],
        coefs={c: [float(beta[i + 1]), float(se[i + 1])] for i, c in enumerate(CONF)},
        r2=float(1 - resid.var() / dw.var()))

    with open(os.path.join(RESULTS, "s3_robust.json"), "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        if k == "adjusted":
            print(f"adjusted intercept {v['intercept']:+.4f} +- {v['se']:.4f} "
                  f"CI={[round(x, 4) for x in v['ci95']]} R2={v['r2']:.3f}")
        else:
            print(f"{k:26s} n={v['n']:3d} median={v['median']:+.4f} "
                  f"CI={[round(x, 4) for x in v['ci95']]} frac<0={v['frac']:.3f}")
    return out


if __name__ == "__main__":
    main()
