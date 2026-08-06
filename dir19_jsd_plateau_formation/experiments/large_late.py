"""S5: does the 64k -> final WIDENING reproduce on the 1,000-pair bank?

The 60-pair set shows median w rising from 0.512 at step 64000 to 0.541 at step 143000. A median
over 60 pairs is easy to move by chance, so the prespecified follow-up re-tests it on the frozen
1,000-pair bank at the same two checkpoints.

Endpoint tokens RECUR across pairs in this bank, so pairs are not independent. Uncertainty comes
from a dyadic endpoint bootstrap: resample the endpoint tokens with replacement and weight each
pair by the product of its two endpoints' multiplicities. The same weights give the weighted
Spearman rho(J_hold, w), so the ordering result is on the same footing.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from common import RESULTS

N_BOOT = 4000
RNG = np.random.default_rng(11)


def weighted_rank(x, wt):
    o = np.argsort(x, kind="mergesort")
    cw = np.cumsum(wt[o])
    r = np.empty_like(x, dtype=float)
    r[o] = cw - (wt[o] - 1) / 2.0
    return r


def wspearman(x, y, wt):
    k = wt > 0
    x, y, wt = x[k], y[k], wt[k]
    rx, ry = weighted_rank(x, wt), weighted_rank(y, wt)
    mx, my = np.average(rx, weights=wt), np.average(ry, weights=wt)
    cov = np.average((rx - mx) * (ry - my), weights=wt)
    return cov / np.sqrt(np.average((rx - mx) ** 2, weights=wt)
                         * np.average((ry - my) ** 2, weights=wt))


def wmedian(x, wt):
    k = wt > 0
    x, wt = x[k], wt[k]
    o = np.argsort(x)
    c = np.cumsum(wt[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def main():
    man = json.load(open(os.path.join(RESULTS, "pair_manifest_large.json")))
    eps = sorted({p["ep_a"] for p in man["pairs"]} | {p["ep_b"] for p in man["pairs"]})
    pos = {e: i for i, e in enumerate(eps)}
    ea = np.array([pos[p["ep_a"]] for p in man["pairs"]])
    eb = np.array([pos[p["ep_b"]] for p in man["pairs"]])

    W, J = {}, None
    tags = ["large_step0", "large_step64000", "large_step143000"]
    tags += [t for t in ("large_step8", "large_step32")          # ordering-onset bracket validation
             if os.path.exists(os.path.join(RESULTS, f"assay_{t}.json"))]
    for tag in tags:
        a = json.load(open(os.path.join(RESULTS, f"assay_{tag}.json")))
        W[tag] = np.array([r["w"] for r in a["rows"]], dtype=float)
        if J is None:
            J = np.array([r["jsd_B"] for r in a["rows"]], dtype=float)
    d = W["large_step143000"] - W["large_step64000"]
    ok = np.isfinite(d)

    mult = RNG.multinomial(len(eps), np.full(len(eps), 1 / len(eps)), size=N_BOOT)
    wt = mult[:, ea] * mult[:, eb]                       # dyadic weight of each pair per resample

    boot_med = np.array([wmedian(d[ok], wt[b][ok]) for b in range(N_BOOT)])
    boot_frac = np.array([np.average((d[ok] > 0).astype(float), weights=wt[b][ok])
                          for b in range(N_BOOT)])
    rho = {t: float(spearmanr(J, W[t], nan_policy="omit").statistic) for t in W}
    rho_ci = {t: [float(np.percentile([wspearman(J, W[t], wt[b].astype(float))
                                       for b in range(500)], q)) for q in (2.5, 97.5)] for t in W}

    # the same 64k -> final comparison on the 60-pair controlled set, bootstrapped over pairs
    z = np.load(os.path.join(RESULTS, "per_pair_trajectories.npz"))
    st = list(z["steps"])
    ds = z["W"][st.index(143000)] - z["W"][st.index(64000)]
    oks = np.isfinite(ds)
    bi = RNG.integers(0, int(oks.sum()), size=(N_BOOT, int(oks.sum())))
    sm = np.median(ds[oks][bi], axis=1)

    out = dict(
        n_pairs=int(len(d)), n_endpoints=len(eps), n_valid=int(ok.sum()), n_boot=N_BOOT,
        small_median_dw=float(np.median(ds[oks])),
        small_median_dw_ci=[float(np.percentile(sm, 2.5)), float(np.percentile(sm, 97.5))],
        small_frac_blunter=float((ds[oks] > 0).mean()), small_n=int(oks.sum()),
        small_wilcoxon_p=float(wilcoxon(ds[oks]).pvalue),
        median_w={t: float(np.nanmedian(W[t])) for t in W},
        median_dw_64k_to_final=float(np.nanmedian(d[ok])),
        median_dw_ci=[float(np.percentile(boot_med, 2.5)), float(np.percentile(boot_med, 97.5))],
        frac_blunter=float((d[ok] > 0).mean()),
        frac_blunter_ci=[float(np.percentile(boot_frac, 2.5)),
                         float(np.percentile(boot_frac, 97.5))],
        wilcoxon_p_ignoring_endpoint_reuse=float(wilcoxon(d[ok]).pvalue),
        rho=rho, rho_endpoint_ci=rho_ci,
        reproduces=bool(np.percentile(boot_med, 2.5) > 0),
    )
    json.dump(out, open(os.path.join(RESULTS, "large_late.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
