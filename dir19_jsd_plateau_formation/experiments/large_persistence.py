"""Are the graded-ordering onset and the ranking lock-in ONE event?

Result 14 brackets the graded ordering (the 600 middle-range pairs of the 1,000-pair bank) at
"after step 32, by step 256". Result 8 brackets the ranking lock-in (agreement of the 60-pair
per-pair widths with the final ranking) at "after step 64, by step 128" — inside that window. With
only steps 32 and 256 measured on the large bank, the two could be a single event.

scan_large.py has now been run at step64 and step128, so both statistics can be evaluated on the
SAME bank at the SAME checkpoints. Here we compute, on the 1,000-pair bank at every measured
checkpoint:

  pi(s)        Spearman(w_s, w_final)                  — how much of the final ranking is present
  pi_perp(s)   the same, with rank(J) partialled out   — the part not explained by corpus divergence
  dpi(s)       pi(s) - pi(0)                           — the part acquired by training

A label permutation is not defined for pi: both variables are widths, there are no labels to
relabel. Inference is therefore the dyadic endpoint bootstrap used everywhere else on this bank
(resample the 123 endpoint tokens, weight each pair by the product of its two endpoints' multi-
plicities), which respects the fact that pairs sharing an endpoint token are dependent. Because
pi(0) is already positive for reasons that have nothing to do with training, the onset rule is
applied to dpi with a SIMULTANEOUS band: the same resampled endpoints are used at every checkpoint
and the band is the 95th percentile of the maximum absolute deviation across checkpoints.

CPU only. Reads results/pair_manifest_large.json and results/assay_large_step*.json.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

from common import RESULTS

N_BOOT = 2000
STEPS = [0, 8, 32, 64, 128, 256, 1000, 8000, 64000, 143000]
REF = 143000
RNG = np.random.default_rng(20260806)


def weighted_rank(x, wt):
    o = np.argsort(x, kind="mergesort")
    cw = np.cumsum(wt[o])
    r = np.empty_like(x, dtype=float)
    r[o] = cw - (wt[o] - 1) / 2.0
    return r


def wcorr(rx, ry, wt):
    mx, my = np.average(rx, weights=wt), np.average(ry, weights=wt)
    cov = np.average((rx - mx) * (ry - my), weights=wt)
    return cov / np.sqrt(np.average((rx - mx) ** 2, weights=wt)
                         * np.average((ry - my) ** 2, weights=wt) + 1e-300)


def wpartial(x, y, z, wt):
    """Weighted partial Spearman of x and y given z (all residualised on rank z)."""
    k = wt > 0
    x, y, z, wt = x[k], y[k], z[k], wt[k]
    rx, ry, rz = weighted_rank(x, wt), weighted_rank(y, wt), weighted_rank(z, wt)
    out = []
    for r in (rx, ry):
        m = np.average(r, weights=wt)
        mz = np.average(rz, weights=wt)
        b = (np.average((r - m) * (rz - mz), weights=wt)
             / (np.average((rz - mz) ** 2, weights=wt) + 1e-300))
        out.append(r - m - b * (rz - mz))
    return wcorr(out[0], out[1], wt)


def wspear(x, y, wt):
    k = wt > 0
    x, y, wt = x[k], y[k], wt[k]
    return wcorr(weighted_rank(x, wt), weighted_rank(y, wt), wt)


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = []
    for r in (rx, ry):
        b = np.cov(r, rz)[0, 1] / np.var(rz)
        res.append(r - r.mean() - b * (rz - rz.mean()))
    return float(np.corrcoef(res[0], res[1])[0, 1])


def main():
    man = json.load(open(os.path.join(RESULTS, "pair_manifest_large.json")))
    M = np.asarray(man["jsd_holdout_matrix"], dtype=float)
    pos = {e: i for i, e in enumerate(man["eligible_endpoints"])}
    ea = np.array([pos[p["ep_a"]] for p in man["pairs"]])
    eb = np.array([pos[p["ep_b"]] for p in man["pairs"]])
    J = M[ea, eb]
    n_ep = M.shape[0]

    steps = [s for s in STEPS
             if os.path.exists(os.path.join(RESULTS, f"assay_large_step{s}.json"))]
    W = {}
    for s in steps:
        rows = json.load(open(os.path.join(RESULTS, f"assay_large_step{s}.json")))["rows"]
        W[s] = np.array([r["w"] for r in rows], dtype=float)
    wf = W[REF]
    print(f"checkpoints: {steps}  (reference {REF})")

    mult = RNG.multinomial(n_ep, np.full(n_ep, 1 / n_ep), size=N_BOOT)
    wt = (mult[:, ea] * mult[:, eb]).astype(float)

    obs_pi, obs_pp, boot_pi, boot_pp = {}, {}, {}, {}
    for s in steps:
        obs_pi[s] = float(np.corrcoef(rankdata(W[s]), rankdata(wf))[0, 1])
        obs_pp[s] = partial(W[s], wf, J)
        boot_pi[s] = np.array([wspear(W[s], wf, wt[b]) for b in range(N_BOOT)])
        boot_pp[s] = np.array([wpartial(W[s], wf, J, wt[b]) for b in range(N_BOOT)])

    # simultaneous band for dpi(s) = pi(s) - pi(0): same resampled endpoints at every checkpoint
    dboot = np.stack([boot_pi[s] - boot_pi[0] for s in steps], axis=1)     # (N_BOOT, n_steps)
    dobs = np.array([obs_pi[s] - obs_pi[0] for s in steps])
    dev = np.abs(dboot - dboot.mean(axis=0, keepdims=True)).max(axis=1)
    sim_half = float(np.quantile(dev, 0.95))

    out = {"n_boot": N_BOOT, "n_endpoints": int(n_ep), "n_pairs": int(len(J)),
           "reference_step": REF, "steps": steps, "sim_halfwidth_dpi": sim_half, "rows": {}}
    for i, s in enumerate(steps):
        lo_p, hi_p = np.percentile(boot_pi[s], [2.5, 97.5])
        lo_q, hi_q = np.percentile(boot_pp[s], [2.5, 97.5])
        out["rows"][str(s)] = dict(
            pi=obs_pi[s], pi_ci=[float(lo_p), float(hi_p)],
            pi_perp=obs_pp[s], pi_perp_ci=[float(lo_q), float(hi_q)],
            dpi=float(dobs[i]), dpi_sim_lo=float(dobs[i] - sim_half),
            dpi_sim_hi=float(dobs[i] + sim_half),
            sig_simultaneous=bool(dobs[i] - sim_half > 0))

    sig = [out["rows"][str(s)]["sig_simultaneous"] for s in steps]
    br = None
    for i in range(len(steps) - 1):
        if sig[i] and sig[i + 1]:
            j = i
            while j > 0 and sig[j - 1]:
                j -= 1
            br = (steps[j - 1] if j > 0 else None, steps[j])
            break
    out["bracket_dpi"] = br
    print(f"\nsimultaneous half-width for dpi = {sim_half:.3f};  bracket = {br}")
    for s in steps:
        r = out["rows"][str(s)]
        print(f"  step {s:>6}  pi={r['pi']:+.3f} [{r['pi_ci'][0]:+.3f},{r['pi_ci'][1]:+.3f}]  "
              f"pi_perp={r['pi_perp']:+.3f} [{r['pi_perp_ci'][0]:+.3f},{r['pi_perp_ci'][1]:+.3f}]  "
              f"dpi={r['dpi']:+.3f} [{r['dpi_sim_lo']:+.3f},{r['dpi_sim_hi']:+.3f}]"
              f"{'  SIG' if r['sig_simultaneous'] else ''}")

    json.dump(out, open(os.path.join(RESULTS, "large_persistence.json"), "w"), indent=2)
    print("\nwrote results/large_persistence.json")


if __name__ == "__main__":
    main()
