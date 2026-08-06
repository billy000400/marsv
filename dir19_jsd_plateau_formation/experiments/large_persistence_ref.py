"""Does the 1,000-pair ranking clock depend on WHICH checkpoint we call final?

Result 15 dates the ranking lock-in on the 1,000-pair bank by scoring every checkpoint's per-pair
widths against step 143000's. That reference is the last released checkpoint, not a checkpoint at
which the model has stopped moving — Result 5 shows the widths still change between step 64000 and
step 143000. Result 11 already answered this on the 60-pair bank (the step 64 -> 128 bracket is the
same under five references); this is the same test on the bank where Result 15 lives.

For each reference the statistics are exactly those of large_persistence.py:

    pi_ref(s)       = Spearman(w_s, w_ref)
    pi_perp_ref(s)  = partial Spearman(w_s, w_ref | J)
    dpi_ref(s)      = pi_ref(s) - pi_ref(0)          <- the onset rule is applied to this

with the dyadic endpoint bootstrap (resample the 123 endpoint tokens, weight each pair by the
product of its two endpoints' multiplicities) and a SIMULTANEOUS 95% band over the checkpoints at
or before the reference. Bracket = (last checkpoint whose band includes zero, first of two
consecutive checkpoints whose band excludes it), searched strictly before the reference.

The same resampled endpoints are reused across references, so the step-143000 column reproduces
results/large_persistence.json exactly rather than redrawing it.

CPU only. Reads results/pair_manifest_large.json and results/assay_large_step*.json.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

from common import RESULTS
from large_persistence import N_BOOT, STEPS, partial, wpartial, wspear

REF_STEPS = [8000, 64000, 143000]
RNG = np.random.default_rng(20260806)


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
    W = {s: np.array([r["w"] for r in
                      json.load(open(os.path.join(RESULTS, f"assay_large_step{s}.json")))["rows"]],
                     dtype=float) for s in steps}
    print(f"checkpoints: {steps}   references: {REF_STEPS}")

    mult = RNG.multinomial(n_ep, np.full(n_ep, 1 / n_ep), size=N_BOOT)
    wt = (mult[:, ea] * mult[:, eb]).astype(float)

    out = {"n_boot": N_BOOT, "n_endpoints": int(n_ep), "n_pairs": int(len(J)),
           "steps": steps, "references": {}}

    for ref in REF_STEPS:
        wf = W[ref]
        sub = [s for s in steps if s <= ref]
        obs_pi, obs_pp, boot_pi = {}, {}, {}
        pp_ci = {}
        for s in sub:
            obs_pi[s] = float(np.corrcoef(rankdata(W[s]), rankdata(wf))[0, 1])
            obs_pp[s] = partial(W[s], wf, J)
            boot_pi[s] = np.array([wspear(W[s], wf, wt[b]) for b in range(N_BOOT)])
            bpp = np.array([wpartial(W[s], wf, J, wt[b]) for b in range(N_BOOT)])
            pp_ci[s] = [float(np.percentile(bpp, 2.5)), float(np.percentile(bpp, 97.5))]

        dboot = np.stack([boot_pi[s] - boot_pi[0] for s in sub], axis=1)
        dobs = np.array([obs_pi[s] - obs_pi[0] for s in sub])
        sim_half = float(np.quantile(
            np.abs(dboot - dboot.mean(axis=0, keepdims=True)).max(axis=1), 0.95))

        rows, sig = {}, []
        for i, s in enumerate(sub):
            lo, hi = np.percentile(boot_pi[s], [2.5, 97.5])
            ok = bool(dobs[i] - sim_half > 0)
            sig.append(ok)
            rows[str(s)] = dict(pi=obs_pi[s], pi_ci=[float(lo), float(hi)],
                                pi_perp=obs_pp[s], pi_perp_ci=pp_ci[s],
                                dpi=float(dobs[i]), dpi_sim_lo=float(dobs[i] - sim_half),
                                dpi_sim_hi=float(dobs[i] + sim_half), sig_simultaneous=ok)

        lim = len(sub) - 1                      # search strictly before the reference
        br = None
        for i in range(lim - 1):
            if sig[i] and sig[i + 1]:
                j = i
                while j > 0 and sig[j - 1]:
                    j -= 1
                br = [sub[j - 1] if j > 0 else None, sub[j]]
                break

        out["references"][str(ref)] = dict(steps=sub, sim_halfwidth_dpi=sim_half,
                                           bracket_dpi=br, rows=rows)
        print(f"\nref step {ref:>6}: bracket {br}  sim half-width {sim_half:.3f}")
        for s in sub:
            r = rows[str(s)]
            print(f"   step {s:>6}  pi={r['pi']:+.3f}  pi_perp={r['pi_perp']:+.3f} "
                  f"[{r['pi_perp_ci'][0]:+.3f},{r['pi_perp_ci'][1]:+.3f}]  "
                  f"dpi={r['dpi']:+.3f} [{r['dpi_sim_lo']:+.3f},{r['dpi_sim_hi']:+.3f}]"
                  f"{'  SIG' if r['sig_simultaneous'] else ''}")

    json.dump(out, open(os.path.join(RESULTS, "large_persistence_ref.json"), "w"), indent=2)
    print("\nwrote results/large_persistence_ref.json")


if __name__ == "__main__":
    main()
