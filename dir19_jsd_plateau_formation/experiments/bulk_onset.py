"""When does the GRADED ordering appear — as opposed to the top-quintile separation?

Result 13 showed the step-32 ordering is carried entirely by the highest-divergence quintile: on
the 1,000-pair bank the 600 middle-range pairs give rho = -0.055 (p = 0.35) at step 32 and -0.300
(p < 0.0001) at step 143000. That leaves the obvious question unanswered — the bulk relation is
absent early and present at the end, but WHEN does it arrive, and is that the ordering clock
(step 8 -> 32) or the plateau-shape clock (step 1000 -> 2000)?

Answering it needs the large bank at intermediate checkpoints, so scan_large.py was run at step256,
step1000 and step8000, giving eight measured checkpoints. Here we run the primary ordering rule
separately on three nested subsets of that bank:

  all       all 1,000 pairs
  middle3   the 600 pairs in divergence quintiles 2-4 (the crowded middle of the range)
  topQ      the 200 pairs in quintile 5 (the most distinguishable pairs)

Inference matches the rest of the large-bank analysis: significance from the endpoint-label (QAP)
permutation, in which the pairs are held fixed and only the correspondence between the 123 endpoint
tokens and their divergences is relabelled, so pairs sharing a token stay dependent; uncertainty
from the dyadic endpoint bootstrap. The same permutation draws are used at every checkpoint, so a
SIMULTANEOUS envelope over the eight checkpoints is available and is what the onset rule uses.

CPU only. Reads results/pair_manifest_large.json and results/assay_large_step*.json.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata, spearmanr

from common import RESULTS

B = 20000
N_BOOT = 2000
STEPS = [0, 8, 32, 64, 128, 256, 1000, 8000, 64000, 143000]
RNG = np.random.default_rng(4242)


def rho_rows(rw, rJ):
    """Spearman between one width-rank vector and each row of a divergence-rank matrix."""
    a = rw - rw.mean()
    Bc = rJ - rJ.mean(axis=1, keepdims=True)
    return (Bc @ a) / (np.linalg.norm(Bc, axis=1) * np.linalg.norm(a) + 1e-300)


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


def bracket(steps, sig):
    """Prespecified onset rule: after the last non-significant checkpoint, by the first of two
    consecutive significant ones. Returns (open, close) or None."""
    for i in range(len(steps) - 1):
        if sig[i] and sig[i + 1]:
            j = i
            while j > 0 and sig[j - 1]:
                j -= 1
            return (steps[j - 1] if j > 0 else None, steps[j])
    return None


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
    print(f"checkpoints: {steps}")

    edges = np.quantile(J, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(J, edges)
    subsets = [("all", np.ones(len(J), bool)),
               ("middle3", (bins != 0) & (bins != 4)),
               ("topQ", bins == 4)]

    perms = np.stack([RNG.permutation(n_ep) for _ in range(B)])
    Jnull = M[perms[:, ea], perms[:, eb]]                        # (B, 1000)
    mult = RNG.multinomial(n_ep, np.full(n_ep, 1 / n_ep), size=N_BOOT)
    wt = (mult[:, ea] * mult[:, eb]).astype(float)               # dyadic pair weights

    out = {"n_perm": B, "n_boot": N_BOOT, "n_endpoints": int(n_ep), "steps": steps,
           "quintile_edges": [float(x) for x in edges], "subsets": {}}

    for name, m in subsets:
        rJ_obs = rankdata(J[m])[None, :]
        rJ_null = rankdata(Jnull[:, m], axis=1)
        rows, null_by_step = {}, []
        for s in steps:
            rw = rankdata(W[s][m])
            obs = float(rho_rows(rw, rJ_obs)[0])
            null = rho_rows(rw, rJ_null)
            null_by_step.append(np.abs(null))
            ci = np.percentile([wspearman(J[m], W[s][m], wt[b][m]) for b in range(N_BOOT)],
                               [2.5, 97.5])
            rows[str(s)] = dict(
                rho=obs, p=float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + B)),
                ci=[float(ci[0]), float(ci[1])],
                null95_pointwise=float(np.quantile(np.abs(null), 0.95)),
                median_w=float(np.median(W[s][m])))
        sim95 = float(np.quantile(np.stack(null_by_step, 1).max(axis=1), 0.95))
        sig = [rows[str(s)]["rho"] < 0 and abs(rows[str(s)]["rho"]) > sim95 for s in steps]
        # family-wise p: max |rho| over checkpoints under each permutation
        maxnull = np.stack(null_by_step, 1).max(axis=1)
        for i, s in enumerate(steps):
            rows[str(s)]["p_fw"] = float(
                (1 + np.sum(maxnull >= abs(rows[str(s)]["rho"]) - 1e-12)) / (1 + B))
            rows[str(s)]["sig_simultaneous"] = bool(sig[i])
        br = bracket(steps, sig)
        out["subsets"][name] = dict(n=int(m.sum()), J_lo=float(J[m].min()), J_hi=float(J[m].max()),
                                    sim95=sim95, bracket=br, steps=rows)
        print(f"\n== {name}  n={m.sum()}  J[{J[m].min():.3f},{J[m].max():.3f}]  "
              f"simultaneous null95={sim95:.3f}  bracket={br}")
        for s in steps:
            r = rows[str(s)]
            print(f"   step {s:>6}  rho={r['rho']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  "
                  f"p={r['p']:.4f}  p_fw={r['p_fw']:.4f}  {'SIG' if r['sig_simultaneous'] else ''}")

    # difference between the two subsets at each checkpoint, on the same bootstrap resamples
    diff = {}
    mm, mt = subsets[1][1], subsets[2][1]
    for s in steps:
        d = np.array([wspearman(J[mm], W[s][mm], wt[b][mm])
                      - wspearman(J[mt], W[s][mt], wt[b][mt]) for b in range(N_BOOT)])
        obs = (spearmanr(J[mm], W[s][mm]).statistic - spearmanr(J[mt], W[s][mt]).statistic)
        diff[str(s)] = dict(obs=float(obs),
                            ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        print(f"middle3 - topQ  step {s:>6}: {obs:+.3f} "
              f"[{diff[str(s)]['ci'][0]:+.3f},{diff[str(s)]['ci'][1]:+.3f}]")
    out["middle_minus_top"] = diff

    # Group-level separation: is the top quintile sharper than the rest as a GROUP? A within-subset
    # correlation cannot see that, and it is what "the ordering starts as a top-quintile effect"
    # actually asserts. Null: relabel endpoint tokens, so group membership moves with J.
    top, rest = subsets[2][1], ~subsets[2][1]
    Bg = 5000
    top_null = Jnull[:Bg] >= edges[3]
    gap = {}
    for s in steps:
        w = W[s]
        obs = float(np.median(w[top]) - np.median(w[rest]))
        db = np.array([wmedian(w[top], wt[b][top]) - wmedian(w[rest], wt[b][rest])
                       for b in range(N_BOOT)])
        null = np.array([np.median(w[top_null[b]]) - np.median(w[~top_null[b]])
                         for b in range(Bg)])
        gap[str(s)] = dict(obs=obs, ci=[float(np.percentile(db, 2.5)),
                                        float(np.percentile(db, 97.5))],
                           p=float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + Bg)),
                           median_w_top=float(np.median(w[top])),
                           median_w_rest=float(np.median(w[rest])))
        print(f"topQ - rest median w  step {s:>6}: {obs:+.4f} "
              f"[{gap[str(s)]['ci'][0]:+.4f},{gap[str(s)]['ci'][1]:+.4f}]  p={gap[str(s)]['p']:.4f}")
    out["group_gap"] = gap

    json.dump(out, open(os.path.join(RESULTS, "bulk_onset.json"), "w"), indent=2)
    print("\nwrote results/bulk_onset.json")


if __name__ == "__main__":
    main()
