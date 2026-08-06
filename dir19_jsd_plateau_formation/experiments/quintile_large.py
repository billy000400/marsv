"""Does the 1,000-pair bank also need its top divergence quintile at step 32?

quintile_loo.py shows the 60-pair step-32 ordering disappears when the highest-divergence quintile
is removed and survives removal of any other. That bank has 10 to 14 pairs per quintile, so "the
graded relation is absent in the bulk" could just be its power in the middle of the range. The frozen
1,000-pair bank has ~200 pairs per quintile and deliberately fills that middle, so it is the test
that can tell those apart.

Same subsets (drop each quintile; middle three only) at step 32 and, as a mature control, at step
143000. Inference is the endpoint-label (QAP) permutation of permtest.py restricted to the subset:
the set of pairs is held fixed at what was observed, and only the correspondence between the 123
endpoint tokens and their divergences is relabelled, so pairs sharing a token stay dependent.

CPU only; reads results/pair_manifest_large.json and results/assay_large_step{32,143000}.json.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

from common import RESULTS

B = 20000
STEPS = [32, 143000]
RNG = np.random.default_rng(3100)


def rho_rows(rw, rJ):
    """Spearman between one width-rank vector and each row of a divergence-rank matrix."""
    a = rw - rw.mean()
    Bc = rJ - rJ.mean(axis=1, keepdims=True)
    return (Bc @ a) / (np.linalg.norm(Bc, axis=1) * np.linalg.norm(a) + 1e-300)


def main():
    man = json.load(open(os.path.join(RESULTS, "pair_manifest_large.json")))
    M = np.asarray(man["jsd_holdout_matrix"], dtype=float)
    pos = {e: i for i, e in enumerate(man["eligible_endpoints"])}
    ea = np.array([pos[p["ep_a"]] for p in man["pairs"]])
    eb = np.array([pos[p["ep_b"]] for p in man["pairs"]])
    J = M[ea, eb]
    n_ep = M.shape[0]

    # quintiles of the large bank's own divergence distribution
    edges = np.quantile(J, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(J, edges)

    perms = np.stack([RNG.permutation(n_ep) for _ in range(B)])
    Jnull = M[perms[:, ea], perms[:, eb]]                       # (B, 1000)

    subsets = [("all", np.ones(len(J), bool))]
    for q in range(5):
        subsets.append((f"drop_Q{q + 1}", bins != q))
    subsets.append(("middle3", (bins != 0) & (bins != 4)))

    out = {"n_perm": B, "n_pairs": int(len(J)), "n_endpoints": int(n_ep),
           "quintile_edges": [float(x) for x in edges],
           "quintile_n": [int((bins == q).sum()) for q in range(5)], "steps": STEPS, "subsets": {}}

    for s in STEPS:
        rows = json.load(open(os.path.join(RESULTS, f"assay_large_step{s}.json")))["rows"]
        w = np.array([r["w"] for r in rows])
        for name, m in subsets:
            rw = rankdata(w[m])
            obs = float(rho_rows(rw, rankdata(J[m])[None, :])[0])
            null = rho_rows(rw, rankdata(Jnull[:, m], axis=1))
            p = float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + B))
            out["subsets"].setdefault(name, {})[str(s)] = dict(
                n=int(m.sum()), rho=obs, p=p,
                null95=float(np.quantile(np.abs(null), 0.95)),
                J_lo=float(J[m].min()), J_hi=float(J[m].max()),
                median_w=float(np.median(w[m])))
            print(f"step {s:>6} {name:>9} n={m.sum():>4} J[{J[m].min():.3f},{J[m].max():.3f}] "
                  f"rho={obs:+.3f} p={p:.4f} (null95={np.quantile(np.abs(null), 0.95):.3f})")

    json.dump(out, open(os.path.join(RESULTS, "quintile_large.json"), "w"), indent=2)
    print("wrote results/quintile_large.json")


if __name__ == "__main__":
    main()
