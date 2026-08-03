"""Secondary ~1000-pair bank: same prespecified endpoint rules, endpoint REUSE allowed.

The primary bank forbids endpoint reuse, which caps it at 61 pairs given 123 eligible endpoints.
To test generality we build a much larger bank from the SAME eligible endpoints and the SAME
frequency-ratio rule, dropping only endpoint-disjointness. Pairs are selected without looking at any
plateau curve: strata are quintiles of the SELECTION-split JSD, and a per-endpoint use cap stops a
few endpoints from dominating. Because endpoints recur, these pairs are NOT independent observations;
inference is endpoint-clustered (see large_analysis.py).
"""
import json
import os

import numpy as np

from build_pairs import MAX_FREQ_RATIO, jsd_rows
from common import DATA, RESULTS

N_BINS = 5
N_PER_BIN = 200          # 1000 pairs total
MAX_USE = 20             # a single endpoint may appear in at most this many pairs
SEED = 7

if __name__ == "__main__":
    cand = json.load(open(os.path.join(RESULTS, "endpoint_candidates.json")))
    valid = np.load(os.path.join(DATA, "reliability_bank.npz"))["valid"]
    pool = np.array(sorted(cand["pool"]))
    cA = np.load(os.path.join(DATA, "counts_A.npy"))
    cB = np.load(os.path.join(DATA, "counts_B.npy"))
    tot_A, tot_B = cA.sum((0, 2)), cB.sum((0, 2))
    strict = set(cand["candidates"])
    ok = np.array([(pool[i] in strict) and tot_A[i] >= 20_000 and tot_B[i] >= 20_000
                   for i in range(len(pool))])
    E = np.flatnonzero(ok)
    print(f"eligible endpoints: {len(E)}")

    def norm(c):
        x = c[:, valid].astype(np.float64)
        return x / np.maximum(x.sum(1, keepdims=True), 1)

    pA, pB = norm(cA.sum(0)), norm(cB.sum(0))
    ent = -(np.where(pB > 0, pB * np.log2(np.maximum(pB, 1e-300)), 0.0)).sum(1)
    surp = -np.mean([cand["pool_ctx_logprob"][c] for c in cand["contexts"]], axis=0) / np.log(2)
    freq = (tot_A + tot_B).astype(np.float64)

    pairs = [(E[ii], E[jj]) for ii in range(len(E)) for jj in range(ii + 1, len(E))
             if 1 / MAX_FREQ_RATIO <= freq[E[ii]] / freq[E[jj]] <= MAX_FREQ_RATIO]
    print(f"candidate pairs passing the frequency-ratio rule: {len(pairs)}")
    jA = np.array([jsd_rows(pA, i, j) for i, j in pairs])
    jB = np.array([jsd_rows(pB, i, j) for i, j in pairs])

    edges = np.quantile(jA, np.linspace(0, 1, N_BINS + 1))
    edges[-1] += 1e-9
    binid = np.clip(np.digitize(jA, edges[1:-1]), 0, N_BINS - 1)

    # Greedy fill: inside each stratum repeatedly take the still-eligible pair whose endpoints are
    # least used so far, so the endpoint distribution stays as flat as the cap allows.
    rng = np.random.default_rng(SEED)
    use = {int(i): 0 for i in E}
    chosen = []
    for b in range(N_BINS):
        cands = list(rng.permutation(np.flatnonzero(binid == b)))
        taken = 0
        while taken < N_PER_BIN:
            best, best_cost = None, None
            for k in cands:
                i, j = pairs[k]
                if use[int(i)] >= MAX_USE or use[int(j)] >= MAX_USE:
                    continue
                c = use[int(i)] + use[int(j)]
                if best_cost is None or c < best_cost:
                    best, best_cost = k, c
                    if c == 0:
                        break
            if best is None:
                print(f"  bin {b}: only {taken} pairs available under the use cap")
                break
            cands.remove(best)
            i, j = pairs[best]
            use[int(i)] += 1
            use[int(j)] += 1
            taken += 1
            chosen.append(dict(bin=b, k=int(best), a=int(pool[i]), b_tok=int(pool[j]),
                               jsd_A=float(jA[best]), jsd_B=float(jB[best]),
                               a_str=cand["pool_strings"][i], b_str=cand["pool_strings"][j],
                               ep_a=int(i), ep_b=int(j),
                               count_a=int(freq[i]), count_b=int(freq[j]),
                               surp_a=float(surp[i]), surp_b=float(surp[j]),
                               ent_a=float(ent[i]), ent_b=float(ent[j])))
    chosen.sort(key=lambda c: (c["bin"], c["jsd_A"]))
    uses = np.array([v for v in use.values() if v > 0])
    print(f"bank: {len(chosen)} pairs over {len(uses)} endpoints; "
          f"uses per endpoint min/median/max = {uses.min()}/{np.median(uses):.0f}/{uses.max()}")

    # Full endpoint x endpoint holdout-JSD matrix: the null distribution for the endpoint-label
    # permutation test needs J for pairs that were never assayed.
    Jm = np.zeros((len(E), len(E)))
    for ii in range(len(E)):
        for jj in range(ii + 1, len(E)):
            Jm[ii, jj] = Jm[jj, ii] = jsd_rows(pB, E[ii], E[jj])

    man = dict(contexts=cand["contexts"], topk_used=cand["topk"], seed=SEED, n_bins=N_BINS,
               n_per_bin=N_PER_BIN, max_use_per_endpoint=MAX_USE, max_freq_ratio=MAX_FREQ_RATIO,
               jsd_A_bin_edges=[float(x) for x in edges], n_eligible_endpoints=int(len(E)),
               n_candidate_pairs=int(len(pairs)), n_endpoints_used=int(len(uses)),
               endpoint_uses={str(k): int(v) for k, v in use.items() if v},
               eligible_endpoints=[int(x) for x in E],
               jsd_holdout_matrix=[[round(float(x), 6) for x in row] for row in Jm],
               calibration_idx=[], pairs=chosen,
               selection_rules=("identical to the primary bank except endpoint-disjointness is "
                                "replaced by a per-endpoint use cap; strata are selection-split JSD "
                                "quintiles; no plateau curve was consulted"))
    json.dump(man, open(os.path.join(RESULTS, "pair_manifest_large.json"), "w"), indent=2)
    print("saved results/pair_manifest_large.json")
