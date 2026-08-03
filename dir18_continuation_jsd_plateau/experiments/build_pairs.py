"""S2: freeze the endpoint-pair bank BEFORE any plateau curve is seen.

Selection rules (all fixed in advance):
  * endpoint is a lowercase alphabetic word-start token in the trained model's top-K eligible word
    continuations of all three carrier contexts (K = 256 if that supplies 75 unique-endpoint pairs,
    else 512 -- see JOURNAL);
  * endpoint occurs >= 20,000 times in EACH corpus split;
  * within a pair the two endpoint corpus frequencies differ by at most a factor of two;
  * no endpoint token is reused anywhere in the primary bank (pairs are endpoint-disjoint);
  * 15 pairs in each JSD_A quintile, with corpus log-frequency and model endpoint surprisal balanced
    across the five bins.
JSD_A selects; JSD_B is the predictor used in every analysis.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import kruskal

from common import DATA, RESULTS

N_PER_BIN = 15
N_BINS = 5
MAX_FREQ_RATIO = 2.0
SEED = 0


def jsd_rows(p, i, j):
    a, b = p[i], p[j]
    m = 0.5 * (a + b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ka = np.where(a > 0, a * np.log2(a / m), 0.0).sum()
        kb = np.where(b > 0, b * np.log2(b / m), 0.0).sum()
    return 0.5 * (ka + kb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["strict", "relaxed"], default="strict",
                    help="strict = the prespecified top-256 filter; relaxed = post-hoc top-512")
    ap.add_argument("--out", default="pair_manifest.json")
    ap.add_argument("--n-per-bin", type=int, default=N_PER_BIN)
    args = ap.parse_args()
    N_PER_BIN = args.n_per_bin
    cand = json.load(open(os.path.join(RESULTS, "endpoint_candidates.json")))
    rel = json.load(open(os.path.join(RESULTS, "jsd_reliability.json")))
    bank_npz = np.load(os.path.join(DATA, "reliability_bank.npz"))
    valid = bank_npz["valid"]
    pool = np.array(sorted(cand["pool"]))
    cA = np.load(os.path.join(DATA, "counts_A.npy"))
    cB = np.load(os.path.join(DATA, "counts_B.npy"))

    tot_A = cA.sum((0, 2))
    tot_B = cB.sum((0, 2))
    strict = set(cand["candidates"])          # top-256
    relaxed = set(cand["bank_pool"])          # top-512
    use = strict if args.pool == "strict" else relaxed
    topk_used = cand["topk"] if args.pool == "strict" else cand["topk_bank"]
    ok = np.array([(pool[i] in use) and tot_A[i] >= 20_000 and tot_B[i] >= 20_000
                   for i in range(len(pool))])
    E = np.flatnonzero(ok)
    print(f"eligible endpoints: {len(E)} (top-{topk_used} pool, count >= 20k in both splits)")

    def norm(c):
        """Rows are indexed by POOL position (same as `pool`, `tot_A`, `freq`, `surp`)."""
        x = c[:, valid].astype(np.float64)
        return x / np.maximum(x.sum(1, keepdims=True), 1)

    pA = norm(cA.sum(0))
    pB = norm(cB.sum(0))
    ent = -(np.where(pB > 0, pB * np.log2(np.maximum(pB, 1e-300)), 0.0)).sum(1)

    surp = -np.mean([cand["pool_ctx_logprob"][c] for c in cand["contexts"]], axis=0) / np.log(2)
    freq = (tot_A + tot_B).astype(np.float64)
    logf = np.log10(freq)

    # every endpoint-disjoint candidate pair that passes the frequency-ratio rule
    pairs = []
    for ii in range(len(E)):
        for jj in range(ii + 1, len(E)):
            i, j = E[ii], E[jj]
            r = freq[i] / freq[j]
            if r > MAX_FREQ_RATIO or r < 1 / MAX_FREQ_RATIO:
                continue
            pairs.append((i, j))
    print(f"candidate pairs passing the frequency-ratio rule: {len(pairs)}")

    jA = np.array([jsd_rows(pA, i, j) for i, j in pairs])
    jB = np.array([jsd_rows(pB, i, j) for i, j in pairs])

    edges = np.quantile(jA, np.linspace(0, 1, N_BINS + 1))
    edges[-1] += 1e-9
    binid = np.clip(np.digitize(jA, edges[1:-1]), 0, N_BINS - 1)

    tgt_f = np.median(logf[E])
    tgt_s = np.median(surp[E])
    sd_f, sd_s = logf[E].std(), surp[E].std()

    # Round-robin over bins: with the strict top-256 pool the endpoint-disjoint constraint is tight,
    # so filling bin 0 to quota before touching bin 4 would starve the later bins.
    order = {}
    for b in range(N_BINS):
        cands = np.flatnonzero(binid == b)
        cost = np.array([abs((logf[pairs[k][0]] + logf[pairs[k][1]]) / 2 - tgt_f) / sd_f
                         + abs((surp[pairs[k][0]] + surp[pairs[k][1]]) / 2 - tgt_s) / sd_s
                         for k in cands])
        order[b] = list(cands[np.argsort(cost)])
    chosen, used = [], set()
    for _ in range(N_PER_BIN):
        for b in range(N_BINS):
            if sum(c["bin"] == b for c in chosen) >= N_PER_BIN:
                continue
            for n_k, k in enumerate(order[b]):
                i, j = pairs[k]
                if i in used or j in used:
                    continue
                used.update((i, j))
                order[b] = order[b][n_k + 1:]
                chosen.append(dict(bin=b, k=int(k), a=int(pool[i]), b_tok=int(pool[j]),
                                   jsd_A=float(jA[k]), jsd_B=float(jB[k]),
                                   a_str=cand["pool_strings"][i], b_str=cand["pool_strings"][j],
                                   count_a=int(freq[i]), count_b=int(freq[j]),
                                   surp_a=float(surp[i]), surp_b=float(surp[j]),
                                   ent_a=float(ent[i]), ent_b=float(ent[j]),
                                   strict_topk=bool(pool[i] in strict and pool[j] in strict)))
                break
    chosen.sort(key=lambda c: (c["bin"], c["jsd_A"]))
    print(f"bank: {len(chosen)} pairs, per-bin {[sum(c['bin']==b for c in chosen) for b in range(N_BINS)]}")

    lf = [[np.log10(c["count_a"] * c["count_b"]) / 2 for c in chosen if c["bin"] == b]
          for b in range(N_BINS)]
    sp = [[(c["surp_a"] + c["surp_b"]) / 2 for c in chosen if c["bin"] == b] for b in range(N_BINS)]
    p_f = kruskal(*lf).pvalue
    p_s = kruskal(*sp).pvalue
    print(f"balance across bins  log-frequency p={p_f:.3f}  surprisal p={p_s:.3f} (large = balanced)")

    rng = np.random.default_rng(SEED)
    calib = []
    for b in range(N_BINS):
        idx = [n for n, c in enumerate(chosen) if c["bin"] == b]
        calib += [int(x) for x in rng.choice(idx, 3, replace=False)]

    man = dict(
        contexts=cand["contexts"], topk_used=topk_used, seed=SEED,
        n_bins=N_BINS, n_per_bin=N_PER_BIN, max_freq_ratio=MAX_FREQ_RATIO,
        jsd_A_bin_edges=[float(x) for x in edges],
        n_eligible_endpoints=int(len(E)), n_candidate_pairs=int(len(pairs)),
        balance_kruskal_p_logfreq=float(p_f), balance_kruskal_p_surprisal=float(p_s),
        n_valid_targets=int(len(valid)),
        calibration_idx=sorted(calib), pairs=chosen,
        selection_rules=("top-K model word continuations in all 3 contexts; >=20k occurrences per "
                         "split; within-pair frequency ratio <= 2; endpoint-disjoint; 15 pairs per "
                         "JSD_A quintile balanced on log-frequency and surprisal"),
    )
    json.dump(man, open(os.path.join(RESULTS, args.out), "w"), indent=2)
    print(f"saved results/{args.out}")
