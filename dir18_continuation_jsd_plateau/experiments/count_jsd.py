"""S1: count endpoint -> successor bigrams in Pythia's released training stream and build the JSD table.

Only adjacent transitions INSIDE a 2049-token row are counted; rows are never joined.
Each 500,000-row split is counted in two halves so that same-token split-half JSD (a noise floor)
can be measured without a second pass over the data.
"""
import json
import os

import numpy as np

from common import RESULTS, DATA

SEQ = 2049
N_ROWS = 500_000
V = 50304
BLOCK = 20_000
POOL_TOPK = 1024  # counting superset; the frozen bank is drawn from a stricter subset


def count_split(name, pool_ids):
    """Return (quarter_counts[2, P, V] int32, unigram[V] int64) for one split, halves separately."""
    P = len(pool_ids)
    lut = np.full(V, -1, dtype=np.int32)
    lut[pool_ids] = np.arange(P, dtype=np.int32)
    arr = np.memmap(os.path.join(DATA, f"split{name}.bin"), dtype=np.uint16, mode="r",
                    shape=(N_ROWS, SEQ))
    counts = np.zeros((2, P, V), dtype=np.int32)
    uni = np.zeros(V, dtype=np.int64)
    for start in range(0, N_ROWS, BLOCK):
        blk = np.asarray(arr[start:start + BLOCK], dtype=np.int64)
        half = 0 if start < N_ROWS // 2 else 1
        uni += np.bincount(blk.ravel(), minlength=V)
        rows = lut[blk[:, :-1]]
        succ = blk[:, 1:]
        m = rows >= 0
        flat = rows[m].astype(np.int64) * V + succ[m]
        counts[half] += np.bincount(flat, minlength=P * V).reshape(P, V).astype(np.int32)
        if (start // BLOCK) % 5 == 0:
            print(f"  split{name} rows {start + BLOCK}/{N_ROWS}", flush=True)
    return counts, uni


def jsd_matrix(p, pairs):
    """Symmetric base-2 Jensen-Shannon divergence for a list of (i, j) row pairs of p."""
    out = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        a, b = p[i], p[j]
        m = 0.5 * (a + b)
        with np.errstate(divide="ignore", invalid="ignore"):
            ka = np.where(a > 0, a * np.log2(a / m), 0.0)
            kb = np.where(b > 0, b * np.log2(b / m), 0.0)
        out[k] = 0.5 * (ka.sum() + kb.sum())
    return out


def normalize(counts, valid):
    c = counts[:, valid].astype(np.float64)
    tot = c.sum(1, keepdims=True)
    return c / np.maximum(tot, 1), tot[:, 0]


if __name__ == "__main__":
    cand = json.load(open(os.path.join(RESULTS, "endpoint_candidates.json")))
    pool = np.array(sorted(cand["pool"]), dtype=np.int64)  # top-1024 counting superset
    print(f"pool size {len(pool)}")

    cA, uA = count_split("A", pool)
    cB, uB = count_split("B", pool)
    uni = uA + uB
    valid = np.flatnonzero(uni > 0)  # target IDs that actually occur in training
    print(f"valid target IDs: {len(valid)} / {V}")

    np.save(os.path.join(DATA, "counts_A.npy"), cA)
    np.save(os.path.join(DATA, "counts_B.npy"), cB)
    np.save(os.path.join(DATA, "unigram.npy"), np.stack([uA, uB]))

    totA = cA.sum((0, 2))
    totB = cB.sum((0, 2))
    keep = np.flatnonzero((totA >= 20_000) & (totB >= 20_000))
    print(f"endpoints with >=20,000 occurrences in BOTH splits: {len(keep)} / {len(pool)}")

    pA, nA = normalize(cA.sum(0), valid)
    pB, nB = normalize(cB.sum(0), valid)
    pA1, _ = normalize(cA[0], valid)
    pA2, _ = normalize(cA[1], valid)

    rng = np.random.default_rng(0)
    kk = keep
    pairs = set()
    while len(pairs) < 10_000 and len(kk) > 1:
        i, j = rng.choice(len(kk), 2, replace=False)
        pairs.add((min(i, j), max(i, j)))
    pairs = sorted(pairs)
    idx = [(kk[i], kk[j]) for i, j in pairs]

    jA = jsd_matrix(pA, idx)
    jB = jsd_matrix(pB, idx)
    # split-half noise floor: same token, two disjoint halves of split A
    sh = np.empty(len(kk))
    for t, k in enumerate(kk):
        a, b = pA1[k], pA2[k]
        m = 0.5 * (a + b)
        with np.errstate(divide="ignore", invalid="ignore"):
            sh[t] = 0.5 * (np.where(a > 0, a * np.log2(a / m), 0).sum()
                           + np.where(b > 0, b * np.log2(b / m), 0).sum())

    from scipy.stats import spearmanr
    rho = spearmanr(jA, jB).statistic
    ratio = float(np.median(sh) / np.median(jB))
    print(f"Spearman(JSD_A, JSD_B) = {rho:.4f}  (need >= 0.90)")
    print(f"median split-half JSD / median between-token JSD = {ratio:.4f}  (need < 0.25)")

    out = {
        "pool_size": int(len(pool)), "n_valid_targets": int(len(valid)),
        "n_endpoints_pass_count": int(len(keep)),
        "count_threshold": 20000,
        "reliability_pairs": int(len(idx)),
        "spearman_A_B": float(rho), "splithalf_ratio": ratio,
        "median_jsd_B": float(np.median(jB)), "median_splithalf_jsd": float(np.median(sh)),
        "pass": bool(rho >= 0.90 and ratio < 0.25),
        "endpoint_ids": [int(x) for x in kk],
        "endpoint_count_A": [int(x) for x in totA[kk]],
        "endpoint_count_B": [int(x) for x in totB[kk]],
    }
    json.dump(out, open(os.path.join(RESULTS, "jsd_reliability.json"), "w"), indent=2)
    np.savez(os.path.join(DATA, "reliability_bank.npz"),
             pairs=np.array(idx), jsd_A=jA, jsd_B=jB, splithalf=sh, endpoints=kk, valid=valid)
    print("saved")
