"""Corpus next-token JSD for the three reference endpoints ` big`, ` large`, ` in`.

Feedback #4 asks whether Pythia plateaus on "My house is big/large" but not on "My house is
big/in". Those three tokens are not all in the frozen 123-endpoint bank, so this script recounts
their successors in the SAME two 500,000-row splits used everywhere else (selection = split A,
holdout = split B) and reports the pair JSDs on the same scale as the frozen bank.

Only 3 endpoint rows are tracked, so the count table is tiny; the unigram pass rebuilds the same
`valid` target-ID list (IDs that occur at all in the sample) that the assay restricts logits to.
"""
import json
import os

import numpy as np

from common import DATA, RESULTS

SEQ = 2049
N_ROWS = 500_000
V = 50304
BLOCK = 10_000
REF_WORDS = [" big", " large", " in"]


def count_split(name, ref_ids):
    """(counts[2, 3, V] int64 by split-half, unigram[V] int64) for one split."""
    lut = np.full(V, -1, dtype=np.int64)
    lut[ref_ids] = np.arange(len(ref_ids))
    arr = np.memmap(os.path.join(DATA, f"split{name}.bin"), dtype=np.uint16, mode="r",
                    shape=(N_ROWS, SEQ))
    counts = np.zeros((2, len(ref_ids), V), dtype=np.int64)
    uni = np.zeros(V, dtype=np.int64)
    for start in range(0, N_ROWS, BLOCK):
        blk = np.asarray(arr[start:start + BLOCK], dtype=np.int64)
        half = 0 if start < N_ROWS // 2 else 1
        uni += np.bincount(blk.ravel(), minlength=V)
        rows = lut[blk[:, :-1]]
        succ = blk[:, 1:]
        m = rows >= 0
        flat = rows[m] * V + succ[m]
        counts[half] += np.bincount(flat, minlength=len(ref_ids) * V).reshape(len(ref_ids), V)
        if (start // BLOCK) % 10 == 0:
            print(f"  split{name} rows {start + BLOCK}/{N_ROWS}", flush=True)
    return counts, uni


def jsd(a, b):
    """Symmetric base-2 Jensen-Shannon divergence between two probability vectors."""
    m = 0.5 * (a + b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ka = np.where(a > 0, a * np.log2(a / m), 0.0).sum()
        kb = np.where(b > 0, b * np.log2(b / m), 0.0).sum()
    return float(0.5 * (ka + kb))


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped", revision="step143000")
    ref_ids = [tok(w)["input_ids"][0] for w in REF_WORDS]
    assert all(len(tok(w)["input_ids"]) == 1 for w in REF_WORDS), "reference endpoints must be single tokens"
    print(dict(zip(REF_WORDS, ref_ids)))

    cA, uA = count_split("A", ref_ids)
    cB, uB = count_split("B", ref_ids)
    valid = np.flatnonzero(uA + uB > 0)
    np.save(os.path.join(DATA, "reference_valid.npy"), valid)
    print(f"valid target IDs: {len(valid)}")

    def probs(c):
        x = c[:, valid].astype(np.float64)
        return x / np.maximum(x.sum(1, keepdims=True), 1)

    pA, pB = probs(cA.sum(0)), probs(cB.sum(0))
    pA1, pA2 = probs(cA[0]), probs(cA[1])
    i = {w: k for k, w in enumerate(REF_WORDS)}

    out = {
        "endpoint_ids": dict(zip(REF_WORDS, [int(x) for x in ref_ids])),
        "counts_selection": {w: int(cA.sum(0)[i[w]].sum()) for w in REF_WORDS},
        "counts_holdout": {w: int(cB.sum(0)[i[w]].sum()) for w in REF_WORDS},
        "n_valid_targets": int(len(valid)),
        "pairs": {},
        "splithalf_noise": {w: jsd(pA1[i[w]], pA2[i[w]]) for w in REF_WORDS},
    }
    for a, b in [(" big", " large"), (" big", " in")]:
        out["pairs"][f"{a}|{b}"] = {
            "jsd_selection": jsd(pA[i[a]], pA[i[b]]),
            "jsd_holdout": jsd(pB[i[a]], pB[i[b]]),
        }
    json.dump(out, open(os.path.join(RESULTS, "reference_jsd.json"), "w"), indent=2)
    print(json.dumps(out["pairs"], indent=2))
    print(json.dumps(out["splithalf_noise"], indent=2))
