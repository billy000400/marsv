"""S1 — freeze the three disjoint context banks and the random pair banks.

Windows: non-overlapping 32-token slices of the wikitext-103 raw validation split (held out from
GPT-2's training corpus in the sense that it is a standard evaluation split; no fitting is done on
it here). One shuffle with seed 0 splits the windows into three DISJOINT pools:

    primary  pairs  (screen)      validation pairs (confirmation)      reference contexts (S5)

Within a pool, consecutive shuffled windows are paired without replacement; a pair is rejected only
if the two contexts' unpatched top-1 next tokens are identical. No filtering on semantics,
confidence, activation distance, or anything visible in the interpolation.

Writes results/manifest.json (provenance + pair lists) and results/acts.npy
([n_window, n_layer, d_model] final-position resid_post) + results/ctx.npz (token ids, top-1 ids,
top-1 probs, norms).
"""
import json
import os

import numpy as np
import torch

from common import (CTX_LEN, LAYERS, MODEL, N_ALPHA, RESULTS, REVISION, SEED, Runner,
                    build_windows)

N_PRIMARY_PAIRS = 1000
N_VALID_PAIRS = 300
N_REFERENCE = 2000
BATCH = 32


def make_pairs(idx, top1, n_target):
    """Pair consecutive shuffled indices, skipping same-prediction pairs. Returns (pairs, n_rej)."""
    pairs, rej, i = [], 0, 0
    while i + 1 < len(idx) and len(pairs) < n_target:
        a, b = int(idx[i]), int(idx[i + 1])
        if top1[a] == top1[b]:
            rej += 1
        else:
            pairs.append([a, b])
        i += 2
    return pairs, rej


def main():
    os.makedirs(RESULTS, exist_ok=True)
    run = Runner()
    n_need = 2 * (N_PRIMARY_PAIRS + N_VALID_PAIRS) + N_REFERENCE
    n_windows = int(n_need * 1.3)          # slack for same-prediction rejections
    W, prov = build_windows(run.tok, n_windows)
    n = W.shape[0]
    print(f"windows: {n} x {CTX_LEN} tokens   {prov}")

    acts = np.empty((n, len(LAYERS), run.model.config.n_embd), dtype=np.float32)
    top1 = np.empty(n, dtype=np.int64)
    top1_p = np.empty(n, dtype=np.float32)
    for i0 in range(0, n, BATCH):
        ids = torch.from_numpy(W[i0:i0 + BATCH]).long()
        rec, lg = run.forward(ids)
        for j, l in enumerate(LAYERS):
            acts[i0:i0 + BATCH, j] = rec[l].float().cpu().numpy()
        p = torch.softmax(lg.float(), -1)
        mp, mi = p.max(-1)
        top1[i0:i0 + BATCH] = mi.cpu().numpy()
        top1_p[i0:i0 + BATCH] = mp.cpu().numpy()
        if i0 % (BATCH * 20) == 0:
            print(f"  cached {i0 + ids.shape[0]}/{n}", flush=True)
    assert np.isfinite(acts).all(), "non-finite activations"

    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    n_pri, n_val = 2 * int(N_PRIMARY_PAIRS * 1.3), 2 * int(N_VALID_PAIRS * 1.3)
    pri_pool, val_pool = order[:n_pri], order[n_pri:n_pri + n_val]
    ref_pool = order[n_pri + n_val:n_pri + n_val + N_REFERENCE]
    assert len(set(pri_pool) & set(val_pool)) == 0 and len(set(pri_pool) & set(ref_pool)) == 0 \
        and len(set(val_pool) & set(ref_pool)) == 0

    pri, rej_p = make_pairs(pri_pool, top1, N_PRIMARY_PAIRS)
    val, rej_v = make_pairs(val_pool, top1, N_VALID_PAIRS)
    print(f"primary pairs {len(pri)} (rejected {rej_p} same-prediction), "
          f"validation pairs {len(val)} (rejected {rej_v}), reference {len(ref_pool)}")

    man = {"model": MODEL, "revision": REVISION, "dtype": "float32", "seed": SEED,
           "ctx_len": CTX_LEN, "layers": list(LAYERS), "n_alpha": N_ALPHA,
           "corpus": prov, "n_windows": n,
           "n_primary_pairs": len(pri), "n_validation_pairs": len(val),
           "n_reference": int(len(ref_pool)),
           "rejected_same_prediction": {"primary": rej_p, "validation": rej_v},
           "primary_pairs": pri, "validation_pairs": val,
           "reference_idx": [int(i) for i in ref_pool],
           "versions": {"torch": torch.__version__}}
    with open(os.path.join(RESULTS, "manifest.json"), "w") as f:
        json.dump(man, f)
    np.save(os.path.join(RESULTS, "acts.npy"), acts)
    np.savez(os.path.join(RESULTS, "ctx.npz"), windows=W, top1=top1, top1_p=top1_p)
    ex = pri[0]
    print("example pair:", repr(run.tok.decode(W[ex[0]])), "->", repr(run.tok.decode(W[ex[1]])))
    print("  preds:", repr(run.tok.decode([top1[ex[0]]])), repr(run.tok.decode([top1[ex[1]]])))
    print("wrote manifest.json / acts.npy / ctx.npz")


if __name__ == "__main__":
    main()
