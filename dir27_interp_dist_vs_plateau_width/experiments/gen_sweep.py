"""Feedback 3 sweep: ' big' -> every other token of one model. Usage: gen_sweep.py MODEL_KEY [TOK_PER_BATCH]

Writes results/gen/<key>/shard_XXXXXX.npz (resumable) with the same fields as s2_sweep.py.
"""
import json
import os
import sys
import time

import numpy as np
import torch

from common import TS, width, y_curve
from gen_common import GEN, Runner, load

SHARD = 4096


def main():
    key = sys.argv[1]
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    out = os.path.join(GEN, key)
    os.makedirs(out, exist_ok=True)
    tok, m = load(key)
    r = Runner(tok, m)
    V = len(tok)
    t = torch.tensor(TS, device="cuda", dtype=torch.float32).view(1, -1, 1)
    ea = r.rows([r.id_a])[0]

    # check: cached last-position path vs full-prompt path on 32 random tokens x 101 steps.
    # For bf16 models the two paths differ by rounding; |dW| measures the resulting precision of W.
    chk = np.random.default_rng(0).choice(V, 32, replace=False)
    eb = r.rows(chk)
    h = ((1 - t) * ea + t * eb.unsqueeze(1)).reshape(-1, ea.shape[0])
    a = [x.clone() for x in r.run(h)]
    b = r.run_full(h)
    rel = [float(((x - y).norm(dim=-1) / y.norm(dim=-1)).max()) for x, y in zip(a, b)]
    dW = [np.abs([width(TS, u)[2] - width(TS, v)[2] for u, v in
                  zip(y_curve(x.view(32, len(TS), -1)).cpu().numpy(), y_curve(y.view(32, len(TS), -1)).cpu().numpy())])
          for x, y in zip(a, b)]
    meta = dict(model=key, n_blocks=r.n_blocks, mid_block=r.mid, final_block=r.final, vocab=V,
                prompt_ids=r.ids, token_a=tok.decode([r.id_a]), cache_check_max_rel_diff=rel,
                cache_check_abs_dW_median=[float(np.median(d)) for d in dW],
                cache_check_abs_dW_max=[float(d.max()) for d in dW])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(meta, flush=True)

    ids_all = np.array([i for i in range(V) if i != r.id_a])
    for s0 in range(0, len(ids_all), SHARD):
        path = os.path.join(out, f"shard_{s0:06d}.npz")
        if os.path.exists(path):
            continue
        ids = ids_all[s0:s0 + SHARD]
        tic = time.time()
        ym_all, yf_all, D = [], [], []
        for b0 in range(0, len(ids), per):
            eb = r.rows(ids[b0:b0 + per])
            h = (1 - t) * ea.view(1, 1, -1) + t * eb.unsqueeze(1)
            mid, fin = r.run(h.reshape(-1, h.shape[-1]))
            k = len(eb)
            ym_all.append(y_curve(mid.view(k, len(TS), -1)).cpu().numpy())
            yf_all.append(y_curve(fin.view(k, len(TS), -1)).cpu().numpy())
            D.append((eb - ea).norm(dim=-1).cpu().numpy())
        ym, yf, D = np.concatenate(ym_all), np.concatenate(yf_all), np.concatenate(D)
        gpu = time.time() - tic
        stats = np.array([width(TS, ym[i]) + width(TS, yf[i]) for i in range(len(ids))])
        np.savez(path, ids=ids, D=D, y_mid=ym.astype(np.float16), y_final=yf.astype(np.float16),
                 stats=stats)
        print(f"shard {s0}: {len(ids)} tokens, gpu {gpu:.1f}s total {time.time() - tic:.1f}s", flush=True)


if __name__ == "__main__":
    main()
