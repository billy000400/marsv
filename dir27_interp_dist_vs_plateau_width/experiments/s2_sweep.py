"""S2: ' big' -> every other GPT-2 token. Saves D, W_mid, W_final, flags and all y(t) curves.

Resumable: writes results/sweep/shard_XXXXX.npz per chunk of token ids.
"""
import os
import sys
import time

import numpy as np
import torch

from common import RESULTS, TS, Runner, load, width, y_curve

TOK_PER_BATCH = 8          # 8 tokens x 101 t-points = 808 sequences per forward
SHARD = 2048
OUT = os.path.join(RESULTS, "sweep")
os.makedirs(OUT, exist_ok=True)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    tok, m, dev = load()
    r = Runner(tok, m, dev)
    V = m.transformer.wte.weight.shape[0]
    ids_all = np.array([i for i in range(V) if i != r.id_a])
    if limit:
        ids_all = ids_all[:limit]
    t = torch.tensor(TS, device=dev, dtype=torch.float32).view(1, -1, 1)
    ea = r.wte[r.id_a]
    for s0 in range(0, len(ids_all), SHARD):
        path = os.path.join(OUT, f"shard_{s0:05d}.npz")
        if os.path.exists(path) and not limit:
            continue
        ids = ids_all[s0:s0 + SHARD]
        tic = time.time()
        ym_all, yf_all, D = [], [], []
        for b0 in range(0, len(ids), TOK_PER_BATCH):
            ib = torch.tensor(ids[b0:b0 + TOK_PER_BATCH], device=dev)
            eb = r.wte[ib]                                        # (k, d)
            h = (1 - t) * ea.view(1, 1, -1) + t * eb.unsqueeze(1)  # (k, 101, d)
            mid, fin = r.run(h.reshape(-1, h.shape[-1]))
            k = len(ib)
            ym_all.append(y_curve(mid.view(k, len(TS), -1)).cpu().numpy())
            yf_all.append(y_curve(fin.view(k, len(TS), -1)).cpu().numpy())
            D.append((eb - ea).norm(dim=-1).cpu().numpy())
        ym, yf, D = np.concatenate(ym_all), np.concatenate(yf_all), np.concatenate(D)
        stats = np.array([width(TS, ym[i]) + width(TS, yf[i]) for i in range(len(ids))])
        np.savez(path, ids=ids, D=D, y_mid=ym.astype(np.float16), y_final=yf.astype(np.float16),
                 stats=stats)
        print(f"shard {s0}: {len(ids)} tokens in {time.time() - tic:.1f}s", flush=True)


if __name__ == "__main__":
    main()
