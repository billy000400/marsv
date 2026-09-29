"""' big' -> every other GPT-2 token under a fixed prefix, using Direction 27's code unchanged.

Usage: sweep.py CONTEXT_ID [LIMIT]
CONTEXT_ID in {control, s1, s2, s3, s4}. `control` = Direction 27's prompt "The house was big".
Only the prompt changes: Runner, TS (101-point alpha grid), y_curve and width are imported from dir27.
Writes results/<ctx>/shard_XXXXX.npz (resumable; same fields as dir27 s2_sweep.py).
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "dir27_interp_dist_vs_plateau_width", "experiments"))
import common as d27  # noqa: E402

PROMPTS = {"control": "The house was big", "s1": "My house is big",
           "s2": "The opposite of small is big", "s3": "He dreamed big", "s4": "The elephant was big"}
TOK_PER_BATCH = 16
SHARD = 2048


def main():
    ctx = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    d27.PROMPT = PROMPTS[ctx]          # Runner reads the module-level PROMPT at construction
    tok, m, dev = d27.load()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.45)
    r = d27.Runner(tok, m, dev)
    assert tok.decode([r.id_a]) == " big", r.ids
    print(ctx, repr(d27.PROMPT), r.ids, flush=True)
    out = os.path.join(ROOT, "results", ctx + (f"_limit{limit}" if limit else ""))
    os.makedirs(out, exist_ok=True)
    V = m.transformer.wte.weight.shape[0]
    ids_all = np.array([i for i in range(V) if i != r.id_a])
    if limit:
        ids_all = ids_all[:limit]
    TS = d27.TS
    t = torch.tensor(TS, device=dev, dtype=torch.float32).view(1, -1, 1)
    ea = r.wte[r.id_a]
    for s0 in range(0, len(ids_all), SHARD):
        path = os.path.join(out, f"shard_{s0:05d}.npz")
        if os.path.exists(path):
            continue
        ids = ids_all[s0:s0 + SHARD]
        tic = time.time()
        ym_all, yf_all, D = [], [], []
        for b0 in range(0, len(ids), TOK_PER_BATCH):
            ib = torch.tensor(ids[b0:b0 + TOK_PER_BATCH], device=dev)
            eb = r.wte[ib]
            h = (1 - t) * ea.view(1, 1, -1) + t * eb.unsqueeze(1)
            mid, fin = r.run(h.reshape(-1, h.shape[-1]))
            k = len(ib)
            ym_all.append(d27.y_curve(mid.view(k, len(TS), -1)).cpu().numpy())
            yf_all.append(d27.y_curve(fin.view(k, len(TS), -1)).cpu().numpy())
            D.append((eb - ea).norm(dim=-1).cpu().numpy())
        ym, yf, D = np.concatenate(ym_all), np.concatenate(yf_all), np.concatenate(D)
        stats = np.array([d27.width(TS, ym[i]) + d27.width(TS, yf[i]) for i in range(len(ids))])
        np.savez(path, ids=ids, D=D, y_mid=ym.astype(np.float16), y_final=yf.astype(np.float16),
                 stats=stats, prompt=d27.PROMPT, prompt_ids=np.array(r.ids))
        print(f"shard {s0}: {len(ids)} tokens in {time.time() - tic:.1f}s", flush=True)


if __name__ == "__main__":
    main()
