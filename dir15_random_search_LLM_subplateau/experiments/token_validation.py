"""Disjoint confirmation bank for the same-context token screen — rule applied unchanged.

The primary token bank (token_interp.py, seed 21) is confirmed here on 300 pairs built from windows
it never touched (seed 22, pool = the 5,980 windows minus every index used by the primary bank).
Only the token-embedding hook is screened, and no threshold is retuned.

usage: python token_validation.py
"""
import json
import os

import numpy as np
import torch

from common import LAYERS, RESULTS, Runner, wilson
from plot_token import D_HI, D_LO, RHO_FLAT, shelf
from token_interp import analyse, screen

N_VAL = 300
VAL_SEED = 22


def main():
    ctx = np.load(os.path.join(RESULTS, "ctx.npz"))
    W, top1_A = ctx["windows"], ctx["top1"]
    prim = json.load(open(os.path.join(RESULTS, "token_interp.json")))
    used = {i for p in prim["pairs"] for i in p[:2]}
    pool = np.array([i for i in range(len(W)) if i not in used])
    print(f"primary bank used {len(used)} windows; disjoint pool = {len(pool)}", flush=True)

    rng = np.random.default_rng(VAL_SEED)
    order = pool[rng.permutation(len(pool))]
    raw = [(int(order[2 * m]), int(order[2 * m + 1])) for m in range(len(order) // 2)]
    raw = [(i, j) for i, j in raw if W[i, -1] != W[j, -1]]

    run = Runner()
    idsB = np.stack([np.concatenate([W[i, :-1], W[j, -1:]]) for i, j in raw])
    top1B = np.zeros(len(raw), dtype=np.int64)
    for s in range(0, len(raw), 64):
        _, lg = run.forward(torch.from_numpy(idsB[s:s + 64]).long(), rec_layers=())
        top1B[s:s + 64] = lg.argmax(-1).cpu().numpy()
    bank = [(p, k) for k, p in enumerate(raw) if top1_A[p[0]] != top1B[k]][:N_VAL]
    print(f"validation bank: {len(bank)} pairs", flush=True)

    acts = np.zeros((1, len(LAYERS), 1), dtype=np.float32)      # unused at the embedding hook
    rows, D, _ = screen(run, W, bank, "embed", acts, acts, idsB, tag="val:")
    res = analyse(rows, D, "token_validation_embed")

    cand = [n for n, r in enumerate(rows) if r["is_candidate"]]
    sc = [shelf(D[n], rows[n]["k_in"], rows[n]["k_out"]) for n in cand]
    n_strict = sum(1 for rho, db in sc if rho < RHO_FLAT and D_LO < db < D_HI)
    res.update({"n_strict_subplateau": n_strict,
                "strict_sub_rate": n_strict / res["n_eligible"],
                "strict_sub_ci": list(wilson(n_strict, res["n_eligible"])),
                "median_rho": float(np.median([r for r, _ in sc])) if sc else None,
                "val_seed": VAL_SEED, "disjoint_pool_size": int(len(pool))})
    with open(os.path.join(RESULTS, "token_validation.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("rho_candidate", "rho_control")}, indent=1))
    print("saved results/token_validation.json")


if __name__ == "__main__":
    main()
