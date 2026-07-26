"""Validity checks: determinism of a re-run subset, and batching invariance of the endpoints.

1. Re-run 20 primary paths from scratch and compare every retained summary array with the stored
   screen output (exact equality for top-1 ids, allclose for probabilities).
2. Every context is exactly 32 tokens, so no padding or attention masking is involved; the only
   batching variable is batch size. Run 32 contexts one at a time and as one batch of 32 and compare
   logits and top-1 predictions.
"""
import json
import os
import pickle

import numpy as np
import torch

from common import ALPHAS, LAYERS, RESULTS, Runner, path_summary, slerp_rescale
from screen import build_jobs

N_PATHS, N_CTX = 20, 32


def main():
    man = json.load(open(os.path.join(RESULTS, "manifest.json")))
    acts = np.load(os.path.join(RESULTS, "acts.npy"))
    ctx = np.load(os.path.join(RESULTS, "ctx.npz"))
    W = ctx["windows"]
    pri = pickle.load(open(os.path.join(RESULTS, "screen_primary.pkl"), "rb"))
    pairs = pri["pairs"]
    run = Runner()
    ts = torch.tensor(ALPHAS, dtype=torch.float32, device=run.device)
    K = len(ALPHAS)

    jobs = build_jobs(pairs, LAYERS)
    max_top1_mismatch, max_p_dev = 0, 0.0
    for n in range(N_PATHS):
        pi, l, c = jobs[n]
        li = LAYERS.index(l)
        H = slerp_rescale(torch.from_numpy(acts[pairs[pi][0], li]).to(run.device),
                          torch.from_numpy(acts[pairs[pi][1], li]).to(run.device), ts)
        ids = torch.from_numpy(W[pairs[pi][c]][None]).long().repeat(K, 1)
        _, lg = run.forward(ids, patch=(l, H), rec_layers=())
        s = path_summary(torch.softmax(lg.float(), -1))
        ref = pri["paths"][n]
        max_top1_mismatch = max(max_top1_mismatch, int((s["top1"] != ref["top1"]).sum()))
        max_p_dev = max(max_p_dev, float(np.abs(s["top1_p"] - ref["top1_p"]).max()))

    ids = torch.from_numpy(W[:N_CTX]).long()
    _, lg_batch = run.forward(ids, rec_layers=())
    single = torch.cat([run.forward(ids[i:i + 1], rec_layers=())[1] for i in range(N_CTX)])
    d_logit = float((lg_batch - single).abs().max())
    n_top1_diff = int((lg_batch.argmax(-1) != single.argmax(-1)).sum())

    out = {"rerun_paths": N_PATHS, "rerun_top1_mismatches": max_top1_mismatch,
           "rerun_max_prob_deviation": max_p_dev,
           "batching_contexts": N_CTX, "batching_max_abs_logit_diff": d_logit,
           "batching_top1_changes": n_top1_diff, "padding": "none (all contexts are 32 tokens)"}
    json.dump(out, open(os.path.join(RESULTS, "checks.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
