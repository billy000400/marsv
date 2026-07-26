"""S2 — run the interpolation screen over a frozen pair bank.

For every (pair, layer, conditioning context) the interpolated final-position resid_post is patched
into the conditioning context's forward pass at that block and propagated through all remaining
blocks + the LM head. The FULL next-token distribution is computed at all 50 alphas; only frozen
summaries (top-1 trajectory, entropy, adjacent-alpha JSD, probability trajectories of every token
that is top-1 somewhere) are retained.

Conditioning: a pair (a, b) yields two paths per layer — one conditioned on context a (where alpha=0
reproduces a's own unpatched state exactly) and one conditioned on context b (where alpha=1 does).

usage: python screen.py --bank primary [--interp slerp|lerp] [--control same_pred|self] [--n N]
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

from common import ALPHAS, LAYERS, RESULTS, Runner, detect, lerp, path_summary, slerp_rescale

PATHS_PER_BATCH = 4          # 4 paths x 50 alphas = 200 rows of 32 tokens


def build_jobs(pairs, layers):
    """One job per (pair, layer, conditioning). cond=0 -> condition on context a, 1 -> on b."""
    return [(pi, l, c) for pi in range(len(pairs)) for l in layers for c in (0, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="primary")
    ap.add_argument("--interp", default="slerp", choices=["slerp", "lerp"])
    ap.add_argument("--control", default="none", choices=["none", "same_pred", "self"])
    ap.add_argument("--n", type=int, default=0, help="limit number of pairs (0 = all)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    man = json.load(open(os.path.join(RESULTS, "manifest.json")))
    acts = np.load(os.path.join(RESULTS, "acts.npy"))
    ctx = np.load(os.path.join(RESULTS, "ctx.npz"))
    W, top1 = ctx["windows"], ctx["top1"]

    if args.control == "same_pred":
        # negative control: pairs of DIFFERENT contexts with the SAME unpatched top-1 prediction,
        # drawn from the reference pool so they never enter the primary denominator.
        rng = np.random.default_rng(1)
        pool = np.array(man["reference_idx"])
        by = {}
        for i in pool:
            by.setdefault(int(top1[i]), []).append(int(i))
        cands = [v for v in by.values() if len(v) >= 2]
        pairs = []
        for v in cands:
            rng.shuffle(v)
            for j in range(0, len(v) - 1, 2):
                pairs.append([v[j], v[j + 1]])
        rng.shuffle(pairs)
    elif args.control == "self":
        pool = np.array(man["reference_idx"])[:300]
        pairs = [[int(i), int(i)] for i in pool]
    else:
        pairs = man[f"{args.bank}_pairs"]
    if args.n:
        pairs = pairs[: args.n]
    name = args.out or f"screen_{args.bank}" + (f"_{args.interp}" if args.interp != "slerp" else "") \
        + (f"_{args.control}" if args.control != "none" else "")

    run = Runner()
    dev = run.device
    ts = torch.tensor(ALPHAS, dtype=torch.float32, device=dev)
    interp = slerp_rescale if args.interp == "slerp" else lerp
    K = len(ALPHAS)

    jobs = build_jobs(pairs, LAYERS)
    rows, paths, t0 = [], [], time.time()
    fid = []                                   # exact endpoint-fidelity checks on a subset
    for j0 in range(0, len(jobs), PATHS_PER_BATCH):
        batch = jobs[j0:j0 + PATHS_PER_BATCH]
        assert len({l for _, l, _ in batch}) == 1 or True
        # group strictly by layer so one patch hook layer applies to the whole batch
        by_layer = {}
        for job in batch:
            by_layer.setdefault(job[1], []).append(job)
        for l, group in by_layer.items():
            li = LAYERS.index(l)
            ids = torch.from_numpy(np.stack([W[pairs[pi][c]] for pi, _, c in group])).long()
            ids = ids.repeat_interleave(K, dim=0)
            H = torch.cat([interp(torch.from_numpy(acts[pairs[pi][0], li]).to(dev),
                                  torch.from_numpy(acts[pairs[pi][1], li]).to(dev), ts)
                           for pi, _, c in group], dim=0)
            _, lg = run.forward(ids, patch=(l, H), rec_layers=())
            probs = torch.softmax(lg.float(), -1)
            for m, (pi, _, c) in enumerate(group):
                s = path_summary(probs[m * K:(m + 1) * K])
                d = detect(s)
                a, b = pairs[pi]
                d.update({"pair": pi, "ia": a, "ib": b, "layer": l, "cond": c,
                          "ep0_match": int(s["top1"][0] == top1[[a, b][c]]),
                          "ep1_match_foreign": int(s["top1"][-1] == top1[[a, b][1 - c]]),
                          "unpatched_A": int(top1[a]), "unpatched_B": int(top1[b])})
                rows.append(d)
                paths.append(s)
        if j0 % (PATHS_PER_BATCH * 100) == 0:
            el = time.time() - t0
            print(f"  {j0}/{len(jobs)} paths  {el:.0f}s  eta {el / max(j0, 1) * (len(jobs) - j0):.0f}s",
                  flush=True)

    # exact endpoint fidelity: patched alpha=0 logits vs unpatched logits, same context
    for pi, l, c in jobs[:40:4]:
        li = LAYERS.index(l)
        idx = pairs[pi][c]
        ids = torch.from_numpy(W[idx][None]).long()
        _, lg_un = run.forward(ids, rec_layers=())
        hA = torch.from_numpy(acts[pairs[pi][0], li]).to(dev)
        hB = torch.from_numpy(acts[pairs[pi][1], li]).to(dev)
        Hp = interp(hA, hB, ts)
        _, lg_p = run.forward(ids, patch=(l, Hp[0 if c == 0 else -1][None]), rec_layers=())
        fid.append(float((lg_p - lg_un).abs().max()))

    out = os.path.join(RESULTS, f"{name}.pkl")
    with open(out, "wb") as f:
        pickle.dump({"rows": rows, "paths": paths, "pairs": pairs, "alphas": ALPHAS,
                     "endpoint_fidelity_maxabs_logit": fid,
                     "args": vars(args)}, f)
    n_el = sum(r["eligible"] for r in rows)
    n_cd = sum(r["is_candidate"] for r in rows)
    print(f"{name}: {len(rows)} paths, {n_el} eligible, {n_cd} candidates, "
          f"endpoint fidelity max|dlogit| = {max(fid):.2e}, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
