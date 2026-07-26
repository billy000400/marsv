"""Secondary, clearly-labelled extension: how do the third-token rate and the plateau flatness
behave at LATER interpolation blocks?

The preregistered primary screen only covers blocks 0/2/4/6 of GPT-2 Large's 36, and both the
candidate rate (8.2% -> 27.7%) and the flatness of the third region (median rho 2.52 -> 1.54) improve
monotonically up to block 6, the deepest one frozen. This script runs the SAME 1,000 primary pairs
and the SAME frozen detector at blocks 12, 18, 24 and 30 (chosen as an even spread of the remaining
depth, fixed before running), and additionally recomputes Matthew's d(t) and the C-window flatness
rho for every candidate found.

Nothing here enters the primary prevalence estimate; it is reported as an exploratory depth sweep.

usage: python depth_extension.py
"""
import json
import os
import time

import numpy as np
import torch

from common import ALPHAS, RESULTS, Runner, detect, path_summary, slerp_rescale, wilson
from matthew_examples import d_curve

EXT_LAYERS = (12, 18, 24, 30)      # frozen before running
PATHS_PER_BATCH = 4


def main():
    man = json.load(open(os.path.join(RESULTS, "manifest.json")))
    ctx = np.load(os.path.join(RESULTS, "ctx.npz"))
    W, top1 = ctx["windows"], ctx["top1"]
    pairs = man["primary_pairs"]
    need = sorted({i for p in pairs for i in p})
    pos = {c: j for j, c in enumerate(need)}

    run = Runner()
    dev = run.device
    ts = torch.tensor(ALPHAS, dtype=torch.float32, device=dev)
    K = len(ALPHAS)

    # ---------------------------------------------------------------- activations at the new blocks
    t0 = time.time()
    A = np.zeros((len(need), len(EXT_LAYERS), 1280), dtype=np.float32)
    for j0 in range(0, len(need), 32):
        idx = need[j0:j0 + 32]
        ids = torch.from_numpy(W[idx]).long()
        rec, _ = run.forward(ids, rec_layers=EXT_LAYERS)
        for li, l in enumerate(EXT_LAYERS):
            A[j0:j0 + len(idx), li] = rec[l].float().cpu().numpy()
    print(f"activations at {EXT_LAYERS}: {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------------------- screen
    jobs = [(pi, l, c) for pi in range(len(pairs)) for l in EXT_LAYERS for c in (0, 1)]
    rows, rho, dbar = [], [], []
    for j0 in range(0, len(jobs), PATHS_PER_BATCH):
        batch = jobs[j0:j0 + PATHS_PER_BATCH]
        by_layer = {}
        for job in batch:
            by_layer.setdefault(job[1], []).append(job)
        for l, group in by_layer.items():
            li = EXT_LAYERS.index(l)
            ids = torch.from_numpy(np.stack([W[pairs[pi][c]] for pi, _, c in group])).long()
            ids = ids.repeat_interleave(K, dim=0)
            H = torch.cat([slerp_rescale(torch.from_numpy(A[pos[pairs[pi][0]], li]).to(dev),
                                         torch.from_numpy(A[pos[pairs[pi][1]], li]).to(dev), ts)
                           for pi, _, c in group], dim=0)
            _, lg = run.forward(ids, patch=(l, H), rec_layers=())
            probs = torch.softmax(lg.float(), -1)
            for m, (pi, _, c) in enumerate(group):
                s = path_summary(probs[m * K:(m + 1) * K])
                d = detect(s)
                a, b = pairs[pi]
                d.update({"pair": pi, "layer": l, "cond": c,
                          "ep0_match": int(s["top1"][0] == top1[[a, b][c]])})
                rows.append(d)
                if d["is_candidate"]:
                    dc = d_curve(lg[m * K:(m + 1) * K].float())
                    seg = dc[d["k_in"]:d["k_out"] + 1]
                    rho.append((l, float(seg.max() - seg.min())
                                / float(ALPHAS[d["k_out"]] - ALPHAS[d["k_in"]])))
                    dbar.append(float(seg.mean()))
        if j0 % (PATHS_PER_BATCH * 200) == 0:
            el = time.time() - t0
            print(f"  {j0}/{len(jobs)}  {el:.0f}s "
                  f"eta {el / max(j0, 1) * (len(jobs) - j0):.0f}s", flush=True)

    # ------------------------------------------------------------------------------- summarise
    out = {"layers": list(EXT_LAYERS), "n_paths": len(rows), "by_layer": {}}
    rho = np.array([r for _, r in rho]), np.array([l for l, _ in rho])
    rho_v, rho_l = rho
    for l in EXT_LAYERS:
        sel = [r for r in rows if r["layer"] == l]
        n_el = sum(r["eligible"] for r in sel)
        n_cd = sum(r["is_candidate"] for r in sel)
        m = rho_l == l
        lo, hi = wilson(n_cd, n_el)
        out["by_layer"][str(l)] = {
            "n_paths": len(sel), "n_eligible": n_el, "n_candidates": n_cd,
            "rate": n_cd / n_el if n_el else float("nan"), "ci": [lo, hi],
            "clean_frac": float(np.mean([r["clean"] for r in sel if r["is_candidate"]])) if n_cd else float("nan"),
            "ep0_match": float(np.mean([r["ep0_match"] for r in sel])),
            "median_rho": float(np.median(rho_v[m])) if m.sum() else float("nan"),
            "frac_rho_lt_0.5": float(np.mean(rho_v[m] < 0.5)) if m.sum() else float("nan"),
            "subplateau_rate": float((rho_v[m] < 0.5).sum() / n_el) if n_el else float("nan"),
            "subplateau_ci": [float(x) for x in wilson(int((rho_v[m] < 0.5).sum()), n_el)],
            "median_run_len": float(np.median([r["run_len"] for r in sel if r["is_candidate"]])) if n_cd else float("nan"),
        }
    out["d_mean_median"] = float(np.median(dbar)) if dbar else float("nan")
    with open(os.path.join(RESULTS, "depth_extension.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(RESULTS, "depth_extension_rho.npz"),
                        rho=rho_v, layer=rho_l)
    print(json.dumps(out, indent=1))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
