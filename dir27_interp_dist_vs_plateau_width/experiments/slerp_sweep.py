"""Feedback 7: the ' big' -> every-other-token sweep with the SLERP path injected at block-0 resid_post.

Usage: DIR27_SITE=block0_resid_post python -u slerp_sweep.py MODEL_KEY [TOK_PER_BATCH] [SCRATCH_DIR]
MODEL_KEY: gpt2-large | gpt2-xl | pythia-1.4b | llama-3.1-8b | qwen3-8b-base

Same prompt, tokens, endpoints, readout blocks, y(t) and width as s2_sweep.py (GPT-2 Large) and gen_sweep.py
(other models). The only change is the injected path: path_control.slerp_rescale (the validated port of
Matthew's slerp_rescale) in place of the straight line. Default TOK_PER_BATCH is the linear sweep's value.

Resumable shards (with the y(t) curves) go to SCRATCH_DIR/<key>/ (default /tmp/dir27_slerp). The deliverable is
results/block0_resid_post/slerp/<key>.csv, one row per token:
  D            L2 distance between the two block-0 endpoints (same quantity as the linear sweep's D)
  L_slerp      length of the injected SLERP path: sum of the 100 segment lengths between the 101 injected points
  cos          cosine between the two endpoints
  W_mid, W_final, flag_nonmonotonic   as in the linear CSVs
  mono_final   1 when the final-block curve alone passes the monotonic test (the summary figures' filter)
"""
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from common import RESULTS, SITE, TS, width, y_curve
from path_control import slerp_rescale

assert SITE == "block0_resid_post"
OUT = os.path.join(RESULTS, "slerp")
os.makedirs(OUT, exist_ok=True)
DEFAULT_PER = {"gpt2-large": 8, "gpt2-xl": 64, "pythia-1.4b": 64, "llama-3.1-8b": 128, "qwen3-8b-base": 128}
SHARD = 4096
BACKSTEP = 0.05
NAN_STATS = (float("nan"),) * 6


def setup(key):
    """-> (tokenizer, vocabulary size, id of ' big', endpoint(ids) -> (k, d), run_patched(r) -> (mid, final))."""
    if key == "gpt2-large":
        import common
        tok, m, dev = common.load()
        r = common.Runner(tok, m, dev)
        return (tok, m.transformer.wte.weight.shape[0], r.id_a,
                lambda ids: r.resid0(torch.as_tensor(ids, device=dev)).clone(), r.run_patched)
    from gen_common import Runner, load
    tok, m = load(key)
    r = Runner(tok, m)
    return tok, len(tok), r.id_a, lambda ids: r.resid0(r.rows(ids)).clone(), r.run_patched


def safe_width(y):
    """common.width; NaN when the curve is not finite (SLERP is undefined for parallel endpoints)."""
    return width(TS, y) if np.isfinite(y).all() else NAN_STATS


def main():
    key = sys.argv[1]
    per = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PER[key]
    scratch = os.path.join(sys.argv[3] if len(sys.argv) > 3 else "/tmp/dir27_slerp", key)
    os.makedirs(scratch, exist_ok=True)
    tok, V, id_a, endpoint, run_patched = setup(key)
    assert tok.decode([id_a]) == " big"
    t = torch.tensor(TS, device="cuda", dtype=torch.float32).view(-1, 1)
    ea = endpoint([id_a])[0]
    rel = lambda x, y: float(((x - y).norm(dim=-1) / y.norm(dim=-1)).max())

    ids_all = np.array([i for i in range(V) if i != id_a])
    for s0 in range(0, len(ids_all), SHARD):
        path = os.path.join(scratch, f"shard_{s0:06d}.npz")
        if os.path.exists(path):
            continue
        ids = ids_all[s0:s0 + SHARD]
        tic = time.time()
        ym_all, yf_all, D, L, cos = [], [], [], [], []
        nonfinite, e0, e1 = 0, 0.0, 0.0
        for b0 in range(0, len(ids), per):
            eb = endpoint(ids[b0:b0 + per])
            p = slerp_rescale(ea, eb, t)                                   # (k, 101, d)
            k = len(eb)
            ok = torch.isfinite(p).all(dim=-1).all(dim=-1)
            nonfinite += int((~ok).sum())
            if ok.any():                                                   # path really starts at a, ends at b
                e0 = max(e0, rel(p[ok, 0], ea.expand(int(ok.sum()), -1)))
                e1 = max(e1, rel(p[ok, -1], eb[ok]))
            mid, fin = run_patched(p.reshape(-1, p.shape[-1]))
            ym_all.append(y_curve(mid.view(k, len(TS), -1)).cpu().numpy())
            yf_all.append(y_curve(fin.view(k, len(TS), -1)).cpu().numpy())
            D.append((eb - ea).norm(dim=-1).cpu().numpy())
            L.append((p[:, 1:] - p[:, :-1]).double().norm(dim=-1).sum(dim=1).cpu().numpy())
            cos.append(torch.nn.functional.cosine_similarity(eb, ea.unsqueeze(0), dim=-1).cpu().numpy())
        ym, yf = np.concatenate(ym_all), np.concatenate(yf_all)
        stats = np.array([safe_width(ym[i]) + safe_width(yf[i]) for i in range(len(ids))])
        np.savez(path, ids=ids, D=np.concatenate(D), L_slerp=np.concatenate(L), cos=np.concatenate(cos),
                 y_mid=ym.astype(np.float16), y_final=yf.astype(np.float16), stats=stats,
                 nonfinite=nonfinite, path_t0_vs_a_max_rel_l2=e0, path_t1_vs_b_max_rel_l2=e1)
        print(f"shard {s0}: {len(ids)} tokens in {time.time() - tic:.1f}s, nonfinite paths {nonfinite}, "
              f"endpoint rel err {e0:.1e} {e1:.1e}", flush=True)

    sh = [np.load(p) for p in sorted(glob.glob(os.path.join(scratch, "shard_*.npz")))]
    ids = np.concatenate([s["ids"] for s in sh])
    assert np.array_equal(ids, ids_all)
    D, L, cos, st = (np.concatenate([s[k] for s in sh]) for k in ("D", "L_slerp", "cos", "stats"))
    flag = ((st[:, [3, 4, 9, 10]] > 1).any(1)) | (st[:, [5, 11]] > BACKSTEP).any(1) | ~np.isfinite(st).all(1)
    mono_final = (st[:, 9] == 1) & (st[:, 10] == 1) & (st[:, 11] <= BACKSTEP)
    with open(os.path.join(OUT, f"{key}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token_id", "token", "D", "L_slerp", "cos", "W_mid", "W_final", "flag_nonmonotonic", "mono_final"])
        for i in range(len(ids)):
            w.writerow([ids[i], repr(tok.decode([int(ids[i])])), f"{D[i]:.4f}", f"{L[i]:.4f}", f"{cos[i]:.4f}",
                        f"{st[i, 2]:.4f}", f"{st[i, 8]:.4f}", int(flag[i]), int(mono_final[i])])
    checks = dict(model=key, n_tokens=int(len(ids)), tok_per_batch=per,
                  nonfinite_paths=int(sum(int(s["nonfinite"]) for s in sh)),
                  path_t0_vs_a_max_rel_l2=float(max(float(s["path_t0_vs_a_max_rel_l2"]) for s in sh)),
                  path_t1_vs_b_max_rel_l2=float(max(float(s["path_t1_vs_b_max_rel_l2"]) for s in sh)),
                  n_negative_cosine=int((cos < 0).sum()), cos_min=float(cos.min()),
                  L_over_D_min_median_max=[float(x) for x in np.nanpercentile(L / D, [0, 50, 100])])
    json.dump(checks, open(os.path.join(scratch, "checks.json"), "w"), indent=1)
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
