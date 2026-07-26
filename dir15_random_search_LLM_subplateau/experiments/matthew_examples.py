"""Operator feedback (human_feedback_1): show WHERE the sub-plateau shows up, what the top-1
sequence is, which two contexts are being interpolated, and draw it in the style of Matthew
Shinkle & StefanHex's "Activation Plateaus" post — i.e. the normalized downstream distance

    d(t) = ||x(t) - x_A|| / (||x(t) - x_A|| + ||x(t) - x_B||)

on the final logits, against the interpolation coefficient. A plateau is a flat stretch of d;
a *sub*-plateau is a flat stretch at an INTERMEDIATE height (0 < d < 1), i.e. a third step of a
staircase rather than a single sigmoid.

This re-runs the already-frozen paths (no new pairs, no new thresholds) and records d(t):
  * every A|C|B candidate path in the primary screen (1,290);
  * a seed-13 sample of the same size drawn from eligible NON-candidate paths, each matched to a
    candidate so the same alpha window [k_in, k_out] can be scored on it (control);
  * the 6 frozen inspection paths (3 top-scoring + 3 random, identical to inspect_cands.py), whose
    full d(t) curve, top-1 run-length sequence and context texts are saved for the example figure.

usage: python matthew_examples.py
"""
import json
import os
import pickle
import time

import numpy as np
import torch
from transformers import GPT2TokenizerFast

from common import ALPHAS, LAYERS, MODEL, RESULTS, REVISION, Runner, rle, slerp_rescale

PATHS_PER_BATCH = 4


def w10_90(d):
    """Matthew's transition width: t(d=0.9) - t(d=0.1), linear interpolation on the raw curve."""
    def cross(y):
        k = np.argmax(d >= y)
        if d[k] < y:
            return np.nan
        if k == 0:
            return float(ALPHAS[0])
        x0, x1, y0, y1 = ALPHAS[k - 1], ALPHAS[k], d[k - 1], d[k]
        return float(x0 + (y - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(x1)
    return cross(0.9) - cross(0.1)


def d_curve(logits):
    """logits: [K, V] tensor -> normalized distance-to-A curve d(t), numpy [K]."""
    xA, xB = logits[0], logits[-1]
    dA = torch.linalg.norm(logits - xA, dim=-1)
    dB = torch.linalg.norm(logits - xB, dim=-1)
    return (dA / (dA + dB).clamp_min(1e-9)).float().cpu().numpy()


def main():
    tok = GPT2TokenizerFast.from_pretrained(MODEL, revision=REVISION)
    acts = np.load(os.path.join(RESULTS, "acts.npy"))
    ctx = np.load(os.path.join(RESULTS, "ctx.npz"))
    W = ctx["windows"]
    pri = pickle.load(open(os.path.join(RESULTS, "screen_primary.pkl"), "rb"))
    rows, paths = pri["rows"], pri["paths"]

    cand = [i for i, r in enumerate(rows) if r["is_candidate"]]
    noncand = [i for i, r in enumerate(rows) if r["eligible"] and not r["is_candidate"]]
    rng = np.random.default_rng(13)
    ctrl = list(rng.choice(noncand, size=len(cand), replace=False))   # 1:1 matched to `cand`

    # the frozen inspection set, reproduced exactly as in inspect_cands.py
    order = sorted(cand, key=lambda i: -rows[i]["score"])
    rest = [i for i in cand if i not in order[:3]]
    examples = order[:3] + [int(x) for x in np.random.default_rng(7).choice(rest, size=3,
                                                                           replace=False)]

    run = Runner()
    dev = run.device
    ts = torch.tensor(ALPHAS, dtype=torch.float32, device=dev)
    K = len(ALPHAS)

    todo = sorted(set(cand) | set(map(int, ctrl)) | set(examples))
    D = {}
    t0 = time.time()
    for j0 in range(0, len(todo), PATHS_PER_BATCH):
        group = todo[j0:j0 + PATHS_PER_BATCH]
        by_layer = {}
        for pi in group:
            by_layer.setdefault(rows[pi]["layer"], []).append(pi)
        for l, g in by_layer.items():
            li = LAYERS.index(l)
            ids = torch.from_numpy(np.stack([W[[rows[pi]["ia"], rows[pi]["ib"]][rows[pi]["cond"]]]
                                             for pi in g])).long()
            ids = ids.repeat_interleave(K, dim=0)
            H = torch.cat([slerp_rescale(torch.from_numpy(acts[rows[pi]["ia"], li]).to(dev),
                                         torch.from_numpy(acts[rows[pi]["ib"], li]).to(dev), ts)
                           for pi in g], dim=0)
            _, lg = run.forward(ids, patch=(l, H), rec_layers=())
            for m, pi in enumerate(g):
                D[pi] = d_curve(lg[m * K:(m + 1) * K].float())
        if j0 % (PATHS_PER_BATCH * 200) == 0:
            el = time.time() - t0
            print(f"  {j0}/{len(todo)} paths  {el:.0f}s  "
                  f"eta {el / max(j0, 1) * (len(todo) - j0):.0f}s", flush=True)

    # ------------------------------------------------------------------ per-candidate dwell stats
    def window_stats(d, kin, kout):
        seg = d[kin:kout + 1]
        dt = float(ALPHAS[kout] - ALPHAS[kin])
        dd = float(seg.max() - seg.min())
        return {"d_mean": float(seg.mean()), "dd": dd, "dt": dt, "rho": dd / dt,
                "d_in": float(d[kin]), "d_out": float(d[kout])}

    cand_stats, ctrl_stats = [], []
    for pi, cj in zip(cand, map(int, ctrl)):
        r = rows[pi]
        cand_stats.append({"path": pi, "layer": r["layer"], "run_len": r["run_len"],
                           "score": r["score"], **window_stats(D[pi], r["k_in"], r["k_out"])})
        ctrl_stats.append({"path": cj, "layer": rows[cj]["layer"],
                           **window_stats(D[cj], r["k_in"], r["k_out"])})

    w_cand = np.array([w10_90(D[i]) for i in cand])
    w_ctrl = np.array([w10_90(D[i]) for i in map(int, ctrl)])

    # ------------------------------------------------------------------------ example detail
    ex = []
    for pi in examples:
        r = rows[pi]
        s = paths[pi]
        runs = rle(s["top1"])
        seq = [{"token": tok.decode([t]), "token_id": int(t),
                "alpha_lo": float(ALPHAS[a]), "alpha_hi": float(ALPHAS[b]), "n_alpha": b - a + 1}
               for t, a, b in runs]
        ex.append({
            "path": pi, "layer": r["layer"], "cond": "A" if r["cond"] == 0 else "B",
            "score": float(r["score"]), "clean": bool(r["clean"]),
            "rank": ("top-%d" % (order.index(pi) + 1)) if pi in order[:3] else "random",
            "A": tok.decode([r["A"]]), "C": tok.decode([r["C"]]), "B": tok.decode([r["B"]]),
            "k_in": int(r["k_in"]), "k_out": int(r["k_out"]),
            "alpha_in": float(ALPHAS[r["k_in"]]), "alpha_out": float(ALPHAS[r["k_out"]]),
            "margin_min": float(r["margin_min"]), "run_len": int(r["run_len"]),
            "ctx_A": tok.decode(W[r["ia"]]), "ctx_B": tok.decode(W[r["ib"]]),
            "sequence": seq,
            "d": D[pi].tolist(),
            "w10_90": float(w10_90(D[pi])),
            **window_stats(D[pi], r["k_in"], r["k_out"]),
        })

    def summ(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
                "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}

    out = {
        "n_candidates": len(cand), "n_control": len(ctrl),
        "rho_candidate": summ([c["rho"] for c in cand_stats]),
        "rho_control": summ([c["rho"] for c in ctrl_stats]),
        "d_mean_candidate": summ([c["d_mean"] for c in cand_stats]),
        "frac_rho_lt_1": float(np.mean([c["rho"] < 1 for c in cand_stats])),
        "frac_rho_lt_1_control": float(np.mean([c["rho"] < 1 for c in ctrl_stats])),
        "frac_rho_lt_0.5": float(np.mean([c["rho"] < 0.5 for c in cand_stats])),
        "frac_rho_lt_0.5_control": float(np.mean([c["rho"] < 0.5 for c in ctrl_stats])),
        "frac_d_mid": float(np.mean([0.2 < c["d_mean"] < 0.8 for c in cand_stats])),
        "w10_90_candidate": summ(w_cand), "w10_90_control": summ(w_ctrl),
        "examples": ex,
    }
    with open(os.path.join(RESULTS, "matthew_examples.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(RESULTS, "matthew_d_curves.npz"),
                        cand_idx=np.array(cand), ctrl_idx=np.array(ctrl, dtype=np.int64),
                        d_cand=np.stack([D[i] for i in cand]),
                        d_ctrl=np.stack([D[int(i)] for i in ctrl]),
                        kin=np.array([rows[i]["k_in"] for i in cand]),
                        kout=np.array([rows[i]["k_out"] for i in cand]))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
