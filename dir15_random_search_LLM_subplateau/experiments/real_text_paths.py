"""Operator feedback (human_feedback_2): "Can you check if sub-plateau exists in real language data?"

The primary screen builds its path by PATCHING a synthetic activation (a slerp of two real
activations) into the residual stream.  Every intermediate point of that path is an activation the
model would never produce from any real text -- and indeed the neighbour analysis showed C-region
points sit further off the natural manifold than the endpoints do.  So the third-token region might
be an artefact of leaving the data manifold.

This script removes the synthetic step entirely.  It builds paths from context A to context B whose
EVERY point is a real token sequence run through the unmodified model (no hooks, no patching):

    step k  =  B[0:k] ++ A[k:32]        k = 0 .. 32

i.e. the distant context is replaced first, token by token, and the recent context last.  k=0 is
context A, k=32 is context B, and all 33 points are real GPT-2 inputs.

Two frozen banks:
  R1  the SAME 1,000 primary pairs used by the activation screen -> directly comparable rate.
  R2  1,000 new pairs whose 32nd token is IDENTICAL (seed 15).  The token being predicted from
      never changes along the path, so the morph is purely contextual and the path has no
      final-token discontinuity.  This is the cleanest real-language interpolation available.

The frozen A|C|B detector, the flatness rho and the transition width w(10->90) are reused unchanged
from the activation screen so the numbers are comparable.

usage: python real_text_paths.py
"""
import json
import os
import time

import numpy as np
import torch
from transformers import GPT2TokenizerFast

from common import (CTX_LEN, MODEL, RESULTS, REVISION, Runner, detect, path_summary, rle,
                    wilson)
from matthew_examples import d_curve

K_STEPS = CTX_LEN + 1                       # 33 points: k = 0..32
T_GRID = np.linspace(0.0, 1.0, K_STEPS)
PATHS_PER_BATCH = 4
BANK2_SEED = 15
CTRL_SEED = 13


def w10_90(d):
    """Transition width t(d=0.9) - t(d=0.1) on the raw curve (same definition as the act. screen)."""
    def cross(y):
        k = int(np.argmax(d >= y))
        if d[k] < y:
            return np.nan
        if k == 0:
            return float(T_GRID[0])
        x0, x1, y0, y1 = T_GRID[k - 1], T_GRID[k], d[k - 1], d[k]
        return float(x0 + (y - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(x1)
    return cross(0.9) - cross(0.1)


def build_bank2(W, top1, n_pairs=1000, seed=BANK2_SEED):
    """Frozen bank of pairs sharing the final token, with different unpatched top-1 predictions."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(W))
    groups = {}
    for i in order:
        groups.setdefault(int(W[i, -1]), []).append(int(i))
    pairs, used = [], set()
    for tokid in sorted(groups):
        g = groups[tokid]
        for a in range(0, len(g) - 1, 2):
            i, j = g[a], g[a + 1]
            if top1[i] != top1[j] and i not in used and j not in used:
                pairs.append([i, j])
                used.update((i, j))
    rng.shuffle(pairs)
    return [list(map(int, p)) for p in pairs[:n_pairs]]


def morph(WA, WB):
    """[33, 32] int64: step k is B's first k tokens followed by A's remaining tokens."""
    out = np.stack([np.concatenate([WB[:k], WA[k:]]) for k in range(K_STEPS)])
    return out


def run_bank(run, W, pairs, tag):
    """Screen one bank of pairs -> (rows, D, summaries): detector row, d(t) and summary per pair."""
    rows, D, S = [], {}, []
    t0 = time.time()
    for j0 in range(0, len(pairs), PATHS_PER_BATCH):
        group = pairs[j0:j0 + PATHS_PER_BATCH]
        ids = torch.from_numpy(np.concatenate([morph(W[ia], W[ib]) for ia, ib in group])).long()
        _, lg = run.forward(ids, patch=None, rec_layers=())
        for m, (ia, ib) in enumerate(group):
            L = lg[m * K_STEPS:(m + 1) * K_STEPS].float()
            summ = path_summary(torch.softmax(L, dim=-1))
            r = detect(summ, alphas=T_GRID)
            runs = rle(summ["top1"])
            r.update({"ia": int(ia), "ib": int(ib), "n_runs_total": len(runs),
                      "k_first_B": int(next((s for c, s, _ in runs if c == r["B"]), -1)),
                      "entropy_mid": float(summ["entropy"][K_STEPS // 2]),
                      "top1_seq": summ["top1"].tolist()})
            rows.append(r)
            S.append(summ)
            D[j0 + m] = d_curve(L)
        if j0 % (PATHS_PER_BATCH * 100) == 0:
            el = time.time() - t0
            print(f"  [{tag}] {j0}/{len(pairs)} {el:.0f}s "
                  f"eta {el / max(j0, 1) * (len(pairs) - j0):.0f}s", flush=True)
    print(f"  [{tag}] done {time.time() - t0:.0f}s", flush=True)
    return rows, D, S


def window_stats(d, kin, kout):
    seg = d[kin:kout + 1]
    dt = float(T_GRID[kout] - T_GRID[kin])
    dd = float(seg.max() - seg.min())
    return {"d_mean": float(seg.mean()), "dd": dd, "dt": dt, "rho": dd / dt}


def summ_stats(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}


def analyse(rows, D, tok, W, tag):
    elig = [i for i, r in enumerate(rows) if r["eligible"]]
    cand = [i for i in elig if rows[i]["is_candidate"]]
    n_e, n_c = len(elig), len(cand)

    # matched non-candidate control: same C window scored on a random eligible non-candidate path
    noncand = [i for i in elig if not rows[i]["is_candidate"]]
    rng = np.random.default_rng(CTRL_SEED)
    ctrl = list(rng.choice(noncand, size=min(n_c, len(noncand)), replace=False)) if n_c else []

    cs = [{"path": i, **window_stats(D[i], rows[i]["k_in"], rows[i]["k_out"])} for i in cand]
    ks = [{"path": int(j), **window_stats(D[int(j)], rows[i]["k_in"], rows[i]["k_out"])}
          for i, j in zip(cand, ctrl)]
    rho = np.array([c["rho"] for c in cs]) if cs else np.zeros(0)
    n_sub = int((rho < 0.5).sum())

    # symmetric variant: the A and B regions must themselves persist for >= MIN_RUN points, so a
    # "B plateau" that is only the final grid point does not count.
    def flanks_ok(i):
        rr = rle(np.array(rows[i]["top1_seq"]))
        return (rr[0][2] - rr[0][1] + 1) >= 3 and (rr[-1][2] - rr[-1][1] + 1) >= 3
    strict = [i for i in cand if flanks_ok(i)]
    strict_sub = [i for i, c in zip(cand, cs) if i in set(strict) and c["rho"] < 0.5]

    out = {
        "tag": tag, "n_paths": len(rows), "n_eligible": n_e,
        "n_candidates": n_c,
        "rate": n_c / n_e if n_e else np.nan,
        "rate_ci": list(wilson(n_c, n_e)),
        "n_clean": int(sum(rows[i]["clean"] for i in cand)),
        "n_strict": len(strict),
        "strict_rate": len(strict) / n_e if n_e else np.nan,
        "strict_rate_ci": list(wilson(len(strict), n_e)),
        "n_strict_subplateau": len(strict_sub),
        "strict_sub_rate": len(strict_sub) / n_e if n_e else np.nan,
        "strict_sub_rate_ci": list(wilson(len(strict_sub), n_e)),
        "n_subplateau": n_sub,
        "sub_rate": n_sub / n_e if n_e else np.nan,
        "sub_rate_ci": list(wilson(n_sub, n_e)),
        "rho_candidate": summ_stats(rho),
        "rho_control": summ_stats([c["rho"] for c in ks]),
        "frac_rho_lt_1": float((rho < 1).mean()) if n_c else np.nan,
        "frac_rho_lt_0.5": float((rho < 0.5).mean()) if n_c else np.nan,
        "d_mean_candidate": summ_stats([c["d_mean"] for c in cs]),
        "w10_90": summ_stats([w10_90(D[i]) for i in elig]),
        "n_runs_total": summ_stats([rows[i]["n_runs_total"] for i in elig]),
        "frac_single_step_to_B": float(np.mean([rows[i]["k_first_B"] == K_STEPS - 1
                                                for i in elig])),
        "frac_B_from_k31": float(np.mean([rows[i]["k_first_B"] >= K_STEPS - 2 for i in elig])),
        "median_k_first_B": float(np.median([rows[i]["k_first_B"] for i in elig])),
    }

    # worked examples: the 3 highest-scoring candidates
    order = sorted(strict or cand, key=lambda i: -rows[i]["score"])
    ex = []
    for i in order[:3]:
        r = rows[i]
        seq = [{"token": tok.decode([t]), "k_lo": int(a), "k_hi": int(b)}
               for t, a, b in rle(np.array(r["top1_seq"]))]
        ex.append({"rank": len(ex) + 1, "path": i, "ia": r["ia"], "ib": r["ib"],
                   "A": tok.decode([r["A"]]), "C": tok.decode([r["C"]]), "B": tok.decode([r["B"]]),
                   "k_in": r["k_in"], "k_out": r["k_out"], "run_len": r["run_len"],
                   "score": r["score"], "clean": bool(r["clean"]),
                   "rho": float(window_stats(D[i], r["k_in"], r["k_out"])["rho"]),
                   "d_mean": float(window_stats(D[i], r["k_in"], r["k_out"])["d_mean"]),
                   "ctx_A": tok.decode(W[r["ia"]]), "ctx_B": tok.decode(W[r["ib"]]),
                   "text_at_C": tok.decode(morph(W[r["ia"]], W[r["ib"]])[
                       (r["k_in"] + r["k_out"]) // 2]),
                   "sequence": seq})
    out["examples"] = ex
    return out, cand, ctrl


def sensitivity(rows, summaries, min_runs=(2, 3, 5)):
    """Re-run the frozen detector at other persistence thresholds on the stored path summaries."""
    n_e = sum(r["eligible"] for r in rows)
    out = {}
    for mr in min_runs:
        k = sum(detect(s, min_run=mr, alphas=T_GRID)["is_candidate"] for s in summaries)
        out[str(mr)] = {"n": int(k), "rate": k / n_e if n_e else np.nan,
                        "ci": list(wilson(k, n_e))}
    return out


def main():
    tok = GPT2TokenizerFast.from_pretrained(MODEL, revision=REVISION)
    ctx = np.load(os.path.join(RESULTS, "ctx.npz"))
    W, top1 = ctx["windows"], ctx["top1"]
    man = json.load(open(os.path.join(RESULTS, "manifest.json")))
    bank1 = [list(map(int, p)) for p in man["primary_pairs"]]
    bank2 = build_bank2(W, top1)
    print(f"bank1 (same primary pairs): {len(bank1)}   bank2 (final-token matched): {len(bank2)}")

    run = Runner()
    out = {"k_steps": K_STEPS, "bank2_seed": BANK2_SEED, "ctrl_seed": CTRL_SEED,
           "bank2_pairs": bank2}
    curves = {}
    for tag, pairs in (("random_pairs", bank1), ("final_token_matched", bank2)):
        rows, D, S = run_bank(run, W, pairs, tag)
        res, cand, ctrl = analyse(rows, D, tok, W, tag)
        res["min_run_sensitivity"] = sensitivity(rows, S)
        out[tag] = res
        curves[tag + "_d"] = np.stack([D[i] for i in range(len(rows))])
        curves[tag + "_kin"] = np.array([r["k_in"] for r in rows])
        curves[tag + "_kout"] = np.array([r["k_out"] for r in rows])
        curves[tag + "_iscand"] = np.array([r["is_candidate"] for r in rows])
        curves[tag + "_score"] = np.array([r["score"] for r in rows])
        curves[tag + "_top1"] = np.array([r["top1_seq"] for r in rows])
        print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))

    with open(os.path.join(RESULTS, "real_text_paths.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(RESULTS, "real_text_curves.npz"), **curves)
    print("saved results/real_text_paths.json + real_text_curves.npz")


if __name__ == "__main__":
    main()
