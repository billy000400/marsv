"""Operator feedback (human_feedback_3): "I was looking for interpolating from one token to another
token with the SAME context inducing a plateau. Can you redesign your experiment and search again?"

Every screen in this direction so far moved from one CONTEXT to another context (activation slerp
between two different 32-token windows, or a text splice that rewrites the context).  This script
holds the context FIXED and moves only the final token:

    S_A = c ++ [t_A]        S_B = c ++ [t_B]        c = the same 31 tokens in both

Both endpoints are real 32-token GPT-2 inputs run with no hooks.  A = top-1 next token of S_A,
B = top-1 of S_B, and pairs are kept only when A != B.  The path between them is built at five
frozen hook points, all applied at the FINAL sequence position only:

  * `embed`  -- interpolate the two token embeddings wte[t_A], wte[t_B] and run the model from
                `inputs_embeds`.  This is the literal "token to token" path: nothing else in the
                input changes, and alpha=0 / alpha=1 reproduce S_A / S_B exactly.
  * blocks 0, 2, 4, 6 -- interpolate the final-position `resid_post` of S_A and S_B at that block
                and patch it into the shared context's forward pass (the same hook points and the
                same `slerp_rescale` geometry as the context-to-context screen, so the two screens
                are directly comparable).

Controls: (i) same-prediction pairs (A == B, held out of the primary denominator); (ii) linear
interpolation at `embed` as a geometry control; (iii) a discrete path that snaps every interpolated
embedding to its nearest real vocabulary token and runs that fully real sequence unpatched;
(iv) exact endpoint fidelity of every hook point against the unpatched endpoint logits.

usage: python token_interp.py
"""
import json
import os
import pickle
import time

import numpy as np
import torch

from common import (ALPHAS, CTX_LEN, LAYERS, RESULTS, Runner, detect, lerp, path_summary, rle,
                    slerp_rescale, wilson)
from matthew_examples import d_curve

HOOKS = ("embed", 0, 2, 4, 6)      # frozen hook points for the token path
BANK_SEED = 21
CTRL_SEED = 13
N_PAIRS = 1000                     # primary token-pair bank
N_SAMEPRED = 500                   # same-prediction negative control
N_DISCRETE = 500                   # pairs used for the nearest-real-token path
PATHS_PER_BATCH = 4
K = len(ALPHAS)


# ------------------------------------------------------------------------------- path geometry
def w10_90(d):
    """Transition width t(d=0.9) - t(d=0.1) of the output-distance curve."""
    def cross(y):
        k = int(np.argmax(d >= y))
        if d[k] < y:
            return np.nan
        if k == 0:
            return float(ALPHAS[0])
        x0, x1, y0, y1 = ALPHAS[k - 1], ALPHAS[k], d[k - 1], d[k]
        return float(x0 + (y - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(x1)
    return cross(0.9) - cross(0.1)


def kappa(d):
    """Share of the total output motion sum|delta d| carried by the sharpest 10% of steps."""
    s = np.abs(np.diff(d))
    if s.sum() <= 0:
        return np.nan
    k = max(1, int(round(0.1 * len(s))))
    return float(np.sort(s)[-k:].sum() / s.sum())


def rho_of(d, kin, kout):
    """Flatness of the C window: output-distance range divided by alpha width (0 = perfectly flat)."""
    seg = d[kin:kout + 1]
    return float(seg.max() - seg.min()) / float(ALPHAS[kout] - ALPHAS[kin])


def detour(summ, min_run=3, jsd_floor=0.005):
    """The frozen A|C|B rule with the A != B requirement dropped: is there ANY interior run of a
    third token C that is top-1 for >= min_run points, strictly beats both endpoint tokens there,
    and is entered and left through a real distribution change?

    Needed for the same-prediction control, where A == B makes a path ineligible by construction
    but the question -- does the model still detour to a third token in between? -- still stands.
    """
    top1, jsd = summ["top1"], summ["jsd"]
    A, B = int(top1[0]), int(top1[-1])
    pidx = {int(t): j for j, t in enumerate(summ["tok_ids"])}
    pA, pB = summ["tok_probs"][:, pidx[A]], summ["tok_probs"][:, pidx[B]]
    for (c, s, e) in rle(top1)[1:-1]:
        if c in (A, B) or (e - s + 1) < min_run:
            continue
        pC = summ["tok_probs"][:, pidx[c]]
        if (pC[s:e + 1] - np.maximum(pA[s:e + 1], pB[s:e + 1])).min() <= 0:
            continue
        if float(jsd[s - 1]) < jsd_floor or float(jsd[e]) < jsd_floor:
            continue
        return True
    return False


def summ_stats(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}


# ------------------------------------------------------------------------------------ the bank
def build_bank(run, W, top1_A, seed=BANK_SEED):
    """Frozen token-pair bank: shared 31-token context, two different final tokens.

    Returns (pairs, sameipred, tokB_top1, actsB) where a pair is (i, j): context and t_A come from
    window i, t_B from window j's final token.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(W))
    raw = [(int(order[2 * m]), int(order[2 * m + 1])) for m in range(len(order) // 2)]
    raw = [(i, j) for i, j in raw if W[i, -1] != W[j, -1]]

    idsB = np.stack([np.concatenate([W[i, :-1], W[j, -1:]]) for i, j in raw])
    top1B = np.zeros(len(raw), dtype=np.int64)
    actsB = np.zeros((len(raw), len(LAYERS), run.model.config.n_embd), dtype=np.float32)
    for s in range(0, len(raw), 64):
        ids = torch.from_numpy(idsB[s:s + 64]).long()
        rec, lg = run.forward(ids, rec_layers=LAYERS)
        top1B[s:s + 64] = lg.argmax(-1).cpu().numpy()
        for li, l in enumerate(LAYERS):
            actsB[s:s + 64, li] = rec[l].float().cpu().numpy()
    diff = [(p, k) for k, p in enumerate(raw) if top1_A[p[0]] != top1B[k]]
    same = [(p, k) for k, p in enumerate(raw) if top1_A[p[0]] == top1B[k]]
    print(f"bank: {len(raw)} token pairs with distinct final tokens -> "
          f"{len(diff)} different-prediction, {len(same)} same-prediction", flush=True)
    return diff[:N_PAIRS], same[:N_SAMEPRED], top1B, actsB, idsB


# ---------------------------------------------------------------------------------- the screen
def screen(run, W, bank, hook, actsA, actsB, idsB, interp=slerp_rescale, tag=""):
    """Interpolate the final token at one hook point for every pair; return detector rows + curves."""
    dev = run.device
    ts = torch.tensor(ALPHAS, dtype=torch.float32, device=dev)
    wte = run.model.transformer.wte.weight
    rows, D, S, t0 = [], [], [], time.time()
    for s in range(0, len(bank), PATHS_PER_BATCH):
        group = bank[s:s + PATHS_PER_BATCH]
        ids = torch.from_numpy(np.stack([W[i] for (i, _), _ in group])).long()
        ids = ids.repeat_interleave(K, dim=0)
        if hook == "embed":
            H = torch.cat([interp(wte[int(W[i, -1])].float(), wte[int(W[j, -1])].float(), ts)
                           for (i, j), _ in group], 0)
            lg = run.forward_embeds(ids, last_embed=H)
        else:
            li = LAYERS.index(hook)
            H = torch.cat([interp(torch.from_numpy(actsA[i, li]).to(dev),
                                  torch.from_numpy(actsB[k, li]).to(dev), ts)
                           for (i, _), k in group], 0)
            _, lg = run.forward(ids, patch=(hook, H), rec_layers=())
        probs = torch.softmax(lg.float(), -1)
        for m, ((i, j), k) in enumerate(group):
            summ = path_summary(probs[m * K:(m + 1) * K])
            r = detect(summ)
            d = d_curve(lg[m * K:(m + 1) * K].float())
            r.update({"i": int(i), "j": int(j), "k": int(k), "hook": str(hook),
                      "t_A": int(W[i, -1]), "t_B": int(W[j, -1]),
                      "n_runs_total": len(rle(summ["top1"])),
                      "w10_90": float(w10_90(d)), "kappa": float(kappa(d)),
                      "entropy_mid": float(summ["entropy"][K // 2]),
                      "detour": bool(detour(summ)), "top1_seq": summ["top1"].tolist()})
            r["rho"] = rho_of(d, r["k_in"], r["k_out"]) if r["is_candidate"] else np.nan
            rows.append(r)
            D.append(d)
            S.append(summ)
        if s % (PATHS_PER_BATCH * 100) == 0:
            el = time.time() - t0
            print(f"  [{tag}{hook}] {s}/{len(bank)} {el:.0f}s "
                  f"eta {el / max(s, 1) * (len(bank) - s):.0f}s", flush=True)
    print(f"  [{tag}{hook}] done {time.time() - t0:.0f}s", flush=True)
    return rows, np.stack(D), S


def analyse(rows, D, tag):
    """Prevalence of a persistent third token and of a genuinely flat sub-plateau."""
    elig = [n for n, r in enumerate(rows) if r["eligible"]]
    cand = [n for n in elig if rows[n]["is_candidate"]]
    non = [n for n in elig if not rows[n]["is_candidate"]]
    n_e, n_c = len(elig), len(cand)
    rho = np.array([rows[n]["rho"] for n in cand]) if cand else np.zeros(0)
    n_sub = int((rho < 0.5).sum())
    # matched control: score each candidate's own C window on a random non-candidate path
    rng = np.random.default_rng(CTRL_SEED)
    ctrl = rng.choice(non, size=min(n_c, len(non)), replace=False) if n_c and non else []
    rho_ctrl = [rho_of(D[int(m)], rows[n]["k_in"], rows[n]["k_out"]) for n, m in zip(cand, ctrl)]
    n_det = int(sum(r["detour"] for r in rows))
    return {
        "tag": tag, "n_paths": len(rows), "n_eligible": n_e,
        "n_detour": n_det, "detour_rate": n_det / len(rows) if rows else np.nan,
        "detour_rate_ci": list(wilson(n_det, len(rows))),
        "n_candidates": n_c, "rate": n_c / n_e if n_e else np.nan,
        "rate_ci": list(wilson(n_c, n_e)),
        "n_clean": int(sum(rows[n]["clean"] for n in cand)),
        "n_subplateau": n_sub, "sub_rate": n_sub / n_e if n_e else np.nan,
        "sub_rate_ci": list(wilson(n_sub, n_e)),
        "frac_rho_lt_0.5": float((rho < 0.5).mean()) if n_c else np.nan,
        "rho_candidate": summ_stats(rho), "rho_control": summ_stats(rho_ctrl),
        "run_len": summ_stats([rows[n]["run_len"] for n in cand]),
        "margin_min": summ_stats([rows[n]["margin_min"] for n in cand]),
        "w10_90": summ_stats([rows[n]["w10_90"] for n in elig]),
        "kappa": summ_stats([rows[n]["kappa"] for n in elig]),
        "n_runs_total": summ_stats([rows[n]["n_runs_total"] for n in elig]),
        "entropy_mid": summ_stats([rows[n]["entropy_mid"] for n in elig]),
        "entropy_mid_cand": summ_stats([rows[n]["entropy_mid"] for n in cand]),
    }


def sensitivity(rows, S, min_runs=(2, 3, 5)):
    n_e = sum(r["eligible"] for r in rows)
    out = {}
    for mr in min_runs:
        k = sum(detect(s, min_run=mr)["is_candidate"] for s in S)
        out[str(mr)] = {"n": int(k), "rate": k / n_e if n_e else np.nan, "ci": list(wilson(k, n_e))}
    return out


# ---------------------------------------------------------- discrete nearest-real-token control
def discrete_paths(run, W, bank, n=N_DISCRETE):
    """Snap every interpolated embedding to its nearest real vocabulary token (cosine) and run the
    resulting REAL sequence unpatched, so no point of the path is a synthetic activation."""
    dev = run.device
    ts = torch.tensor(ALPHAS, dtype=torch.float32, device=dev)
    wte = run.model.transformer.wte.weight.float()
    wn = wte / wte.norm(dim=-1, keepdim=True)
    rows, D, t0 = [], [], time.time()
    for (i, j), _ in bank[:n]:
        H = slerp_rescale(wte[int(W[i, -1])], wte[int(W[j, -1])], ts)
        snap = (H / H.norm(dim=-1, keepdim=True) @ wn.T).argmax(-1)          # [K]
        ids = torch.from_numpy(W[i][None]).long().repeat(K, 1)
        ids[:, -1] = snap.cpu()
        lg = run.forward_embeds(ids)
        summ = path_summary(torch.softmax(lg.float(), -1))
        r = detect(summ)
        d = d_curve(lg.float())
        r.update({"i": int(i), "j": int(j), "hook": "discrete",
                  "n_distinct_tokens": int(len(torch.unique(snap))),
                  "n_runs_total": len(rle(summ["top1"])), "w10_90": float(w10_90(d)),
                  "kappa": float(kappa(d)), "entropy_mid": float(summ["entropy"][K // 2]),
                  "detour": bool(detour(summ)),
                  "snap_seq": snap.cpu().numpy().tolist(), "top1_seq": summ["top1"].tolist()})
        r["rho"] = rho_of(d, r["k_in"], r["k_out"]) if r["is_candidate"] else np.nan
        rows.append(r)
        D.append(d)
        if len(rows) % 100 == 0:
            print(f"  [discrete] {len(rows)}/{n} {time.time() - t0:.0f}s", flush=True)
    print(f"  [discrete] done {time.time() - t0:.0f}s", flush=True)
    return rows, np.stack(D)


# -------------------------------------------------------------------------- endpoint fidelity
def fidelity(run, W, bank, actsA, actsB, idsB, n=20):
    """alpha=0 must reproduce S_A's unpatched logits and alpha=1 must reproduce S_B's, exactly."""
    dev, out = run.device, {}
    ts = torch.tensor([0.0, 1.0], dtype=torch.float32, device=dev)
    wte = run.model.transformer.wte.weight
    for hook in HOOKS:
        e0, e1 = [], []
        for (i, j), k in bank[:n]:
            idsA = torch.from_numpy(W[i][None]).long()
            _, lgA = run.forward(idsA, rec_layers=())
            _, lgB = run.forward(torch.from_numpy(idsB[k][None]).long(), rec_layers=())
            if hook == "embed":
                H = slerp_rescale(wte[int(W[i, -1])].float(), wte[int(W[j, -1])].float(), ts)
                lg = run.forward_embeds(idsA.repeat(2, 1), last_embed=H)
            else:
                li = LAYERS.index(hook)
                H = slerp_rescale(torch.from_numpy(actsA[i, li]).to(dev),
                                  torch.from_numpy(actsB[k, li]).to(dev), ts)
                _, lg = run.forward(idsA.repeat(2, 1), patch=(hook, H), rec_layers=())
            e0.append(float((lg[0] - lgA[0]).abs().max()))
            e1.append(float((lg[1] - lgB[0]).abs().max()))
        out[str(hook)] = {"max_abs_dlogit_alpha0_vs_S_A": max(e0),
                          "max_abs_dlogit_alpha1_vs_S_B": max(e1)}
        print(f"  fidelity {hook}: a0 {max(e0):.2e}  a1 {max(e1):.2e}", flush=True)
    return out


def main():
    ctx = np.load(os.path.join(RESULTS, "ctx.npz"))
    W, top1_A = ctx["windows"], ctx["top1"]
    actsA = np.load(os.path.join(RESULTS, "acts.npy"), mmap_mode="r")
    assert W.shape[1] == CTX_LEN

    run = Runner()
    bank, samepred, top1B, actsB, idsB = build_bank(run, W, top1_A)
    out = {"hooks": [str(h) for h in HOOKS], "bank_seed": BANK_SEED, "n_alpha": K,
           "n_pairs": len(bank), "n_samepred": len(samepred),
           "pairs": [[int(i), int(j), int(k)] for (i, j), k in bank]}
    out["endpoint_fidelity"] = fidelity(run, W, bank, actsA, actsB, idsB)

    curves, rows_all = {}, {}
    for hook in HOOKS:
        rows, D, S = screen(run, W, bank, hook, actsA, actsB, idsB)
        res = analyse(rows, D, f"token_{hook}")
        res["min_run_sensitivity"] = sensitivity(rows, S)
        out[f"token_{hook}"] = res
        rows_all[f"token_{hook}"] = rows
        curves[f"token_{hook}_d"] = D
        curves[f"token_{hook}_kin"] = np.array([r["k_in"] for r in rows])
        curves[f"token_{hook}_kout"] = np.array([r["k_out"] for r in rows])
        curves[f"token_{hook}_iscand"] = np.array([r["is_candidate"] for r in rows])
        curves[f"token_{hook}_score"] = np.array([r["score"] for r in rows])
        print(json.dumps({k: v for k, v in res.items() if k != "min_run_sensitivity"}, indent=1),
              flush=True)

    # controls
    rows, D, _ = screen(run, W, samepred, "embed", actsA, actsB, idsB, tag="samepred:")
    out["control_same_prediction_embed"] = analyse(rows, D, "same_prediction_embed")
    rows_all["control_same_prediction_embed"] = rows

    self_bank = [((i, i), k) for (i, _), k in bank[:300]]
    rows, D, _ = screen(run, W, self_bank, "embed", actsA, actsB, idsB, tag="self:")
    out["control_self_token_embed"] = analyse(rows, D, "self_token_embed")
    rows_all["control_self_token_embed"] = rows

    rows, D, _ = screen(run, W, bank, "embed", actsA, actsB, idsB, interp=lerp, tag="lerp:")
    out["control_lerp_embed"] = analyse(rows, D, "lerp_embed")
    rows_all["control_lerp_embed"] = rows
    curves["control_lerp_embed_d"] = D

    rows, D = discrete_paths(run, W, bank)
    out["discrete_real_token"] = analyse(rows, D, "discrete_real_token")
    out["discrete_real_token"]["n_distinct_tokens"] = summ_stats(
        [r["n_distinct_tokens"] for r in rows])
    rows_all["discrete_real_token"] = rows
    curves["discrete_real_token_d"] = D

    with open(os.path.join(RESULTS, "token_interp.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RESULTS, "token_interp_rows.pkl"), "wb") as f:
        pickle.dump({"rows": rows_all, "alphas": ALPHAS, "bank": bank}, f)
    np.savez_compressed(os.path.join(RESULTS, "token_interp_curves.npz"), **curves)
    print("saved results/token_interp.json + token_interp_rows.pkl + token_interp_curves.npz")


if __name__ == "__main__":
    main()
