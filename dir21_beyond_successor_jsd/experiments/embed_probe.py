"""Is the per-token width trait already in the static embedding?

Two parts, both aimed at replacing the forward-pass measurement of w_hat_u by a lookup:

A. anchor widths measured with the interpolation site at the INPUT EMBEDDING (before block 0),
   using the same six anchors as the block-0 measurement;
B. a ridge probe from a token's static embedding row W_E[i] to its block-0 anchor width, fitted on
   80 of the 123 endpoint tokens and tested on the other 43 (repeated random splits), plus
   out-of-fold probe predictions used as the per-token feature in the pair-level model.

Writes results/embed.json.
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

sys.path.append("/workspace/marsv_agent_haoyang/dir18_continuation_jsd_plateau/experiments")
import curve_metrics                                             # noqa: F401  (used via run_pair)

from anchor_width import N_ANCHOR, run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint
from common import D18, RESULTS, load
from explore1 import cv_r2

GATE = 0.2
N_TRAIN = 80
N_SPLIT = 50
ALPHAS = np.logspace(-1, 5, 13)

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def ridge_fit(X, y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def probe(F, y, rng, n_train, n_split=N_SPLIT):
    """Repeated train/test splits of a ridge probe. Returns test rho / R^2 per split."""
    n = len(y)
    rhos, r2s, alphas = [], [], []
    for _ in range(n_split):
        perm = rng.permutation(n)
        tr, te = perm[:n_train], perm[n_train:]
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
        Xtr = np.hstack([np.ones((len(tr), 1)), (F[tr] - mu) / sd])
        Xte = np.hstack([np.ones((len(te), 1)), (F[te] - mu) / sd])
        ym = y[tr].mean()
        # inner 5-fold CV on the training tokens only, to pick the ridge strength
        best, best_lam = -np.inf, ALPHAS[0]
        ip = rng.permutation(len(tr))
        for lam in ALPHAS:
            pred = np.empty(len(tr))
            for f in range(5):
                ite = ip[f::5]
                itr = np.setdiff1d(ip, ite)
                b = ridge_fit(Xtr[itr], y[tr][itr] - ym, lam)
                pred[ite] = Xtr[ite] @ b + ym
            s = 1 - ((y[tr] - pred) ** 2).sum() / ((y[tr] - y[tr].mean()) ** 2).sum()
            if s > best:
                best, best_lam = s, lam
        b = ridge_fit(Xtr, y[tr] - ym, best_lam)
        p = Xte @ b + ym
        rhos.append(float(spearmanr(p, y[te])[0]))
        r2s.append(float(1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()))
        alphas.append(float(best_lam))
    return np.array(rhos), np.array(r2s), float(np.median(alphas))


def oof(F, y, rng, folds=5):
    """Out-of-fold probe predictions for every token (no token predicts itself)."""
    n = len(y)
    perm = rng.permutation(n)
    pred = np.empty(n)
    for f in range(folds):
        te = perm[f::folds]
        tr = np.setdiff1d(perm, te)
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
        Xtr = np.hstack([np.ones((len(tr), 1)), (F[tr] - mu) / sd])
        Xte = np.hstack([np.ones((len(te), 1)), (F[te] - mu) / sd])
        ym = y[tr].mean()
        b = ridge_fit(Xtr, y[tr] - ym, 100.0)
        pred[te] = Xte @ b + ym
    return pred


def measure_embedding_site(tok, model, endpoints, anchors):
    """Part A: anchor widths with the interpolation site at the input embedding."""
    patcher = Patcher(model, layer="emb")
    rows = {s: [] for s, _ in endpoints}
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        anc = [endpoint(model, patcher,
                        torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
               for a in anchors]
        for k, (s, i) in enumerate(endpoints):
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            x, z = endpoint(model, patcher, ids)
            for xb, zb in anc:
                rows[s].append(run_pair(model, patcher, ids, x, z, xb, zb)[0])
            if k % 60 == 0:
                print(f"emb site, frame {frame!r} token {k}/{len(endpoints)}", flush=True)
    patcher.close()
    valid = float(np.mean(~np.isnan(np.array([rows[s] for s, _ in endpoints]))))
    return {s: float(np.nanmedian(v)) for s, v in rows.items()}, valid


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    endpoints = sorted(ids_by_str.items())
    used = set(ids_by_str.values())
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]
    assert len(anchors) == N_ANCHOR

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()

    out_path = os.path.join(RESULTS, "embed.json")
    cached = json.load(open(out_path)) if os.path.exists(out_path) else {}
    if "w_emb" in cached:
        w_emb, valid = cached["w_emb"], cached["emb_valid"]
        print("reusing cached embedding-site widths")
    else:
        w_emb, valid = measure_embedding_site(tok, model, endpoints, anchors)
        print(f"embedding site: valid curves {valid:.3f}", flush=True)

    E = model.gpt_neox.embed_in.weight.detach().float().cpu().numpy()
    del model
    torch.cuda.empty_cache()

    # targets
    w0 = {s: float(np.nanmedian(v["w"]))
          for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    tokfx = json.load(open(f"{RESULTS}/explore1.json"))["token_effects"]
    a_by_str = dict(zip(tokfx["tokens"], tokfx["effect"]))

    names = [s for s, _ in endpoints]
    F = np.array([E[ids_by_str[s]] for s in names])
    y0 = np.array([w0[s] for s in names])
    ye = np.array([w_emb[s] for s in names])
    fx_names = [s for s in names if s in a_by_str]
    Ffx = np.array([E[ids_by_str[s]] for s in fx_names])
    yfx = np.array([a_by_str[s] for s in fx_names])

    res = {"anchors": [tok.convert_ids_to_tokens(x) for x in anchors],
           "n_tokens": len(names), "emb_valid": valid, "w_emb": w_emb,
           "n_train": N_TRAIN, "n_split": N_SPLIT}

    # Part A: how the embedding-site measurement compares with the block-0 one
    res["emb_site"] = dict(
        rho_vs_block0=[float(x) for x in spearmanr(ye, y0)[:2]],
        rho_vs_token_effect=[float(x) for x in spearmanr(
            np.array([w_emb[s] for s in fx_names]), yfx)[:2]],
        median=float(np.median(ye)), iqr=float(np.subtract(*np.percentile(ye, [75, 25]))))

    # Part B: probe from the static embedding
    rng = np.random.default_rng(0)
    for key, Fk, yk, ntr in (("w_block0", F, y0, N_TRAIN),
                             ("w_emb_site", F, ye, N_TRAIN),
                             ("token_effect", Ffx, yfx, N_TRAIN)):
        rho, r2, lam = probe(Fk, yk, np.random.default_rng(1), ntr)
        rho_null, r2_null, _ = probe(Fk, rng.permutation(yk), np.random.default_rng(1), ntr)
        res[f"probe_{key}"] = dict(
            n=len(yk), rho_mean=float(rho.mean()), rho_sd=float(rho.std()),
            rho_frac_pos=float((rho > 0).mean()),
            r2_mean=float(r2.mean()), r2_sd=float(r2.std()), median_alpha=lam,
            null_rho_mean=float(rho_null.mean()), null_rho_sd=float(rho_null.std()),
            null_r2_mean=float(r2_null.mean()))
        print(f"probe -> {key}: test rho {rho.mean():+.3f} +- {rho.std():.3f} "
              f"(null {rho_null.mean():+.3f}), test R2 {r2.mean():+.3f}", flush=True)

    # simple univariate baseline: does embedding norm alone carry it?
    nrm = np.linalg.norm(F, axis=1)
    res["emb_norm"] = dict(rho_vs_w0=[float(x) for x in spearmanr(nrm, y0)[:2]],
                           rho_vs_w_emb=[float(x) for x in spearmanr(nrm, ye)[:2]])

    # Part C: pair-level value of the lookup (out-of-fold probe predictions per token)
    t, _, _ = load()
    m = t["out_jsd_min"] >= GATE
    y = t["w"][m]
    one = np.ones((len(y), 1))
    col = lambda d: (np.array([d[x] for x in t["a_str"]])[m]
                     + np.array([d[x] for x in t["b_str"]])[m])[:, None]
    J = t["jsd_B"][m][:, None]
    p_hat = dict(zip(names, oof(F, y0, np.random.default_rng(2))))
    res["pair_level"] = dict(
        measured_w0=cv_r2(np.hstack([one, col(w0)]), y)[0],
        emb_site_w=cv_r2(np.hstack([one, col(w_emb)]), y)[0],
        probe_lookup=cv_r2(np.hstack([one, col(p_hat)]), y)[0],
        probe_lookup_plus_J=cv_r2(np.hstack([one, col(p_hat), J]), y)[0],
        measured_w0_plus_J=cv_r2(np.hstack([one, col(w0), J]), y)[0],
        n_pairs=int(m.sum()))
    res["probe_pred"] = {s: float(v) for s, v in p_hat.items()}
    print("pair-level:", {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in res["pair_level"].items()}, flush=True)

    json.dump(res, open(out_path, "w"), indent=1)
    print("wrote results/embed.json")


if __name__ == "__main__":
    main()
