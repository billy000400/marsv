"""S5c — ID robustness diagnostics requested by CODEX_REVIEW 2026-06-23.

Addresses two review findings on the layer-6 local-ID estimate:
  (#4) Duplicate / self-masking fragility. The original id_estimate.knn_dists
       drops column 0 of topk(k+1) as "self". If exact-duplicate vectors exist
       (plausible after fp16 storage + position pooling), a duplicate at
       distance 0 (different index) can survive as the "nearest neighbour",
       leaking r1=0 into TwoNN (filtered) / MLE (floored to 1e-12, biasing).
       Here we (a) count exact duplicate rows, (b) count zero-distance nearest
       neighbours, and (c) recompute TwoNN/MLE with EXPLICIT self-index masking
       plus zero-distance-neighbour filtering, and compare to the naive method.
  (rec#5) Bootstrap CIs. We draw B disjoint subsamples of size n from the 200k
       layer-6 pool and report mean / std / 2.5-97.5 percentile of TwoNN & MLE.

Runs on GPU (RTX 3090) with VRAM capped per BUDGET.md; falls back to CPU.
"""
import os, json, time
import numpy as np
import torch

THREADS = int(os.environ.get("CPU_THREADS_PER_AGENT", "2"))
torch.set_num_threads(THREADS)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.45)

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
CHUNK = 2000
K_MLE = 20
rng = np.random.default_rng(20260623)


def knn_dists(X, k, mask_self_by_index):
    """k nearest distances. If mask_self_by_index, set the self entry (global
    index match) to +inf before topk so self is excluded BY INDEX, not by
    'smallest distance' (which fails when a distinct duplicate sits at d=0)."""
    Xt = torch.as_tensor(X, device=DEV)
    n = Xt.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    take = k if mask_self_by_index else k + 1
    for i in range(0, n, CHUNK):
        q = Xt[i:i + CHUNK]
        d = torch.cdist(q, Xt)                       # (chunk, n)
        if mask_self_by_index:
            rows = torch.arange(q.shape[0], device=DEV)
            d[rows, i + rows] = float("inf")         # mask self by index
            vals, _ = torch.topk(d, take, dim=1, largest=False)
            out[i:i + q.shape[0]] = vals.double().cpu().numpy()
        else:
            vals, _ = torch.topk(d, take, dim=1, largest=False)
            out[i:i + q.shape[0]] = vals[:, 1:].double().cpu().numpy()  # drop col0
    return out


def twonn(dk2, drop_zero):
    r1, r2 = dk2[:, 0].copy(), dk2[:, 1].copy()
    if drop_zero:
        good = r1 > 0
    else:
        good = r1 > 0  # TwoNN always needs r1>0 to form the ratio
    mu = r2[good] / r1[good]
    mu = mu[mu > 1.0 + 1e-12]
    mu.sort()
    N = mu.shape[0]
    keep = int(N * 0.90)
    mu = mu[:keep]
    F = np.arange(1, keep + 1) / (N + 1.0)
    x = np.log(mu); y = -np.log(1.0 - F)
    return float((x @ y) / (x @ x)), int((~good).sum())


def mle(dk, drop_zero):
    dk = dk.copy()
    n_zero_nbr = int((dk[:, 0] <= 0).sum())
    if drop_zero:
        # keep only points whose K neighbours are all strictly positive
        keep = (dk > 0).all(1)
        dk = dk[keep]
    dk = np.maximum(dk, 1e-12)
    k = dk.shape[1]
    logTk = np.log(dk[:, k - 1:k])
    inv = (logTk - np.log(dk[:, :k - 1])).sum(1) / (k - 1)
    inv = inv[inv > 0]
    return float(1.0 / inv.mean()), n_zero_nbr


def main():
    t0 = time.time()
    L = 6
    Xall = np.load(os.path.join(DATA, f"acts_layer{L}.npy")).astype(np.float32)
    mu_all = Xall.mean(0, keepdims=True)
    n_avail = Xall.shape[0]
    report = {"layer": L, "device": DEV, "n_avail": int(n_avail)}

    # ---- (a) exact-duplicate diagnostic on a 50k centered subsample ----
    n_diag = 50_000
    idx = rng.choice(n_avail, size=n_diag, replace=False)
    Xs = np.ascontiguousarray(Xall[idx] - mu_all, dtype=np.float32)
    uniq = np.unique(Xs, axis=0)
    n_dupe_rows = n_diag - uniq.shape[0]

    # naive (drop-col0) vs robust (mask-self-by-index) kNN
    dk_naive = knn_dists(Xs, K_MLE, mask_self_by_index=False)
    dk_robust = knn_dists(Xs, K_MLE, mask_self_by_index=True)

    tn_naive, tn_nbad = twonn(dk_naive[:, :2], drop_zero=False)
    tn_robust, _ = twonn(dk_robust[:, :2], drop_zero=False)
    ml_naive, ml_zero_naive = mle(dk_naive, drop_zero=False)
    ml_robust, ml_zero_robust = mle(dk_robust, drop_zero=True)

    report["duplicate_diagnostic"] = {
        "n_subsample": n_diag,
        "exact_duplicate_rows": int(n_dupe_rows),
        "zero_distance_nn_naive": int(ml_zero_naive),
        "zero_distance_nn_robust_byindex": int(ml_zero_robust),
        "twonn_naive": tn_naive, "twonn_robust": tn_robust,
        "mle_naive": ml_naive, "mle_robust": ml_robust,
        "twonn_delta": tn_robust - tn_naive,
        "mle_delta": ml_robust - ml_naive,
    }
    print(f"[{time.time()-t0:.0f}s] dupes={n_dupe_rows} "
          f"zeroNN naive={ml_zero_naive} robust={ml_zero_robust} | "
          f"TwoNN {tn_naive:.2f}->{tn_robust:.2f}  MLE {ml_naive:.2f}->{ml_robust:.2f}",
          flush=True)

    # ---- (b) bootstrap CIs (robust kNN), B disjoint n=20k subsamples ----
    B, n_boot = 20, 20_000
    tns, mls = [], []
    for b in range(B):
        ib = rng.choice(n_avail, size=n_boot, replace=False)
        Xb = np.ascontiguousarray(Xall[ib] - mu_all, dtype=np.float32)
        dkb = knn_dists(Xb, K_MLE, mask_self_by_index=True)
        tb, _ = twonn(dkb[:, :2], drop_zero=False)
        mb, _ = mle(dkb, drop_zero=True)
        tns.append(tb); mls.append(mb)
        print(f"[{time.time()-t0:.0f}s] boot {b+1}/{B} TwoNN={tb:.2f} MLE={mb:.2f}",
              flush=True)
    tns = np.array(tns); mls = np.array(mls)
    report["bootstrap"] = {
        "B": B, "n_per_sample": n_boot, "sampling": "disjoint draws w/o replacement",
        "twonn_mean": float(tns.mean()), "twonn_std": float(tns.std(ddof=1)),
        "twonn_ci95": [float(np.percentile(tns, 2.5)), float(np.percentile(tns, 97.5))],
        "mle_mean": float(mls.mean()), "mle_std": float(mls.std(ddof=1)),
        "mle_ci95": [float(np.percentile(mls, 2.5)), float(np.percentile(mls, 97.5))],
    }
    print(f"[{time.time()-t0:.0f}s] BOOTSTRAP TwoNN {tns.mean():.2f}"
          f"±{tns.std(ddof=1):.2f}  MLE {mls.mean():.2f}±{mls.std(ddof=1):.2f}",
          flush=True)

    json.dump(report, open(os.path.join(RES, "id_diagnostics.json"), "w"), indent=2)
    print("DONE id_diagnostics", flush=True)


if __name__ == "__main__":
    main()
