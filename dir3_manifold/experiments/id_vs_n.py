"""Operator ask 2026-07-08 #1: "TwoNN depends on how many points to sample —
do a study varying the number of points and check if it is noise."

We sweep the subsample size n and, at each n, draw R independent subsamples of
the layer-6 residual-stream pool and recompute TwoNN + MLE (robust self-masking
kNN, reused from id_diagnostics). For each n we report the across-repeat
mean/std. This separates two things the operator conflated:

  * SAMPLING NOISE at fixed n  = the across-repeat std (shrinks ~1/sqrt(n));
  * SYSTEMATIC n-DEPENDENCE    = the drift of the mean as n grows.

If the mean drifts by more than the noise band as n grows, the estimate has a
finite-sample bias (a known TwoNN/MLE property) and a single-n number is not a
stable ID; if the mean is flat within the band, the earlier 11-15 spread is just
noise. Runs on GPU with VRAM capped per BUDGET.md; falls back to CPU.
"""
import os, json, time
import numpy as np
import torch

THREADS = int(os.environ.get("CPU_THREADS_PER_AGENT", "1"))
torch.set_num_threads(THREADS)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.18)

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
K_MLE = 20
rng = np.random.default_rng(20260708)


def knn_dists(X, k, chunk):
    """k nearest distances, self excluded BY INDEX (robust to d=0 duplicates)."""
    Xt = torch.as_tensor(X, device=DEV)
    n = Xt.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    for i in range(0, n, chunk):
        q = Xt[i:i + chunk]
        d = torch.cdist(q, Xt)
        rows = torch.arange(q.shape[0], device=DEV)
        d[rows, i + rows] = float("inf")
        vals, _ = torch.topk(d, k, dim=1, largest=False)
        out[i:i + q.shape[0]] = vals.double().cpu().numpy()
    del Xt
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


def twonn(dk2):
    r1, r2 = dk2[:, 0], dk2[:, 1]
    good = r1 > 0
    mu = r2[good] / r1[good]
    mu = mu[mu > 1.0 + 1e-12]
    mu.sort()
    N = mu.shape[0]
    keep = int(N * 0.90)
    mu = mu[:keep]
    F = np.arange(1, keep + 1) / (N + 1.0)
    x = np.log(mu); y = -np.log(1.0 - F)
    return float((x @ y) / (x @ x))


def mle(dk):
    keep = (dk > 0).all(1)
    dk = np.maximum(dk[keep], 1e-12)
    k = dk.shape[1]
    logTk = np.log(dk[:, k - 1:k])
    inv = (logTk - np.log(dk[:, :k - 1])).sum(1) / (k - 1)
    inv = inv[inv > 0]
    return float(1.0 / inv.mean())


def main():
    t0 = time.time()
    L = 6
    Xall = np.load(os.path.join(DATA, f"acts_layer{L}.npy")).astype(np.float32)
    mu_all = Xall.mean(0, keepdims=True)
    n_avail = Xall.shape[0]

    ns = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    R = 8
    rows = []
    for n in ns:
        chunk = 1000 if n > 50000 else 2000
        reps_tn, reps_ml = [], []
        r_here = 1 if n >= n_avail else R
        for r in range(r_here):
            idx = rng.choice(n_avail, size=min(n, n_avail), replace=False)
            Xs = np.ascontiguousarray(Xall[idx] - mu_all, dtype=np.float32)
            dk = knn_dists(Xs, K_MLE, chunk)
            reps_tn.append(twonn(dk[:, :2]))
            reps_ml.append(mle(dk))
        tn = np.array(reps_tn); ml = np.array(reps_ml)
        row = {
            "n": n, "repeats": r_here,
            "twonn_mean": float(tn.mean()), "twonn_std": float(tn.std(ddof=1) if r_here > 1 else 0.0),
            "mle_mean": float(ml.mean()), "mle_std": float(ml.std(ddof=1) if r_here > 1 else 0.0),
        }
        rows.append(row)
        print(f"[{time.time()-t0:.0f}s] n={n:>6} R={r_here} "
              f"TwoNN={row['twonn_mean']:.2f}±{row['twonn_std']:.2f} "
              f"MLE={row['mle_mean']:.2f}±{row['mle_std']:.2f}", flush=True)

    out = {
        "layer": L, "device": DEV, "n_avail": int(n_avail),
        "K_MLE": K_MLE, "repeats_per_n": R, "sweep": rows,
    }
    json.dump(out, open(os.path.join(RES, "id_vs_n.json"), "w"), indent=2)
    print("DONE id_vs_n", flush=True)


if __name__ == "__main__":
    main()
