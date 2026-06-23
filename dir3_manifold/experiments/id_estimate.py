"""S2b — nonlinear intrinsic dimension (TwoNN + MLE), pure numpy/torch, CPU.

No skdim/sklearn available, so both estimators are hand-rolled:
  - TwoNN (Facco et al. 2017): ID from the ratio mu=r2/r1 of the 1st/2nd NN
    distances, via the linear fit -log(1-F(mu)) = d*log(mu) through the origin.
  - MLE (Levina-Bickel 2004) with k neighbours; report the MacKay-Ghahramani
    inverse-averaged estimate (mean over points of 1/m_k, then invert).

kNN is chunked brute-force with torch.cdist on CPU (no full NxN matrix).

Two preprocessings per layer (the PCA stage found one massive-activation dim
holding 78-94% of variance from layer 3 on, which dominates raw distances):
  - 'centered'      : mean-centered only (raw geometry, outlier dim included)
  - 'standardized'  : z-scored per dim (outlier dim's dominance removed)
"""
import os, json, time
import numpy as np
import torch

THREADS = int(os.environ.get("CPU_THREADS_PER_AGENT", "2"))
torch.set_num_threads(THREADS)
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
LAYERS = [0, 3, 6, 9, 11]
SUBSAMPLES = [10_000, 50_000]
K_MLE = 20
CHUNK = 1000
rng = np.random.default_rng(0)


def knn_dists(X, k):
    """Return distances to the k nearest *non-self* neighbours, shape (n,k)."""
    Xt = torch.from_numpy(X)  # float32
    n = Xt.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    for i in range(0, n, CHUNK):
        q = Xt[i:i + CHUNK]
        d = torch.cdist(q, Xt)                     # (chunk, n)
        vals, _ = torch.topk(d, k + 1, dim=1, largest=False)  # incl. self(0)
        out[i:i + q.shape[0]] = vals[:, 1:k + 1].numpy()       # drop self
    return out


def twonn(X, discard_frac=0.10):
    d2 = knn_dists(X, 2)
    r1, r2 = d2[:, 0], d2[:, 1]
    good = r1 > 0
    mu = r2[good] / r1[good]
    mu = mu[mu > 1.0 + 1e-12]
    mu.sort()
    N = mu.shape[0]
    keep = int(N * (1.0 - discard_frac))
    mu = mu[:keep]
    F = (np.arange(1, keep + 1)) / (N + 1.0)        # empirical CDF
    x = np.log(mu)
    y = -np.log(1.0 - F)
    d = float((x @ y) / (x @ x))                    # LS through origin
    return d


def mle(X, k=K_MLE):
    dk = knn_dists(X, k)                             # (n,k), ascending
    dk = np.maximum(dk, 1e-12)
    logTk = np.log(dk[:, k - 1:k])                   # (n,1)
    # m_k(i)^-1 = (1/(k-1)) sum_{j=1..k-1} log(T_k/T_j)
    inv = (logTk - np.log(dk[:, :k - 1])).sum(1) / (k - 1)
    inv = inv[inv > 0]
    m_invavg = float(1.0 / inv.mean())               # MacKay-Ghahramani
    m_meanpt = float((1.0 / inv).mean())             # mean of per-point estimates
    return m_invavg, m_meanpt


def main():
  results = []
  t0 = time.time()
  for L in LAYERS:
    path = os.path.join(DATA, f"acts_layer{L}.npy")
    if not os.path.exists(path):
        print(f"L{L} missing", flush=True); continue
    Xall = np.load(path).astype(np.float32)
    n_avail = Xall.shape[0]
    mu_all = Xall.mean(0, keepdims=True)
    sd_all = Xall.std(0, keepdims=True) + 1e-6
    for n in SUBSAMPLES:
        if n > n_avail:
            continue
        idx = rng.choice(n_avail, size=n, replace=False)
        Xs = Xall[idx]
        for prep, X in (("centered", (Xs - mu_all)),
                        ("standardized", (Xs - mu_all) / sd_all)):
            X = np.ascontiguousarray(X, dtype=np.float32)
            tn = twonn(X)
            m_inv, m_pt = mle(X)
            row = {"layer": L, "n": n, "prep": prep,
                   "twonn": tn, "mle_invavg": m_inv, "mle_meanpt": m_pt}
            results.append(row)
            print(f"[{time.time()-t0:.0f}s] L{L} n={n} {prep:12s} "
                  f"TwoNN={tn:.2f} MLE={m_inv:.2f}", flush=True)
            json.dump(results, open(os.path.join(RES, "id_nonlinear.json"), "w"),
                      indent=2)
    del Xall
  print("DONE id_nonlinear", flush=True)


if __name__ == "__main__":
    main()
