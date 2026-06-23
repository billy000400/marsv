"""S2a — linear PCA participation ratio per layer (cheap, full 200k).

PR = (Σλ)^2 / Σλ^2 over the eigenvalues of the mean-centered covariance.
Also report dims to reach 90/95/99% cumulative variance. No unit-normalization.
Uses the 768x768 covariance -> eigvalsh (fast, exact for PCA spectrum).
"""
import os, json, time
import numpy as np

torch_threads = int(os.environ.get("CPU_THREADS_PER_AGENT", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
LAYERS = [0, 3, 6, 9, 11]

rows = []
t0 = time.time()
for L in LAYERS:
    X = np.load(os.path.join(DATA, f"acts_layer{L}.npy")).astype(np.float32)
    n = X.shape[0]
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (n - 1)          # 768x768
    eig = np.linalg.eigvalsh(cov)         # ascending
    eig = np.clip(eig[::-1], 0, None)     # descending, nonneg
    tot = eig.sum()
    pr = float((tot ** 2) / (eig ** 2).sum())
    cum = np.cumsum(eig) / tot
    d90 = int(np.searchsorted(cum, 0.90) + 1)
    d95 = int(np.searchsorted(cum, 0.95) + 1)
    d99 = int(np.searchsorted(cum, 0.99) + 1)
    row = {"layer": L, "n": n, "pca_pr": pr,
           "pca_d90": d90, "pca_d95": d95, "pca_d99": d99,
           "top_eig_frac": float(eig[0] / tot)}
    rows.append(row)
    print(f"[{time.time()-t0:.0f}s] L{L} PR={pr:.1f} d90={d90} d95={d95} "
          f"d99={d99} top1={row['top_eig_frac']:.3f}", flush=True)

json.dump(rows, open(os.path.join(RES, "pca_pr.json"), "w"), indent=2)
print("DONE pca_pr", flush=True)
