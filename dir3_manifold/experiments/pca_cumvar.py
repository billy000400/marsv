"""Cumulative PCA variance curve per layer (operator request 2026-07-02).

Same mean-centered 768x768 covariance -> eigvalsh spectrum as pca_pr.py, but
also SAVE the full cumulative-variance-explained curve so it can be plotted with
the 95% / 99% crossings marked. One layer at a time to stay within RAM budget.
"""
import os, json, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
LAYERS = [0, 3, 6, 9, 11]

rows = []
t0 = time.time()
for L in LAYERS:
    X = np.load(os.path.join(DATA, f"acts_layer{L}.npy")).astype(np.float32)
    n = X.shape[0]
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / (n - 1)            # 768x768
    eig = np.clip(np.linalg.eigvalsh(cov)[::-1], 0, None)  # descending, nonneg
    cum = (np.cumsum(eig) / eig.sum())     # length 768, cumulative variance fraction
    d95 = int(np.searchsorted(cum, 0.95) + 1)
    d99 = int(np.searchsorted(cum, 0.99) + 1)
    rows.append({"layer": L, "n": n,
                 "cumvar": [round(float(v), 6) for v in cum],
                 "d95": d95, "d99": d99})
    print(f"[{time.time()-t0:.0f}s] L{L} d95={d95} d99={d99}", flush=True)
    del X, Xc, cov, eig

json.dump(rows, open(os.path.join(RES, "pca_cumvar.json"), "w"))
print("DONE pca_cumvar", flush=True)
