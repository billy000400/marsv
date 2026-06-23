"""Validation artifact for the TwoNN + MLE estimators (addresses REVIEW false-claim #4).

Runs the SAME hand-rolled twonn()/mle() used on the real activations on synthetic
data of KNOWN intrinsic dimension d in {5,10,20,50}: an isotropic d-dim Gaussian
linearly embedded into the ambient 768-d space (random orthonormal columns) plus a
tiny isotropic noise floor. A correct estimator should recover ~d at low d, with the
known TwoNN/MLE downward bias at large d. Saves results/id_validation.json so the
"validated on synthetic Gaussians" claim is substantiated by a saved artifact.
"""
import os, json, time
import numpy as np
from id_estimate import twonn, mle

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
AMBIENT = 768
N = 20000
NOISE = 1e-3
DIMS = [5, 10, 20, 50]
rng = np.random.default_rng(0)

results = []
t0 = time.time()
for d in DIMS:
    # random orthonormal embedding d -> 768
    A = rng.standard_normal((AMBIENT, d))
    Q, _ = np.linalg.qr(A)                      # (768, d), orthonormal cols
    Z = rng.standard_normal((N, d))             # latent isotropic Gaussian
    X = (Z @ Q.T).astype(np.float32)            # embed
    X += NOISE * rng.standard_normal(X.shape).astype(np.float32)
    tn = twonn(X)
    m_inv, m_pt = mle(X)
    row = {"true_d": d, "ambient": AMBIENT, "n": N,
           "twonn": tn, "mle_invavg": m_inv, "mle_meanpt": m_pt}
    results.append(row)
    json.dump(results, open(os.path.join(RES, "id_validation.json"), "w"), indent=2)
    print(f"[{time.time()-t0:.0f}s] true_d={d:3d}  TwoNN={tn:.2f}  "
          f"MLE_invavg={m_inv:.2f}  MLE_meanpt={m_pt:.2f}", flush=True)
print("DONE validate", flush=True)
