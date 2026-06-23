"""S3 — autoencoder bottleneck sweep on layer-6 resid stream.

Fixed deep MLP  768->512->256->k->256->512->768 (GELU).
Vary ONLY k. Identical optimizer/LR/batch/steps/split across all k.
Metric: held-out fraction-of-variance-unexplained (FVU) =
    mean_i ||x_i - xhat_i||^2  /  mean_i ||x_i - mu_train||^2
Mean-centering: subtract TRAIN mean (no unit-normalization).
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
# CPU-sized: B=2048 ~182ms/step, so 1200 steps ~= 3.6 min/k (~36 min for 10 k's).
# Identical steps/batch across all k -> the FVU-vs-k elbow comparison stays valid.
KS = [2, 4, 8, 16, 24, 32, 48, 64, 128, 256]
STEPS = 1200
BATCH = 2048
LR = 1e-3
# GPU (V100 sm_70) is unusable with this cu130 torch build -> force CPU.
dev = "cpu"
torch.set_num_threads(int(os.environ.get("CPU_THREADS_PER_AGENT", "2")))
torch.manual_seed(0)

X = np.load(os.path.join(DATA, "acts_layer6.npy")).astype(np.float32)
rng = np.random.default_rng(0)
perm = rng.permutation(X.shape[0])
n_val = X.shape[0] // 10
val_idx, tr_idx = perm[:n_val], perm[n_val:]
mu = X[tr_idx].mean(0, keepdims=True)
Xtr = torch.from_numpy(X[tr_idx] - mu).to(dev)
Xval = torch.from_numpy(X[val_idx] - mu).to(dev)
denom = (Xval ** 2).sum(1).mean().item()   # mean ||x - mu_train||^2 on val
print(f"train {Xtr.shape} val {Xval.shape} denom(var)={denom:.2f}", flush=True)


def make_ae(k):
    return nn.Sequential(
        nn.Linear(768, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, k),   nn.GELU(),
        nn.Linear(k, 256),   nn.GELU(),
        nn.Linear(256, 512), nn.GELU(),
        nn.Linear(512, 768),
    ).to(dev)


def fvu_on(model, data):
    model.eval()
    with torch.no_grad():
        err = ((model(data) - data) ** 2).sum(1).mean().item()
    return err / denom


outpath = os.path.join(RES, "ae_results.json")
results = json.load(open(outpath)) if os.path.exists(outpath) else []
done_ks = {r["k"] for r in results}
ntr = Xtr.shape[0]
t0 = time.time()
for k in KS:
    if k in done_ks:
        continue
    model = make_ae(k)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    model.train()
    for step in range(STEPS):
        idx = torch.randint(0, ntr, (BATCH,), device=dev)
        xb = Xtr[idx]
        opt.zero_grad()
        loss = lossf(model(xb), xb)
        loss.backward()
        opt.step()
    train_fvu = fvu_on(model, Xtr)
    val_fvu = fvu_on(model, Xval)
    row = {"k": k, "val_fvu": val_fvu, "train_fvu": train_fvu,
           "val_loss": val_fvu * denom, "train_loss": train_fvu * denom}
    results.append(row)
    json.dump(results, open(outpath, "w"), indent=2)
    print(f"[{time.time()-t0:.0f}s] k={k} val_FVU={val_fvu:.4f} "
          f"train_FVU={train_fvu:.4f}", flush=True)
print("DONE ae", flush=True)
