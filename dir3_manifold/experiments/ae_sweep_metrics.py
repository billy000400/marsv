"""Operator request (human_feedback_07060326): re-derive the AE bottleneck 'ID'
(elbow-k) using RECONSTRUCTION ERROR and COSINE SIMILARITY instead of only FVU.

Identical architecture / optimizer / data / split / budget as ae_sweep_gpu.py
(768->512->256->k->256->512->768 GELU, Adam 1e-3, STEPS=10000, BATCH=4096, seed 0,
raw layer-6 activations, train-mean centered, 90/10 split) so the three metrics are
measured on the SAME trained models and are directly comparable to ae_results_gpu.json.

Per k, on the held-out val set (centered vectors x' = x - mu_train, which is what the
AE reconstructs), we record:
  val_fvu  = mean||x'-xhat|| ^2 / mean||x'|| ^2         (cross-check w/ ae_results_gpu)
  val_rmse = sqrt( mean_over(i,dim) (x'_ij - xhat_ij)^2 )   [per-dimension RMSE]
  val_cos  = mean_i  <x'_i, xhat_i> / (||x'_i|| ||xhat_i||)  [mean cosine similarity]
Elbow (kneedle on log2 k) is computed for each metric downstream (analyze script).
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)

KS = [2, 4, 8, 16, 24, 32, 48, 64, 128, 256]
STEPS = 10000
BATCH = 4096
LR = 1e-3

assert torch.cuda.is_available(), "expected GPU"
torch.cuda.set_per_process_memory_fraction(0.180)  # shared box: 1/5 of RTX 3090
torch.set_num_threads(1)
dev = "cuda"
torch.manual_seed(0)

X = np.load(os.path.join(DATA, "acts_layer6.npy")).astype(np.float32)
rng = np.random.default_rng(0)
perm = rng.permutation(X.shape[0])
n_val = X.shape[0] // 10
val_idx, tr_idx = perm[:n_val], perm[n_val:]
mu = X[tr_idx].mean(0, keepdims=True)
Xtr = torch.from_numpy(X[tr_idx] - mu).to(dev)
Xval = torch.from_numpy(X[val_idx] - mu).to(dev)
denom = (Xval ** 2).sum(1).mean().item()          # mean ||x'||^2 (FVU denominator)
print(f"train {tuple(Xtr.shape)} val {tuple(Xval.shape)} denom={denom:.2f} "
      f"dev={torch.cuda.get_device_name(0)}", flush=True)


def make_ae(k):
    return nn.Sequential(
        nn.Linear(768, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, k),   nn.GELU(),
        nn.Linear(k, 256),   nn.GELU(),
        nn.Linear(256, 512), nn.GELU(),
        nn.Linear(512, 768),
    ).to(dev)


def metrics_on(model, data):
    model.eval()
    with torch.no_grad():
        xhat = model(data)
        diff = xhat - data
        fvu = (diff ** 2).sum(1).mean().item() / denom
        rmse = ((diff ** 2).mean()).sqrt().item()   # per-dimension RMSE
        cos = torch.nn.functional.cosine_similarity(data, xhat, dim=1).mean().item()
    return fvu, rmse, cos


outpath = os.path.join(RES, "ae_results_metrics.json")
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
    tr_fvu, tr_rmse, tr_cos = metrics_on(model, Xtr)
    v_fvu, v_rmse, v_cos = metrics_on(model, Xval)
    row = {"k": k, "val_fvu": v_fvu, "val_rmse": v_rmse, "val_cos": v_cos,
           "train_fvu": tr_fvu, "train_rmse": tr_rmse, "train_cos": tr_cos,
           "steps": STEPS, "batch": BATCH}
    results.append(row)
    json.dump(results, open(outpath, "w"), indent=2)
    del model, opt
    torch.cuda.empty_cache()
    print(f"[{time.time()-t0:.0f}s] k={k} FVU={v_fvu:.4f} RMSE={v_rmse:.4f} "
          f"cos={v_cos:.4f}", flush=True)
print("DONE ae_metrics", flush=True)
