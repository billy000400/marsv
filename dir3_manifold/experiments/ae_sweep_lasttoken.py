"""Operator ask 2026-07-08 #2: re-run the layer-6 AE bottleneck sweep on LAST-token
activations (acts_layer6_lasttoken.npy) instead of the pooled all-token cloud, to
test whether an elbow appears when we stop averaging "too much information".

Same fixed deep MLP 768->512->256->k->256->512->768 (GELU) and identical
optimizer/LR/batch/steps/split across k as experiments/ae_sweep.py, so the pooled
vs last-token elbow comparison is apples-to-apples. Scores held-out FVU, per-sample
rel-L2, and per-sample cosine (elbow-relevant metrics from REPORT_AE).
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
KS = [2, 4, 8, 16, 24, 32, 48, 64, 128, 256]
STEPS = 1200
BATCH = 2048
LR = 1e-3
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ.get("CPU_THREADS_PER_AGENT", "1")))
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.18)
torch.manual_seed(0)

X = np.load(os.path.join(DATA, "acts_layer6_lasttoken.npy")).astype(np.float32)
rng = np.random.default_rng(0)
perm = rng.permutation(X.shape[0])
n_val = X.shape[0] // 10
val_idx, tr_idx = perm[:n_val], perm[n_val:]
mu = X[tr_idx].mean(0, keepdims=True)
Xtr = torch.from_numpy(X[tr_idx] - mu).to(dev)
Xval = torch.from_numpy(X[val_idx] - mu).to(dev)
denom = (Xval ** 2).sum(1).mean().item()
# raw (uncentered) val, for cosine/rel-L2 on the reconstructed RAW vector
Xval_raw = torch.from_numpy(X[val_idx]).to(dev)
mu_t = torch.from_numpy(mu).to(dev)
var = X[tr_idx].var(0)
top1 = float(var.max() / var.sum())
print(f"last-token: train {Xtr.shape} val {Xval.shape} denom={denom:.2f} "
      f"top1_var_frac={top1:.4f}", flush=True)


def make_ae(k):
    return nn.Sequential(
        nn.Linear(768, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, k),   nn.GELU(),
        nn.Linear(k, 256),   nn.GELU(),
        nn.Linear(256, 512), nn.GELU(),
        nn.Linear(512, 768),
    ).to(dev)


def metrics(model):
    model.eval()
    with torch.no_grad():
        rec_c = model(Xval)                 # reconstruction of centered vector
        fvu = ((rec_c - Xval) ** 2).sum(1).mean().item() / denom
        rec_raw = rec_c + mu_t              # back to raw space
        rel = (((Xval_raw - rec_raw).norm(dim=1)) / Xval_raw.norm(dim=1)).mean().item()
        cos = F.cosine_similarity(Xval_raw, rec_raw, dim=1).mean().item()
        tr_fvu = ((model(Xtr) - Xtr) ** 2).sum(1).mean().item() / \
            (Xtr ** 2).sum(1).mean().item()
    return fvu, rel, cos, tr_fvu


ntr = Xtr.shape[0]
results = []
t0 = time.time()
for k in KS:
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
    fvu, rel, cos, tr_fvu = metrics(model)
    row = {"k": k, "val_fvu": fvu, "train_fvu": tr_fvu,
           "val_rel_l2": rel, "val_cos": cos}
    results.append(row)
    print(f"  k={k:4d} val_FVU={fvu:.4f} rel_L2={rel:.4f} cos={cos:.4f} "
          f"train_FVU={tr_fvu:.4f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"pooling": "last_token", "top1_var_frac": top1,
               "n_points": int(X.shape[0]), "steps": STEPS, "rows": results},
              open(os.path.join(RES, "ae_results_lasttoken.json"), "w"), indent=1)
print("DONE", flush=True)
