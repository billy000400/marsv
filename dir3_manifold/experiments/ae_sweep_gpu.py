"""S3-redux — GPU AE bottleneck sweep on layer-6 resid stream.

Same architecture/metric as ae_sweep.py (768->512->256->k->256->512->768, GELU,
FVU on held-out), but the box now has a usable A10 GPU (sm_86, cu130 torch runs
CUDA fine — the journal's "V100 dead" premise no longer holds). So we give the AE
a MUCH larger training budget to lower the FVU floor and sharpen the k-elbow,
which was the one documented soft spot of the CPU run. Writes a SEPARATE file
(ae_results_gpu.json) so the original CPU sweep is preserved for comparison.
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
# Respect the shared-box VRAM budget (half of the 23GB A10).
torch.cuda.set_per_process_memory_fraction(0.45)
torch.set_num_threads(int(os.environ.get("CPU_THREADS_PER_AGENT", "2")))
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
denom = (Xval ** 2).sum(1).mean().item()
print(f"train {tuple(Xtr.shape)} val {tuple(Xval.shape)} denom(var)={denom:.2f} "
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


def fvu_on(model, data):
    model.eval()
    with torch.no_grad():
        err = ((model(data) - data) ** 2).sum(1).mean().item()
    return err / denom


outpath = os.path.join(RES, "ae_results_gpu.json")
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
           "steps": STEPS, "batch": BATCH}
    results.append(row)
    json.dump(results, open(outpath, "w"), indent=2)
    del model, opt
    torch.cuda.empty_cache()
    print(f"[{time.time()-t0:.0f}s] k={k} val_FVU={val_fvu:.4f} "
          f"train_FVU={train_fvu:.4f}", flush=True)
print("DONE ae_gpu", flush=True)
