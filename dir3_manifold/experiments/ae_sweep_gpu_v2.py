"""S3-redux-v2 — AE sweep robustness checks (addresses REVIEW follow-ups 1,2,3).

Same AE (768->512->256->k->256->512->768, GELU) and held-out FVU metric, on GPU
with the longer 10000-step budget. Adds the checks the review asked for:
  - multiple random seeds {0,1,2} on raw (centered) activations -> elbow stability
  - a sweep on STANDARDIZED (z-scored) activations -> does the knee survive removing
    the massive-activation dim's dominance?
  - exact parameter count per k recorded (param count is NOT held constant; we report
    it so the confound is explicit). Counts rise monotonically with k (~1.05M->1.18M),
    so if params drove FVU the elbow would move toward HIGH k, not create a low-k knee.
Writes results/ae_results_gpu_v2.json and results/ae_param_counts.json.
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
RAW_SEEDS = [0, 1, 2]
STD_SEEDS = [0]

assert torch.cuda.is_available(), "expected GPU"
torch.cuda.set_per_process_memory_fraction(0.45)
torch.set_num_threads(int(os.environ.get("CPU_THREADS_PER_AGENT", "2")))
dev = "cuda"

Xraw = np.load(os.path.join(DATA, "acts_layer6.npy")).astype(np.float32)
rng = np.random.default_rng(0)
perm = rng.permutation(Xraw.shape[0])
n_val = Xraw.shape[0] // 10
val_idx, tr_idx = perm[:n_val], perm[n_val:]


def make_ae(k):
    return nn.Sequential(
        nn.Linear(768, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, k),   nn.GELU(),
        nn.Linear(k, 256),   nn.GELU(),
        nn.Linear(256, 512), nn.GELU(),
        nn.Linear(512, 768),
    ).to(dev)


# --- exact parameter counts per k ---
pcounts = {}
for k in KS:
    m = make_ae(k)
    pcounts[k] = int(sum(p.numel() for p in m.parameters()))
    del m
torch.cuda.empty_cache()
json.dump(pcounts, open(os.path.join(RES, "ae_param_counts.json"), "w"), indent=2)
print("param counts:", pcounts, flush=True)


def prep_data(standardize):
    mu = Xraw[tr_idx].mean(0, keepdims=True)
    if standardize:
        sd = Xraw[tr_idx].std(0, keepdims=True) + 1e-6
        tr = (Xraw[tr_idx] - mu) / sd
        va = (Xraw[val_idx] - mu) / sd
    else:
        tr = Xraw[tr_idx] - mu
        va = Xraw[val_idx] - mu
    Xtr = torch.from_numpy(tr).to(dev)
    Xval = torch.from_numpy(va).to(dev)
    denom = (Xval ** 2).sum(1).mean().item()
    return Xtr, Xval, denom


def fvu_on(model, data, denom):
    model.eval()
    with torch.no_grad():
        err = ((model(data) - data) ** 2).sum(1).mean().item()
    return err / denom


outpath = os.path.join(RES, "ae_results_gpu_v2.json")
results = json.load(open(outpath)) if os.path.exists(outpath) else []
done = {(r["prep"], r["seed"], r["k"]) for r in results}
t0 = time.time()

for prep, seeds in (("standardized", STD_SEEDS), ("raw", RAW_SEEDS)):
    Xtr, Xval, denom = prep_data(prep == "standardized")
    ntr = Xtr.shape[0]
    print(f"=== prep={prep} denom={denom:.2f} ===", flush=True)
    for seed in seeds:
        for k in KS:
            if (prep, seed, k) in done:
                continue
            torch.manual_seed(seed)
            model = make_ae(k)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            lossf = nn.MSELoss()
            model.train()
            g = torch.Generator(device=dev); g.manual_seed(seed * 100003 + k)
            for step in range(STEPS):
                idx = torch.randint(0, ntr, (BATCH,), device=dev, generator=g)
                xb = Xtr[idx]
                opt.zero_grad()
                loss = lossf(model(xb), xb)
                loss.backward()
                opt.step()
            tr_fvu = fvu_on(model, Xtr, denom)
            va_fvu = fvu_on(model, Xval, denom)
            results.append({"prep": prep, "seed": seed, "k": k,
                            "val_fvu": va_fvu, "train_fvu": tr_fvu,
                            "n_params": pcounts[k], "steps": STEPS})
            json.dump(results, open(outpath, "w"), indent=2)
            del model, opt
            torch.cuda.empty_cache()
            print(f"[{time.time()-t0:.0f}s] {prep} seed={seed} k={k} "
                  f"val_FVU={va_fvu:.4f}", flush=True)
    del Xtr, Xval
    torch.cuda.empty_cache()
print("DONE ae_v2", flush=True)
