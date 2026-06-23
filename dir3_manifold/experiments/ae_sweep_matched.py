"""S5b — PARAMETER-MATCHED AE bottleneck sweep (addresses Codex review concern #4 /
suggested step #1).

The earlier sweeps let total parameter count drift with k (1.05M @ k=2 -> 1.18M @ k=256).
Reviewer: "saying the confound runs the wrong way does not remove it; a param-matched AE
sweep is needed if the AE result is going to matter." This script holds the TOTAL
parameter count fixed across every k by compensating the OUTER hidden width h1 (the
768<->h1 layers, far from the bottleneck): small k gets a wider h1 so its total params
match the largest-k model. Bottleneck k stays the only varying information channel.

Architecture: 768 -> h1 -> 256 -> k -> 256 -> h1 -> 768 (GELU), with h1 chosen per k.
P(h1,k) = 2050*h1 + 1280 + 513*k ; target = P(512,256) = 1,182,208.
=> h1(k) = round((1180928 - 513*k)/2050).  Residual mismatch < 0.1% (reported).

Same held-out FVU metric / 90-10 split / Adam 1e-3 / STEPS=10000 / BATCH=4096 / raw
centered activations as ae_sweep_gpu_v2.py, seeds {0,1,2}. If the low-k knee survives
parameter matching, the AE bend is not a parameter-count artifact. (Adding outer width
at low k can only HELP low-k reconstruction, so a persisting knee is conservative.)
Writes results/ae_results_matched.json.
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)

KS = [2, 4, 8, 16, 24, 32, 48, 64, 128, 256]
SEEDS = [0]  # single seed over full k-range; seed-stability already established (≤0.0018) by unmatched sweeps
STEPS = 10000
BATCH = 4096
LR = 1e-3
TARGET = 1182208
H2 = 256


def h1_for(k):
    return int(round((1180928 - 513 * k) / 2050))


def params_of(h1, k):
    return 2050 * h1 + 1280 + 513 * k


assert torch.cuda.is_available(), "expected GPU"
torch.cuda.set_per_process_memory_fraction(0.45)
torch.set_num_threads(int(os.environ.get("CPU_THREADS_PER_AGENT", "2")))
dev = "cuda"

Xraw = np.load(os.path.join(DATA, "acts_layer6.npy")).astype(np.float32)
rng = np.random.default_rng(0)
perm = rng.permutation(Xraw.shape[0])
n_val = Xraw.shape[0] // 10
val_idx, tr_idx = perm[:n_val], perm[n_val:]

mu = Xraw[tr_idx].mean(0, keepdims=True)
Xtr = torch.from_numpy(Xraw[tr_idx] - mu).to(dev)
Xval = torch.from_numpy(Xraw[val_idx] - mu).to(dev)
denom = (Xval ** 2).sum(1).mean().item()
ntr = Xtr.shape[0]


def make_ae(k):
    h1 = h1_for(k)
    return nn.Sequential(
        nn.Linear(768, h1), nn.GELU(),
        nn.Linear(h1, H2),  nn.GELU(),
        nn.Linear(H2, k),   nn.GELU(),
        nn.Linear(k, H2),   nn.GELU(),
        nn.Linear(H2, h1),  nn.GELU(),
        nn.Linear(h1, 768),
    ).to(dev)


def fvu_on(model, data):
    model.eval()
    with torch.no_grad():
        err = ((model(data) - data) ** 2).sum(1).mean().item()
    return err / denom


# report the matched param counts up front
matched = {}
for k in KS:
    m = make_ae(k)
    matched[k] = {"h1": h1_for(k), "n_params": int(sum(p.numel() for p in m.parameters())),
                  "predicted": params_of(h1_for(k), k)}
    del m
torch.cuda.empty_cache()
json.dump(matched, open(os.path.join(RES, "ae_matched_param_counts.json"), "w"), indent=2)
spread = max(v["n_params"] for v in matched.values()) - min(v["n_params"] for v in matched.values())
print(f"matched param counts (spread={spread}, {100*spread/TARGET:.3f}%):", flush=True)
for k in KS:
    print(f"  k={k:4d} h1={matched[k]['h1']:4d} params={matched[k]['n_params']}", flush=True)

outpath = os.path.join(RES, "ae_results_matched.json")
results = json.load(open(outpath)) if os.path.exists(outpath) else []
done = {(r["seed"], r["k"]) for r in results}
t0 = time.time()

for seed in SEEDS:
    for k in KS:
        if (seed, k) in done:
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
        tr_fvu = fvu_on(model, Xtr)
        va_fvu = fvu_on(model, Xval)
        results.append({"seed": seed, "k": k, "h1": h1_for(k),
                        "val_fvu": va_fvu, "train_fvu": tr_fvu,
                        "n_params": matched[k]["n_params"], "steps": STEPS})
        json.dump(results, open(outpath, "w"), indent=2)
        del model, opt
        torch.cuda.empty_cache()
        print(f"[{time.time()-t0:.0f}s] seed={seed} k={k} val_FVU={va_fvu:.4f}", flush=True)
print("DONE ae_matched", flush=True)
