"""Operator ask 2026-07-08 #3: for the lasse.png reproduction, check whether the
LARGE-k autoencoders were undertrained. Plot the learning curve (train + held-out
val loss vs training step) for each k, and re-read the k-sweep at an early
checkpoint (matching the 3000-step lasse repro) vs a long budget, to see what the
U-shape becomes if the large-k models are allowed to converge.

Signature of undertraining, already visible in the 3000-step repro: TRAIN rel-L2
RISES with k (k=500 train 0.513 > k=100 train 0.467). At convergence a bigger
bottleneck can only fit the train set better, so a rising train error at fixed step
budget means the big models simply haven't finished training. This script trains
long enough to test whether that reverses (train error becomes monotone in k) and
whether the held-out U flattens.

Identical DeepAutoencoder / data / split / optimizer as ae_sweep_lasse.py; only adds
per-step logging and multiple checkpoints. GPU, VRAM capped per BUDGET.md.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ae_share"))
from src.autoencoders import DeepAutoencoder  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.180)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")


@torch.no_grad()
def evaluate(ae, data, mu, chunk=8192):
    n = data.shape[0]; se = 0.0; rel = 0.0; cos = 0.0; denom = 0.0
    for i in range(0, n, chunk):
        b = data[i:i + chunk].float().cuda()
        r, _ = ae(b)
        se += ((b - r) ** 2).sum().item()
        denom += ((b - mu) ** 2).sum().item()
        rel += ((b - r).norm(dim=-1) / b.norm(dim=-1)).sum().item()
        cos += F.cosine_similarity(b, r, dim=-1).sum().item()
    return {"mse": se / n, "fvu": se / denom, "rel_l2": rel / n, "cos": cos / n}


def train_with_curve(train_gpu, val, train_eval, mu, d_model, k,
                     n_steps, batch, lr, eval_every, seed):
    torch.manual_seed(seed)
    ae = DeepAutoencoder(d_in=d_model, k=k, hidden_dims=[4096, 4096, 2048]).cuda()
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    n = train_gpu.shape[0]
    curve = {"step": [], "train_mse": [], "val_mse": [],
             "val_rel_l2": [], "val_cos": [], "train_rel_l2": []}
    run_loss = 0.0; cnt = 0
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n, (batch,), device=train_gpu.device)
        b = train_gpu[idx].float()
        r, _ = ae(b)
        loss = F.mse_loss(r, b)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        run_loss += loss.item(); cnt += 1
        if step % eval_every == 0 or step == n_steps:
            ae.eval()
            v = evaluate(ae, val, mu)
            tr = evaluate(ae, train_eval, mu)
            curve["step"].append(step)
            curve["train_mse"].append(run_loss / cnt)
            curve["val_mse"].append(v["mse"])
            curve["val_rel_l2"].append(v["rel_l2"])
            curve["val_cos"].append(v["cos"])
            curve["train_rel_l2"].append(tr["rel_l2"])
            run_loss = 0.0; cnt = 0
            ae.train()
    ae.eval()
    return curve, sum(p.numel() for p in ae.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100, 200, 500])
    ap.add_argument("--n_steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts = torch.from_numpy(np.load(os.path.join(CACHE, f"acts_qwen_L{args.layer}.npy"))).float()
    d_model = acts.shape[1]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(acts.shape[0], generator=g)
    n_val = acts.shape[0] // 10
    val = acts[perm[:n_val]].contiguous()
    train = acts[perm[n_val:]].contiguous()
    mu = train.mean(0, keepdim=True).cuda()
    train_gpu = train.half().cuda()
    train_eval = train[:n_val].contiguous()
    print(f"layer {args.layer}: train {train.shape} val {val.shape} d={d_model} "
          f"steps={args.n_steps} ks={args.ks}", flush=True)

    out = {"layer": args.layer, "n_steps": args.n_steps, "batch": args.batch,
           "lr": args.lr, "eval_every": args.eval_every, "curves": {}}
    t0 = time.time()
    for k in args.ks:
        curve, npar = train_with_curve(train_gpu, val, train_eval, mu, d_model, k,
                                       args.n_steps, args.batch, args.lr,
                                       args.eval_every, args.seed)
        out["curves"][str(k)] = {"n_params": npar, **curve}
        print(f"  k={k:4d} final val_rel_l2={curve['val_rel_l2'][-1]:.4f} "
              f"train_rel_l2={curve['train_rel_l2'][-1]:.4f} "
              f"cos={curve['val_cos'][-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(RESULTS, f"qwen_lcurve_L{args.layer}.json"), "w"), indent=1)
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
