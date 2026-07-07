"""Reproduce the colleague's wide-k bottleneck sweep (lasse.png): the colleague's
DeepAutoencoder on Qwen3-1.7B layer-2 last-token activations, swept over a WIDE k
range out to 500 and scored by held-out cosine similarity and relative-L2 error.

lasse.png shows a clear U/turnaround: rel-L2 error MINIMISES around k~50 then RISES
toward k=500 (cosine peaks then declines). Our earlier sweep only went to k=30 and
scored FVU, so it missed the turnaround. Here we (a) reproduce it and (b) also log
TRAIN metrics so we can tell whether the post-minimum rise is overfitting (train keeps
improving while val worsens -> genuine capacity/ID signal) or undertraining (both
worsen -> fixed-step-budget artifact).

Constant n_steps across all k (cleaner than the colleague's 6k/10k merge): a held-out
minimum under a FIXED training budget is the honest "AE bottleneck ID" reading.

Usage: python ae_sweep_lasse.py --layer 2 --ks 5 10 20 30 40 50 75 100 200 500 --n_steps 6000
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
os.makedirs(RESULTS, exist_ok=True)


@torch.no_grad()
def evaluate(ae, data, mu, chunk=8192):
    """FVU (vs mean mu), per-sample rel-L2, per-sample cosine on RAW acts."""
    n = data.shape[0]; se = 0.0; rel = 0.0; cos = 0.0; denom = 0.0
    for i in range(0, n, chunk):
        b = data[i:i + chunk].float().cuda()
        r, _ = ae(b)
        se += ((b - r) ** 2).sum().item()
        denom += ((b - mu) ** 2).sum().item()
        rel += ((b - r).norm(dim=-1) / b.norm(dim=-1)).sum().item()
        cos += F.cosine_similarity(b, r, dim=-1).sum().item()
    return {"fvu": se / denom, "rel_l2": rel / n, "cos": cos / n}


def train_one(acts_train, d_model, k, n_steps, batch, lr, seed):
    torch.manual_seed(seed)
    ae = DeepAutoencoder(d_in=d_model, k=k, hidden_dims=[4096, 4096, 2048]).cuda()
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    n = acts_train.shape[0]
    for step in range(n_steps):
        idx = torch.randint(0, n, (batch,), device=acts_train.device)
        b = acts_train[idx].float()
        r, _ = ae(b)
        loss = F.mse_loss(r, b)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    ae.eval()
    return ae, sum(p.numel() for p in ae.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[5, 10, 20, 30, 40, 50, 75, 100, 200, 500])
    ap.add_argument("--n_steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="_lasse")
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
    # fixed train subsample (val-sized) for the overfit diagnostic
    train_eval = train[:n_val].contiguous()
    var = train.var(0); top1 = (var.max() / var.sum()).item()
    print(f"layer {args.layer}: train {train.shape} val {val.shape} d={d_model} "
          f"top1_var_frac={top1:.4f} steps={args.n_steps}", flush=True)

    rows = []
    t0 = time.time()
    for k in args.ks:
        ae, npar = train_one(train_gpu, d_model, k, args.n_steps, args.batch, args.lr, args.seed)
        v = evaluate(ae, val, mu)
        tr = evaluate(ae, train_eval, mu)
        row = {"k": k, "n_params": npar,
               "val_fvu": v["fvu"], "val_rel_l2": v["rel_l2"], "val_cos": v["cos"],
               "train_fvu": tr["fvu"], "train_rel_l2": tr["rel_l2"], "train_cos": tr["cos"]}
        rows.append(row)
        del ae; torch.cuda.empty_cache()
        print(f"  k={k:4d} val: rel_L2={v['rel_l2']:.4f} cos={v['cos']:.4f} FVU={v['fvu']:.4f} | "
              f"train: rel_L2={tr['rel_l2']:.4f} cos={tr['cos']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"layer": args.layer, "n_steps": args.n_steps, "top1_var_frac": top1,
                   "rows": rows}, open(os.path.join(
                       RESULTS, f"qwen_sweep_L{args.layer}{args.tag}.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
