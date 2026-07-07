"""Bottleneck-k sweep of the colleague's DeepAutoencoder on Qwen3-1.7B last-token
activations. Faithful to autoencoder_share: arch 2048->4096->4096->2048->k (GELU?
no — ReLU per DeepAutoencoder), MSE on RAW activations, Adam lr=3e-4, CosineAnnealing.

Reports, on a held-out 10% split, three reconstruction metrics vs k and a Kneedle
elbow for each, so we can see whether Qwen layer-2/10 last-token activations show a
genuine elbow (unlike our GPT-2 all-token layer-6 sweep, which only softly bends).

Usage: python ae_sweep_qwen.py --layer 2 --ks 5 10 15 20 25 30 --n_steps 6000
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
torch.backends.cuda.matmul.allow_tf32 = True   # ~4x fp32 matmul on Ampere, negligible acc change
torch.backends.cudnn.allow_tf32 = True
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def kneedle_log2(ks, ys):
    """Elbow on the (log2 k, y) curve: max |normalized curve - first->last chord|.
    Works for monotone increasing or decreasing y. Returns the elbow k."""
    x = np.log2(np.asarray(ks, float)); y = np.asarray(ys, float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    if yn[-1] < yn[0]:      # decreasing -> flip so it's increasing
        yn = 1 - yn
    d = yn - xn             # distance below the diagonal chord
    return int(ks[int(np.argmax(np.abs(d)))])


@torch.no_grad()
def evaluate(ae, val, mu, chunk=8192):
    """val FVU (vs train mean mu), per-sample rel-L2, per-sample cosine — all on RAW acts."""
    n = val.shape[0]; se = 0.0; rel = 0.0; cos = 0.0; denom = 0.0
    mu = mu.cuda()
    for i in range(0, n, chunk):
        b = val[i:i + chunk].cuda()
        r, _ = ae(b)
        se += ((b - r) ** 2).sum().item()
        denom += ((b - mu) ** 2).sum().item()
        rel += ((b - r).norm(dim=-1) / b.norm(dim=-1)).sum().item()
        cos += F.cosine_similarity(b, r, dim=-1).sum().item()
    return {"val_fvu": se / denom, "val_rel_l2": rel / n, "val_cos": cos / n}


def train_one(acts_train, mu, d_model, k, n_steps, batch, lr, seed):
    torch.manual_seed(seed)
    ae = DeepAutoencoder(d_in=d_model, k=k, hidden_dims=[4096, 4096, 2048]).cuda()
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    n = acts_train.shape[0]
    for step in range(n_steps):
        idx = torch.randint(0, n, (batch,))
        b = acts_train[idx].cuda()
        r, _ = ae(b)
        loss = F.mse_loss(r, b)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    ae.eval()
    return ae, sum(p.numel() for p in ae.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    ap.add_argument("--n_steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--acts", default=None, help="override activation npy path")
    ap.add_argument("--inject_massive", type=float, default=0.0,
                    help="if >0, scale coord 0 so it carries ~this variance fraction "
                         "(mimics GPT-2 massive-activation dim)")
    args = ap.parse_args()

    path = args.acts or os.path.join(CACHE, f"acts_qwen_L{args.layer}.npy")
    acts = np.load(path)
    acts = torch.from_numpy(acts).float()
    if args.inject_massive > 0:
        # rescale one coordinate so its variance = (frac/(1-frac)) * (variance of the rest)
        rest_var = acts.var(0).sum().item() - acts[:, 0].var().item()
        target = args.inject_massive / (1 - args.inject_massive) * rest_var
        acts[:, 0] *= (target / acts[:, 0].var().item()) ** 0.5
    d_model = acts.shape[1]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(acts.shape[0], generator=g)
    n_val = acts.shape[0] // 10
    val = acts[perm[:n_val]].contiguous()
    train = acts[perm[n_val:]].contiguous()
    mu = train.mean(0, keepdim=True).cuda()
    # top-1 variance fraction of the RAW activations (massive-activation diagnostic)
    var = train.var(0); top1 = (var.max() / var.sum()).item()
    print(f"layer {args.layer}: train {train.shape} val {val.shape} d={d_model} "
          f"top1_var_frac={top1:.4f}", flush=True)

    rows = []
    t0 = time.time()
    for k in args.ks:
        ae, npar = train_one(train, mu, d_model, k, args.n_steps, args.batch, args.lr, args.seed)
        m = evaluate(ae, val, mu)
        m.update(k=k, n_params=npar)
        rows.append(m)
        del ae; torch.cuda.empty_cache()
        print(f"  k={k:3d} FVU={m['val_fvu']:.4f} rel_L2={m['val_rel_l2']:.4f} "
              f"cos={m['val_cos']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"layer": args.layer, "n_steps": args.n_steps, "top1_var_frac": top1,
                   "rows": rows}, open(os.path.join(
                       RESULTS, f"qwen_sweep_L{args.layer}{args.tag}.json"), "w"), indent=1)

    ks = [r["k"] for r in rows]
    elbow = {m: kneedle_log2(ks, [r[m] for r in rows])
             for m in ("val_fvu", "val_rel_l2", "val_cos")}
    out = {"layer": args.layer, "n_steps": args.n_steps, "top1_var_frac": top1,
           "elbow": elbow, "rows": rows}
    json.dump(out, open(os.path.join(RESULTS, f"qwen_sweep_L{args.layer}{args.tag}.json"), "w"), indent=1)
    print("elbow:", elbow, flush=True)


if __name__ == "__main__":
    main()
