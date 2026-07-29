"""S2 — train the five target-sharpness models (identical data + hyperparameters).

Usage:
    python train.py --epochs 4000 --seeds 0 --tag pilot
    python train.py --epochs <chosen> --seeds 0 1 2 --tag main
"""
import argparse
import json
import os

import numpy as np
import torch

import common as C

def train_one(k, seed, n_epochs, eval_every, device, n_train=C.N_TRAIN, cosine=True):
    steps_per_epoch = n_train // C.BATCH
    data = C.build_dataset(seed, n_train)
    xtr = data['xtr'].to(device)
    xva = data['xva'].to(device)
    ytr = C.target_fn_t(data['btr'], k).to(device)[:, None]
    yva = C.target_fn_t(data['bva'], k).to(device)[:, None]

    torch.manual_seed(seed)                    # identical init across k
    model = C.MLP(n_out=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WD)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * steps_per_epoch)
             if cosine else None)
    lossf = torch.nn.MSELoss()
    g = torch.Generator(device='cpu').manual_seed(10_000 + seed)   # identical batch order across k

    hist = {'epoch': [], 'train_loss': [], 'val_loss': []}
    best = (np.inf, None, -1)
    for ep in range(n_epochs):
        perm = torch.randperm(n_train, generator=g).to(device)
        for s in range(steps_per_epoch):
            idx = perm[s * C.BATCH:(s + 1) * C.BATCH]
            loss = lossf(model(xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if sched is not None:
                sched.step()
        if ep % eval_every == 0 or ep == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                full_tr = lossf(model(xtr), ytr).item()
                vl = lossf(model(xva), yva).item()
            model.train()
            hist['epoch'].append(ep); hist['train_loss'].append(full_tr)
            hist['val_loss'].append(vl)
            if vl < best[0]:
                best = (vl, {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}, ep)
    return model, hist, best


def adequacy(hist):
    tr, va = np.array(hist['train_loss']), np.array(hist['val_loss'])
    ep = np.array(hist['epoch'])
    imin = int(va.argmin())
    tail = tr[-max(2, len(tr) // 10):]
    return {'train_tail_max_over_min': float(tail.max() / tail.min()),
            'val_min': float(va[imin]), 'val_min_epoch': int(ep[imin]),
            'val_final': float(va[-1]), 'val_final_over_min': float(va[-1] / va[imin]),
            'train_min': float(tr.min()), 'train_final': float(tr[-1]),
            'train_final_over_min': float(tr[-1] / tr.min()),
            'val_min_before_end': bool(imin < len(va) - 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=4000)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--tag', default='pilot')
    p.add_argument('--n_train', type=int, default=C.N_TRAIN)
    p.add_argument('--no_cosine', action='store_true')
    p.add_argument('--save_ckpt', action='store_true')
    p.add_argument('--ckpt_prefix', default='ckpt')
    p.add_argument('--ks', type=float, nargs='+', default=None,
                   help='subset of k values to train (default: all of C.K_VALUES)')
    a = p.parse_args()
    ks = a.ks if a.ks else list(C.K_VALUES)

    C.setup_torch()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(C.RESULTS, exist_ok=True)

    # merge into an existing history file for this tag so that adding k values
    # does not require retraining the ones already done
    path = os.path.join(C.RESULTS, f"train_{a.tag}.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for seed in a.seeds:
        for k in ks:
            model, hist, best = train_one(k, seed, a.epochs, a.eval_every, device,
                                          n_train=a.n_train, cosine=not a.no_cosine)
            ad = adequacy(hist)
            key = f"k{k:g}_s{seed}"
            out[key] = {'history': hist, 'adequacy': ad, 'k': k, 'seed': seed,
                        'epochs': a.epochs, 'n_train': a.n_train,
                        'cosine': not a.no_cosine}
            print(f"[{a.tag}] {key}: val_min={ad['val_min']:.5f}@ep{ad['val_min_epoch']} "
                  f"val_final={ad['val_final']:.5f} ratio={ad['val_final_over_min']:.3f} "
                  f"train_final/min={ad['train_final_over_min']:.3f}", flush=True)
            if a.save_ckpt:
                torch.save({'final': {kk: v.detach().cpu() for kk, v in model.state_dict().items()},
                            'best_val': best[1], 'best_val_epoch': best[2],
                            'k': k, 'seed': seed, 'epochs': a.epochs, 'n_train': a.n_train},
                           os.path.join(C.RESULTS, f"{a.ckpt_prefix}_{key}.pt"))
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f"saved results/train_{a.tag}.json")


if __name__ == '__main__':
    main()
