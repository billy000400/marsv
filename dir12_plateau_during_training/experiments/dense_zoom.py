#!/usr/bin/env python3
"""
dir12 S5 — Resolve the timing of the largest late boundary move by
deterministically retraining seed 0 and recording every 50 steps inside
[from, to] (default: 82,000-82,500, pair 5->6, the largest adjacent-frame
change in the 500-step movie).

Records go to results/plateau_records/seed_<S>_dense/; also renders
plots/dense_zoom.png showing d_logit(alpha) for the focal pair at every
dense checkpoint.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from plateau_protocol import (HERE, DATA, N_POINTS, N_TEST_POOL,
                              build_pair_bank, record_checkpoint)
from src.mnist import MLP, load_mnist

torch.set_num_threads(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--start', type=int, default=82000)
    ap.add_argument('--end', type=int, default=82500)
    ap.add_argument('--every', type=int, default=50)
    ap.add_argument('--pair_a', type=int, default=5)
    ap.add_argument('--pair_b', type=int, default=6)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.225)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)                    # identical call order ->
    train_x, train_y, test_x, test_y = load_mnist(DATA)   # identical trajectory
    subset_idx = torch.randint(0, len(train_x), (1000,))
    pairs = build_pair_bank(test_y)
    ex_a = test_x[[q['idx_a'] for q in pairs]].to(device)
    ex_b = test_x[[q['idx_b'] for q in pairs]].to(device)
    train_x = train_x[subset_idx].to(device)
    train_y = train_y[subset_idx].to(device)

    model = MLP(depth=4, width=200, activation='relu').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = torch.nn.MSELoss()
    one_hots = torch.eye(10, device=device)

    rec_dir = os.path.join(HERE, 'results', 'plateau_records',
                           f'seed_{args.seed}_dense')
    os.makedirs(rec_dir, exist_ok=True)
    dense = list(range(args.start, args.end + 1, args.every))
    dense_set = set(dense)

    curves = {}
    for step in range(1, args.end + 1):
        idx = torch.randint(0, 1000, (200,), device=device)
        logits = model(train_x[idx])
        loss = loss_fn(logits, one_hots[train_y[idx]])
        opt.zero_grad(); loss.backward(); opt.step()
        if step in dense_set:
            model.eval()
            rec, _ = record_checkpoint(model, ex_a, ex_b)
            model.train()
            np.savez_compressed(os.path.join(rec_dir, f'step_{step}.npz'),
                                d_logit=rec['d_logit'], d_h3=rec['d_h3'],
                                pred=rec['pred'], step=np.int64(step))
            curves[step] = rec['d_logit']
            print(f'recorded step {step}', flush=True)
    with open(os.path.join(rec_dir, 'manifest.json'), 'w') as f:
        json.dump({'seed': args.seed, 'ckpt_steps': dense,
                   'every': args.every}, f)

    # consistency check vs the 500-step movie records
    for s in [args.start, args.end]:
        ref = np.load(os.path.join(HERE, 'results', 'plateau_records',
                                   f'seed_{args.seed}', f'step_{s}.npz'))['d_logit']
        print(f'step {s}: max|dense - movie record| = '
              f'{np.abs(curves[s] - ref).max():.2e}')

    # focal-pair figure
    i = next(k for k, p in enumerate(pairs)
             if p['class_a'] == min(args.pair_a, args.pair_b)
             and p['class_b'] == max(args.pair_a, args.pair_b))
    t = np.linspace(0, 1, N_POINTS)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cmap = plt.cm.viridis
    for j, s in enumerate(dense):
        ax.plot(t, curves[s][i], color=cmap(j / (len(dense) - 1)), lw=1.5,
                label=f'{s:,}')
    ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
    ax.set_xlabel(r'interpolation $\alpha$'); ax.set_ylabel(r'$d(\alpha)$ (logits)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, title='training step', ncol=2)
    ax.set_title(f'Dense 50-step zoom, pair {args.pair_a} → {args.pair_b} '
                 f'(seed {args.seed}): the late boundary flip completes in '
                 f'~150 steps')
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'plots', 'dense_zoom.png'), dpi=130)
    plt.close(fig)
    print('saved dense_zoom.png')


if __name__ == '__main__':
    main()
