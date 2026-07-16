#!/usr/bin/env python3
"""
dir12 — Early-training zoom (operator feedback 07161530, points 2 & 8):
deterministically retrain seed 0 and record the frozen SLERP protocol every 5
steps from 0 to 1,000 — 201 LINEARLY spaced frames covering the period from
the start of training until train/test accuracy has been flat for hundreds of
steps. Renders a linear-time animation with accuracy/confidence AND
train/test-loss insets, plus a static linear-time heatmap.

Records -> results/plateau_records/seed_<S>_early/ (compact: d-curves + preds
+ metrics; the run is deterministic and verified bit-exact against the main
movie records at the overlapping steps).

Outputs: plots/plateau_evolution_early.gif, plots/plateau_early_heatmap.png
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
import numpy as np
import torch

from plateau_protocol import (HERE, DATA, N_POINTS, N_TEST_POOL, ANIM_PAIRS,
                              build_pair_bank, record_checkpoint)
from train_and_record import eval_metrics
from src.mnist import MLP, load_mnist

torch.set_num_threads(2)
PLOTS = os.path.join(HERE, 'plots')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--end', type=int, default=1000)
    ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--fps', type=int, default=12)
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
    test_x = test_x[:N_TEST_POOL].to(device)
    test_y = test_y[:N_TEST_POOL].to(device)

    model = MLP(depth=4, width=200, activation='relu').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = torch.nn.MSELoss()
    one_hots = torch.eye(10, device=device)

    rec_dir = os.path.join(HERE, 'results', 'plateau_records',
                           f'seed_{args.seed}_early')
    os.makedirs(rec_dir, exist_ok=True)
    steps = list(range(0, args.end + 1, args.every))
    step_set = set(steps)

    hist = {'step': [], 'train_loss': [], 'train_acc': [], 'test_loss': [],
            'test_acc': [], 'train_conf': [], 'test_conf': []}
    d_logit, d_h3, d_h2, pred = [], [], [], []

    def checkpoint(step):
        model.eval()
        rec, _ = record_checkpoint(model, ex_a, ex_b)
        model.train()
        tr_acc, tr_conf, tr_loss = eval_metrics(model, train_x, train_y, one_hots, loss_fn)
        te_acc, te_conf, te_loss = eval_metrics(model, test_x, test_y, one_hots, loss_fn)
        for k, v in [('step', step), ('train_loss', tr_loss), ('train_acc', tr_acc),
                     ('test_loss', te_loss), ('test_acc', te_acc),
                     ('train_conf', tr_conf), ('test_conf', te_conf)]:
            hist[k].append(v)
        np.savez_compressed(os.path.join(rec_dir, f'step_{step}.npz'),
                            d_logit=rec['d_logit'], d_h3=rec['d_h3'],
                            d_h2=rec['d_h2'], pred=rec['pred'],
                            step=np.int64(step))
        d_logit.append(rec['d_logit']); d_h3.append(rec['d_h3'])
        d_h2.append(rec['d_h2']); pred.append(rec['pred'])

    checkpoint(0)
    for step in range(1, args.end + 1):
        idx = torch.randint(0, 1000, (200,), device=device)
        logits = model(train_x[idx])
        loss = loss_fn(logits, one_hots[train_y[idx]])
        opt.zero_grad(); loss.backward(); opt.step()
        if step in step_set:
            checkpoint(step)
    print(f'recorded {len(steps)} checkpoints (every {args.every} steps to '
          f'{args.end})', flush=True)

    with open(os.path.join(rec_dir, 'manifest.json'), 'w') as f:
        json.dump({'seed': args.seed, 'ckpt_steps': steps,
                   'every': args.every}, f)
    with open(os.path.join(rec_dir, 'metrics.json'), 'w') as f:
        json.dump(hist, f)

    D = np.stack(d_logit)
    PRED = np.stack(pred)
    steps = np.array(steps)

    # consistency check vs the main-movie records at overlapping steps
    for s in [0, 10, 30, 100, 300, 500, 1000]:
        ref = np.load(os.path.join(HERE, 'results', 'plateau_records',
                                   f'seed_{args.seed}', f'step_{s}.npz'))['d_logit']
        j = int(np.where(steps == s)[0][0])
        print(f'step {s}: max|early - movie record| = '
              f'{np.abs(D[j] - ref).max():.2e}')

    def pair_index(a, b):
        for i, p in enumerate(pairs):
            if p['class_a'] == min(a, b) and p['class_b'] == max(a, b):
                return i
        raise KeyError((a, b))
    anim_idx = [pair_index(a, b) for a, b in ANIM_PAIRS]
    t = np.linspace(0, 1, N_POINTS)

    # ---------------- animation (linear time scale) ----------------
    fig = plt.figure(figsize=(15, 5.6))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.3)
    lines, dots = [], []
    for k, (a, b) in enumerate(ANIM_PAIRS):
        ax = fig.add_subplot(gs[k // 5, k % 5])
        ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
        ln, = ax.plot([], [], lw=2, color='C0')
        sc = ax.scatter(t, np.full(N_POINTS, -0.09), s=6, c=np.zeros(N_POINTS),
                        cmap='tab10', vmin=0, vmax=9, marker='s')
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.15, 1.05)
        ax.set_title(f'{a} → {b}', fontsize=10)
        ax.grid(alpha=0.25)
        if k % 5 == 0:
            ax.set_ylabel(r'$d(\alpha)$', fontsize=9)
        if k // 5 == 1:
            ax.set_xlabel(r'$\alpha$', fontsize=9)
        ax.tick_params(labelsize=7)
        lines.append(ln); dots.append(sc)
    ax_acc = fig.add_subplot(gs[0, 5])
    ax_acc.plot(hist['step'], hist['test_acc'], color='C3', lw=1.5, label='test acc')
    ax_acc.plot(hist['step'], hist['train_acc'], color='C3', lw=1, ls='--', alpha=0.6,
                label='train acc')
    ax_acc.plot(hist['step'], hist['test_conf'], color='C4', lw=1.5, label='test conf')
    ax_acc.set_xlim(0, args.end); ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=6, loc='lower right')
    ax_acc.tick_params(labelsize=7); ax_acc.grid(alpha=0.25)
    vline_a = ax_acc.axvline(0, color='k', lw=1)
    ax_loss = fig.add_subplot(gs[1, 5])
    ax_loss.plot(hist['step'], hist['train_loss'], color='C0', lw=1.5, label='train loss')
    ax_loss.plot(hist['step'], hist['test_loss'], color='C1', lw=1.5, label='test loss')
    ax_loss.set_yscale('log'); ax_loss.set_xlim(0, args.end)
    ax_loss.set_xlabel('step', fontsize=8)
    ax_loss.legend(fontsize=6, loc='upper right')
    ax_loss.tick_params(labelsize=7); ax_loss.grid(alpha=0.25)
    vline_l = ax_loss.axvline(0, color='k', lw=1)
    title = fig.suptitle('', fontsize=13)

    def draw(f):
        for k, i in enumerate(anim_idx):
            lines[k].set_data(t, D[f, i])
            dots[k].set_array(PRED[f, i].astype(float))
        vline_a.set_xdata([steps[f]] * 2)
        vline_l.set_xdata([steps[f]] * 2)
        title.set_text(f'Early training, LINEAR time (seed {args.seed}) — step '
                       f'{steps[f]:,} of {args.end:,}   |   logit-space '
                       f'$d(\\alpha)$, squares: predicted class')
        return lines

    ani = animation.FuncAnimation(fig, draw, frames=len(steps), blit=False)
    ani.save(os.path.join(PLOTS, 'plateau_evolution_early.gif'), fps=args.fps,
             writer='pillow', dpi=72)
    plt.close(fig)
    print('saved plateau_evolution_early.gif')

    # ---------------- static linear-time heatmap ----------------
    show_pairs = ANIM_PAIRS + [(0, 0), (7, 7)]
    fig, axg = plt.subplots(2, 6, figsize=(16, 5.4), sharex=True, sharey=True)
    half = args.every / 2
    for k, (a, b) in enumerate(show_pairs):
        ax = axg[k // 6, k % 6]
        i = pair_index(a, b)
        im = ax.imshow(D[:, i, :], aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=0, vmax=1, extent=[0, 1, -half, args.end + half])
        ax.set_title(f'{a} → {b}' + (' (within)' if a == b else ''), fontsize=10)
        ax.tick_params(labelsize=7)
        if k % 6 == 0:
            ax.set_ylabel('training step (linear)', fontsize=8)
        if k // 6 == 1:
            ax.set_xlabel(r'$\alpha$', fontsize=9)
    cb = fig.colorbar(im, ax=axg, fraction=0.015, pad=0.01)
    cb.set_label(r'$d(\alpha)$ (logits)', fontsize=9)
    fig.suptitle('Early training on a LINEAR time axis: logit-space '
                 '$d(\\alpha)$, one row per 5 steps (seed %d)' % args.seed,
                 y=0.99)
    plt.savefig(os.path.join(PLOTS, 'plateau_early_heatmap.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)
    print('saved plateau_early_heatmap.png')


if __name__ == '__main__':
    main()
