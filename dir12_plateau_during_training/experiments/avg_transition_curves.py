"""
Operator feedback human_feedback_2: the plateau-evolution animation only tracked
ONE image pair per transition. Here we instead evaluate 100 image pairs per
transition and plot the MEAN d(alpha) curve with a +/-1 std shadow band, at
every full-60k seed-0 weight checkpoint.

A "transition" = one of the ten fixed cross-class digit pairs shown in the main
animation (ANIM_PAIRS). For transition (a,b) we build 100 endpoint pairs by
pairing the rank-i class-a test image with the rank-i class-b test image
(i = 0..99), all inside the first 2,000 test images (the frozen test pool). The
per-transition digit pair is fixed before viewing results; only the number of
example image-pairs is increased from 1 to 100.

Outputs (numeric first, then figures):
  results/avg_transition_curves.npz      mean/std d(alpha) [n_steps, 10, 50]
  plots/avg_transition_curves.gif        animated mean+std band, 10 transitions
  plots/avg_transition_curves_frames.png selected-step static (report-safe)
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

sys.path.insert(0, '/workspace/mars-plateaus-image')
sys.path.insert(0, os.path.dirname(__file__))
from plateau_protocol import (record_checkpoint, load_state_model, ANIM_PAIRS,
                              N_POINTS, N_TEST_POOL)

HERE = '/workspace/marsv_agent_haoyang/dir12_plateau_during_training'
DATA = '/workspace/mars-plateaus-image/data/mnist'
PLOTS = os.path.join(HERE, 'plots')
CKPTS = os.path.join(HERE, 'results', 'full_mnist_from_scratch', 'seed_0', 'ckpts')
N_PER = 100                        # image pairs per transition

torch.set_num_threads(2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.225)


def build_100_bank(test_y):
    """For each ANIM transition (a,b): 100 endpoint pairs (rank-i a, rank-i b),
    ranks in test order within the first 2,000 test images. Returns idx_a, idx_b
    of shape [10, 100]."""
    pool_y = test_y[:N_TEST_POOL]
    by_class = {c: torch.where(pool_y == c)[0] for c in range(10)}
    idx_a = np.zeros((len(ANIM_PAIRS), N_PER), dtype=np.int64)
    idx_b = np.zeros((len(ANIM_PAIRS), N_PER), dtype=np.int64)
    for k, (a, b) in enumerate(ANIM_PAIRS):
        assert len(by_class[a]) >= N_PER and len(by_class[b]) >= N_PER, (a, b)
        idx_a[k] = by_class[a][:N_PER].cpu().numpy()
        idx_b[k] = by_class[b][:N_PER].cpu().numpy()
    return idx_a, idx_b


def main():
    from src.mnist import load_mnist
    _tx, _ty, test_x, test_y = load_mnist(DATA)
    idx_a, idx_b = build_100_bank(test_y)                    # [10,100]
    flat_a = idx_a.reshape(-1)
    flat_b = idx_b.reshape(-1)
    ex_a = test_x[flat_a].to(device)                         # [1000,784]
    ex_b = test_x[flat_b].to(device)

    steps = sorted(int(f[4:-3]) for f in os.listdir(CKPTS)
                   if f.startswith('step') and f.endswith('.pt'))
    print('checkpoints:', steps)

    n_tr = len(ANIM_PAIRS)
    mean = np.zeros((len(steps), n_tr, N_POINTS), dtype=np.float32)
    std = np.zeros_like(mean)
    for si, s in enumerate(steps):
        model = load_state_model(os.path.join(CKPTS, f'step{s}.pt'), device)
        rec, _ = record_checkpoint(model, ex_a, ex_b)        # d_logit [1000,50]
        d = rec['d_logit'].reshape(n_tr, N_PER, N_POINTS)    # [10,100,50]
        mean[si] = d.mean(axis=1)
        std[si] = d.std(axis=1)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        print(f'  step {s}: mean-PF '
              f'{np.mean((mean[si] < 0.1) | (mean[si] > 0.9)):.3f}')

    os.makedirs(os.path.dirname(os.path.join(HERE, 'results', 'x')), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, 'results', 'avg_transition_curves.npz'),
                        steps=np.array(steps), mean=mean, std=std,
                        anim_pairs=np.array(ANIM_PAIRS), n_per=N_PER)
    print('saved results/avg_transition_curves.npz')

    alpha = np.linspace(0, 1, N_POINTS)
    # ---- animation: 2x5 grid, mean line + std band, one frame per checkpoint --
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2), sharex=True, sharey=True)
    axes = axes.ravel()
    lines, bands, ttls = [], [], []
    for k, (a, b) in enumerate(ANIM_PAIRS):
        ax = axes[k]
        ax.plot([0, 1], [0, 1], ls=':', c='0.7', lw=1)       # diagonal ref
        (ln,) = ax.plot([], [], c='C2', lw=2)
        lines.append(ln)
        bands.append(None)
        ax.set_title(f'{a}→{b}', fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if k >= 5:
            ax.set_xlabel(r'$\alpha$')
        if k % 5 == 0:
            ax.set_ylabel(r'$d(\alpha)$ (logit)')
    sup = fig.suptitle('', fontsize=14)

    def draw(si):
        for k in range(n_tr):
            ax = axes[k]
            if bands[k] is not None:
                bands[k].remove()
            m, sd = mean[si, k], std[si, k]
            lines[k].set_data(alpha, m)
            bands[k] = ax.fill_between(alpha, np.clip(m - sd, 0, 1),
                                       np.clip(m + sd, 0, 1), color='C2', alpha=0.25)
        sup.set_text(f'Mean $d(\\alpha)$ over {N_PER} pairs/transition '
                     f'(±1 std band) — full-60k seed 0, step {steps[si]:,}')
        return lines

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ani = animation.FuncAnimation(fig, draw, frames=len(steps), blit=False)
    ani.save(os.path.join(PLOTS, 'avg_transition_curves.gif'), fps=2,
             dpi=90, writer='pillow')
    plt.close(fig)
    print('saved plots/avg_transition_curves.gif')

    # ---- static selected-step frames -----------------------------------------
    sel = [s for s in [0, 100, 300, 3000, 30000] if s in steps]
    sel_i = [steps.index(s) for s in sel]
    fig, axes = plt.subplots(len(sel), n_tr, figsize=(2.0 * n_tr, 2.0 * len(sel)),
                             sharex=True, sharey=True)
    for ri, si in enumerate(sel_i):
        for k, (a, b) in enumerate(ANIM_PAIRS):
            ax = axes[ri, k]
            ax.plot([0, 1], [0, 1], ls=':', c='0.7', lw=1)
            m, sd = mean[si, k], std[si, k]
            ax.plot(alpha, m, c='C2', lw=1.8)
            ax.fill_between(alpha, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1),
                            color='C2', alpha=0.25)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if ri == 0:
                ax.set_title(f'{a}→{b}', fontsize=10)
            if k == 0:
                ax.set_ylabel(f'step {steps[si]:,}\n' r'$d(\alpha)$', fontsize=9)
            if ri == len(sel) - 1:
                ax.set_xlabel(r'$\alpha$', fontsize=9)
    fig.suptitle(f'Mean $d(\\alpha)$ over {N_PER} image pairs per transition '
                 f'(±1 std shadow) — full-60k seed 0', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOTS, 'avg_transition_curves_frames.png'), dpi=130)
    plt.close(fig)
    print('saved plots/avg_transition_curves_frames.png')

    # summary numbers for the report
    pf_by_step = [float(np.mean((mean[si] < 0.1) | (mean[si] > 0.9)))
                  for si in range(len(steps))]
    mean_band = [float(std[si].mean()) for si in range(len(steps))]
    print('PF(mean curve) by step:', dict(zip(steps, [round(x, 3) for x in pf_by_step])))
    print('mean std width by step:', dict(zip(steps, [round(x, 3) for x in mean_band])))
    json.dump({'steps': steps, 'pf_mean_curve': pf_by_step,
               'mean_std_width': mean_band, 'n_per': N_PER},
              open(os.path.join(HERE, 'results', 'avg_transition_curves_summary.json'), 'w'),
              indent=2)


if __name__ == '__main__':
    main()
