#!/usr/bin/env python3
"""
dir12 — Early-training zoom of the full-60k run (feedback human_feedback_1:
rerun every 1k-only experiment on the 60k data). Loads the records written by
`train_full60k.py --early` (frozen protocol every 5 steps, 0-1,000, LINEAR
time) and renders the linear-time animation + heatmap, after verifying the
early records bit-match the main 60k run at the overlapping steps.

Usage: python experiments/render_early_60k.py [_pl_f0.5_p100]
Outputs: plots/plateau_evolution_early_60k.gif, plots/plateau_early_heatmap_60k.png
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
import numpy as np

from plateau_protocol import HERE, N_POINTS, ANIM_PAIRS

PLOTS = os.path.join(HERE, 'plots')
FM = os.path.join(HERE, 'results', 'full_mnist_from_scratch')
SFX = sys.argv[1] if len(sys.argv) > 1 else '_pl_f0.5_p100'
END, EVERY, FPS = 1000, 5, 12


def main():
    rec_dir = os.path.join(FM, f'seed_0{SFX}_early')
    man = json.load(open(os.path.join(rec_dir, 'manifest.json')))
    steps = np.array(man['ckpt_steps'])
    hist = json.load(open(os.path.join(rec_dir, 'history.json')))
    D, PRED = [], []
    for s in man['ckpt_steps']:
        z = np.load(os.path.join(rec_dir, f'step_{s}.npz'))
        D.append(z['d_logit']); PRED.append(z['pred'])
    D, PRED = np.stack(D), np.stack(PRED)

    # consistency check vs the main 60k run at overlapping steps
    main_dir = os.path.join(FM, f'seed_0{SFX}')
    for s in [0, 10, 30, 100, 300, 600, 900]:
        ref = np.load(os.path.join(main_dir, f'step_{s}.npz'))['d_logit']
        j = int(np.where(steps == s)[0][0])
        print(f'step {s}: max|early - main 60k record| = '
              f'{np.abs(D[j] - ref).max():.2e}')

    def pair_index(a, b):
        for i, p in enumerate(man['pairs']):
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
        ln, = ax.plot([], [], lw=2, color='C2')
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
    ax_acc.set_xlim(0, END); ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=6, loc='lower right')
    ax_acc.tick_params(labelsize=7); ax_acc.grid(alpha=0.25)
    vline_a = ax_acc.axvline(0, color='k', lw=1)
    ax_loss = fig.add_subplot(gs[1, 5])
    ax_loss.plot(hist['step'], hist['train_loss'], color='C0', lw=1.5, label='train loss')
    ax_loss.plot(hist['step'], hist['test_loss'], color='C1', lw=1.5, label='test loss')
    ax_loss.set_yscale('log'); ax_loss.set_xlim(0, END)
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
        title.set_text(f'Early training on all 60k images, LINEAR time (seed 0) — '
                       f'step {steps[f]:,} of {END:,}   |   logit-space '
                       f'$d(\\alpha)$, squares: predicted class')
        return lines

    ani = animation.FuncAnimation(fig, draw, frames=len(steps), blit=False)
    ani.save(os.path.join(PLOTS, 'plateau_evolution_early_60k.gif'), fps=FPS,
             writer='pillow', dpi=72)
    plt.close(fig)
    print('saved plateau_evolution_early_60k.gif')

    # ---------------- static linear-time heatmap ----------------
    show_pairs = ANIM_PAIRS + [(0, 0), (7, 7)]
    fig, axg = plt.subplots(2, 6, figsize=(16, 5.4), sharex=True, sharey=True)
    half = EVERY / 2
    for k, (a, b) in enumerate(show_pairs):
        ax = axg[k // 6, k % 6]
        i = pair_index(a, b)
        im = ax.imshow(D[:, i, :], aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=0, vmax=1, extent=[0, 1, -half, END + half])
        ax.set_title(f'{a} → {b}' + (' (within)' if a == b else ''), fontsize=10)
        ax.tick_params(labelsize=7)
        if k % 6 == 0:
            ax.set_ylabel('training step (linear)', fontsize=8)
        if k // 6 == 1:
            ax.set_xlabel(r'$\alpha$', fontsize=9)
    cb = fig.colorbar(im, ax=axg, fraction=0.015, pad=0.01)
    cb.set_label(r'$d(\alpha)$ (logits)', fontsize=9)
    fig.suptitle('Early training on all 60k images, LINEAR time axis: logit-space '
                 '$d(\\alpha)$, one row per 5 steps (seed 0)', y=0.99)
    plt.savefig(os.path.join(PLOTS, 'plateau_early_heatmap_60k.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)
    print('saved plateau_early_heatmap_60k.png')


if __name__ == '__main__':
    main()
