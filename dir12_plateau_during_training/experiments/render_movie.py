#!/usr/bin/env python3
"""
dir12 S4 — Render the plateau-evolution animation and static summaries from
saved per-checkpoint records (no model needed).

Outputs (plots/):
  plateau_evolution.gif        10 preregistered pairs, d_logit(alpha) per frame
  plateau_training_heatmap.png step x alpha heatmap of d_logit per pair
  layerwise_selected_steps.png d at h2/h3/logits, early/middle/late
  training_context.png         train/test accuracy + confidence vs step
  frames_selected_steps.png    static early/middle/late animation frames
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
import numpy as np

from plateau_protocol import HERE, N_POINTS, ANIM_PAIRS

PLOTS = os.path.join(HERE, 'plots')


def load_records(seed, sfx='', dir60k=False):
    base = ('full_mnist_from_scratch' if dir60k else 'plateau_records')
    rec_dir = os.path.join(HERE, 'results', base, f'seed_{seed}{sfx}')
    man = json.load(open(os.path.join(rec_dir, 'manifest.json')))
    steps = man['ckpt_steps']
    d_logit, d_h3, d_h2, pred = [], [], [], []
    test_acc = []
    for s in steps:
        z = np.load(os.path.join(rec_dir, f'step_{s}.npz'))
        d_logit.append(z['d_logit']); d_h3.append(z['d_h3']); d_h2.append(z['d_h2'])
        pred.append(z['pred']); test_acc.append(float(z['test_acc']))
    return (man, np.array(steps), np.stack(d_logit), np.stack(d_h3),
            np.stack(d_h2), np.stack(pred), np.array(test_acc))


def pair_index(man, a, b):
    for i, p in enumerate(man['pairs']):
        if p['class_a'] == min(a, b) and p['class_b'] == max(a, b):
            return i
    raise KeyError((a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--suffix', default='', help="e.g. _ce for the CE-loss run")
    ap.add_argument('--dir60k', action='store_true',
                    help='render a full-60k run (results/full_mnist_from_scratch)')
    ap.add_argument('--out-suffix', default=None,
                    help='figure filename suffix (default: --suffix, +_60k if --dir60k)')
    args = ap.parse_args()
    seed, sfx = args.seed, args.suffix
    tag = (' — CE loss' if '_ce' in sfx else '') + \
          (' — full 60k' if args.dir60k else '') + \
          (' — LR scheduler' if ('_sched' in sfx or '_pl_' in sfx or
                                 '_cos' in sfx) else '')

    man, steps, D, D3, D2, PRED, test_acc = load_records(seed, sfx, args.dir60k)
    # confidence: max raw output for MSE runs; max softmax prob for CE runs
    conf_key = 'test_prob' if '_ce' in sfx else 'test_conf'
    conf_lab = 'test conf (max prob)' if '_ce' in sfx else 'test conf (max raw output)'
    if args.dir60k:
        hist = json.load(open(os.path.join(HERE, 'results', 'full_mnist_from_scratch',
                                           f'seed_{seed}{sfx}', 'history.json')))
    else:
        hist = json.load(open(os.path.join(HERE, 'results', 'ckpts_movie',
                                           f'seed{seed}{sfx}', 'history.json')))
    out = args.out_suffix if args.out_suffix is not None else \
        (('_60k' + sfx) if args.dir60k else sfx)
    SEL5 = [0, 100, 1500, 10500, 30000] if args.dir60k else \
        [0, 100, 1000, 20000, 100000]
    TICKS = [0, 100, 1000, 10000, 30000] if args.dir60k else \
        [0, 100, 1000, 10000, 100000]
    SEL3 = [100, 3000, 30000] if args.dir60k else [100, 5000, 100000]
    tr_lab = 'train acc (60k)' if args.dir60k else 'train acc (1000 subset)'
    t = np.linspace(0, 1, N_POINTS)
    anim_idx = [pair_index(man, a, b) for a, b in ANIM_PAIRS]
    n_ck = len(steps)

    # ---------------- animation ----------------
    fig = plt.figure(figsize=(15, 5.6))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.3)
    axes, lines, dots = [], [], []
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
        axes.append(ax); lines.append(ln); dots.append(sc)
    ax_acc = fig.add_subplot(gs[0, 5])
    ax_acc.plot(hist['step'], hist['test_acc'], color='C3', lw=1.5, label='test acc')
    ax_acc.plot(hist['step'], hist['train_acc'], color='C3', lw=1, ls='--', alpha=0.6,
                label='train acc')
    ax_acc.plot(hist['step'], hist[conf_key], color='C4', lw=1.5, label='test conf')
    ax_acc.set_xscale('symlog', linthresh=10)
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=6, loc='lower right')
    ax_acc.tick_params(labelsize=7); ax_acc.grid(alpha=0.25)
    vline = ax_acc.axvline(0, color='k', lw=1)
    ax_loss = fig.add_subplot(gs[1, 5])
    ax_loss.plot(hist['step'], hist['train_loss'], color='C0', lw=1.5, label='train loss')
    ax_loss.plot(hist['step'], hist['test_loss'], color='C1', lw=1.5, label='test loss')
    ax_loss.set_xscale('symlog', linthresh=10); ax_loss.set_yscale('log')
    ax_loss.set_xlabel('step', fontsize=8)
    ax_loss.legend(fontsize=6, loc='lower left')
    ax_loss.tick_params(labelsize=7); ax_loss.grid(alpha=0.25)
    vline_l = ax_loss.axvline(0, color='k', lw=1)
    title = fig.suptitle('', fontsize=13)

    def draw(f):
        for k, i in enumerate(anim_idx):
            lines[k].set_data(t, D[f, i])
            dots[k].set_array(PRED[f, i].astype(float))
        vline.set_xdata([max(steps[f], 1e-3)] * 2)
        vline_l.set_xdata([max(steps[f], 1e-3)] * 2)
        title.set_text(f'Plateau evolution (seed {seed}{tag}) — training step '
                       f'{steps[f]:,}   |   logit-space $d(\\alpha)$, squares: '
                       f'predicted class along the path')
        return lines

    ani = animation.FuncAnimation(fig, draw, frames=n_ck, blit=False)
    ani.save(os.path.join(PLOTS, f'plateau_evolution{out}.gif'), fps=args.fps,
             writer='pillow', dpi=72)
    plt.close(fig)
    print('saved plateau_evolution.gif')

    # ---------------- static frames (early / middle / late) ----------------
    sel = [int(np.argmin(np.abs(steps - s))) for s in SEL5]
    fig, axg = plt.subplots(len(sel), 10, figsize=(16, 2.0 * len(sel)),
                            sharex=True, sharey=True)
    for r, f in enumerate(sel):
        for k, i in enumerate(anim_idx):
            ax = axg[r, k]
            ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
            ax.plot(t, D[f, i], lw=1.6, color='C0')
            ax.scatter(t, np.full(N_POINTS, -0.09), s=4, c=PRED[f, i],
                       cmap='tab10', vmin=0, vmax=9, marker='s')
            ax.set_ylim(-0.15, 1.05); ax.grid(alpha=0.2); ax.tick_params(labelsize=6)
            if r == 0:
                a, b = ANIM_PAIRS[k]; ax.set_title(f'{a} → {b}', fontsize=9)
            if k == 0:
                ax.set_ylabel(f'step {steps[f]:,}\n' + r'$d(\alpha)$', fontsize=8)
    for k in range(10):
        axg[-1, k].set_xlabel(r'$\alpha$', fontsize=8)
    fig.suptitle('Selected animation frames: logit-space $d(\\alpha)$ '
                 '(squares: predicted class along the path)', y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f'frames_selected_steps{out}.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)
    print('saved frames_selected_steps.png')

    # ---------------- heatmap: step x alpha per pair ----------------
    show_pairs = ANIM_PAIRS + [(0, 0), (7, 7)]
    fig, axg = plt.subplots(2, 6, figsize=(16, 5.4), sharex=True, sharey=True)
    tick_steps = TICKS
    tick_rows = [int(np.argmin(np.abs(steps - s))) for s in tick_steps]
    for k, (a, b) in enumerate(show_pairs):
        ax = axg[k // 6, k % 6]
        i = pair_index(man, a, b)
        im = ax.imshow(D[:, i, :], aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=0, vmax=1, extent=[0, 1, -0.5, n_ck - 0.5])
        ax.set_title(f'{a} → {b}' + (' (within)' if a == b else ''), fontsize=10)
        ax.set_yticks(tick_rows); ax.set_yticklabels([f'{s:,}' for s in tick_steps],
                                                     fontsize=7)
        ax.tick_params(labelsize=7)
        if k % 6 == 0:
            ax.set_ylabel('training step\n(checkpoint row)', fontsize=8)
        if k // 6 == 1:
            ax.set_xlabel(r'$\alpha$', fontsize=9)
    cb = fig.colorbar(im, ax=axg, fraction=0.015, pad=0.01)
    cb.set_label(r'$d(\alpha)$ (logits)', fontsize=9)
    fig.suptitle('Plateau formation: relative endpoint distance $d(\\alpha)$ over '
                 'training (each panel: one pair; rows = checkpoints, non-uniform '
                 'early steps)', y=0.99)
    plt.savefig(os.path.join(PLOTS, f'plateau_training_heatmap{out}.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)
    print('saved plateau_training_heatmap.png')

    # ---------------- layerwise early / middle / late ----------------
    sel3 = [int(np.argmin(np.abs(steps - s))) for s in SEL3]
    fig, axg = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    i01 = pair_index(man, 0, 1)
    cross = np.arange(45)
    for c, f in enumerate(sel3):
        ax = axg[c]
        for D_l, name, col in [(D2, r'$h_2$', 'C2'), (D3, r'$h_3$', 'C1'),
                               (D, 'logits', 'C0')]:
            ax.plot(t, D_l[f, i01], color=col, lw=2, label=f'{name} (pair 0→1)')
            ax.plot(t, D_l[f, cross].mean(axis=0), color=col, lw=1.2, ls='--',
                    alpha=0.7, label=f'{name} (mean, 45 pairs)')
        ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
        ax.set_title(f'step {steps[f]:,}')
        ax.set_xlabel(r'$\alpha$'); ax.grid(alpha=0.3)
    axg[0].set_ylabel(r'$d(\alpha)$')
    axg[0].legend(fontsize=7)
    fig.suptitle('Layerwise sharpening of the plateau (seed %d): relative distance '
                 'at $h_2$, $h_3$ (last hidden), and logits' % seed)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f'layerwise_selected_steps{out}.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)
    print('saved layerwise_selected_steps.png')

    # ---------------- training context ----------------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(hist['step'], hist['train_acc'], lw=1.5, color='C0', label=tr_lab)
    ax.plot(hist['step'], hist['test_acc'], lw=1.5, color='C3', label='test acc (first 2000)')
    ax.plot(hist['step'], hist[conf_key], lw=1.5, color='C4', label=conf_lab)
    ax.set_xscale('symlog', linthresh=10)
    ax.set_xlabel('training step'); ax.set_ylabel('accuracy / confidence')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f'Training context (seed {seed}{tag})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f'training_context{out}.png'), dpi=130)
    plt.close(fig)
    print('saved training_context.png')

    # ---------------- transition-timing diagnostics (S5 input) ----------------
    # frame-to-frame mean |change| in d_logit over the 45 cross pairs
    diff = np.abs(np.diff(D[:, :45, :], axis=0)).mean(axis=2)   # [n_ck-1, 45]
    worst = diff.max(axis=1)
    order = np.argsort(worst)[::-1][:8]
    print('\nlargest adjacent-frame changes (step_from -> step_to: max-pair mean |dd|):')
    for j in order:
        print(f'  {steps[j]:>6,} -> {steps[j+1]:>6,}: {worst[j]:.3f} '
              f'(pair {man["pairs"][int(np.argmax(diff[j]))]["class_a"]}->'
              f'{man["pairs"][int(np.argmax(diff[j]))]["class_b"]})')


if __name__ == '__main__':
    main()
