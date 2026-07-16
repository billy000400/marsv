#!/usr/bin/env python3
"""
dir12 S5 — Compare plateau emergence across seeds 0/1/2.

Left: plateau fraction vs training step per seed — the single transparent
curve-derived summary (PLAN allows at most one): the fraction of the 50 path
points with d_logit < 0.1 or > 0.9, averaged over the 45 cross-class pairs.
Right: overlaid d_logit(alpha) curves of the 45 cross pairs at matched steps,
one row per seed, to show the qualitative movie is stable.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plateau_protocol import HERE, N_POINTS

t = np.linspace(0, 1, N_POINTS)
SEL = [0, 100, 1000, 20000, 100000]


def load(seed):
    rec_dir = os.path.join(HERE, 'results', 'plateau_records', f'seed_{seed}')
    man = json.load(open(os.path.join(rec_dir, 'manifest.json')))
    steps = np.array(man['ckpt_steps'])
    D = np.stack([np.load(os.path.join(rec_dir, f'step_{s}.npz'))['d_logit']
                  for s in man['ckpt_steps']])
    return steps, D


fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(3, 6, width_ratios=[1.8, 1, 1, 1, 1, 1], wspace=0.3,
                      hspace=0.45)
axL = fig.add_subplot(gs[:, 0])
for seed, col in zip([0, 1, 2], ['C0', 'C1', 'C2']):
    steps, D = load(seed)
    frac = ((D[:, :45, :] < 0.1) | (D[:, :45, :] > 0.9)).mean(axis=(1, 2))
    axL.plot(np.maximum(steps, 0.5), frac, color=col, lw=1.5,
             label=f'seed {seed}' + (' (205 ckpts)' if seed == 0 else ' (56 ckpts)'))
    for r, s in enumerate(SEL):
        f = int(np.argmin(np.abs(steps - s)))
        ax = fig.add_subplot(gs[seed, r + 1])
        for i in range(45):
            ax.plot(t, D[f, i], color=col, lw=0.6, alpha=0.35)
        ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=6); ax.grid(alpha=0.2)
        if seed == 0:
            ax.set_title(f'step {s:,}', fontsize=9)
        if r == 0:
            ax.set_ylabel(f'seed {seed}\n' + r'$d(\alpha)$', fontsize=8)
        if seed == 2:
            ax.set_xlabel(r'$\alpha$', fontsize=8)
axL.set_xscale('symlog', linthresh=10)
axL.set_xlabel('training step'); axL.set_ylabel('plateau fraction')
axL.set_ylim(0, 1); axL.grid(alpha=0.3); axL.legend(fontsize=8)
axL.set_title('Plateau fraction: mean fraction of path points\n'
              r'with $d<0.1$ or $d>0.9$ (45 cross-class pairs)', fontsize=10)
fig.suptitle('Plateau emergence is consistent across seeds — gradual sharpening, '
             'no sudden transition (right: all 45 cross-pair $d(\\alpha)$ curves '
             'overlaid)', y=1.0)
plt.savefig(os.path.join(HERE, 'plots', 'seed_comparison.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)

# print emergence summary for the report
for seed in [0, 1, 2]:
    steps, D = load(seed)
    frac = ((D[:, :45, :] < 0.1) | (D[:, :45, :] > 0.9)).mean(axis=(1, 2))
    for thr in [0.25, 0.5]:
        above = np.where(frac >= thr)[0]
        s = steps[above[0]] if len(above) else None
        print(f'seed {seed}: plateau fraction first >= {thr} at step {s}; '
              f'final = {frac[-1]:.3f}')
print('saved seed_comparison.png')
