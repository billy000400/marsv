#!/usr/bin/env python3
"""
dir12 — Feedback 07161834: figure documenting that the primary runs now
converge SMOOTHLY. Shows only the winning-scheduler runs (ReduceLROnPlateau
factor 0.5, patience 100): per-step full-train loss (MSE seeds 0-2 + CE seed
0), the LR cascades, and test accuracy. No oscillating traces anywhere.

Output: plots/smooth_convergence.png + results/smooth_convergence.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plateau_protocol import HERE

PLOTS = os.path.join(HERE, 'plots')
DIR60K = '--dir60k' in sys.argv                  # full-60k runs (feedback 1)
argv = [a for a in sys.argv[1:] if a != '--dir60k']
SFX = argv[0] if argv else '_pl_f0.5_p100'       # scheduler suffix of the runs
OUT = '_60k' if DIR60K else ''
if DIR60K:
    RUNS = [  # (record dir under full_mnist_from_scratch, label, color, ls)
        (f'seed_0{SFX}',    'MSE seed 0', 'C0', '-'),
        (f'seed_1{SFX}',    'MSE seed 1', 'C2', '-'),
        (f'seed_2{SFX}',    'MSE seed 2', 'C1', '-'),
        (f'seed_0_ce{SFX}', 'CE seed 0',  'C3', '--'),
    ]
    BASE = os.path.join(HERE, 'results', 'full_mnist_from_scratch')
else:
    RUNS = [  # (ckpt dir, label, color, ls)
        ('seed0_pl_f0.5_p100',    'MSE seed 0', 'C0', '-'),
        ('seed1_pl_f0.5_p100',    'MSE seed 1', 'C2', '-'),
        ('seed2_pl_f0.5_p100',    'MSE seed 2', 'C1', '-'),
        ('seed0_ce_pl_f0.5_p100', 'CE seed 0',  'C3', '--'),
    ]
    BASE = os.path.join(HERE, 'results', 'ckpts_movie')

summary = {}
fig, axg = plt.subplots(1, 3, figsize=(16, 4.2))
for ck, label, col, ls in RUNS:
    z = np.load(os.path.join(BASE, ck, 'sched_trace.npz'))
    tr, lr = z['full_train_loss'], z['lr']
    hist = json.load(open(os.path.join(BASE, ck, 'history.json')))
    cmin = np.minimum.accumulate(tr)
    ratio = tr[1000:] / np.maximum(cmin[1000:], 1e-30)
    red = np.where(lr[1:] < lr[:-1])[0] + 1
    summary[label] = {
        'spike_max': float(ratio.max()),
        'spike_frac_gt2': float((ratio > 2).mean()),
        'tail_range_last5k': float(tr[-5000:].max() / tr[-5000:].min()),
        'final_train_loss': float(tr[-1]), 'final_lr': float(lr[-1]),
        'lr_cascade': [int(red[0]), int(red[-1])],
        'final_test_acc': hist['test_acc'][-1],
    }
    st = np.arange(len(tr))
    axg[0].plot(st, tr, lw=0.8, color=col, ls=ls, label=label)
    axg[1].plot(st, lr, lw=1.3, color=col, ls=ls, label=label)
    axg[2].plot(hist['step'], hist['test_acc'], lw=1.3, color=col, ls=ls, label=label)

for ax, title, ylab in [
        (axg[0], 'Full-train loss at every step (all smooth)', 'train loss (log)'),
        (axg[1], 'Learning-rate cascade', 'LR (log)'),
        (axg[2], 'Test accuracy (first 2,000 test images)', 'test accuracy')]:
    ax.set_xscale('symlog', linthresh=10); ax.grid(alpha=0.3)
    ax.set_xlabel('training step'); ax.set_title(title); ax.set_ylabel(ylab)
    ax.legend(fontsize=8)
axg[0].set_yscale('log'); axg[1].set_yscale('log')
axg[2].set_ylim(0, 1.0)
fig.suptitle('Converged training'
             + (' on all 60,000 MNIST images' if DIR60K else '')
             + ': ReduceLROnPlateau on the per-step full-train loss — '
             'the primary runs of this report', y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, f'smooth_convergence{OUT}.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)

with open(os.path.join(HERE, 'results', f'smooth_convergence{OUT}.json'), 'w') as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
print(f'saved plots/smooth_convergence{OUT}.png')
