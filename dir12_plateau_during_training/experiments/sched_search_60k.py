#!/usr/bin/env python3
"""
dir12 — Feedback human_feedback_1 (2026-07-17): "the 60K training shows a
noisy training loss curve — use a more sophisticated LR scheduler to make it
smooth". LR-scheduler search on the full-60k from-scratch run (seed 0, MSE,
identical init/data/batch order): cosine (the previous primary schedule) vs
ReduceLROnPlateau variants, scored with the same smoothness/convergence
metrics as the 1k search (sched_search.py):

  spike_max  = max_{s>1000} L_s / min_{u<=s} L_u   (1 = perfectly monotone)
  spike_frac = fraction of steps s>1000 with L_s > 2 * min_{u<=s} L_u
  tail_range = max(L)/min(L) over the last 5,000 steps (converged -> ~1)

where L_s is the per-step FULL-train (60k) loss.

Outputs plots/lr_scheduler_search_60k.png + results/lr_scheduler_search_60k.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plateau_protocol import HERE

PLOTS = os.path.join(HERE, 'plots')
FM = os.path.join(HERE, 'results', 'full_mnist_from_scratch')

RUNS = [  # (record dir, label, color)
    ('seed_0_search',               'cosine 1e-3 -> 1e-6 (previous)',      'C0'),
    ('seed_0_pl_f0.5_p100_search',  'ReduceLROnPlateau f=0.5 p=100',       'C2'),
    ('seed_0_pl_f0.5_p300_search',  'ReduceLROnPlateau f=0.5 p=300',       'C3'),
]


def main():
    summary = {}
    fig, axg = plt.subplots(1, 3, figsize=(16, 4.2))
    for d, label, col in RUNS:
        rec_dir = os.path.join(FM, d)
        z = np.load(os.path.join(rec_dir, 'sched_trace.npz'))
        tr, lr = z['full_train_loss'], z['lr']
        man = json.load(open(os.path.join(rec_dir, 'manifest.json')))
        hist = json.load(open(os.path.join(rec_dir, 'history.json')))
        pf_steps, pf = [], []
        for s in man['ckpt_steps']:
            rec = np.load(os.path.join(rec_dir, f'step_{s}.npz'))
            dd = rec['d_logit'][:45]
            pf_steps.append(s)
            pf.append(float(((dd < 0.1) | (dd > 0.9)).mean()))
        cmin = np.minimum.accumulate(tr)
        ratio = tr[1000:] / np.maximum(cmin[1000:], 1e-30)
        tail = tr[-5000:]
        summary[label] = {
            'dir': d,
            'spike_max': float(ratio.max()),
            'spike_frac': float((ratio > 2).mean()),
            'tail_range': float(tail.max() / max(tail.min(), 1e-30)),
            'final_train_loss': float(tr[-1]),
            'final_lr': float(lr[-1]),
            'pf_final': float(pf[-1]),
            'final_test_acc': hist['test_acc'][-1],
        }
        axg[0].plot(np.arange(len(tr)), tr, lw=0.7, color=col, label=label)
        axg[1].plot(np.arange(len(lr)), lr, lw=1.4, color=col, label=label)
        axg[2].plot(pf_steps, pf, lw=1.4, color=col, label=label)

    for ax, title, ylab in [(axg[0], 'Full-train (60k) loss, every step',
                             'MSE train loss (log)'),
                            (axg[1], 'Learning rate', 'LR (log)'),
                            (axg[2], 'Plateau fraction (logit d)', 'PF')]:
        ax.set_xscale('symlog', linthresh=10); ax.grid(alpha=0.3)
        ax.set_xlabel('training step'); ax.set_title(title); ax.set_ylabel(ylab)
        ax.legend(fontsize=7)
    axg[0].set_yscale('log'); axg[1].set_yscale('log')
    axg[2].axhline(0.20, color='gray', ls=':', lw=1)
    fig.suptitle('LR-scheduler search on the full-60k run (seed 0, MSE, identical '
                 'init/data/batches)', y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, 'lr_scheduler_search_60k.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(HERE, 'results', 'lr_scheduler_search_60k.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print('saved plots/lr_scheduler_search_60k.png')


if __name__ == '__main__':
    main()
