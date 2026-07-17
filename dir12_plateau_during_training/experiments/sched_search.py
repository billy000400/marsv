#!/usr/bin/env python3
"""
dir12 — Feedback 07161834: "only show the result that is smoothly converged;
if none, optimize your LR scheduler to find a better one".

Scores every available seed-0 MSE scheduler run on (a) smoothness of the
per-step full-train loss, (b) convergence, (c) final plateau fraction, and
saves a comparison figure + JSON so the winner can be picked on numbers.

Smoothness/convergence metrics (per-step full-train loss trace L_s):
  spike_max  = max_{s>1000} L_s / min_{u<=s} L_u   (1 = perfectly monotone)
  spike_frac = fraction of steps s>1000 with L_s > 2 * min_{u<=s} L_u
  tail_range = max(L)/min(L) over the last 5,000 steps (converged -> ~1)

Outputs plots/lr_scheduler_search.png + results/lr_scheduler_search.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plateau_protocol import HERE

PLOTS = os.path.join(HERE, 'plots')

RUNS = [  # (suffix, label, color)
    ('',                'constant LR 1e-3',                        'C7'),
    ('_sched',          'ReduceLROnPlateau f=0.5 p=10',            'C1'),
    ('_pl_f0.5_p100',   'ReduceLROnPlateau f=0.5 p=100',           'C2'),
    ('_pl_f0.9_p50',    'ReduceLROnPlateau f=0.9 p=50',            'C3'),
    ('_cos',            'cosine anneal 1e-3 -> 1e-6',              'C0'),
]


def load_trace(sfx):
    ck = os.path.join(HERE, 'results', 'ckpts_movie', f'seed0{sfx}')
    if sfx == '':
        z = np.load(os.path.join(ck, 'base_trace.npz'))
        return z['full_train_loss'], None
    z = np.load(os.path.join(ck, 'sched_trace.npz'))
    return z['full_train_loss'], z['lr']


def load_pf(sfx):
    rec_dir = os.path.join(HERE, 'results', 'plateau_records', f'seed_0{sfx}')
    man = json.load(open(os.path.join(rec_dir, 'manifest.json')))
    steps, pf = [], []
    for s in man['ckpt_steps']:
        z = np.load(os.path.join(rec_dir, f'step_{s}.npz'))
        d = z['d_logit'][:45]
        steps.append(s)
        pf.append(float(((d < 0.1) | (d > 0.9)).mean()))
    return np.array(steps), np.array(pf)


def main():
    summary = {}
    fig, axg = plt.subplots(1, 3, figsize=(16, 4.2))
    for sfx, label, col in RUNS:
        try:
            tr, lr = load_trace(sfx)
            steps, pf = load_pf(sfx)
        except FileNotFoundError as e:
            print(f'skip {label}: {e}')
            continue
        hist = json.load(open(os.path.join(HERE, 'results', 'ckpts_movie',
                                           f'seed0{sfx}', 'history.json')))
        cmin = np.minimum.accumulate(tr)
        ratio = tr[1000:] / np.maximum(cmin[1000:], 1e-30)
        tail = tr[-5000:]
        summary[label] = {
            'suffix': sfx,
            'spike_max': float(ratio.max()),
            'spike_frac': float((ratio > 2).mean()),
            'tail_range': float(tail.max() / max(tail.min(), 1e-30)),
            'final_train_loss': float(tr[-1]),
            'final_lr': float(lr[-1]) if lr is not None else 1e-3,
            'pf_100k': float(pf[-1]),
            'final_test_acc': hist['test_acc'][-1],
        }
        axg[0].plot(np.arange(len(tr)), tr, lw=0.7, color=col, label=label)
        if lr is not None:
            axg[1].plot(np.arange(len(lr)), lr, lw=1.4, color=col, label=label)
        axg[2].plot(steps, pf, lw=1.4, color=col, label=label)

    for ax, title, ylab in [(axg[0], 'Full-train loss, every step', 'MSE train loss (log)'),
                            (axg[1], 'Learning rate', 'LR (log)'),
                            (axg[2], 'Plateau fraction (logit d)', 'PF')]:
        ax.set_xscale('symlog', linthresh=10); ax.grid(alpha=0.3)
        ax.set_xlabel('training step'); ax.set_title(title); ax.set_ylabel(ylab)
        ax.legend(fontsize=7)
    axg[0].set_yscale('log'); axg[1].set_yscale('log')
    axg[2].axhline(0.20, color='gray', ls=':', lw=1)
    fig.suptitle('LR-scheduler search (seed 0, MSE, identical init/data/batches)', y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, 'lr_scheduler_search.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(HERE, 'results', 'lr_scheduler_search.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print('saved plots/lr_scheduler_search.png')


if __name__ == '__main__':
    main()
