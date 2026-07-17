#!/usr/bin/env python3
"""
dir12 — Feedback 07161650: static CE-run frames showing d(alpha) in LOGIT
space (blue) and PROBABILITY (softmax) space (red) side by side at selected
steps, for the ten preregistered animation pairs. Under CE the logit-space
curve stays near the diagonal while the probability-space curve develops
plateaus sharper than the MSE run's.

Output: plots/frames_selected_steps_ce_prob.png
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plateau_protocol import HERE, N_POINTS, ANIM_PAIRS

PLOTS = os.path.join(HERE, 'plots')
SEL_STEPS = [0, 100, 1000, 20000, 100000]


def rel_dist_np(x, eps=1e-10):
    d_a = np.linalg.norm(x - x[:, :1], axis=2)
    d_b = np.linalg.norm(x - x[:, -1:], axis=2)
    return d_a / (d_a + d_b + eps)


def main():
    sched_sfx = sys.argv[1] if len(sys.argv) > 1 else ''   # e.g. _pl_f0.5_p100
    rec_dir = os.path.join(HERE, 'results', 'plateau_records',
                           f'seed_0_ce{sched_sfx}')
    man = json.load(open(os.path.join(rec_dir, 'manifest.json')))
    idx = []
    for a, b in ANIM_PAIRS:
        idx.append(next(i for i, p in enumerate(man['pairs'])
                        if p['class_a'] == a and p['class_b'] == b))
    t = np.linspace(0, 1, N_POINTS)

    fig, axg = plt.subplots(len(SEL_STEPS), 10, figsize=(16, 2.0 * len(SEL_STEPS)),
                            sharex=True, sharey=True)
    for r, s in enumerate(SEL_STEPS):
        z = np.load(os.path.join(rec_dir, f'step_{s}.npz'))
        logit = z['logits'].astype(np.float32)
        logit -= logit.max(axis=2, keepdims=True)
        prob = np.exp(logit); prob /= prob.sum(axis=2, keepdims=True)
        d_prob = rel_dist_np(prob)
        for k, i in enumerate(idx):
            ax = axg[r, k]
            ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
            ax.plot(t, z['d_logit'][i], lw=1.5, color='C0',
                    label='logit space' if (r, k) == (0, 0) else None)
            ax.plot(t, d_prob[i], lw=1.5, color='C3',
                    label='probability space' if (r, k) == (0, 0) else None)
            ax.scatter(t, np.full(N_POINTS, -0.09), s=4, c=z['pred'][i],
                       cmap='tab10', vmin=0, vmax=9, marker='s')
            ax.set_ylim(-0.15, 1.05); ax.grid(alpha=0.2); ax.tick_params(labelsize=6)
            if r == 0:
                a, b = ANIM_PAIRS[k]; ax.set_title(f'{a} → {b}', fontsize=9)
            if k == 0:
                ax.set_ylabel(f'step {s:,}\n' + r'$d(\alpha)$', fontsize=8)
    for k in range(10):
        axg[-1, k].set_xlabel(r'$\alpha$', fontsize=8)
    fig.legend(loc='upper right', fontsize=10, ncol=2, bbox_to_anchor=(0.99, 1.035))
    fig.suptitle('CE-loss run (seed 0): $d(\\alpha)$ in logit space (blue) stays near '
                 'the diagonal, while probability space (red) develops sharp plateaus '
                 '(squares: predicted class along the path)', y=1.02, x=0.42)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f'frames_selected_steps_ce_prob{sched_sfx}.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'saved plots/frames_selected_steps_ce_prob{sched_sfx}.png')


if __name__ == '__main__':
    main()
