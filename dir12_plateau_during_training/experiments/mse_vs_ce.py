#!/usr/bin/env python3
"""
dir12 — Feedback 07161650, point 1: train accuracy is flat at 1.0 while train
loss keeps falling. Is that an artifact of the MSE-on-one-hot loss, and does
cross-entropy (CE, the maximum-likelihood loss) behave differently?

Compares the seed-0 MSE run with a seed-0 CE run that shares the SAME
initialization, data subset, and batch order (identical RNG stream; only the
loss differs): train/test loss, train/test accuracy, confidence, and the
plateau fraction PF(t) from the frozen interpolation protocol.

Outputs plots/mse_vs_ce_training.png + results/mse_vs_ce.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plateau_protocol import HERE

PLOTS = os.path.join(HERE, 'plots')


def rel_dist_np(x, eps=1e-10):
    d_a = np.linalg.norm(x - x[:, :1], axis=2)
    d_b = np.linalg.norm(x - x[:, -1:], axis=2)
    return d_a / (d_a + d_b + eps)


DIR60K = '--dir60k' in sys.argv                  # full-60k runs (feedback 1)


def load(sfx):
    if DIR60K:
        rec_dir = os.path.join(HERE, 'results', 'full_mnist_from_scratch',
                               f'seed_0{sfx}')
        hist = json.load(open(os.path.join(rec_dir, 'history.json')))
    else:
        hist = json.load(open(os.path.join(HERE, 'results', 'ckpts_movie',
                                           f'seed0{sfx}', 'history.json')))
        rec_dir = os.path.join(HERE, 'results', 'plateau_records', f'seed_0{sfx}')
    man = json.load(open(os.path.join(rec_dir, 'manifest.json')))
    steps, pf, pf_prob = [], [], []
    for s in man['ckpt_steps']:
        z = np.load(os.path.join(rec_dir, f'step_{s}.npz'))
        d = z['d_logit'][:45]
        logit = z['logits'][:45].astype(np.float32)
        logit -= logit.max(axis=2, keepdims=True)
        prob = np.exp(logit); prob /= prob.sum(axis=2, keepdims=True)
        dp = rel_dist_np(prob)
        steps.append(s)
        pf.append(float(((d < 0.1) | (d > 0.9)).mean()))
        pf_prob.append(float(((dp < 0.1) | (dp > 0.9)).mean()))
    return hist, np.array(steps), np.array(pf), np.array(pf_prob)


def main():
    # optional scheduler suffix (feedback 07161834: compare the SMOOTH runs)
    argv = [a for a in sys.argv[1:] if a != '--dir60k']
    sched_sfx = argv[0] if argv else ''
    out = '_60k' if DIR60K else sched_sfx
    h_mse, st_mse, pf_mse, pfp_mse = load(sched_sfx)
    h_ce, st_ce, pf_ce, pfp_ce = load('_ce' + sched_sfx)

    def acc1_step(h):
        for s, a in zip(h['step'], h['train_acc']):
            if a >= 1.0:
                return s
        return None

    summary = {
        'train_acc_first_1.0': {'mse': acc1_step(h_mse), 'ce': acc1_step(h_ce)},
        'final_train_loss': {'mse': h_mse['train_loss'][-1], 'ce': h_ce['train_loss'][-1]},
        'final_test_acc': {'mse': h_mse['test_acc'][-1], 'ce': h_ce['test_acc'][-1]},
        'final_test_prob': {'mse': h_mse['test_prob'][-1] if 'test_prob' in h_mse else None,
                            'ce': h_ce['test_prob'][-1]},
        'pf_100k': {'mse': float(pf_mse[-1]), 'ce': float(pf_ce[-1])},
        'pf_prob_100k': {'mse': float(pfp_mse[-1]), 'ce': float(pfp_ce[-1])},
        'pf_curve_ce': {'steps': st_ce.tolist(), 'pf': pf_ce.tolist(),
                        'pf_prob': pfp_ce.tolist()},
    }
    with open(os.path.join(HERE, 'results', f'mse_vs_ce{out}.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'pf_curve_ce'}, indent=1))

    fig, axg = plt.subplots(2, 2, figsize=(12.5, 7.6))

    ax = axg[0, 0]
    ax.plot(h_mse['step'], h_mse['train_loss'], color='C0', lw=1.6, label='train loss')
    ax.plot(h_mse['step'], h_mse['test_loss'], color='C1', lw=1.6, label='test loss')
    _acc1 = summary['train_acc_first_1.0']['mse']
    if _acc1 is not None:                     # 60k MSE never hits exactly 1.0
        ax.axvline(_acc1, color='gray', ls='--', lw=1, label='train acc reaches 1.0')
    ax.set_xscale('symlog', linthresh=10); ax.set_yscale('log')
    ax.set_title('MSE run: loss keeps falling after train acc = 1.0')
    ax.set_ylabel('MSE loss (log)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axg[0, 1]
    ax.plot(h_ce['step'], h_ce['train_loss'], color='C0', lw=1.6, label='train loss')
    ax.plot(h_ce['step'], h_ce['test_loss'], color='C1', lw=1.6, label='test loss')
    ax.axvline(summary['train_acc_first_1.0']['ce'], color='gray', ls='--', lw=1,
               label='train acc reaches 1.0')
    ax.set_xscale('symlog', linthresh=10); ax.set_yscale('log')
    ax.set_title('CE run: same — loss falls for 100k steps at train acc = 1.0')
    ax.set_ylabel('CE loss (log)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axg[1, 0]
    ax.plot(h_mse['step'], h_mse['train_acc'], color='C0', lw=1.4, label='train acc (MSE)')
    ax.plot(h_mse['step'], h_mse['test_acc'], color='C0', lw=1.4, ls='--', label='test acc (MSE)')
    ax.plot(h_ce['step'], h_ce['train_acc'], color='C3', lw=1.4, label='train acc (CE)')
    ax.plot(h_ce['step'], h_ce['test_acc'], color='C3', lw=1.4, ls='--', label='test acc (CE)')
    ax.plot(h_ce['step'], h_ce['test_prob'], color='C4', lw=1.2, ls=':',
            label='test confidence (CE, max prob)')
    ax.set_xscale('symlog', linthresh=10)
    ax.set_xlabel('training step'); ax.set_ylabel('accuracy / probability')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Accuracy: identical story under both losses')

    ax = axg[1, 1]
    ax.plot(st_mse, pf_mse, color='C0', lw=1.4, label='PF, logit space (MSE)')
    ax.plot(st_mse, pfp_mse, color='C0', lw=1.2, ls='--', label='PF, prob space (MSE)')
    ax.plot(st_ce, pf_ce, color='C3', lw=1.4, label='PF, logit space (CE)')
    ax.plot(st_ce, pfp_ce, color='C3', lw=1.2, ls='--', label='PF, prob space (CE)')
    ax.axhline(0.20, color='gray', ls=':', lw=1, label='diagonal floor ≈ 0.20')
    ax.set_xscale('symlog', linthresh=10)
    ax.set_xlabel('training step'); ax.set_ylabel('plateau fraction PF')
    ax.set_ylim(0, 1.0); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title('Plateau fraction: logit space vs probability space')

    for ax in axg.ravel():
        ax.set_xlim(-0.5, 3.5e4 if DIR60K else 1.2e5)
    fig.suptitle('MSE-on-one-hot vs cross-entropy (seed 0, identical init/data/batches'
                 + (', full 60k images' if DIR60K else '')
                 + (', ReduceLROnPlateau' if sched_sfx else '')
                 + '; x-axes log-scale)', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f'mse_vs_ce_training{out}.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)
    print(f'saved plots/mse_vs_ce_training{out}.png')


if __name__ == '__main__':
    main()
