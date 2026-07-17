#!/usr/bin/env python3
"""
dir12 — Feedback 07161721: "trainings are too chaotic in the end, not
converging — use a ReduceLROnPlateau-style scheduler (LR shrinks if training
loss does not improve for ~10 steps)".

Compares the constant-LR seed-0 runs (MSE + CE) with seed-0 reruns that use
ReduceLROnPlateau(factor=0.5, patience=10, threshold=1e-4 rel, min_lr=1e-8)
stepped every optimization step on the full-train-set loss. All four runs
share the same init/data/batch order (the scheduler consumes no RNG).

For the baseline runs the per-step full-train loss was never stored, so it is
re-traced here by deterministically re-running the training loop (verified
against the stored checkpoint history).

Outputs plots/lr_scheduler_comparison.png + results/lr_scheduler.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from plateau_protocol import HERE, DATA
from src.mnist import MLP, load_mnist

torch.set_num_threads(2)
PLOTS = os.path.join(HERE, 'plots')
N_STEPS = 100_000


def retrace_baseline(loss_name, device):
    """Re-run the constant-LR seed-0 training loop, recording only the
    per-step full-train-set loss (identical RNG semantics to train_and_record)."""
    torch.manual_seed(0)
    train_x, train_y, _, _ = load_mnist(DATA)
    subset_idx = torch.randint(0, len(train_x), (1000,))
    train_x = train_x[subset_idx].to(device)
    train_y = train_y[subset_idx].to(device)
    model = MLP(depth=4, width=200, activation='relu').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    one_hots = torch.eye(10, device=device)
    if loss_name == 'ce':
        ce_fn = torch.nn.CrossEntropyLoss()
        loss_of = lambda logits, y: ce_fn(logits, y)
    else:
        mse_fn = torch.nn.MSELoss()
        loss_of = lambda logits, y: mse_fn(logits, one_hots[y])
    trace = np.zeros(N_STEPS + 1, np.float32)
    with torch.no_grad():
        trace[0] = loss_of(model(train_x), train_y).item()
    for step in range(1, N_STEPS + 1):
        idx = torch.randint(0, 1000, (200,), device=device)
        loss = loss_of(model(train_x[idx]), train_y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            trace[step] = loss_of(model(train_x), train_y).item()
    return trace


def rel_dist_np(x, eps=1e-10):
    d_a = np.linalg.norm(x - x[:, :1], axis=2)
    d_b = np.linalg.norm(x - x[:, -1:], axis=2)
    return d_a / (d_a + d_b + eps)


def load_run(sfx):
    """History, PF curves (logit + prob space) and d-curve stacks for one run."""
    hist = json.load(open(os.path.join(HERE, 'results', 'ckpts_movie',
                                       f'seed0{sfx}', 'history.json')))
    rec_dir = os.path.join(HERE, 'results', 'plateau_records', f'seed_0{sfx}')
    man = json.load(open(os.path.join(rec_dir, 'manifest.json')))
    steps, pf, pf_prob, d_all, dp_all = [], [], [], [], []
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
        d_all.append(d); dp_all.append(dp)
    return (hist, np.array(steps), np.array(pf), np.array(pf_prob),
            np.stack(d_all), np.stack(dp_all))


def motion(steps, D):
    """Curve motion between adjacent checkpoints: mean |d_{t+1}-d_t| over the
    45 cross pairs and 50 path points. Returned at the mid-step of each gap."""
    m = np.abs(np.diff(D, axis=0)).mean(axis=(1, 2))
    return 0.5 * (steps[1:] + steps[:-1]), m


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.225)

    runs = {}
    for key, sfx in [('mse', ''), ('mse_sched', '_sched'),
                     ('ce', '_ce'), ('ce_sched', '_ce_sched')]:
        runs[key] = load_run(sfx)

    # per-step traces: scheduled runs stored them; baselines are re-traced
    traces = {}
    for key, sfx in [('mse_sched', '_sched'), ('ce_sched', '_ce_sched')]:
        z = np.load(os.path.join(HERE, 'results', 'ckpts_movie',
                                 f'seed0{sfx}', 'sched_trace.npz'))
        traces[key] = (z['full_train_loss'], z['lr'])
    for key, loss_name in [('mse', 'mse'), ('ce', 'ce')]:
        print(f'retracing baseline {loss_name} per-step full-train loss ...', flush=True)
        tr = retrace_baseline(loss_name, device)
        stored = runs[key][0]['train_loss'][-1]
        print(f'  retrace final={tr[-1]:.3e}  stored history final={stored:.3e}')
        traces[key] = (tr, None)

    # ---- summary numbers ----
    summary = {}
    for key in runs:
        hist, steps, pf, pfp, D, DP = runs[key]
        tr, lr = traces[key]
        late = steps >= 50_000
        mid_steps, m_logit = motion(steps, D)
        _, m_prob = motion(steps, DP)
        late_gap = mid_steps >= 50_000
        entry = {
            'final_train_loss': float(tr[-1]),
            'final_test_acc': hist['test_acc'][-1],
            'pf_logit_100k': float(pf[-1]), 'pf_prob_100k': float(pfp[-1]),
            'late_motion_logit_mean': float(m_logit[late_gap].mean()),
            'late_motion_prob_mean': float(m_prob[late_gap].mean()),
        }
        if lr is not None:
            red_steps = np.where(lr[1:] < lr[:-1])[0] + 1
            at_min = np.where(lr <= 1.01e-8)[0]
            entry.update({
                'final_lr': float(lr[-1]),
                'n_lr_reductions': int(len(red_steps)),
                'first_lr_reduction_step': int(red_steps[0]) if len(red_steps) else None,
                'lr_at_min_from_step': int(at_min[0]) if len(at_min) else None,
            })
        summary[key] = entry
    with open(os.path.join(HERE, 'results', 'lr_scheduler.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    # ---- figure ----
    st = np.arange(N_STEPS + 1)
    fig, axg = plt.subplots(2, 3, figsize=(16, 7.6))

    ax = axg[0, 0]
    ax.plot(st, traces['mse'][0], color='C0', lw=0.8, label='constant LR')
    ax.plot(st, traces['mse_sched'][0], color='C3', lw=0.8, label='ReduceLROnPlateau')
    ax.set_xscale('symlog', linthresh=10); ax.set_yscale('log')
    ax.set_title('MSE: full-train loss, every step')
    ax.set_ylabel('MSE train loss (log)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axg[0, 1]
    ax.plot(st, traces['ce'][0], color='C0', lw=0.8, label='constant LR')
    ax.plot(st, traces['ce_sched'][0], color='C3', lw=0.8, label='ReduceLROnPlateau')
    ax.set_xscale('symlog', linthresh=10); ax.set_yscale('log')
    ax.set_title('CE: full-train loss, every step')
    ax.set_ylabel('CE train loss (log)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axg[0, 2]
    ax.plot(st, traces['mse_sched'][1], color='C0', lw=1.4, label='MSE run')
    ax.plot(st, traces['ce_sched'][1], color='C3', lw=1.4, label='CE run')
    ax.set_xscale('symlog', linthresh=10); ax.set_yscale('log')
    ax.set_title('Scheduled runs: learning rate')
    ax.set_ylabel('learning rate (log)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axg[1, 0]
    for key, col, ls, lab in [('mse', 'C0', '-', 'MSE, const LR'),
                              ('mse_sched', 'C0', '--', 'MSE, scheduled'),
                              ('ce', 'C3', '-', 'CE, const LR'),
                              ('ce_sched', 'C3', '--', 'CE, scheduled')]:
        hist = runs[key][0]
        ax.plot(hist['step'], hist['test_acc'], color=col, ls=ls, lw=1.3, label=lab)
    ax.set_xscale('symlog', linthresh=10)
    ax.set_ylim(0, 1.0); ax.set_ylabel('test accuracy (first 2,000)')
    ax.set_xlabel('training step'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title('Test accuracy')

    ax = axg[1, 1]
    for key, col, ls, lab in [('mse', 'C0', '-', 'MSE const (logit d)'),
                              ('mse_sched', 'C0', '--', 'MSE sched (logit d)'),
                              ('ce', 'C3', '-', 'CE const (prob d)'),
                              ('ce_sched', 'C3', '--', 'CE sched (prob d)')]:
        _, steps, pf, pfp, _, _ = runs[key]
        ax.plot(steps, pf if key.startswith('mse') else pfp, color=col, ls=ls,
                lw=1.3, label=lab)
    ax.axhline(0.20, color='gray', ls=':', lw=1, label='diagonal floor ≈ 0.20')
    ax.set_xscale('symlog', linthresh=10)
    ax.set_ylim(0, 1.0); ax.set_ylabel('plateau fraction PF')
    ax.set_xlabel('training step'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title('Plateau fraction (each loss in its plateau space)')

    ax = axg[1, 2]
    for key, col, ls, lab in [('mse', 'C0', '-', 'MSE const (logit d)'),
                              ('mse_sched', 'C0', '--', 'MSE sched (logit d)'),
                              ('ce', 'C3', '-', 'CE const (prob d)'),
                              ('ce_sched', 'C3', '--', 'CE sched (prob d)')]:
        _, steps, _, _, D, DP = runs[key]
        ms, m = motion(steps, D if key.startswith('mse') else DP)
        ax.plot(ms, m, color=col, ls=ls, lw=1.1, label=lab)
    ax.set_xscale('symlog', linthresh=10); ax.set_yscale('log')
    ax.set_ylabel(r'curve motion $M(t)$ = mean $|\Delta d|$ per checkpoint gap')
    ax.set_xlabel('training step'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title('How much the curves still move')

    fig.suptitle('Constant LR vs ReduceLROnPlateau (seed 0, identical init/data/batches; '
                 'scheduler: factor 0.5, patience 10 steps on the full-train loss)', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, 'lr_scheduler_comparison.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)
    print('saved plots/lr_scheduler_comparison.png')


if __name__ == '__main__':
    main()
