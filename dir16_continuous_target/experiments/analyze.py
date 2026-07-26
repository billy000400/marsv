#!/usr/bin/env python3
"""
dir16 S3 — aggregate the probe, bootstrap the paired comparison, make figures.

Writes results/aggregate.json and every PNG in plots/.
Usage: python experiments/analyze.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (C_CLF, C_REG, CVD, MLP, N_POINTS, PLOTS, RESULTS,
                    build_dataset, build_pairs, setup, slerp_batch)

plt.rcParams.update({'axes.prop_cycle': plt.cycler(color=CVD),
                     'figure.dpi': 130, 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': .25})
SEEDS = [0, 1, 2]
LAYERS = [('d_h2', 'hidden layer 2'), ('d_h3', 'hidden layer 3'),
          ('d_out', 'output layer')]
KINDS = [('clf', 'classifier', C_CLF, '-', 'o'),
         ('reg', 'regressor', C_REG, '--', 's')]
ALPHA = np.linspace(0, 1, N_POINTS)
HAND = [(6, 7), (3, 5), (0, 1), (4, 9)]     # hand-selected transitions
NBOOT = 10_000


def load():
    P = {}
    for s in SEEDS:
        for k, _, _, _, _ in KINDS:
            P[(s, k)] = dict(np.load(os.path.join(RESULTS, f'seed{s}',
                                                  f'probe_{k}.npz')))
    return P


def lin_dev(d):
    """mean_alpha |d(alpha) - alpha| per pair. [P,T] -> [P]."""
    return np.abs(d - ALPHA[None, :]).mean(1)


def max_jump(d):
    """(T-1) * max_alpha |d(alpha_{i+1}) - d(alpha_i)| per pair. [P,T] -> [P]."""
    return (N_POINTS - 1) * np.abs(np.diff(d, axis=1)).max(1)


def boot_ci(x, rng, n=NBOOT):
    """Percentile bootstrap 95% CI of the mean, resampling pairs."""
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------- figures
def fig_training():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for k, lbl, c, ls, _ in KINDS:
        ax = axes[0] if k == 'clf' else axes[1]
        for i, s in enumerate(SEEDS):
            h = json.load(open(os.path.join(RESULTS, f'seed{s}', f'{k}_history.json')))
            st = np.array(h['step'])
            ax.plot(st[1:], h['train_loss'][1:], color=CVD[0], ls='-', lw=1,
                    alpha=.8, label='train loss' if i == 0 else None)
            ax.plot(st[1:], h['val_loss'][1:], color=CVD[1], ls='--', lw=1,
                    alpha=.8, label='val loss' if i == 0 else None)
            if k == 'clf':
                vl = np.array(h['val_loss'])
                j = int(np.argmin(vl[1:])) + 1
                axes[2].plot(st[1:], vl[1:] / vl[j], color=CVD[1], ls='--', lw=1,
                             alpha=.85, label='classifier val loss' if i == 0 else None)
                axes[2].scatter([st[j]], [1.0], color=CVD[0], marker='v', s=30,
                                zorder=4, label='minimum' if i == 0 else None)
        ax.set_yscale('log'); ax.set_xlabel('training step')
        ax.set_ylabel('MSE per output unit')
        ax.set_title(f'{lbl}  (3 seeds)'); ax.legend(fontsize=7)
    axes[2].set_xlabel('training step'); axes[2].set_ylim(0.98, 1.10)
    axes[2].set_ylabel('val loss / its minimum')
    axes[2].set_title('classifier val loss, rescaled (3 seeds)')
    axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'training_curves.png'))
    plt.close(fig)


def fig_control(P):
    """Early-stopped classifier control: does the gap survive matched fitting?"""
    info = json.load(open(os.path.join(RESULTS, 'control_earlystop.json')))
    Pe = {s: dict(np.load(os.path.join(RESULTS, f'seed{s}', 'probe_clf_early.npz')))
          for s in SEEDS}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    series = [('classifier (final, step 30k)', 'clf', C_CLF, '-', 'o'),
              ('classifier (early-stopped)', None, CVD[2], '-.', '^'),
              ('regressor (final, step 30k)', 'reg', C_REG, '--', 's')]
    out = {}
    for c, (key, lname) in enumerate(LAYERS):
        ax = axes[c]
        vals = []
        for lbl, k, col, ls, mk in series:
            cur = (np.stack([P[(s, k)][key] for s in SEEDS]) if k
                   else np.stack([Pe[s][key] for s in SEEDS]))
            ax.plot(ALPHA, cur.mean((0, 1)), color=col, ls=ls, lw=1.8, label=lbl,
                    marker=mk, markevery=13, ms=4)
            vals.append(float(np.mean([lin_dev(x) for x in cur])))
        ax.plot([0, 1], [0, 1], color='0.6', lw=.8, ls=':', label='straight line')
        ax.set_title(lname); ax.set_xlabel(r'interpolation position $\alpha$')
        out[key] = dict(zip(['clf_final', 'clf_early', 'reg_final'], vals))
    axes[0].set_ylabel(r'mean $d(\alpha)$ over 90 pairs, 3 seeds')
    axes[0].legend(fontsize=6.5, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'control_earlystop.png'))
    plt.close(fig)
    # paired CI: early-stopped classifier minus regressor
    rng = np.random.default_rng(1)
    for key, _ in LAYERS:
        c = np.mean([lin_dev(Pe[s][key]) for s in SEEDS], 0)
        r = np.mean([lin_dev(P[(s, 'reg')][key]) for s in SEEDS], 0)
        lo, hi = boot_ci(c - r, rng)
        out[key].update({'diff_early_minus_reg': float((c - r).mean()),
                         'ci': [lo, hi],
                         'frac_pairs_reg_lower': float((r < c).mean())})
    out['stop_info'] = info
    return out


def fig_recon(device):
    """Regressor task quality: corrupted input, its 7x7 pooling, prediction, target."""
    ds = build_dataset()
    m = MLP(49).to(device)
    m.load_state_dict(torch.load(os.path.join(RESULTS, 'seed0', 'reg.pt'),
                                 map_location=device))
    m.eval()
    idx = [int(torch.where(ds['te_lab'] == c)[0][0]) for c in range(10)]
    x = ds['te_in'][idx].to(device)
    with torch.no_grad():
        pred = m(x).cpu().numpy().reshape(-1, 7, 7)
    fig, axes = plt.subplots(3, 10, figsize=(10.5, 3.4))
    for j in range(10):
        for r, (img, t) in enumerate([
                (ds['te_in'][idx[j]].reshape(28, 28), 'corrupted input'),
                (pred[j], 'regressor output'),
                (ds['te_rec'][idx[j]].reshape(7, 7), 'clean 7x7 target')]):
            axes[r, j].imshow(np.asarray(img), cmap='gray', vmin=0, vmax=1)
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([]); axes[r, j].grid(False)
            if j == 0:
                axes[r, j].set_ylabel(t, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'reconstructions.png'))
    plt.close(fig)


def fig_hand(P):
    """d(alpha) on the hand-selected transitions, classifier vs regressor."""
    meta = P[(0, 'clf')]
    fig, axes = plt.subplots(len(HAND), 3, figsize=(9.5, 2.4 * len(HAND)),
                             sharex=True, sharey=True)
    for r, (a, b) in enumerate(HAND):
        pi = int(np.where((meta['class_a'] == a) & (meta['class_b'] == b)
                          & (meta['rep'] == 0))[0][0])
        for c, (key, lname) in enumerate(LAYERS):
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], color='0.6', lw=.8, ls=':', label='straight line')
            for k, lbl, col, ls, mk in KINDS:
                for i, s in enumerate(SEEDS):
                    ax.plot(ALPHA, P[(s, k)][key][pi], color=col, ls=ls, lw=1.2,
                            alpha=.85, label=lbl if (i == 0) else None)
            if r == 0:
                ax.set_title(lname)
            if c == 0:
                ax.set_ylabel(f'{a} $\\rightarrow$ {b}\n$d(\\alpha)$')
            if r == len(HAND) - 1:
                ax.set_xlabel(r'interpolation position $\alpha$')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'hand_selected_curves.png'))
    plt.close(fig)


def fig_mean_curves(P):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for c, (key, lname) in enumerate(LAYERS):
        ax = axes[c]
        ax.plot([0, 1], [0, 1], color='0.6', lw=.8, ls=':', label='straight line')
        for k, lbl, col, ls, mk in KINDS:
            cur = np.stack([P[(s, k)][key] for s in SEEDS])      # [S,P,T]
            m = cur.mean((0, 1))
            lo, hi = np.percentile(cur.reshape(-1, N_POINTS), [25, 75], axis=0)
            ax.plot(ALPHA, m, color=col, ls=ls, lw=1.8, label=lbl,
                    marker=mk, markevery=12, ms=4)
            ax.fill_between(ALPHA, lo, hi, color=col, alpha=.15,
                            hatch='//' if k == 'clf' else '\\\\', lw=0)
        ax.set_title(lname); ax.set_xlabel(r'interpolation position $\alpha$')
    axes[0].set_ylabel(r'$d(\alpha)$'); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'mean_curves.png'))
    plt.close(fig)


def fig_paired(agg):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for pane, (met, name) in enumerate([('lin_dev', 'linearity deviation'),
                                        ('max_jump', 'max normalized jump')]):
        ax = axes[pane]
        xs = np.arange(len(LAYERS))
        for si, s in enumerate(SEEDS):
            v = [agg['per_seed'][str(s)][met][key]['diff_mean'] for key, _ in LAYERS]
            ax.scatter(xs + .12 * (si - 1), v, color=CVD[3], marker='^', s=22,
                       zorder=3, label='per-seed mean' if si == 0 else None)
        m = [agg['pooled'][met][key]['diff_mean'] for key, _ in LAYERS]
        lo = [agg['pooled'][met][key]['ci'][0] for key, _ in LAYERS]
        hi = [agg['pooled'][met][key]['ci'][1] for key, _ in LAYERS]
        ax.errorbar(xs, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt='o',
                    color=CVD[0], capsize=4, lw=1.6, ms=6, zorder=4,
                    label='seed-averaged mean, 95% bootstrap CI')
        ax.axhline(0, color='0.4', lw=.9, ls='--')
        ax.set_xticks(xs); ax.set_xticklabels([n for _, n in LAYERS])
        ax.set_ylabel(f'classifier $-$ regressor\n({name})')
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'paired_difference.png'))
    plt.close(fig)


def fig_scatter(P):
    """Per-pair classifier vs regressor deviation at h3 — is the effect uniform?"""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for c, (key, lname) in enumerate(LAYERS):
        ax = axes[c]
        x = np.mean([lin_dev(P[(s, 'clf')][key]) for s in SEEDS], 0)
        y = np.mean([lin_dev(P[(s, 'reg')][key]) for s in SEEDS], 0)
        ax.scatter(x, y, s=16, color=CVD[0], marker='o', alpha=.75)
        hi = max(x.max(), y.max()) * 1.05
        ax.plot([0, hi], [0, hi], color='0.4', ls='--', lw=1, label='equal (y = x)')
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel('classifier deviation'); ax.set_title(lname)
        frac = float((y < x).mean())
        ax.text(.05, .93, f'regressor lower on {frac*100:.0f}% of pairs',
                transform=ax.transAxes, fontsize=7.5)
        if c == 0:
            ax.set_ylabel('regressor deviation'); ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'per_pair_scatter.png'))
    plt.close(fig)


def fig_path_recon(device):
    """Regressor output rendered along the 6->7 interpolation path."""
    ds = build_dataset()
    pairs = build_pairs(ds['te_lab'])
    pi = [i for i, p in enumerate(pairs)
          if (p['class_a'], p['class_b'], p['rep']) == (6, 7, 0)][0]
    xa = ds['te_in'][[pairs[pi]['idx_a']]].to(device)
    xb = ds['te_in'][[pairs[pi]['idx_b']]].to(device)
    show = np.linspace(0, N_POINTS - 1, 11).astype(int)
    fig, axes = plt.subplots(2, 11, figsize=(11, 2.5))
    for row, (kind, n_out) in enumerate([('reg', 49), ('clf', 10)]):
        m = MLP(n_out).to(device)
        m.load_state_dict(torch.load(os.path.join(RESULTS, 'seed0', f'{kind}.pt'),
                                     map_location=device))
        m.eval()
        with torch.no_grad():
            ha, _ = m.hidden_activations(xa)
            hb, _ = m.hidden_activations(xb)
            h1 = slerp_batch(ha[0], hb[0], N_POINTS)
            out, _ = m.forward_from(h1.reshape(-1, 200), 1)
            out = out.reshape(N_POINTS, -1).cpu().numpy()
        for j, t in enumerate(show):
            ax = axes[row, j]
            ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
            if kind == 'reg':
                ax.imshow(out[t].reshape(7, 7), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'{ALPHA[t]:.1f}', fontsize=7)
            else:
                ax.bar(range(10), out[t], color=CVD[0])
                ax.set_ylim(-.4, 1.2)
                ax.set_xlabel(str(int(out[t].argmax())), fontsize=7, labelpad=1)
        axes[row, 0].set_ylabel('regressor\noutput 7x7' if kind == 'reg'
                                else 'classifier\n10 logits', fontsize=7)
    fig.suptitle(r'6 $\rightarrow$ 7 path: model output vs $\alpha$ (top labels)',
                 fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'path_reconstructions.png'))
    plt.close(fig)


def main():
    device = setup()
    os.makedirs(PLOTS, exist_ok=True)
    P = load()
    rng = np.random.default_rng(0)

    agg = {'per_seed': {}, 'pooled': {}}
    for met, fn in [('lin_dev', lin_dev), ('max_jump', max_jump)]:
        agg['pooled'][met] = {}
        for key, _ in LAYERS:
            c = np.mean([fn(P[(s, 'clf')][key]) for s in SEEDS], 0)
            r = np.mean([fn(P[(s, 'reg')][key]) for s in SEEDS], 0)
            lo, hi = boot_ci(c - r, rng)
            agg['pooled'][met][key] = {
                'clf_mean': float(c.mean()), 'reg_mean': float(r.mean()),
                'diff_mean': float((c - r).mean()), 'ci': [lo, hi],
                'frac_pairs_reg_lower': float((r < c).mean()),
                'significant': bool(lo > 0 or hi < 0)}
    for s in SEEDS:
        agg['per_seed'][str(s)] = {}
        for met, fn in [('lin_dev', lin_dev), ('max_jump', max_jump)]:
            agg['per_seed'][str(s)][met] = {}
            for key, _ in LAYERS:
                c, r = fn(P[(s, 'clf')][key]), fn(P[(s, 'reg')][key])
                lo, hi = boot_ci(c - r, rng)
                agg['per_seed'][str(s)][met][key] = {
                    'clf_mean': float(c.mean()), 'reg_mean': float(r.mean()),
                    'diff_mean': float((c - r).mean()), 'ci': [lo, hi]}
    # dir12 fraction-form robustness check on linearity deviation
    agg['frac_form'] = {}
    for key in ['f_h2', 'f_h3', 'f_out']:
        c = np.mean([lin_dev(P[(s, 'clf')][key]) for s in SEEDS], 0)
        r = np.mean([lin_dev(P[(s, 'reg')][key]) for s in SEEDS], 0)
        lo, hi = boot_ci(c - r, rng)
        agg['frac_form'][key] = {'clf_mean': float(c.mean()),
                                 'reg_mean': float(r.mean()),
                                 'diff_mean': float((c - r).mean()), 'ci': [lo, hi]}
    agg['n_pairs'] = int(P[(0, 'clf')]['d_h3'].shape[0])
    agg['seeds'] = SEEDS

    fig_training(); fig_recon(device); fig_hand(P); fig_mean_curves(P)
    fig_paired(agg); fig_scatter(P); fig_path_recon(device)
    agg['control_earlystop'] = fig_control(P)
    json.dump(agg, open(os.path.join(RESULTS, 'aggregate.json'), 'w'), indent=1)
    print(json.dumps(agg['pooled'], indent=1))
    print(json.dumps(agg['control_earlystop'], indent=1))
    print('figures written to', PLOTS)


if __name__ == '__main__':
    main()
