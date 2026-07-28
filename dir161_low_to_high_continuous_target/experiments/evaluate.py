#!/usr/bin/env python3
"""
dir161 S2b — task-adequacy gates and the frozen evaluation-only baselines.

Everything here is measured on the untouched endpoint pool test[:2000], at each
model's best-validation-loss checkpoint.  Baselines (frozen before training):
mean training image, block repetition U(z), 7x7->28x28 bicubic, and the
privileged digit-template diagnostic U(z) + mean_train[P(y) | digit].

Writes results/task_quality.json and plots/superres_panel.png,
plots/baseline_bars.png.

Usage: python experiments/evaluate.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CVD, D, MLP, N_TEST_POOL, PLOTS, RESULTS, U, Pdet, bicubic,
                    build_dataset, setup)

plt.rcParams.update({'axes.prop_cycle': plt.cycler(color=CVD), 'figure.dpi': 130,
                     'font.size': 9, 'axes.grid': True, 'grid.alpha': .25})
SEEDS = [0, 1, 2]
NBOOT = 10_000


def per_image(yhat, y, z):
    """Per-image full MSE, removed-detail MSE, low-res consistency MSE."""
    return {'mse': ((yhat - y) ** 2).mean(1).cpu().numpy(),
            'detail_mse': ((Pdet(yhat) - Pdet(y)) ** 2).mean(1).cpu().numpy(),
            'lowres_mse': ((D(yhat) - z) ** 2).mean(1).cpu().numpy(),
            'sse_det': ((Pdet(yhat) - Pdet(y)) ** 2).sum(1).cpu().numpy()}


def boot_ci(x, rng, n=NBOOT):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_r2(sse, denom, rng, n=NBOOT):
    """Bootstrap CI of R2_detail = 1 - sum(sse)/sum(denom), resampling images."""
    idx = rng.integers(0, len(sse), size=(n, len(sse)))
    r2 = 1 - sse[idx].sum(1) / denom[idx].sum(1)
    return [float(np.percentile(r2, 2.5)), float(np.percentile(r2, 97.5))]


def main():
    device = setup()
    rng = np.random.default_rng(0)
    ds = build_dataset()
    ep = slice(0, N_TEST_POOL)
    z = ds['te_in'][ep].to(device)
    y = ds['te_img'][ep].to(device)
    lab = ds['te_lab'][ep]
    denom = (Pdet(y) ** 2).sum(1).cpu().numpy()

    # --- frozen baselines ----------------------------------------------------
    mean_img = ds['tr_img'].mean(0, keepdim=True).to(device).expand_as(y)
    blk = U(z)
    bic = bicubic(z)
    tr_det = Pdet(ds['tr_img'].to(device))
    tmpl = torch.stack([tr_det[ds['tr_lab'].to(device) == c].mean(0) for c in range(10)])
    dtmpl = blk + tmpl[lab.to(device)]

    preds = {'mean image': mean_img, 'block repeat $U(z)$': blk,
             'bicubic': bic, 'digit template': dtmpl}

    # --- model predictions, per seed ----------------------------------------
    model_pred = {}
    for s in SEEDS:
        m = MLP(784).to(device)
        m.load_state_dict(torch.load(os.path.join(RESULTS, f'seed{s}', 'pre_best.pt'),
                                     map_location=device))
        m.eval()
        with torch.no_grad():
            model_pred[s] = m(z)
    preds['predictor'] = model_pred[0]

    stats = {name: per_image(p, y, z) for name, p in preds.items()}
    # seed-averaged per-image model error (primary row)
    mstat = {k: np.mean([per_image(model_pred[s], y, z)[k] for s in SEEDS], 0)
             for k in ['mse', 'detail_mse', 'lowres_mse', 'sse_det']}
    stats['predictor'] = mstat

    out = {'gate_pass': {}, 'baselines': {}, 'per_seed': {}}
    for name, st in stats.items():
        out['baselines'][name] = {
            'mse': float(st['mse'].mean()), 'mse_ci': boot_ci(st['mse'], rng),
            'detail_mse': float(st['detail_mse'].mean()),
            'detail_mse_ci': boot_ci(st['detail_mse'], rng),
            'lowres_mse': float(st['lowres_mse'].mean()),
            'r2_detail': float(1 - st['sse_det'].sum() / denom.sum()),
            'r2_detail_ci': boot_r2(st['sse_det'], denom, rng)}
    # paired model-vs-baseline differences (positive = model better)
    for name in ['mean image', 'block repeat $U(z)$', 'bicubic', 'digit template']:
        for met in ['mse', 'detail_mse']:
            dv = stats[name][met] - mstat[met]
            out['baselines'][name][f'minus_model_{met}'] = float(dv.mean())
            out['baselines'][name][f'minus_model_{met}_ci'] = boot_ci(dv, rng)

    for s in SEEDS:
        st = per_image(model_pred[s], y, z)
        summ = json.load(open(os.path.join(RESULTS, f'seed{s}', 'summary.json')))
        out['per_seed'][str(s)] = {
            'clf_acc_pool': summ['clf']['best_pool_acc'],
            'clf_val_min_step': summ['clf']['val_min_step'],
            'pre_val_min_step': summ['pre']['val_min_step'],
            'mse': float(st['mse'].mean()), 'detail_mse': float(st['detail_mse'].mean()),
            'lowres_mse': float(st['lowres_mse'].mean()),
            'r2_detail': float(1 - st['sse_det'].sum() / denom.sum())}

    accs = [out['per_seed'][str(s)]['clf_acc_pool'] for s in SEEDS]
    b = out['baselines']
    beats = all(b[n][f'minus_model_{m}_ci'][0] > 0
                for n in ['block repeat $U(z)$', 'bicubic'] for m in ['mse', 'detail_mse'])
    out['gate_pass'] = {
        'classifier_acc_min': min(accs), 'classifier_gate': bool(min(accs) >= 0.95),
        'predictor_beats_fixed_upsamplers': bool(beats),
        'r2_detail_lower_bound': b['predictor']['r2_detail_ci'][0],
        'predictor_gate': bool(beats and b['predictor']['r2_detail_ci'][0] > 0),
        'beats_digit_template_detail':
            bool(b['digit template']['minus_model_detail_mse_ci'][0] > 0)}

    # --- figure: super-resolution panel -------------------------------------
    idx = [int(torch.where(lab == c)[0][0]) for c in range(10)]
    rows = [('input $z$ shown as $U(z)$', blk, 0, 1, 'gray'),
            ('bicubic', bic, 0, 1, 'gray'),
            ('digit template', dtmpl, 0, 1, 'gray'),
            ('predictor $\\hat{y}$', model_pred[0], 0, 1, 'gray'),
            ('target $y$', y, 0, 1, 'gray'),
            ('predicted detail $P(\\hat{y})$', Pdet(model_pred[0]), -.5, .5, 'coolwarm'),
            ('true detail $P(y)$', Pdet(y), -.5, .5, 'coolwarm')]
    fig, axes = plt.subplots(len(rows), 10, figsize=(10.5, 1.15 * len(rows)))
    for ri, (name, t, vmin, vmax, cm) in enumerate(rows):
        im = t[idx].reshape(-1, 28, 28).detach().cpu().numpy()
        for cj in range(10):
            ax = axes[ri, cj]
            ax.imshow(im[cj], cmap=cm, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if cj == 0:
                ax.set_ylabel(name, fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'superres_panel.png'))
    plt.close(fig)

    # --- figure: baselines with uncertainty ---------------------------------
    names = ['mean image', 'block repeat $U(z)$', 'bicubic', 'digit template', 'predictor']
    short = ['mean\nimage', 'block\nrepeat', 'bicubic', 'digit\ntemplate', 'predictor']
    hatches = ['//', '\\\\', '..', 'xx', '']
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for pane, (key, ttl, ylab) in enumerate([
            ('mse', 'full-image MSE (lower better)', 'MSE per pixel'),
            ('detail_mse', 'removed-detail MSE (lower better)', 'MSE per pixel'),
            ('r2_detail', 'detail recovery $R^2_{\\mathrm{detail}}$ (higher better)',
             '$R^2_{\\mathrm{detail}}$')]):
        ax = axes[pane]
        v = [b[n][key] for n in names]
        ci = np.array([b[n][key + ('_ci' if key != 'r2_detail' else '_ci')] for n in names])
        err = np.abs(ci.T - np.array(v))
        for i in range(len(names)):
            ax.bar(i, v[i], color=CVD[i % 5], hatch=hatches[i], edgecolor='0.25',
                   lw=.6, yerr=err[:, i:i + 1], capsize=3, ecolor='0.2')
        ax.set_xticks(range(len(names))); ax.set_xticklabels(short, fontsize=7)
        ax.set_title(ttl, fontsize=8); ax.set_ylabel(ylab, fontsize=8)
        if key == 'r2_detail':
            ax.axhline(0, color='0.3', lw=.9, ls='--')
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'baseline_bars.png'))
    plt.close(fig)

    json.dump(out, open(os.path.join(RESULTS, 'task_quality.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
