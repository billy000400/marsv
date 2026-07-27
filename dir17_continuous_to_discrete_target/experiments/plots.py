"""Figures for dir17. Reads results/train_main.json + results/analysis.json."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common as C

A = json.load(open(os.path.join(C.RESULTS, 'analysis.json')))
T = json.load(open(os.path.join(C.RESULTS, 'train_main.json')))
GRID = np.array(A['grid'])
BMID = 0.5 * (GRID[:-1] + GRID[1:])
SEEDS = A['seeds']
KS = list(C.K_VALUES)
LAYERS = [1, 2, 3]


def get(which, k, seed):
    return A['models'][f"{which}|k{k:g}|s{seed}"]


def stack(which, k, field, layer=None):
    """[n_seeds, ...] array of a per-model quantity."""
    if layer is None:
        return np.array([get(which, k, s)[field] for s in SEEDS])
    return np.array([get(which, k, s)[f'L{layer}'][field] for s in SEEDS])


def ci95(a, axis=0):
    n = a.shape[axis]
    return 1.96 * a.std(axis=axis, ddof=1) / np.sqrt(n)


def R_stats(which, layer):
    """mean and 95% CI (across seeds) of R_l(k) = C_l(k)/0.2, per k."""
    m, e = [], []
    for k in KS:
        c = stack(which, k, 'C_mean', layer) / 0.2
        m.append(c.mean()); e.append(ci95(c))
    return np.array(m), np.array(e)


FLANK = C.flank_mask()


def F_stats(which, layer):
    """mean and 95% CI (across seeds) of the flank movement fraction F_l(k)."""
    m, e = [], []
    for k in KS:
        f = stack(which, k, 's_mean', layer)[:, FLANK].sum(axis=1)
        m.append(f.mean()); e.append(ci95(f))
    return np.array(m), np.array(e)


def F_target():
    return np.array([np.array(A['target'][f"{k:g}"]['s'])[FLANK].sum() for k in KS])


def F_pred():
    return np.array([np.mean([np.abs(np.diff(get('final', k, s)['pred_mean']))[FLANK].sum()
                              / np.abs(np.diff(get('final', k, s)['pred_mean'])).sum()
                              for s in SEEDS]) for k in KS])


# --------------------------------------------------------- Fig 2: training
def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for i, k in enumerate(KS):
        h = T[f"k{k:g}_s0"]['history']
        ep = np.array(h['epoch'])
        for ax, key in zip(axes, ['train_loss', 'val_loss']):
            ax.plot(ep, h[key], color=C.CVD[i], ls=C.LINESTYLES[i], lw=1.4,
                    label=f"$k={k:g}$")
        ad = T[f"k{k:g}_s0"]['adequacy']
        axes[1].plot([ad['val_min_epoch']], [ad['val_min']], marker=C.MARKERS[i],
                     color=C.CVD[i], ms=7, mew=1.2, mfc='none')
    axes[0].set_yscale('log'); axes[0].set_ylabel('training MSE (full train set)')
    axes[0].set_title('Training loss')
    axes[1].set_ylabel('validation MSE (2000 held-out images)')
    axes[1].set_title('Validation loss (open marker = minimum)')
    for ax in axes:
        ax.set_xlabel('epoch'); ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'training_curves.png'), dpi=150)
    plt.close()


# ------------------------------------------------------ Fig 3: predictions
def fig_predictions():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)
    for i, (k, ax) in enumerate(zip(KS, axes)):
        y = C.target_fn(GRID, k)
        pm = stack('final', k, 'pred_mean').mean(axis=0)
        ps = stack('final', k, 'pred_mean').std(axis=0)
        img_sd = stack('final', k, 'pred_std').mean(axis=0)
        ax.fill_between(GRID, pm - img_sd, pm + img_sd, color=C.CVD[i], alpha=0.22, lw=0)
        ax.plot(GRID, y, color='0.25', ls=':', lw=2, label='target $y_k(b)$')
        ax.plot(GRID, pm, color=C.CVD[i], ls='-', lw=1.8, label='mean prediction')
        r2 = np.mean([get('final', k, s)['sweep_r2'] for s in SEEDS])
        ax.set_title(f"$k={k:g}$   $R^2={r2:.3f}$", fontsize=10)
        ax.set_xlabel('brightness $b$'); ax.grid(alpha=0.3)
        ax.axvspan(C.CENTER_LO, C.CENTER_HI, color='0.85', alpha=0.7, zorder=0)
        if i == 0:
            ax.set_ylabel('target / prediction')
            ax.legend(frameon=False, fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'prediction_sweeps.png'), dpi=150)
    plt.close()


# --------------------------------------------------- Fig 4: movement curves
def fig_movement():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=False)
    # panel 0: the target's own movement, as the reference sharpness
    for i, k in enumerate(KS):
        axes[0].plot(BMID, A['target'][f"{k:g}"]['s'], color=C.CVD[i],
                     ls=C.LINESTYLES[i], lw=1.6, label=f"$k={k:g}$")
    axes[0].set_title('target $|\\Delta y_k|$ (normalized)', fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    for j, l in enumerate(LAYERS):
        ax = axes[j + 1]
        for i, k in enumerate(KS):
            s = stack('final', k, 's_mean', l)          # [seeds, 200]
            m = s.mean(axis=0)
            ax.plot(BMID, m, color=C.CVD[i], ls=C.LINESTYLES[i], lw=1.6,
                    label=f"$k={k:g}$")
            if len(SEEDS) > 1:
                e = ci95(s)
                ax.fill_between(BMID, m - e, m + e, color=C.CVD[i], alpha=0.2, lw=0)
        ax.set_title(f'hidden layer {l}' + (' (deepest)' if l == 3 else ''), fontsize=10)
    for ax in axes:
        ax.axhline(1 / (C.N_SWEEP - 1), color='0.35', lw=1, ls=(0, (1, 2)))
        ax.axvspan(C.CENTER_LO, C.CENTER_HI, color='0.85', alpha=0.7, zorder=0)
        ax.set_xlabel('brightness $b$'); ax.grid(alpha=0.3)
    axes[0].set_ylabel('normalized movement $s(b)$')
    axes[1].set_ylabel('normalized movement $s_l(b)$')
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'activation_movement_by_k.png'), dpi=150)
    plt.close()


# ------------------------------------------------ Fig 5: concentration vs k
def fig_concentration():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    tgt = np.array([A['target'][f"{k:g}"]['C'] for k in KS]) / 0.2
    ax.plot(KS, tgt, color='0.25', ls=':', lw=2, marker='*', ms=11,
            label='target $y_k$ (reference)')
    prd = np.array([np.mean([get('final', k, s)['pred_C'] for s in SEEDS]) for k in KS]) / 0.2
    ax.plot(KS, prd, color=C.CVD[4], ls=(0, (3, 1, 1, 1)), lw=1.6, marker=C.MARKERS[4],
            ms=6, label='model prediction $\\hat y$')
    for j, l in enumerate(LAYERS):
        m, e = R_stats('final', l)
        ax.errorbar(KS, m, yerr=e, color=C.CVD[j], ls=C.LINESTYLES[j], lw=1.8,
                    marker=C.MARKERS[j], ms=6, capsize=3, label=f'hidden layer {l}')
    ax.axhline(1.0, color='0.45', lw=1, ls=(0, (1, 2)))
    ax.set_ylim(bottom=0.94)
    ax.text(3.2, 0.955, 'uniform movement ($R=1$)', fontsize=8, color='0.35')
    ax.set_xscale('log'); ax.set_xticks(KS); ax.set_xticklabels([f"{k:g}" for k in KS])
    ax.set_xlabel('target sharpness $k$  (log scale)')
    ax.set_ylabel('concentration ratio $R = C/0.2$')
    ax.set_title('(a) movement in the middle 20% of brightness')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax2.plot(KS, F_target(), color='0.25', ls=':', lw=2, marker='*', ms=11,
             label='target $y_k$ (reference)')
    ax2.plot(KS, F_pred(), color=C.CVD[4], ls=(0, (3, 1, 1, 1)), lw=1.6,
             marker=C.MARKERS[4], ms=6, label='model prediction $\\hat y$')
    for j, l in enumerate(LAYERS):
        m, e = F_stats('final', l)
        ax2.errorbar(KS, m, yerr=e, color=C.CVD[j], ls=C.LINESTYLES[j], lw=1.8,
                     marker=C.MARKERS[j], ms=6, capsize=3, label=f'hidden layer {l}')
    ax2.axhline(0.4, color='0.45', lw=1, ls=(0, (1, 2)))
    ax2.text(0.52, 0.415, 'uniform movement ($F=0.4$)', fontsize=8, color='0.35')
    ax2.set_ylim(-0.02, 0.47)
    ax2.set_xscale('log'); ax2.set_xticks(KS); ax2.set_xticklabels([f"{k:g}" for k in KS])
    ax2.set_xlabel('target sharpness $k$  (log scale)')
    ax2.set_ylabel('flank movement fraction $F$')
    ax2.set_title('(b) movement left in the outer 40% ($F\\to0$ = true plateau)')
    ax2.legend(frameon=False, fontsize=9, loc='lower left')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'concentration_vs_k.png'), dpi=150)
    plt.close()


# --------------------------------------- Fig 6: checkpoint robustness check
def fig_robustness():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.3))
    for j, l in enumerate(LAYERS):
        for w, (which, ls, mk) in enumerate([('final', '-', 'o'), ('best_val', '--', 's')]):
            m, e = R_stats(which, l)
            ax.errorbar(KS, m, yerr=e, color=C.CVD[j], ls=ls, lw=1.6, marker=mk,
                        ms=5, capsize=3, mfc='none' if w else None,
                        label=f"layer {l} — {'min-val' if w else 'final'} ckpt")
    ax.axhline(1.0, color='0.45', lw=1, ls=(0, (1, 2)))
    ax.set_xscale('log'); ax.set_xticks(KS); ax.set_xticklabels([f"{k:g}" for k in KS])
    ax.set_xlabel('target sharpness $k$  (log scale)')
    ax.set_ylabel('concentration ratio $R_l(k)$')
    ax.set_title('(a) final vs minimum-validation-loss checkpoint')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # (b) primary (1000 training images) vs 10x larger training set
    N10 = json.load(open(os.path.join(C.RESULTS, 'summary_table_n10k.json')))
    tgt = np.array([A['target'][f"{k:g}"]['C'] for k in KS]) / 0.2
    ax2.plot(KS, tgt, color='0.25', ls=':', lw=2, marker='*', ms=11,
             label='target $y_k$ (reference)')
    m, e = R_stats('final', 3)
    ax2.errorbar(KS, m, yerr=e, color=C.CVD[2], ls='-', lw=1.8, marker='^', ms=6,
                 capsize=3, label='layer 3 — 1000 train images (primary)')
    m10 = np.array([r['R3'] for r in N10]); e10 = np.array([r['R3_ci'] for r in N10])
    ax2.errorbar(KS, m10, yerr=e10, color=C.CVD[2], ls='--', lw=1.8, marker='^', ms=7,
                 mfc='none', capsize=3, label='layer 3 — 10000 train images')
    ax2.plot(KS, [np.mean([get('final', k, s)['pred_C'] for s in SEEDS]) / 0.2 for k in KS],
             color=C.CVD[4], ls='-', lw=1.4, marker='v', ms=5,
             label='prediction — 1000 train images')
    ax2.plot(KS, [r['predR'] for r in N10], color=C.CVD[4], ls='--', lw=1.4, marker='v',
             ms=6, mfc='none', label='prediction — 10000 train images')
    ax2.axhline(1.0, color='0.45', lw=1, ls=(0, (1, 2)))
    ax2.set_xscale('log'); ax2.set_xticks(KS); ax2.set_xticklabels([f"{k:g}" for k in KS])
    ax2.set_xlabel('target sharpness $k$  (log scale)')
    ax2.set_ylabel('concentration ratio $R$')
    ax2.set_title('(b) training-set size (deepest layer + output)')
    ax2.legend(frameon=False, fontsize=8, loc='upper left')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'checkpoint_robustness.png'), dpi=150)
    plt.close()


# ----------------------------------------------------- Fig 7: main summary
def fig_summary():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for i, k in enumerate(KS):
        axes[0].plot(GRID, C.target_fn(GRID, k), color=C.CVD[i], ls=C.LINESTYLES[i],
                     lw=1.8, label=f"$k={k:g}$")
    axes[0].set_ylabel('target $y_k(b)$'); axes[0].set_title('(a) target family', fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')

    for i, k in enumerate(KS):
        pm = stack('final', k, 'pred_mean').mean(axis=0)
        axes[1].plot(GRID, C.target_fn(GRID, k), color='0.6', ls=':', lw=1.2)
        axes[1].plot(GRID, pm, color=C.CVD[i], ls=C.LINESTYLES[i], lw=1.8,
                     label=f"$k={k:g}$")
    axes[1].set_ylabel('mean prediction $\\hat y(b)$')
    axes[1].set_title('(b) predictions (dotted grey = target)', fontsize=10)

    for i, k in enumerate(KS):
        s = stack('final', k, 's_mean', 3)
        m = s.mean(axis=0); e = ci95(s)
        axes[2].plot(BMID, m, color=C.CVD[i], ls=C.LINESTYLES[i], lw=1.6, label=f"$k={k:g}$")
        axes[2].fill_between(BMID, m - e, m + e, color=C.CVD[i], alpha=0.2, lw=0)
    axes[2].axhline(1 / (C.N_SWEEP - 1), color='0.35', lw=1, ls=(0, (1, 2)))
    axes[2].set_ylabel('$s_3(b)$ (deepest layer)')
    axes[2].set_title('(c) normalized activation movement', fontsize=10)

    tgt = np.array([A['target'][f"{k:g}"]['C'] for k in KS]) / 0.2
    axes[3].plot(KS, tgt, color='0.25', ls=':', lw=2, marker='*', ms=11, label='target')
    for j, l in enumerate(LAYERS):
        m, e = R_stats('final', l)
        axes[3].errorbar(KS, m, yerr=e, color=C.CVD[j], ls=C.LINESTYLES[j], lw=1.8,
                         marker=C.MARKERS[j], ms=6, capsize=3, label=f'layer {l}')
    axes[3].axhline(1.0, color='0.45', lw=1, ls=(0, (1, 2)))
    axes[3].set_xscale('log'); axes[3].set_xticks(KS)
    axes[3].set_xticklabels([f"{k:g}" for k in KS])
    axes[3].set_xlabel('target sharpness $k$ (log)')
    axes[3].set_ylabel('$R_l(k)=C_l(k)/0.2$')
    axes[3].set_title('(d) concentration score', fontsize=10)
    axes[3].legend(frameon=False, fontsize=8)

    for ax in axes[:3]:
        ax.set_xlabel('brightness $b$')
        ax.axvspan(C.CENTER_LO, C.CENTER_HI, color='0.85', alpha=0.7, zorder=0)
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'main_summary.png'), dpi=150)
    plt.close()


def table():
    rows = []
    for k in KS:
        r = {'k': k, 'target_R': A['target'][f"{k:g}"]['C'] / 0.2,
             'pred_R': float(np.mean([get('final', k, s)['pred_C'] for s in SEEDS]) / 0.2),
             'sweep_r2': float(np.mean([get('final', k, s)['sweep_r2'] for s in SEEDS])),
             'val_mse': float(np.mean([T[f"k{k:g}_s{s}"]['adequacy']['val_final'] for s in SEEDS])),
             'val_ratio': float(np.mean([T[f"k{k:g}_s{s}"]['adequacy']['val_final_over_min']
                                         for s in SEEDS]))}
        r['target_F'] = float(F_target()[KS.index(k)])
        r['pred_F'] = float(F_pred()[KS.index(k)])
        for l in LAYERS:
            f = stack('final', k, 's_mean', l)[:, FLANK].sum(axis=1)
            r[f'F{l}'] = float(f.mean()); r[f'F{l}_ci'] = float(ci95(f))
            c = stack('final', k, 'C_mean', l) / 0.2
            r[f'R{l}'] = float(c.mean()); r[f'R{l}_ci'] = float(ci95(c))
            r[f'R{l}_img_sd'] = float(stack('final', k, 'C_std_images', l).mean() / 0.2)
            cb = stack('best_val', k, 'C_mean', l) / 0.2
            r[f'R{l}_bestval'] = float(cb.mean())
        rows.append(r)
    with open(os.path.join(C.RESULTS, 'summary_table.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    for r in rows:
        print(f"k={r['k']:>4g}  tgtR={r['target_R']:5.2f}  predR={r['pred_R']:5.2f}  "
              f"R1={r['R1']:.2f}±{r['R1_ci']:.2f}  R2={r['R2']:.2f}±{r['R2_ci']:.2f}  "
              f"R3={r['R3']:.2f}±{r['R3_ci']:.2f}  sweepR2={r['sweep_r2']:.3f}")
    return rows


if __name__ == '__main__':
    os.makedirs(C.PLOTS, exist_ok=True)
    fig_training(); fig_predictions(); fig_movement()
    fig_concentration(); fig_robustness(); fig_summary()
    table()
    print('figures written to plots/')
