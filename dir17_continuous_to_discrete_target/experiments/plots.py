"""Figures + summary tables for dir17.

Reads results/train_main.json + results/analysis.json (primary, 1000 training
images) and results/train_n10k.json + results/analysis_n10k.json (10,000-image
robustness grid).

Notation (renamed after operator feedback #1, which flagged that the old
"R_l" collides visually with the coefficient of determination R^2):
    Gamma_l(k) = C_l(k) / 0.2  -- concentration gain  (1 = uniform, 5 = max)
    Phi_l(k)                   -- flank share         (0.4 = uniform, 0 = plateau)
    R^2                        -- kept ONLY for the sweep goodness-of-fit.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common as C

A = json.load(open(os.path.join(C.RESULTS, 'analysis.json')))
T = json.load(open(os.path.join(C.RESULTS, 'train_main.json')))
A10 = json.load(open(os.path.join(C.RESULTS, 'analysis_n10k.json')))
T10 = json.load(open(os.path.join(C.RESULTS, 'train_n10k.json')))
GRID = np.array(A['grid'])
BMID = 0.5 * (GRID[:-1] + GRID[1:])
SEEDS = A['seeds']
KS = list(C.K_VALUES)
LAYERS = [1, 2, 3]
FLANK = C.flank_mask()
GAM = r'$\Gamma$'
PHI = r'$\Phi$'


# ------------------------------------------------------------ accessors
def get(which, k, seed, an=None):
    return (an or A)['models'][f"{which}|k{k:g}|s{seed}"]


def stack(which, k, field, layer=None, an=None):
    """[n_seeds, ...] array of a per-model quantity."""
    if layer is None:
        return np.array([get(which, k, s, an)[field] for s in SEEDS])
    return np.array([get(which, k, s, an)[f'L{layer}'][field] for s in SEEDS])


def ci95(a, axis=0):
    n = a.shape[axis]
    return 1.96 * a.std(axis=axis, ddof=1) / np.sqrt(n)


def gamma_stats(which, layer, an=None):
    """mean and 95% CI (across seeds) of Gamma_l(k) = C_l(k)/0.2, per k."""
    m, e = [], []
    for k in KS:
        c = stack(which, k, 'C_mean', layer, an) / 0.2
        m.append(c.mean()); e.append(ci95(c))
    return np.array(m), np.array(e)


def phi_stats(which, layer, an=None):
    """mean and 95% CI (across seeds) of the flank share Phi_l(k)."""
    m, e = [], []
    for k in KS:
        f = stack(which, k, 's_mean', layer, an)[:, FLANK].sum(axis=1)
        m.append(f.mean()); e.append(ci95(f))
    return np.array(m), np.array(e)


def gamma_target():
    return np.array([A['target'][f"{k:g}"]['C'] for k in KS]) / 0.2


def phi_target():
    return np.array([np.array(A['target'][f"{k:g}"]['s'])[FLANK].sum() for k in KS])


def gamma_pred(an=None):
    return np.array([np.mean([get('final', k, s, an)['pred_C'] for s in SEEDS])
                     for k in KS]) / 0.2


def phi_pred(an=None):
    out = []
    for k in KS:
        vals = []
        for s in SEEDS:
            d = np.abs(np.diff(get('final', k, s, an)['pred_mean']))
            vals.append(d[FLANK].sum() / d.sum())
        out.append(np.mean(vals))
    return np.array(out)


def klabel(ax, ticks=(0.5, 2, 10, 40, 160)):
    ax.set_xscale('log')
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()


def k_legend(ax, **kw):
    """Compact legend for the 10 ordered k values (two columns)."""
    ax.legend(frameon=False, fontsize=7, ncol=2, handlelength=2.4,
              columnspacing=1.0, labelspacing=0.25, **kw)


# ------------------------------------------------- Fig 1: target functions
def fig_targets():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for i, k in enumerate(KS):
        st = C.k_style(i)
        for ax in axes:
            ax.plot(GRID, C.target_fn(GRID, k), lw=1.7, label=f"$k={k:g}$", **st)
    for ax in axes:
        ax.axvspan(C.CENTER_LO, C.CENTER_HI, color='0.85', alpha=0.7, zorder=0)
        ax.axvline(C.B0, color='0.35', lw=1, ls=(0, (1, 2)))
        ax.set_xlabel('brightness $b$'); ax.grid(alpha=0.3)
    axes[0].set_ylabel('target $y_k(b)$')
    axes[0].set_title('(a) full brightness range', fontsize=10)
    k_legend(axes[0], loc='upper left')
    axes[1].set_xlim(0.60, 0.80)
    axes[1].set_title('(b) zoom on the transition', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'target_functions.png'), dpi=150)
    plt.close()


# --------------------------------------------------------- Fig 2: training
def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for i, k in enumerate(KS):
        h = T[f"k{k:g}_s0"]['history']
        st = C.k_style(i)
        ep = np.array(h['epoch'])
        for ax, key in zip(axes, ['train_loss', 'val_loss']):
            ax.plot(ep, h[key], lw=1.3, label=f"$k={k:g}$", **st)
        ad = T[f"k{k:g}_s0"]['adequacy']
        axes[1].plot([ad['val_min_epoch']], [ad['val_min']], marker='o',
                     color=st['color'], ms=6, mew=1.2, mfc='none')
    axes[0].set_yscale('log'); axes[0].set_ylabel('training MSE (full train set)')
    axes[0].set_title('Training loss', fontsize=10)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('validation MSE (2000 held-out images)')
    axes[1].set_title('Validation loss (open marker = minimum)', fontsize=10)
    for ax in axes:
        ax.set_xlabel('epoch'); ax.grid(alpha=0.3)
    k_legend(axes[0], loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'training_curves.png'), dpi=150)
    plt.close()


# ------------------------------------------------------ Fig 3: predictions
def fig_predictions():
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2), sharey=True, sharex=True)
    for i, (k, ax) in enumerate(zip(KS, axes.ravel())):
        st = C.k_style(i)
        y = C.target_fn(GRID, k)
        pm = stack('final', k, 'pred_mean').mean(axis=0)
        img_sd = stack('final', k, 'pred_std').mean(axis=0)
        ax.fill_between(GRID, pm - img_sd, pm + img_sd, color=st['color'],
                        alpha=0.22, lw=0)
        ax.plot(GRID, y, color='0.25', ls=':', lw=2, label='target $y_k(b)$')
        ax.plot(GRID, pm, lw=1.8, label='mean prediction', **st)
        r2 = np.mean([get('final', k, s)['sweep_r2'] for s in SEEDS])
        ax.set_title(f"$k={k:g}$   $R^2={r2:.3f}$", fontsize=10)
        ax.grid(alpha=0.3)
        ax.axvspan(C.CENTER_LO, C.CENTER_HI, color='0.85', alpha=0.7, zorder=0)
        if i >= 5:
            ax.set_xlabel('brightness $b$')
        if i % 5 == 0:
            ax.set_ylabel('target / prediction')
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'prediction_sweeps.png'), dpi=150)
    plt.close()


# --------------------------------------------------- Fig 4: movement curves
def fig_movement():
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
    for i, k in enumerate(KS):
        axes[0].plot(BMID, A['target'][f"{k:g}"]['s'], lw=1.5,
                     label=f"$k={k:g}$", **C.k_style(i))
    axes[0].set_title('target $|\\Delta y_k|$ (normalized)', fontsize=10)
    axes[0].set_yscale('log')
    axes[0].set_ylim(1e-5, 2.0)
    k_legend(axes[0], loc='lower left')
    for j, l in enumerate(LAYERS):
        ax = axes[j + 1]
        for i, k in enumerate(KS):
            st = C.k_style(i)
            s = stack('final', k, 's_mean', l)          # [seeds, 200]
            m = s.mean(axis=0)
            ax.plot(BMID, m, lw=1.5, label=f"$k={k:g}$", **st)
            e = ci95(s)
            ax.fill_between(BMID, m - e, m + e, color=st['color'], alpha=0.18, lw=0)
        ax.set_title(f'hidden layer {l}' + (' (deepest)' if l == 3 else ''), fontsize=10)
    for ax in axes:
        ax.axhline(1 / (C.N_SWEEP - 1), color='0.35', lw=1, ls=(0, (1, 2)))
        ax.axvspan(C.CENTER_LO, C.CENTER_HI, color='0.85', alpha=0.7, zorder=0)
        ax.set_xlabel('brightness $b$'); ax.grid(alpha=0.3)
    axes[0].set_ylabel('normalized movement $s(b)$  (log)')
    axes[1].set_ylabel('normalized movement $s_l(b)$')
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'activation_movement_by_k.png'), dpi=150)
    plt.close()


# ------------------------------------------------ Fig 5: concentration vs k
def fig_concentration():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax.plot(KS, gamma_target(), color='0.25', ls=':', lw=2, marker='*', ms=11,
            label='target $y_k$ (reference)')
    ax.plot(KS, gamma_pred(), color=C.CVD[4], ls=(0, (3, 1, 1, 1)), lw=1.6,
            marker='v', ms=6, label='model output $\\hat y$')
    for j, l in enumerate(LAYERS):
        m, e = gamma_stats('final', l)
        ax.errorbar(KS, m, yerr=e, color=C.CVD[j], ls=C.LINESTYLES[j], lw=1.8,
                    marker=C.MARKERS[j], ms=6, capsize=3, label=f'hidden layer {l}')
    ax.axhline(1.0, color='0.45', lw=1, ls=(0, (1, 2)))
    ax.axhline(5.0, color='0.45', lw=1, ls=(0, (4, 2)))
    ax.text(0.52, 0.76, 'uniform movement ($\\Gamma=1$)', fontsize=8, color='0.35')
    ax.text(0.52, 4.62, 'maximum ($\\Gamma=5$: all movement in the window)',
            fontsize=8, color='0.35')
    ax.set_ylim(0.7, 5.5)
    klabel(ax)
    ax.set_xlabel('target sharpness $k$  (log scale)')
    ax.set_ylabel('concentration gain $\\Gamma = C/0.2$')
    ax.set_title('(a) movement in the middle 20% of brightness', fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc='center left')
    ax.grid(alpha=0.3)

    ax2.plot(KS, phi_target(), color='0.25', ls=':', lw=2, marker='*', ms=11,
             label='target $y_k$ (reference)')
    ax2.plot(KS, phi_pred(), color=C.CVD[4], ls=(0, (3, 1, 1, 1)), lw=1.6,
             marker='v', ms=6, label='model output $\\hat y$')
    for j, l in enumerate(LAYERS):
        m, e = phi_stats('final', l)
        ax2.errorbar(KS, m, yerr=e, color=C.CVD[j], ls=C.LINESTYLES[j], lw=1.8,
                     marker=C.MARKERS[j], ms=6, capsize=3, label=f'hidden layer {l}')
    ax2.axhline(0.4, color='0.45', lw=1, ls=(0, (1, 2)))
    ax2.text(0.52, 0.415, 'uniform movement ($\\Phi=0.4$)', fontsize=8, color='0.35')
    ax2.set_ylim(-0.02, 0.47)
    klabel(ax2)
    ax2.set_xlabel('target sharpness $k$  (log scale)')
    ax2.set_ylabel('flank share $\\Phi$')
    ax2.set_title('(b) movement left in the outer 40% ($\\Phi\\to0$ = true plateau)',
                  fontsize=10)
    ax2.legend(frameon=False, fontsize=9, loc='center left')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'concentration_vs_k.png'), dpi=150)
    plt.close()


# ------------------------------------- Fig 6: saturation / fit-quality check
def fig_saturation():
    """Does the deepest layer keep sharpening once the target is a step, and is
    the ceiling an artefact of the model failing to fit the step?"""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.5))
    ax.plot(KS, gamma_target(), color='0.25', ls=':', lw=2, marker='*', ms=11,
            label='target $y_k$ (reference)')
    for an, tag, ls, mfc in [(A, '1000 train images (primary)', '-', None),
                             (A10, '10000 train images', '--', 'none')]:
        m, e = gamma_stats('final', 3, an)
        ax.errorbar(KS, m, yerr=e, color=C.CVD[2], ls=ls, lw=1.8, marker='^',
                    ms=6, mfc=mfc, capsize=3, label=f'layer 3 — {tag}')
        ax.plot(KS, gamma_pred(an), color=C.CVD[4], ls=ls, lw=1.4, marker='v',
                ms=5, mfc=mfc, label=f'model output — {tag}')
    ax.axhline(1.0, color='0.45', lw=1, ls=(0, (1, 2)))
    ax.axhline(5.0, color='0.45', lw=1, ls=(0, (4, 2)))
    ax.set_ylim(0.7, 5.5)
    klabel(ax)
    ax.set_xlabel('target sharpness $k$  (log scale)')
    ax.set_ylabel('concentration gain $\\Gamma$')
    ax.set_title('(a) the deepest layer saturates near $\\Gamma\\approx1.5-1.9$', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    for an, tag, ls, mfc in [(A, '1000 train images (primary)', '-', None),
                             (A10, '10000 train images', '--', 'none')]:
        r2 = [np.mean([get('final', k, s, an)['sweep_r2'] for s in SEEDS]) for k in KS]
        ax2.plot(KS, r2, color=C.CVD[0], ls=ls, lw=1.8, marker='o', ms=6,
                 mfc=mfc, label=f'sweep $R^2$ — {tag}')
    ax2.axhline(1.0, color='0.45', lw=1, ls=(0, (1, 2)))
    klabel(ax2)
    ax2.set_ylim(0.5, 1.03)
    ax2.set_xlabel('target sharpness $k$  (log scale)')
    ax2.set_ylabel('sweep $R^2$ (prediction vs target)')
    ax2.set_title('(b) how well the model actually fits the target', fontsize=10)
    ax2.legend(frameon=False, fontsize=8, loc='lower left')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'saturation_and_fit.png'), dpi=150)
    plt.close()


# --------------------------------------- Fig 7: checkpoint robustness check
def fig_robustness():
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    for j, l in enumerate(LAYERS):
        for w, (which, ls, mk) in enumerate([('final', '-', 'o'), ('best_val', '--', 's')]):
            m, e = gamma_stats(which, l)
            ax.errorbar(KS, m, yerr=e, color=C.CVD[j], ls=ls, lw=1.6, marker=mk,
                        ms=5, capsize=3, mfc='none' if w else None,
                        label=f"layer {l} — {'min-val' if w else 'final'} ckpt")
    ax.axhline(1.0, color='0.45', lw=1, ls=(0, (1, 2)))
    klabel(ax)
    ax.set_xlabel('target sharpness $k$  (log scale)')
    ax.set_ylabel('concentration gain $\\Gamma_l(k)$')
    ax.set_title('final vs minimum-validation-loss checkpoint', fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'checkpoint_robustness.png'), dpi=150)
    plt.close()


# ----------------------------------------------------- Fig 8: main summary
def fig_summary():
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.7))
    for i, k in enumerate(KS):
        axes[0].plot(GRID, C.target_fn(GRID, k), lw=1.7, label=f"$k={k:g}$",
                     **C.k_style(i))
    axes[0].set_ylabel('target $y_k(b)$')
    axes[0].set_title('(a) target family', fontsize=10)
    k_legend(axes[0], loc='upper left')

    for i, k in enumerate(KS):
        pm = stack('final', k, 'pred_mean').mean(axis=0)
        axes[1].plot(GRID, pm, lw=1.7, **C.k_style(i))
    axes[1].set_ylabel('mean prediction $\\hat y(b)$')
    axes[1].set_title('(b) predictions', fontsize=10)

    for i, k in enumerate(KS):
        st = C.k_style(i)
        s = stack('final', k, 's_mean', 3)
        m = s.mean(axis=0); e = ci95(s)
        axes[2].plot(BMID, m, lw=1.5, **st)
        axes[2].fill_between(BMID, m - e, m + e, color=st['color'], alpha=0.18, lw=0)
    axes[2].axhline(1 / (C.N_SWEEP - 1), color='0.35', lw=1, ls=(0, (1, 2)))
    axes[2].set_ylabel('$s_3(b)$ (deepest layer)')
    axes[2].set_title('(c) normalized activation movement', fontsize=10)

    axes[3].plot(KS, gamma_target(), color='0.25', ls=':', lw=2, marker='*',
                 ms=11, label='target')
    for j, l in enumerate(LAYERS):
        m, e = gamma_stats('final', l)
        axes[3].errorbar(KS, m, yerr=e, color=C.CVD[j], ls=C.LINESTYLES[j], lw=1.8,
                         marker=C.MARKERS[j], ms=6, capsize=3, label=f'layer {l}')
    axes[3].axhline(1.0, color='0.45', lw=1, ls=(0, (1, 2)))
    axes[3].axhline(5.0, color='0.45', lw=1, ls=(0, (4, 2)))
    axes[3].set_ylim(0.7, 5.5)
    klabel(axes[3])
    axes[3].set_xlabel('target sharpness $k$ (log)')
    axes[3].set_ylabel('$\\Gamma_l(k)=C_l(k)/0.2$')
    axes[3].set_title('(d) concentration gain', fontsize=10)
    axes[3].legend(frameon=False, fontsize=8)

    for ax in axes[:3]:
        ax.set_xlabel('brightness $b$')
        ax.axvspan(C.CENTER_LO, C.CENTER_HI, color='0.85', alpha=0.7, zorder=0)
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.PLOTS, 'main_summary.png'), dpi=150)
    plt.close()


# ------------------------------------------------------------- tables
def table(an, tr, out_name):
    rows = []
    gt, pt = gamma_target(), phi_target()
    gp, pp = gamma_pred(an), phi_pred(an)
    for i, k in enumerate(KS):
        r = {'k': k, 'target_G': float(gt[i]), 'target_P': float(pt[i]),
             'pred_G': float(gp[i]), 'pred_P': float(pp[i]),
             'r2': float(np.mean([get('final', k, s, an)['sweep_r2'] for s in SEEDS])),
             'val': float(np.mean([tr[f"k{k:g}_s{s}"]['adequacy']['val_final'] for s in SEEDS])),
             'rho': float(np.mean([tr[f"k{k:g}_s{s}"]['adequacy']['val_final_over_min']
                                   for s in SEEDS])),
             'val_min_ep': float(np.mean([tr[f"k{k:g}_s{s}"]['adequacy']['val_min_epoch']
                                          for s in SEEDS]))}
        for l in LAYERS:
            g = stack('final', k, 'C_mean', l, an) / 0.2
            f = stack('final', k, 's_mean', l, an)[:, FLANK].sum(axis=1)
            r[f'G{l}'] = float(g.mean()); r[f'G{l}_ci'] = float(ci95(g))
            r[f'P{l}'] = float(f.mean()); r[f'P{l}_ci'] = float(ci95(f))
            r[f'G{l}_img_sd'] = float(stack('final', k, 'C_std_images', l, an).mean() / 0.2)
            r[f'G{l}_bestval'] = float((stack('best_val', k, 'C_mean', l, an) / 0.2).mean())
        rows.append(r)
    with open(os.path.join(C.RESULTS, out_name), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"--- {out_name}")
    for r in rows:
        print(f"k={r['k']:>5g}  tgtG={r['target_G']:5.2f}  outG={r['pred_G']:5.2f}  "
              f"G1={r['G1']:.3f}  G2={r['G2']:.3f}  G3={r['G3']:.3f}+-{r['G3_ci']:.3f}  "
              f"P3={r['P3']:.3f}  R2={r['r2']:.3f}  val={r['val']:.4f}")
    return rows


if __name__ == '__main__':
    os.makedirs(C.PLOTS, exist_ok=True)
    fig_targets(); fig_training(); fig_predictions(); fig_movement()
    fig_concentration(); fig_saturation(); fig_robustness(); fig_summary()
    table(A, T, 'summary_table.json')
    table(A10, T10, 'summary_table_n10k.json')
    print('figures written to plots/')
