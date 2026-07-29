"""Figures for operator feedback #2 — movement INSIDE the target transition.

Reads results/zoom_curves_<tag>.npz + results/zoom_<tag>.json (written by
zoom.py, 6001-point brightness grid) and produces:

  plots/transition_zoom.png       primary grid, 2 zoom levels x 5 signals
  plots/transition_zoom_n10k.png  10,000-image control at the tightest zoom
  plots/transition_scale.png      scale-resolved concentration + peak rate

Plotted quantity is the movement RATE relative to uniform,
    g(b_i) = s(b_i) * (S-1),      g = 1 <=> movement spread evenly over b,
so all panels can share one log y-axis and the target's spike is comparable
with the layers' on the same scale.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common as C

KS = list(C.K_VALUES)
SEEDS = [0, 1, 2]
SIGNALS = [('target', 'target $|\\Delta y_k|$'), ('out', 'model output $\\hat y$'),
           ('L1', 'hidden layer 1'), ('L2', 'hidden layer 2'),
           ('L3', 'hidden layer 3 (deepest)')]
WINDOWS = [0.06, 0.03, 0.01, 0.005, 0.0025]


def load(tag):
    z = np.load(os.path.join(C.RESULTS, f'zoom_curves_{tag}.npz'))
    j = json.load(open(os.path.join(C.RESULTS, f'zoom_{tag}.json')))
    grid = z['grid']
    bmid = 0.5 * (grid[:-1] + grid[1:])
    n_int = len(bmid)
    return z, j, bmid, n_int


def rate(z, tag_key, n_int):
    """movement share -> rate relative to uniform (1 = uniform)."""
    return z[tag_key] * n_int


def mean_rate(z, sig, k, n_int, ks_avail):
    if sig == 'target':
        return rate(z, f'target_k{k:g}', n_int)
    if k not in ks_avail:
        return None
    return np.mean([rate(z, f'k{k:g}|s{s}|{sig}', n_int) for s in SEEDS], axis=0)


def available_ks(j):
    return sorted({float(key.split('|')[0][1:]) for key in j['models']})


def zoom_panels(ax, z, bmid, n_int, sig, half, ks_avail, logy, legend=False):
    sel = np.abs(bmid - C.B0) <= half
    for i, k in enumerate(KS):
        y = mean_rate(z, sig, k, n_int, ks_avail)
        if y is None:
            continue
        st = C.k_style(i)
        ax.plot(bmid[sel], y[sel], lw=1.5, label=f"$k={k:g}$", **st)
    ax.axhline(1.0, color='0.35', ls=':', lw=1)
    ax.axvline(C.B0, color='0.6', ls=':', lw=0.8)
    if logy:
        ax.set_yscale('log')
        ax.set_ylim(0.3, 300)
    ax.set_xlim(C.B0 - half, C.B0 + half)
    ax.grid(alpha=0.3, which='both')
    if legend:
        ax.legend(fontsize=6, ncol=2, loc='upper left', framealpha=0.9)


def fig_zoom(tag, out_name, title):
    """Row 1: window +-0.04 on ONE shared log axis (the scale gap).
    Row 2: window +-0.0025 (finer than the k=320 transition), each panel
    autoscaled linearly so the layers' shapes are visible."""
    z, j, bmid, n_int = load(tag)
    ks_avail = available_ks(j)
    rows = [(0.04, True), (0.0025, False)]
    nc = len(SIGNALS)
    fig, axes = plt.subplots(len(rows), nc, figsize=(4.0 * nc, 3.4 * len(rows)),
                             squeeze=False)
    for r, (half, logy) in enumerate(rows):
        for c, (sig, name) in enumerate(SIGNALS):
            ax = axes[r][c]
            zoom_panels(ax, z, bmid, n_int, sig, half, ks_avail, logy,
                        legend=(r == 0 and c == 0))
            if r == 0:
                ax.set_title(name, fontsize=11)
            else:
                ax.set_xlabel('brightness $b$')
                ax.ticklabel_format(axis='x', useOffset=False)
                ax.tick_params(axis='x', labelrotation=20, labelsize=8)
            if c == 0:
                lab = ('shared log axis' if logy else 'own linear axis')
                ax.set_ylabel(f'movement rate $g(b)$  (x uniform)\n'
                              f'zoom $b_0 \\pm {half:g}$, {lab}', fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(C.PLOTS, out_name), dpi=150)
    plt.close(fig)


STYLE = {'target': dict(color='0.15', ls=':', marker='*', ms=9),
         'out': dict(color=C.CVD[4], ls='-.', marker='v', ms=5),
         'L1': dict(color=C.CVD[0], ls='-', marker='o', ms=4),
         'L2': dict(color=C.CVD[1], ls='--', marker='s', ms=4),
         'L3': dict(color=C.CVD[2], ls='-.', marker='^', ms=5)}


def _series(j, sig, field, ks):
    return [np.mean([j['models'][f'k{k:g}|s{s}'][sig][field] for s in SEEDS])
            for k in ks]


def fig_scale(kx=320.0):
    """(a) Gamma(w) at k=320 -- window centred on b0, every scale.
       (b) Lambda(w) at k=320 -- best window ANYWHERE, per image.
       (c) Lambda at the finest scale vs k."""
    _, jm, _, _ = load('main')
    _, j1, _, _ = load('n10k')
    ks1 = available_ks(j1)
    w = np.array(WINDOWS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    for ax, field, sym, ttl in (
            (axes[0], 'gamma', '\\Gamma',
             f'(a) window centred on $b_0$, $k={kx:g}$'),
            (axes[1], 'lambda', '\\Lambda',
             f'(b) best window anywhere, $k={kx:g}$')):
        ax.plot(w, 0.3 / w, color='0.55', ls='--', lw=1, label='maximum possible')
        for sig, name in SIGNALS:
            if sig == 'target':
                ax.plot(w, [jm['target'][f'k{kx:g}'][f'{field}_{ww:g}'] for ww in w],
                        label=name, **STYLE[sig])
                continue
            for j, lab, dashed in ((jm, ' (1k imgs)', False), (j1, ' (10k imgs)', True)):
                y = [np.mean([j['models'][f'k{kx:g}|s{s}'][sig][f'{field}_{ww:g}']
                              for s in SEEDS]) for ww in w]
                st = dict(STYLE[sig])
                if dashed:
                    st.update(mfc='none', alpha=0.75, ls=(0, (1, 1)))
                ax.plot(w, y, label=name + lab, **st)
        ax.axhline(1.0, color='0.35', ls=':', lw=1)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
        ax.set_xticks(w); ax.set_xticklabels([f'{ww:g}' for ww in w])
        ax.minorticks_off()
        ax.set_xlabel('window half-width $w$  (brightness units, zooming in $\\to$)')
        ax.set_ylabel(f'concentration gain ${sym}(w)$   (1 = uniform)')
        ax.set_title(ttl)
        ax.grid(alpha=0.3, which='both')

    ax = axes[2]
    ww = min(WINDOWS)
    ax.plot(KS, [jm['target'][f'k{k:g}'][f'lambda_{ww:g}'] for k in KS],
            label=SIGNALS[0][1], **STYLE['target'])
    for sig, name in SIGNALS[1:]:
        ax.plot(KS, _series(jm, sig, f'lambda_{ww:g}', KS),
                label=name + ' (1k imgs)', **STYLE[sig])
        ks_ok = [k for k in KS if k in ks1]
        st = dict(STYLE[sig]); st.update(mfc='none', alpha=0.75, ls=(0, (1, 1)))
        ax.plot(ks_ok, _series(j1, sig, f'lambda_{ww:g}', ks_ok),
                label=name + ' (10k imgs)', **st)
    ax.axhline(1.0, color='0.35', ls=':', lw=1)
    ax.axhline(0.3 / ww, color='0.55', ls='--', lw=1, label='maximum possible')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('target sharpness $k$ (log)')
    ax.set_ylabel(f'$\\Lambda(w={ww:g})$   (1 = uniform)')
    ax.set_title(f'(c) finest scale ($w={ww:g}$) versus $k$')
    ax.grid(alpha=0.3, which='both')

    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=6, loc='lower center', frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(os.path.join(C.PLOTS, 'transition_scale.png'), dpi=150)
    plt.close(fig)


def tables():
    """Numbers quoted in RESULTS.md / REPORT.md (mean +- 95% CI across seeds)."""
    _, jm, _, _ = load('main')
    _, j1, _, _ = load('n10k')
    ks1 = available_ks(j1)

    def cell(j, k, sig, field):
        v = np.array([j['models'][f'k{k:g}|s{s}'][sig][field] for s in SEEDS])
        return f"{v.mean():.3f} ± {1.96 * v.std(ddof=1) / np.sqrt(3):.3f}"

    for field, name in (('gamma_0.0025', 'Gamma(w=0.0025), window centred on b0'),
                        ('lambda_0.0025', 'Lambda(w=0.0025), best window anywhere')):
        for j, tag, avail in ((jm, '1000 images', KS), (j1, '10,000 images', ks1)):
            print(f'\n== {name} — {tag} ==')
            print('| k | target | model output | layer 1 | layer 2 | layer 3 |')
            for k in KS:
                if k not in avail:
                    continue
                t = jm['target'][f'k{k:g}'][field]
                print(f"| {k:g} | {t:.2f} | " + ' | '.join(
                    cell(j, k, s, field) for s in ('out', 'L1', 'L2', 'L3')) + ' |')

    print('\n== dense vs coarse grid, Gamma(w=0.06) = the report Table 1 window ==')
    print('| k | L3 dense (6001 pts) | L3 10k-grid dense |')
    for k in KS:
        row = f"| {k:g} | {cell(jm, k, 'L3', 'gamma_0.06')} | "
        row += (cell(j1, k, 'L3', 'gamma_0.06') if k in ks1 else '-') + ' |'
        print(row)


if __name__ == '__main__':
    fig_zoom('main', 'transition_zoom.png',
             'Movement rate inside the target transition — 1000 training images '
             '(mean over 100 held-out images and 3 seeds)')
    fig_zoom('n10k', 'transition_zoom_n10k.png',
             'Movement rate inside the transition — 10,000-image control '
             '(the grid whose OUTPUT is a genuine switch)')
    fig_scale()
    tables()
    print('\nsaved plots/transition_zoom.png, transition_zoom_n10k.png, '
          'transition_scale.png')
