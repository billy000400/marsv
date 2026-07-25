"""
S1/S2 analysis — apply the FROZEN rules (JOURNAL.md, iter 1 part A) to the
45-transition census and emit the classification table plus Stage-1 figures.

Rule 1 (stable third-class): some digit z not in {a,b} appears as a contiguous
predicted-class segment in >= 50/100 paths, with median longest-z-run >= 3
alpha points.
Rule 2 (stable sub-plateau): the 100-pair mean d(alpha) curve contains a
maximal flat run of >= 5 alpha points with range <= 0.05 whose mean level is in
[0.15, 0.85].
Rule 3 (robustness): leave-one-out over the 100 pairs must not change the
Rule-2 label; 1,000 bootstrap resamples must agree >= 90%.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = '/workspace/marsv_agent_haoyang/dir121_third_class_prediction'
TAU_FLAT, L_MIN, LEVEL_LO, LEVEL_HI = 0.05, 5, 0.15, 0.85
PREVALENCE, MED_RUN = 50, 3
N_BOOT = 1000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- frozen rules
def maximal_flat_runs(m):
    """Maximal contiguous runs of m with max-min <= TAU_FLAT. -> [(i, j)]."""
    n, runs, prev_j = len(m), [], -1
    for i in range(n):
        j = i
        while j + 1 < n and m[i:j + 2].max() - m[i:j + 2].min() <= TAU_FLAT:
            j += 1
        if j > prev_j:
            runs.append((i, j))
            prev_j = j
    return runs


def shelf(m):
    """Rule 2. -> (has_shelf, widest intermediate run as dict or None)."""
    cand = []
    for i, j in maximal_flat_runs(m):
        if j - i + 1 >= L_MIN and LEVEL_LO <= m[i:j + 1].mean() <= LEVEL_HI:
            cand.append((i, j))
    if not cand:
        return False, None
    i, j = max(cand, key=lambda r: r[1] - r[0])
    n = len(m)
    return True, {'i': int(i), 'j': int(j), 'n_points': int(j - i + 1),
                  'alpha_lo': round(i / (n - 1), 4), 'alpha_hi': round(j / (n - 1), 4),
                  'level': round(float(m[i:j + 1].mean()), 4)}


def third_runs(pred_row, a, b):
    """Longest contiguous run length per third digit z on one path. -> {z: len}."""
    out = {}
    z, start = None, 0
    for t in range(len(pred_row) + 1):
        cur = int(pred_row[t]) if t < len(pred_row) else -1
        if cur != z:
            if z is not None and z != a and z != b:
                out[z] = max(out.get(z, 0), t - start)
            z, start = cur, t
    return out


def third_stats(pred_tr, a, b):
    """Per-transition third-class stats over the paths in pred_tr [n_paths,50]."""
    n = len(pred_tr)
    runs = [third_runs(p, a, b) for p in pred_tr]
    stats = {}
    for zc in range(10):
        if zc in (a, b):
            continue
        lens = [r[zc] for r in runs if zc in r]
        if not lens:
            continue
        stats[zc] = {'n_paths': len(lens), 'prevalence': len(lens) / n,
                     'median_run': float(np.median(lens)),
                     'iqr_run': [float(np.percentile(lens, 25)),
                                 float(np.percentile(lens, 75))],
                     'runs': lens}
    any_third = float(np.mean([len(r) > 0 for r in runs]))
    return stats, any_third


def stable_third(stats, n_paths, prevalence=PREVALENCE):
    """Rule 1 -> (label, dominant z or None). Ties broken by higher prevalence."""
    ok = [(z, s) for z, s in stats.items()
          if s['n_paths'] >= prevalence * n_paths / 100 and s['median_run'] >= MED_RUN]
    if not ok:
        return False, None
    return True, int(max(ok, key=lambda kv: (kv[1]['n_paths'], kv[1]['median_run']))[0])


# ------------------------------------------------------------------- analysis
def main():
    z = np.load(os.path.join(HERE, 'results', 's1_census.npz'), allow_pickle=False)
    trans = z['transitions']                      # [45,2]
    pred = z['pred']                              # [n_seed,4500,50]
    d = z['d_logit']
    end_pred, lab_a, lab_b = z['end_pred'], z['label_a'], z['label_b']
    seeds, n_per, n_pts = z['seeds'], int(z['n_per']), int(z['n_points'])
    n_tr = len(trans)
    pred = pred.reshape(len(seeds), n_tr, n_per, n_pts)
    d = d.reshape(len(seeds), n_tr, n_per, n_pts)
    end_pred = end_pred.reshape(len(seeds), n_tr, n_per, 2)
    alpha = np.linspace(0, 1, n_pts)

    table = []
    for k, (a, b) in enumerate(trans):
        row = {'a': int(a), 'b': int(b), 'transition': f'{a}->{b}', 'seeds': {}}
        good = ((end_pred[:, k, :, 0] == lab_a[k][None, :]) &
                (end_pred[:, k, :, 1] == lab_b[k][None, :]))       # [n_seed,100]
        for si, seed in enumerate(seeds):
            m, sd = d[si, k].mean(0), d[si, k].std(0)
            has_shelf, sh = shelf(m)
            st, any_third = third_stats(pred[si, k], a, b)
            lab3, zdom = stable_third(st, n_per)
            # correct-endpoint subset
            sub = pred[si, k][good[si]]
            st_sub, any_third_sub = third_stats(sub, a, b) if len(sub) else ({}, 0.0)
            lab3_sub, zdom_sub = (stable_third(st_sub, len(sub))
                                  if len(sub) else (False, None))
            e = {'stable_third': bool(lab3), 'z_dominant': zdom,
                 'prevalence_z': (round(st[zdom]['prevalence'], 3) if zdom is not None else 0.0),
                 'median_run_z': (st[zdom]['median_run'] if zdom is not None else 0.0),
                 'iqr_run_z': (st[zdom]['iqr_run'] if zdom is not None else None),
                 'any_third_frac': round(any_third, 3),
                 'sub_plateau': bool(has_shelf), 'shelf': sh,
                 'endpoint_correct_frac': round(float(good[si].mean()), 3),
                 'n_correct_endpoint_paths': int(good[si].sum()),
                 'stable_third_correct_subset': bool(lab3_sub),
                 'z_dominant_correct_subset': zdom_sub,
                 'any_third_frac_correct_subset': round(any_third_sub, 3),
                 'prevalence_all_z': {int(zz): round(s['prevalence'], 3)
                                      for zz, s in sorted(st.items())},
                 'sensitivity': {p: bool(stable_third(st, n_per, p)[0])
                                 for p in (25, 50, 75)}}
            if si == 0:                                   # robustness on seed 0 only
                dk = d[si, k]
                loo = [shelf(np.delete(dk, i, axis=0).mean(0))[0] for i in range(n_per)]
                bs = [shelf(dk[RNG.integers(0, n_per, n_per)].mean(0))[0]
                      for _ in range(N_BOOT)]
                e['loo_agree'] = int(np.sum(np.array(loo) == has_shelf))
                e['boot_agree'] = round(float(np.mean(np.array(bs) == has_shelf)), 3)
                loo3 = [stable_third(third_stats(np.delete(pred[si, k], i, 0), a, b)[0],
                                     n_per - 1)[0] for i in range(n_per)]
                e['loo_agree_third'] = int(np.sum(np.array(loo3) == lab3))
            row['seeds'][int(seed)] = e
        table.append(row)

    meta = {'ckpt_paths': [str(p) for p in z['ckpt_paths']],
            'ckpt_sha256': [str(h) for h in z['ckpt_sha256']],
            'step': int(z['step']), 'n_per': n_per, 'n_points': n_pts,
            'rules': {'tau_flat': TAU_FLAT, 'l_min': L_MIN,
                      'level_band': [LEVEL_LO, LEVEL_HI],
                      'prevalence_pct': PREVALENCE, 'median_run_min': MED_RUN,
                      'n_bootstrap': N_BOOT}}
    json.dump({'meta': meta, 'transitions': table},
              open(os.path.join(HERE, 'results', 's1_classification.json'), 'w'), indent=1)

    with open(os.path.join(HERE, 'results', 's1_classification.csv'), 'w') as f:
        f.write('transition,seed,stable_third,z,prevalence_z,median_run_z,any_third_frac,'
                'sub_plateau,shelf_alpha_lo,shelf_alpha_hi,shelf_level,endpoint_correct_frac,'
                'stable_third_correct_subset,loo_agree,boot_agree\n')
        for r in table:
            for seed, e in r['seeds'].items():
                sh = e['shelf'] or {}
                f.write(f"{r['transition']},{seed},{int(e['stable_third'])},"
                        f"{e['z_dominant'] if e['z_dominant'] is not None else ''},"
                        f"{e['prevalence_z']},{e['median_run_z']},{e['any_third_frac']},"
                        f"{int(e['sub_plateau'])},{sh.get('alpha_lo','')},"
                        f"{sh.get('alpha_hi','')},{sh.get('level','')},"
                        f"{e['endpoint_correct_frac']},{int(e['stable_third_correct_subset'])},"
                        f"{e.get('loo_agree','')},{e.get('boot_agree','')}\n")
    print('saved results/s1_classification.{json,csv}')
    make_figures(z, table, trans, pred, d, alpha)
    print_summary(table)


# --------------------------------------------------------------------- figures
CAT_COLORS = {0: '#e8e8e8', 1: '#4C78A8', 2: '#F58518', 3: '#54A24B'}
CAT_NAMES = {0: 'neither', 1: 'stable third-class only',
             2: 'stable sub-plateau only', 3: 'both'}
PLOTS = os.path.join(HERE, 'plots')


def category(e):
    return (1 if e['stable_third'] else 0) + (2 if e['sub_plateau'] else 0)


def make_figures(z, table, trans, pred, d, alpha):
    os.makedirs(PLOTS, exist_ok=True)
    n_pts = len(alpha)

    # 1 — 10x10 category matrix (seed 0), diagonal blank
    M = np.full((10, 10), np.nan)
    for r in table:
        c = category(r['seeds'][0])
        M[r['a'], r['b']] = c
        M[r['b'], r['a']] = c
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    rgb = np.ones((10, 10, 3))
    for i in range(10):
        for j in range(10):
            if not np.isnan(M[i, j]):
                rgb[i, j] = matplotlib.colors.to_rgb(CAT_COLORS[int(M[i, j])])
    ax.imshow(rgb)
    for r in table:
        e = r['seeds'][0]
        if e['stable_third']:
            for (i, j) in ((r['a'], r['b']), (r['b'], r['a'])):
                ax.text(j, i, str(e['z_dominant']), ha='center', va='center',
                        fontsize=9, color='w', fontweight='bold')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('endpoint digit b'); ax.set_ylabel('endpoint digit a')
    ax.set_title('Digit-transition census (seed 0, step 30k, 100 pairs each)\n'
                 'cell text = dominant third digit z', fontsize=11)
    ax.legend(handles=[Patch(facecolor=CAT_COLORS[c], label=CAT_NAMES[c]) for c in range(4)],
              loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 's1_transition_matrix.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # 2 — 5x9 grid of all 45 mean curves with +/-1 std band
    fig, axes = plt.subplots(5, 9, figsize=(20, 11.5), sharex=True, sharey=True)
    for k, r in enumerate(table):
        ax = axes[k // 9, k % 9]
        e = r['seeds'][0]
        m, sd = d[0, k].mean(0), d[0, k].std(0)
        ax.plot([0, 1], [0, 1], ls=':', c='0.75', lw=1)
        ax.plot(alpha, m, c=CAT_COLORS[category(e)] if category(e) else '0.35', lw=2)
        ax.fill_between(alpha, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1),
                        color=CAT_COLORS[category(e)] if category(e) else '0.35', alpha=0.22)
        if e['shelf']:
            ax.axhspan(e['shelf']['level'] - 0.025, e['shelf']['level'] + 0.025,
                       xmin=e['shelf']['alpha_lo'], xmax=e['shelf']['alpha_hi'],
                       color='k', alpha=0.18)
        ttl = f"{r['transition']}"
        if e['stable_third']:
            ttl += f"  z={e['z_dominant']} ({e['prevalence_z']:.0%})"
        ax.set_title(ttl, fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if k // 9 == 4:
            ax.set_xlabel(r'$\alpha$', fontsize=9)
        if k % 9 == 0:
            ax.set_ylabel(r'$d(\alpha)$ (logit space)', fontsize=9)
    fig.suptitle('Mean logit-space relative endpoint distance over 100 image pairs, all 45 '
                 'digit transitions (seed 0, step 30k); band = ±1 std across pairs; '
                 'grey bar = detected sub-plateau shelf', fontsize=13)
    fig.legend(handles=[Patch(facecolor=CAT_COLORS[c], label=CAT_NAMES[c]) for c in range(4)],
               loc='lower center', ncol=4, fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.965])
    fig.savefig(os.path.join(PLOTS, 's1_mean_curves_grid.png'), dpi=110)
    plt.close(fig)

    # 3 & 4 — class composition and segment widths for stable third-class transitions
    pos = [k for k, r in enumerate(table) if r['seeds'][0]['stable_third']]
    if pos:
        ncol = min(5, len(pos))
        nrow = int(np.ceil(len(pos) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.9 * nrow),
                                 squeeze=False, sharex=True, sharey=True)
        for ax in axes.ravel():
            ax.axis('off')
        for n, k in enumerate(pos):
            r, e = table[k], table[k]['seeds'][0]
            ax = axes[n // ncol, n % ncol]
            ax.axis('on')
            comp = np.stack([(pred[0, k] == c).mean(0) for c in range(10)])   # [10,50]
            ax.stackplot(alpha, comp, colors=[plt.cm.tab10(c) for c in range(10)])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{r['transition']}   z={e['z_dominant']}", fontsize=10)
            if n // ncol == nrow - 1:
                ax.set_xlabel(r'$\alpha$', fontsize=9)
            if n % ncol == 0:
                ax.set_ylabel('fraction of 100 paths', fontsize=9)
        fig.suptitle('Predicted-class composition across the interpolation, stable '
                     'third-class transitions (seed 0)', fontsize=12)
        fig.legend(handles=[Patch(facecolor=plt.cm.tab10(c), label=f'predicted {c}')
                            for c in range(10)],
                   loc='lower center', ncol=10, fontsize=9)
        fig.tight_layout(rect=[0, 0.045, 1, 0.94])
        fig.savefig(os.path.join(PLOTS, 's1_class_composition.png'), dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(1.05 * len(pos) + 3.2, 4.4))
        data, labels = [], []
        for k in pos:
            r, e = table[k], table[k]['seeds'][0]
            zz = e['z_dominant']
            data.append([third_runs(p, r['a'], r['b']).get(zz, 0) for p in pred[0, k]])
            labels.append(f"{r['transition']}\nz={zz}")
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.axhline(MED_RUN, ls='--', c='r', lw=1,
                   label=f'frozen median-run threshold = {MED_RUN}')
        ax.set_ylabel('longest run of consecutive $\\alpha$ points predicted $z$\n'
                      '(0 = that path never predicts $z$)')
        ax.set_xlabel('transition (of 50 alpha points per path)')
        ax.set_title('Per-pair third-class segment width, all 100 pairs (seed 0)', fontsize=11)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, 's1_segment_widths.png'), dpi=150)
        plt.close(fig)

    # 5 — cross-seed agreement of both labels, and the third-digit identity
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))
    for ax, key, name in zip(axes[:2], ['stable_third', 'sub_plateau'],
                             ['stable third-class prediction', 'stable sub-plateau']):
        A = np.zeros((len(table), 3))
        for k, r in enumerate(table):
            for si, s in enumerate((0, 1, 2)):
                A[k, si] = r['seeds'][s][key]
        ax.imshow(A.T, aspect='auto', cmap='Greys', vmin=0, vmax=1.6)
        ax.set_yticks(range(3)); ax.set_yticklabels([f'seed {s}' for s in (0, 1, 2)])
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels([r['transition'] for r in table], rotation=90, fontsize=6)
        ax.set_title(f'{name}: black = label true (per seed, {int(A[:, 0].sum())}/'
                     f'{int(A[:, 1].sum())}/{int(A[:, 2].sum())} of 45 for seeds 0/1/2); '
                     f'all three seeds agree on {int((A.sum(1) % 3 == 0).sum())}/45 transitions',
                     fontsize=11)
        ax.set_xlabel('digit transition')
    ax = axes[2]
    Z = np.full((len(table), 3), np.nan)
    for k, r in enumerate(table):
        for si, s in enumerate((0, 1, 2)):
            if r['seeds'][s]['stable_third']:
                Z[k, si] = r['seeds'][s]['z_dominant']
    ax.imshow(np.ma.masked_invalid(Z.T), aspect='auto', cmap='tab10', vmin=-0.5, vmax=9.5)
    for k in range(len(table)):
        for si in range(3):
            if not np.isnan(Z[k, si]):
                ax.text(k, si, str(int(Z[k, si])), ha='center', va='center',
                        fontsize=7, color='w', fontweight='bold')
    ax.set_yticks(range(3)); ax.set_yticklabels([f'seed {s}' for s in (0, 1, 2)])
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels([r['transition'] for r in table], rotation=90, fontsize=6)
    ax.set_xlabel('digit transition')
    ax.set_title('Dominant third digit $z$ where the transition is labelled stable third-class '
                 '(white = not stable at that seed).\nEach seed has its own preferred third '
                 'digit: 7/8 at seed 0, 1 at seed 1, 2/8 at seed 2.', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 's1_seed_agreement.png'), dpi=150)
    plt.close(fig)
    print('saved 5 figures to plots/')


def print_summary(table):
    cats = [category(r['seeds'][0]) for r in table]
    print('\nseed-0 category counts:',
          {CAT_NAMES[c]: cats.count(c) for c in range(4)})
    for r in table:
        e = r['seeds'][0]
        if e['stable_third'] or e['sub_plateau']:
            print(f"  {r['transition']:6s} cat={CAT_NAMES[category(e)]:26s} "
                  f"z={e['z_dominant']} prev={e['prevalence_z']} med={e['median_run_z']} "
                  f"shelf={e['shelf']} loo={e.get('loo_agree')} boot={e.get('boot_agree')} "
                  f"seed1={e is None or table[0] is None or ''}"
                  f"{r['seeds'][1]['stable_third']}/{r['seeds'][1]['sub_plateau']} "
                  f"seed2={r['seeds'][2]['stable_third']}/{r['seeds'][2]['sub_plateau']}")


if __name__ == '__main__':
    main()
