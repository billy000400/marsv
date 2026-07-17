#!/usr/bin/env python3
"""
dir12 — Feedback 07161650, point 2: is 3-vs-5 harder to classify than other
digit pairs, and does the shape of a pair's interpolation curve (wide mid-level
shelf instead of two endpoint plateaus) track pairwise discriminability?

For every cross-class pair (a,b) at the final MSE seed-0 checkpoint (step
100,000) we compute, on the first 2,000 test images:
  - pairwise AUROC: score s = logit_a - logit_b over images with true label in
    {a,b}; AUROC = P(s_a > s_b) for a random (true-a, true-b) image pair
    (rank/Mann-Whitney estimator, ties counted 1/2);
  - pairwise confusion rate: among true-a and true-b images, the fraction
    predicted as the OTHER class of the pair;
and, from the saved interpolation record at the same checkpoint:
  - mid fraction: fraction of the 50 path points with 0.1 < d(alpha) < 0.9
    (the complement of the per-pair plateau fraction);
  - third-class fraction: fraction of path points predicted as a digit
    outside {a,b}.

Outputs results/pairwise_auc.json + plots/pairwise_auc.png (heatmap, ranking,
scatter vs mid fraction, and the annotated 3->5 curve). Also prints the CE
model's 3-vs-5 AUROC if the CE checkpoint exists.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from plateau_protocol import HERE, DATA, N_TEST_POOL, N_POINTS, load_state_model
from src.mnist import load_mnist

torch.set_num_threads(2)
DIR60K = '--dir60k' in sys.argv                  # full-60k runs (feedback 1)
STEP = 30_000 if DIR60K else 100_000
FM = os.path.join(HERE, 'results', 'full_mnist_from_scratch')
PLOTS = os.path.join(HERE, 'plots')


def auroc(pos, neg):
    """Rank-based AUROC = P(score_pos > score_neg), ties -> 1/2."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_s) + 1)
    # average ranks over ties
    s_sorted = all_s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_p, n_n = len(pos), len(neg)
    return (ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return np.corrcoef(rx, ry)[0, 1]


@torch.no_grad()
def logits_of(model, x, device, batch=2000):
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch].to(device)).cpu())
    return torch.cat(out).numpy()


def main():
    argv = [a for a in sys.argv[1:] if a != '--dir60k']
    sched_sfx = argv[0] if argv else ''   # e.g. _pl_f0.5_p100
    out_sfx = '_60k' if DIR60K else sched_sfx
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.225)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, test_x, test_y = load_mnist(DATA)
    test_x, test_y = test_x[:N_TEST_POOL], test_y[:N_TEST_POOL].numpy()

    if DIR60K:
        model = load_state_model(os.path.join(FM, f'seed_0{sched_sfx}', 'ckpts',
                                              f'step{STEP}.pt'), device)
    else:
        model = load_state_model(os.path.join(HERE, 'results', 'ckpts_movie',
                                              f'seed0{sched_sfx}',
                                              f'step{STEP}.pt'), device)
    L = logits_of(model, test_x, device)
    pred = L.argmax(1)

    rec_base = os.path.join(FM, f'seed_0{sched_sfx}') if DIR60K else \
        os.path.join(HERE, 'results', 'plateau_records', f'seed_0{sched_sfx}')
    rec = np.load(os.path.join(rec_base, f'step_{STEP}.npz'))
    man = json.load(open(os.path.join(rec_base, 'manifest.json')))

    rows = []
    auc_mat = np.full((10, 10), np.nan)
    for i, p in enumerate(man['pairs'][:45]):
        a, b = p['class_a'], p['class_b']
        ia, ib = test_y == a, test_y == b
        score = L[:, a] - L[:, b]
        auc = auroc(score[ia], score[ib])
        conf_rate = ((pred[ia] == b).sum() + (pred[ib] == a).sum()) / (ia.sum() + ib.sum())
        d = rec['d_logit'][i]
        pr = rec['pred'][i]
        mid = float(((d > 0.1) & (d < 0.9)).mean())
        third = float(np.mean((pr != a) & (pr != b)))
        rows.append({'a': a, 'b': b, 'auroc': float(auc),
                     'confusion_rate': float(conf_rate), 'mid_frac': mid,
                     'third_class_frac': third,
                     'n_a': int(ia.sum()), 'n_b': int(ib.sum())})
        auc_mat[a, b] = auc_mat[b, a] = auc

    rows_sorted = sorted(rows, key=lambda r: r['auroc'])
    r35 = next(r for r in rows if (r['a'], r['b']) == (3, 5))
    rank35 = rows_sorted.index(r35) + 1
    aucs = np.array([r['auroc'] for r in rows])
    mids = np.array([r['mid_frac'] for r in rows])
    rho_mid = spearman(aucs, mids)
    rho_third = spearman(aucs, np.array([r['third_class_frac'] for r in rows]))

    out = {'step': STEP, 'seed': 0, 'loss': 'mse', 'data': ('60k' if DIR60K else '1k'),
           'pairs': rows,
           'rank_3v5_from_worst': rank35, 'spearman_auc_vs_midfrac': float(rho_mid),
           'spearman_auc_vs_thirdfrac': float(rho_third)}

    # CE model, if trained
    ce_path = (os.path.join(FM, f'seed_0_ce{sched_sfx}', 'ckpts', f'step{STEP}.pt')
               if DIR60K else
               os.path.join(HERE, 'results', 'ckpts_movie',
                            f'seed0_ce{sched_sfx}', f'step{STEP}.pt'))
    if os.path.exists(ce_path):
        Lce = logits_of(load_state_model(ce_path, device), test_x, device)
        ce_rows = {}
        for p in man['pairs'][:45]:
            a, b = p['class_a'], p['class_b']
            s = Lce[:, a] - Lce[:, b]
            ce_rows[f'{a}v{b}'] = float(auroc(s[test_y == a], s[test_y == b]))
        out['ce_auroc'] = ce_rows
        print(f"CE 3v5 AUROC: {ce_rows['3v5']:.4f} "
              f"(worst CE pair: {min(ce_rows, key=ce_rows.get)} "
              f"{min(ce_rows.values()):.4f})")

    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', f'pairwise_auc{out_sfx}.json'), 'w') as f:
        json.dump(out, f, indent=1)

    print(f"3v5: AUROC={r35['auroc']:.4f}, confusion={r35['confusion_rate']:.4f}, "
          f"mid_frac={r35['mid_frac']:.2f}, rank {rank35}/45 from worst")
    print("5 worst pairs:", [(f"{r['a']}v{r['b']}", round(r['auroc'], 4))
                             for r in rows_sorted[:5]])
    print(f"Spearman(AUROC, mid_frac) = {rho_mid:.3f}; "
          f"Spearman(AUROC, third_frac) = {rho_third:.3f}")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(auc_mat, cmap='viridis', vmin=np.nanmin(auc_mat), vmax=1.0)
    for a in range(10):
        for b in range(10):
            if a != b:
                ax.text(b, a, f'{auc_mat[a, b]:.3f}'.lstrip('0'), fontsize=5.5,
                        ha='center', va='center',
                        color='white' if auc_mat[a, b] < 0.9985 else 'black')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('digit $b$'); ax.set_ylabel('digit $a$')
    ax.set_title(f'Pairwise AUROC (logit difference), step {STEP:,}')
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    names = [f"{r['a']}v{r['b']}" for r in rows_sorted]
    vals = [1 - r['auroc'] for r in rows_sorted]
    cols = ['C3' if n == '3v5' else 'C0' for n in names]
    ax.bar(range(45), vals, color=cols)
    ax.set_xticks(range(45)); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yscale('log')
    ax.set_ylabel('1 − AUROC (log scale)')
    ax.set_title(f'Pair difficulty ranking (3v5 in red: rank {rank35}/45 from worst)')
    ax.grid(alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(mids, 1 - aucs, s=25, color='C0')
    ax.scatter([r35['mid_frac']], [1 - r35['auroc']], s=90, color='C3',
               zorder=5, label='3v5')
    for r in rows:
        if 1 - r['auroc'] > 0.004 or r['mid_frac'] > 0.62 or (r['a'], r['b']) == (3, 5):
            ax.annotate(f"{r['a']}v{r['b']}", (r['mid_frac'], 1 - r['auroc']),
                        fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_yscale('log')
    ax.set_xlabel('mid fraction of the interpolation curve '
                  r'($0.1 < d(\alpha) < 0.9$), final step')
    ax.set_ylabel('1 − AUROC (log scale)')
    ax.set_title(f'Curve shape vs pair difficulty (Spearman ρ = {rho_mid:.2f})')
    ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 1, N_POINTS)
    i35 = next(i for i, p in enumerate(man['pairs'][:45])
               if (p['class_a'], p['class_b']) == (3, 5))
    d, pr = rec['d_logit'][i35], rec['pred'][i35]
    ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
    ax.plot(t, d, lw=2, color='C0')
    ax.scatter(t, np.full(N_POINTS, -0.07), s=14, c=pr, cmap='tab10',
               vmin=0, vmax=9, marker='s')
    ax.set_ylim(-0.13, 1.05); ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$d(\alpha)$')
    # data-driven segment labels: contiguous runs of one predicted class
    segs, s0 = [], 0
    for k in range(1, N_POINTS + 1):
        if k == N_POINTS or pr[k] != pr[s0]:
            segs.append((s0, k - 1, int(pr[s0]))); s0 = k
    n_seg = len(segs)
    for s0, s1, cls in segs:
        if s1 - s0 < 4:                       # skip tiny transition segments
            continue
        xm = 0.5 * (s0 + s1) / (N_POINTS - 1)
        ax.annotate(f'predicted {cls}', (xm, float(d[(s0 + s1) // 2]) + 0.07),
                    fontsize=8, ha='center')
    ep_note = ('endpoint "3" misclassified as %d' % pr[0]) if pr[0] != 3 else \
              'both endpoints correctly classified'
    ax.set_title(f'Pair 3→5 at step {STEP:,}: {n_seg} predicted-class segments\n'
                 f'({ep_note})', fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle('Is 3-vs-5 harder? Pairwise discriminability vs interpolation-curve '
                 'shape (MSE seed 0'
                 + (', full 60k images' if DIR60K else '')
                 + (', ReduceLROnPlateau' if sched_sfx else '')
                 + f', step {STEP:,}, first 2,000 test images)', y=0.98)
    plt.savefig(os.path.join(PLOTS, f'pairwise_auc{out_sfx}.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)
    print(f'saved plots/pairwise_auc{out_sfx}.png')


if __name__ == '__main__':
    main()
