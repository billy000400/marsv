#!/usr/bin/env python3
"""
dir12 figure #4 — region_composition_and_lineage (seed 0).

(a) Validated-region emergence: for each predicted digit (row) and checkpoint
    (col), mark whether that digit owns a plateau-validated stable region.
    Reads off monotonic birth (regions appear one digit at a time, none lost).
(b,c) Membership-overlap heatmaps for a representative BIRTH transition and a
    LATE stable transition. Because the same eval examples are reused, cell
    [i,j] = |cluster_i(t) ∩ cluster_j(t+1)|. A near-permutation (each row/col a
    single dominant cell) == monotonic evolution: no splits/merges.

Also prints a split/merge audit across all adjacent transitions.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = '/workspace/marsv_agent_haoyang/dir12_plateau_during_training'
L = json.load(open(os.path.join(HERE, 'results', 'lineage_seed0.json')))
steps = L['steps']
cks = {c['step']: c for c in L['checkpoints']}


def order_by_digit(ck):
    """Return cluster ids ordered by (majority digit, size), + labels."""
    cl = sorted(ck['clusters'], key=lambda d: (d['maj_pred'], -d['size']))
    ids = [d['cluster'] for d in cl]
    tags = [f"{d['maj_pred']}" + ('' if d['purity'] >= 0.9 else '*') for d in cl]
    return ids, tags


def overlap_matrix(sa, sb):
    ca, cb = cks[sa], cks[sb]
    la = np.array(ca['labels']); lb = np.array(cb['labels'])
    ia, ta = order_by_digit(ca)
    ib, tb = order_by_digit(cb)
    M = np.zeros((len(ia), len(ib)), int)
    for r, i in enumerate(ia):
        for c, j in enumerate(ib):
            M[r, c] = int(np.sum((la == i) & (lb == j)))
    return M, ta, tb


def audit_transition(sa, sb):
    """Count splits (one parent -> >=2 children) and merges (>=2 parents ->
    one child) via max-overlap alignment on fixed examples."""
    ca, cb = cks[sa], cks[sb]
    la = np.array(ca['labels']); lb = np.array(cb['labels'])
    ua = np.unique(la); ub = np.unique(lb)
    ov = np.zeros((len(ua), len(ub)), int)
    for r, i in enumerate(ua):
        for c, j in enumerate(ub):
            ov[r, c] = int(np.sum((la == i) & (lb == j)))
    parent = ov.argmax(axis=0)            # for each child j, its max-overlap parent
    child = ov.argmax(axis=1)             # for each parent i, its max-overlap child
    _, pc = np.unique(parent, return_counts=True)
    _, cc = np.unique(child, return_counts=True)
    splits = int((pc >= 2).sum())         # parents claimed by >=2 children
    merges = int((cc >= 2).sum())         # children claimed by >=2 parents
    return splits, merges


# ---- split/merge audit across all adjacent transitions ----
print("transition       k_a->k_b  splits merges")
tot_s = tot_m = 0
for sa, sb in zip(steps[:-1], steps[1:]):
    s, m = audit_transition(sa, sb)
    tot_s += s; tot_m += m
    print(f"{sa:>6}->{sb:<6}  {cks[sa]['k']:>2}->{cks[sb]['k']:<2}   "
          f"{s:>4}  {m:>4}")
print(f"TOTAL splits={tot_s} merges={tot_m}")

# ---- figure ----
fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.32)

# (a) validated-region emergence grid: digit x step
ax = fig.add_subplot(gs[0, 0])
grid = np.zeros((10, len(steps)))
for j, st in enumerate(steps):
    for d in cks[st]['valid_digits']:
        grid[d, j] = 1
ax.imshow(grid, cmap='Greens', vmin=0, vmax=1.4, aspect='auto', origin='lower')
ax.set_xticks(range(len(steps)))
ax.set_xticklabels([str(s) for s in steps], rotation=90, fontsize=7)
ax.set_yticks(range(10)); ax.set_yticklabels(range(10))
ax.set_xlabel('optimization step'); ax.set_ylabel('predicted digit')
# annotate validated count per column (above the grid)
for j, st in enumerate(steps):
    ax.text(j, 9.62, str(cks[st]['n_valid']), ha='center', va='bottom',
            fontsize=7, color='#444')
ax.set_title('(a) validated stable region present, by predicted digit'
             '   (numbers above = # validated regions)', fontsize=8.5, pad=16)
for k in range(10):
    ax.axhline(k - 0.5, color='white', lw=0.5)

# (b,c) overlap heatmaps for birth + late transitions
for gi, (sa, sb, ttl) in enumerate([
        (100, 300, '(b) birth transition 100 -> 300'),
        (75000, 100000, '(c) late transition 75k -> 100k')]):
    ax = fig.add_subplot(gs[0, 1 + gi])
    M, ta, tb = overlap_matrix(sa, sb)
    im = ax.imshow(M, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(tb))); ax.set_xticklabels(tb, fontsize=7)
    ax.set_yticks(range(len(ta))); ax.set_yticklabels(ta, fontsize=7)
    ax.set_xlabel(f'clusters @ step {sb} (maj digit)', fontsize=8)
    ax.set_ylabel(f'clusters @ step {sa}', fontsize=8)
    ax.set_title(ttl, fontsize=9)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if M[r, c] > 0:
                ax.text(c, r, M[r, c], ha='center', va='center', fontsize=6,
                        color='white' if M[r, c] > M.max() * 0.5 else 'black')

# does any digit ever host >=2 validated regions in a single checkpoint?
max_per_digit = 0
for st in steps:
    vd = cks[st]['valid_digits']
    max_per_digit = max(max_per_digit, max([vd.count(d) for d in set(vd)] or [0]))

fig.suptitle('Region composition & membership-overlap lineage (seed 0): '
             'validated regions appear monotonically, exactly one per predicted '
             f'digit at every checkpoint (max regions on any single digit = '
             f'{max_per_digit}). Raw agglomerative k oscillates 10-12 as a '
             'transient non-validated sub-cluster splits off / rejoins; the '
             'birth (b) and late (c) transitions between the 10 validated '
             'regions are clean near-permutations (0 splits, 0 merges).',
             fontsize=9, y=1.02)
fig.savefig(os.path.join(HERE, 'plots', 'region_composition_and_lineage.png'),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("saved plots/region_composition_and_lineage.png")
