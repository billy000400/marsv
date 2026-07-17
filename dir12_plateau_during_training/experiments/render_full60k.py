#!/usr/bin/env python3
"""
dir12 S9 — Render the full-60k vs 1,000-example comparison from saved records.

Outputs (plots/):
  full_vs_1k_evolution.gif      synchronized side-by-side animation, 10 pairs,
                                frames aligned by optimizer step (25 common)
  full_vs_1k_frames.png         static early/mid/late rows of the same view
  full_mnist_3v5_training.gif   50-path 3->5 bank through full-60k training
  full_mnist_3v5_summary.png    original 3->5 path + 50-path distribution +
                                segment/detour/endpoint evolution, both runs
  full_mnist_training_context.png  acc/conf/loss, 60k seeds 0/1/2 vs 1k ref
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
import numpy as np

from plateau_protocol import HERE, N_POINTS, ANIM_PAIRS

PLOTS = os.path.join(HERE, 'plots')
FM = os.path.join(HERE, 'results', 'full_mnist_from_scratch')
REF_SFX = '_pl_f0.5_p100'          # converged primary 1k run
# suffix of the 60k runs to compare (feedback human_feedback_1: the smooth
# ReduceLROnPlateau runs are now the primary 60k runs)
SFX60 = sys.argv[1] if len(sys.argv) > 1 else ''
t = np.linspace(0, 1, N_POINTS)


def load_dir(rec_dir):
    man = json.load(open(os.path.join(rec_dir, 'manifest.json')))
    steps = man['ckpt_steps']
    out = {k: [] for k in ['d_logit', 'pred', 'seg_count']}
    for s in steps:
        z = np.load(os.path.join(rec_dir, f'step_{s}.npz'))
        for k in out:
            out[k].append(z[k] if k in z.files else None)
    return man, np.array(steps), {k: np.stack(v) if v[0] is not None else None
                                  for k, v in out.items()}


def pair_index(man, a, b, start=0, stop=None):
    for i, p in enumerate(man['pairs'][start:stop], start):
        if p['class_a'] == min(a, b) and p['class_b'] == max(a, b):
            return i
    raise KeyError((a, b))


def detour(pred_rows):
    """[N, T] -> [N] bool: any interior class differing from BOTH endpoint
    predictions (endpoint-misclassification segments are not detours)."""
    ends = np.stack([pred_rows[:, 0], pred_rows[:, -1]], 1)
    return np.array([len(set(row.tolist()) - set(e.tolist())) > 0
                     for row, e in zip(pred_rows, ends)])


# ---------------- load everything ----------------
man60, steps60, r60 = load_dir(os.path.join(FM, f'seed_0{SFX60}'))
man1k, steps1k, r1k = load_dir(os.path.join(HERE, 'results', 'plateau_records',
                                            f'seed_0{REF_SFX}'))
manR, stepsR, rR = load_dir(os.path.join(FM, 'ref1k_seed_0'))   # 105 pairs, anchors
hist60 = json.load(open(os.path.join(FM, f'seed_0{SFX60}', 'history.json')))
hist1k = json.load(open(os.path.join(HERE, 'results', 'ckpts_movie',
                                     f'seed0{REF_SFX}', 'history.json')))
D60, P60 = r60['d_logit'], r60['pred']
D1k, P1k = r1k['d_logit'], r1k['pred']
anim60 = [pair_index(man60, a, b, stop=55) for a, b in ANIM_PAIRS]
anim1k = [pair_index(man1k, a, b) for a, b in ANIM_PAIRS]
i35_60 = pair_index(man60, 3, 5, stop=55)
i35_1k = pair_index(man1k, 3, 5)
rows35 = np.arange(55, 105)                       # frozen 50-path 3v5 bank

# ---------------- 1. synchronized side-by-side animation ----------------
common = [0, 10, 30, 100, 300] + list(range(1500, 30_001, 1500))
f60 = [int(np.where(steps60 == s)[0][0]) for s in common]
f1k = [int(np.where(steps1k == s)[0][0]) for s in common]

fig = plt.figure(figsize=(15, 5.8))
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.5, wspace=0.3)
lines, dots = [], []
for row, lab in [(0, '1k subset'), (1, 'full 60k')]:
    for k, (a, b) in enumerate(ANIM_PAIRS[:5]):
        pass
axes = []
for row in range(2):
    for k in range(5):
        a, b = ANIM_PAIRS[k + 5 * 0] if False else ANIM_PAIRS[k]
        ax = fig.add_subplot(gs[row, k])
        ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
        ln, = ax.plot([], [], lw=2, color=('C0' if row == 0 else 'C2'))
        sc = ax.scatter(t, np.full(N_POINTS, -0.09), s=6, c=np.zeros(N_POINTS),
                        cmap='tab10', vmin=0, vmax=9, marker='s')
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.15, 1.05)
        ax.set_title(f'{a} → {b}', fontsize=9)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
        if k == 0:
            ax.set_ylabel(('1k subset\n' if row == 0 else 'full 60k\n')
                          + r'$d(\alpha)$', fontsize=9)
        if row == 1:
            ax.set_xlabel(r'$\alpha$', fontsize=9)
        axes.append(ax); lines.append(ln); dots.append(sc)
ax_acc = fig.add_subplot(gs[0, 5])
ax_acc.plot(hist1k['step'], hist1k['test_acc'], color='C0', lw=1.5, label='1k test acc')
ax_acc.plot(hist60['step'], hist60['test_acc'], color='C2', lw=1.5, label='60k test acc')
ax_acc.set_xscale('symlog', linthresh=10); ax_acc.set_xlim(0, 30000)
ax_acc.set_ylim(0, 1.05); ax_acc.legend(fontsize=6, loc='lower right')
ax_acc.tick_params(labelsize=7); ax_acc.grid(alpha=0.25)
vline = ax_acc.axvline(0, color='k', lw=1)
ax_ls = fig.add_subplot(gs[1, 5])
ax_ls.plot(hist1k['step'], hist1k['train_loss'], color='C0', lw=1.5, label='1k train loss')
ax_ls.plot(hist60['step'], hist60['train_loss'], color='C2', lw=1.5, label='60k train loss')
ax_ls.set_xscale('symlog', linthresh=10); ax_ls.set_xlim(0, 30000)
ax_ls.set_yscale('log'); ax_ls.set_xlabel('step', fontsize=8)
ax_ls.legend(fontsize=6, loc='lower left'); ax_ls.tick_params(labelsize=7)
ax_ls.grid(alpha=0.25)
vline_l = ax_ls.axvline(0, color='k', lw=1)
title = fig.suptitle('', fontsize=12)

# note: 5 of the 10 ANIM pairs per row (2 rows x 5 pairs x 2 runs won't fit) —
# instead show the SAME 5 pairs for both runs; second GIF covers the rest? No:
# use the first 5 ANIM pairs; the 3->5 GIF + static frames cover the full set.
SHOW = ANIM_PAIRS[:5]


def draw(fi):
    for k in range(5):
        lines[k].set_data(t, D1k[f1k[fi], anim1k[k]])
        dots[k].set_array(P1k[f1k[fi], anim1k[k]].astype(float))
        lines[5 + k].set_data(t, D60[f60[fi], anim60[k]])
        dots[5 + k].set_array(P60[f60[fi], anim60[k]].astype(float))
    vline.set_xdata([max(common[fi], 1e-3)] * 2)
    vline_l.set_xdata([max(common[fi], 1e-3)] * 2)
    title.set_text(f'1k-subset vs full-60k training, aligned at step '
                   f'{common[fi]:,} — logit $d(\\alpha)$, squares: predicted class')
    return lines


ani = animation.FuncAnimation(fig, draw, frames=len(common), blit=False)
ani.save(os.path.join(PLOTS, 'full_vs_1k_evolution.gif'), fps=4,
         writer='pillow', dpi=72)
plt.close(fig)
print('saved full_vs_1k_evolution.gif')

# ---------------- 2. static aligned frames, all 10 pairs ----------------
sel_steps = [0, 300, 3000, 30000]
fig, axg = plt.subplots(len(sel_steps) * 2, 10,
                        figsize=(16, 1.55 * 2 * len(sel_steps)),
                        sharex=True, sharey=True)
for r, s in enumerate(sel_steps):
    j1 = int(np.where(steps1k == s)[0][0])
    j6 = int(np.where(steps60 == s)[0][0])
    for k in range(10):
        for rr, (D_, P_, idx, col, lab) in enumerate(
                [(D1k, P1k, anim1k, 'C0', '1k'), (D60, P60, anim60, 'C2', '60k')]):
            ax = axg[2 * r + rr, k]
            ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.7)
            ax.plot(t, D_[j1 if rr == 0 else j6, idx[k]], lw=1.5, color=col)
            ax.scatter(t, np.full(N_POINTS, -0.09), s=4,
                       c=P_[j1 if rr == 0 else j6, idx[k]],
                       cmap='tab10', vmin=0, vmax=9, marker='s')
            ax.set_ylim(-0.15, 1.05); ax.grid(alpha=0.2); ax.tick_params(labelsize=6)
            if 2 * r + rr == 0:
                a, b = ANIM_PAIRS[k]; ax.set_title(f'{a} → {b}', fontsize=9)
            if k == 0:
                ax.set_ylabel(f'step {s:,}\n{lab}', fontsize=8)
for k in range(10):
    axg[-1, k].set_xlabel(r'$\alpha$', fontsize=8)
fig.suptitle('Aligned frames: 1k-subset (blue) vs full-60k (green) logit '
             '$d(\\alpha)$; squares = predicted class along the path', y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'full_vs_1k_frames.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)
print('saved full_vs_1k_frames.png')

# ---------------- 3. the 50-path 3->5 bank through training (GIF) ----------------
fig = plt.figure(figsize=(11, 4.4))
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1], wspace=0.28)
axL = fig.add_subplot(gs[0, 0])
axL.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
spag = [axL.plot([], [], lw=0.8, color='C2', alpha=0.35)[0] for _ in rows35]
orig_ln, = axL.plot([], [], lw=2.5, color='C3', label='original 3→5 pair')
axL.set_xlim(-0.02, 1.02); axL.set_ylim(-0.02, 1.05)
axL.set_xlabel(r'$\alpha$'); axL.set_ylabel(r'$d(\alpha)$ (logits)')
axL.legend(fontsize=8, loc='upper left'); axL.grid(alpha=0.25)
axR = fig.add_subplot(gs[0, 1])
seg60 = r60['seg_count'][:, rows35].astype(float)         # [F, 50]
det60 = np.stack([detour(P60[f, rows35]) for f in range(len(steps60))])
end_ok60 = (P60[:, rows35, 0] == 3) & (P60[:, rows35, -1] == 5)
axR.plot(steps60, seg60.mean(1), color='C2', lw=1.8, label='mean segment count')
axR.plot(steps60, 1 + 4 * det60.mean(1), color='C1', lw=1.5,
         label='detour fraction (scaled 1+4f)')
axR.plot(steps60, 1 + 4 * end_ok60.mean(1), color='C4', lw=1.5, ls='--',
         label='endpoints-correct fraction (scaled)')
axR.set_xscale('symlog', linthresh=10); axR.set_ylim(0.8, 5.2)
axR.set_xlabel('training step'); axR.grid(alpha=0.25); axR.legend(fontsize=7)
vln = axR.axvline(0, color='k', lw=1)
ttl = fig.suptitle('', fontsize=12)


def draw35(f):
    for j, ln in enumerate(spag):
        ln.set_data(t, D60[f, rows35[j]])
    orig_ln.set_data(t, D60[f, i35_60])
    vln.set_xdata([max(steps60[f], 1e-3)] * 2)
    ttl.set_text(f'Full-60k run, 50 frozen 3→5 paths — step {steps60[f]:,}')
    return spag


ani = animation.FuncAnimation(fig, draw35, frames=len(steps60), blit=False)
ani.save(os.path.join(PLOTS, 'full_mnist_3v5_training.gif'), fps=10,
         writer='pillow', dpi=72)
plt.close(fig)
print('saved full_mnist_3v5_training.gif')

# ---------------- 4. static 3->5 summary ----------------
# ref1k metrics on the same 50 paths at anchor steps (<= 30000 for x-range)
segR = rR['seg_count'][:, rows35].astype(float)
PR = rR['pred']
detR = np.stack([detour(PR[f, rows35]) for f in range(len(stepsR))])
end_okR = (PR[:, rows35, 0] == 3) & (PR[:, rows35, -1] == 5)
ok0_60 = end_ok60[0]                       # endpoint-correct at step 0 (60k)
ok0_R = end_okR[0]

fig, axs = plt.subplots(2, 2, figsize=(12.5, 8))
# (a) original 3->5 path, final checkpoints
ax = axs[0, 0]
jR = int(np.where(stepsR == 30000)[0][0])
ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
ax.plot(t, rR['d_logit'][jR, i35_60], lw=2.2, color='C0', label='1k subset (step 30k)')
ax.plot(t, D60[-1, i35_60], lw=2.2, color='C2', label='full 60k (step 30k)')
ax.scatter(t, np.full(N_POINTS, -0.07), s=14, c=PR[jR, i35_60], cmap='tab10',
           vmin=0, vmax=9, marker='s')
ax.scatter(t, np.full(N_POINTS, -0.13), s=14, c=P60[-1, i35_60], cmap='tab10',
           vmin=0, vmax=9, marker='s')
ax.text(0.01, -0.055, '1k pred', fontsize=7); ax.text(0.01, -0.125, '60k pred', fontsize=7)
ax.set_ylim(-0.18, 1.05); ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$d(\alpha)$ (logits)')
ax.set_title('Original 3→5 pair at matched step 30,000'); ax.legend(fontsize=8)
ax.grid(alpha=0.25)
# (b) 50-path spaghetti at final step, both runs
ax = axs[0, 1]
ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=0.8)
for j in rows35:
    ax.plot(t, rR['d_logit'][jR, j], lw=0.7, color='C0', alpha=0.3)
    ax.plot(t, D60[-1, j], lw=0.7, color='C2', alpha=0.3)
ax.plot([], [], color='C0', lw=1.5, label='1k subset (50 paths)')
ax.plot([], [], color='C2', lw=1.5, label='full 60k (50 paths)')
ax.set_xlabel(r'$\alpha$'); ax.set_title('All 50 frozen 3→5 paths at step 30,000')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
# (c) segment count evolution
ax = axs[1, 0]
q1, q3 = np.percentile(seg60, [25, 75], axis=1)
ax.fill_between(steps60, q1, q3, color='C2', alpha=0.2)
ax.plot(steps60, seg60.mean(1), color='C2', lw=2, label='full 60k (mean, IQR)')
ax.plot(stepsR, segR.mean(1), color='C0', lw=2, marker='o', ms=4,
        label='1k subset (mean, anchors)')
ax.plot(steps60, seg60[:, ok0_60].mean(1) if ok0_60.any() else
        np.full(len(steps60), np.nan), color='C2', ls='--', lw=1.2,
        label='60k, endpoints-correct-at-step-0 subset')
ax.set_xscale('symlog', linthresh=10); ax.set_xlim(0, 30000)
ax.set_xlabel('training step'); ax.set_ylabel('predicted-class segments per path')
ax.set_title('Run-length segment count, 50 3→5 paths'); ax.legend(fontsize=8)
ax.grid(alpha=0.25)
# (d) detour + endpoint correctness evolution
ax = axs[1, 1]
ax.plot(steps60, det60.mean(1), color='C2', lw=2, label='60k: third-class detour frac')
ax.plot(stepsR, detR.mean(1), color='C0', lw=2, marker='o', ms=4,
        label='1k: third-class detour frac')
ax.plot(steps60, end_ok60.mean(1), color='C2', lw=1.5, ls='--',
        label='60k: endpoints correct frac')
ax.plot(stepsR, end_okR.mean(1), color='C0', lw=1.5, ls='--',
        label='1k: endpoints correct frac')
ax.set_xscale('symlog', linthresh=10); ax.set_xlim(0, 30000)
ax.set_ylim(-0.03, 1.05); ax.set_xlabel('training step'); ax.set_ylabel('fraction of 50 paths')
ax.set_title('Detour presence and endpoint correctness'); ax.legend(fontsize=7)
ax.grid(alpha=0.25)
fig.suptitle('Frozen 50-path 3→5 bank: full-60k from-scratch run vs the '
             '1,000-example reference', y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'full_mnist_3v5_summary.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)
print('saved full_mnist_3v5_summary.png')

# ---------------- 5. training context: 60k seeds 0/1/2 vs 1k ----------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.8))
ax = axs[0]
for s, col in [(0, 'C2'), (1, 'C8'), (2, 'C9')]:
    h = json.load(open(os.path.join(FM, f'seed_{s}{SFX60}', 'history.json')))
    ax.plot(h['step'], h['test_acc'], lw=1.5, color=col, label=f'60k seed {s} test acc')
ax.plot(hist1k['step'], hist1k['test_acc'], lw=1.5, color='C0', label='1k seed 0 test acc')
ax.plot(hist60['step'], hist60['test_conf'], lw=1.2, color='C4', ls='--',
        label='60k seed 0 test conf (max raw)')
ax.set_xscale('symlog', linthresh=10); ax.set_xlim(0, 30000); ax.set_ylim(0, 1.05)
ax.set_xlabel('training step'); ax.set_ylabel('accuracy / confidence')
ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.set_title('Accuracy and confidence')
ax = axs[1]
ax.plot(hist60['step'], hist60['train_loss'], lw=1.5, color='C2', label='60k train loss')
ax.plot(hist60['step'], hist60['test_loss'], lw=1.5, color='C1', label='60k test loss')
ax.plot(hist1k['step'], hist1k['train_loss'], lw=1.5, color='C0', label='1k train loss')
ax.set_xscale('symlog', linthresh=10); ax.set_xlim(0, 30000); ax.set_yscale('log')
ax.set_xlabel('training step'); ax.set_ylabel('MSE loss'); ax.grid(alpha=0.3)
ax.legend(fontsize=7); ax.set_title('Loss (log scale)')
fig.suptitle('Full-60k from-scratch training context (seed 0 primary, seeds 1–2 '
             'confirmation) vs the 1k reference', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'full_mnist_training_context.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)
print('saved full_mnist_training_context.png')

# ---------------- console summary for the report ----------------
def rle(row):
    out = [int(row[0])]
    for v in row[1:]:
        if int(v) != out[-1]:
            out.append(int(v))
    return out

print('\n=== summary numbers ===')
print(f'endpoints-correct at step 0: 60k {int(end_ok60[0].sum())}/50, '
      f'1k-ref {int(end_okR[0].sum())}/50')
print(f'final (step 30k): 60k seg mean {seg60[-1].mean():.2f}, detour frac '
      f'{det60[-1].mean():.2f}, endpoints-correct {int(end_ok60[-1].sum())}/50')
jR30 = int(np.where(stepsR == 30000)[0][0])
print(f'1k-ref at 30k:   seg mean {segR[jR30].mean():.2f}, detour frac '
      f'{detR[jR30].mean():.2f}, endpoints-correct {int(end_okR[jR30].sum())}/50')
jR100 = int(np.where(stepsR == 100000)[0][0])
print(f'1k-ref at 100k:  seg mean {segR[jR100].mean():.2f}, detour frac '
      f'{detR[jR100].mean():.2f}, endpoints-correct {int(end_okR[jR100].sum())}/50')
print(f'original 3->5 RLE: 1k@30k {rle(PR[jR30, i35_60])}, '
      f'1k@100k {rle(PR[jR100, i35_60])}, 60k@30k {rle(P60[-1, i35_60])}')
# seeds 1/2 replication at final step
for s in (1, 2):
    m, st, r = load_dir(os.path.join(FM, f'seed_{s}{SFX60}'))
    p = r['pred']; sc = r['seg_count'][:, rows35].astype(float)
    dt = detour(p[-1, rows35]); eo = (p[-1, rows35, 0] == 3) & (p[-1, rows35, -1] == 5)
    print(f'60k seed {s} final: seg mean {sc[-1].mean():.2f}, detour frac '
          f'{dt.mean():.2f}, endpoints-correct {int(eo.sum())}/50, '
          f'orig 3->5 RLE {rle(p[-1, pair_index(m, 3, 5, stop=55)])}')
