#!/usr/bin/env python3
"""dir12 — figures for plateau emergence over training. Reads results/*.json."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = '/workspace/marsv_agent_haoyang/dir12_plateau_during_training'
PLOT = os.path.join(HERE, 'plots'); os.makedirs(PLOT, exist_ok=True)

SEEDS = [s for s in [0, 1, 2]
         if os.path.exists(os.path.join(HERE, 'results', f'sweep_seed{s}.json'))]
sweeps = {s: json.load(open(os.path.join(HERE, 'results', f'sweep_seed{s}.json'))) for s in SEEDS}
hist = {s: json.load(open(os.path.join(HERE, 'results', 'ckpts', f'seed{s}', 'history.json')))
        for s in SEEDS}
curves = json.load(open(os.path.join(HERE, 'results', 'curves_seed0.json')))
S0 = sweeps[0]
rhos = np.array(S0['rhos'])


def xlog(step):
    return max(step, 0.5)


# ---------- Fig 1: training dynamics ----------
fig, ax = plt.subplots(figsize=(8, 5))
h = hist[0]
st = [xlog(s) for s in h['step']]
ax.plot(st, h['train_acc'], 'C0-', label='train acc')
ax.plot(st, h['test_acc'], 'C1-', label='test acc')
ax.plot(st, h['train_conf'], 'C2:', alpha=0.6, label='train softmax conf')
ax.set_xscale('log'); ax.set_xlabel('optimization step'); ax.set_ylabel('accuracy / confidence')
ax.grid(alpha=0.3)
ax2 = ax.twinx()
steps = [c['step'] for c in S0['checkpoints']]
ax2.plot([xlog(s) for s in steps], [c['mean_conf'] for c in S0['checkpoints']],
         'C3-o', ms=4, label='mean max-output (eval)')
ax2.set_ylabel('mean max raw output (confidence)', color='C3')
ax2.tick_params(axis='y', labelcolor='C3')
ax.axvline(300, color='gray', ls='--', lw=1)
ax.text(320, 0.2, 'test acc peaks', rotation=90, fontsize=8, color='gray')
lines1, l1 = ax.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, l1 + l2, loc='center right', fontsize=8)
ax.set_title('MNIST MLP (d4/w200/ReLU, n=1000) training dynamics — seed 0')
plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'training_dynamics.png'), dpi=140); plt.close()

# ---------- Fig 2: response curves early/mid/late ----------
stages = [(100, 'early (step 100)'), (10000, 'mid (step 10k)'), (100000, 'late (step 100k)')]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
for ax, (step, title) in zip(axes, stages):
    c = curves[str(step)]
    ax.plot(rhos, c['R_data_median'], 'C0-o', ms=3, label='all natural')
    if c['R_conf_correct']:
        ax.plot(rhos, c['R_conf_correct'], 'C2-', label='confident-correct')
    if c['R_conf_wrong']:
        ax.plot(rhos, c['R_conf_wrong'], 'C1-', label='confident-wrong')
    if c['R_unc']:
        ax.plot(rhos, c['R_unc'], 'C4-', label='uncertain')
    ax.plot(rhos, c['R_rand_median'], 'C3--', label='matched-random control')
    ax.axvspan(0, S0['rho_small'], color='gray', alpha=0.1)
    ax.set_title(title); ax.set_xlabel(r'relative radius $\rho$'); ax.grid(alpha=0.3)
axes[0].set_ylabel(r'normalized downstream response $R(\rho)$')
axes[0].legend(fontsize=8)
fig.suptitle('Response curves by stage: natural activations stay flatter than matched-random near 0\n'
             r'(shaded = small-radius interval $\rho\in[0,0.2]$ used for plateau contrast)', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'plateau_curves_by_stage.png'), dpi=140); plt.close()

# ---------- Fig 3: contrast + region count vs step (multi-seed) ----------
fig, (axc, axr) = plt.subplots(1, 2, figsize=(14, 4.8))
# contrast: mean +- across seeds if available, plus per-seed CI band for seed0
allsteps = S0['steps']
cmat = []
for s in SEEDS:
    cmat.append([c['contrast'] for c in sweeps[s]['checkpoints']])
cmat = np.array(cmat)
xs = [xlog(s) for s in allsteps]
if len(SEEDS) > 1:
    axc.plot(xs, cmat.mean(0), 'C0-o', ms=4, label=f'mean of {len(SEEDS)} seeds')
    axc.fill_between(xs, cmat.min(0), cmat.max(0), color='C0', alpha=0.2, label='seed min-max')
else:
    lo = [c['contrast_ci'][0] for c in S0['checkpoints']]
    hi = [c['contrast_ci'][1] for c in S0['checkpoints']]
    axc.plot(xs, cmat[0], 'C0-o', ms=4, label='seed 0')
    axc.fill_between(xs, lo, hi, color='C0', alpha=0.2, label='95% bootstrap CI')
axc.axhline(0, color='k', lw=0.7)
axc.set_xscale('log'); axc.set_xlabel('optimization step'); axc.set_ylabel('plateau contrast')
axc.set_title('Plateau contrast vs training step')
axc.grid(alpha=0.3); axc.legend(fontsize=8)
# region count
for s in SEEDS:
    vc = [c['regions']['cosine']['n_valid'] for c in sweeps[s]['checkpoints']]
    axr.plot([xlog(x) for x in sweeps[s]['steps']], vc, '-o', ms=4, label=f'seed {s} (cosine)')
axr.axhline(10, color='gray', ls=':', label='10 (one per digit)')
axr.set_xscale('log'); axr.set_xlabel('optimization step')
axr.set_ylabel('validated stable-region count')
axr.set_title('Validated stable-region count vs training step')
axr.set_ylim(-0.5, 12.5); axr.grid(alpha=0.3); axr.legend(fontsize=8)
fig.suptitle('Plateau strength keeps rising after test accuracy saturates; region count converges to ~10',
             fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'plateau_contrast_and_region_count.png'), dpi=140)
plt.close()

# ---------- Fig 4: contrast by confidence/correctness group ----------
fig, ax = plt.subplots(figsize=(8.5, 5))
groups = [('conf_correct', 'C2', 'confident-correct'),
          ('conf_wrong', 'C1', 'confident-wrong'),
          ('unc_correct', 'C4', 'uncertain-correct'),
          ('unc_wrong', 'C5', 'uncertain-wrong')]
for gk, col, lab in groups:
    ys, xsg = [], []
    for c in S0['checkpoints']:
        v = c['groups'][gk]['contrast']
        if v is not None and c['groups'][gk]['n'] >= 10:
            ys.append(v); xsg.append(xlog(c['step']))
    if ys:
        ax.plot(xsg, ys, '-o', ms=4, color=col, label=lab)
ax.axhline(0, color='k', lw=0.7)
ax.set_xscale('log'); ax.set_xlabel('optimization step'); ax.set_ylabel('plateau contrast')
ax.set_title('Plateau contrast by confidence x correctness (seed 0)\n'
             'confident-wrong plateaus like confident-correct; uncertain is weakest')
ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'contrast_by_group.png'), dpi=140); plt.close()

print("saved 4 figures to plots/ ; seeds used:", SEEDS)
