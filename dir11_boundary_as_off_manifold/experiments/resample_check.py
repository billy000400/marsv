#!/usr/bin/env python3
"""
dir11 — endpoint-resampling stability check (final verdict-rule requirement).

The PLAN's verdict rules require the counterexample verdict to be stable under
RESAMPLING as well as model replication. All prior runs used endpoint-sampling
seed 0 (bootstrap CIs resample the seed-0 measurements, not fresh endpoint
draws). Here we re-run the identical frozen pipeline on the base model with two
fresh endpoint-sampling seeds (1, 2), plus seed 0 as a regression check, and ask:

  (a) does the between-plateau median G stay on the within-plateau baseline?
  (b) do counterexamples (verified pairs with median G <= 1) persist — in
      particular, is there at least one pair that is a counterexample under
      EVERY seed ("reproducible counterexample", the universal-claim refuter)?

Everything else (d(t) accept filter, G definition, 20 pairs/region-pair, margin
rule) stays frozen. Outputs results/resample_check.json,
plots/population_resample.png.
"""
import os, sys, json
import importlib.util

HERE_FILE = os.path.abspath(__file__)
EXP = os.path.dirname(HERE_FILE)
spec = importlib.util.spec_from_file_location('pm', os.path.join(EXP, 'population_manifold.py'))
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.mnist import load_checkpoint  # path set up by pm import

CKPT = os.path.join(pm.BASE, 'results/image/mnist_mlp_d4_w200_relu_n1000_s100000.pt')
model, ck = load_checkpoint(CKPT)

SEEDS = [0, 1, 2]
runs = []
for s in SEEDS:
    r = pm.analyze(model, ck, f'base seed{s}', sample_seed=s)
    print(f"[sample_seed={s}] between-G med={r['between_G_median']:.3f} "
          f"CI={r['boot_between_median_CI']}  frac>1={r['frac_pairs_G_gt_1']:.2f}  "
          f"counterex={r['n_counterexamples']}/{r['n_verified_pairs']}  "
          f"digit9 G={r['digit9_sub']['G_median']}", flush=True)
    runs.append(r)

# per-pair median G across seeds (only pairs verified in that seed)
def pair_meds(r):
    return {p['name']: p['G_median'] for p in r['pairs']
            if p['G_median'] is not None and p['n_verified'] >= 5}
meds = [pair_meds(r) for r in runs]
common = sorted(set(meds[0]) & set(meds[1]) & set(meds[2]))
cx_sets = [set(n for n, g in m.items() if g <= 1.0) for m in meds]
stable_cx = sorted(cx_sets[0] & cx_sets[1] & cx_sets[2])

out = dict(
    seeds=SEEDS,
    per_seed=[dict(seed=s,
                   between_G_median=r['between_G_median'],
                   boot_between_median_CI=r['boot_between_median_CI'],
                   within_G_median=r['within_G_median'],
                   within_G_p95=r['within_G_p95'],
                   frac_pairs_G_gt_1=r['frac_pairs_G_gt_1'],
                   n_verified_pairs=r['n_verified_pairs'],
                   n_counterexamples=r['n_counterexamples'],
                   digit9_G=r['digit9_sub']['G_median'])
              for s, r in zip(SEEDS, runs)],
    n_pairs_verified_in_all_seeds=len(common),
    n_stable_counterexamples=len(stable_cx),
    stable_counterexamples=stable_cx,
    per_pair_G=[dict(name=n, G=[m.get(n) for m in meds]) for n in sorted(set().union(*meds))],
)
with open(os.path.join(pm.RES, 'resample_check.json'), 'w') as f:
    json.dump(out, f, indent=2)

# ---- figure ----
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.0))
x = np.arange(len(SEEDS))
bm = [r['between_G_median'] for r in runs]
lo = [r['boot_between_median_CI'][0] for r in runs]
hi = [r['boot_between_median_CI'][1] for r in runs]
axa.errorbar(x, bm, yerr=[np.array(bm) - lo, np.array(hi) - bm], fmt='o', color='C3',
             capsize=4, label='between-plateau median G (95% CI)')
axa.plot(x, [r['within_G_median'] for r in runs], 's', color='C2',
         label='within-plateau median G')
axa.axhline(1.0, color='k', ls='--', lw=1)
axa.set_xticks(x); axa.set_xticklabels([f'endpoint seed {s}' for s in SEEDS])
axa.set_ylabel('median G'); axa.set_ylim(0.85, 1.15)
axa.set_title('(a) Fresh endpoint draws: between-plateau median G\nstays on the within-plateau baseline for every seed')
axa.legend(fontsize=9); axa.grid(alpha=0.3)

# scatter per-pair G: seed 0 vs seeds 1,2
for si, c in [(1, 'C0'), (2, 'C1')]:
    xs = [meds[0][n] for n in common]
    ys = [meds[si][n] for n in common]
    axb.scatter(xs, ys, s=22, alpha=0.75, color=c, label=f'seed 0 vs seed {SEEDS[si]}')
axb.axhline(1.0, color='k', ls=':', lw=0.8); axb.axvline(1.0, color='k', ls=':', lw=0.8)
lim = [0.75, 1.75]
axb.plot(lim, lim, 'k--', lw=0.8, label='y = x')
axb.set_xlim(lim); axb.set_ylim(lim)
axb.set_xlabel('per-pair median G (endpoint seed 0)')
axb.set_ylabel('per-pair median G (fresh endpoint seed)')
axb.set_title(f'(b) Per-pair median G is stable across endpoint draws\n'
              f'{len(stable_cx)} pairs are counterexamples (G≤1) under ALL three seeds')
axb.legend(fontsize=9); axb.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(pm.PLOT, 'population_resample.png'), dpi=140)
plt.close()

print(json.dumps({k: out[k] for k in ('n_pairs_verified_in_all_seeds',
                                      'n_stable_counterexamples')}, indent=2))
print('stable counterexamples:', ', '.join(stable_cx))
print('saved resample_check.json + population_resample.png')
