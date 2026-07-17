#!/usr/bin/env python3
"""
dir12 S2 — Manifest test: verify every expected per-checkpoint record exists
and contains the full frozen schema; flag missing/corrupt files.

Usage: python experiments/manifest_check.py --seed 0
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from plateau_protocol import HERE, N_POINTS

REQUIRED = ['d_h2', 'd_h3', 'd_logit', 'pred', 'max_prob', 'max_raw', 'logits',
            'end_logits', 'end_h1', 'end_h2', 'end_h3', 't', 'idx_a', 'idx_b',
            'true_a', 'true_b', 'test_acc', 'step', 'seed']
BIG = ['h1_interp', 'h2', 'h3']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--suffix', default='', help='e.g. _ce, _sched, _ce_sched')
    args = ap.parse_args()
    rec_dir = os.path.join(HERE, 'results', 'plateau_records',
                           f'seed_{args.seed}{args.suffix}')
    man = json.load(open(os.path.join(rec_dir, 'manifest.json')))
    P = len(man['pairs'])
    n_ok, problems = 0, []
    for s in man['ckpt_steps']:
        path = os.path.join(rec_dir, f'step_{s}.npz')
        if not os.path.exists(path):
            problems.append(f'step {s}: MISSING'); continue
        try:
            z = np.load(path)
            missing = [k for k in REQUIRED if k not in z.files]
            if s in man['anchor_steps']:
                missing += [k for k in BIG if k not in z.files]
            bad_shape = []
            for k in ['d_logit', 'd_h3', 'd_h2', 'pred']:
                if k in z.files and z[k].shape != (P, N_POINTS):
                    bad_shape.append(k)
            if 'logits' in z.files and z['logits'].shape != (P, N_POINTS, 10):
                bad_shape.append('logits')
            if missing or bad_shape:
                problems.append(f'step {s}: missing={missing} bad_shape={bad_shape}')
            else:
                n_ok += 1
        except Exception as e:
            problems.append(f'step {s}: CORRUPT ({e})')
        # state dict presence (suffix runs store state_dicts at anchors only)
        if args.suffix == '' or s in man['anchor_steps']:
            ck = os.path.join(HERE, 'results', 'ckpts_movie',
                              f'seed{args.seed}{args.suffix}', f'step{s}.pt')
            if not os.path.exists(ck):
                problems.append(f'step {s}: state_dict MISSING')
    print(f'seed {args.seed}: {n_ok}/{len(man["ckpt_steps"])} records OK, '
          f'{len(problems)} problems')
    for p in problems:
        print(' ', p)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
