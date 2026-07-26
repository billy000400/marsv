#!/usr/bin/env python3
"""
dir16 S2 — run the frozen activation-interpolation probe on both trained models.

Every model is probed at its BEST-VALIDATION-LOSS checkpoint (`{kind}_best.pt`
-> `probe_{kind}_best.npz`), which is what the report analyses; the final
step-30,000 checkpoint (`{kind}.pt` -> `probe_{kind}.npz`) is probed too, only
to support the training-length control.

For every seed and both models: patch the 90 fixed cross-digit pairs at h1,
SLERP over 101 alphas, propagate, and record d(alpha) at h2, h3 and the output
layer (both the PLAN.md endpoint-normalized form and dir12's fraction form).
The regressor's 49-d output along each path is kept so reconstructions can be
rendered.

Usage: python experiments/probe.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from common import (MLP, N_TEST_POOL, RESULTS, build_dataset, build_pairs,
                    probe, setup)

SEEDS = [0, 1, 2]


def main():
    device = setup()
    ds = build_dataset()
    pairs = build_pairs(ds['te_lab'])
    ex_a = ds['te_in'][[p['idx_a'] for p in pairs]].to(device)
    ex_b = ds['te_in'][[p['idx_b'] for p in pairs]].to(device)
    meta = {'class_a': np.array([p['class_a'] for p in pairs]),
            'class_b': np.array([p['class_b'] for p in pairs]),
            'rep': np.array([p['rep'] for p in pairs]),
            'idx_a': np.array([p['idx_a'] for p in pairs]),
            'idx_b': np.array([p['idx_b'] for p in pairs])}

    for seed in SEEDS:
        d = os.path.join(RESULTS, f'seed{seed}')
        for kind, n_out in [('clf', 10), ('reg', 49)]:
            for tag in ['_best', '']:          # best-val checkpoint, then final
                m = MLP(n_out).to(device)
                m.load_state_dict(torch.load(os.path.join(d, f'{kind}{tag}.pt'),
                                             map_location=device))
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                rec = probe(m, ex_a, ex_b)
                np.savez_compressed(os.path.join(d, f'probe_{kind}{tag}.npz'),
                                    **rec, **meta)
                print(f'seed{seed} {kind}{tag or "(final)"}: d_h3 mean|d-a| = '
                      f'{np.abs(rec["d_h3"] - np.linspace(0,1,rec["d_h3"].shape[1])).mean():.4f}',
                      flush=True)
    np.savez_compressed(os.path.join(RESULTS, 'pairs.npz'), **meta)
    print('done')


if __name__ == '__main__':
    main()
