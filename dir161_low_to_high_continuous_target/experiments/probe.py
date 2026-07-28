#!/usr/bin/env python3
"""
dir161 S3 — run the frozen activation-interpolation probe on both trained models.

Every model is probed at its BEST-VALIDATION-LOSS checkpoint (`{kind}_best.pt`
-> `probe_{kind}_best.npz`), which is what the report analyses; the final
step-30,000 checkpoint (`{kind}.pt` -> `probe_{kind}.npz`) is probed too, only
to support the training-length control.

For every seed and both models: patch the 90 fixed cross-digit pairs at h1,
SLERP over 101 alphas, propagate, and record d(alpha) at h2, h3 and the output
layer (both the PLAN.md endpoint-normalized form and dir12's fraction form).
The predictor's 784-d output along each path is kept so high-resolution path
predictions can be rendered.

Endpoint fidelity (alpha 0 and 1 must reproduce the unpatched activations and
outputs) and a deterministic rerun are checked and recorded.

Usage: python experiments/probe.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from common import (MLP, RESULTS, build_dataset, build_pairs, probe, setup)

SEEDS = [0, 1, 2]


@torch.no_grad()
def endpoint_check(m, ex_a, ex_b):
    """Max |patched - unpatched| at alpha=0 and alpha=1, for h2, h3, output."""
    from common import N_POINTS, slerp_batch
    ha, oa = m.hidden_activations(ex_a)
    hb, ob = m.hidden_activations(ex_b)
    h1 = slerp_batch(ha[0], hb[0], N_POINTS)
    P = ex_a.shape[0]
    out, hs = m.forward_from(h1.reshape(P * N_POINTS, -1), 1)
    out = out.reshape(P, N_POINTS, -1)
    h2 = hs[0].reshape(P, N_POINTS, -1); h3 = hs[1].reshape(P, N_POINTS, -1)
    e = [(h2[:, 0] - ha[1]).abs().max(), (h3[:, 0] - ha[2]).abs().max(),
         (out[:, 0] - oa).abs().max(), (h2[:, -1] - hb[1]).abs().max(),
         (h3[:, -1] - hb[2]).abs().max(), (out[:, -1] - ob).abs().max()]
    return float(max(x.item() for x in e))


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

    checks = {'endpoint_max_abs_err': {}, 'determinism_max_abs_diff': {}}
    for seed in SEEDS:
        d = os.path.join(RESULTS, f'seed{seed}')
        for kind, n_out in [('clf', 10), ('pre', 784)]:
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
                key = f'seed{seed}_{kind}{tag or "_final"}'
                checks['endpoint_max_abs_err'][key] = endpoint_check(m, ex_a, ex_b)
                rec2 = probe(m, ex_a, ex_b, keep_out=False)
                checks['determinism_max_abs_diff'][key] = float(
                    max(np.abs(rec[k] - rec2[k]).max() for k in rec2))
                print(f'seed{seed} {kind}{tag or "(final)"}: d_h3 mean|d-a| = '
                      f'{np.abs(rec["d_h3"] - np.linspace(0,1,rec["d_h3"].shape[1])).mean():.4f}'
                      f'  endpoint_err={checks["endpoint_max_abs_err"][key]:.2e}',
                      flush=True)
    np.savez_compressed(os.path.join(RESULTS, 'pairs.npz'), **meta)
    checks['endpoint_tolerance'] = 1e-4
    checks['all_endpoints_within_tolerance'] = bool(
        max(checks['endpoint_max_abs_err'].values()) < 1e-4)
    checks['all_reruns_identical'] = bool(
        max(checks['determinism_max_abs_diff'].values()) == 0.0)
    json.dump(checks, open(os.path.join(RESULTS, 'probe_checks.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in checks.items() if not isinstance(v, dict)}))


if __name__ == '__main__':
    main()
