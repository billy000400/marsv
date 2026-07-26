#!/usr/bin/env python3
"""
dir16 S3 control — is the classifier's extra curvature just overtraining?

At step 30,000 the classifier has memorized the training set (train MSE ~1e-7)
while the regressor is still improving.  A skeptic can therefore attribute the
smoothness gap to training regime rather than to target type.  This control
retrains each classifier with the IDENTICAL seed, batch order and schedule but
stops at that seed's validation-loss minimum (the best-generalizing
classifier), then runs the same probe.  If the gap survives, it is not an
overtraining artifact.

Usage: python experiments/control_earlystop.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from common import (MLP, N_STEPS, N_TRAIN, N_TEST_POOL, BATCH, STEPS_PER_EPOCH,
                    RESULTS, build_dataset, build_pairs, probe, setup)

SEEDS = [0, 1, 2]


def main():
    device = setup()
    ds = build_dataset()
    pairs = build_pairs(ds['te_lab'])
    ex_a = ds['te_in'][[p['idx_a'] for p in pairs]].to(device)
    ex_b = ds['te_in'][[p['idx_b'] for p in pairs]].to(device)
    tr_in = ds['tr_in'].to(device)
    tr_oh = torch.eye(10, device=device)[ds['tr_lab'].to(device)]
    te_in, te_lab = ds['te_in'].to(device), ds['te_lab'].to(device)

    info = {}
    for seed in SEEDS:
        d = os.path.join(RESULTS, f'seed{seed}')
        stop = json.load(open(os.path.join(d, 'summary.json')))['clf']['val_min_step']
        torch.manual_seed(seed)
        model = MLP(10).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=N_STEPS, eta_min=1e-6)          # same schedule, cut short
        g = torch.Generator().manual_seed(seed)
        perm = None
        for step in range(1, stop + 1):
            pos = (step - 1) % STEPS_PER_EPOCH
            if pos == 0:
                perm = torch.randperm(N_TRAIN, generator=g)
            idx = perm[pos * BATCH:(pos + 1) * BATCH].to(device)
            loss = torch.nn.functional.mse_loss(model(tr_in[idx]), tr_oh[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(te_in).argmax(1) == te_lab).float().mean())
            trl = float(torch.nn.functional.mse_loss(model(tr_in[:10_000]),
                                                     tr_oh[:10_000]))
        rec = probe(model, ex_a, ex_b)
        np.savez_compressed(os.path.join(d, 'probe_clf_early.npz'), **rec)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(d, 'clf_early.pt'))
        info[str(seed)] = {'stop_step': stop, 'test_acc': acc, 'train_loss': trl}
        print(f'seed{seed}: early-stop @{stop} acc={acc:.4f} train_mse={trl:.2e}',
              flush=True)
    json.dump(info, open(os.path.join(RESULTS, 'control_earlystop.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main()
