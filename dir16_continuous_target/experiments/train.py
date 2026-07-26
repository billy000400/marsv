#!/usr/bin/env python3
"""
dir16 S1 — train the matched classifier / regressor pair.

Both models: 784-200-200-200-out ReLU MLP, AdamW(1e-3, wd 0.01), batch 200,
30,000 steps (100 epochs over 60k, shuffled without replacement each epoch),
cosine LR 1e-3 -> 1e-6, MSE loss.  Only the head differs: 10 one-hot logits
(classifier) vs 49 pooled pixels (regressor).  Same seed => bit-identical
initial weights in the three shared layers (asserted) and identical batch order.

Usage: python experiments/train.py --seed 0 [--steps N]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from common import (MLP, N_STEPS, N_TRAIN, N_VAL, N_TEST_POOL, BATCH,
                    STEPS_PER_EPOCH, RESULTS, build_dataset, pool7, setup)

EVAL_EVERY = 100


@torch.no_grad()
def evaluate(model, x, tgt, lab, chunk=10_000):
    """Mean MSE loss on `tgt`, plus classification accuracy when lab is given."""
    tot, correct = 0.0, 0
    for i in range(0, len(x), chunk):
        o = model(x[i:i + chunk])
        tot += torch.nn.functional.mse_loss(o, tgt[i:i + chunk],
                                            reduction='sum').item()
        if lab is not None:
            correct += (o.argmax(1) == lab[i:i + chunk]).sum().item()
    return tot / (len(x) * tgt.shape[1]), (correct / len(x) if lab is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=N_STEPS)
    args = ap.parse_args()
    device = setup()
    ds = build_dataset()

    tr_in = ds['tr_in'].to(device)
    tr_oh = torch.eye(10, device=device)[ds['tr_lab'].to(device)]
    tr_rec = ds['tr_rec'].to(device)
    tr_lab = ds['tr_lab'].to(device)
    # val split = test[2000:] (disjoint from the probe endpoint pool test[:2000])
    va = slice(N_TEST_POOL, N_TEST_POOL + N_VAL)
    va_in = ds['te_in'][va].to(device)
    va_oh = torch.eye(10, device=device)[ds['te_lab'][va].to(device)]
    va_rec = ds['te_rec'][va].to(device)
    va_lab = ds['te_lab'][va].to(device)
    te_in = ds['te_in'].to(device)
    te_oh = torch.eye(10, device=device)[ds['te_lab'].to(device)]
    te_rec = ds['te_rec'].to(device)
    te_lab = ds['te_lab'].to(device)
    # fixed 10k train subset for the (cheap) train-loss trace
    trs = slice(0, 10_000)

    # --- regressor baselines (target = clean 7x7 pooled image), on the test set
    mean_tgt = tr_rec.mean(0, keepdim=True)
    base_mean = torch.nn.functional.mse_loss(mean_tgt.expand_as(te_rec), te_rec).item()
    base_pool = torch.nn.functional.mse_loss(pool7(ds['te_in']).to(device), te_rec).item()

    out = {}
    for kind, n_out in [('clf', 10), ('reg', 49)]:
        torch.manual_seed(args.seed)
        model = MLP(n_out).to(device)
        if kind == 'clf':
            shared0 = [model.linears[i].weight.detach().cpu().clone() for i in range(3)]
        else:
            for i in range(3):
                assert torch.equal(shared0[i], model.linears[i].weight.detach().cpu()), \
                    f'shared layer {i} init differs between clf and reg'
            print('[verify] clf and reg share bit-identical init in layers 1-3',
                  flush=True)
        tgt_tr = tr_oh if kind == 'clf' else tr_rec
        tgt_va = va_oh if kind == 'clf' else va_rec
        tgt_te = te_oh if kind == 'clf' else te_rec
        lab_va = va_lab if kind == 'clf' else None
        lab_te = te_lab if kind == 'clf' else None

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=1e-6)
        g = torch.Generator().manual_seed(args.seed)   # identical batch order
        hist = {'step': [], 'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        t0, perm = time.time(), None

        def record(step):
            model.eval()
            trl, _ = evaluate(model, tr_in[trs], tgt_tr[trs], None)
            vl, vacc = evaluate(model, va_in, tgt_va, lab_va)
            model.train()
            for k, v in [('step', step), ('train_loss', trl), ('val_loss', vl),
                         ('val_acc', vacc), ('lr', opt.param_groups[0]['lr'])]:
                hist[k].append(v)

        record(0)
        for step in range(1, args.steps + 1):
            pos = (step - 1) % STEPS_PER_EPOCH
            if pos == 0:
                perm = torch.randperm(N_TRAIN, generator=g)
            idx = perm[pos * BATCH:(pos + 1) * BATCH].to(device)
            loss = torch.nn.functional.mse_loss(model(tr_in[idx]), tgt_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % EVAL_EVERY == 0:
                record(step)
                if step % 5000 == 0:
                    print(f'  {kind} step {step}: train={hist["train_loss"][-1]:.5f} '
                          f'val={hist["val_loss"][-1]:.5f} '
                          f'acc={hist["val_acc"][-1]} ({time.time()-t0:.0f}s)',
                          flush=True)

        model.eval()
        te_loss, te_acc = evaluate(model, te_in, tgt_te, lab_te)
        d = os.path.join(RESULTS, f'seed{args.seed}')
        os.makedirs(d, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(d, f'{kind}.pt'))
        vmin = int(np.argmin(hist['val_loss']))
        out[kind] = {'test_loss': te_loss, 'test_acc': te_acc,
                     'final_train_loss': hist['train_loss'][-1],
                     'val_min_loss': hist['val_loss'][vmin],
                     'val_min_step': hist['step'][vmin],
                     'val_final_loss': hist['val_loss'][-1],
                     'overfits': hist['val_loss'][-1] > hist['val_loss'][vmin],
                     'seconds': time.time() - t0}
        json.dump(hist, open(os.path.join(d, f'{kind}_history.json'), 'w'))
        print(f'[{kind}] test_loss={te_loss:.6f} test_acc={te_acc} '
              f'val_min={hist["val_loss"][vmin]:.6f}@{hist["step"][vmin]} '
              f'val_final={hist["val_loss"][-1]:.6f}', flush=True)

    out['baselines'] = {'reg_mean_target': base_mean, 'reg_pool_corrupted': base_pool}
    out['config'] = {'seed': args.seed, 'steps': args.steps, 'batch': BATCH,
                     'sigma': 0.3, 'opt': 'AdamW(1e-3, wd=0.01)',
                     'sched': 'CosineAnnealingLR(eta_min=1e-6)'}
    json.dump(out, open(os.path.join(RESULTS, f'seed{args.seed}', 'summary.json'), 'w'),
              indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
