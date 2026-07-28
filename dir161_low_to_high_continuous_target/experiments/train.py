#!/usr/bin/env python3
"""
dir161 S2 — train the matched classifier / low-to-high predictor pair.

Both models: 49-200-200-200-out ReLU MLP, AdamW(1e-3, wd 0.01), batch 200,
30,000 steps (100 epochs over 60k, shuffled without replacement each epoch),
cosine LR 1e-3 -> 1e-6, per-output-unit MSE loss.  Only the head differs:
10 one-hot logits (classifier) vs 784 clean pixels (predictor).  Same seed =>
bit-identical initial weights in the three shared layers (asserted) and
identical batch order.

Two checkpoints per model: `{kind}_best.pt` (lowest validation loss over the
run, evaluated every 100 steps) — the primary probe checkpoint — and `{kind}.pt`
(final step), kept only as the training-length control.

Usage: python experiments/train.py --seed 0 [--steps N]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from common import (MLP, N_STEPS, N_TRAIN, N_VAL, N_TEST_POOL, BATCH,
                    STEPS_PER_EPOCH, RESULTS, build_dataset, setup)

EVAL_EVERY = 100


@torch.no_grad()
def evaluate(model, x, tgt, lab, chunk=5_000):
    """Mean per-output-unit MSE on `tgt`, plus accuracy when lab is given."""
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
    tr_img = ds['tr_img'].to(device)
    # val split = test[2000:10000] (disjoint from the endpoint pool test[:2000])
    va = slice(N_TEST_POOL, N_TEST_POOL + N_VAL)
    va_in = ds['te_in'][va].to(device)
    va_oh = torch.eye(10, device=device)[ds['te_lab'][va].to(device)]
    va_img = ds['te_img'][va].to(device)
    va_lab = ds['te_lab'][va].to(device)
    # task-adequacy pool = untouched test[:2000]
    ep = slice(0, N_TEST_POOL)
    ep_in = ds['te_in'][ep].to(device)
    ep_oh = torch.eye(10, device=device)[ds['te_lab'][ep].to(device)]
    ep_img = ds['te_img'][ep].to(device)
    ep_lab = ds['te_lab'][ep].to(device)
    trs = slice(0, 10_000)      # fixed train subset for the cheap train-loss trace

    out = {}
    for kind, n_out in [('clf', 10), ('pre', 784)]:
        torch.manual_seed(args.seed)
        model = MLP(n_out).to(device)
        if kind == 'clf':
            shared0 = [model.linears[i].weight.detach().cpu().clone() for i in range(3)]
        else:
            for i in range(3):
                assert torch.equal(shared0[i], model.linears[i].weight.detach().cpu()), \
                    f'shared layer {i} init differs between clf and pre'
            print('[verify] clf and pre share bit-identical init in layers 1-3', flush=True)
        tgt_tr = tr_oh if kind == 'clf' else tr_img
        tgt_va = va_oh if kind == 'clf' else va_img
        tgt_ep = ep_oh if kind == 'clf' else ep_img
        lab_va = va_lab if kind == 'clf' else None
        lab_ep = ep_lab if kind == 'clf' else None

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

        snap = lambda: {k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}
        record(0)
        best = {'val': hist['val_loss'][0], 'step': 0, 'sd': snap()}
        for step in range(1, args.steps + 1):
            pos = (step - 1) % STEPS_PER_EPOCH
            if pos == 0:
                perm = torch.randperm(N_TRAIN, generator=g)
            idx = perm[pos * BATCH:(pos + 1) * BATCH].to(device)
            loss = torch.nn.functional.mse_loss(model(tr_in[idx]), tgt_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % EVAL_EVERY == 0:
                record(step)
                if hist['val_loss'][-1] < best['val']:
                    best = {'val': hist['val_loss'][-1], 'step': step, 'sd': snap()}
                if step % 5000 == 0:
                    print(f'  {kind} step {step}: train={hist["train_loss"][-1]:.5f} '
                          f'val={hist["val_loss"][-1]:.5f} '
                          f'acc={hist["val_acc"][-1]} ({time.time()-t0:.0f}s)', flush=True)

        model.eval()
        ep_loss, ep_acc = evaluate(model, ep_in, tgt_ep, lab_ep)
        d = os.path.join(RESULTS, f'seed{args.seed}')
        os.makedirs(d, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(d, f'{kind}.pt'))
        torch.save(best['sd'], os.path.join(d, f'{kind}_best.pt'))
        model.load_state_dict(best['sd'])
        bl, bacc = evaluate(model, ep_in, tgt_ep, lab_ep)
        b_train, _ = evaluate(model, tr_in[trs], tgt_tr[trs], None)
        vmin = int(np.argmin(hist['val_loss']))
        assert hist['step'][vmin] == best['step'], 'best-checkpoint step mismatch'
        out[kind] = {'final_pool_loss': ep_loss, 'final_pool_acc': ep_acc,
                     'final_train_loss': hist['train_loss'][-1],
                     'val_min_loss': hist['val_loss'][vmin],
                     'val_min_step': hist['step'][vmin],
                     'val_final_loss': hist['val_loss'][-1],
                     'overfits': hist['val_loss'][-1] > hist['val_loss'][vmin],
                     'best_pool_loss': bl, 'best_pool_acc': bacc,
                     'best_train_loss': b_train, 'seconds': time.time() - t0}
        json.dump(hist, open(os.path.join(d, f'{kind}_history.json'), 'w'))
        print(f'[{kind}] pool_loss={ep_loss:.6f} acc={ep_acc} '
              f'val_min={hist["val_loss"][vmin]:.6f}@{hist["step"][vmin]} '
              f'best_ckpt: pool_loss={bl:.6f} acc={bacc}', flush=True)

    out['config'] = {'seed': args.seed, 'steps': args.steps, 'batch': BATCH,
                     'opt': 'AdamW(1e-3, wd=0.01)',
                     'sched': 'CosineAnnealingLR(eta_min=1e-6)'}
    json.dump(out, open(os.path.join(RESULTS, f'seed{args.seed}', 'summary.json'), 'w'),
              indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
