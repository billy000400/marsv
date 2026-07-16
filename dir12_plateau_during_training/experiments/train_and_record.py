#!/usr/bin/env python3
"""
dir12 S3 — Train the MNIST MLP and run the frozen SLERP plateau protocol at
every scheduled checkpoint, saving state_dicts + per-checkpoint records.

Exact branch training config (as experiments/train_checkpoints.py): depth-4
width-200 ReLU MLP, 1000-sample MNIST subset (seed-defined), AdamW(1e-3, wd
0.01), MSE on one-hot, batch 200, 100k steps. Test metrics on test[:2000]
(operator feedback 07161151).

Per checkpoint: state_dict -> results/ckpts_movie/seed<S>/step<T>.pt, protocol
record -> results/plateau_records/seed_<S>/step_<T>.npz (d-curves, per-point
logits/preds/probs, endpoint activations at every hidden layer; raw 50-point
h1/h2/h3 activation arrays additionally at ANCHOR_STEPS — every other frame is
regenerable from state_dict + saved endpoint h1 via the deterministic slerp).

Usage: python experiments/train_and_record.py --seed 0 --schedule full
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from plateau_protocol import (HERE, DATA, N_POINTS, N_TEST_POOL, FULL_SCHEDULE,
                              FALLBACK_SCHEDULE, ANCHOR_STEPS, build_pair_bank,
                              record_checkpoint)
from src.mnist import MLP, load_mnist

torch.set_num_threads(2)


@torch.no_grad()
def eval_metrics(model, x, y, one_hots, loss_fn, batch=2000):
    model.eval()
    correct, conf_sum, loss_sum = 0, 0.0, 0.0
    for i in range(0, len(x), batch):
        logits = model(x[i:i + batch])
        loss_sum += loss_fn(logits, one_hots[y[i:i + batch]]).item() * len(logits)
        conf, pred = logits.max(dim=1)          # max raw output = confidence
        correct += (pred == y[i:i + batch]).sum().item()
        conf_sum += conf.sum().item()
    model.train()
    n = len(x)
    return correct / n, conf_sum / n, loss_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--schedule', default='full', choices=['full', 'fallback'])
    args = p.parse_args()
    schedule = FULL_SCHEDULE if args.schedule == 'full' else FALLBACK_SCHEDULE
    anchors = set(ANCHOR_STEPS) if args.schedule == 'full' else {0, 100000}

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.225)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # exact call order of the branch config so seed semantics stay identical
    torch.manual_seed(args.seed)
    train_x, train_y, test_x, test_y = load_mnist(DATA)
    subset_idx = torch.randint(0, len(train_x), (1000,))
    pairs = build_pair_bank(test_y)
    ex_a = test_x[[q['idx_a'] for q in pairs]].to(device)
    ex_b = test_x[[q['idx_b'] for q in pairs]].to(device)
    true_a = np.array([q['class_a'] for q in pairs])
    true_b = np.array([q['class_b'] for q in pairs])
    train_x = train_x[subset_idx].to(device)
    train_y = train_y[subset_idx].to(device)
    test_x = test_x[:N_TEST_POOL].to(device)
    test_y = test_y[:N_TEST_POOL].to(device)

    model = MLP(depth=4, width=200, activation='relu').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = torch.nn.MSELoss()
    one_hots = torch.eye(10, device=device)

    ck_dir = os.path.join(HERE, 'results', 'ckpts_movie', f'seed{args.seed}')
    rec_dir = os.path.join(HERE, 'results', 'plateau_records', f'seed_{args.seed}')
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)

    history = {'step': [], 'train_loss': [], 'train_acc': [], 'test_loss': [],
               'test_acc': [], 'train_conf': [], 'test_conf': []}

    def checkpoint(step):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state, os.path.join(ck_dir, f'step{step}.pt'))
        model.eval()
        rec, big = record_checkpoint(model, ex_a, ex_b)
        model.train()
        tr_acc, tr_conf, tr_loss = eval_metrics(model, train_x, train_y, one_hots, loss_fn)
        te_acc, te_conf, te_loss = eval_metrics(model, test_x, test_y, one_hots, loss_fn)
        for k, v in [('step', step), ('train_loss', tr_loss), ('train_acc', tr_acc),
                     ('test_loss', te_loss), ('test_acc', te_acc),
                     ('train_conf', tr_conf), ('test_conf', te_conf)]:
            history[k].append(v)
        arrays = dict(rec, step=np.int64(step), seed=np.int64(args.seed),
                      t=np.linspace(0, 1, N_POINTS).astype(np.float32),
                      idx_a=np.array([q['idx_a'] for q in pairs]),
                      idx_b=np.array([q['idx_b'] for q in pairs]),
                      true_a=true_a, true_b=true_b,
                      test_acc=np.float32(te_acc), test_conf=np.float32(te_conf))
        if step in anchors:
            arrays.update(big)
        np.savez_compressed(os.path.join(rec_dir, f'step_{step}.npz'), **arrays)
        print(f"step {step}: loss={tr_loss:.5f} tr_acc={tr_acc:.3f} "
              f"te_acc={te_acc:.3f} te_conf={te_conf:.3f}", flush=True)

    t0 = time.time()
    ck_set = set(schedule)
    checkpoint(0)
    for step in range(1, max(schedule) + 1):
        idx = torch.randint(0, 1000, (200,), device=device)
        logits = model(train_x[idx])
        loss = loss_fn(logits, one_hots[train_y[idx]])
        opt.zero_grad(); loss.backward(); opt.step()
        if step in ck_set:
            checkpoint(step)

    with open(os.path.join(ck_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    manifest = {'seed': args.seed, 'schedule': args.schedule,
                'ckpt_steps': schedule, 'anchor_steps': sorted(anchors),
                'n_points': N_POINTS, 'n_test_pool': N_TEST_POOL,
                'pairs': pairs, 'subset_idx': subset_idx.tolist(),
                'config': {'depth': 4, 'width': 200, 'activation': 'relu',
                           'train_points': 1000, 'batch': 200, 'lr': 1e-3,
                           'weight_decay': 0.01, 'loss': 'MSE-on-one-hot',
                           'opt': 'AdamW', 'steps': max(schedule)}}
    with open(os.path.join(rec_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f"done: {len(schedule)} checkpoints in {time.time() - t0:.0f}s -> {rec_dir}")


if __name__ == '__main__':
    main()
