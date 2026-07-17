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
def eval_metrics(model, x, y, loss_of, batch=2000):
    model.eval()
    correct, conf_sum, prob_sum, loss_sum = 0, 0.0, 0.0, 0.0
    for i in range(0, len(x), batch):
        logits = model(x[i:i + batch])
        loss_sum += loss_of(logits, y[i:i + batch]).item() * len(logits)
        conf, pred = logits.max(dim=1)          # max raw output = confidence
        correct += (pred == y[i:i + batch]).sum().item()
        conf_sum += conf.sum().item()
        prob_sum += torch.softmax(logits, 1).max(dim=1).values.sum().item()
    model.train()
    n = len(x)
    return correct / n, conf_sum / n, prob_sum / n, loss_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--schedule', default='full', choices=['full', 'fallback'])
    p.add_argument('--loss', default='mse', choices=['mse', 'ce'])
    # feedback 07161721: 'plateau' = ReduceLROnPlateau stepped EVERY optimization
    # step on the full-train-set loss (patience 10 steps, factor 0.5)
    # feedback 07161834: scheduler search for a SMOOTHLY converging run —
    # 'cosine' = CosineAnnealingLR over all steps; --factor/--patience retune
    # 'plateau' (non-default values get their own output suffix).
    p.add_argument('--sched', default='const', choices=['const', 'plateau', 'cosine'])
    p.add_argument('--factor', type=float, default=0.5)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--eta-min', type=float, default=1e-6)
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
    one_hots = torch.eye(10, device=device)
    if args.loss == 'ce':                       # CE on integer labels (MLE loss)
        ce_fn = torch.nn.CrossEntropyLoss()
        loss_of = lambda logits, y: ce_fn(logits, y)
    else:                                       # branch default: MSE on one-hot
        mse_fn = torch.nn.MSELoss()
        loss_of = lambda logits, y: mse_fn(logits, one_hots[y])
    if args.sched == 'const':
        sched_sfx = ''
    elif args.sched == 'cosine':
        sched_sfx = '_cos'
    elif (args.factor, args.patience) == (0.5, 10):
        sched_sfx = '_sched'                     # the original 07161721 run
    else:
        sched_sfx = f'_pl_f{args.factor:g}_p{args.patience}'
    sfx = ('' if args.loss == 'mse' else '_ce') + sched_sfx

    sched = None
    if args.sched == 'plateau':
        # monitored quantity = full-train-set loss, recomputed after every step
        # (deterministic; the per-batch loss is far too noisy for patience=10).
        # No RNG is consumed -> identical init/subset/batch order as the const run.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=args.factor, patience=args.patience,
            threshold=1e-4, threshold_mode='rel', min_lr=1e-8)
    elif args.sched == 'cosine':
        # no RNG consumed either; stepped after every optimization step
        n_total = max(FULL_SCHEDULE if args.schedule == 'full' else FALLBACK_SCHEDULE)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_total, eta_min=args.eta_min)

    ck_dir = os.path.join(HERE, 'results', 'ckpts_movie', f'seed{args.seed}{sfx}')
    rec_dir = os.path.join(HERE, 'results', 'plateau_records', f'seed_{args.seed}{sfx}')
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)

    history = {'step': [], 'train_loss': [], 'train_acc': [], 'test_loss': [],
               'test_acc': [], 'train_conf': [], 'test_conf': [],
               'train_prob': [], 'test_prob': [], 'lr': []}

    def checkpoint(step):
        # CE/scheduled runs keep state_dicts only at anchor steps (run is
        # deterministic & regenerable; avoids ~190 MB extra in the shared repo)
        if (args.loss == 'mse' and args.sched == 'const') or step in anchors:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(state, os.path.join(ck_dir, f'step{step}.pt'))
        model.eval()
        rec, big = record_checkpoint(model, ex_a, ex_b)
        model.train()
        tr_acc, tr_conf, tr_prob, tr_loss = eval_metrics(model, train_x, train_y, loss_of)
        te_acc, te_conf, te_prob, te_loss = eval_metrics(model, test_x, test_y, loss_of)
        for k, v in [('step', step), ('train_loss', tr_loss), ('train_acc', tr_acc),
                     ('test_loss', te_loss), ('test_acc', te_acc),
                     ('train_conf', tr_conf), ('test_conf', te_conf),
                     ('train_prob', tr_prob), ('test_prob', te_prob),
                     ('lr', opt.param_groups[0]['lr'])]:
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

    @torch.no_grad()
    def full_train_loss():
        return loss_of(model(train_x), train_y).item()

    t0 = time.time()
    ck_set = set(schedule)
    n_steps = max(schedule)
    if sched is not None:
        trace_loss = np.zeros(n_steps + 1, np.float32)
        trace_lr = np.zeros(n_steps + 1, np.float32)
        trace_loss[0] = full_train_loss()
        trace_lr[0] = opt.param_groups[0]['lr']
    checkpoint(0)
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, 1000, (200,), device=device)
        logits = model(train_x[idx])
        loss = loss_of(logits, train_y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if sched is not None:
            mon = full_train_loss()
            if args.sched == 'plateau':
                sched.step(mon)
            else:
                sched.step()
            trace_loss[step] = mon
            trace_lr[step] = opt.param_groups[0]['lr']
        if step in ck_set:
            checkpoint(step)
    if sched is not None:
        np.savez_compressed(os.path.join(ck_dir, 'sched_trace.npz'),
                            full_train_loss=trace_loss, lr=trace_lr)

    with open(os.path.join(ck_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    manifest = {'seed': args.seed, 'schedule': args.schedule,
                'ckpt_steps': schedule, 'anchor_steps': sorted(anchors),
                'n_points': N_POINTS, 'n_test_pool': N_TEST_POOL,
                'pairs': pairs, 'subset_idx': subset_idx.tolist(),
                'config': {'depth': 4, 'width': 200, 'activation': 'relu',
                           'train_points': 1000, 'batch': 200, 'lr': 1e-3,
                           'weight_decay': 0.01,
                           'loss': 'MSE-on-one-hot' if args.loss == 'mse'
                                   else 'CE-on-labels',
                           'lr_schedule': ('constant' if args.sched == 'const' else
                                   f'CosineAnnealingLR(T_max=steps, eta_min={args.eta_min:g}) '
                                   'stepped every step' if args.sched == 'cosine' else
                                   f'ReduceLROnPlateau(factor={args.factor:g}, '
                                   f'patience={args.patience}, threshold=1e-4 rel, '
                                   'min_lr=1e-8) stepped every step on the '
                                   'full-train-set loss'),
                           'opt': 'AdamW', 'steps': max(schedule)}}
    with open(os.path.join(rec_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f"done: {len(schedule)} checkpoints in {time.time() - t0:.0f}s -> {rec_dir}")


if __name__ == '__main__':
    main()
