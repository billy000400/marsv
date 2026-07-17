#!/usr/bin/env python3
"""
dir12 S7/S8 — Fresh full-MNIST training from the saved step-0 untrained
weights, with the frozen SLERP plateau protocol at every scheduled checkpoint.

Per PLAN.md (reopened extension): load the ORIGINAL step-0 untrained MSE
checkpoint (never any trained weights), train on all 60,000 MNIST training
images shuffled WITHOUT replacement each epoch (300 steps/epoch at batch 200),
AdamW(1e-3, wd 0.01), MSE on one-hot, fixed cosine schedule 1e-3 -> 1e-6 over
30,000 steps. Evaluate the unchanged protocol on the original 55-pair bank
PLUS a frozen 50-path 3-to-5 bank (unfiltered, selected before viewing any
full-data result).

Startup verifications (S7, asserted before training):
  1. loaded checkpoint bit-equals a fresh re-derivation of the seed's
     initialization (proves it is untrained / step 0);
  2. it also bit-equals the constant-LR run's step0.pt;
  3. during training: the first epoch uses all 60,000 indices exactly once.
After training a manifest check verifies every scheduled record exists with
the full schema (also standalone: --check).

Usage: python experiments/train_full60k.py --seed 0 [--schedule fallback]
                                           [--max-step N] [--check]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from plateau_protocol import (HERE, DATA, N_POINTS, N_TEST_POOL,
                              build_pair_bank, record_checkpoint)
from train_and_record import eval_metrics
from src.mnist import MLP, load_mnist

torch.set_num_threads(2)

# PLAN.md checkpoint schedule for the full-data run: early steps + one frame
# per full-MNIST epoch (300 steps) through 30,000 -> 104 frames.
FULL60K_SCHEDULE = [0, 10, 30, 100] + list(range(300, 30_001, 300))
# fallback density for confirmation seeds 1/2 -> 25 frames.
FULL60K_FALLBACK = [0, 10, 30, 100, 300] + list(range(1500, 30_001, 1500))
# anchors storing the big raw activation arrays + state_dicts (PLAN fallback
# step list; every other frame regenerable from anchor state + determinism).
FULL60K_ANCHORS = [0, 10, 30, 100, 300, 1500, 3000, 6000, 15000, 30000]

N_TRAIN, BATCH, STEPS_PER_EPOCH, N_STEPS = 60_000, 200, 300, 30_000
N_3V5 = 50

REQUIRED = ['d_h2', 'd_h3', 'd_logit', 'pred', 'max_prob', 'max_raw', 'logits',
            'end_logits', 'end_h1', 'end_h2', 'end_h3', 't', 'idx_a', 'idx_b',
            'true_a', 'true_b', 'test_acc', 'step', 'seed', 'seg_count']
BIG = ['h1_interp', 'h2', 'h3']


def build_3v5_bank(test_y):
    """50 deterministic unfiltered 3->5 pairs: rank-i 3 with rank-i 5 (test
    order, within test[:N_TEST_POOL]), i = 0..49. Frozen before any full-data
    result was viewed."""
    pool_y = test_y[:N_TEST_POOL]
    threes = torch.where(pool_y == 3)[0]
    fives = torch.where(pool_y == 5)[0]
    assert len(threes) >= N_3V5 and len(fives) >= N_3V5
    return [{'class_a': 3, 'class_b': 5,
             'idx_a': int(threes[i]), 'idx_b': int(fives[i])}
            for i in range(N_3V5)]


def seg_counts(pred):
    """Run-length segment count per path. pred: [P, T] -> [P] int8."""
    changes = (pred[:, 1:] != pred[:, :-1]).sum(axis=1)
    return (changes + 1).astype(np.int8)


def verify_untrained(seed, state, device):
    """Re-derive the seed's initialization with the exact original call order
    (train_and_record.py) and assert bit-equality; also compare against the
    constant-LR run's step0.pt."""
    torch.manual_seed(seed)
    _tx, _ty, _sx, _sy = load_mnist(DATA)
    _ = torch.randint(0, len(_tx), (1000,))          # subset draw (consumed RNG)
    ref = MLP(depth=4, width=200, activation='relu')
    for k, v in ref.state_dict().items():
        assert torch.equal(v, state[k]), f'init mismatch at {k}'
    const0 = os.path.join(HERE, 'results', 'ckpts_movie', f'seed{seed}', 'step0.pt')
    cstate = torch.load(const0, map_location='cpu')
    for k, v in cstate.items():
        assert torch.equal(v, state[k]), f'const-run step0 mismatch at {k}'
    print(f'[verify] step0 checkpoint == fresh seed-{seed} init == const-run '
          f'step0 (bit-exact, all tensors) -> untrained confirmed')


def check_manifest(rec_dir):
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
            bad = [k for k in ['d_logit', 'd_h3', 'd_h2', 'pred']
                   if k in z.files and z[k].shape != (P, N_POINTS)]
            if 'logits' in z.files and z['logits'].shape != (P, N_POINTS, 10):
                bad.append('logits')
            if missing or bad:
                problems.append(f'step {s}: missing={missing} bad_shape={bad}')
            else:
                n_ok += 1
        except Exception as e:
            problems.append(f'step {s}: CORRUPT ({e})')
        if s in man['anchor_steps']:
            ck = os.path.join(rec_dir, 'ckpts', f'step{s}.pt')
            if not os.path.exists(ck):
                problems.append(f'step {s}: state_dict MISSING')
    print(f'{rec_dir}: {n_ok}/{len(man["ckpt_steps"])} records OK, '
          f'{len(problems)} problems')
    for p in problems:
        print(' ', p)
    return not problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--schedule', default='full', choices=['full', 'fallback'])
    p.add_argument('--max-step', type=int, default=None,
                   help='truncate the run (smoke test); writes to *_smoke')
    p.add_argument('--check', action='store_true',
                   help='only run the manifest check on an existing dir')
    args = p.parse_args()

    smoke = args.max_step is not None
    rec_dir = os.path.join(HERE, 'results', 'full_mnist_from_scratch',
                           f'seed_{args.seed}' + ('_smoke' if smoke else ''))
    if args.check:
        sys.exit(0 if check_manifest(rec_dir) else 1)

    schedule = FULL60K_SCHEDULE if args.schedule == 'full' else FULL60K_FALLBACK
    anchors = set(FULL60K_ANCHORS)
    n_steps = N_STEPS
    if smoke:
        schedule = [s for s in schedule if s <= args.max_step]
        anchors = {s for s in anchors if s <= args.max_step}
        n_steps = args.max_step

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.225)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- load the untrained step-0 weights of the primary converged 1k run
    ck0 = os.path.join(HERE, 'results', 'ckpts_movie',
                       f'seed{args.seed}_pl_f0.5_p100', 'step0.pt')
    state = torch.load(ck0, map_location='cpu')
    verify_untrained(args.seed, state, device)

    train_x, train_y, test_x, test_y = load_mnist(DATA)
    pairs = build_pair_bank(test_y) + build_3v5_bank(test_y)
    ex_a = test_x[[q['idx_a'] for q in pairs]].to(device)
    ex_b = test_x[[q['idx_b'] for q in pairs]].to(device)
    true_a = np.array([q['class_a'] for q in pairs])
    true_b = np.array([q['class_b'] for q in pairs])
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x = test_x[:N_TEST_POOL].to(device)
    test_y = test_y[:N_TEST_POOL].to(device)

    model = MLP(depth=4, width=200, activation='relu').to(device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=N_STEPS, eta_min=1e-6)
    one_hots = torch.eye(10, device=device)
    mse_fn = torch.nn.MSELoss()
    loss_of = lambda logits, y: mse_fn(logits, one_hots[y])

    os.makedirs(os.path.join(rec_dir, 'ckpts'), exist_ok=True)
    history = {'step': [], 'train_loss': [], 'train_acc': [], 'test_loss': [],
               'test_acc': [], 'train_conf': [], 'test_conf': [],
               'train_prob': [], 'test_prob': [], 'lr': []}

    def checkpoint(step):
        if step in anchors:
            st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(st, os.path.join(rec_dir, 'ckpts', f'step{step}.pt'))
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
                      seg_count=seg_counts(rec['pred']),
                      test_acc=np.float32(te_acc), test_conf=np.float32(te_conf))
        if step in anchors:
            arrays.update(big)
        np.savez_compressed(os.path.join(rec_dir, f'step_{step}.npz'), **arrays)
        print(f"step {step}: loss={tr_loss:.6f} tr_acc={tr_acc:.4f} "
              f"te_acc={te_acc:.4f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)

    # --- training loop: shuffle WITHOUT replacement every epoch
    g = torch.Generator().manual_seed(args.seed)   # dedicated shuffle RNG
    t0 = time.time()
    ck_set = set(schedule)
    trace_loss = np.zeros(n_steps + 1, np.float32)  # per-step BATCH loss
    trace_lr = np.zeros(n_steps + 1, np.float32)
    trace_lr[0] = opt.param_groups[0]['lr']
    checkpoint(0)
    perm = None
    first_epoch = []
    for step in range(1, n_steps + 1):
        pos = (step - 1) % STEPS_PER_EPOCH
        if pos == 0:
            perm = torch.randperm(N_TRAIN, generator=g)
        idx = perm[pos * BATCH:(pos + 1) * BATCH].to(device)
        if step <= STEPS_PER_EPOCH:
            first_epoch.append(idx.cpu())
        logits = model(train_x[idx])
        loss = loss_of(logits, train_y[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        trace_loss[step] = loss.item()
        trace_lr[step] = opt.param_groups[0]['lr']
        if step == STEPS_PER_EPOCH:
            seen = torch.cat(first_epoch)
            assert torch.equal(seen.sort().values, torch.arange(N_TRAIN)), \
                'first epoch does not cover all 60,000 indices exactly once'
            print('[verify] first epoch used all 60,000 training indices '
                  'exactly once (no replacement)', flush=True)
        if step in ck_set:
            checkpoint(step)

    np.savez_compressed(os.path.join(rec_dir, 'trace.npz'),
                        batch_loss=trace_loss, lr=trace_lr)
    with open(os.path.join(rec_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    manifest = {'seed': args.seed, 'schedule': args.schedule,
                'ckpt_steps': schedule, 'anchor_steps': sorted(anchors),
                'n_points': N_POINTS, 'n_test_pool': N_TEST_POOL,
                'pairs': pairs, 'n_original_pairs': len(pairs) - N_3V5,
                'n_3v5_pairs': N_3V5,
                'init_from': ck0,
                'config': {'depth': 4, 'width': 200, 'activation': 'relu',
                           'train_points': N_TRAIN, 'batch': BATCH,
                           'shuffle': 'without replacement each epoch '
                                      f'({STEPS_PER_EPOCH} steps/epoch)',
                           'lr': 1e-3, 'weight_decay': 0.01,
                           'loss': 'MSE-on-one-hot',
                           'lr_schedule': 'CosineAnnealingLR(T_max=30000, '
                                          'eta_min=1e-6) stepped every step',
                           'opt': 'AdamW', 'steps': n_steps}}
    with open(os.path.join(rec_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f"done: {len(schedule)} checkpoints in {time.time() - t0:.0f}s -> {rec_dir}")
    ok = check_manifest(rec_dir)
    print('[verify] manifest complete' if ok else '[verify] MANIFEST PROBLEMS')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
