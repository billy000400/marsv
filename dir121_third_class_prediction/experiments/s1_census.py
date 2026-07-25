"""
S1/S2 — 45-transition x 100-pair interpolation census at the final converged
direction-12 checkpoints (seed 0 primary, seeds 1 and 2 for confirmation).

Reuses direction 12's frozen protocol by import (no copy, no retraining):
50-point norm-rescaled SLERP on post-ReLU h1, patched at h1, propagated to the
logits, with the logit-space relative endpoint distance d(alpha).

Pair bank: for each unordered digit pair (a<b), the rank-i class-a test image is
paired with the rank-i class-b test image, i = 0..99, ranks taken inside the
first 2,000 MNIST test images (direction 12's construction, extended from its
10 animation transitions to all 45).

Writes results/s1_census.npz only; classification lives in s1_analyze.py.
"""
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/mars-plateaus-image')
sys.path.insert(0, '/workspace/marsv_agent_haoyang/dir12_plateau_during_training/experiments')
from plateau_protocol import record_checkpoint, load_state_model, N_POINTS, N_TEST_POOL

HERE = '/workspace/marsv_agent_haoyang/dir121_third_class_prediction'
DIR12 = '/workspace/marsv_agent_haoyang/dir12_plateau_during_training'
DATA = '/workspace/mars-plateaus-image/data/mnist'
SEEDS = [0, 1, 2]
FINAL_STEP = 30000
N_PER = 100
CHUNK = 900

torch.set_num_threads(2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.225)

TRANSITIONS = [(a, b) for a in range(10) for b in range(a + 1, 10)]   # 45


def build_bank(test_y):
    """[45,100] endpoint index arrays inside the first 2,000 test images."""
    pool_y = test_y[:N_TEST_POOL]
    by_class = {c: torch.where(pool_y == c)[0] for c in range(10)}
    idx_a = np.zeros((len(TRANSITIONS), N_PER), dtype=np.int64)
    idx_b = np.zeros((len(TRANSITIONS), N_PER), dtype=np.int64)
    for k, (a, b) in enumerate(TRANSITIONS):
        assert len(by_class[a]) >= N_PER and len(by_class[b]) >= N_PER, (a, b)
        idx_a[k] = by_class[a][:N_PER].cpu().numpy()
        idx_b[k] = by_class[b][:N_PER].cpu().numpy()
    return idx_a, idx_b


def ckpt_path(seed):
    return os.path.join(DIR12, 'results', 'full_mnist_from_scratch',
                        f'seed_{seed}', 'ckpts', f'step{FINAL_STEP}.pt')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from src.mnist import load_mnist
    _tx, _ty, test_x, test_y = load_mnist(DATA)
    idx_a, idx_b = build_bank(test_y)
    P = idx_a.size                                        # 4500
    flat_a, flat_b = idx_a.reshape(-1), idx_b.reshape(-1)
    ex_a = test_x[flat_a].to(device)
    ex_b = test_x[flat_b].to(device)

    n_seed = len(SEEDS)
    pred = np.zeros((n_seed, P, N_POINTS), dtype=np.int8)
    d_logit = np.zeros((n_seed, P, N_POINTS), dtype=np.float32)
    end_pred = np.zeros((n_seed, P, 2), dtype=np.int8)
    hashes = []
    for si, seed in enumerate(SEEDS):
        path = ckpt_path(seed)
        hashes.append(sha256(path))
        model = load_state_model(path, device)
        for s in range(0, P, CHUNK):
            e = min(s + CHUNK, P)
            rec, _big = record_checkpoint(model, ex_a[s:e], ex_b[s:e])
            pred[si, s:e] = rec['pred']
            d_logit[si, s:e] = rec['d_logit']
            end_pred[si, s:e] = rec['end_logits'].argmax(axis=2)
            del rec, _big
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        acc = (end_pred[si].reshape(-1) ==
               np.concatenate([test_y[flat_a].numpy(), test_y[flat_b].numpy()])
               .reshape(2, P).T.reshape(-1)).mean()
        print(f'seed {seed}: endpoint accuracy over the {2*P} bank endpoints = {acc:.4f}')

    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    np.savez_compressed(
        os.path.join(HERE, 'results', 's1_census.npz'),
        transitions=np.array(TRANSITIONS), idx_a=idx_a, idx_b=idx_b,
        seeds=np.array(SEEDS), step=FINAL_STEP, pred=pred, d_logit=d_logit,
        end_pred=end_pred, label_a=test_y[flat_a].numpy().reshape(45, N_PER),
        label_b=test_y[flat_b].numpy().reshape(45, N_PER),
        ckpt_paths=np.array([ckpt_path(s) for s in SEEDS]),
        ckpt_sha256=np.array(hashes), n_per=N_PER, n_points=N_POINTS)
    print('saved results/s1_census.npz')


if __name__ == '__main__':
    main()
