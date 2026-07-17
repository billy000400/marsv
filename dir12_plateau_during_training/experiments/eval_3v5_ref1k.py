#!/usr/bin/env python3
"""
dir12 S9 — Evaluate the frozen 50-path 3->5 bank on the EXISTING 1,000-example
reference run (converged primary, seed0_pl_f0.5_p100) at its 16 anchor
checkpoints, so the full-60k run's 3->5 distribution has a paired baseline at
matched optimizer steps. No training; states are loaded from disk.

Output: results/full_mnist_from_scratch/ref1k_seed_0/step_<t>.npz (+ manifest)
with the same record schema as train_full60k.py.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from plateau_protocol import (HERE, DATA, N_POINTS, N_TEST_POOL, ANCHOR_STEPS,
                              build_pair_bank, record_checkpoint,
                              load_state_model)
from train_full60k import build_3v5_bank, seg_counts
from src.mnist import load_mnist

torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.225)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED, SFX = 0, '_pl_f0.5_p100'
out_dir = os.path.join(HERE, 'results', 'full_mnist_from_scratch', f'ref1k_seed_{SEED}')
os.makedirs(out_dir, exist_ok=True)

_tx, _ty, test_x, test_y = load_mnist(DATA)
pairs = build_pair_bank(test_y) + build_3v5_bank(test_y)   # identical 105-pair bank
ex_a = test_x[[q['idx_a'] for q in pairs]].to(device)
ex_b = test_x[[q['idx_b'] for q in pairs]].to(device)
true_a = np.array([q['class_a'] for q in pairs])
true_b = np.array([q['class_b'] for q in pairs])

for step in ANCHOR_STEPS:
    ck = os.path.join(HERE, 'results', 'ckpts_movie', f'seed{SEED}{SFX}',
                      f'step{step}.pt')
    model = load_state_model(ck, device)
    rec, _big = record_checkpoint(model, ex_a, ex_b)
    arrays = dict(rec, step=np.int64(step), seed=np.int64(SEED),
                  t=np.linspace(0, 1, N_POINTS).astype(np.float32),
                  idx_a=np.array([q['idx_a'] for q in pairs]),
                  idx_b=np.array([q['idx_b'] for q in pairs]),
                  true_a=true_a, true_b=true_b, seg_count=seg_counts(rec['pred']))
    np.savez_compressed(os.path.join(out_dir, f'step_{step}.npz'), **arrays)
    print(f'ref1k step {step}: done', flush=True)

with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
    json.dump({'seed': SEED, 'source_run': f'seed{SEED}{SFX}',
               'ckpt_steps': ANCHOR_STEPS, 'anchor_steps': [],
               'pairs': pairs, 'n_3v5_pairs': 50,
               'note': '1k-reference run evaluated on the 105-pair bank '
                       '(55 original + 50 frozen 3v5) at anchor steps'}, f, indent=1)
print('done ->', out_dir)
