#!/usr/bin/env python3
"""
dir12 S1 — Match the post at the final checkpoint.

1. slerp_batch must reproduce src.paths.slerp_path exactly (per-pair loop).
2. alpha=0/1 patched outputs must reproduce the unpatched endpoint outputs.
3. Run the frozen 55-pair protocol on the existing seed-0 step-100000
   checkpoint and save example plateau / non-plateau / within-class curves.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from plateau_protocol import (HERE, DATA, N_POINTS, build_pair_bank,
                              slerp_batch, record_checkpoint, load_state_model)
from src.mnist import load_mnist
from src.paths import slerp_path

torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.225)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

_, _, test_x, test_y = load_mnist(DATA)
pairs = build_pair_bank(test_y)
print(f"pair bank: {len(pairs)} pairs; max test index used:",
      max(max(p['idx_a'], p['idx_b']) for p in pairs))

model = load_state_model(os.path.join(HERE, 'results', 'ckpts', 'seed0',
                                      'step100000.pt'), device)
ex_a = test_x[[p['idx_a'] for p in pairs]].to(device)
ex_b = test_x[[p['idx_b'] for p in pairs]].to(device)

# --- check 1: vectorized slerp == reference slerp_path -----------------------
hid_a, _ = model.hidden_activations(ex_a)
hid_b, _ = model.hidden_activations(ex_b)
batch = slerp_batch(hid_a[0], hid_b[0], N_POINTS)
worst = 0.0
for i in range(len(pairs)):
    ref = slerp_path(hid_a[0][i], hid_b[0][i], N_POINTS)
    worst = max(worst, (batch[i] - ref).abs().max().item())
print(f"check 1  slerp_batch vs slerp_path: max|diff| = {worst:.2e}")
assert worst < 1e-4

# --- check 2: endpoints reproduce unpatched outputs --------------------------
rec, big = record_checkpoint(model, ex_a, ex_b)
end = rec['end_logits']                       # [P,2,10] unpatched
d0 = np.abs(rec['logits'][:, 0, :].astype(np.float32) - end[:, 0]).max()
d1 = np.abs(rec['logits'][:, -1, :].astype(np.float32) - end[:, 1]).max()
print(f"check 2  alpha=0/1 logit reproduction: max|diff| = {max(d0, d1):.2e} "
      f"(float16 storage quantum ~1e-3)")
assert max(d0, d1) < 5e-2

# also negative-entry sanity: slerp of two nonneg vectors stays nonneg
print("check 3  min h1_interp entry:", float(big['h1_interp'].min()))

# --- example curves at the final checkpoint ----------------------------------
t = np.linspace(0, 1, N_POINTS)
# pick examples deterministically by pair identity (not by result):
show = [(0, 1), (3, 5), (4, 9), (7, 7)]       # 3 cross + 1 within-class
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
for ax, (a, b) in zip(axes, show):
    i = next(k for k, p in enumerate(pairs)
             if p['class_a'] == min(a, b) and p['class_b'] == max(a, b))
    ax.plot(t, rec['d_logit'][i], lw=2, color='C0', label='logits')
    ax.plot(t, rec['d_h3'][i], lw=1.2, color='C1', alpha=0.8, label='h3')
    ax.plot(t, rec['d_h2'][i], lw=1.2, color='C2', alpha=0.8, label='h2')
    ax.plot([0, 1], [0, 1], ls=':', color='gray', lw=1)
    for k in range(N_POINTS):
        ax.plot(t[k], -0.05, marker='s', ms=3, color=f"C{rec['pred'][i, k] % 10}")
    ax.set_title(f"{a} → {b}" + ("  (within-class)" if a == b else ""))
    ax.set_xlabel(r'interpolation $\alpha$')
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r'$d(\alpha)$')
axes[0].legend(fontsize=8)
fig.suptitle('Step 100,000 (seed 0): relative endpoint distance along the SLERP path '
             '(squares: predicted class)', y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'plots', 's1_final_checkpoint_examples.png'),
            dpi=140, bbox_inches='tight')
plt.close()

# summary stats over all 45 cross pairs
cross = slice(0, 45)
mid = rec['d_logit'][cross, N_POINTS // 2]
n_flat = int(((rec['d_logit'][cross, :10] < 0.2).all(axis=1) &
              (rec['d_logit'][cross, -10:] > 0.8).all(axis=1)).sum())
print(f"cross-class pairs: {n_flat}/45 have d<0.2 on first fifth AND d>0.8 on last fifth")
print("median |d(0.5)-0.5| =", float(np.median(np.abs(mid - 0.5))))
print("S1 OK")
