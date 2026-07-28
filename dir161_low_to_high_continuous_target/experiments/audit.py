#!/usr/bin/env python3
"""
dir161 S1 — freeze and validate the protocol before any training.

Checks the operator identities D(U(z)) = z and D(P(y)) = 0 numerically, records
the held-out energy of the removed-detail component r = P(y) (the audit that the
784-value target contains structure NOT explicitly present in the 49-value
input), writes the frozen split/pair manifest with checksums, and renders the
data/detail audit figure.

Usage: python experiments/audit.py
"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CVD, D, N_TEST_POOL, N_VAL, PLOTS, RESULTS, U, Pdet,
                    bicubic, build_dataset, build_pairs, setup)

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9})


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    setup()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)
    ds = build_dataset()
    te_img, te_in, te_lab = ds['te_img'], ds['te_in'], ds['te_lab']
    pool = slice(0, N_TEST_POOL)

    # --- operator identities -------------------------------------------------
    z = te_in[pool]
    y = te_img[pool]
    id1 = (D(U(z)) - z).abs().max().item()          # D U = I on 49-space
    r = Pdet(y)
    id2 = D(r).abs().max().item()                   # D P = 0
    id3 = (U(D(y)) + r - y).abs().max().item()      # U D + P = I
    assert id1 < 1e-5 and id2 < 1e-5 and id3 < 1e-5, (id1, id2, id3)

    # --- removed-detail energy ----------------------------------------------
    e_img = (y ** 2).mean().item()
    e_det = (r ** 2).mean().item()
    e_blk = (U(D(y)) ** 2).mean().item()
    audit = {
        'identity_max_abs_err': {'D_U_z_minus_z': id1, 'D_P_y': id2,
                                 'UD_plus_P_minus_I': id3},
        'per_pixel_mean_square': {'image_y': e_img, 'block_mean_UDy': e_blk,
                                  'removed_detail_Py': e_det},
        'detail_energy_fraction': e_det / e_img,
        'n_endpoint_pool': int(N_TEST_POOL), 'n_val': int(N_VAL),
        'input_dim': 49, 'target_dim': 784,
    }

    # --- frozen manifest -----------------------------------------------------
    pairs = build_pairs(te_lab)
    man = {
        'split': {'train': [0, 60000], 'endpoint_pool_test': [0, N_TEST_POOL],
                  'val_test': [N_TEST_POOL, N_TEST_POOL + N_VAL]},
        'preprocessing': 'pixels/255 in [0,1]; input z = 4x4 avg-pool of clean y '
                         '(49 values); classifier target = one-hot label; '
                         'low-to-high target = clean y (784 values); no corruption',
        'operators': {'D': '4x4 non-overlapping average pooling',
                      'U': 'block repetition of each 7x7 cell into its 4x4 block',
                      'P': 'I - U D'},
        'seeds': [0, 1, 2], 'n_pairs': len(pairs),
        'checksums': {'te_in_pool': sha(z.numpy()), 'te_img_pool': sha(y.numpy()),
                      'pair_idx_a': sha(np.array([p['idx_a'] for p in pairs])),
                      'pair_idx_b': sha(np.array([p['idx_b'] for p in pairs]))},
        'pairs': pairs,
    }
    json.dump(man, open(os.path.join(RESULTS, 'manifest.json'), 'w'), indent=1)
    json.dump(audit, open(os.path.join(RESULTS, 'audit.json'), 'w'), indent=1)

    # --- figure 1: data / detail audit --------------------------------------
    idx = [int(torch.where(te_lab[pool] == c)[0][0]) for c in [0, 3, 6, 7, 9]]
    rows = [('clean target $y$ (28x28)', y[idx].reshape(-1, 28, 28), 0, 1, 'gray'),
            ('model input $z=D(y)$ (7x7)', z[idx].reshape(-1, 7, 7), 0, 1, 'gray'),
            ('block repeat $U(z)$', U(z[idx]).reshape(-1, 28, 28), 0, 1, 'gray'),
            ('bicubic upsample', bicubic(z[idx]).reshape(-1, 28, 28), 0, 1, 'gray'),
            ('removed detail $r=P(y)$', r[idx].reshape(-1, 28, 28), -.5, .5, 'coolwarm')]
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(1.35 * len(idx), 1.4 * len(rows)))
    for ri, (name, imgs, vmin, vmax, cm) in enumerate(rows):
        for cj in range(len(idx)):
            ax = axes[ri, cj]
            ax.imshow(np.asarray(imgs[cj]), cmap=cm, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if cj == 0:
                ax.set_ylabel(name, fontsize=6.5)
    fig.suptitle(f'removed detail carries {100*e_det/e_img:.1f}% of pixel energy '
                 f'(mean square {e_det:.4f} vs {e_img:.4f})', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'data_audit.png'))
    plt.close(fig)

    print(json.dumps(audit, indent=1))
    print('manifest ->', os.path.join(RESULTS, 'manifest.json'))


if __name__ == '__main__':
    main()
