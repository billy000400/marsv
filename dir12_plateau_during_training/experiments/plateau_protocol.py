"""
dir12 — Frozen SLERP-interpolation plateau protocol (shared by all scripts).

Implements the interpolation experiment of "Activation Plateaus: Where and How
They Emerge" for the MNIST MLP, anchored on the branch's
scripts/image/interpolate_digits.py and src/paths.py slerp convention
(spherical direction interpolation + linearly interpolated endpoint norms).

Protocol (locked before checkpointed runs; see PLAN.md):
  - endpoints: fixed test images, ALL drawn from the first 2,000 of the 10K
    test images (operator feedback 07161151);
  - 55 pairs: 45 cross-class (one per unordered digit pair) + 10 within-class;
  - interpolate the post-ReLU first-hidden-layer activation h1 with 50-point
    norm-rescaled SLERP, patch at h1, propagate;
  - record h2, h3 (last hidden), logits, and the relative endpoint distance
        d(a) = ||x(a) - x(0)|| / (||x(a) - x(0)|| + ||x(a) - x(1)||)
    at every recorded layer, plus predicted class and max softmax prob.
"""
import os
import sys
sys.path.insert(0, '/workspace/mars-plateaus-image')

import numpy as np
import torch

HERE = '/workspace/marsv_agent_haoyang/dir12_plateau_during_training'
DATA = '/workspace/mars-plateaus-image/data/mnist'

N_POINTS = 50
N_TEST_POOL = 2000          # endpoints + test accuracy use test[:2000] only
EPS = 1e-10                 # matches interpolate_digits.py denominator guard

# checkpoint schedules (PLAN.md): full = 205 frames, fallback = 56 frames
FULL_SCHEDULE = [0, 10, 30, 100, 300] + list(range(500, 100_001, 500))
FALLBACK_SCHEDULE = [0, 10, 30, 100, 300] + list(range(2000, 100_001, 2000))
# anchor steps where the big raw activation arrays (h1 interp, h2, h3) are
# stored; every checkpoint stores state_dict + endpoint activations + logits
# + d-curves, so every frame is regenerable without retraining either way.
ANCHOR_STEPS = [0, 10, 30, 100, 300, 500, 1000, 2000, 3000, 5000, 10000,
                20000, 30000, 50000, 75000, 100000]

# fixed readable animation subset: ten cross-class pairs chosen by digit
# identity BEFORE viewing any results — every digit appears exactly twice.
ANIM_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
              (0, 8), (1, 7), (3, 5), (4, 9), (2, 6)]


def build_pair_bank(test_y):
    """55 deterministic pairs of test-set indices, all within test[:2000].

    Cross-class pair (a,b), a<b: rank-b image of class a and rank-a image of
    class b (ranks in test order), so each pair uses distinct images.
    Within-class pair for digit c: rank-10 and rank-11 images of class c
    (ranks 0..9 are reserved by the cross-class scheme).
    """
    pool_y = test_y[:N_TEST_POOL]
    by_class = {c: torch.where(pool_y == c)[0] for c in range(10)}
    pairs = []
    for a in range(10):
        for b in range(a + 1, 10):
            pairs.append({'class_a': a, 'class_b': b,
                          'idx_a': int(by_class[a][b]), 'idx_b': int(by_class[b][a])})
    for c in range(10):
        pairs.append({'class_a': c, 'class_b': c,
                      'idx_a': int(by_class[c][10]), 'idx_b': int(by_class[c][11])})
    return pairs


def slerp_batch(A, B, n_points):
    """Vectorized slerp_path for a batch of pairs.

    A, B: [P, d]. Returns [P, n_points, d]. Matches src.paths.slerp_path
    (great-circle direction, linearly interpolated magnitude) exactly,
    including the theta < 1e-6 fallback.
    """
    t = torch.linspace(0, 1, n_points, device=A.device)          # [T]
    nA = A.norm(dim=1, keepdim=True)                             # [P,1]
    nB = B.norm(dim=1, keepdim=True)
    uA, uB = A / nA, B / nB
    dot = torch.clamp((uA * uB).sum(dim=1, keepdim=True), -1.0, 1.0)  # [P,1]
    theta = torch.acos(dot)                                      # [P,1]
    sin_theta = torch.sin(theta)
    small = theta.abs() < 1e-6                                   # [P,1]
    safe_sin = torch.where(small, torch.ones_like(sin_theta), sin_theta)
    # coefficients [P,T]
    cA = torch.sin((1 - t)[None, :] * theta) / safe_sin
    cB = torch.sin(t[None, :] * theta) / safe_sin
    direction = cA[:, :, None] * uA[:, None, :] + cB[:, :, None] * uB[:, None, :]
    direction = torch.where(small[:, :, None], uA[:, None, :].expand_as(direction),
                            direction)
    mag = (1 - t)[None, :, None] * nA[:, None, :] + t[None, :, None] * nB[:, None, :]
    return direction * mag


def rel_dist(x):
    """d(a) per pair/point relative to path endpoints. x: [P, T, d] -> [P, T]."""
    d_a = (x - x[:, :1, :]).norm(dim=2)
    d_b = (x - x[:, -1:, :]).norm(dim=2)
    return d_a / (d_a + d_b + EPS)


@torch.no_grad()
def record_checkpoint(model, ex_a, ex_b):
    """Run the frozen protocol on one model.

    ex_a, ex_b: [P, 784] endpoint images (device = model device).
    Returns dict of numpy arrays (float32 curves, float16 activations).
    """
    P = ex_a.shape[0]
    hid_a, log_a = model.hidden_activations(ex_a)   # lists of [P,200], [P,10]
    hid_b, log_b = model.hidden_activations(ex_b)
    h1_path = slerp_batch(hid_a[0], hid_b[0], N_POINTS)          # [P,T,200]
    flat = h1_path.reshape(P * N_POINTS, -1)
    logits, hs = model.forward_from(flat, 1)
    h2 = hs[0].reshape(P, N_POINTS, -1)
    h3 = hs[1].reshape(P, N_POINTS, -1)
    logits = logits.reshape(P, N_POINTS, -1)

    prob = torch.softmax(logits, dim=2)
    rec = {
        'd_h2': rel_dist(h2).cpu().numpy().astype(np.float32),
        'd_h3': rel_dist(h3).cpu().numpy().astype(np.float32),
        'd_logit': rel_dist(logits).cpu().numpy().astype(np.float32),
        'pred': logits.argmax(dim=2).cpu().numpy().astype(np.int8),
        'max_prob': prob.max(dim=2).values.cpu().numpy().astype(np.float16),
        'max_raw': logits.max(dim=2).values.cpu().numpy().astype(np.float16),
        'logits': logits.cpu().numpy().astype(np.float16),
        'end_logits': torch.stack([log_a, log_b], dim=1).cpu().numpy().astype(np.float32),
        'end_h1': torch.stack([hid_a[0], hid_b[0]], 1).cpu().numpy().astype(np.float16),
        'end_h2': torch.stack([hid_a[1], hid_b[1]], 1).cpu().numpy().astype(np.float16),
        'end_h3': torch.stack([hid_a[2], hid_b[2]], 1).cpu().numpy().astype(np.float16),
    }
    big = {
        'h1_interp': h1_path.cpu().numpy().astype(np.float16),
        'h2': h2.cpu().numpy().astype(np.float16),
        'h3': h3.cpu().numpy().astype(np.float16),
    }
    return rec, big


def load_state_model(path, device):
    from src.mnist import MLP
    state = torch.load(path, map_location=device)
    m = MLP(depth=4, width=200, activation='relu').to(device)
    m.load_state_dict(state)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m
