"""
dir17 — shared setup: brightness-regression MNIST dataset, tanh target family,
4-layer ReLU MLP with a scalar head, and the brightness-sweep probe.

Inputs are L2-normalized MNIST images rescaled to a sampled brightness b:
    x_b = b * x / ||x||_2
The target is a continuous, sharpness-controlled function of b alone:
    y_k(b) = tanh(k (b - b0)) / tanh(0.3 k),   b0 = 0.7
so k = 0.5 is nearly linear and k = 10 is nearly a binary switch, with the
endpoint range fixed at [-1, 1] for every k.

Architecture / optimizer follow the project's canonical MNIST plateau MLP
(784 -> 200 -> 200 -> 200 -> out, ReLU, AdamW lr 1e-3 wd 0.01, batch 200,
MSE); only the head width changes (10 -> 1).
"""
import gzip
import os
import struct

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = '/workspace/mars-plateaus-image/data/mnist'
RESULTS = os.path.join(HERE, 'results')
PLOTS = os.path.join(HERE, 'plots')

# ---- fixed experiment constants (PLAN.md) --------------------------------
B_LO, B_HI, B0 = 0.4, 1.0, 0.7
K_VALUES = (0.5, 1.0, 2.0, 5.0, 10.0)
N_SWEEP = 201                    # brightness grid points for the probe
N_PROBE_IMAGES = 100             # digit-balanced held-out test images
CENTER_LO, CENTER_HI = 0.64, 0.76
FLANK_LO, FLANK_HI = 0.52, 0.88
EPS = 1e-12

WIDTH, DEPTH = 200, 4
N_TRAIN, N_VAL, BATCH = 1000, 2000, 200
LR, WD = 1e-3, 0.01

# CVD-safe palette (CLAUDE.md rule 13): blue, vermillion, reddish purple,
# sky blue, orange. Never red-vs-green.
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v']


def setup_torch():
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.225)


# --------------------------------------------------------------------- data
def _read_idx(path):
    with gzip.open(path, 'rb') as f:
        _z, _d, ndim = struct.unpack('>HBB', f.read(4))
        shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist():
    f = {'tr_x': 'train-images-idx3-ubyte.gz', 'tr_y': 'train-labels-idx1-ubyte.gz',
         'te_x': 't10k-images-idx3-ubyte.gz', 'te_y': 't10k-labels-idx1-ubyte.gz'}
    a = {k: _read_idx(os.path.join(DATA, v)) for k, v in f.items()}
    return (torch.from_numpy(a['tr_x'].copy()).float().reshape(-1, 784) / 255.,
            torch.from_numpy(a['tr_y'].copy()).long(),
            torch.from_numpy(a['te_x'].copy()).float().reshape(-1, 784) / 255.,
            torch.from_numpy(a['te_y'].copy()).long())


def l2_normalize(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


def balanced_indices(labels, n_total, rng, pool_mask=None):
    """Digit-balanced sample of n_total indices (n_total/10 per digit)."""
    per = n_total // 10
    idx = []
    for d in range(10):
        cand = torch.nonzero(labels == d, as_tuple=False).squeeze(1).numpy()
        if pool_mask is not None:
            cand = cand[pool_mask[cand]]
        idx.append(rng.choice(cand, size=per, replace=False))
    return torch.from_numpy(np.sort(np.concatenate(idx))).long()


def target_fn(b, k):
    """y_k(b) = tanh(k (b - b0)) / tanh(0.3 k); range fixed to [-1, 1]."""
    return np.tanh(k * (b - B0)) / np.tanh(0.3 * k)


def target_fn_t(b, k):
    return torch.tanh(k * (b - B0)) / float(np.tanh(0.3 * k))


def build_dataset(seed, n_train=N_TRAIN):
    """Fixed splits + fixed brightness assignments for one seed.

    Splits come from the original MNIST images and are IDENTICAL for all five
    k values; only the scalar target changes with k. Digit labels are used
    only to make the splits digit-balanced, never as a learning target.
    Each split draws from its own generator, so the held-out probe images are
    the same regardless of the training-set size.
    """
    tr_x, tr_y, te_x, te_y = load_mnist()
    rtr = np.random.default_rng(1000 + seed)
    rva = np.random.default_rng(2000 + seed)
    rte = np.random.default_rng(3000)          # probe images fixed across seeds

    # train pool = MNIST train[:50000], val pool = MNIST train[50000:]
    pool_tr = np.zeros(len(tr_y), dtype=bool); pool_tr[:50000] = True
    pool_va = ~pool_tr
    tr_idx = balanced_indices(tr_y, n_train, rtr, pool_tr)
    va_idx = balanced_indices(tr_y, N_VAL, rva, pool_va)
    te_idx = balanced_indices(te_y, N_PROBE_IMAGES, rte)

    xtr = l2_normalize(tr_x[tr_idx])
    xva = l2_normalize(tr_x[va_idx])
    xte = l2_normalize(te_x[te_idx])

    btr = torch.from_numpy(rtr.uniform(B_LO, B_HI, n_train)).float()
    bva = torch.from_numpy(rva.uniform(B_LO, B_HI, N_VAL)).float()

    return {'xtr': xtr * btr[:, None], 'btr': btr,
            'xva': xva * bva[:, None], 'bva': bva,
            'xte_unit': xte, 'yte_lab': te_y[te_idx]}


# -------------------------------------------------------------------- model
class MLP(nn.Module):
    """784 -> 200 -> 200 -> 200 -> 1 ReLU MLP; post-ReLU hidden layers 1..3."""

    def __init__(self, n_out=1):
        super().__init__()
        self.depth = DEPTH
        self.act = nn.ReLU()
        self.linears = nn.ModuleList(
            [nn.Linear(784 if i == 0 else WIDTH,
                       n_out if i == DEPTH - 1 else WIDTH) for i in range(DEPTH)])

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < self.depth - 1:
                x = self.act(x)
        return x

    def hidden_activations(self, x):
        hs, out = [], x
        for i, lin in enumerate(self.linears):
            out = lin(out)
            if i < self.depth - 1:
                out = self.act(out)
                hs.append(out)
        return hs, out


# ------------------------------------------------------------------- probe
def brightness_grid():
    return np.linspace(B_LO, B_HI, N_SWEEP)


def center_mask():
    """Boolean mask over the 200 movement points whose left endpoint b_i is in
    [0.64, 0.76). Exactly 40/200 = 20% of the points, so uniform movement
    gives concentration 0.2."""
    b = brightness_grid()[:-1]
    return (b >= CENTER_LO - 1e-9) & (b < CENTER_HI - 1e-9)


def flank_mask():
    """Outer 40% of the brightness range: b_i < 0.52 or b_i >= 0.88.
    Uniform movement puts fraction 0.4 there; a true plateau puts ~0."""
    b = brightness_grid()[:-1]
    return (b < FLANK_LO - 1e-9) | (b >= FLANK_HI - 1e-9)


@torch.no_grad()
def sweep(model, x_unit, device):
    """Brightness sweep for a batch of unit-norm images.

    Returns preds [N, S] and per-layer activations list of [N, S, W].
    """
    grid = torch.from_numpy(brightness_grid()).float().to(device)
    x_unit = x_unit.to(device)
    n, s = x_unit.shape[0], len(grid)
    preds = torch.empty(n, s)
    acts = [torch.empty(n, s, WIDTH) for _ in range(DEPTH - 1)]
    for j, b in enumerate(grid):
        hs, out = model.hidden_activations(x_unit * b)
        preds[:, j] = out.squeeze(1).cpu()
        for l in range(DEPTH - 1):
            acts[l][:, j, :] = hs[l].cpu()
    return preds.numpy(), [a.numpy() for a in acts]


def normalized_movement(act):
    """s_l(b_i) = ||h(b_{i+1}) - h(b_i)|| / sum_j ||.||, per image. act [N,S,W]."""
    m = np.linalg.norm(np.diff(act, axis=1), axis=2)          # [N, S-1]
    return m / (m.sum(axis=1, keepdims=True) + EPS)


def concentration(s_curve):
    """C = sum of normalized movement inside the central 20% window."""
    return s_curve[..., center_mask()].sum(axis=-1)
