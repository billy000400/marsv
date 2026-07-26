"""
dir16 — shared setup: corrupted-MNIST dataset, matched MLP, frozen SLERP probe.

One dataset feeds BOTH models: input = clean MNIST image + one fixed mild
Gaussian corruption (clipped to [0,1]); classifier target = digit label
(one-hot, MSE); regressor target = the CLEAN image average-pooled to 7x7 (49
continuous values, MSE). Same images, same order, same corruption for both.

The interpolation protocol is dir12's frozen SLERP protocol
(dir12_plateau_during_training/experiments/plateau_protocol.py): patch at the
post-ReLU first hidden layer h1, spherically interpolate endpoint activations
with linearly interpolated norms, propagate through the remaining layers.
"""
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = '/workspace/mars-plateaus-image/data/mnist'
RESULTS = os.path.join(HERE, 'results')
PLOTS = os.path.join(HERE, 'plots')

N_POINTS = 101              # alpha grid (PLAN.md)
N_TEST_POOL = 2000          # probe endpoints drawn from test[:2000] (dir12)
N_VAL = 8000                # val split for the overfitting check = test[2000:]
SIGMA = 0.3                 # fixed mild Gaussian corruption
CORRUPT_SEED = 1234
EPS = 1e-10

WIDTH, DEPTH = 200, 4
N_TRAIN, BATCH, STEPS_PER_EPOCH, N_STEPS = 60_000, 200, 300, 30_000

# CVD-safe palette (CLAUDE.md rule 13): blue, vermillion, reddish purple,
# sky blue, orange.  Never red-vs-green.
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
C_CLF, C_REG = CVD[0], CVD[1]


# --------------------------------------------------------------------------- data
def _read_idx(path):
    import gzip, struct
    with gzip.open(path, 'rb') as f:
        _zeros, _dtype, ndim = struct.unpack('>HBB', f.read(4))
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


def pool7(x):
    """Average-pool a [N,784] clean image batch to [N,49] (4x4 mean pooling)."""
    return torch.nn.functional.avg_pool2d(x.reshape(-1, 1, 28, 28), 4).reshape(-1, 49)


def build_dataset():
    """Return the shared corrupted-input / dual-target dataset.

    corruption is drawn once with a FIXED seed, so both models and all training
    seeds see bit-identical inputs.
    """
    tr_x, tr_y, te_x, te_y = load_mnist()
    g = torch.Generator().manual_seed(CORRUPT_SEED)
    tr_in = (tr_x + SIGMA * torch.randn(tr_x.shape, generator=g)).clamp_(0, 1)
    te_in = (te_x + SIGMA * torch.randn(te_x.shape, generator=g)).clamp_(0, 1)
    return {'tr_in': tr_in, 'tr_lab': tr_y, 'tr_rec': pool7(tr_x),
            'te_in': te_in, 'te_lab': te_y, 'te_rec': pool7(te_x),
            'te_clean': te_x}


# -------------------------------------------------------------------------- model
class MLP(nn.Module):
    """784 -> 200 -> 200 -> 200 -> n_out ReLU MLP (dir12 config, generic head).

    Layers 0..2 are constructed first, so for a fixed torch seed the classifier
    (n_out=10) and regressor (n_out=49) receive BIT-IDENTICAL initial weights in
    every shared layer; only the head differs.
    """

    def __init__(self, n_out):
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
        hs = []
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < self.depth - 1:
                x = self.act(x)
                hs.append(x)
        return hs, x

    def forward_from(self, h, layer):
        """Run from post-activation hidden `layer` (1-indexed) to the output."""
        hs, x = [], h
        for i in range(layer, self.depth):
            x = self.linears[i](x)
            if i < self.depth - 1:
                x = self.act(x)
                hs.append(x)
        return x, hs


# --------------------------------------------------------------------- pair bank
def build_pairs(te_lab):
    """90 fixed cross-digit test pairs: 2 per unordered digit pair (45 x 2).

    Replica-0 of each digit pair is EXACTLY dir12's cross-class pair (rank-b
    image of class a with rank-a image of class b, ranks in test order within
    test[:2000]) so the hand-selected transitions (incl. 6->7) are directly
    comparable with prior results; replica-1 uses ranks shifted by 12.
    """
    pool = te_lab[:N_TEST_POOL]
    by_c = {c: torch.where(pool == c)[0] for c in range(10)}
    pairs = []
    for a in range(10):
        for b in range(a + 1, 10):
            for r in range(2):
                pairs.append({'class_a': a, 'class_b': b, 'rep': r,
                              'idx_a': int(by_c[a][b + 12 * r]),
                              'idx_b': int(by_c[b][a + 12 * r])})
    return pairs


# ------------------------------------------------------------------------- probe
def slerp_batch(A, B, n_points):
    """dir12 slerp_path, vectorized. A,B: [P,d] -> [P,T,d]."""
    t = torch.linspace(0, 1, n_points, device=A.device)
    nA, nB = A.norm(dim=1, keepdim=True), B.norm(dim=1, keepdim=True)
    uA, uB = A / nA, B / nB
    dot = torch.clamp((uA * uB).sum(dim=1, keepdim=True), -1., 1.)
    theta = torch.acos(dot)
    sin_t = torch.sin(theta)
    small = theta.abs() < 1e-6
    safe = torch.where(small, torch.ones_like(sin_t), sin_t)
    cA = torch.sin((1 - t)[None, :] * theta) / safe
    cB = torch.sin(t[None, :] * theta) / safe
    d = cA[:, :, None] * uA[:, None, :] + cB[:, :, None] * uB[:, None, :]
    d = torch.where(small[:, :, None], uA[:, None, :].expand_as(d), d)
    mag = (1 - t)[None, :, None] * nA[:, None, :] + t[None, :, None] * nB[:, None, :]
    return d * mag


def d_norm(x):
    """PLAN.md d(alpha) = ||x(a)-x(0)|| / ||x(1)-x(0)||.  [P,T,d] -> [P,T]."""
    num = (x - x[:, :1, :]).norm(dim=2)
    den = (x[:, -1:, :] - x[:, :1, :]).norm(dim=2)
    return num / (den + EPS)


def d_frac(x):
    """dir12 relative distance d = d_a / (d_a + d_b).  [P,T,d] -> [P,T]."""
    da = (x - x[:, :1, :]).norm(dim=2)
    db = (x - x[:, -1:, :]).norm(dim=2)
    return da / (da + db + EPS)


@torch.no_grad()
def probe(model, ex_a, ex_b):
    """Frozen protocol on one model. Returns d-curves per layer + raw outputs."""
    P = ex_a.shape[0]
    ha, _ = model.hidden_activations(ex_a)
    hb, _ = model.hidden_activations(ex_b)
    h1 = slerp_batch(ha[0], hb[0], N_POINTS)
    out, hs = model.forward_from(h1.reshape(P * N_POINTS, -1), 1)
    h2 = hs[0].reshape(P, N_POINTS, -1)
    h3 = hs[1].reshape(P, N_POINTS, -1)
    out = out.reshape(P, N_POINTS, -1)
    f32 = lambda z: z.cpu().numpy().astype(np.float32)
    return {'d_h2': f32(d_norm(h2)), 'd_h3': f32(d_norm(h3)),
            'd_out': f32(d_norm(out)),
            'f_h2': f32(d_frac(h2)), 'f_h3': f32(d_frac(h3)),
            'f_out': f32(d_frac(out)),
            'out': out.cpu().numpy().astype(np.float16)}


def setup(frac=0.225, threads=2):
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)
        return 'cuda'
    return 'cpu'
