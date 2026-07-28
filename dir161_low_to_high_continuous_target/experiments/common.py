"""
dir161 — shared setup: low-resolution MNIST input, dual targets, frozen SLERP probe.

Both models see the SAME single input z = D(y): the clean 28x28 MNIST image y
average-pooled 4x4 to 7x7 (49 values).  The classifier target is the one-hot
digit label; the low-to-high predictor target is the original clean 28x28 image
(784 values).  Both trained with per-output-unit MSE.

Operators (PLAN.md):
  D : 784 -> 49    non-overlapping 4x4 average pooling
  U : 49  -> 784   block repetition (each 7x7 value copied into its 4x4 block)
  P = I - U D      removed-detail projector; D(P(y)) = 0 and D(U(z)) = z

The interpolation protocol is dir12/dir16's frozen SLERP protocol: patch at the
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
N_TEST_POOL = 2000          # untouched endpoint / evaluation pool = test[:2000]
N_VAL = 8000                # checkpoint-validation split = test[2000:10000]
EPS = 1e-10

D_IN, WIDTH, DEPTH = 49, 200, 4
N_TRAIN, BATCH, STEPS_PER_EPOCH, N_STEPS = 60_000, 200, 300, 30_000

# CVD-safe palette (CLAUDE.md rule 13): blue, vermillion, reddish purple,
# sky blue, orange.  Never red-vs-green.
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
C_CLF, C_PRE = CVD[0], CVD[1]


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


def D(y):
    """4x4 average pooling, [N,784] -> [N,49]."""
    return torch.nn.functional.avg_pool2d(y.reshape(-1, 1, 28, 28), 4).reshape(-1, 49)


def U(z):
    """Block repetition, [N,49] -> [N,784].  Each 7x7 cell fills its 4x4 block."""
    return z.reshape(-1, 1, 7, 7).repeat_interleave(4, 2).repeat_interleave(4, 3) \
            .reshape(-1, 784)


def Pdet(y):
    """Removed-detail projector P = I - U D, [N,784] -> [N,784]."""
    return y - U(D(y))


def build_dataset():
    """Shared low-resolution input / dual-target dataset (no corruption)."""
    tr_y, tr_lab, te_y, te_lab = load_mnist()
    return {'tr_in': D(tr_y), 'tr_lab': tr_lab, 'tr_img': tr_y,
            'te_in': D(te_y), 'te_lab': te_lab, 'te_img': te_y}


def bicubic(z):
    """Fixed 7x7 -> 28x28 bicubic upsample baseline, clipped to [0,1]."""
    up = torch.nn.functional.interpolate(z.reshape(-1, 1, 7, 7), size=(28, 28),
                                         mode='bicubic', align_corners=False)
    return up.clamp(0, 1).reshape(-1, 784)


# -------------------------------------------------------------------------- model
class MLP(nn.Module):
    """49 -> 200 -> 200 -> 200 -> n_out ReLU MLP (generic head).

    Layers 0..2 are constructed first, so for a fixed torch seed the classifier
    (n_out=10) and the low-to-high predictor (n_out=784) receive BIT-IDENTICAL
    initial weights in every shared layer; only the head differs.
    """

    def __init__(self, n_out):
        super().__init__()
        self.depth = DEPTH
        self.act = nn.ReLU()
        self.linears = nn.ModuleList(
            [nn.Linear(D_IN if i == 0 else WIDTH,
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

    Identical construction to dir16 (and dir12 for replica 0), so the
    hand-selected transitions (incl. 6->7) match earlier directions.
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
def probe(model, ex_a, ex_b, keep_out=True):
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
    rec = {'d_h2': f32(d_norm(h2)), 'd_h3': f32(d_norm(h3)),
           'd_out': f32(d_norm(out)),
           'f_h2': f32(d_frac(h2)), 'f_h3': f32(d_frac(h3)),
           'f_out': f32(d_frac(out))}
    if keep_out:
        rec['out'] = out.cpu().numpy().astype(np.float16)
    return rec


def setup(frac=0.225, threads=2):
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)
        return 'cuda'
    return 'cpu'
