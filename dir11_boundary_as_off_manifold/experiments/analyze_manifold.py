#!/usr/bin/env python3
"""
dir11 — Are plateau-separated stable regions manifold components?

Reuses the image-models MNIST MLP (depth 4, width 200) checkpoint and its
slerp interpolation. Two questions:

  (1) Direct-path test: does the plateau boundary (max |d'(t)|) of a
      cross-region digit-9->9 slerp coincide with the lowest-support part of
      the path (highest kNN radius to the natural first-layer activation cloud)?

  (2) Component test: in a mutual-kNN graph of natural first-layer activations,
      do the two candidate digit-9 regions merge into one connected component at
      a NORMAL within-manifold neighborhood scale, or only at an anomalously
      large scale (like genuinely different digits)?

Layers: interpolate first hidden layer L1 (post-ReLU, 200-dim); measure
downstream distance at last hidden layer L3 (200-dim). Natural reference cloud =
L1 activations of correctly classified MNIST test images.

Outputs: results/*.pt (numbers), plots/*.png. CPU-only (tiny model).
"""
import os, sys, json
sys.path.insert(0, '/network/mars-plateaus-image')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from src.mnist import load_checkpoint, load_mnist
from src.paths import slerp_path

torch.set_num_threads(2)
torch.manual_seed(0)
np.random.seed(0)

HERE = '/network/marsv_agent_haoyang/dir11_boundary_as_off_manifold'
RES = os.path.join(HERE, 'results')
PLOT = os.path.join(HERE, 'plots')
os.makedirs(RES, exist_ok=True); os.makedirs(PLOT, exist_ok=True)
CKPT = '/network/mars-plateaus-image/results/image/mnist_mlp_d4_w200_relu_n1000_s100000.pt'
DATA = '/network/mars-plateaus-image/data/mnist'

N_POINTS = 200
KNN_SUPPORT = 10          # k for support radius along paths
SMOOTH = 5                # smoothing window for |d'(t)|

# ------------------------------------------------------------------ load
model, ck = load_checkpoint(CKPT)
n_test = ck['config']['n_test']
_, _, test_x, test_y = load_mnist(DATA)
test_x, test_y = test_x[:n_test], test_y[:n_test]
with torch.no_grad():
    hiddens, logits = model.hidden_activations(test_x)
L1 = hiddens[0]           # [n,200] first hidden (interpolation layer)
L3 = hiddens[-1]          # [n,200] last hidden (measurement layer)
pred = logits.argmax(1)
correct = pred == test_y
cidx = torch.where(correct)[0]           # indices of correctly classified
nat = L1[cidx].numpy()                    # natural reference cloud (L1)
nat_lab = test_y[cidx].numpy()
print(f"natural cloud: {nat.shape[0]} correctly-classified L1 activations")

# ------------------------------------------------------------- 9 regions
nine_mask = (test_y == 9) & correct
nine_idx = torch.where(nine_mask)[0]
nine_L3 = L3[nine_idx].numpy()
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(nine_L3)
lab9 = km.labels_
# medoid of each cluster = image closest to centroid (downstream)
def medoid(idxs, feats, centroid):
    d = np.linalg.norm(feats - centroid, axis=1)
    return idxs[int(d.argmin())]
regionA = nine_idx[lab9 == 0].numpy()
regionB = nine_idx[lab9 == 1].numpy()
medA = medoid(regionA, L3[regionA].numpy(), km.cluster_centers_[0])
medB = medoid(regionB, L3[regionB].numpy(), km.cluster_centers_[1])
print(f"region A: {len(regionA)} nines, region B: {len(regionB)} nines")

# --------------------------------------------- paths + support along path
nn_support = NearestNeighbors(n_neighbors=KNN_SUPPORT + 1).fit(nat)

def path_analysis(idx0, idx1):
    a = L1[idx0]; b = L1[idx1]
    pts = slerp_path(a, b, N_POINTS)
    with torch.no_grad():
        lo, hs = model.forward_from(pts, 1)
    he = hs[-1]
    ha, hb = he[0], he[-1]
    da = (he - ha).norm(dim=1); db = (he - hb).norm(dim=1)
    d = (da / (da + db + 1e-10)).numpy()
    # plateau boundary: max smoothed |d'|
    dp = np.abs(np.gradient(d))
    ker = np.ones(SMOOTH) / SMOOTH
    dps = np.convolve(dp, ker, mode='same')
    bnd = int(dps.argmax())
    # support: kNN radius to natural cloud along path
    dist, _ = nn_support.kneighbors(pts.numpy())
    rad = dist[:, -1]          # k-th neighbor distance
    nn1 = dist[:, 1]           # nearest (col0 may be self if endpoint in cloud)
    t = np.linspace(0, 1, N_POINTS)
    return dict(t=t, d=d, dps=dps, bnd=bnd, rad=rad, nn1=nn1,
                pred=lo.argmax(1).numpy())

# natural-point baseline kNN radius distribution (self excluded)
nat_dist, _ = nn_support.kneighbors(nat)
nat_rad = nat_dist[:, -1]      # each natural point's own k-th-NN radius
rad_ref_med = float(np.median(nat_rad))
rad_ref_p95 = float(np.percentile(nat_rad, 95))

paths = {
    'same_region_9': path_analysis(regionA[0], regionA[1] if len(regionA) > 1 else regionA[0]),
    'cross_region_9': path_analysis(medA, medB),
    'cross_digit_9v4': path_analysis(medA, torch.where((test_y == 4) & correct)[0][0].item()),
    'cross_digit_9v0': path_analysis(medA, torch.where((test_y == 0) & correct)[0][0].item()),
}

def pct_at_boundary(p):
    return float((nat_rad < p['rad'][p['bnd']]).mean() * 100)

print("\n=== direct-path test ===")
for name, p in paths.items():
    print(f"{name}: boundary t={p['t'][p['bnd']]:.3f}  rad@bnd={p['rad'][p['bnd']]:.3f} "
          f"(pctile {pct_at_boundary(p):.0f})  max_rad={p['rad'].max():.3f}@t={p['t'][p['rad'].argmax()]:.3f}")

# ---------------------------------------- mutual-kNN graph connectivity
def mutual_knn_components(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, nbr = nn.kneighbors(X)
    nbr = nbr[:, 1:]                      # drop self
    n = X.shape[0]
    rows, cols = [], []
    nbrset = [set(nbr[i]) for i in range(n)]
    for i in range(n):
        for j in nbr[i]:
            if i in nbrset[j]:            # mutual
                rows.append(i); cols.append(j)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)
    ncomp, comp = connected_components(A, directed=False)
    return comp, A

# node-set indices within the natural cloud (positions in `nat`)
pos = {int(g): i for i, g in enumerate(cidx.numpy())}
setA = np.array([pos[i] for i in regionA])
setB = np.array([pos[i] for i in regionB])
set4 = np.array([pos[i] for i in torch.where((test_y == 4) & correct)[0].numpy()])
set0 = np.array([pos[i] for i in torch.where((test_y == 0) & correct)[0].numpy()])

def merge_k(comp_fn, S1, S2, ks):
    for k in ks:
        comp, _ = comp_fn(k)
        c1 = set(comp[S1]); c2 = set(comp[S2])
        if c1 & c2:
            return k
    return None

KS = list(range(3, 41, 1))
comp_cache = {}
def comp_fn(k):
    if k not in comp_cache:
        comp_cache[k] = mutual_knn_components(nat, k)
    return comp_cache[k]

mk_AB = merge_k(comp_fn, setA, setB, KS)
mk_94 = merge_k(comp_fn, setA, set4, KS)
mk_90 = merge_k(comp_fn, setB, set0, KS)
# within-region internal connectivity scale: k at which >=90% of region in one comp
def internal_k(S, ks):
    for k in ks:
        comp, _ = comp_fn(k)
        cvals, counts = np.unique(comp[S], return_counts=True)
        if counts.max() >= 0.9 * len(S):
            return k
    return None
ik_A = internal_k(setA, KS)
ik_B = internal_k(setB, KS)

print("\n=== component test (mutual-kNN merge scale) ===")
print(f"within region A internal-connect k: {ik_A}")
print(f"within region B internal-connect k: {ik_B}")
print(f"region A <-> region B merge k: {mk_AB}")
print(f"region A(9) <-> digit 4 merge k: {mk_94}")
print(f"region B(9) <-> digit 0 merge k: {mk_90}")

# bottleneck: at AB merge k, min-max-edge (bottleneck) Euclidean path A->B
def bottleneck(S1, S2, k):
    comp, A = comp_fn(k)
    # weight adjacency by euclidean length
    Acoo = A.tocoo()
    w = np.linalg.norm(nat[Acoo.row] - nat[Acoo.col], axis=1)
    W = csr_matrix((w, (Acoo.row, Acoo.col)), shape=A.shape)
    # dijkstra from all S1, take min over S2 of path; approximate bottleneck by
    # max edge on the shortest (sum-weight) path
    dmat, pred_p = dijkstra(W, directed=False, indices=S1, return_predecessors=True)
    best = None
    for si, s in enumerate(S1):
        for t_ in S2:
            if np.isfinite(dmat[si, t_]):
                # trace path
                path = []
                cur = t_
                while cur != s and cur >= 0:
                    path.append(cur); cur = pred_p[si, cur]
                path.append(s)
                edges = [np.linalg.norm(nat[path[q]] - nat[path[q+1]]) for q in range(len(path)-1)]
                mx = max(edges) if edges else np.inf
                if best is None or mx < best:
                    best = mx
    return best

bn_AB = bottleneck(setA, setB, mk_AB) if mk_AB else None
print(f"A<->B bottleneck edge @k={mk_AB}: {bn_AB:.3f}  "
      f"(natural median kNN radius {rad_ref_med:.3f}, p95 {rad_ref_p95:.3f})")

# --- fixed-k discriminative comparison (merge_k saturates at k=3) ---
# At a within-region neighborhood scale K_FIX, compare geodesic hop distance and
# bottleneck edge for the three pair categories. Node endpoints = region medoids.
K_FIX = 10
idx4 = torch.where((test_y == 4) & correct)[0].numpy()
idx0 = torch.where((test_y == 0) & correct)[0].numpy()
med4 = medoid(idx4, L3[idx4].numpy(), L3[idx4].numpy().mean(0))
med0 = medoid(idx0, L3[idx0].numpy(), L3[idx0].numpy().mean(0))
pmedA, pmedB = pos[medA], pos[medB]
pmed4, pmed0 = pos[med4], pos[med0]
# a within-region A pair: medoid A vs farthest-downstream A node
pA2 = pos[regionA[int(np.linalg.norm(L3[regionA].numpy() - L3[medA].numpy(), axis=1).argmax())]]

def geo_and_bottleneck(s, t_, k):
    comp, A = comp_fn(k)
    Acoo = A.tocoo()
    w = np.linalg.norm(nat[Acoo.row] - nat[Acoo.col], axis=1)
    W = csr_matrix((w, (Acoo.row, Acoo.col)), shape=A.shape)
    # unweighted hop distance
    dh, predh = dijkstra(A, directed=False, indices=[s], return_predecessors=True)
    hops = dh[0, t_]
    # bottleneck edge along euclidean-shortest path
    dw, predw = dijkstra(W, directed=False, indices=[s], return_predecessors=True)
    path, cur = [], t_
    if np.isfinite(dw[0, t_]):
        while cur != s and cur >= 0:
            path.append(cur); cur = predw[0, cur]
        path.append(s)
        edges = [np.linalg.norm(nat[path[q]] - nat[path[q+1]]) for q in range(len(path)-1)]
        bn = max(edges) if edges else np.inf
    else:
        bn = np.inf
    return (None if not np.isfinite(hops) else int(hops)), float(bn)

fixed = {}
for nm, (s, t_) in {'within_A': (pmedA, pA2), 'A_vs_B_9regions': (pmedA, pmedB),
                    '9_vs_4': (pmedA, pmed4), '9_vs_0': (pmedB, pmed0)}.items():
    fixed[nm] = geo_and_bottleneck(s, t_, K_FIX)
print(f"\n=== fixed-k={K_FIX} geodesic hops / bottleneck edge ===")
for nm, (h, bn) in fixed.items():
    print(f"{nm}: hops={h}  bottleneck={bn:.3f}  (nat median rad {rad_ref_med:.3f})")

# ------------------------------------------------------------------ save
summary = dict(
    n_natural=int(nat.shape[0]), n_nine=int(nine_mask.sum()),
    region_sizes=[len(regionA), len(regionB)],
    rad_ref_med=rad_ref_med, rad_ref_p95=rad_ref_p95,
    boundary={n: dict(t=float(p['t'][p['bnd']]),
                      rad_at_bnd=float(p['rad'][p['bnd']]),
                      rad_pctile=pct_at_boundary(p),
                      max_rad=float(p['rad'].max()),
                      t_max_rad=float(p['t'][p['rad'].argmax()]))
              for n, p in paths.items()},
    internal_k=[ik_A, ik_B], merge_AB=mk_AB, merge_9v4=mk_94, merge_9v0=mk_90,
    bottleneck_AB=None if bn_AB is None else float(bn_AB),
    K_FIX=K_FIX, fixed={nm: dict(hops=h, bottleneck=bn) for nm, (h, bn) in fixed.items()},
)
with open(os.path.join(RES, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
torch.save(dict(paths=paths, summary=summary), os.path.join(RES, 'analysis.pt'))

# ------------------------------------------------------------------ plots
# Fig 1: direct-path d(t) + support radius for the 4 paths
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, (name, p) in zip(axes.flat, paths.items()):
    ax.plot(p['t'], p['d'], 'C0', lw=2, label='d(t) downstream')
    ax.axvline(p['t'][p['bnd']], color='k', ls='--', lw=1, label='plateau boundary')
    ax.set_ylim(-0.05, 1.15); ax.set_xlabel('t'); ax.set_ylabel('d(t)')
    ax2 = ax.twinx()
    ax2.plot(p['t'], p['rad'], 'C3', lw=1.4, alpha=0.8, label=f'kNN radius (k={KNN_SUPPORT})')
    ax2.axhline(rad_ref_med, color='C3', ls=':', lw=1, alpha=0.6)
    ax2.axhline(rad_ref_p95, color='C1', ls=':', lw=1, alpha=0.6)
    ax2.set_ylabel('support: kNN radius')
    ax.set_title(f"{name}  (boundary rad pctile {pct_at_boundary(p):.0f})")
    ax.grid(alpha=0.3)
    if name == 'same_region_9':
        ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='lower right', fontsize=8)
fig.suptitle('Direct-path test: downstream d(t) vs off-manifold support along slerp\n'
             'dotted C3=natural median kNN radius, dotted C1=natural p95', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT, 'direct_path_support.png'), dpi=140); plt.close()

# Fig 2: fixed-k geodesic hops + bottleneck edge per pair category
order = ['within_A', 'A_vs_B_9regions', '9_vs_4', '9_vs_0']
disp = ['within\nregion A', 'A<->B\n(9 regions)', '9 <-> 4', '9 <-> 0']
cols = ['C2', 'C0', 'C3', 'C3']
fig, (axh, axb) = plt.subplots(1, 2, figsize=(13, 5))
hops = [fixed[o][0] if fixed[o][0] is not None else 0 for o in order]
axh.bar(disp, hops, color=cols)
for i, h in enumerate(hops):
    axh.text(i, h + 0.1, str(fixed[order[i]][0]), ha='center', fontsize=10)
axh.set_ylabel(f'geodesic hop distance (mutual-kNN, k={K_FIX})')
axh.set_title('Graph distance between region medoids\n(lower = same component / closer)')
axh.grid(alpha=0.3, axis='y')
bns = [fixed[o][1] for o in order]
axb.bar(disp, bns, color=cols)
axb.axhline(rad_ref_med, color='k', ls='--', lw=1, label=f'natural median kNN radius {rad_ref_med:.2f}')
axb.axhline(rad_ref_p95, color='gray', ls=':', lw=1, label=f'natural p95 {rad_ref_p95:.2f}')
for i, bn in enumerate(bns):
    axb.text(i, bn + 0.03, f'{bn:.2f}', ha='center', fontsize=10)
axb.set_ylabel('bottleneck edge on connecting path')
axb.set_title('Support of the worst edge linking the two node-sets\n(below dashed line = connects through high-support region)')
axb.legend(fontsize=8); axb.grid(alpha=0.3, axis='y')
fig.suptitle(f'Component test at within-manifold scale k={K_FIX}: are the two digit-9 regions '
             f'their own component?', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT, 'component_test.png'), dpi=140); plt.close()

print("\nsaved plots + results")
print(json.dumps(summary, indent=2))
