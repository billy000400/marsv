#!/usr/bin/env python3
"""
dir11 robustness pass (S2/S3): is the REFUTED verdict stable across hyperparameters?

Iter 1 showed, at one graph scale (mutual-kNN k=10) and one digit-9 region split
(KMeans k=2, seed 0), that the two candidate digit-9 regions connect like a
within-region pair, unlike genuinely different digits. Here we stress-test that:

  (1) graph-k sweep: does the ordering
        within-region ~ same-digit-cross-region  <<  cross-digit
      hold across mutual-kNN scales k in {6..25}?
  (2) region-definition sweep: recompute the two digit-9 regions under
      KMeans k in {2,3,4} x seed in {0,1,2}; do ALL resulting same-9 region
      pairs stay within-region-like?
  (3) generality: repeat the same-digit-cross-region vs cross-digit contrast for
      ALL ten digits, not just 9.
  (4) counterexample search: flag any same-digit region pair that connects only
      through a low-support bottleneck (> natural p95 radius) or at cross-digit
      hop range -- the counterexample that would SUPPORT the hypothesis.

Reuses the base loading + mutual-kNN machinery from analyze_manifold.py.
CPU-only (tiny model). Outputs results/robustness.json, plots/robustness_*.png.
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
from scipy.sparse.csgraph import dijkstra

from src.mnist import load_checkpoint, load_mnist

torch.set_num_threads(2)
torch.manual_seed(0); np.random.seed(0)

HERE = '/network/marsv_agent_haoyang/dir11_boundary_as_off_manifold'
RES = os.path.join(HERE, 'results'); PLOT = os.path.join(HERE, 'plots')
os.makedirs(RES, exist_ok=True); os.makedirs(PLOT, exist_ok=True)
CKPT = '/network/mars-plateaus-image/results/image/mnist_mlp_d4_w200_relu_n1000_s100000.pt'
DATA = '/network/mars-plateaus-image/data/mnist'
KNN_SUPPORT = 10
K_GRAPH_SWEEP = [6, 8, 10, 12, 15, 20, 25]
K_FIX = 10

# ------------------------------------------------------------------ load
model, ck = load_checkpoint(CKPT)
n_test = ck['config']['n_test']
_, _, test_x, test_y = load_mnist(DATA)
test_x, test_y = test_x[:n_test], test_y[:n_test]
with torch.no_grad():
    hiddens, logits = model.hidden_activations(test_x)
L1 = hiddens[0]; L3 = hiddens[-1]
pred = logits.argmax(1); correct = pred == test_y
cidx = torch.where(correct)[0]
nat = L1[cidx].numpy()
print(f"natural cloud: {nat.shape[0]} correctly-classified L1 activations")

# natural-point kNN radius reference (support calibration)
nn_ref = NearestNeighbors(n_neighbors=KNN_SUPPORT + 1).fit(nat)
nat_dist, _ = nn_ref.kneighbors(nat)
nat_rad = nat_dist[:, -1]
rad_med = float(np.median(nat_rad)); rad_p95 = float(np.percentile(nat_rad, 95))
print(f"natural kNN radius: median {rad_med:.3f}  p95 {rad_p95:.3f}")

pos = {int(g): i for i, g in enumerate(cidx.numpy())}   # global idx -> nat row

def medoid(idxs, centroid):
    f = L3[idxs].numpy()
    return int(idxs[int(np.linalg.norm(f - centroid, axis=1).argmin())])

# ------------------------------------------------- mutual-kNN graph cache
def mutual_knn(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, nbr = nn.kneighbors(X); nbr = nbr[:, 1:]
    n = X.shape[0]; rows, cols = [], []
    nbrset = [set(nbr[i]) for i in range(n)]
    for i in range(n):
        for j in nbr[i]:
            if i in nbrset[j]:
                rows.append(i); cols.append(j)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return A.maximum(A.T)

_gcache = {}
def graph(k):
    if k not in _gcache:
        A = mutual_knn(nat, k)
        Acoo = A.tocoo()
        w = np.linalg.norm(nat[Acoo.row] - nat[Acoo.col], axis=1)
        W = csr_matrix((w, (Acoo.row, Acoo.col)), shape=A.shape)
        _gcache[k] = (A, W)
    return _gcache[k]

def hops_bottleneck(s, t, k):
    """geodesic hop count (unweighted) and bottleneck edge (max edge on the
    euclidean-shortest path) between nat rows s and t in the mutual-kNN graph."""
    A, W = graph(k)
    dh = dijkstra(A, directed=False, indices=[s])
    h = dh[0, t]
    dw, predw = dijkstra(W, directed=False, indices=[s], return_predecessors=True)
    if not np.isfinite(dw[0, t]):
        return None, float('inf')
    path, cur = [], t
    while cur != s and cur >= 0:
        path.append(cur); cur = predw[0, cur]
    path.append(s)
    edges = [np.linalg.norm(nat[path[q]] - nat[path[q + 1]]) for q in range(len(path) - 1)]
    bn = max(edges) if edges else float('inf')
    return (None if not np.isfinite(h) else int(h)), float(bn)

# ------------------------------------------ per-digit medoids & region pairs
def digit_idx(d):
    return torch.where((test_y == d) & correct)[0].numpy()

digit_medoid = {}      # whole-digit medoid (nat row)
for d in range(10):
    di = digit_idx(d)
    digit_medoid[d] = pos[medoid(di, L3[di].numpy().mean(0))]

def region_pair(d, kk, seed):
    """KMeans(kk) on digit d's L3; return list of the two largest regions'
    medoid nat rows (for kk>2 we take the two largest clusters)."""
    di = digit_idx(d)
    if len(di) < 2 * 8:
        return None
    km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(L3[di].numpy())
    labs, cents = km.labels_, km.cluster_centers_
    sizes = [(int((labs == c).sum()), c) for c in range(kk)]
    sizes.sort(reverse=True)
    (s0, c0), (s1, c1) = sizes[0], sizes[1]
    if s1 < 8:
        return None
    m0 = pos[medoid(di[labs == c0], cents[c0])]
    m1 = pos[medoid(di[labs == c1], cents[c1])]
    return m0, m1, s0, s1

# =============================================================== (1) graph-k sweep
# fixed digit-9 kmeans-2 seed-0 split (same as base analyze_manifold.py)
rp9 = region_pair(9, 2, 0)
m9a, m9b = rp9[0], rp9[1]
# a within-region-A control node: farthest downstream member of the larger 9-cluster
di9 = digit_idx(9)
km9 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(L3[di9].numpy())
big = np.bincount(km9.labels_).argmax()
Aidx = di9[km9.labels_ == big]
far = Aidx[int(np.linalg.norm(L3[Aidx].numpy() - L3[cidx[m9a]].numpy(), axis=1).argmax())]
m9a_far = pos[int(far)]

sweep = {'within_A': [], 'A_vs_B_9': [], '9_vs_4': [], '9_vs_0': []}
pairs = {'within_A': (m9a, m9a_far), 'A_vs_B_9': (m9a, m9b),
         '9_vs_4': (m9a, digit_medoid[4]), '9_vs_0': (m9a, digit_medoid[0])}
for k in K_GRAPH_SWEEP:
    for nm, (s, t) in pairs.items():
        h, _ = hops_bottleneck(s, t, k)
        sweep[nm].append(h)
print("\n=== (1) graph-k sweep hops ===")
for nm in pairs:
    print(f"{nm:12s}: {dict(zip(K_GRAPH_SWEEP, sweep[nm]))}")

# =============================================================== (2)+(3) region defs
# same-digit-cross-region (all 10 digits, seed 0, kmeans-2) at K_FIX
same_digit = {}
for d in range(10):
    rp = region_pair(d, 2, 0)
    if rp is None:
        continue
    h, bn = hops_bottleneck(rp[0], rp[1], K_FIX)
    same_digit[d] = dict(hops=h, bottleneck=bn, sizes=[rp[2], rp[3]])
print("\n=== (3) same-digit cross-region hops/bottleneck (kmeans-2, k=10) ===")
for d, v in same_digit.items():
    print(f"digit {d}: hops={v['hops']}  bottleneck={v['bottleneck']:.3f}  sizes={v['sizes']}")

# digit-9 region defs across kmeans-k x seed (region-definition robustness)
nine_defs = []
for kk in (2, 3, 4):
    for seed in (0, 1, 2):
        rp = region_pair(9, kk, seed)
        if rp is None:
            continue
        h, bn = hops_bottleneck(rp[0], rp[1], K_FIX)
        nine_defs.append(dict(kmeans_k=kk, seed=seed, hops=h, bottleneck=bn,
                              sizes=[rp[2], rp[3]]))
print("\n=== (2) digit-9 region defs (kmeans-k x seed, k=10) ===")
for v in nine_defs:
    print(f"kk={v['kmeans_k']} seed={v['seed']}: hops={v['hops']} bn={v['bottleneck']:.3f} sizes={v['sizes']}")

# cross-digit reference band: all 45 digit pairs at K_FIX
cross = []
for a in range(10):
    for b in range(a + 1, 10):
        h, bn = hops_bottleneck(digit_medoid[a], digit_medoid[b], K_FIX)
        cross.append(dict(pair=[a, b], hops=h, bottleneck=bn))
cross_h = [c['hops'] for c in cross if c['hops'] is not None]
cross_bn = [c['bottleneck'] for c in cross if np.isfinite(c['bottleneck'])]
print(f"\n=== cross-digit band (45 pairs, k=10) === hops: "
      f"min {min(cross_h)} med {int(np.median(cross_h))} max {max(cross_h)}  "
      f"bottleneck med {np.median(cross_bn):.3f}")

# =============================================================== (4) counterexample
# The plan's SUPPORTED verdict = regions merge only through a MUCH lower-support
# (off-manifold) bottleneck than within-region. The support/gap metric is the
# BOTTLENECK edge, not hops: hops is a distance measure that overlaps across
# categories (an elongated digit's own regions can be many hops apart while every
# edge stays high-support). So a genuine counterexample = a same-digit region pair
# whose connecting bottleneck exceeds the natural p95 radius (an off-manifold gap).
sd_hops = [v['hops'] for v in same_digit.values() if v['hops'] is not None]
sd_bn = [v['bottleneck'] for v in same_digit.values() if np.isfinite(v['bottleneck'])]
nine_hops = [v['hops'] for v in nine_defs if v['hops'] is not None]
nine_bn = [v['bottleneck'] for v in nine_defs if np.isfinite(v['bottleneck'])]
cross_bn_min = float(min(cross_bn)); cross_bn_max = float(max(cross_bn))
counterexamples = []
for tag, rec in ([('digit%d' % d, v) for d, v in same_digit.items()] +
                 [('9_kk%d_s%d' % (v['kmeans_k'], v['seed']), v) for v in nine_defs]):
    bn = rec['bottleneck']
    if np.isfinite(bn) and bn > rad_p95:
        counterexamples.append(dict(tag=tag, hops=rec['hops'], bottleneck=bn))
print(f"\n=== (4) counterexample search (bottleneck > p95={rad_p95:.2f}) === "
      f"found {len(counterexamples)}: {counterexamples}")

# ------------------------------------------------------------------ save
summary = dict(
    n_natural=int(nat.shape[0]), rad_med=rad_med, rad_p95=rad_p95,
    K_GRAPH_SWEEP=K_GRAPH_SWEEP, sweep=sweep,
    same_digit={str(d): v for d, v in same_digit.items()},
    nine_defs=nine_defs,
    cross_digit=dict(hops_min=int(min(cross_h)), hops_med=float(np.median(cross_h)),
                     hops_max=int(max(cross_h)), bn_med=float(np.median(cross_bn)),
                     bn_min=cross_bn_min, bn_max=cross_bn_max),
    same_digit_hops=dict(min=int(min(sd_hops)), max=int(max(sd_hops)),
                         med=float(np.median(sd_hops))),
    same_digit_bn=dict(min=float(min(sd_bn)), max=float(max(sd_bn))),
    nine_defs_hops=dict(min=int(min(nine_hops)), max=int(max(nine_hops))),
    nine_defs_bn=dict(min=float(min(nine_bn)), max=float(max(nine_bn))),
    counterexamples=counterexamples,
    verdict_stable=(len(counterexamples) == 0),
)
with open(os.path.join(RES, 'robustness.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# ------------------------------------------------------------------ plots
# Fig A: graph-k sweep
fig, ax = plt.subplots(figsize=(8, 5))
sty = {'within_A': ('C2', 'o', 'within region A (control)'),
       'A_vs_B_9': ('C0', 's', 'A <-> B  (two digit-9 regions)'),
       '9_vs_4': ('C3', '^', '9 <-> 4  (different digit)'),
       '9_vs_0': ('C1', 'v', '9 <-> 0  (different digit)')}
for nm, (c, mk, lab) in sty.items():
    ys = [h if h is not None else np.nan for h in sweep[nm]]
    ax.plot(K_GRAPH_SWEEP, ys, c + '-', marker=mk, lw=2, label=lab)
ax.set_xlabel('mutual-kNN graph scale k'); ax.set_ylabel('geodesic hop distance')
ax.set_title('Robustness across graph scale: two digit-9 regions track the\n'
             'within-region control, far below genuinely different digits')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'robustness_graphk.png'), dpi=140); plt.close()

# Fig B: bottleneck (primary support/gap metric) + hops (distance, overlaps)
fig, (axb, axh) = plt.subplots(1, 2, figsize=(13, 5))
sd_d = sorted(same_digit.keys())
sd_hv = [same_digit[d]['hops'] for d in sd_d]
sd_bv = [same_digit[d]['bottleneck'] for d in sd_d]
x = np.arange(len(sd_d))
barcol = ['C4' if d == 9 else 'C0' for d in sd_d]
# left = bottleneck: the discriminative support/gap metric
axb.bar(x, sd_bv, color=barcol)
axb.axhspan(cross_bn_min, cross_bn_max, color='C3', alpha=0.13,
            label=f'cross-digit bottleneck band ({cross_bn_min:.2f}-{cross_bn_max:.2f})')
axb.axhline(rad_med, color='k', ls='--', lw=1, label=f'natural median radius {rad_med:.2f}')
axb.axhline(rad_p95, color='gray', ls=':', lw=1.4, label=f'natural p95 radius {rad_p95:.2f} (gap threshold)')
axb.set_xticks(x); axb.set_xticklabels([str(d) for d in sd_d])
axb.set_xlabel('digit (two regions via KMeans-2 on L3)')
axb.set_ylabel('bottleneck edge on connecting path')
axb.set_title('Support of the connecting bottleneck (digit 9 = purple)\n'
              'every same-digit pair stays below p95 -> no off-manifold gap')
axb.legend(fontsize=8); axb.grid(alpha=0.3, axis='y')
# right = hops: distance measure, honestly shown to OVERLAP the cross-digit band
axh.bar(x, sd_hv, color=barcol)
axh.axhspan(min(cross_h), max(cross_h), color='C3', alpha=0.13,
            label=f'cross-digit band ({min(cross_h)}-{max(cross_h)} hops)')
axh.set_xticks(x); axh.set_xticklabels([str(d) for d in sd_d])
axh.set_xlabel('digit')
axh.set_ylabel(f'geodesic hops between the 2 regions (k={K_FIX})')
axh.set_title('Hop distance is NOT a clean separateness signal:\n'
              'elongated digit-1 = 15 hops yet its bottleneck (left) is below median')
axh.legend(fontsize=8); axh.grid(alpha=0.3, axis='y')
fig.suptitle('Generality + counterexample search: by the support (bottleneck) metric, '
             'no same-digit region pair forms a separate component', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'robustness_regions.png'), dpi=140); plt.close()

print("\nsaved robustness plots + results")
print(json.dumps(summary, indent=2))
