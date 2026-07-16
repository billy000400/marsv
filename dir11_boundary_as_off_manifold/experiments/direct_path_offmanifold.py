#!/usr/bin/env python3
"""
dir11 — Investigation 2 (operator feedback 2026-07-16): the low-density corridor.

The population analysis (population_manifold.py) answered the component question:
plateau pairs are NOT separate empirical manifold components (G ~= 1). The operator
feedback asks to ALSO keep the complementary finding as its own investigation: the
DIRECT slerp path between two plateaus passes through a low-density region that real
activations do not occupy — even though a high-support natural detour exists.

Frozen conventions (identical to population_manifold.py):
  * model  : base mnist_mlp_d4_w200_relu (784->200->200->200->10, MSE-to-one-hot).
  * interpolation layer : FIRST hidden layer L1 (200-dim, post-ReLU) — slerp lives here.
  * d(t) measurement layer : LAST hidden layer L3 (200-dim).
  * support measurement layer : L1 (same layer as the interpolation; distance from the
    interpolated L1 point to the natural L1 cloud).
  * natural cloud : L1 activations of the 1705 correctly-classified test images.
  * regions : 10 confident digit classes (margin >= 0.5) + digit-9 A/B sub-plateau.
  * sampling : 20 endpoint pairs per region pair, seed 0; d(t) accept filter for
    between-region paths (plateau fraction >= 0.5, start < 0.2, end > 0.8).

Support observable (per path point x_t): r_k(x_t) = distance to the k=10-th nearest
neighbor in the natural cloud. Baseline = each natural point's own r_k (self excluded);
median 2.85 / p95 4.23 on the base model. Off-manifold excursion of a path =
max_t r_k(x_t) expressed as a percentile of the natural baseline.

Outputs:
  plots/direct_path_support.png      — annotated 4-panel digit-9 illustration
                                       (regenerates the iter-1 figure the operator liked,
                                       now with layers + sample sizes written on it).
  plots/direct_path_population.png   — population-level comparison: between-plateau
                                       verified paths vs within-plateau control paths.
  results/direct_path.json
"""
import os, sys, json, itertools
BASE = '/network/mars-plateaus-image' if os.path.isdir('/network/mars-plateaus-image') \
    else '/workspace/mars-plateaus-image'
sys.path.insert(0, BASE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from src.mnist import load_checkpoint, load_mnist

torch.set_num_threads(2)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results'); PLOT = os.path.join(HERE, 'plots')
DATA = os.path.join(BASE, 'data/mnist')
CKPT = os.path.join(BASE, 'results/image/mnist_mlp_d4_w200_relu_n1000_s100000.pt')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population_manifold import slerp_path, MARGIN, N_PAIRS, PLATEAU_FRAC

K_SUPPORT = 10        # k for the support radius r_k
N_POINTS_ILLU = 200   # slerp resolution for the 4-panel illustration (as in iter 1)
N_POINTS_POP = 120    # slerp resolution for the population sweep (as in population run)
SAMPLE_SEED = 0

# ------------------------------------------------------------------ load
model, ck = load_checkpoint(CKPT)
model = model.to('cpu')
n_test = ck['config']['n_test']
_, _, test_x, test_y = load_mnist(DATA)
test_x, test_y = test_x[:n_test], test_y[:n_test]
with torch.no_grad():
    hiddens, logits = model.hidden_activations(test_x)
L1, L3 = hiddens[0], hiddens[-1]
pred = logits.argmax(1); correct = pred == test_y
top2 = logits.topk(2, 1).values
conf = correct & ((top2[:, 0] - top2[:, 1]) >= MARGIN)
cidx = torch.where(correct)[0]
nat = L1[cidx].numpy().astype(np.float64)
Nc = nat.shape[0]
pos = {int(g): i for i, g in enumerate(cidx.numpy())}
cloud_global = cidx.numpy()
print(f"natural cloud: {Nc} correct L1 activations; confident endpoints: {int(conf.sum())}")

# ------------------------------------------------- support radius machinery
nn_q = NearestNeighbors(n_neighbors=K_SUPPORT).fit(nat)        # for path points
nn_b = NearestNeighbors(n_neighbors=K_SUPPORT + 1).fit(nat)    # for natural points (excl self)
nat_rad = nn_b.kneighbors(nat)[0][:, -1]                       # each point's own k-th-NN radius
nat_med = float(np.median(nat_rad)); nat_p95 = float(np.percentile(nat_rad, 95))
nat_sorted = np.sort(nat_rad)
print(f"natural r_{K_SUPPORT}: median {nat_med:.3f}, p95 {nat_p95:.3f}")

def path_radii(pts):
    return nn_q.kneighbors(pts.numpy())[0][:, -1]

def pctile(r):
    """percentile of r within the natural baseline distribution"""
    return float(np.searchsorted(nat_sorted, r) / len(nat_sorted) * 100)

def d_of_t(ga, gb, n_points):
    pts = slerp_path(L1[ga], L1[gb], n_points)
    with torch.no_grad():
        _, hs = model.forward_from(pts, 1)
    he = hs[-1]
    da = (he - he[0]).norm(dim=1); db = (he - he[-1]).norm(dim=1)
    return (da / (da + db + 1e-10)).numpy(), pts

def plateau_ok(d):
    n = len(d); early = d[: n // 5]; late = d[-n // 5:]
    frac = float(((d < 0.2) | (d > 0.8)).mean())
    return (early.min() < 0.2) and (late.max() > 0.8) and (frac >= PLATEAU_FRAC)

# =============================================================== Figure 1:
# annotated 4-panel digit-9 illustration (regenerate iter-1 figure, same endpoints)
nine_idx = torch.where((test_y == 9) & correct)[0]
nine_L3 = L3[nine_idx].numpy()
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(nine_L3)
lab9 = km.labels_
def medoid(idxs, feats, centroid):
    return idxs[int(np.linalg.norm(feats - centroid, axis=1).argmin())]
regionA = nine_idx[lab9 == 0].numpy(); regionB = nine_idx[lab9 == 1].numpy()
medA = medoid(regionA, L3[regionA].numpy(), km.cluster_centers_[0])
medB = medoid(regionB, L3[regionB].numpy(), km.cluster_centers_[1])

illu = {}
pairs_illu = {
    'same-region 9→9': (int(regionA[0]), int(regionA[1])),
    'cross-region 9A→9B': (int(medA), int(medB)),
    'cross-digit 9→4': (int(medA), int(torch.where((test_y == 4) & correct)[0][0])),
    'cross-digit 9→0': (int(medA), int(torch.where((test_y == 0) & correct)[0][0])),
}
for name, (ga, gb) in pairs_illu.items():
    d, pts = d_of_t(ga, gb, N_POINTS_ILLU)
    rad = path_radii(pts)
    illu[name] = dict(d=d, rad=rad, max_pct=pctile(rad.max()))

t_illu = np.linspace(0, 1, N_POINTS_ILLU)
fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
for ax, (name, p) in zip(axes.flat, illu.items()):
    ax.plot(t_illu, p['d'], 'C0', lw=2, label='d(t)  [measured at L3]')
    ax.set_ylim(-0.05, 1.15); ax.set_xlabel('t (slerp position in L1)'); ax.set_ylabel('d(t)')
    ax2 = ax.twinx()
    ax2.plot(t_illu, p['rad'], 'C3', lw=1.4, alpha=0.85,
             label=f'support radius $r_{{{K_SUPPORT}}}$(t)  [measured at L1]')
    ax2.axhline(nat_med, color='C3', ls=':', lw=1, alpha=0.7)
    ax2.axhline(nat_p95, color='C1', ls='--', lw=1.2, alpha=0.9)
    ax2.set_ylabel(f'$r_{{{K_SUPPORT}}}$: dist to {K_SUPPORT}th-NN in natural L1 cloud')
    ax.set_title(f"{name}   (max $r_{{{K_SUPPORT}}}$ = {p['rad'].max():.2f} "
                 f"= {p['max_pct']:.0f}th pctile of natural)")
    ax.grid(alpha=0.3)
    if name.startswith('same'):
        ax.legend(loc='center left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
fig.suptitle(
    'Direct-path test — downstream distance d(t) vs off-manifold support along the slerp path\n'
    f'Interpolation: spherical (slerp) in the FIRST hidden layer L1 (200-d, post-ReLU); d(t) at the LAST hidden layer L3\n'
    f'support $r_{{{K_SUPPORT}}}$ at L1 vs the {Nc}-pt natural cloud; {N_POINTS_ILLU} pts/path, '
    f'1 endpoint pair per panel (region medoids)\n'
    f'dotted red = natural median $r_{{{K_SUPPORT}}}$ ({nat_med:.2f}), dashed orange = natural p95 ({nat_p95:.2f})',
    fontsize=9.5)
plt.tight_layout(rect=[0, 0, 1, 0.885])
plt.savefig(os.path.join(PLOT, 'direct_path_support.png'), dpi=140); plt.close()
print("saved annotated direct_path_support.png")
for name, p in illu.items():
    print(f"  {name}: max r = {p['rad'].max():.3f} ({p['max_pct']:.0f}th pctile)")

# =============================================================== Population sweep:
# excursion of verified between-plateau paths vs within-plateau control paths
rng = np.random.RandomState(SAMPLE_SEED)
def sample_pairs(rA, rB, n, same):
    out = []
    for _ in range(n * 6):
        a = int(rA[rng.randint(len(rA))]); b = int(rB[rng.randint(len(rB))])
        if same and a == b:
            continue
        out.append((a, b))
        if len(out) == n:
            break
    return out

def rows(mask):
    return np.array([pos[int(i)] for i in torch.where(mask)[0].numpy()])
regions = {str(d): rows(conf & (test_y == d)) for d in range(10)}
nine_g = torch.where(conf & (test_y == 9))[0].numpy()
lab9c = KMeans(n_clusters=2, n_init=10, random_state=0).fit(L3[nine_g].numpy()).labels_
if (lab9c == 0).sum() < (lab9c == 1).sum():
    lab9c = 1 - lab9c
sub9A = np.array([pos[int(i)] for i in nine_g[lab9c == 0]])
sub9B = np.array([pos[int(i)] for i in nine_g[lab9c == 1]])

t_pop = np.linspace(0, 1, N_POINTS_POP)
def eval_paths(rI, rJ, same):
    """returns per-path (accepted, max_pct, radius profile normalized by nat median)"""
    recs = []
    for a, b in sample_pairs(rI, rJ, N_PAIRS, same):
        ga, gb = int(cloud_global[a]), int(cloud_global[b])
        d, pts = d_of_t(ga, gb, N_POINTS_POP)
        ok = plateau_ok(d) if not same else True
        rad = path_radii(pts)
        recs.append(dict(ok=ok, max_pct=pctile(rad.max()),
                         above_p95=bool(rad.max() > nat_p95),
                         prof=(rad / nat_med).tolist()))
    return recs

between, within = [], []
for di, dj in itertools.combinations([str(d) for d in range(10)], 2):
    between.extend(eval_paths(regions[di], regions[dj], False))
between.extend(eval_paths(sub9A, sub9B, False))          # digit-9 sub-plateau, same rules
for d in range(10):
    within.extend(eval_paths(regions[str(d)], regions[str(d)], True))

btw_ok = [r for r in between if r['ok']]
b_pct = np.array([r['max_pct'] for r in btw_ok])
w_pct = np.array([r['max_pct'] for r in within])
b_above = float(np.mean([r['above_p95'] for r in btw_ok]))
w_above = float(np.mean([r['above_p95'] for r in within]))
b_prof = np.array([r['prof'] for r in btw_ok])
w_prof = np.array([r['prof'] for r in within])
print(f"\nverified between-plateau paths: {len(btw_ok)}/{len(between)}  "
      f"median max-r pctile {np.median(b_pct):.1f}, frac > p95: {b_above:.2f}")
print(f"within-plateau control paths:  {len(within)}  "
      f"median max-r pctile {np.median(w_pct):.1f}, frac > p95: {w_above:.2f}")

json.dump(dict(
    k=K_SUPPORT, n_natural=Nc, nat_med=nat_med, nat_p95=nat_p95,
    n_points_pop=N_POINTS_POP, n_pairs_per_region_pair=N_PAIRS, sample_seed=SAMPLE_SEED,
    illustration={n: dict(max_r=float(p['rad'].max()), max_pctile=p['max_pct'],
                          n_points=N_POINTS_ILLU) for n, p in illu.items()},
    between=dict(n_paths_sampled=len(between), n_verified=len(btw_ok),
                 median_max_pctile=float(np.median(b_pct)),
                 iqr_max_pctile=[float(np.percentile(b_pct, 25)), float(np.percentile(b_pct, 75))],
                 frac_above_p95=b_above),
    within=dict(n_paths=len(within), median_max_pctile=float(np.median(w_pct)),
                iqr_max_pctile=[float(np.percentile(w_pct, 25)), float(np.percentile(w_pct, 75))],
                frac_above_p95=w_above),
), open(os.path.join(RES, 'direct_path.json'), 'w'), indent=2)

# ---- population figure ----
fig, (axa, axb) = plt.subplots(1, 2, figsize=(14, 5.2))
bins = np.linspace(0, 100, 26)
axa.hist(w_pct, bins=bins, alpha=0.6, color='C2', density=True,
         label=f'within-plateau controls (n={len(within)})')
axa.hist(b_pct, bins=bins, alpha=0.6, color='C3', density=True,
         label=f'verified between-plateau paths (n={len(btw_ok)})')
axa.axvline(95, color='k', ls='--', lw=1.2, label='natural p95')
axa.set_xlabel(f'off-manifold excursion: max $r_{{{K_SUPPORT}}}$ along path, '
               'percentile of natural baseline')
axa.set_ylabel('density')
axa.set_title(f'(a) Direct between-plateau paths reach lower-density territory\n'
              f'median pctile {np.median(b_pct):.0f} vs {np.median(w_pct):.0f} within; '
              f'{b_above*100:.0f}% vs {w_above*100:.0f}% exceed the natural p95')
axa.legend(fontsize=9); axa.grid(alpha=0.3)

for prof, c, lab in ((w_prof, 'C2', 'within-plateau controls'),
                     (b_prof, 'C3', 'between-plateau verified')):
    med = np.median(prof, axis=0)
    lo, hi = np.percentile(prof, 25, axis=0), np.percentile(prof, 75, axis=0)
    axb.plot(t_pop, med, color=c, lw=2, label=f'{lab} (median)')
    axb.fill_between(t_pop, lo, hi, color=c, alpha=0.25)
axb.axhline(1.0, color='k', ls=':', lw=1, label='natural median $r_{10}$')
axb.axhline(nat_p95 / nat_med, color='k', ls='--', lw=1, label='natural p95')
axb.set_xlabel('t (slerp position in L1)')
axb.set_ylabel(f'$r_{{{K_SUPPORT}}}$(t) / natural median')
axb.set_title('(b) Support-radius profile along the path (median, IQR band)\n'
              'between-plateau paths bulge mid-path: a low-density corridor')
axb.legend(fontsize=8); axb.grid(alpha=0.3)
fig.suptitle(f'Population direct-path test — slerp in L1, support $r_{{{K_SUPPORT}}}$ at L1 vs the '
             f'{Nc}-pt natural cloud, d(t) filter at L3; {N_PAIRS} endpoint pairs per region pair '
             f'(46 region pairs), {N_POINTS_POP} pts/path, seed {SAMPLE_SEED}', fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(PLOT, 'direct_path_population.png'), dpi=140); plt.close()
print("saved direct_path_population.png + results/direct_path.json")
