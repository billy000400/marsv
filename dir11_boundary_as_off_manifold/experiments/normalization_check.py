#!/usr/bin/env python3
"""
dir11 — operator feedback (human_feedback_2): why normalize G by max(s_i, s_j)?
Is it a true reflection at the boundary?

Two empirical answers, base model, frozen seed-0 sampling (identical rng stream to
population_manifold.analyze, so the max-variant doubles as a regression check):

  1. Normalization sensitivity: recompute every per-pair G under four denominators
       max(s_i, s_j)   (frozen choice)
       min(s_i, s_j)
       mean(s_i, s_j)
       s_global        (median over ALL pooled within-region bottlenecks)
     and report the verdict quantities (between-plateau median G + CI, counterexample
     count, digit-9 sub G, %pairs>1) for each. Within-plateau baseline: for max/min/
     mean the two regions coincide (s_i = s_j), so the control distribution is the
     SAME under all three; only s_global changes it.

  2. Boundary diagnostic: for every verified between-plateau path, locate the actual
     bottleneck EDGE (the argmax edge on the MST path) and check the digit labels of
     its two endpoints — is the forced biggest hop a genuine boundary crossing
     (different classes) or an ordinary hop inside one region's cloud?

Outputs results/normalization_check.json, plots/normalization_check.png.
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
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans

from src.mnist import load_checkpoint, load_mnist

torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.225)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results'); PLOT = os.path.join(HERE, 'plots')
DATA = os.path.join(BASE, 'data/mnist')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population_manifold import slerp_path, MARGIN, N_PAIRS, N_POINTS, PLATEAU_FRAC

# ---------------------------------------------------------------- load base model
ckpt = os.path.join(BASE, 'results/image/mnist_mlp_d4_w200_relu_n1000_s100000.pt')
model, ck = load_checkpoint(ckpt); model = model.to('cpu')
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
cloud_label = test_y[cidx].numpy()          # digit class of each natural cloud point

# ---------------------------------------------------------------- MST + bottleneck WITH edge tracking
D = cdist(nat, nat)
mst = minimum_spanning_tree(D).toarray(); mst = np.maximum(mst, mst.T)
adj = [[] for _ in range(Nc)]
ii, jj = np.where(mst > 0)
for a, b in zip(ii, jj):
    if a < b:
        w = float(mst[a, b]); adj[a].append((b, w)); adj[b].append((a, w))
cache = {}
def maxedge_from(src):
    """me[v] = largest edge on MST path src->v; (ep,eq)[v] = that edge's endpoints."""
    if src in cache:
        return cache[src]
    me = np.full(Nc, np.nan); me[src] = 0.0
    ep = np.full(Nc, -1, dtype=np.int32); eq = np.full(Nc, -1, dtype=np.int32)
    stack = [src]
    while stack:
        u = stack.pop(); mu = me[u]
        for v, w in adj[u]:
            if np.isnan(me[v]):
                if w > mu:
                    me[v] = w; ep[v] = u; eq[v] = v
                else:
                    me[v] = mu; ep[v] = ep[u]; eq[v] = eq[u]
                stack.append(v)
    cache[src] = (me, ep, eq); return cache[src]
def bottleneck(u, v):
    me, ep, eq = maxedge_from(u)
    return float(me[v]), int(ep[v]), int(eq[v])

# ---------------------------------------------------------------- regions (identical to frozen pipeline)
def rows(mask):
    return np.array([pos[int(i)] for i in torch.where(mask)[0].numpy()])
regions = {str(d): rows(conf & (test_y == d)) for d in range(10)}

nine_g = torch.where(conf & (test_y == 9))[0].numpy()
lab9 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(L3[nine_g].numpy()).labels_
if (lab9 == 0).sum() < (lab9 == 1).sum():
    lab9 = 1 - lab9
sub9A = np.array([pos[int(i)] for i in nine_g[lab9 == 0]])
sub9B = np.array([pos[int(i)] for i in nine_g[lab9 == 1]])

rng = np.random.RandomState(0)   # SAME stream/order as population_manifold.analyze(seed 0)
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

def d_of_t(a_row, b_row):
    ga, gb = int(cloud_global[a_row]), int(cloud_global[b_row])
    pts = slerp_path(L1[ga], L1[gb], N_POINTS)
    with torch.no_grad():
        _, hs = model.forward_from(pts, 1)
    he = hs[-1]
    da = (he - he[0]).norm(dim=1); db = (he - he[-1]).norm(dim=1)
    return (da / (da + db + 1e-10)).numpy()

def plateau_ok(d):
    n = len(d); early = d[: n // 5]; late = d[-n // 5:]
    frac = float(((d < 0.2) | (d > 0.8)).mean())
    return (early.min() < 0.2) and (late.max() > 0.8) and (frac >= PLATEAU_FRAC), frac

# ---- within-plateau scale (same rng order: regions '0'..'9') ----
within_scale = {}; within_B = {}
within_edge_cross = []   # is the within-pair bottleneck edge cross-class?
for k, rr in regions.items():
    bs = []
    for a, b in sample_pairs(rr, rr, N_PAIRS, True):
        B, p, q = bottleneck(a, b)
        bs.append(B)
        within_edge_cross.append(bool(cloud_label[p] != cloud_label[q]))
    within_B[k] = np.array(bs)
    within_scale[k] = float(np.median(bs))
s_global = float(np.median(np.concatenate(list(within_B.values()))))

def sub_scale(rr):
    return float(np.median([bottleneck(a, b)[0] for a, b in sample_pairs(rr, rr, N_PAIRS, True)]))

# ---- between-plateau pairs (same rng order: combinations, then 9A-9B) ----
def eval_pair(name, rI, rJ, si, sj, cross_digit):
    Bs = []; accept = 0; edge_cross = []
    for a, b in sample_pairs(rI, rJ, N_PAIRS, False):
        d = d_of_t(a, b); ok, _ = plateau_ok(d)
        B, p, q = bottleneck(a, b)
        if ok:
            accept += 1; Bs.append(B)
            if cross_digit:
                edge_cross.append(bool(cloud_label[p] != cloud_label[q]))
    return dict(name=name, si=si, sj=sj, n_verified=accept,
                B_median=float(np.median(Bs)) if Bs else None,
                edge_cross=edge_cross)

pairs = []
for di, dj in itertools.combinations([str(d) for d in range(10)], 2):
    pairs.append(eval_pair(f"{di}-{dj}", regions[di], regions[dj],
                           within_scale[di], within_scale[dj], True))
s9A, s9B = sub_scale(sub9A), sub_scale(sub9B)
pairs.append(eval_pair("9A-9B", sub9A, sub9B, s9A, s9B, False))

verified = [r for r in pairs if r['n_verified'] >= 5 and r['B_median'] is not None]

# ---------------------------------------------------------------- G under 4 normalizations
DENOMS = {
    'max(si,sj) [frozen]': lambda si, sj: max(si, sj),
    'min(si,sj)':          lambda si, sj: min(si, sj),
    'mean(si,sj)':         lambda si, sj: 0.5 * (si + sj),
    'global s':            lambda si, sj: s_global,
}
boot_rng = np.random.RandomState(12345)
def boot_ci(x, n=2000):
    x = np.asarray(x)
    idx = boot_rng.randint(0, len(x), size=(n, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

variants = {}
for vname, fn in DENOMS.items():
    meds = np.array([r['B_median'] / fn(r['si'], r['sj']) for r in verified])
    names = [r['name'] for r in verified]
    d9 = next((m for n, m in zip(names, meds) if n == '9A-9B'), None)
    # within-plateau distribution under this variant (si = sj within a region)
    wG = np.concatenate([within_B[k] / fn(within_scale[k], within_scale[k])
                         for k in regions])
    variants[vname] = dict(
        per_pair={n: float(m) for n, m in zip(names, meds)},
        between_median=float(np.median(meds)), between_CI=boot_ci(meds),
        n_counterexamples=int((meds <= 1.0).sum()), n_verified=len(meds),
        frac_gt_1=float((meds > 1.0).mean()), digit9_sub=float(d9),
        within_median=float(np.median(wG)), within_p95=float(np.percentile(wG, 95)),
        within_CI=boot_ci(wG),
    )
    print(f"[{vname:22s}] between med={variants[vname]['between_median']:.3f} "
          f"CI={variants[vname]['between_CI']} cx={variants[vname]['n_counterexamples']}"
          f"/{len(meds)} frac>1={variants[vname]['frac_gt_1']:.2f} 9A-9B={d9:.2f} "
          f"within med={variants[vname]['within_median']:.3f} p95={variants[vname]['within_p95']:.3f}",
          flush=True)

# regression check vs published frozen numbers
fro = variants['max(si,sj) [frozen]']
print(f"\n[regression] frozen-max between med {fro['between_median']:.3f} (published 0.996), "
      f"cx {fro['n_counterexamples']}/{fro['n_verified']} (published 25/45), "
      f"9A-9B {fro['digit9_sub']:.2f} (published 1.00)", flush=True)

# ---------------------------------------------------------------- boundary-edge diagnostic
btw_cross = np.concatenate([np.array(r['edge_cross'], dtype=bool) for r in verified
                            if r['name'] != '9A-9B' and len(r['edge_cross'])])
within_edge_cross = np.array(within_edge_cross, dtype=bool)
edge_diag = dict(
    between_frac_cross_class=float(btw_cross.mean()), n_between_paths=int(len(btw_cross)),
    within_frac_cross_class=float(within_edge_cross.mean()), n_within_paths=int(len(within_edge_cross)),
)
print(f"[edge diag] bottleneck edge joins two DIFFERENT digits' points on "
      f"{edge_diag['between_frac_cross_class']*100:.0f}% of {edge_diag['n_between_paths']} verified "
      f"between-plateau paths (within controls: {edge_diag['within_frac_cross_class']*100:.0f}%)", flush=True)

# ---------------------------------------------------------------- persist
with open(os.path.join(RES, 'normalization_check.json'), 'w') as f:
    json.dump(dict(s_global=s_global, within_scale=within_scale, s9A=s9A, s9B=s9B,
                   variants=variants, edge_diagnostic=edge_diag), f, indent=2)

# ---------------------------------------------------------------- figure
fig, (axa, axb) = plt.subplots(1, 2, figsize=(14, 5.2))

x = np.arange(len(DENOMS)); vnames = list(DENOMS)
bm = [variants[v]['between_median'] for v in vnames]
lo = [variants[v]['between_CI'][0] for v in vnames]
hi = [variants[v]['between_CI'][1] for v in vnames]
wm = [variants[v]['within_median'] for v in vnames]
wp = [variants[v]['within_p95'] for v in vnames]
axa.errorbar(x, bm, yerr=[np.array(bm) - lo, np.array(hi) - bm], fmt='o', color='C3',
             capsize=4, ms=7, label='between-plateau median G (95% CI)')
axa.plot(x, wm, 's', color='C2', ms=7, label='within-plateau median G')
axa.plot(x, wp, '^', color='C2', ms=7, mfc='none', label='within-plateau p95')
for xi, v in zip(x, vnames):
    axa.annotate(f"cx {variants[v]['n_counterexamples']}/{variants[v]['n_verified']}",
                 (xi, bm[int(xi)]), textcoords='offset points', xytext=(0, 12),
                 ha='center', fontsize=9, color='C3')
axa.axhline(1.0, color='k', ls='--', lw=1)
axa.set_xticks(x); axa.set_xticklabels(vnames, fontsize=9)
axa.set_ylabel('median G')
axa.set_title('(a) Verdict quantities under four normalizations\n'
              '(cx = counterexample pairs, median G <= 1)')
axa.legend(fontsize=8); axa.grid(alpha=0.3)

asym = np.array([max(r['si'], r['sj']) / min(r['si'], r['sj']) for r in verified])
gmax = np.array([variants['max(si,sj) [frozen]']['per_pair'][r['name']] for r in verified])
gmin = np.array([variants['min(si,sj)']['per_pair'][r['name']] for r in verified])
axb.scatter(asym, gmax, color='C0', s=30, label='G with max(si,sj)  [frozen]')
axb.scatter(asym, gmin, color='C3', s=30, marker='^', label='G with min(si,sj)')
for r, a, g in zip(verified, asym, gmin):
    if '1' in r['name'].split('-') and g > 1.5:
        axb.annotate(r['name'], (a, g), textcoords='offset points', xytext=(4, 2), fontsize=7)
axb.axhline(1.0, color='k', ls='--', lw=1)
axb.set_xlabel('region scale asymmetry  max(si,sj) / min(si,sj)')
axb.set_ylabel('per-pair median G')
axb.set_title('(b) min-normalized G grows with scale asymmetry\n'
              '(it tracks region-density mismatch, not a boundary gap)')
axb.legend(fontsize=8); axb.grid(alpha=0.3)

plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'normalization_check.png'), dpi=140); plt.close()
print("saved results/normalization_check.json + plots/normalization_check.png")
