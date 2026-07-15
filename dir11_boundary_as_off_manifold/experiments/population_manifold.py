#!/usr/bin/env python3
"""
dir11 (reframed) — population-level test:

  When an activation path moves from one stable output plateau to another, do the
  two plateaus lie on DIFFERENT empirical components of the natural activation
  manifold?

We answer this across ALL sufficiently populated plateau pairs, not just the
digit-9 case, with ONE frozen manifold observable:

  Normalized connection bottleneck G.
    Build a Euclidean minimum spanning tree (MST) over the natural first-hidden-
    layer (L1) activation cloud. For two endpoints, B = the largest edge on their
    unique MST path = the smallest step size needed to connect them through the
    sampled natural cloud. Normalize by the within-plateau connection scale:
      G = B / max(s_i, s_j)   for a pair whose endpoints lie in regions i, j,
    where s_r = median within-region MST bottleneck (frozen from within-plateau
    endpoint pairs, BEFORE looking at any between-plateau result).
    G = 1  -> no larger gap than is normally required inside a plateau.
    G > 1  -> an unusually large bridge is required (candidate manifold-component
              separation).

The plateau observable d(t) (normalized downstream distance at the last hidden
layer L3) is used ONLY to verify that a sampled slerp path genuinely begins in one
stable region and ends in another (a sharp plateau-to-plateau transition).

FROZEN definitions (fixed before between-plateau results were examined):
  * Plateau regions   : the 10 digit classes (class-aligned stable output regions),
                        restricted to correctly-classified test images with output
                        margin (top1-top2 logit) >= 0.5 (confidence inclusion rule).
  * Natural cloud     : L1 activations of all correctly-classified test images.
  * d(t) accept rule  : a between-region slerp path counts as a verified plateau
                        transition iff its plateau fraction (fraction of t with
                        d<0.2 or d>0.8) >= 0.5 AND it starts near A (min early d<0.2)
                        and ends near B (max late d>0.8).
  * G normalization   : s_r = median within-region MST bottleneck over 20 sampled
                        within-region endpoint pairs; G = B / max(s_i, s_j).
  * Sampling          : 20 endpoint pairs per region pair, seed 0.
  * Digit-9 sub-plateau: the previously studied A/B split of digit 9 (KMeans-2 on
                        L3) is included as ONE extra transition, treated like any
                        other pair.

Runs the base checkpoint (detailed figures) plus the existing second seed and three
architecture checkpoints (replication). CPU analysis (tiny model).
Outputs results/population.json, plots/population_G.png, plots/population_heatmap.png,
plots/population_dt.png, plots/population_replication.png.
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
os.makedirs(RES, exist_ok=True); os.makedirs(PLOT, exist_ok=True)
DATA = os.path.join(BASE, 'data/mnist')

MARGIN = 0.5          # confidence inclusion rule for endpoints
N_PAIRS = 20          # endpoint pairs per region pair
N_POINTS = 120        # slerp resolution for d(t)
PLATEAU_FRAC = 0.5    # d(t) accept: >= this fraction of path on a plateau

# ---- slerp (own copy so d(t) endpoints are the sampled natural activations) ----
def slerp_path(a, b, n):
    a = a.float(); b = b.float()
    na, nb = a.norm(), b.norm()
    ua, ub = a / (na + 1e-10), b / (nb + 1e-10)
    dot = torch.clamp((ua * ub).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    t = torch.linspace(0, 1, n).unsqueeze(1)
    if omega < 1e-4:
        pts = (1 - t) * a + t * b
    else:
        so = torch.sin(omega)
        pts = (torch.sin((1 - t) * omega) / so) * ua + (torch.sin(t * omega) / so) * ub
        r = (1 - t.squeeze(1)) * na + t.squeeze(1) * nb
        pts = pts * r.unsqueeze(1)
    return pts

# ============================================================ per-model analysis
def analyze(model, ck, label, detailed=False, n_pairs=N_PAIRS):
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

    # ---- MST + bottleneck (max edge on unique MST path) ----
    D = cdist(nat, nat)
    mst = minimum_spanning_tree(D).toarray(); mst = np.maximum(mst, mst.T)
    adj = [[] for _ in range(Nc)]
    ii, jj = np.where(mst > 0)
    for a, b in zip(ii, jj):
        if a < b:
            w = float(mst[a, b]); adj[a].append((b, w)); adj[b].append((a, w))
    cache = {}
    def maxedge_from(src):
        if src in cache:
            return cache[src]
        me = np.full(Nc, np.nan); me[src] = 0.0; stack = [src]
        while stack:
            u = stack.pop(); mu = me[u]
            for v, w in adj[u]:
                if np.isnan(me[v]):
                    me[v] = mu if mu > w else w; stack.append(v)
        cache[src] = me; return me
    def bottleneck(u, v):
        return float(maxedge_from(u)[v])

    # ---- regions ----
    def rows(mask):
        return np.array([pos[int(i)] for i in torch.where(mask)[0].numpy()])
    regions = {str(d): rows(conf & (test_y == d)) for d in range(10)}
    sizes = {k: len(v) for k, v in regions.items()}

    nine_g = torch.where(conf & (test_y == 9))[0].numpy()
    lab9 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(L3[nine_g].numpy()).labels_
    if (lab9 == 0).sum() < (lab9 == 1).sum():
        lab9 = 1 - lab9
    sub9A = np.array([pos[int(i)] for i in nine_g[lab9 == 0]])
    sub9B = np.array([pos[int(i)] for i in nine_g[lab9 == 1]])

    rng = np.random.RandomState(0)
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

    # ---- within-plateau scale (frozen) ----
    within_scale = {}; within_G = []
    for k, rr in regions.items():
        bns = np.array([bottleneck(a, b) for a, b in sample_pairs(rr, rr, n_pairs, True)])
        within_scale[k] = float(np.median(bns))
        within_G.extend((bns / within_scale[k]).tolist())
    within_G = np.array(within_G)

    def sub_scale(rr):
        return float(np.median([bottleneck(a, b) for a, b in sample_pairs(rr, rr, n_pairs, True)]))

    # ---- between-plateau pairs ----
    def eval_pair(name, ri, rj, rI, rJ, si, sj):
        scale = max(si, sj); gs = []; fracs = []; accept = 0; ex = None
        for a, b in sample_pairs(rI, rJ, n_pairs, False):
            d = d_of_t(a, b); ok, frac = plateau_ok(d); fracs.append(frac)
            g = bottleneck(a, b) / scale
            if ok:
                accept += 1; gs.append(g)
                if ex is None:
                    ex = (a, b, d.tolist(), g)
        gs = np.array(gs)
        return dict(name=name, ri=ri, rj=rj, n_verified=int(accept),
                    plateau_frac_mean=float(np.mean(fracs)), scale=scale,
                    G_median=float(np.median(gs)) if len(gs) else None,
                    G_min=float(gs.min()) if len(gs) else None,
                    G_max=float(gs.max()) if len(gs) else None,
                    G_all=gs.tolist(), example=ex)

    pair_records = []; between_G_all = []
    for di, dj in itertools.combinations([str(d) for d in range(10)], 2):
        rec = eval_pair(f"{di}-{dj}", di, dj, regions[di], regions[dj],
                        within_scale[di], within_scale[dj])
        pair_records.append(rec)
        if rec['G_median'] is not None:
            between_G_all.extend(rec['G_all'])
    rec9 = eval_pair("9A-9B", "9A", "9B", sub9A, sub9B, sub_scale(sub9A), sub_scale(sub9B))
    pair_records.append(rec9)
    between_G_all = np.array(between_G_all)

    verified = [r for r in pair_records if r['n_verified'] >= 5 and r['G_median'] is not None]
    counterex = [r for r in verified if r['G_median'] <= 1.0]
    btw_meds = np.array([r['G_median'] for r in verified])

    def boot_med(x, n=2000):
        idx = rng.randint(0, len(x), size=(n, len(x)))
        return np.median(x[idx], axis=1)
    b_btw = boot_med(btw_meds) if len(btw_meds) else np.array([np.nan])
    b_wth = boot_med(within_G)

    out = dict(
        label=label, arch=f"depth={ck['config']['depth']}, width={ck['config']['width']}",
        test_acc=float(correct.float().mean()), n_natural=Nc, n_confident=int(conf.sum()),
        region_sizes=sizes, within_scale=within_scale,
        within_G_median=float(np.median(within_G)), within_G_p95=float(np.percentile(within_G, 95)),
        n_verified_pairs=len(verified), n_counterexamples=len(counterex),
        counterexamples=[r['name'] for r in counterex],
        between_G_median=float(np.median(btw_meds)) if len(btw_meds) else None,
        between_G_min=float(btw_meds.min()) if len(btw_meds) else None,
        between_G_max=float(btw_meds.max()) if len(btw_meds) else None,
        frac_pairs_G_gt_1=float(np.mean(btw_meds > 1.0)) if len(btw_meds) else 0.0,
        boot_between_median_CI=[float(np.percentile(b_btw, 2.5)), float(np.percentile(b_btw, 97.5))],
        boot_within_median_CI=[float(np.percentile(b_wth, 2.5)), float(np.percentile(b_wth, 97.5))],
        digit9_sub=dict(G_median=rec9['G_median'], n_verified=rec9['n_verified']),
        pairs=[{k: r[k] for k in ('name', 'ri', 'rj', 'n_verified', 'G_median',
                                  'G_min', 'G_max', 'plateau_frac_mean')} for r in pair_records],
    )
    if detailed:
        out['_within_G'] = within_G; out['_between_G_all'] = between_G_all
        out['_verified'] = verified; out['_rec9'] = rec9
    return out


# ============================================================ run all models
CHECKPOINTS = [
    ("base d4w200 (seed 0)", os.path.join(BASE, 'results/image/mnist_mlp_d4_w200_relu_n1000_s100000.pt'), True),
    ("seed 1 d4w200", os.path.join(RES, 'mnist_mlp_d4_w200_relu_n1000_seed1.pt'), False),
    ("d3w200 shallower", os.path.join(RES, 'mnist_mlp_d3_w200_relu_n1000_seed2.pt'), False),
    ("d4w400 wider", os.path.join(RES, 'mnist_mlp_d4_w400_relu_n1000_seed3.pt'), False),
    ("d5w200 deeper", os.path.join(RES, 'mnist_mlp_d5_w200_relu_n1000_seed4.pt'), False),
]
results = []
base = None
for label, ckpt, detailed in CHECKPOINTS:
    if not os.path.exists(ckpt):
        print(f"[skip] missing {ckpt}"); continue
    model, ck = load_checkpoint(ckpt)
    r = analyze(model, ck, label, detailed=detailed)
    print(f"[{label}] acc={r['test_acc']:.3f}  within-G med={r['within_G_median']:.3f} p95={r['within_G_p95']:.3f}"
          f"  between-G med={r['between_G_median']:.3f} ({r['between_G_min']:.3f}-{r['between_G_max']:.3f})"
          f"  frac>1={r['frac_pairs_G_gt_1']:.2f}  counterex={r['n_counterexamples']}/{r['n_verified_pairs']}",
          flush=True)
    if detailed:
        base = r
    results.append(r)

# ============================================================ shallow-net power restoration
# d3w200 is under-powered at 20 endpoint pairs (its downstream distance ramps rather than
# plateaus, so the frozen d(t) accept filter rejects almost every path -> only 1 verified
# region-pair). PLAN S3 asks us to restore power (sample MORE endpoint pairs) and confirm the
# verdict is unchanged. We re-run ONLY the shallow net at increasing n_pairs until it clears
# >=20 verified region-pairs (or a cap), keeping every definition (d(t) filter, G, threshold)
# frozen. Base and the other models stay at the frozen 20 pairs.
shallow_ck = os.path.join(RES, 'mnist_mlp_d3_w200_relu_n1000_seed2.pt')
shallow_power = None; shallow_sweep = []
if os.path.exists(shallow_ck):
    m_s, c_s = load_checkpoint(shallow_ck)
    for np_try in (20, 60, 120, 200):
        sp = analyze(m_s, c_s, f"d3w200 shallower ({np_try} pairs)", n_pairs=np_try)
        print(f"[shallow-power np={np_try}] verified_pairs={sp['n_verified_pairs']} "
              f"between-G med={sp['between_G_median']} counterex={sp['n_counterexamples']}"
              f"/{sp['n_verified_pairs']} frac>1={sp['frac_pairs_G_gt_1']:.2f}", flush=True)
        rec = dict(n_pairs=np_try, **{k: sp[k] for k in (
            'n_verified_pairs', 'n_counterexamples', 'between_G_median', 'between_G_min',
            'between_G_max', 'frac_pairs_G_gt_1', 'within_G_median', 'within_G_p95',
            'boot_between_median_CI')})
        shallow_sweep.append(rec)
        shallow_power = rec
        if sp['n_verified_pairs'] >= 20:
            break

    # ---- figure: why the shallow net can't be powered up (structural, not sampling) ----
    d3_main = next(r for r in results if 'd3w200' in r['label'])
    fr_base = np.array([p['plateau_frac_mean'] for p in base['pairs']])
    fr_d3 = np.array([p['plateau_frac_mean'] for p in d3_main['pairs']])
    fig, (axp, axq) = plt.subplots(1, 2, figsize=(13, 4.8))
    axp.hist(fr_base, bins=np.linspace(0, 1, 21), alpha=0.6, color='C0',
             label=f'base d4w200 ({int((fr_base>=0.5).sum())}/{len(fr_base)} region-pairs plateau)')
    axp.hist(fr_d3, bins=np.linspace(0, 1, 21), alpha=0.6, color='C3',
             label=f'd3w200 shallow ({int((fr_d3>=0.5).sum())}/{len(fr_d3)} region-pairs plateau)')
    axp.axvline(0.5, color='k', ls='--', lw=1.2, label='d(t) accept threshold (0.5)')
    axp.set_xlabel('mean plateau fraction of d(t) per region pair')
    axp.set_ylabel('number of region pairs')
    axp.set_title('(a) Shallow net rarely plateaus: d(t) ramps, so no path\npasses the accept filter — a structural gap, not sampling noise')
    axp.legend(fontsize=8); axp.grid(alpha=0.3)
    nn = [r['n_pairs'] for r in shallow_sweep]
    axq.plot(nn, [r['n_verified_pairs'] for r in shallow_sweep], 'o-', color='C0',
             label='verified plateau pairs')
    axq.axhline(20, color='k', ls='--', lw=1, label='well-powered target (20)')
    for r in shallow_sweep:
        axq.annotate(f"G={r['between_G_median']:.2f}", (r['n_pairs'], r['n_verified_pairs']),
                     textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center', color='C3')
    axq.set_xlabel('endpoint pairs sampled per region pair')
    axq.set_ylabel('verified plateau pairs (n_verified >= 5)'); axq.set_ylim(0, 22)
    axq.set_title('(b) Sampling 10x more endpoint pairs does NOT restore power;\nthe few genuine plateaus it finds are all G<=1 (counterexamples)')
    axq.legend(fontsize=8); axq.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'population_shallow_power.png'), dpi=140); plt.close()

# ---- persist (drop bulky arrays) ----
def clean(r):
    return {k: v for k, v in r.items() if not k.startswith('_')}
with open(os.path.join(RES, 'population.json'), 'w') as f:
    json.dump(dict(base=clean(base), replication=[clean(r) for r in results],
                   shallow_power=shallow_power), f, indent=2)

# ============================================================ figures (base)
within_G = base['_within_G']; between_G_all = base['_between_G_all']
verified = base['_verified']; rec9 = base['_rec9']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
ax1.hist(within_G, bins=30, alpha=0.55, color='C2', density=True, label='within-plateau controls')
ax1.hist(between_G_all, bins=30, alpha=0.55, color='C3', density=True, label='between-plateau pairs')
ax1.axvline(1.0, color='k', ls='--', lw=1.2, label='G = 1 (normal within-plateau gap)')
ax1.set_xlabel('normalized connection bottleneck  G = B / within-plateau scale')
ax1.set_ylabel('density'); ax1.set_title('(a) Per-pair G: between-plateau vs within-plateau')
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

names = [r['name'] for r in verified]; meds = [r['G_median'] for r in verified]
order = np.argsort(meds); names = [names[i] for i in order]; meds = [meds[i] for i in order]
colors = ['C1' if n == '9A-9B' else ('C3' if m > 1 else 'C0') for n, m in zip(names, meds)]
ax2.barh(range(len(meds)), meds, color=colors)
ax2.axvline(1.0, color='k', ls='--', lw=1.2)
ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names, fontsize=6)
ax2.set_xlabel('median G over verified plateau transitions')
ax2.set_title(f'(b) Median G per plateau pair ({len(verified)} pairs)\n'
              f'orange = digit-9 sub-plateau; blue = G<=1 (counterexample); red = G>1')
ax2.grid(alpha=0.3, axis='x')
plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'population_G.png'), dpi=140); plt.close()

# heatmap
H = np.full((10, 10), np.nan)
for r in base['pairs']:
    if r['name'] == '9A-9B' or r['G_median'] is None:
        continue
    i, j = int(r['ri']), int(r['rj']); H[i, j] = H[j, i] = r['G_median']
for d in range(10):
    H[d, d] = 1.0
fig, ax = plt.subplots(figsize=(7.4, 6.2))
vmax = np.nanmax(H); im = ax.imshow(H, cmap='RdBu_r', vmin=2 - vmax, vmax=vmax)
ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.set_xlabel('digit'); ax.set_ylabel('digit')
ax.set_title('Median normalized bottleneck G between plateau pairs\n'
             '(diagonal = within-plateau = 1; blue > 1 = larger bridge required)')
for i in range(10):
    for j in range(10):
        if not np.isnan(H[i, j]):
            ax.text(j, i, f'{H[i,j]:.1f}', ha='center', va='center', fontsize=6,
                    color='white' if abs(H[i, j] - 1) > (vmax - 1) * 0.5 else 'black')
fig.colorbar(im, label='median G'); plt.tight_layout()
plt.savefig(os.path.join(PLOT, 'population_heatmap.png'), dpi=140); plt.close()

# representative d(t)
lowG = min((r for r in verified if r['name'] != '9A-9B'), key=lambda r: r['G_median'])
highG = max((r for r in verified if r['name'] != '9A-9B'), key=lambda r: r['G_median'])
examples = [(f"{highG['name']}  (G={highG['G_median']:.2f}, largest bridge)", highG['example']),
            (f"9A-9B digit-9 sub  (G={rec9['G_median']:.2f})" if rec9['example'] else "9A-9B (none)", rec9['example']),
            (f"{lowG['name']}  (G={lowG['G_median']:.2f}, smallest bridge)", lowG['example'])]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2)); t = np.linspace(0, 1, N_POINTS)
for ax, (title, ex) in zip(axes, examples):
    if ex is None:
        ax.set_title(title + '\n(no verified path)'); continue
    ax.plot(t, ex[2], 'C0', lw=2)
    ax.axhspan(0, 0.2, color='C2', alpha=0.15); ax.axhspan(0.8, 1.0, color='C3', alpha=0.15)
    ax.set_ylim(-0.05, 1.05); ax.set_xlabel('t'); ax.set_ylabel('d(t) downstream'); ax.set_title(title)
    ax.grid(alpha=0.3)
fig.suptitle('Representative verified plateau-to-plateau transitions d(t): flat near A, sharp jump, flat near B\n'
             '(shaded = plateau bands used by the accept rule)', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'population_dt.png'), dpi=140); plt.close()

# replication across 5 models
fig, (axa, axb) = plt.subplots(1, 2, figsize=(14, 5.2))
labels = [r['label'] for r in results]
btw_med = [r['between_G_median'] for r in results]
btw_lo = [r['boot_between_median_CI'][0] for r in results]
btw_hi = [r['boot_between_median_CI'][1] for r in results]
wth_med = [r['within_G_median'] for r in results]
x = np.arange(len(results))
axa.errorbar(x, btw_med, yerr=[np.array(btw_med) - btw_lo, np.array(btw_hi) - btw_med],
             fmt='o', color='C3', capsize=4, label='between-plateau median G (95% CI)')
axa.plot(x, wth_med, 's', color='C2', label='within-plateau median G')
axa.axhline(1.0, color='k', ls='--', lw=1)
axa.set_xticks(x); axa.set_xticklabels(labels, rotation=20, fontsize=8, ha='right')
axa.set_ylabel('median G'); axa.set_title('(a) Between- vs within-plateau median G across models')
axa.legend(fontsize=8); axa.grid(alpha=0.3)

frac = [r['frac_pairs_G_gt_1'] * 100 for r in results]
ncx = [r['n_counterexamples'] for r in results]
nvp = [r['n_verified_pairs'] for r in results]
axb.bar(x - 0.2, frac, 0.4, color='C0', label='% verified pairs with G>1')
axb.bar(x + 0.2, [100 * c / n for c, n in zip(ncx, nvp)], 0.4, color='C1',
        label='% counterexamples (G<=1)')
axb.set_xticks(x); axb.set_xticklabels(labels, rotation=20, fontsize=8, ha='right')
axb.set_ylabel('percent of verified plateau pairs'); axb.set_ylim(0, 100)
axb.set_title('(b) Neither claim holds: ~half of pairs sit each side of G=1')
axb.legend(fontsize=8); axb.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(os.path.join(PLOT, 'population_replication.png'), dpi=140); plt.close()

print("\nsaved population.json + 4 plots")
print(json.dumps({r['label']: dict(acc=round(r['test_acc'], 3),
      between_G_med=round(r['between_G_median'], 3),
      frac_gt1=round(r['frac_pairs_G_gt_1'], 2),
      counterex=f"{r['n_counterexamples']}/{r['n_verified_pairs']}") for r in results}, indent=2))
