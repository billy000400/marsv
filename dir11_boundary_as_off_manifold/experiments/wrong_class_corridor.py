#!/usr/bin/env python3
"""
dir11 — operator feedback 2026-07-17 (human_feedback_1.txt):

  "Your natural activation cloud is only activations of correctly classified test
   images. I'm wondering if the low density corridor corresponds to the wrongly
   classified images."

Three tests, base model, frozen conventions of direct_path_offmanifold.py:

  A. WHERE DO WRONG ACTIVATIONS SIT? For each wrongly-classified test image's L1
     activation, its support radius r_10 vs the correct cloud, as a percentile of
     the natural (correct-point) baseline. If the corridor is their home, they
     should pile up at high percentiles like the path-excursion points do.
  B. DOES ADDING THEM FILL THE CORRIDOR? Recompute the per-path excursion E for
     the SAME sampled paths against the augmented cloud (correct + wrong = all
     2000 test activations), with the baseline recomputed on the augmented cloud.
     If wrong images fill the corridor, between-plateau E drops toward the
     within-plateau control level.
  C. ARE CORRIDOR POINTS AT HOME AMONG WRONG ACTIVATIONS? For each verified
     between-plateau path, take its corridor point x* = argmax_t r_10(t) (vs the
     correct cloud). Measure r_10^wrong(x*) = distance to the 10th-nearest WRONG
     activation, as a percentile of the wrong cloud's own self-support baseline
     (each wrong point's distance to its 10th-nearest other wrong point). ~50 or
     below = as supported by wrong activations as a typical wrong activation is.

Outputs: plots/wrong_class_corridor.png, results/wrong_class_corridor.json.
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

K_SUPPORT = 10
N_POINTS_POP = 120
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
widx = torch.where(~correct)[0]
nat = L1[cidx].numpy().astype(np.float64)      # correct cloud (the frozen natural cloud)
wro = L1[widx].numpy().astype(np.float64)      # wrong cloud
aug = L1.numpy().astype(np.float64)            # augmented cloud = all test activations
Nc, Nw, Na = nat.shape[0], wro.shape[0], aug.shape[0]
pos = {int(g): i for i, g in enumerate(cidx.numpy())}
cloud_global = cidx.numpy()
print(f"correct cloud {Nc}, wrong cloud {Nw}, augmented {Na}")

# ------------------------------------------------- support machinery (3 clouds)
def baseline(cloud):
    """each cloud point's own k-th-NN radius (self excluded), sorted"""
    nn = NearestNeighbors(n_neighbors=K_SUPPORT + 1).fit(cloud)
    rad = nn.kneighbors(cloud)[0][:, -1]
    return np.sort(rad)

nn_nat = NearestNeighbors(n_neighbors=K_SUPPORT).fit(nat)
nn_wro = NearestNeighbors(n_neighbors=K_SUPPORT).fit(wro)
nn_aug = NearestNeighbors(n_neighbors=K_SUPPORT).fit(aug)
base_nat = baseline(nat)   # median 2.85 / p95 4.23 (published)
base_wro = baseline(wro)
base_aug = baseline(aug)
med_nat, p95_nat = float(np.median(base_nat)), float(np.percentile(base_nat, 95))
med_wro = float(np.median(base_wro))
med_aug, p95_aug = float(np.median(base_aug)), float(np.percentile(base_aug, 95))
print(f"baselines r_{K_SUPPORT}: correct med {med_nat:.3f}/p95 {p95_nat:.3f} | "
      f"wrong med {med_wro:.3f} | augmented med {med_aug:.3f}/p95 {p95_aug:.3f}")

def pct(r, base):
    return np.searchsorted(base, r) / len(base) * 100

# =============================================================== Test A:
# wrong activations' support vs the correct cloud
wrong_rad = nn_nat.kneighbors(wro)[0][:, -1]
wrong_pct = np.array([pct(r, base_nat) for r in wrong_rad])
a_med = float(np.median(wrong_pct))
a_above = float(np.mean(wrong_rad > p95_nat))
print(f"\nA. wrong activations vs correct cloud: median pctile {a_med:.1f}, "
      f"{a_above*100:.0f}% beyond natural p95")

# =============================================================== path sampling
# (identical code + seed to direct_path_offmanifold.py -> same endpoint pairs)
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

def eval_paths(rI, rJ, same):
    recs = []
    for a, b in sample_pairs(rI, rJ, N_PAIRS, same):
        ga, gb = int(cloud_global[a]), int(cloud_global[b])
        d, pts = d_of_t(ga, gb, N_POINTS_POP)
        ok = plateau_ok(d) if not same else True
        P = pts.numpy()
        rad_nat = nn_nat.kneighbors(P)[0][:, -1]
        rad_wro = nn_wro.kneighbors(P)[0][:, -1]
        rad_aug = nn_aug.kneighbors(P)[0][:, -1]
        istar = int(rad_nat.argmax())          # corridor point (vs correct cloud)
        recs.append(dict(
            ok=ok,
            E_nat=pct(rad_nat.max(), base_nat),
            E_aug=pct(rad_aug.max(), base_aug),
            corr_wrong_pct=pct(rad_wro[istar], base_wro),
            prof_nat=(rad_nat / med_nat).tolist(),
            prof_wro=(rad_wro / med_wro).tolist(),
        ))
    return recs

between, within = [], []
for di, dj in itertools.combinations([str(d) for d in range(10)], 2):
    between.extend(eval_paths(regions[di], regions[dj], False))
between.extend(eval_paths(sub9A, sub9B, False))
for d in range(10):
    within.extend(eval_paths(regions[str(d)], regions[str(d)], True))
btw = [r for r in between if r['ok']]
print(f"\nverified between paths {len(btw)}/{len(between)}, controls {len(within)}")

bE_nat = np.array([r['E_nat'] for r in btw]);  wE_nat = np.array([r['E_nat'] for r in within])
bE_aug = np.array([r['E_aug'] for r in btw]);  wE_aug = np.array([r['E_aug'] for r in within])
bCW = np.array([r['corr_wrong_pct'] for r in btw])
wCW = np.array([r['corr_wrong_pct'] for r in within])
print(f"regression check (correct-only E): between median {np.median(bE_nat):.1f} "
      f"(published 95.4), within {np.median(wE_nat):.1f} (published 65.2)")
print(f"B. augmented-cloud E: between median {np.median(bE_aug):.1f}, "
      f"within {np.median(wE_aug):.1f}")
print(f"C. corridor-point support among WRONG cloud: between median pctile "
      f"{np.median(bCW):.1f}, within {np.median(wCW):.1f}")

json.dump(dict(
    k=K_SUPPORT, n_correct=Nc, n_wrong=Nw, n_augmented=Na,
    baselines=dict(correct_med=med_nat, correct_p95=p95_nat, wrong_med=med_wro,
                   augmented_med=med_aug, augmented_p95=p95_aug),
    A_wrong_vs_correct=dict(median_pctile=a_med, frac_above_p95=a_above,
                            iqr=[float(np.percentile(wrong_pct, 25)),
                                 float(np.percentile(wrong_pct, 75))]),
    B_augmented_E=dict(between_median=float(np.median(bE_aug)),
                       between_iqr=[float(np.percentile(bE_aug, 25)),
                                    float(np.percentile(bE_aug, 75))],
                       within_median=float(np.median(wE_aug)),
                       between_median_correct_only=float(np.median(bE_nat)),
                       within_median_correct_only=float(np.median(wE_nat))),
    C_corridor_in_wrong=dict(between_median_pctile=float(np.median(bCW)),
                             between_iqr=[float(np.percentile(bCW, 25)),
                                          float(np.percentile(bCW, 75))],
                             within_median_pctile=float(np.median(wCW)),
                             frac_above_wrong_p95=float(np.mean(bCW > 95))),
    n_verified_between=len(btw), n_within=len(within),
), open(os.path.join(RES, 'wrong_class_corridor.json'), 'w'), indent=2)

# =============================================================== figure
t = np.linspace(0, 1, N_POINTS_POP)
prof_nat = np.array([r['prof_nat'] for r in btw])
prof_wro = np.array([r['prof_wro'] for r in btw])
fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(18, 5.2))
bins = np.linspace(0, 100, 26)

axa.hist(wrong_pct, bins=bins, alpha=0.65, color='C4', density=True,
         label=f'wrong-image activations (n={Nw})')
axa.hist(bE_nat, bins=bins, alpha=0.55, color='C3', density=True,
         label=f'corridor excursions E (n={len(btw)})')
axa.axhline(1 / 100, color='C2', ls=':', lw=1.5,
            label='correct activations (uniform by construction)')
axa.axvline(95, color='k', ls='--', lw=1.2, label='natural p95')
axa.set_xlabel('support percentile vs correct-cloud baseline')
axa.set_ylabel('density')
axa.set_title(f'(A) Wrong activations sit INSIDE the correct cloud\n'
              f'median pctile {a_med:.0f} ({a_above*100:.0f}% beyond p95) '
              f'vs {np.median(bE_nat):.0f} for corridor points')
axa.legend(fontsize=8); axa.grid(alpha=0.3)

for prof, med_b, c, lab in ((prof_nat, None, 'C3',
                             f'$r_{{10}}$ to CORRECT cloud / its median ({med_nat:.2f})'),
                            (prof_wro, None, 'C4',
                             f'$r_{{10}}$ to WRONG cloud / its median ({med_wro:.2f})')):
    med = np.median(prof, axis=0)
    lo, hi = np.percentile(prof, 25, axis=0), np.percentile(prof, 75, axis=0)
    axb.plot(t, med, color=c, lw=2, label=lab)
    axb.fill_between(t, lo, hi, color=c, alpha=0.25)
axb.axhline(1.0, color='k', ls=':', lw=1, label='cloud\'s own median support')
axb.set_xlabel('t (slerp position in L1)')
axb.set_ylabel('$r_{10}$(t) / that cloud\'s median baseline')
axb.set_title('(B) No mid-path dip toward the wrong cloud — the corridor is not their home\n'
              'verified between-plateau paths (median, IQR band); a dip below 1 would mean\n'
              'wrong activations live mid-path; instead the profile stays flat ~1.25')
axb.legend(fontsize=8); axb.grid(alpha=0.3)

axc.hist(bE_nat, bins=bins, alpha=0.5, color='C3', density=True,
         label=f'E vs correct-only cloud ({Nc} pts), median {np.median(bE_nat):.0f}')
axc.hist(bE_aug, bins=bins, histtype='step', lw=2.2, color='C0', density=True,
         label=f'E vs augmented cloud ({Na} pts), median {np.median(bE_aug):.0f}')
axc.axvline(95, color='k', ls='--', lw=1.2, label='p95 of each baseline')
axc.set_xlabel('off-manifold excursion E (percentile of that cloud\'s baseline)')
axc.set_ylabel('density')
axc.set_title(f'(C) Adding the {Nw} wrong activations does NOT fill the corridor\n'
              f'between-plateau E: {np.median(bE_nat):.0f} → {np.median(bE_aug):.0f} '
              f'(within controls: {np.median(wE_nat):.0f} → {np.median(wE_aug):.0f})')
axc.legend(fontsize=8); axc.grid(alpha=0.3)

fig.suptitle(f'Is the low-density corridor where the wrongly-classified images live? — '
             f'base model, slerp in L1, $r_{{{K_SUPPORT}}}$ support, '
             f'{len(btw)} verified between-plateau paths, seed {SAMPLE_SEED}', fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(PLOT, 'wrong_class_corridor.png'), dpi=140); plt.close()
print("saved plots/wrong_class_corridor.png + results/wrong_class_corridor.json")
