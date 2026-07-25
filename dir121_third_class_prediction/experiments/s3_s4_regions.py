"""
S3 + S4 + S5 — real-digit activation regions at post-ReLU h1, and whether the
stable third-class segments found in S1 land inside the real activation region
of the digit they are predicted to be.

S3  reference sets: per digit c, mean and coordinate-wise std of h1 over 2,000
    REAL TRAINING images; the 95th percentile of the normalized distance over
    700 HELD-OUT REAL TEST images (test[2000:], disjoint from the endpoint pool
    test[:2000]). Interpolation points never define a region.
S4  for every stable third-class transition: normalized distance to both
    endpoint digits and to the dominant third digit z along alpha, and the
    fraction of third-class-segment points/paths that are both inside z's region
    (ratio < 1) and closer to z than to any other digit.
S5  controls: held-out real images land in their own region; endpoint-predicted
    path portions land in the endpoint region; within-digit interpolations stay
    inside that digit's region; reversing a path reverses alpha only.

Outputs results/s3_s4_regions.json and four figures in plots/.
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/mars-plateaus-image')
sys.path.insert(0, '/workspace/marsv_agent_haoyang/dir12_plateau_during_training/experiments')
from plateau_protocol import slerp_batch, load_state_model, N_POINTS, N_TEST_POOL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_analyze import third_runs
from cvd_style import CVD, MARKERS, use_cvd

use_cvd()

HERE = '/workspace/marsv_agent_haoyang/dir121_third_class_prediction'
DIR12 = '/workspace/marsv_agent_haoyang/dir12_plateau_during_training'
DATA = '/workspace/mars-plateaus-image/data/mnist'
PLOTS = os.path.join(HERE, 'plots')
CKPT = os.path.join(DIR12, 'results', 'full_mnist_from_scratch', 'seed_0',
                    'ckpts', 'step30000.pt')
N_REF_TRAIN, N_REF_TEST, N_PER = 2000, 700, 100
STD_FLOOR = 0.01               # sigma_c,j floored at 1% of the global coord std

torch.set_num_threads(2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.225)


@torch.no_grad()
def h1_of(model, x):
    return model.hidden_activations(x.to(device))[0][0]


def normalized_dist(U, mu, sig):
    """Root-mean-square z-scored distance from each row of U to each digit mean.

    U [n,200], mu/sig [10,200] -> [n,10].
    """
    zsq = ((U[:, None, :] - mu[None]) / sig[None]) ** 2
    return zsq.mean(dim=2).sqrt()


def main():
    from src.mnist import load_mnist
    train_x, train_y, test_x, test_y = load_mnist(DATA)
    model = load_state_model(CKPT, device)

    # ---------------- S3: reference regions from REAL images only -------------
    ref_idx, held_idx = {}, {}
    mu = torch.zeros(10, 200, device=device)
    sig = torch.zeros(10, 200, device=device)
    for c in range(10):
        ref_idx[c] = torch.where(train_y == c)[0][:N_REF_TRAIN]
        held_idx[c] = (torch.where(test_y[N_TEST_POOL:] == c)[0][:N_REF_TEST]
                       + N_TEST_POOL)
        H = h1_of(model, train_x[ref_idx[c]])
        mu[c], sig[c] = H.mean(0), H.std(0)
    glob = h1_of(model, train_x[torch.cat([ref_idx[c] for c in range(10)])]).std(0)
    sig = torch.maximum(sig, STD_FLOOR * glob[None]).clamp_min(1e-6)

    q95 = torch.zeros(10, device=device)
    held_D, held_lab = [], []
    for c in range(10):
        D = normalized_dist(h1_of(model, test_x[held_idx[c]]), mu, sig)   # [n,10]
        q95[c] = torch.quantile(D[:, c], 0.95)
        held_D.append(D)
        held_lab.append(torch.full((len(D),), c, device=device))
    held_D, held_lab = torch.cat(held_D), torch.cat(held_lab)

    def ratio(U):
        return normalized_dist(U, mu, sig) / q95[None]

    # ---- control C1: held-out real images should sit in their own region -----
    R_held = held_D / q95[None]
    c1_argmin = float((R_held.argmin(1) == held_lab).float().mean())
    c1_inside = float((R_held.gather(1, held_lab[:, None])[:, 0] < 1).float().mean())
    c1_both = float(((R_held.argmin(1) == held_lab) &
                     (R_held.gather(1, held_lab[:, None])[:, 0] < 1)).float().mean())
    print(f'C1 held-out real: argmin-correct {c1_argmin:.3f}, inside-own {c1_inside:.3f}, '
          f'both {c1_both:.3f}')

    # ---------------- rebuild the S1 pair bank and its predictions ------------
    cls = json.load(open(os.path.join(HERE, 'results', 's1_classification.json')))
    z1 = np.load(os.path.join(HERE, 'results', 's1_census.npz'))
    idx_a, idx_b, pred_all = z1['idx_a'], z1['idx_b'], z1['pred'][0].reshape(45, N_PER, N_POINTS)
    stable = [(k, r) for k, r in enumerate(cls['transitions']) if r['seeds']['0']['stable_third']]
    print(f'{len(stable)} stable third-class transitions to analyze')

    alpha = np.linspace(0, 1, N_POINTS)
    out, curves = [], {}
    for k, r in stable:
        a, b, zz = r['a'], r['b'], r['seeds']['0']['z_dominant']
        hA = h1_of(model, test_x[idx_a[k]])
        hB = h1_of(model, test_x[idx_b[k]])
        path = slerp_batch(hA, hB, N_POINTS)                     # [100,50,200]
        R = ratio(path.reshape(-1, 200)).reshape(N_PER, N_POINTS, 10).cpu().numpy()
        curves[r['transition']] = {'R_mean': R.mean(0), 'a': a, 'b': b, 'z': zz,
                                   'z_frac': (pred_all[k] == zz).mean(0)}
        pt_in, pt_tot, path_ok, seg_paths, argmin_hist = 0, 0, 0, 0, np.zeros(10)
        rz_vals, rmin_vals = [], []
        for p in range(N_PER):
            runs = third_runs(pred_all[k, p], a, b)
            if zz not in runs:
                continue
            L = runs[zz]
            # locate the longest z-run
            best, cur, s0 = (0, 0), 0, 0
            for t in range(N_POINTS + 1):
                if t < N_POINTS and pred_all[k, p, t] == zz:
                    if cur == 0:
                        s0 = t
                    cur += 1
                else:
                    if cur > best[1] - best[0]:
                        best = (s0, s0 + cur)
                    cur = 0
            seg = slice(*best)
            assert best[1] - best[0] == L
            Rs = R[p, seg]                                        # [L,10]
            ok = (Rs[:, zz] < 1) & (Rs.argmin(1) == zz)
            pt_in += int(ok.sum()); pt_tot += L
            path_ok += int(ok.mean() > 0.5)
            seg_paths += 1
            argmin_hist += np.bincount(Rs.argmin(1), minlength=10)
            rz_vals.append(float(np.median(Rs[:, zz])))
            rmin_vals.append(float(np.median(Rs.min(1))))
        # control C2: endpoint-predicted portions vs the endpoint region, both
        # over ALL such points and restricted to the endpoint plateau
        # (alpha <= 0.1 for endpoint a, alpha >= 0.9 for endpoint b).
        c2, c2_plateau = [], []
        near = np.zeros((N_PER, N_POINTS), bool)
        for endc, sl in ((a, slice(0, 6)), (b, slice(N_POINTS - 6, N_POINTS))):
            m = pred_all[k] == endc
            if m.sum():
                Re = R[m]
                c2.append(float(((Re[:, endc] < 1) & (Re.argmin(1) == endc)).mean()))
            near[:] = False
            near[:, sl] = True
            mp = m & near
            if mp.sum():
                Rp = R[mp]
                c2_plateau.append(float(((Rp[:, endc] < 1) & (Rp.argmin(1) == endc)).mean()))
        out.append({'transition': r['transition'], 'a': a, 'b': b, 'z': zz,
                    'prevalence_z': r['seeds']['0']['prevalence_z'],
                    'n_paths_with_segment': seg_paths, 'n_segment_points': pt_tot,
                    'point_frac_in_z_region': round(pt_in / max(pt_tot, 1), 3),
                    'path_frac_majority_in_z_region': round(path_ok / max(seg_paths, 1), 3),
                    'median_ratio_to_z_on_segment': round(float(np.median(rz_vals)), 3),
                    'median_min_ratio_on_segment': round(float(np.median(rmin_vals)), 3),
                    'segment_argmin_digit_hist': {int(i): int(v) for i, v in
                                                  enumerate(argmin_hist) if v},
                    'control_endpoint_portion_in_region': [round(x, 3) for x in c2],
                    'control_endpoint_plateau_in_region': [round(x, 3) for x in c2_plateau]})
        print(f"  {r['transition']:6s} z={zz} points-in-z-region "
              f"{out[-1]['point_frac_in_z_region']:.3f} paths "
              f"{out[-1]['path_frac_majority_in_z_region']:.3f} "
              f"median R_z {out[-1]['median_ratio_to_z_on_segment']:.2f} "
              f"median min-R {out[-1]['median_min_ratio_on_segment']:.2f} "
              f"endpoint-ctrl {out[-1]['control_endpoint_portion_in_region']} "
              f"plateau {out[-1]['control_endpoint_plateau_in_region']}")
        del path
        if device == 'cuda':
            torch.cuda.empty_cache()

    # ---- control C3: within-digit interpolation stays in its own region ------
    pool_y = test_y[:N_TEST_POOL]
    c3 = {}
    for c in range(10):
        cand = torch.where(pool_y == c)[0]
        n = len(cand)
        ia, ib = cand[:N_PER], cand[(np.arange(N_PER) + n // 2) % n]
        pth = slerp_batch(h1_of(model, test_x[ia]), h1_of(model, test_x[ib]), N_POINTS)
        Rw = ratio(pth.reshape(-1, 200)).reshape(N_PER, N_POINTS, 10)
        logits, _ = model.forward_from(pth.reshape(-1, 200), 1)
        pw = logits.argmax(1).reshape(N_PER, N_POINTS)
        c3[c] = {'frac_points_inside_own_region': round(float((Rw[:, :, c] < 1).float().mean()), 3),
                 'frac_points_argmin_own': round(float((Rw.argmin(2) == c).float().mean()), 3),
                 'frac_points_predicted_own': round(float((pw == c).float().mean()), 3),
                 'mean_ratio_own': round(float(Rw[:, :, c].mean()), 3)}
    print('C3 within-digit:', {c: c3[c]['frac_points_inside_own_region'] for c in range(10)})

    # ---- control C4: reversing a path reverses alpha only --------------------
    k0, r0 = stable[0]
    hA, hB = h1_of(model, test_x[idx_a[k0]]), h1_of(model, test_x[idx_b[k0]])
    fwd = slerp_batch(hA, hB, N_POINTS)
    rev = slerp_batch(hB, hA, N_POINTS).flip(1)
    c4 = float((fwd - rev).abs().max() / fwd.abs().max())
    print(f'C4 reversal max relative deviation: {c4:.2e}')

    # (The earlier PCA view was replaced by s4b_planes.py: a real-class LDA plane
    # and a path-local margin-gradient/SVD decision slice. See PLAN.md S4b.)

    res = {'meta': {'ckpt': CKPT, 'ckpt_sha256': str(z1['ckpt_sha256'][0]),
                    'n_ref_train_per_digit': N_REF_TRAIN,
                    'n_heldout_test_per_digit': N_REF_TEST, 'std_floor': STD_FLOOR,
                    'q95': [round(float(v), 3) for v in q95]},
           'controls': {'C1_heldout_argmin_correct': round(c1_argmin, 3),
                        'C1_heldout_inside_own_region': round(c1_inside, 3),
                        'C1_heldout_both': round(c1_both, 3),
                        'C3_within_digit': c3,
                        'C4_reversal_max_rel_dev': c4},
           'transitions': out}
    json.dump(res, open(os.path.join(HERE, 'results', 's3_s4_regions.json'), 'w'), indent=1)
    print('saved results/s3_s4_regions.json')
    figures(curves, out, c3, R_held, held_lab.cpu().numpy(), alpha)


def figures(curves, out, c3, R_held, held_lab, alpha):
    keys = [o['transition'] for o in out]
    ncol, nrow = 5, int(np.ceil(len(keys) / 5))

    # 1 — full-space normalized-distance view for every stable transition
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow),
                             squeeze=False, sharex=True)
    for ax in axes.ravel():
        ax.axis('off')
    for n, key in enumerate(keys):
        cv = curves[key]
        ax = axes[n // ncol, n % ncol]; ax.axis('on')
        for i, (c, lab) in enumerate(((cv['a'], 'endpoint a'), (cv['b'], 'endpoint b'),
                                      (cv['z'], 'third digit z'))):
            ax.plot(alpha, cv['R_mean'][:, c], ls=['-', '--', '-.'][i], lw=2,
                    color=CVD[i], marker=MARKERS[i], markevery=8, ms=4,
                    label=f'{lab} = {c}')
        ax.axhline(1.0, ls='--', c='k', lw=1)
        ax.fill_between(alpha, 0, 1, where=cv['z_frac'] > 0.5, color='0.8',
                        alpha=0.5, transform=ax.get_xaxis_transform())
        ax.set_title(f"{key}   z={cv['z']}", fontsize=10)
        ax.set_yscale('log'); ax.legend(fontsize=6.5)
        if n // ncol == nrow - 1:
            ax.set_xlabel(r'$\alpha$', fontsize=9)
        if n % ncol == 0:
            ax.set_ylabel('normalized distance ratio\n(log scale)', fontsize=8.5)
    fig.suptitle('Distance from the interpolated $h_1$ activation to each real digit '
                 'activation region, mean over 100 pairs.\nDashed line = 1 (the held-out '
                 '95th percentile for that digit); grey band = alpha range where most '
                 'paths predict $z$.', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(PLOTS, 's4_distance_view.png'), dpi=130)
    plt.close(fig)

    # 2 — headline bar: how much of each third-class segment is inside z's region
    fig, ax = plt.subplots(figsize=(0.55 * len(keys) + 3.5, 4.6))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [o['point_frac_in_z_region'] for o in out], 0.4,
           color=CVD[0], hatch='//', edgecolor='w',
           label='fraction of segment POINTS inside the real-$z$ region')
    ax.bar(x + 0.2, [o['path_frac_majority_in_z_region'] for o in out], 0.4,
           color=CVD[1], hatch='\\\\', edgecolor='w',
           label='fraction of PATHS with a majority of segment points inside')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\nz={o['z']}" for k, o in zip(keys, out)], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel('fraction (bars, left axis)')
    ax.set_xlabel('stable third-class transition')
    ax.set_title('Are stable third-class segments inside the real activation region of the '
                 'digit they are predicted to be?\n(inside = ratio < 1 AND closest of all '
                 'ten digits)', fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(x, [o['median_ratio_to_z_on_segment'] for o in out], 'D-', c='k', ms=5,
             label='median ratio to $z$ on the segment (solid, diamonds)')
    ax2.plot(x, [o['median_min_ratio_on_segment'] for o in out], '^--', c=CVD[2], ms=5,
             label='median ratio to the NEAREST of all ten digits (dashed, triangles)')
    ax2.axhline(1, ls=':', c='k', lw=1)
    ax2.set_ylabel('normalized distance ratio (markers, right axis)')
    ax2.set_ylim(0, None)
    h1_, l1_ = ax.get_legend_handles_labels()
    h2_, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1_ + h2_, l1_ + l2_, fontsize=8.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 's4_region_membership.png'), dpi=150)
    plt.close(fig)

    # 3 — controls
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    own = R_held[np.arange(len(held_lab)), held_lab].cpu().numpy()
    ax.hist(own, bins=60, color=CVD[0])
    ax.axvline(1, ls='--', c='k')
    ax.set_xlabel('normalized distance ratio to its OWN digit region')
    ax.set_ylabel('held-out real test images')
    ax.set_title('C1 — real held-out images vs their own region', fontsize=10)
    ax = axes[1]
    ax.bar(range(10), [c3[c]['frac_points_inside_own_region'] for c in range(10)],
           color=CVD[0], hatch='//', edgecolor='w')
    ax.set_xticks(range(10)); ax.set_xlabel('digit c (within-digit interpolation c$\\to$c)')
    ax.set_ylabel('fraction of interpolation points\nwith ratio < 1 to digit c')
    ax.set_ylim(0, 1.05)
    ax.set_title('C3 — within-digit interpolation stays inside', fontsize=10)
    ax = axes[2]
    ends = np.array([o['control_endpoint_portion_in_region'] for o in out]).ravel()
    endp = np.array([o['control_endpoint_plateau_in_region'] for o in out]).ravel()
    _n, _b, patches = ax.hist([endp, ends], bins=10, range=(0, 1),
                              color=[CVD[0], CVD[1]], edgecolor='w',
                              label=[r'endpoint plateau only ($\alpha\leq0.1$ / $\geq0.9$),'
                                     ' hatch //',
                                     'all endpoint-predicted points, hatch \\\\'])
    for hh, group in zip(('//', '\\\\'), patches):
        for p in group:
            p.set_hatch(hh)
    ax.set_xlabel('fraction of endpoint-predicted points inside\nthat endpoint digit\'s region')
    ax.set_ylabel('count (2 endpoints x 19 transitions)')
    ax.set_title('C2 — endpoint-predicted portions vs endpoint region', fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle('Controls on the mean-and-variance activation-region summary at $h_1$',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(PLOTS, 's5_controls.png'), dpi=150)
    plt.close(fig)
    print('saved 3 figures to plots/')


if __name__ == '__main__':
    main()
