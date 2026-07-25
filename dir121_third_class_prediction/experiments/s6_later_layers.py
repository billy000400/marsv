"""
Preregistered later-layer follow-up (PLAN.md "Optional later-layer follow-up"),
unlocked because the h1 result is a clean null: stable third-class segments are
predicted as z but are NOT close to real z activations at h1.

Repeats exactly the same reference-region construction and normalized-distance
comparison at h2 and h3 (the second and third post-ReLU hidden layers), on the
SAME frozen Stage-1 segments and the same 100-pair bank. Nothing is retuned:
the region definition, the std floor, the held-out 95th-percentile calibration
and the inside-region criterion are identical to s3_s4_regions.py, only the
hook point changes. h1 numbers are recomputed here too so the three layers are
produced by one code path and are directly comparable.

Outputs results/s6_later_layers.json and plots/s6_later_layers.png.
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
from s3_s4_regions import (HERE, CKPT, DATA, PLOTS, N_REF_TRAIN, N_REF_TEST,
                           N_PER, STD_FLOOR, h1_of, normalized_dist, device)
from s1_analyze import third_runs

LAYERS = [1, 2, 3]                     # post-ReLU hidden layers; 1 == h1


@torch.no_grad()
def hidden_of(model, x, layer):
    """Post-ReLU hidden `layer` (1-indexed) for real input images."""
    return model.hidden_activations(x.to(device))[0][layer - 1]


def longest_z_seg(pred_row, zz):
    best, cur, s0 = (0, 0), 0, 0
    for t in range(len(pred_row) + 1):
        if t < len(pred_row) and pred_row[t] == zz:
            if cur == 0:
                s0 = t
            cur += 1
        else:
            if cur > best[1] - best[0]:
                best = (s0, s0 + cur)
            cur = 0
    return best


def main():
    from src.mnist import load_mnist
    train_x, train_y, test_x, test_y = load_mnist(DATA)
    model = load_state_model(CKPT, device)
    cls = json.load(open(os.path.join(HERE, 'results', 's1_classification.json')))
    z1 = np.load(os.path.join(HERE, 'results', 's1_census.npz'))
    idx_a, idx_b = z1['idx_a'], z1['idx_b']
    pred_all = z1['pred'][0].reshape(45, N_PER, N_POINTS)
    stable = [(k, r) for k, r in enumerate(cls['transitions'])
              if r['seeds']['0']['stable_third']]

    ref_idx = {c: torch.where(train_y == c)[0][:N_REF_TRAIN] for c in range(10)}
    held_idx = {c: (torch.where(test_y[N_TEST_POOL:] == c)[0][:N_REF_TEST] + N_TEST_POOL)
                for c in range(10)}

    out = {'meta': {'ckpt': CKPT, 'ckpt_sha256': str(z1['ckpt_sha256'][0]),
                    'n_ref_train_per_digit': N_REF_TRAIN,
                    'n_heldout_test_per_digit': N_REF_TEST,
                    'std_floor': STD_FLOOR, 'note': 'same rules as s3_s4_regions.py, '
                    'only the hook point changes'}, 'layers': {}}

    for L in LAYERS:
        w = hidden_of(model, train_x[ref_idx[0]][:1], L).shape[1]
        mu = torch.zeros(10, w, device=device)
        sig = torch.zeros(10, w, device=device)
        for c in range(10):
            H = hidden_of(model, train_x[ref_idx[c]], L)
            mu[c], sig[c] = H.mean(0), H.std(0)
        glob = hidden_of(model, train_x[torch.cat([ref_idx[c] for c in range(10)])], L).std(0)
        sig = torch.maximum(sig, STD_FLOOR * glob[None]).clamp_min(1e-6)

        q95 = torch.zeros(10, device=device)
        held_ok, held_n = 0, 0
        for c in range(10):
            D = normalized_dist(hidden_of(model, test_x[held_idx[c]], L), mu, sig)
            q95[c] = torch.quantile(D[:, c], 0.95)
        for c in range(10):
            R = normalized_dist(hidden_of(model, test_x[held_idx[c]], L), mu, sig) / q95[None]
            held_ok += int(((R[:, c] < 1) & (R.argmin(1) == c)).sum())
            held_n += len(R)
        c1 = held_ok / held_n

        # control C3 at this layer: within-digit interpolation (patched at h1)
        pool_y = test_y[:N_TEST_POOL]
        c3 = []
        for c in range(10):
            cand = torch.where(pool_y == c)[0]
            n = len(cand)
            ia, ib = cand[:N_PER], cand[(np.arange(N_PER) + n // 2) % n]
            pth = slerp_batch(hidden_of(model, test_x[ia], 1),
                              hidden_of(model, test_x[ib], 1), N_POINTS)
            U = layer_of_path(model, pth, L)
            R = normalized_dist(U, mu, sig) / q95[None]
            c3.append(float((R[:, c] < 1).float().mean()))

        rows, pooled_in, pooled_tot = [], 0, 0
        for k, r in stable:
            a, b, zz = r['a'], r['b'], r['seeds']['0']['z_dominant']
            path = slerp_batch(hidden_of(model, test_x[idx_a[k]], 1),
                               hidden_of(model, test_x[idx_b[k]], 1), N_POINTS)
            U = layer_of_path(model, path, L)
            R = (normalized_dist(U, mu, sig) / q95[None]
                 ).reshape(N_PER, N_POINTS, 10).cpu().numpy()
            pin, ptot, pok, npaths, rz, rmin = 0, 0, 0, 0, [], []
            below, argz, nreg = 0, 0, 0
            for p in range(N_PER):
                if zz not in third_runs(pred_all[k, p], a, b):
                    continue
                s, e = longest_z_seg(pred_all[k, p], zz)
                Rs = R[p, s:e]
                ok = (Rs[:, zz] < 1) & (Rs.argmin(1) == zz)
                pin += int(ok.sum()); ptot += e - s
                pok += int(ok.mean() > 0.5); npaths += 1
                rz.append(float(np.median(Rs[:, zz])))
                rmin.append(float(np.median(Rs.min(1))))
                # decompose the two conditions, and count how many of the ten
                # regions each segment point falls inside at once
                below += int((Rs[:, zz] < 1).sum())
                argz += int((Rs.argmin(1) == zz).sum())
                nreg += int((Rs < 1).sum())
            pooled_in += pin; pooled_tot += ptot
            rows.append({'transition': r['transition'], 'z': zz,
                         'point_frac_in_z_region': round(pin / max(ptot, 1), 3),
                         'path_frac_majority_in_z_region': round(pok / max(npaths, 1), 3),
                         'point_frac_ratio_to_z_below_1': round(below / max(ptot, 1), 3),
                         'point_frac_z_is_nearest': round(argz / max(ptot, 1), 3),
                         'mean_n_regions_containing_point': round(nreg / max(ptot, 1), 2),
                         'median_ratio_to_z_on_segment': round(float(np.median(rz)), 3),
                         'median_min_ratio_on_segment': round(float(np.median(rmin)), 3)})
            del path, U
            if device == 'cuda':
                torch.cuda.empty_cache()
        out['layers'][f'h{L}'] = {
            'width': int(w), 'C1_heldout_both': round(c1, 3),
            'C3_within_digit_inside': [round(v, 3) for v in c3],
            'pooled_point_frac_in_z_region': round(pooled_in / pooled_tot, 4),
            'pooled_segment_points': int(pooled_tot), 'transitions': rows}
        print(f'h{L}: C1 {c1:.3f}  C3 mean {np.mean(c3):.3f}  '
              f'pooled points inside z region {pooled_in / pooled_tot:.4f}  '
              f"median R_z range {min(x['median_ratio_to_z_on_segment'] for x in rows):.2f}"
              f"-{max(x['median_ratio_to_z_on_segment'] for x in rows):.2f}")

    json.dump(out, open(os.path.join(HERE, 'results', 's6_later_layers.json'), 'w'), indent=1)
    print('saved results/s6_later_layers.json')
    figure(out)


@torch.no_grad()
def layer_of_path(model, h1_path, L):
    """Activations at post-ReLU layer L for an h1-patched path. -> [N,width]."""
    flat = h1_path.reshape(-1, h1_path.shape[-1])
    if L == 1:
        return flat
    _logits, hs = model.forward_from(flat, 1)
    return hs[L - 2]


def figure(out):
    keys = [t['transition'] for t in out['layers']['h1']['transitions']]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(3, 1, figsize=(0.75 * len(keys) + 4.5, 12.6))
    ax = axes[0]
    for i, L in enumerate(LAYERS):
        d = out['layers'][f'h{L}']
        ax.bar(x + (i - 1) * 0.27, [t['point_frac_in_z_region'] for t in d['transitions']],
               0.27, label=f"$h_{L}$ (pooled {d['pooled_point_frac_in_z_region']:.1%})")
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel('fraction of third-class segment points\ninside the real-$z$ activation region')
    ax.set_xlabel('stable third-class transition (seed 0)')
    ax.set_title('Does the segment enter the predicted digit\'s activation region at a later '
                 'layer?', fontsize=12)
    ax.legend(title='hook point', fontsize=9)
    ax = axes[1]
    for i, L in enumerate(LAYERS):
        d = out['layers'][f'h{L}']
        ax.plot(x, [t['median_ratio_to_z_on_segment'] for t in d['transitions']],
                'o-', color=f'C{i}', label=f'$h_{L}$: distance to predicted digit $z$')
        ax.plot(x, [t['median_min_ratio_on_segment'] for t in d['transitions']],
                'v--', color=f'C{i}', alpha=0.55,
                label=f'$h_{L}$: distance to the nearest of all ten digits')
    ax.axhline(1, ls=':', c='k', lw=1.2)
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel('median normalized distance ratio\non the segment (log scale)')
    ax.set_xlabel('stable third-class transition (seed 0)')
    ax.set_title('Distance ratios on the same frozen segments at each hook point '
                 '(dotted line = 1, the boundary of the usual spread of real images)',
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax = axes[2]
    for i, L in enumerate(LAYERS):
        d = out['layers'][f'h{L}']
        ax.bar(x + (i - 1) * 0.27,
               [t['mean_n_regions_containing_point'] for t in d['transitions']], 0.27,
               label=f'$h_{L}$')
    ax.axhline(1, ls=':', c='k', lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel('mean number of the ten digit regions\nthat contain the segment point')
    ax.set_xlabel('stable third-class transition (seed 0)')
    ax.set_title('How discriminative is "inside a region" at each hook point? '
                 '(10 = inside every digit\'s region at once)', fontsize=11)
    ax.legend(title='hook point', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 's6_later_layers.png'), dpi=140)
    plt.close(fig)
    print('saved plots/s6_later_layers.png')


if __name__ == '__main__':
    main()
