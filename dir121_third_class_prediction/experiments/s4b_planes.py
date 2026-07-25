"""
S4b — two complementary 2-D views of the third-class segments, replacing the
earlier PCA view (PLAN.md "Two complementary two-dimensional visualizations").

View A  real-class LDA plane. Three-class linear discriminant analysis fitted on
        REAL TRAINING h1 activations of the two endpoint digits a, b and the
        dominant third digit z. Held-out real test activations and all 100
        interpolation paths are projected without refitting. Answers: does the
        path approach the directions that best separate real a/b/z activations?

View B  path-local margin-gradient/SVD decision slice. At every point of the
        representative path's z-segment (plus PAD points either side, so the
        entry and exit boundaries are included) take grad_h(l_a - l_z) and
        grad_h(l_b - l_z); stack them, SVD, keep the top two right-singular
        vectors as an orthonormal plane anchored at the segment mean. Evaluate
        the network on a grid in that plane and colour by predicted class.
        Answers: what does the model's local decision geometry look like there?

Both views are visualization; the conclusion still rests on the full 200-d
distances from S3/S4. Frozen rules (recorded in JOURNAL.md before running):
  * representative path = medoid, by Euclidean distance between full 50x200 h1
    trajectories, among the paths that contain a dominant-z segment;
  * LDA on 2,000 real training images per class, ridge 1e-3 * tr(S_W)/200;
  * View B grid 161x161 spanning the projected path and the projected held-out
    real class means, padded 30%; grid points are NEVER clamped to h>=0.

Outputs results/s4b_planes.json and four figures in plots/.
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, '/workspace/mars-plateaus-image')
sys.path.insert(0, '/workspace/marsv_agent_haoyang/dir12_plateau_during_training/experiments')
from plateau_protocol import slerp_batch, load_state_model, N_POINTS, N_TEST_POOL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvd_style import CVD, MARKERS, use_cvd, digit_color, digit_text_color

use_cvd()

HERE = '/workspace/marsv_agent_haoyang/dir121_third_class_prediction'
DIR12 = '/workspace/marsv_agent_haoyang/dir12_plateau_during_training'
DATA = '/workspace/mars-plateaus-image/data/mnist'
PLOTS = os.path.join(HERE, 'plots')
CKPT = os.path.join(DIR12, 'results', 'full_mnist_from_scratch', 'seed_0',
                    'ckpts', 'step30000.pt')

N_REF_TRAIN, N_REF_TEST, N_PER = 2000, 700, 100
RIDGE = 1e-3          # S_W ridge, as a fraction of tr(S_W)/d
PAD = 5               # alpha points kept either side of the z-segment (View B)
GRID = 161            # View B grid resolution per axis
GRID_PAD = 0.30       # window padding beyond path + real class means
ELLIPSE_SD = 2.0      # LDA spread ellipse radius, in standard deviations
FEATURE = '6->9'      # cross-seed-stable case featured in the main report

torch.set_num_threads(2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.225)

alpha_grid = np.linspace(0, 1, N_POINTS)


@torch.no_grad()
def h1_of(model, x):
    return model.hidden_activations(x.to(device))[0][0]


# ---------------------------------------------------------------- View A -----
def lda_axes(Hs):
    """Three-class LDA on real training activations. Hs: {class: [n,d] float64}.

    Returns global mean, [d,2] axis matrix (unit-norm columns), eigenvalues.
    """
    X = np.concatenate([Hs[c] for c in Hs])
    d = X.shape[1]
    mu_g = X.mean(0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for H in Hs.values():
        mc = H.mean(0)
        Xc = H - mc
        Sw += Xc.T @ Xc
        dm = (mc - mu_g)[:, None]
        Sb += len(H) * (dm @ dm.T)
    Sw /= len(X)
    Sb /= len(X)
    Sw = Sw + RIDGE * np.trace(Sw) / d * np.eye(d)
    L = np.linalg.cholesky(Sw)
    Li = np.linalg.inv(L)
    M = Li @ Sb @ Li.T
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    order = np.argsort(w)[::-1][:2]
    W = Li.T @ V[:, order]
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return mu_g, W, w[order]


def ellipse_params(P):
    """Mean, 2x2 covariance and the Mahalanobis radius test for 2-D points P."""
    m = P.mean(0)
    C = np.cov(P.T)
    Ci = np.linalg.inv(C)
    ev, evec = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
    return m, C, Ci, 2 * ELLIPSE_SD * np.sqrt(ev[::-1]), ang


def inside_ellipse(P, m, Ci):
    D = P - m
    return (np.einsum('ij,jk,ik->i', D, Ci, D) <= ELLIPSE_SD ** 2)


# ---------------------------------------------------------------- View B -----
def margin_grads(model, H, a, b, z):
    """grad_h(l_a - l_z) and grad_h(l_b - l_z) at each row of H. -> [m,d] x2."""
    with torch.enable_grad():
        Hg = H.detach().clone().requires_grad_(True)
        logits, _ = model.forward_from(Hg, 1)
        ga = torch.autograd.grad((logits[:, a] - logits[:, z]).sum(), Hg,
                                 retain_graph=True)[0]
        gb = torch.autograd.grad((logits[:, b] - logits[:, z]).sum(), Hg)[0]
    return ga.detach(), gb.detach()


@torch.no_grad()
def eval_grid(model, anchor, axes_t, us, vs):
    """Predicted class and the two margins on the plane grid (never clamped)."""
    U, V = np.meshgrid(us, vs)                                  # [G,G]
    coords = torch.tensor(np.stack([U.ravel(), V.ravel()], 1),
                          dtype=torch.float32, device=device)
    H = anchor[None] + coords @ axes_t.T                        # [G*G,200]
    # off-ReLU-support diagnostics: the strict "any negative coordinate" test
    # saturates in 200 dimensions, so report two graded companions as well.
    off = {'frac_cells_any_negative': round(float((H < 0).any(dim=1).float().mean()), 3),
           'mean_frac_coords_negative': round(float((H < 0).float().mean()), 3),
           'median_negative_norm_share':
               round(float(torch.median(H.clamp(max=0).norm(dim=1) / H.norm(dim=1))), 3)}
    chunks = []
    for s in range(0, len(H), 8192):
        lg, _ = model.forward_from(H[s:s + 8192], 1)
        chunks.append(lg.cpu().numpy())
    logits = np.concatenate(chunks).reshape(GRID, GRID, 10)
    return logits.argmax(2), logits, off, U, V


def main():
    from src.mnist import load_mnist
    train_x, train_y, test_x, test_y = load_mnist(DATA)
    model = load_state_model(CKPT, device)

    cls = json.load(open(os.path.join(HERE, 'results', 's1_classification.json')))
    reg = json.load(open(os.path.join(HERE, 'results', 's3_s4_regions.json')))
    reg_by_t = {r['transition']: r for r in reg['transitions']}
    z1 = np.load(os.path.join(HERE, 'results', 's1_census.npz'))
    idx_a, idx_b = z1['idx_a'], z1['idx_b']
    pred_all = z1['pred'][0].reshape(45, N_PER, N_POINTS)
    stable = [(k, r) for k, r in enumerate(cls['transitions'])
              if r['seeds']['0']['stable_third']]
    print(f'{len(stable)} stable third-class transitions')

    ref_idx = {c: torch.where(train_y == c)[0][:N_REF_TRAIN] for c in range(10)}
    held_idx = {c: (torch.where(test_y[N_TEST_POOL:] == c)[0][:N_REF_TEST]
                    + N_TEST_POOL) for c in range(10)}

    out, panels = [], {}
    for k, r in stable:
        a, b, z = r['a'], r['b'], r['seeds']['0']['z_dominant']
        key = r['transition']

        # ---- paths and the frozen medoid rule -------------------------------
        hA, hB = h1_of(model, test_x[idx_a[k]]), h1_of(model, test_x[idx_b[k]])
        path = slerp_batch(hA, hB, N_POINTS)                    # [100,50,200]
        segs = {}
        for p in range(N_PER):
            best, cur, s0 = (0, 0), 0, 0
            for t in range(N_POINTS + 1):
                if t < N_POINTS and pred_all[k, p, t] == z:
                    if cur == 0:
                        s0 = t
                    cur += 1
                else:
                    if cur > best[1] - best[0]:
                        best = (s0, s0 + cur)
                    cur = 0
            if best[1] > best[0]:
                segs[p] = best
        cand = sorted(segs)
        flat = path[cand].reshape(len(cand), -1)
        D = torch.cdist(flat, flat)
        med = cand[int(D.sum(1).argmin())]
        s_lo, s_hi = segs[med]                                  # [s_lo, s_hi)
        print(f'{key}: z={z}, {len(cand)} paths with a z-segment, medoid pair '
              f'{med}, segment alpha points [{s_lo},{s_hi})')

        # ---- View A: LDA on real training activations only -------------------
        Htr = {c: h1_of(model, train_x[ref_idx[c]]).double().cpu().numpy()
               for c in (a, b, z)}
        mu_g, W, evals = lda_axes(Htr)
        Wt = torch.tensor(W, dtype=torch.float32, device=device)
        mu_gt = torch.tensor(mu_g, dtype=torch.float32, device=device)

        def projA(T):
            return ((T - mu_gt) @ Wt).cpu().numpy()

        held = {c: h1_of(model, test_x[held_idx[c]]) for c in (a, b, z)}
        heldA = {c: projA(held[c]) for c in (a, b, z)}
        pathA = projA(path.reshape(-1, 200)).reshape(N_PER, N_POINTS, 2)
        ell = {c: ellipse_params(heldA[c]) for c in (a, b, z)}
        # how often do real held-out images land in their own LDA ellipse?
        ell_calib = {int(c): round(float(inside_ellipse(heldA[c], ell[c][0],
                                                        ell[c][2]).mean()), 3)
                     for c in (a, b, z)}
        # the 2-D analogue of the full-space region test, over ALL z-segments
        segpts = np.concatenate([pathA[p, lo:hi] for p, (lo, hi) in segs.items()])
        in_z = float(inside_ellipse(segpts, ell[z][0], ell[z][2]).mean())
        in_a = float(inside_ellipse(segpts, ell[a][0], ell[a][2]).mean())
        in_b = float(inside_ellipse(segpts, ell[b][0], ell[b][2]).mean())

        # ---- View B: path-local margin-gradient/SVD plane ---------------------
        lo = max(0, s_lo - PAD)
        hi = min(N_POINTS, s_hi + PAD)
        Hsel = path[med, lo:hi]                                  # [m,200]
        ga, gb = margin_grads(model, Hsel, a, b, z)
        G = torch.cat([ga, gb], 0).double().cpu().numpy()        # [2m,200]
        _u, sv, Vt = np.linalg.svd(G, full_matrices=False)
        energy = float((sv[:2] ** 2).sum() / (sv ** 2).sum())
        axes_np = Vt[:2].T                                       # [200,2]
        axes_t = torch.tensor(axes_np, dtype=torch.float32, device=device)
        anchor = path[med, s_lo:s_hi].mean(0)                    # segment mean

        def projB(T):
            return ((T - anchor[None]) @ axes_t).cpu().numpy()

        pathB = projB(path[med])                                 # [50,2]
        heldB = {c: projB(held[c]) for c in (a, b, z)}
        meansB = np.stack([heldB[c].mean(0) for c in (a, b, z)])
        span = np.concatenate([pathB, meansB])
        lo2, hi2 = span.min(0), span.max(0)
        pad = GRID_PAD * np.maximum(hi2 - lo2, 1e-6)
        us = np.linspace(lo2[0] - pad[0], hi2[0] + pad[0], GRID)
        vs = np.linspace(lo2[1] - pad[1], hi2[1] + pad[1], GRID)
        pred_grid, logit_grid, off_frac, U, V = eval_grid(model, anchor, axes_t,
                                                          us, vs)
        m_ac = logit_grid[:, :, a] - logit_grid[:, :, z]
        m_bc = logit_grid[:, :, b] - logit_grid[:, :, z]
        vis = {int(c): round(float(((heldB[c][:, 0] >= us[0]) & (heldB[c][:, 0] <= us[-1]) &
                                    (heldB[c][:, 1] >= vs[0]) & (heldB[c][:, 1] <= vs[-1])
                                    ).mean()), 3) for c in (a, b, z)}
        # how much of each projected object actually lies IN the plane: a point
        # drawn at (u,v) still carries a discarded out-of-plane component.
        def in_plane(T):
            Dv = T - anchor[None]
            return round(float(torch.median(
                (Dv @ axes_t).norm(dim=1) / Dv.norm(dim=1).clamp_min(1e-9)) ** 2), 3)
        inplane = {'medoid_path': in_plane(path[med]),
                   **{f'real_{int(c)}': in_plane(held[c]) for c in (a, b, z)}}
        # Does the DRAWN path agree with the real one? Re-classify each path
        # point after collapsing it into the plane; if the two disagree, the
        # picture would show the path in the wrong decision region.
        with torch.no_grad():
            Dv = path[med] - anchor[None]
            proj_h = anchor[None] + (Dv @ axes_t) @ axes_t.T
            lg_proj, _ = model.forward_from(proj_h, 1)
            pred_proj = lg_proj.argmax(1).cpu().numpy()
        pred_true = pred_all[k, med]
        fid = {'all_points': round(float((pred_proj == pred_true).mean()), 3),
               'segment_points': round(float((pred_proj[s_lo:s_hi] ==
                                              pred_true[s_lo:s_hi]).mean()), 3),
               'segment_points_predicted_z': round(float(
                   (pred_proj[s_lo:s_hi] == z).mean()), 3)}

        rec = {'transition': key, 'a': a, 'b': b, 'z': z,
               'medoid_pair_rank': int(med),
               'medoid_idx_a': int(idx_a[k][med]), 'medoid_idx_b': int(idx_b[k][med]),
               'medoid_segment_points': [int(s_lo), int(s_hi)],
               'medoid_segment_alpha': [round(float(alpha_grid[s_lo]), 3),
                                        round(float(alpha_grid[s_hi - 1]), 3)],
               'n_paths_with_segment': len(cand),
               'lda_eigenvalues': [round(float(v), 4) for v in evals],
               'lda_heldout_own_ellipse_frac': ell_calib,
               'lda_segment_points_in_z_ellipse': round(in_z, 3),
               'lda_segment_points_in_a_ellipse': round(in_a, 3),
               'lda_segment_points_in_b_ellipse': round(in_b, 3),
               'fullspace_point_frac_in_z_region':
                   reg_by_t[key]['point_frac_in_z_region'],
               'grad_singular_values': [round(float(v), 4) for v in sv[:4]],
               'grad_two_axis_energy': round(energy, 3),
               'grid_off_relu_support': off_frac,
               'grid_frac_predicted_z': round(float((pred_grid == z).mean()), 3),
               'grid_heldout_real_visible_frac': vis,
               'median_in_plane_energy_share': inplane,
               'projected_path_prediction_fidelity': fid}
        out.append(rec)
        print(f'   LDA: seg pts in z-ellipse {in_z:.3f} (full space '
              f"{rec['fullspace_point_frac_in_z_region']:.3f}); "
              f'SVD: 2-axis energy {energy:.3f}, off-ReLU '
              f"{off_frac['mean_frac_coords_negative']:.3f} coords / "
              f"{off_frac['median_negative_norm_share']:.3f} norm, "
              f'grid predicted z {rec["grid_frac_predicted_z"]:.3f}, '
              f'in-plane share {inplane}, projected-path fidelity {fid}')

        panels[key] = {'heldA': {c: heldA[c][::4] for c in (a, b, z)}, 'ell': ell,
                       'pathA': pathA[cand[:24]], 'medA': pathA[med],
                       'predA': pred_all[k, cand[:24]], 'seg': (s_lo, s_hi),
                       'us': us, 'vs': vs, 'pred_grid': pred_grid,
                       'm_ac': m_ac, 'm_bc': m_bc, 'pathB': pathB,
                       'heldB': {c: heldB[c][::6] for c in (a, b, z)},
                       'a': a, 'b': b, 'z': z, 'rec': rec}
        del path
        if device == 'cuda':
            torch.cuda.empty_cache()

    res = {'meta': {'ckpt': CKPT, 'ckpt_sha256': str(z1['ckpt_sha256'][0]),
                    'lda_train_per_class': N_REF_TRAIN, 'lda_ridge': RIDGE,
                    'heldout_per_class': N_REF_TEST, 'ellipse_sd': ELLIPSE_SD,
                    'gradient_pad_points': PAD, 'grid': GRID,
                    'grid_pad_frac': GRID_PAD,
                    'medoid_rule': 'min sum of Euclidean distances between full '
                                   '50x200 h1 trajectories, among paths with a '
                                   'dominant-z segment'},
           'transitions': out}
    json.dump(res, open(os.path.join(HERE, 'results', 's4b_planes.json'), 'w'),
              indent=1)
    print('saved results/s4b_planes.json')
    figures(panels, out)


# ------------------------------------------------------------------ plots ----
def draw_lda(ax, P, fontsize=8, legend=True):
    a, b, z = P['a'], P['b'], P['z']
    for i, c in enumerate((a, b, z)):
        pts = P['heldA'][c]
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.25, color=CVD[i],
                   marker=MARKERS[i], linewidths=0)
        m, _C, _Ci, wh, ang = P['ell'][c]
        ax.add_patch(Ellipse(m, wh[0], wh[1], angle=ang, fill=False,
                             edgecolor=CVD[i], lw=1.8,
                             ls=['-', '--', '-.'][i], zorder=4))
        ax.scatter(*m, marker='*', s=220, color=CVD[i], edgecolor='k', zorder=6)
    for p in P['pathA']:
        ax.plot(p[:, 0], p[:, 1], c='0.45', lw=0.6, alpha=0.7, zorder=2)
    msk = P['predA'] == z
    ax.scatter(P['pathA'][msk][:, 0], P['pathA'][msk][:, 1], s=9, color='k',
               marker='x', lw=0.7, zorder=5)
    ax.plot(P['medA'][:, 0], P['medA'][:, 1], c='k', lw=2.0, zorder=7)
    s_lo, s_hi = P['seg']
    ax.scatter(P['medA'][[s_lo, s_hi - 1], 0], P['medA'][[s_lo, s_hi - 1], 1],
               s=90, facecolor='none', edgecolor='k', lw=2.0, zorder=8)
    ax.set_xlabel('LDA discriminant axis 1', fontsize=fontsize)
    ax.set_ylabel('LDA discriminant axis 2', fontsize=fontsize)
    if legend:
        hs = [Line2D([], [], color=CVD[i], marker=MARKERS[i], ls=['-', '--', '-.'][i],
                     label=f'real digit {c} ({lab}); ellipse = {ELLIPSE_SD:g} s.d.')
              for i, (c, lab) in enumerate(((a, 'endpoint a'), (b, 'endpoint b'),
                                            (z, 'third digit z')))]
        hs += [Line2D([], [], color='0.45', lw=0.8, label='interpolation paths (24 of 100)'),
               Line2D([], [], color='k', marker='x', ls='none',
                      label=f'point predicted {z}'),
               Line2D([], [], color='k', lw=2, label='medoid path'),
               Line2D([], [], color='k', marker='o', mfc='none', ls='none',
                      label='medoid segment entry / exit')]
        ax.legend(handles=hs, fontsize=fontsize - 1.5, loc='best', framealpha=0.85)


def draw_margin(ax, P, fontsize=8, legend=True):
    a, b, z = P['a'], P['b'], P['z']
    cmap = ListedColormap([digit_color(c) for c in range(10)])
    ax.pcolormesh(P['us'], P['vs'], P['pred_grid'], cmap=cmap,
                  norm=BoundaryNorm(np.arange(-0.5, 10), 10), shading='auto',
                  zorder=0, rasterized=True)
    # thin outlines between decision regions: the sequential ramp alone cannot
    # separate neighbouring digits, so every region also gets a drawn border.
    ax.contour(P['us'], P['vs'], P['pred_grid'].astype(float),
               levels=np.arange(0.5, 9.5), colors='0.15', linewidths=0.7, zorder=2)
    for c in range(10):
        m = P['pred_grid'] == c
        if m.mean() < 0.02:
            continue
        yi, xi = np.nonzero(m)
        ax.text(P['us'][int(np.median(xi))], P['vs'][int(np.median(yi))], str(c),
                color=digit_text_color(c), fontsize=fontsize + 3, fontweight='bold',
                ha='center', va='center', zorder=3)
    # the third digit's decision region additionally carries a hatch, so the
    # region the whole figure is about never depends on hue alone
    ax.contourf(P['us'], P['vs'], (P['pred_grid'] == z).astype(float),
                levels=[0.5, 1.5], colors='none', hatches=['//'], zorder=1)
    ax.contour(P['us'], P['vs'], P['m_ac'], levels=[0], colors='k',
               linestyles='-', linewidths=2.0, zorder=4)
    ax.contour(P['us'], P['vs'], P['m_bc'], levels=[0], colors='k',
               linestyles='--', linewidths=2.0, zorder=4)
    for i, c in enumerate((a, b, z)):
        pts = P['heldB'][c]
        ax.scatter(pts[:, 0], pts[:, 1], s=7, alpha=0.55, color=CVD[i],
                   marker=MARKERS[i], edgecolor='k', linewidths=0.2, zorder=5)
    pb = P['pathB']
    ax.plot(pb[:, 0], pb[:, 1], c='k', lw=2.2, zorder=6)
    s_lo, s_hi = P['seg']
    ax.scatter(pb[[s_lo, s_hi - 1], 0], pb[[s_lo, s_hi - 1], 1], s=110,
               facecolor='none', edgecolor='k', lw=2.2, zorder=7)
    ax.set_xlim(P['us'][0], P['us'][-1]); ax.set_ylim(P['vs'][0], P['vs'][-1])
    ax.set_xlabel('margin-gradient axis 1 (top right-singular vector)', fontsize=fontsize)
    ax.set_ylabel('margin-gradient axis 2', fontsize=fontsize)
    if legend:
        hs = [Line2D([], [], color='k', lw=2, ls='-', label=f'$l_{{{a}}}-l_{{{z}}}=0$ (a/z boundary)'),
              Line2D([], [], color='k', lw=2, ls='--', label=f'$l_{{{b}}}-l_{{{z}}}=0$ (b/z boundary)'),
              Patch(facecolor=digit_color(z), hatch='//', edgecolor='k',
                    label=f'cells predicted {z} (third digit z, hatched)')]
        hs += [Line2D([], [], color=CVD[i], marker=MARKERS[i], ls='none',
                      label=f'real held-out digit {c}')
               for i, c in enumerate((a, b, z))]
        hs += [Line2D([], [], color='k', lw=2, label='medoid path (projected)'),
               Line2D([], [], color='k', marker='o', mfc='none', ls='none',
                      label='segment entry / exit')]
        ax.legend(handles=hs, fontsize=fontsize - 1.5, loc='best', framealpha=0.85)


def figures(panels, out):
    # 1 — featured cross-seed-stable case, both views side by side
    P = panels[FEATURE]
    rec = P['rec']
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
    draw_lda(axes[0], P, fontsize=9)
    axes[0].set_title(f"View A — real-class LDA plane, {FEATURE} (z={P['z']})\n"
                      f"segment points inside the {ELLIPSE_SD:g} s.d. real-{P['z']} "
                      f"ellipse: {rec['lda_segment_points_in_z_ellipse']:.1%}  "
                      f"(full 200-d space: {rec['fullspace_point_frac_in_z_region']:.1%})",
                      fontsize=11)
    draw_margin(axes[1], P, fontsize=9)
    axes[1].set_title(f"View B — path-local margin-gradient/SVD decision slice, {FEATURE}\n"
                      f"two axes hold {rec['grad_two_axis_energy']:.1%} of the squared gradient "
                      f"norm; every grid cell is off the post-ReLU support\n"
                      f"({rec['grid_off_relu_support']['mean_frac_coords_negative']:.1%} of "
                      f"coordinates negative, "
                      f"{rec['grid_off_relu_support']['median_negative_norm_share']:.1%} of "
                      "the norm)", fontsize=11)
    fig.suptitle('Two 2-D views of the stable third-class segment on the '
                 'cross-seed-stable 6→9 transition (medoid path).\nView A is supervised by '
                 'REAL class labels; View B is supervised by the MODEL\'s local margins. '
                 'The conclusion still comes from the full 200-d distances.', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(os.path.join(PLOTS, 's4b_feature_6to9.png'), dpi=140)
    plt.close(fig)

    # 2/3 — contact sheets over all 19 stable transitions
    keys = [o['transition'] for o in out]
    for name, draw, title in (
            ('s4b_lda_contact', draw_lda,
             'View A — real-class LDA planes, all 19 seed-0 stable third-class transitions'),
            ('s4b_margin_contact', draw_margin,
             'View B — path-local margin-gradient/SVD decision slices, all 19 transitions')):
        ncol, nrow = 5, int(np.ceil(len(keys) / 5))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.7 * nrow),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for n, key in enumerate(keys):
            ax = axes[n // ncol, n % ncol]; ax.axis('on')
            draw(ax, panels[key], fontsize=7, legend=False)
            r = panels[key]['rec']
            extra = (f"in-ellipse {r['lda_segment_points_in_z_ellipse']:.0%} | "
                     f"full-space {r['fullspace_point_frac_in_z_region']:.0%}"
                     if name.endswith('lda_contact') else
                     f"energy {r['grad_two_axis_energy']:.0%} | neg. coords "
                     f"{r['grid_off_relu_support']['mean_frac_coords_negative']:.0%}")
            ax.set_title(f"{key}  z={r['z']}\n{extra}", fontsize=8.5)
            ax.set_xlabel(''); ax.set_ylabel('')
            ax.tick_params(labelsize=6)
        draw(axes[0, 0], panels[keys[0]], fontsize=7, legend=True)
        fig.suptitle(title + '.\nAxes, markers and boundaries are as in the featured '
                     'figure; the legend in the first panel applies to all panels '
                     '(digit identities differ per panel, see each title).', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(PLOTS, f'{name}.png'), dpi=110)
        plt.close(fig)

    # 4 — diagnostics required by PLAN.md, plus the 2-D vs full-space comparison
    x = np.arange(len(keys))
    fig, axes = plt.subplots(2, 1, figsize=(0.62 * len(keys) + 4, 8.2), sharex=True)
    ax = axes[0]
    ax.bar(x - 0.2, [o['lda_segment_points_in_z_ellipse'] for o in out], 0.4,
           color=CVD[0], hatch='//', edgecolor='w',
           label=f'inside the {ELLIPSE_SD:g} s.d. real-$z$ ellipse in the LDA plane (2-D)')
    ax.bar(x + 0.2, [o['fullspace_point_frac_in_z_region'] for o in out], 0.4,
           color=CVD[1], hatch='\\\\', edgecolor='w',
           label='inside the real-$z$ region in the full 200-d space (S4)')
    ax.set_ylabel('fraction of third-class\nsegment points')
    ax.set_ylim(0, 0.25)
    ax.legend(fontsize=8.5)
    ax.set_title('The supervised 2-D LDA plane does not rescue the null: third-class '
                 'segments stay outside the real-$z$ ellipse', fontsize=11)
    ax = axes[1]
    ax.bar(x - 0.2, [o['grad_two_axis_energy'] for o in out], 0.4, color=CVD[0],
           hatch='//', edgecolor='w',
           label='fraction of squared margin-gradient norm held by the two axes')
    ax.bar(x + 0.2, [o['grid_off_relu_support']['mean_frac_coords_negative']
                     for o in out], 0.4, color=CVD[2], hatch='xx', edgecolor='w',
           label='mean fraction of the 200 $h_1$ coordinates that are negative on the grid '
                 '(every cell has at least one)')
    ax.plot(x, [o['grid_frac_predicted_z'] for o in out], 'D-', c='k', ms=5,
            label='fraction of plotted grid cells predicted $z$ (solid, diamonds)')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('fraction')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\nz={o['z']}" for k, o in zip(keys, out)], fontsize=8)
    ax.set_xlabel('stable third-class transition')
    ax.legend(fontsize=8.5)
    ax.set_title('View B plane diagnostics', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 's4b_plane_diagnostics.png'), dpi=150)
    plt.close(fig)
    print('saved 4 figures to plots/')


if __name__ == '__main__':
    main()
