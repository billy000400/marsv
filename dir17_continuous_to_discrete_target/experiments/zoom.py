"""Operator feedback #2 — what does movement look like INSIDE the transition?

The 201-point probe grid (spacing 0.003) is coarser than the k=320 target's
transition (width 0.0046), so the main figures cannot show what happens while
the target is actually switching. This script re-runs the brightness sweep on a
30x finer uniform grid (6001 points, spacing 1e-4) and stores, per model:

  g_l(b_i) = s_l(b_i) * (S-1)                       movement rate, 1 = uniform
  Gamma_l(w) = share of movement in [b0-w, b0+w] / (2w/0.6)   scale-resolved
  pi_l       = max of g_l over the central window [0.64, 0.76]

Curves go to results/zoom_curves_<tag>.npz, scalars to results/zoom_<tag>.json.
"""
import json
import os
import sys

import numpy as np
import torch

import common as C

N_DENSE = 6001                       # spacing 1e-4 over [0.4, 1.0]
WINDOWS = [0.06, 0.03, 0.01, 0.005, 0.0025]   # half-widths around b0
SEEDS = [0, 1, 2]

C.setup_torch()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def dense_grid():
    return np.linspace(C.B_LO, C.B_HI, N_DENSE)


@torch.no_grad()
def dense_sweep(model, x_unit, grid, device, chunk=100):
    """Streaming dense sweep: returns preds [N,S] and movement [L][N,S-1].

    Activations are never stored for the whole grid — only the previous chunk's
    last column is kept, so RAM stays at the movement arrays (~7 MB).
    """
    x_unit = x_unit.to(device)
    n, s = x_unit.shape[0], len(grid)
    preds = np.empty((n, s), dtype=np.float32)
    mov = [np.empty((n, s - 1), dtype=np.float32) for _ in range(C.DEPTH - 1)]
    gt = torch.from_numpy(grid).float().to(device)
    prev = None                                    # [L] of [N, W] at last b
    for j0 in range(0, s, chunk):
        bs = gt[j0:j0 + chunk]
        m = len(bs)
        x = (x_unit[:, None, :] * bs[None, :, None]).reshape(n * m, 784)
        hs, out = model.hidden_activations(x)
        preds[:, j0:j0 + m] = out.reshape(n, m).cpu().numpy()
        hh = [hs[l].reshape(n, m, C.WIDTH) for l in range(C.DEPTH - 1)]
        for l, h in enumerate(hh):
            if m > 1:
                d = torch.linalg.norm(h[:, 1:] - h[:, :-1], dim=2)   # [N, m-1]
                mov[l][:, j0:j0 + m - 1] = d.cpu().numpy()
            if prev is not None:                   # step across the chunk seam
                mov[l][:, j0 - 1] = torch.linalg.norm(
                    h[:, 0] - prev[l], dim=1).cpu().numpy()
        prev = [h[:, -1].clone() for h in hh]
        del hs, out, x, hh
    return preds, mov


def sliding_max_gain(s, n_win):
    """Alignment-free concentration: for each image, the largest share of its
    movement inside ANY window of n_win consecutive steps, divided by that
    window's uniform share. Same scale as gamma (1 = uniform), but it does not
    require the sharp turn to sit at b0."""
    cs = np.concatenate([np.zeros((s.shape[0], 1)), np.cumsum(s, axis=1)], axis=1)
    win = cs[:, n_win:] - cs[:, :-n_win]                  # [N, S-1-n_win+1]
    return win.max(axis=1) / (n_win / s.shape[1])


def reduce_curve(m):
    """m [N, S-1] raw movement -> per-image normalized share, then metrics."""
    s = m / (m.sum(axis=1, keepdims=True) + C.EPS)
    b = dense_grid()[:-1]
    out = {'curve': s.mean(axis=0).astype(np.float32)}
    for w in WINDOWS:
        sel = (b >= C.B0 - w - 1e-12) & (b < C.B0 + w - 1e-12)
        share = s[:, sel].sum(axis=1)
        gain = share / (sel.mean())              # 1 = uniform, max = 1/sel.mean()
        out[f'gamma_{w:g}'] = float(gain.mean())
        out[f'gamma_{w:g}_sd_img'] = float(gain.std())
        lam = sliding_max_gain(s, int(sel.sum()))
        out[f'lambda_{w:g}'] = float(lam.mean())
        out[f'lambda_{w:g}_sd_img'] = float(lam.std())
    cen = (b >= C.CENTER_LO - 1e-12) & (b < C.CENTER_HI - 1e-12)
    out['peak'] = float((s.mean(axis=0)[cen] * (len(b))).max())
    fl = (b < C.FLANK_LO - 1e-12) | (b >= C.FLANK_HI - 1e-12)
    out['phi'] = float(s[:, fl].sum(axis=1).mean())
    return out


def target_entries():
    b = dense_grid()
    out, curves = {}, {}
    for k in C.K_VALUES:
        m = np.abs(np.diff(C.target_fn(b, k)))[None, :]
        r = reduce_curve(m)
        curves[f'target_k{k:g}'] = r.pop('curve')
        out[f'k{k:g}'] = r
    return out, curves


def run(prefix, tag):
    data = C.build_dataset(0)
    x_unit = data['xte_unit']
    grid = dense_grid()
    scal, curves = {}, {}
    for k in C.K_VALUES:
        for seed in SEEDS:
            ck = torch.load(os.path.join(C.RESULTS, f"{prefix}_k{k:g}_s{seed}.pt"),
                            map_location='cpu', weights_only=False)
            model = C.MLP(n_out=1).to(DEVICE)
            model.load_state_dict(ck['final'])
            model.eval()
            preds, mov = dense_sweep(model, x_unit, grid, DEVICE)
            del model
            torch.cuda.empty_cache()

            key = f"k{k:g}|s{seed}"
            e = {}
            r = reduce_curve(np.abs(np.diff(preds, axis=1)))
            curves[f'{key}|out'] = r.pop('curve')
            e['out'] = r
            for l in range(C.DEPTH - 1):
                r = reduce_curve(mov[l])
                curves[f'{key}|L{l+1}'] = r.pop('curve')
                e[f'L{l+1}'] = r
            scal[key] = e
            print(f"[{tag}] k={k:g} s={seed}: "
                  f"L3 gamma(0.06)={e['L3']['gamma_0.06']:.3f} "
                  f"gamma(0.005)={e['L3']['gamma_0.005']:.3f} "
                  f"peak={e['L3']['peak']:.2f} | out peak={e['out']['peak']:.1f}",
                  flush=True)
            del preds, mov
    return scal, curves


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'main'
    prefix = 'ckpt' if tag == 'main' else 'ckpt10k'
    scal, curves = run(prefix, tag)
    tscal, tcurves = target_entries()
    with open(os.path.join(C.RESULTS, f'zoom_{tag}.json'), 'w') as f:
        json.dump({'models': scal, 'target': tscal, 'windows': WINDOWS,
                   'n_dense': N_DENSE, 'seeds': SEEDS}, f)
    np.savez_compressed(os.path.join(C.RESULTS, f'zoom_curves_{tag}.npz'),
                        grid=dense_grid().astype(np.float32),
                        **curves, **tcurves)
    print('saved results/zoom_%s.json + zoom_curves_%s.npz' % (tag, tag))
