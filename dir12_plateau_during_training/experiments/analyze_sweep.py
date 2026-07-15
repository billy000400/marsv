#!/usr/bin/env python3
"""
dir12 — Measure plateau emergence over training checkpoints.

Frozen protocol (locked before sweep; see REPORT Methods):
  Layers   : perturb first hidden post-ReLU h1 (200-d); measure downstream at
             last hidden L3 (200-d) via model.forward_from(h1, layer=1).
  Response : R_t(x, rho) = median_u ||G_t(h1 + rho*||h1||*u) - G_t(h1)||_2
                                    / (||G_t(h1)||_2 + eps)
             over N_DIRS fixed random unit directions u, radius grid rho.
  Control  : norm-and-sparsity-matched random h1 (positive-entry count + norm of
             each eval example's h1), same directions/radii.
  Scalar   : plateau_contrast = 1 - AUC(R_data)/AUC(R_random) on rho in
             [0, RHO_SMALL].  >0 => suppression near natural activations.
             Per-example (median over dirs); bootstrap CI over examples.
  Groups   : confident/uncertain (max-softmax >= 0.9) x correct/wrong.
  Regions  : agglomerative clustering (avg linkage) of L3 on the fixed eval set;
             k chosen by silhouette over k=2..15 (cosine and euclidean).
             Validated stable region: >=20 examples, >=90% predicted-label
             purity, and per-cluster plateau_contrast bootstrap CI excludes 0.

Eval set : 50 test images per class (500 total), fixed IDs across checkpoints
             and seeds.  Directions/radii identical across checkpoints in a seed.

Usage: python experiments/analyze_sweep.py --seed 0
"""
import argparse, json, os, sys
sys.path.insert(0, '/workspace/mars-plateaus-image')

import numpy as np
import torch
from src.mnist import MLP, load_mnist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

torch.set_num_threads(2)

HERE = '/workspace/marsv_agent_haoyang/dir12_plateau_during_training'
DATA = '/workspace/mars-plateaus-image/data/mnist'

N_DIRS = 16
N_RHO = 21
RHO_MAX = 0.6
RHO_SMALL = 0.2          # small-radius interval for scalar summary
# Confidence = max raw output. Training uses MSE on one-hot targets, so the
# network drives the correct output toward 1 and others toward 0; the max raw
# output (not softmax, which saturates near 0.23) is the meaningful absolute
# confidence. "Confident" = max output >= 0.7 (fixed absolute threshold).
CONF_THRESH = 0.7
EPS = 1e-8
PER_CLASS = 50
N_BOOT = 1000


def build_eval_set(test_x, test_y):
    """50 test images per class, fixed IDs (deterministic order)."""
    idxs = []
    for c in range(10):
        ci = torch.where(test_y == c)[0][:PER_CLASS]
        idxs.append(ci)
    idx = torch.cat(idxs)
    return idx


def load_model(seed, step, device):
    state = torch.load(os.path.join(HERE, 'results', 'ckpts', f'seed{seed}',
                                    f'step{step}.pt'), map_location=device)
    m = MLP(depth=4, width=200, activation='relu').to(device)
    m.load_state_dict(state)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def response_curves(model, h1, dirs, rhos, device):
    """R for each example/radius, median over dirs.  Returns [n, n_rho]."""
    n = h1.shape[0]
    _, hs = model.forward_from(h1, 1)
    G0 = hs[-1]                                   # [n,200]
    G0n = G0.norm(dim=1) + EPS                    # [n]
    h1n = h1.norm(dim=1, keepdim=True)            # [n,1]
    acc = torch.zeros(n, len(rhos), N_DIRS, device=device)
    for di in range(N_DIRS):
        u = dirs[di]                              # [200] unit
        for ri, rho in enumerate(rhos):
            pts = h1 + rho * h1n * u              # [n,200]
            _, hp = model.forward_from(pts, 1)
            disp = (hp[-1] - G0).norm(dim=1)      # [n]
            acc[:, ri, di] = disp / G0n
    R = acc.median(dim=2).values                  # [n,n_rho]
    return R.cpu().numpy()


def auc_small(R, rhos, rho_small):
    """Trapezoid AUC over rho in [0, rho_small] per example. R:[n,n_rho]."""
    mask = rhos <= rho_small + 1e-9
    r = rhos[mask]
    return np.trapz(R[:, mask], r, axis=1)        # [n]


def contrast_from_auc(auc_data, auc_rand):
    """Scalar plateau_contrast = 1 - mean(AUC_data)/mean(AUC_rand)."""
    return 1.0 - auc_data.mean() / (auc_rand.mean() + EPS)


def bootstrap_contrast(auc_data, auc_rand, rng, n_boot=N_BOOT):
    n = len(auc_data)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        di = rng.integers(0, n, n)
        ri = rng.integers(0, len(auc_rand), len(auc_rand))
        vals[b] = 1.0 - auc_data[di].mean() / (auc_rand[ri].mean() + EPS)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def make_matched_random(h1, gen):
    """Norm-and-sparsity-matched random h1 (non-negative, |Gaussian| on random
    support of same size, scaled to same norm)."""
    n, w = h1.shape
    out = torch.zeros_like(h1)
    ks = (h1 > 0).sum(dim=1)
    norms = h1.norm(dim=1)
    for i in range(n):
        k = int(ks[i])
        g = torch.zeros(w, device=h1.device)
        support = torch.randperm(w, generator=gen, device=h1.device)[:k]
        g[support] = torch.randn(k, generator=gen, device=h1.device).abs()
        gn = g.norm() + EPS
        out[i] = g / gn * norms[i]
    return out


def validate_regions(labels, pred, contrast_per_cluster_ci):
    """Count clusters meeting size>=20, purity>=0.9, contrast CI>0."""
    n_valid = 0
    comp = []
    for c in np.unique(labels):
        m = labels == c
        size = int(m.sum())
        if size == 0:
            continue
        vals, counts = np.unique(pred[m], return_counts=True)
        maj = vals[counts.argmax()]
        purity = counts.max() / size
        lo = contrast_per_cluster_ci.get(int(c), (None, None))[0]
        valid = (size >= 20) and (purity >= 0.9) and (lo is not None and lo > 0)
        comp.append({'cluster': int(c), 'size': size, 'maj_pred': int(maj),
                     'purity': float(purity), 'contrast_lo': lo, 'valid': bool(valid)})
        if valid:
            n_valid += 1
    return n_valid, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.225)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ck_dir = os.path.join(HERE, 'results', 'ckpts', f'seed{args.seed}')
    manifest = json.load(open(os.path.join(ck_dir, 'manifest.json')))
    steps = manifest['ckpt_steps']

    _, _, test_x, test_y = load_mnist(DATA)
    test_x, test_y = test_x.to(device), test_y.to(device)
    eval_idx = build_eval_set(test_x, test_y)
    ex = test_x[eval_idx]                          # [500,784]
    ey = test_y[eval_idx].cpu().numpy()

    # fixed directions + radii (seed-independent -> identical across seeds)
    dgen = torch.Generator(device=device).manual_seed(12345)
    dirs = torch.randn(N_DIRS, 200, generator=dgen, device=device)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    rhos = np.linspace(0, RHO_MAX, N_RHO)
    rgen = torch.Generator(device=device).manual_seed(999)   # matched-random support
    boot_rng = np.random.default_rng(7)

    out = {'seed': args.seed, 'steps': steps, 'rhos': rhos.tolist(),
           'rho_small': RHO_SMALL, 'checkpoints': []}
    rep_curves = {}

    for step in steps:
        model = load_model(args.seed, step, device)
        with torch.no_grad():
            hiddens, logits = model.hidden_activations(ex)
        h1 = hiddens[0]                            # [500,200]
        L3 = hiddens[-1].cpu().numpy()
        conf, pred = logits.max(dim=1)            # max raw output = confidence
        conf = conf.cpu().numpy(); pred = pred.cpu().numpy()
        correct = (pred == ey)

        R_data = response_curves(model, h1, dirs, rhos, device)
        h1_rand = make_matched_random(h1, rgen)
        R_rand = response_curves(model, h1_rand, dirs, rhos, device)

        auc_d = auc_small(R_data, rhos, RHO_SMALL)
        auc_r = auc_small(R_rand, rhos, RHO_SMALL)
        contrast = contrast_from_auc(auc_d, auc_r)
        clo, chi = bootstrap_contrast(auc_d, auc_r, boot_rng)

        # groups
        groups = {}
        conf_mask = conf >= CONF_THRESH
        for gname, gmask in [('conf_correct', conf_mask & correct),
                             ('unc_correct', ~conf_mask & correct),
                             ('conf_wrong', conf_mask & ~correct),
                             ('unc_wrong', ~conf_mask & ~correct)]:
            k = int(gmask.sum())
            if k >= 5:
                gc = contrast_from_auc(auc_d[gmask], auc_r)
                glo, ghi = bootstrap_contrast(auc_d[gmask], auc_r, boot_rng)
            else:
                gc = glo = ghi = None
            groups[gname] = {'n': k, 'contrast': gc, 'ci': [glo, ghi]}

        # regions: cluster L3, silhouette-select k, validate
        region_info = {}
        for metric in ['cosine', 'euclidean']:
            best_k, best_s = 2, -2
            for k in range(2, 16):
                cl = AgglomerativeClustering(n_clusters=k, metric=metric,
                                             linkage='average').fit(L3)
                try:
                    s = silhouette_score(L3, cl.labels_, metric=metric)
                except Exception:
                    s = -2
                if s > best_s:
                    best_s, best_k = s, k
            cl = AgglomerativeClustering(n_clusters=best_k, metric=metric,
                                         linkage='average').fit(L3)
            labels = cl.labels_
            # per-cluster contrast CI
            ci_map = {}
            for c in np.unique(labels):
                m = labels == c
                if m.sum() >= 5:
                    lo, hi = bootstrap_contrast(auc_d[m], auc_r, boot_rng, n_boot=500)
                    ci_map[int(c)] = (lo, hi)
            n_valid, comp = validate_regions(labels, pred, ci_map)
            region_info[metric] = {'best_k': best_k, 'silhouette': float(best_s),
                                    'n_valid': n_valid, 'composition': comp}

        rec = {'step': step,
               'contrast': contrast, 'contrast_ci': [clo, chi],
               'mean_conf': float(conf.mean()),
               'frac_confident': float(conf_mask.mean()),
               'acc_eval': float(correct.mean()),
               'groups': groups, 'regions': region_info}
        out['checkpoints'].append(rec)
        rep_curves[step] = {
            'R_data_median': np.median(R_data, axis=0).tolist(),
            'R_rand_median': np.median(R_rand, axis=0).tolist(),
            'R_conf_correct': (np.median(R_data[conf_mask & correct], axis=0).tolist()
                               if (conf_mask & correct).sum() >= 5 else None),
            'R_conf_wrong': (np.median(R_data[conf_mask & ~correct], axis=0).tolist()
                             if (conf_mask & ~correct).sum() >= 5 else None),
            'R_unc': (np.median(R_data[~conf_mask], axis=0).tolist()
                      if (~conf_mask).sum() >= 5 else None),
        }
        print(f"step {step}: contrast={contrast:.3f} CI[{clo:.3f},{chi:.3f}] "
              f"conf={conf.mean():.3f} acc={correct.mean():.3f} "
              f"regions cos={region_info['cosine']['n_valid']}/"
              f"{region_info['cosine']['best_k']} "
              f"euc={region_info['euclidean']['n_valid']}/"
              f"{region_info['euclidean']['best_k']}", flush=True)

    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', f'sweep_seed{args.seed}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(HERE, 'results', f'curves_seed{args.seed}.json'), 'w') as f:
        json.dump(rep_curves, f)
    print("saved sweep + curves")


if __name__ == '__main__':
    main()
