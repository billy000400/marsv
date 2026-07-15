#!/usr/bin/env python3
"""
dir12 S4 — Region composition + membership-overlap lineage (seed 0).

Reuses the frozen clustering protocol from analyze_sweep.py (average-linkage
agglomerative clustering of L3 on the fixed 500-example eval set, k selected by
silhouette over k=2..15, cosine metric). Because the SAME eval examples are
reused at every checkpoint, adjacent checkpoints are aligned by maximum
membership overlap; births/deaths/splits/merges are read off the overlap matrix.

Outputs results/lineage_seed0.json with, per checkpoint:
  labels (per eval example), pred (per eval example), and validated-cluster info.
No perturbation / bootstrap needed here — validation (contrast CI) is already in
sweep_seed0.json; we reuse its per-cluster `valid` flags by matching majority
predicted digit.

Usage: python experiments/lineage.py --seed 0
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
PER_CLASS = 50
METRIC = 'cosine'   # primary metric (euclidean agrees; see sweep)


def build_eval_set(test_y):
    idxs = [torch.where(test_y == c)[0][:PER_CLASS] for c in range(10)]
    return torch.cat(idxs)


def load_model(seed, step, device):
    state = torch.load(os.path.join(HERE, 'results', 'ckpts', f'seed{seed}',
                                    f'step{step}.pt'), map_location=device)
    m = MLP(depth=4, width=200, activation='relu').to(device)
    m.load_state_dict(state); m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def cluster_L3(L3):
    best_k, best_s = 2, -2
    for k in range(2, 16):
        cl = AgglomerativeClustering(n_clusters=k, metric=METRIC,
                                     linkage='average').fit(L3)
        try:
            s = silhouette_score(L3, cl.labels_, metric=METRIC)
        except Exception:
            s = -2
        if s > best_s:
            best_s, best_k = s, k
    cl = AgglomerativeClustering(n_clusters=best_k, metric=METRIC,
                                 linkage='average').fit(L3)
    return cl.labels_, best_k, float(best_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.225)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ck_dir = os.path.join(HERE, 'results', 'ckpts', f'seed{args.seed}')
    steps = json.load(open(os.path.join(ck_dir, 'manifest.json')))['ckpt_steps']

    _, _, test_x, test_y = load_mnist(DATA)
    test_x, test_y = test_x.to(device), test_y.to(device)
    eval_idx = build_eval_set(test_y)
    ex = test_x[eval_idx]
    ey = test_y[eval_idx].cpu().numpy()

    # reuse sweep validation: map (step) -> set of validated majority-pred digits
    sweep = json.load(open(os.path.join(HERE, 'results',
                                         f'sweep_seed{args.seed}.json')))
    valid_digits = {}
    for rec in sweep['checkpoints']:
        comp = rec['regions'][METRIC]['composition']
        valid_digits[rec['step']] = [c['maj_pred'] for c in comp if c['valid']]

    out = {'seed': args.seed, 'steps': steps, 'metric': METRIC,
           'ey': ey.tolist(), 'checkpoints': []}
    for step in steps:
        model = load_model(args.seed, step, device)
        with torch.no_grad():
            hiddens, logits = model.hidden_activations(ex)
        L3 = hiddens[-1].cpu().numpy()
        pred = logits.max(dim=1).values, logits.argmax(dim=1)
        pred = pred[1].cpu().numpy()
        labels, k, sil = cluster_L3(L3)
        # per-cluster majority predicted digit + purity
        clusters = []
        for c in np.unique(labels):
            m = labels == c
            vals, counts = np.unique(pred[m], return_counts=True)
            maj = int(vals[counts.argmax()])
            clusters.append({'cluster': int(c), 'size': int(m.sum()),
                             'maj_pred': maj,
                             'purity': float(counts.max() / m.sum())})
        out['checkpoints'].append({
            'step': step, 'k': k, 'silhouette': sil,
            'labels': labels.tolist(), 'pred': pred.tolist(),
            'clusters': clusters,
            'valid_digits': sorted(valid_digits.get(step, [])),
            'n_valid': len(valid_digits.get(step, [])),
        })
        print(f"step {step}: k={k} sil={sil:.3f} "
              f"n_valid={len(valid_digits.get(step, []))}", flush=True)

    with open(os.path.join(HERE, 'results', f'lineage_seed{args.seed}.json'),
              'w') as f:
        json.dump(out, f)
    print("saved lineage")


if __name__ == '__main__':
    main()
