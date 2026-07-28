#!/usr/bin/env python3
"""dir161 — export per-pair, per-seed, per-layer LD and MJ as a committed CSV.

The raw d(alpha) arrays live in results/seed*/probe_*.npz (gitignored, large);
this writes the per-pair summaries every aggregate is computed from.
Usage: python experiments/export_pairs.py
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from analyze import LAYERS, SEEDS, lin_dev, load, max_jump

P = load('_best')
meta = P[(0, 'clf')]
rows = []
for s in SEEDS:
    for k in ['clf', 'pre']:
        for key, lname in LAYERS:
            ld, mj = lin_dev(P[(s, k)][key]), max_jump(P[(s, k)][key])
            for i in range(len(ld)):
                rows.append({'seed': s, 'model': k, 'layer': key,
                             'pair': i, 'class_a': int(meta['class_a'][i]),
                             'class_b': int(meta['class_b'][i]),
                             'rep': int(meta['rep'][i]),
                             'idx_a': int(meta['idx_a'][i]),
                             'idx_b': int(meta['idx_b'][i]),
                             'lin_dev': round(float(ld[i]), 6),
                             'max_jump': round(float(mj[i]), 6)})
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', 'per_pair_metrics.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), 'rows ->', out)
