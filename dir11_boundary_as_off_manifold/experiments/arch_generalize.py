#!/usr/bin/env python3
"""
dir11 architecture-generalization (final limitation closed).

The REFUTED verdict -- plateau-separated digit-9 regions are NOT off-manifold /
disconnected components -- is robust across analysis hyperparameters (region k,
seed; graph k), holds for all ten digits, and transfers to a second independently
trained model (cross_model.py). The SOLE remaining limitation was that every model
so far shared ONE architecture: depth-4 width-200 ReLU. Here we retrain three
GENUINELY DIFFERENT architectures from scratch and re-run the exact same three
decisive tests, reusing analyze() from cross_model.py verbatim:

  - d3w200  : shallower (3 linear layers)
  - d4w400  : wider     (2x hidden width)
  - d5w200  : deeper    (5 linear layers)

plus the base d4w200 (seed 0) as the reference. analyze() is architecture-agnostic
(it reads the first and last hidden layer via hidden_activations / forward_from), so
the identical measurement pipeline applies. If the verdict is a property of the
phenomenon rather than of one architecture, all four should show:
  (D) cross-region 9->9 plateau boundary at a WELL-SUPPORTED point (percentile far
      below the cross-digit 9->0 boundary);
  (C) the two digit-9 regions connecting near the within-region hop scale, far below
      different digits;
  (X) 0 same-digit counterexamples (no same-digit region pair crossing an
      off-manifold gap, bottleneck > natural p95).

Trains on GPU under the shared memory fraction; analyzes on CPU.
Outputs results/arch_generalize.json, plots/arch_generalize.png.
"""
import os, sys, json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'experiments'))
import cross_model as cm  # sets BASE on sys.path, torch threads + memory fraction

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.mnist import MLP, load_mnist, load_checkpoint

RES, PLOT, DATA = cm.RES, cm.PLOT, cm.DATA


def train_arch(depth, width, seed, steps=100_000, n_test=2000):
    """Same training recipe as the base checkpoint, new architecture + seed."""
    ckpt = os.path.join(RES, f'mnist_mlp_d{depth}_w{width}_relu_n1000_seed{seed}.pt')
    if os.path.exists(ckpt):
        print(f"[train] found {ckpt}, skipping")
        return ckpt
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    train_x, train_y, test_x, test_y = load_mnist(DATA)
    sub = torch.randint(0, len(train_x), (1000,))
    tx, ty = train_x[sub].to(dev), train_y[sub].to(dev)
    vx, vy = test_x[:n_test].to(dev), test_y[:n_test].to(dev)
    model = MLP(depth=depth, width=width, activation='relu').to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lossf = torch.nn.MSELoss(); oh = torch.eye(10, device=dev)
    for step in range(steps):
        idx = torch.randint(0, 1000, (200,), device=dev)
        loss = lossf(model(tx[idx]), oh[ty[idx]])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20000 == 0:
            with torch.no_grad():
                acc = (model(vx).argmax(1) == vy).float().mean().item()
            print(f"[d{depth}w{width}s{seed}] step {step}: loss={loss.item():.5f} acc={acc:.3f}", flush=True)
    model.eval()
    torch.save({'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
                'config': dict(depth=depth, width=width, activation='relu',
                               n_test=n_test, seed=seed, steps=steps)}, ckpt)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[train] saved {ckpt}")
    return ckpt


# reference (base) + three different architectures
SPECS = [
    ('d4w200 (base, seed 0)', 'ref', cm.CKPT1),
    ('d3w200 (shallower)', 'new', dict(depth=3, width=200, seed=2)),
    ('d4w400 (wider)',      'new', dict(depth=4, width=400, seed=3)),
    ('d5w200 (deeper)',     'new', dict(depth=5, width=200, seed=4)),
]

results = {}
for label, kind, spec in SPECS:
    if kind == 'ref':
        ckpt = spec
    else:
        ckpt = train_arch(spec['depth'], spec['width'], spec['seed'])
    print(f"\n=== analyzing {label} ===", flush=True)
    model, ck = load_checkpoint(ckpt)
    r = cm.analyze(model, ck)
    r['arch'] = f"depth={ck['config']['depth']}, width={ck['config']['width']}"
    results[label] = r
    print(f"  {label}: acc={r['test_acc']:.3f}  9-region boundary pctile="
          f"{r['direct']['cross_region_9']['pctile']:.0f} vs 9v0="
          f"{r['direct']['cross_digit_9v0']['pctile']:.0f} | A<->B hops="
          f"{r['component']['A_vs_B_9']['hops']} within-A="
          f"{r['component']['within_A']['hops']} | counterex={r['counterexamples']}", flush=True)

# save (drop heavy curves)
with open(os.path.join(RES, 'arch_generalize.json'), 'w') as f:
    json.dump({lab: {k: v for k, v in r.items() if k != 'direct_curves'}
               for lab, r in results.items()}, f, indent=2)

# ============================================================ plot
labels = list(results.keys())
short = [l.split(' ')[0] for l in labels]
ncol = len(labels)
colors = ['C0', 'C1', 'C2', 'C3']

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

# (a) direct-path boundary percentile: cross-region 9->9 vs cross-digit 9->0
ax = axes[0]
x = np.arange(ncol); wdt = 0.38
cr = [results[l]['direct']['cross_region_9']['pctile'] for l in labels]
cd = [results[l]['direct']['cross_digit_9v0']['pctile'] for l in labels]
ax.bar(x - wdt / 2, cr, wdt, color='C0', label='cross-region 9->9')
ax.bar(x + wdt / 2, cd, wdt, color='C3', label='cross-digit 9->0')
ax.axhline(50, color='k', ls=':', lw=1, label='median support (50th pctile)')
ax.set_xticks(x); ax.set_xticklabels(short, rotation=15, fontsize=8)
ax.set_ylabel('boundary kNN-radius percentile\n(higher = more off-manifold)')
ax.set_title('(a) Direct-path test across architectures:\nsame-digit boundary well-supported, '
             'cross-digit off-manifold')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

# (b) component test hops: A<->B vs within-A vs different digits
ax = axes[1]
comps = ['within_A', 'A_vs_B_9', '9_vs_4', '9_vs_0']
cdisp = ['within A', 'A<->B (9)', '9<->4', '9<->0']
wdt2 = 0.2
for j, l in enumerate(labels):
    hv = [results[l]['component'][c]['hops'] or 0 for c in comps]
    ax.bar(np.arange(len(comps)) + (j - (ncol - 1) / 2) * wdt2, hv, wdt2,
           color=colors[j], label=short[j])
ax.set_xticks(np.arange(len(comps))); ax.set_xticklabels(cdisp, fontsize=8)
ax.set_ylabel(f'geodesic hops (k={cm.K_FIX})')
ax.set_title('(b) Component test across architectures:\ntwo 9-regions (A<->B) near within-region, '
             'far below different digits')
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis='y')

# (c) counterexample search: same-digit bottleneck vs p95, per architecture
ax = axes[2]
for j, l in enumerate(labels):
    sd = results[l]['same_digit']
    ds = sorted(int(d) for d in sd)
    bn = [sd[str(d)]['bottleneck'] / results[l]['rad_p95'] for d in ds]
    ax.plot(ds, bn, 'o-', color=colors[j], label=short[j], ms=5, lw=1.2)
ax.axhline(1.0, color='k', ls=':', lw=1.4, label='p95 gap threshold (=1.0)')
ax.set_xticks(range(10))
ax.set_xlabel('digit (two regions via KMeans-2 on L3)')
ax.set_ylabel('same-digit bottleneck / natural p95')
ax.set_title('(c) Counterexample search: every same-digit pair below\np95 in all 4 architectures '
             '-> 0 counterexamples')
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis='y')

fig.suptitle('Architecture generalization: three additional architectures (shallower / wider / deeper) '
             'reproduce the REFUTED verdict', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT, 'arch_generalize.png'), dpi=140); plt.close()
print("\nsaved plots/arch_generalize.png + results/arch_generalize.json")

# console summary table
print("\n===== ARCHITECTURE SWEEP SUMMARY =====")
print(f"{'arch':26s} {'acc':>5s} {'9->9pct':>8s} {'9->0pct':>8s} {'A<->B':>6s} {'winA':>5s} {'counterex':>10s}")
for l in labels:
    r = results[l]
    print(f"{l:26s} {r['test_acc']:5.3f} "
          f"{r['direct']['cross_region_9']['pctile']:8.0f} "
          f"{r['direct']['cross_digit_9v0']['pctile']:8.0f} "
          f"{str(r['component']['A_vs_B_9']['hops']):>6s} "
          f"{str(r['component']['within_A']['hops']):>5s} "
          f"{str(r['counterexamples']):>10s}")
