"""S1 — validate the brightness dataset numerically and plot the target family."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common as C

C.setup_torch()
os.makedirs(C.PLOTS, exist_ok=True)
os.makedirs(C.RESULTS, exist_ok=True)

d = C.build_dataset(seed=0)
err_tr = (d['xtr'].norm(dim=1) - d['btr']).abs().max().item()
err_va = (d['xva'].norm(dim=1) - d['bva']).abs().max().item()
unit_err = (d['xte_unit'].norm(dim=1) - 1).abs().max().item()

# identical inputs across k is trivially true (targets are a function of b only),
# but check that two seeds give DIFFERENT brightness assignments as intended.
d1 = C.build_dataset(seed=1)
seed_diff = (d1['btr'] - d['btr']).abs().mean().item()

check = {'max_abs_err_norm_minus_b_train': err_tr,
         'max_abs_err_norm_minus_b_val': err_va,
         'max_abs_err_unit_norm_test': unit_err,
         'mean_abs_brightness_diff_seed0_vs_seed1': seed_diff,
         'n_train': int(len(d['btr'])), 'n_val': int(len(d['bva'])),
         'n_probe_images': int(len(d['xte_unit'])),
         'digit_counts_probe': np.bincount(d['yte_lab'].numpy(), minlength=10).tolist()}
print(json.dumps(check, indent=2))
assert err_tr < 1e-4 and err_va < 1e-4 and unit_err < 1e-5
with open(os.path.join(C.RESULTS, 's1_dataset_check.json'), 'w') as f:
    json.dump(check, f, indent=2)

b = C.brightness_grid()
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for i, k in enumerate(C.K_VALUES):
    ax.plot(b, C.target_fn(b, k), color=C.CVD[i], ls=C.LINESTYLES[i], lw=2,
            label=f"$k={k:g}$")
ax.axvline(C.B0, color='0.4', lw=1, ls=(0, (1, 2)))
ax.text(C.B0 + 0.005, -0.95, '$b_0=0.7$', color='0.35', fontsize=9)
ax.axvspan(C.CENTER_LO, C.CENTER_HI, color='0.85', alpha=0.6, zorder=0)
ax.set_xlabel('brightness $b$')
ax.set_ylabel('target $y_k(b)$')
ax.set_title('Continuous target family: linear ($k$=0.5) to switch-like ($k$=10)')
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(C.PLOTS, 'target_functions.png'), dpi=150)
plt.close()
print('saved plots/target_functions.png')
