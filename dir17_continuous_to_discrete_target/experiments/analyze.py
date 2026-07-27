"""S3/S4 — brightness sweeps, normalized activation movement, concentration scores.

For every trained (k, seed) model: sweep 201 brightness values on 100 held-out
digit-balanced test images, record predictions and post-ReLU activations from
all three hidden layers, and reduce to
  s_l(b_i) = ||h_l(b_{i+1}) - h_l(b_i)|| / sum_j ||h_l(b_{j+1}) - h_l(b_j)||
  C_l(k)   = sum of s_l over b_i in [0.64, 0.76)      (0.2 if movement uniform)
  R_l(k)   = C_l(k) / 0.2
Also computes the same score for the target curve and for the model's own
prediction curve, and the sweep fit quality (R^2 of prediction vs target).
"""
import json
import os

import numpy as np
import torch

import common as C

PREFIX = os.environ.get("CKPT_PREFIX", "ckpt")
OUT = os.environ.get("ANALYSIS_OUT", "analysis.json")

C.setup_torch()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [0, 1, 2]
CKPT_KEYS = ['final', 'best_val']


def target_scores():
    b = C.brightness_grid()
    out = {}
    for k in C.K_VALUES:
        m = np.abs(np.diff(C.target_fn(b, k)))
        s = m / m.sum()
        out[f"{k:g}"] = {'s': s.tolist(), 'C': float(C.concentration(s))}
    return out


def run():
    data = C.build_dataset(0)                 # probe images are seed-independent
    x_unit = data['xte_unit']
    b = C.brightness_grid()
    res = {}
    for which in CKPT_KEYS:
        for k in C.K_VALUES:
            for seed in SEEDS:
                ck = torch.load(os.path.join(C.RESULTS, f"{PREFIX}_k{k:g}_s{seed}.pt"),
                                map_location='cpu', weights_only=False)
                model = C.MLP(n_out=1).to(DEVICE)
                model.load_state_dict(ck[which])
                model.eval()
                preds, acts = C.sweep(model, x_unit, DEVICE)
                del model
                torch.cuda.empty_cache()

                y = C.target_fn(b, k)
                pred_mean = preds.mean(axis=0)
                r2 = 1 - ((preds - y[None, :]) ** 2).mean() / y.var()

                entry = {'pred_mean': pred_mean.tolist(),
                         'pred_std': preds.std(axis=0).tolist(),
                         'sweep_r2': float(r2),
                         'sweep_mse': float(((preds - y[None, :]) ** 2).mean())}
                # prediction-curve concentration (per image, then averaged)
                mp = np.abs(np.diff(preds, axis=1))
                sp = mp / (mp.sum(axis=1, keepdims=True) + C.EPS)
                entry['pred_C'] = float(C.concentration(sp).mean())
                for l in range(C.DEPTH - 1):
                    s = C.normalized_movement(acts[l])          # [N, 200]
                    Ci = C.concentration(s)                      # [N]
                    entry[f'L{l+1}'] = {'s_mean': s.mean(axis=0).tolist(),
                                        's_sem': (s.std(axis=0) / np.sqrt(len(s))).tolist(),
                                        'C_mean': float(Ci.mean()),
                                        'C_std_images': float(Ci.std())}
                del acts, preds
                res[f"{which}|k{k:g}|s{seed}"] = entry
                print(f"{which} k={k:g} s={seed}: R2={r2:.4f} "
                      f"R_L3={entry['L3']['C_mean']/0.2:.3f}", flush=True)
    return res


if __name__ == '__main__':
    out = {'target': target_scores(), 'models': run(),
           'grid': C.brightness_grid().tolist(), 'seeds': SEEDS}
    with open(os.path.join(C.RESULTS, OUT), 'w') as f:
        json.dump(out, f)
    print('saved results/' + OUT)
