#!/usr/bin/env python3
"""
dir12 — feedback human_feedback_1: collect the headline numbers of the smooth
(ReduceLROnPlateau f=0.5 p=100) full-60k runs into one JSON for the report.

For each run: PF (logit; CE also prob) at key steps, late curve motion M,
endpoint / within-class-control audit at the final step, LR-cascade span,
final full-train loss + smoothness metrics, final test acc/conf.

Usage: python3 experiments/collect_60k_numbers.py [dir ...]
       (default: the four primary p100 runs)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from plateau_protocol import HERE

FM = os.path.join(HERE, 'results', 'full_mnist_from_scratch')
DEFAULT = ['seed_0_pl_f0.5_p100', 'seed_1_pl_f0.5_p100', 'seed_2_pl_f0.5_p100',
           'seed_0_ce_pl_f0.5_p100']
KEY_STEPS = [0, 10, 30, 100, 300, 1500, 3000, 6000, 9000, 15000, 30000]


def d_prob(logits, end_logits):
    """d(alpha) in probability space from stored logits [P,T,10] and
    end_logits [P,2,10]."""
    def sm(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    p, pa, pb = sm(logits), sm(end_logits[:, :1]), sm(end_logits[:, 1:2])
    da = np.linalg.norm(p - pa, axis=-1)
    db = np.linalg.norm(p - pb, axis=-1)
    return da / (da + db + 1e-10)


def pf(d):
    return float(((d < 0.1) | (d > 0.9)).mean())


def run_stats(name):
    rd = os.path.join(FM, name)
    man = json.load(open(os.path.join(rd, 'manifest.json')))
    hist = json.load(open(os.path.join(rd, 'history.json')))
    steps = man['ckpt_steps']
    ce = 'CE' in man['config']['loss']
    D, DP, PRED, TA, TB = [], [], [], None, None
    for s in steps:
        z = np.load(os.path.join(rd, f'step_{s}.npz'))
        D.append(z['d_logit'])
        PRED.append(z['pred'])
        if ce:
            DP.append(d_prob(z['logits'], z['end_logits']))
        if TA is None:
            TA, TB = z['true_a'], z['true_b']
    D, PRED = np.stack(D), np.stack(PRED)
    out = {'steps': len(steps),
           'pf_logit': {s: pf(D[steps.index(s)][:45]) for s in KEY_STEPS
                        if s in steps},
           'pf_logit_peak': max((pf(D[i][:45]), steps[i])
                                for i in range(len(steps))),
           'final_test_acc': hist['test_acc'][-1],
           'final_test_conf': hist['test_conf'][-1],
           'final_train_loss_hist': hist['train_loss'][-1]}
    if ce:
        DP = np.stack(DP)
        out['pf_prob'] = {s: pf(DP[steps.index(s)][:45]) for s in KEY_STEPS
                          if s in steps}
        out['pf_prob_peak'] = max((pf(DP[i][:45]), steps[i])
                                  for i in range(len(steps)))
    # late curve motion M: mean |delta d| per adjacent-checkpoint gap over the
    # last 10 gaps (300-step gaps for the full schedule), 45 cross pairs
    dd = D if not ce else DP
    gaps = np.abs(np.diff(dd[:, :45, :], axis=0)).mean(axis=(1, 2))
    out['late_motion_M_last10gaps'] = float(gaps[-10:].mean())
    # endpoint + control audit at the final checkpoint (55-pair original bank)
    P = PRED[-1]
    cross_bad = int((P[:45, 0] != TA[:45]).sum() + (P[:45, -1] != TB[:45]).sum())
    within_single = int(sum((P[i] == P[i, 0]).all() for i in range(45, 55)))
    out['cross_endpoints_misclassified_of_90'] = cross_bad
    out['within_class_single_pred_of_10'] = within_single
    # frozen 3->5 bank (pairs 55..104): endpoint correctness, segments,
    # detours, repeated-class RLEs (merge-capable patterns) at final step +
    # anywhere mid-training
    if len(TA) >= 105:
        p35 = PRED[:, 55:105, :]                      # [ckpt, 50, T]
        seg = 1 + (p35[..., 1:] != p35[..., :-1]).sum(-1)
        ok = (p35[..., 0] == 3) & (p35[..., -1] == 5)
        det = np.array([[len(set(row.tolist()) - {int(row[0]), int(row[-1])}) > 0
                         for row in ck] for ck in p35])
        def rep(ck):                                  # repeated-class RLE count
            n = 0
            for row in ck:
                r = [int(row[0])]
                for c in row[1:]:
                    if int(c) != r[-1]:
                        r.append(int(c))
                if len(set(r)) < len(r):
                    n += 1
            return n
        out['bank3v5'] = {
            'endpoints_correct_final_of_50': int(ok[-1].sum()),
            'endpoints_correct_step0_of_50': int(ok[0].sum()),
            'seg_mean_final': float(seg[-1].mean()),
            'detour_frac_final': float(det[-1].mean()),
            'repeated_class_rle_final': rep(p35[-1]),
            'repeated_class_rle_max_any_ckpt': max(rep(ck) for ck in p35),
        }
    # original 3->5 pair (cross bank): RLE at the final checkpoint
    i35 = next(i for i in range(45)
               if TA[i] == 3 and TB[i] == 5)
    row = PRED[-1, i35]
    rle = [int(row[0])]
    for c in row[1:]:
        if int(c) != rle[-1]:
            rle.append(int(c))
    out['orig_3v5_rle_final'] = rle
    # LR cascade + smoothness from the per-step full-train-loss trace
    z = np.load(os.path.join(rd, 'sched_trace.npz'))
    tr, lr = z['full_train_loss'].astype(np.float64), z['lr']
    ch = np.where(np.diff(lr) != 0)[0] + 1
    cmin = np.minimum.accumulate(tr)
    ratio = tr[1000:] / np.maximum(cmin[1000:], 1e-30)
    out.update(n_lr_cuts=int(len(ch)),
               first_lr_cut=int(ch[0]) if len(ch) else None,
               last_lr_cut=int(ch[-1]) if len(ch) else None,
               final_lr=float(lr[-1]), final_full_train_loss=float(tr[-1]),
               spike_max=float(ratio.max()),
               spike_frac_gt2=float((ratio > 2).mean()),
               tail_range=float(tr[-5000:].max() / max(tr[-5000:].min(), 1e-30)))
    return out


def main():
    names = sys.argv[1:] or DEFAULT
    res = {n: run_stats(n) for n in names}
    out = os.path.join(HERE, 'results', 'numbers_60k_p100.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print('->', out)


if __name__ == '__main__':
    main()
