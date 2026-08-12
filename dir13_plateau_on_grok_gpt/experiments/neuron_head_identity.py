"""Does training strengthen the SAME head units, or swap in different ones?

neuron_bands_time.py showed that what training adds to the band decomposition is the strength of the
head: the eight highest-ranked units per pair remove 7.2% of the trained->untrained width gap at step
831 and 23.9% at step 30,000, while the redundancy of the ranking stays flat. That leaves the
identity question open -- the top slots could be held by the same units throughout, or by different
units at each stage.

This needs no ablations at all, only the importance ranking itself:

    I_j = ||W_proj[:,j]|| * max_t |a_j(t) - chord_j(t)|

recorded from ONE unablated pass per pair per checkpoint. For each pair we take the top-8 and top-32
units at each of the five checkpoints and report (a) the overlap of each checkpoint's set with the
step-30,000 set, and (b) the overlap with the immediately preceding checkpoint. Same 150 pairs,
context, block-0 interpolation and checkpoints as neuron_bands_time.py.

The random baseline for an overlap of k out of n_units is k*k/n_units (0.017 units at k=8, 0.27 at
k=32), so any overlap above ~1 unit is already far above chance; the informative comparison is
between the early and late checkpoints, not against chance.

Raw -> results/neuron_head_identity_raw.npz, stats -> results/neuron_head_identity_summary.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_path import install, uninstall, record_pair, width, CKPT_DIR, N_PAIRS, SEED
from neuron_bands_time import STEPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
KS = [8, 32]


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    all_pairs = list(itertools.combinations(range(len(chars)), 2))
    rng = np.random.default_rng(SEED)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]
    t0 = time.time()

    top = {k: np.zeros((len(STEPS), N_PAIRS, k), dtype=np.int64) for k in KS}
    w_base = np.zeros((len(STEPS), N_PAIRS))
    for si, step in enumerate(STEPS):
        model, _ = load_ckpt(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"), device)
        wraps = install(model)
        for i, (a, b) in enumerate(pairs):
            r, imp = record_pair(model, wraps, seqs, a, b, ts, device)
            w_base[si, i], _ = width(ts, r["d_logit"])
            order = np.argsort(-imp)
            for k in KS:
                top[k][si, i] = order[:k]
        uninstall(model, wraps)
        del model
        torch.cuda.empty_cache()
        print(f"step {step} done ({time.time()-t0:.0f}s)", flush=True)

    def overlap(x, y):     # median over pairs of |x_i ∩ y_i|
        return float(np.median([len(set(x[i]) & set(y[i])) for i in range(N_PAIRS)]))

    summary = {
        "context": CONTEXT, "block": 0, "n_t": N_T, "n_pairs": N_PAIRS, "ckpt_dir": CKPT_DIR,
        "steps": STEPS, "k_list": KS, "n_units": 3840,
        "median_w_baseline": [round(float(np.median(w_base[si])), 4) for si in range(len(STEPS))],
        "overlap_with_final": {
            str(k): [overlap(top[k][si], top[k][-1]) for si in range(len(STEPS))] for k in KS},
        "overlap_with_previous": {
            str(k): [None] + [overlap(top[k][si], top[k][si - 1]) for si in range(1, len(STEPS))]
            for k in KS},
        "chance_overlap": {str(k): round(k * k / 3840, 3) for k in KS},
    }
    with open(os.path.join(RES, "neuron_head_identity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_head_identity_raw.npz"),
                        pairs=np.array(pairs), steps=np.array(STEPS), w_base=w_base,
                        **{f"top{k}": top[k] for k in KS})
    print(json.dumps(summary["overlap_with_final"], indent=2))
    print(json.dumps(summary["overlap_with_previous"], indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
