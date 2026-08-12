"""Is the redundancy of the unit ranking BUILT by training? -- band decomposition over checkpoints.

neuron_bands.py measured, at step 30,000, that the six importance-rank bands each bend the path on
their own and that their effects sum to 111.5% of the 85.2% recovered by linearizing all 3,840 units
at once. That excess is redundancy: several bands remove the same bend. Read at one checkpoint it is
only a description of the finished network.

This script re-measures the same decomposition at five checkpoints of the same run
(steps 831 / 2,038 / 5,000 / 12,500 / 30,000) and asks one question:

    does the redundancy ratio  (sum of band-alone effects) / (all-units effect)  grow as the
    plateau sharpens, or is it there as soon as there is any bend to share?

Only the band-alone runs and the all-units run are needed, so it is 7 conditions per pair per
checkpoint (150 pairs, same context / block-0 interpolation / pairs as everywhere else). The random
controls of neuron_bands.py answer a different question and are not repeated.

Two honest limits, both reported in the output: (a) the per-pair recovered fraction divides by that
pair's trained->untrained width gap, so early checkpoints -- where the network has barely sharpened
-- have few pairs with a usable gap, and n_pairs_used is reported per checkpoint; (b) this is one
training run, so a trend across its checkpoints is a description of that run's development, not a
demonstration that training causes the redundancy.

Raw -> results/neuron_bands_time_raw.npz, stats -> results/neuron_bands_time_summary.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import run_pair, self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_path import install, uninstall, record_pair, ablate_pair, width, BLOCK, CKPT_DIR, \
    N_PAIRS, SEED
from neuron_bands import EDGES, MIN_GAP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
STEPS = [831, 2038, 5000, 12500, 30000]


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    self_test()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    all_pairs = list(itertools.combinations(range(len(chars)), 2))
    rng = np.random.default_rng(SEED)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]
    t0 = time.time()

    # untrained reference, once: the straight-line end of the recovered-fraction scale
    m0, _ = load_ckpt(os.path.join(CKPT_DIR, "ckpt_000000.pt"), device)
    w_init = np.array([width(ts, run_pair(m0, seqs[a], seqs[b], BLOCK, ts, device,
                                          batch_k=len(ts))["d_logit"])[0] for a, b in pairs])
    del m0
    torch.cuda.empty_cache()

    out, W_BASE, W_BAND, W_ALL = [], [], [], []
    for step in STEPS:
        model, _ = load_ckpt(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"), device)
        H = model.blocks[1].mlp.c_fc.out_features
        n_units = 4 * H
        edges = EDGES + [n_units]
        nb = len(edges) - 1
        wraps = install(model)
        w_base = np.zeros(N_PAIRS)
        w_band = np.zeros((nb, N_PAIRS))
        w_all = np.zeros(N_PAIRS)
        for i, (a, b) in enumerate(pairs):
            r, imp = record_pair(model, wraps, seqs, a, b, ts, device)
            w_base[i], _ = width(ts, r["d_logit"])
            order = np.argsort(-imp)
            for bi in range(nb):
                ra = ablate_pair(model, wraps, seqs, a, b, ts, device,
                                 order[edges[bi]:edges[bi + 1]], H)
                w_band[bi, i], _ = width(ts, ra["d_logit"])
            ra = ablate_pair(model, wraps, seqs, a, b, ts, device, order, H)
            w_all[i], _ = width(ts, ra["d_logit"])
        uninstall(model, wraps)
        del model
        torch.cuda.empty_cache()

        gap = w_init - w_base
        ok = gap > MIN_GAP
        r_band = (w_band - w_base[None, :]) / gap[None, :]
        r_all = (w_all - w_base) / gap
        ssum = r_band[:, ok].sum(axis=0)
        out.append({
            "step": step,
            "median_w_baseline": round(float(np.median(w_base)), 4),
            "n_pairs_used": int(ok.sum()),
            "rho_all_units": round(float(np.median(r_all[ok])), 4),
            "rho_band_alone": [round(float(np.median(r_band[bi, ok])), 4) for bi in range(nb)],
            "sum_of_band_alone": round(float(np.median(ssum)), 4),
            "redundancy_ratio": round(float(np.median(ssum / r_all[ok])), 4),
            "frac_pairs_ratio_gt_1": round(float((ssum > r_all[ok]).mean()), 4),
        })
        W_BASE.append(w_base); W_BAND.append(w_band); W_ALL.append(w_all)
        print(json.dumps(out[-1]), f"({time.time()-t0:.0f}s)", flush=True)

    # the per-checkpoint rows above use different pair subsets (an early checkpoint has fewer pairs
    # with a usable gap), so repeat the trend on the pairs usable at EVERY checkpoint
    W_BASE, W_BAND, W_ALL = np.array(W_BASE), np.array(W_BAND), np.array(W_ALL)
    gap_all = w_init[None, :] - W_BASE
    keep = (gap_all > MIN_GAP).all(axis=0)
    common = []
    for si, step in enumerate(STEPS):
        g = gap_all[si, keep]
        rb = (W_BAND[si][:, keep] - W_BASE[si, keep][None, :]) / g[None, :]
        ra = (W_ALL[si, keep] - W_BASE[si, keep]) / g
        ssum = rb.sum(axis=0)
        common.append({
            "step": step,
            "median_w_baseline": round(float(np.median(W_BASE[si, keep])), 4),
            "rho_all_units": round(float(np.median(ra)), 4),
            "rho_band_alone": [round(float(v), 4) for v in np.median(rb, axis=1)],
            "sum_of_band_alone": round(float(np.median(ssum)), 4),
            "redundancy_ratio": round(float(np.median(ssum / ra)), 4),
            "frac_pairs_ratio_gt_1": round(float((ssum > ra).mean()), 4),
        })

    summary = {
        "context": CONTEXT, "block": BLOCK, "n_t": N_T, "n_pairs": N_PAIRS, "ckpt_dir": CKPT_DIR,
        "band_edges": EDGES + [4 * 960], "min_gap": MIN_GAP,
        "median_w_untrained": round(float(np.median(w_init)), 4),
        "checkpoints": out,
        "common_subset": {"n_pairs": int(keep.sum()), "checkpoints": common},
    }
    with open(os.path.join(RES, "neuron_bands_time_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_bands_time_raw.npz"),
                        pairs=np.array(pairs), steps=np.array(STEPS), w_init=w_init,
                        w_base=W_BASE, w_band=W_BAND, w_all=W_ALL, keep=keep)
    print(json.dumps(summary["common_subset"], indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
