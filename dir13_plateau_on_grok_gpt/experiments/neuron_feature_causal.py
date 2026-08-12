"""S24c held-out causal test: do units selected by CORPUS TUNING ALONE carry the bend?

neuron_feature.py shows that a unit's character tuning, measured on the training corpus with no
interpolation, predicts which units the assay-derived importance ranking selects (AUROC 0.847). That
is a correlation between two rankings. This script asks the causal question directly, with the
selection rule made blind to the assay:

    rank units by differential tuning |z_a(j) - z_b(j)| -- a quantity computed from ordinary text --
    take the top k, linearize exactly those along the path (neuron_path.py's chord substitution),
    and measure how much of the trained->untrained width gap disappears.

That is a held-out prediction: nothing in the selection has seen d(t), the importance score I_j, or
the pair's own curve. It is compared at matched k against the two rules neuron_path.py already ran --
the assay's own per-pair top-k (the ceiling, which is fitted on the curve it is tested on) and random
k (the floor).

Raw -> results/neuron_feature_causal.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import run_pair, self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_path import install, uninstall, record_pair, ablate_pair, width, CKPT_DIR, BLOCK, EARLY, N_PAIRS, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
K_LIST = [8, 32, 128, 512]


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

    prior = np.load(os.path.join(RES, "neuron_path_raw.npz"))
    assert (prior["pairs"] == np.array(pairs)).all(), "pair subsample must match neuron_path.py"
    w_base_prior, w_init = prior["w_base"], prior["w_init"]
    z = np.load(os.path.join(RES, "neuron_feature_raw.npz"))["z"]      # [65, 3840] corpus tuning

    model, _ = load_ckpt(os.path.join(CKPT_DIR, "ckpt_030000.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    wraps = install(model)
    t0 = time.time()

    W = np.full((len(K_LIST), N_PAIRS), np.nan)
    w_base = np.zeros(N_PAIRS)
    ep_worst = 0.0
    for i, (a, b) in enumerate(pairs):
        r, _ = record_pair(model, wraps, seqs, a, b, ts, device)        # refreshes the chords
        w_base[i], _ = width(ts, r["d_logit"])
        order = np.argsort(-np.abs(z[a] - z[b]))                        # corpus tuning only
        for ki, k in enumerate(K_LIST):
            ra = ablate_pair(model, wraps, seqs, a, b, ts, device, order[:k], H)
            W[ki, i], _ = width(ts, ra["d_logit"])
            ep_worst = max(ep_worst, abs(ra["d0"]), abs(1 - ra["d1"]))
        if (i + 1) % 50 == 0:
            print(f"  pair {i+1}/{N_PAIRS} ({time.time()-t0:.0f}s)", flush=True)
    uninstall(model, wraps)

    wb, wi = float(np.median(w_base)), float(np.median(w_init))
    gap = wi - wb
    prior_sum = json.load(open(os.path.join(RES, "neuron_path_summary.json")))
    prior_k = prior_sum["k_list"]
    out = {
        "reproduction_check": {"median_w_baseline_here": round(wb, 4),
                               "median_w_baseline_neuron_path": prior_sum["baseline"]["median_w"],
                               "max_abs_diff_per_pair": round(float(np.abs(w_base - w_base_prior).max()), 6)},
        "untrained_median_w": round(wi, 4), "worst_endpoint_error": round(ep_worst, 6),
        "k_list": K_LIST, "n_pairs": N_PAIRS,
        "tuning_selected": {
            "median_w": [round(float(np.median(W[ki])), 4) for ki in range(len(K_LIST))],
            "recovered_frac": [round(float((np.median(W[ki]) - wb) / gap), 4) for ki in range(len(K_LIST))]},
        "reference_rules": {m: {"median_w": [prior_sum["curves"][m]["median_w"][prior_k.index(k)] for k in K_LIST],
                                "recovered_frac": [prior_sum["curves"][m]["recovered_frac"][prior_k.index(k)]
                                                   for k in K_LIST]}
                            for m in ["pair", "global", "random"]},
    }
    # paired tests at k = 32 against the assay-ranked and random rules
    k32 = K_LIST.index(32)
    out["paired_at_k32"] = {
        "p_vs_random": float(wilcoxon(W[k32], prior["w_random"][prior_k.index(32)]).pvalue),
        "p_vs_pair_ranked": float(wilcoxon(W[k32], prior["w_pair"][prior_k.index(32)]).pvalue),
        "p_vs_global": float(wilcoxon(W[k32], prior["w_global"][prior_k.index(32)]).pvalue),
        "frac_pairs_wider_than_baseline": round(float((W[k32] > w_base).mean()), 4)}
    with open(os.path.join(RES, "neuron_feature_causal.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_feature_causal_raw.npz"),
                        w_tuning=W, w_base=w_base, k_list=np.array(K_LIST))
    print(json.dumps(out, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
