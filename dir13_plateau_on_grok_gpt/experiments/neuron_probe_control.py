"""S24e control: is the probe rule's advantage the fitted context, or just the activation scale?

neuron_probe.py ranks units for pair (a, b) by |yhat_a - yhat_b|, the fitted probe's predicted
activation difference at the assay's own context, in RAW activation units. The character rule it
beats (neuron_feature_causal.py) ranks by |z_a - z_b|, a profile standardized within each unit. Two
things therefore differ at once: the probe knows the context, and it keeps each unit's own scale --
and a unit whose activation swings by a lot in absolute terms moves the residual stream more.

This script separates them by running two more selection rules through the identical intervention:

    raw_char    |sel_a - sel_b|    the character profile WITHOUT standardization (scale, no context)
    probe_z     |zhat_a - zhat_b|  the probe's prediction standardized per unit (context, no scale)

Both are blind to d(t) in exactly the same way as the two rules they bracket.

Raw -> results/neuron_probe_control.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_path import install, uninstall, record_pair, ablate_pair, width, CKPT_DIR, EARLY, N_PAIRS, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
K_LIST = [8, 32, 128]


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    t0 = time.time()
    self_test()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    V = len(chars)
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[ch] for ch in CONTEXT + c], dtype=np.int64) for c in chars]
    all_pairs = list(itertools.combinations(range(V), 2))
    rng = np.random.default_rng(SEED)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]

    prior = np.load(os.path.join(RES, "neuron_path_raw.npz"))
    assert (prior["pairs"] == np.array(pairs)).all(), "pair subsample must match neuron_path.py"
    sel = np.load(os.path.join(RES, "neuron_feature_raw.npz"))["sel"]       # [V, U] raw means
    yhat = np.load(os.path.join(RES, "neuron_probe_raw.npz"))["yhat"]       # [V, U] probe at context
    zhat = (yhat - yhat.mean(axis=0)) / np.maximum(yhat.std(axis=0), 1e-8)

    model, _ = load_ckpt(os.path.join(CKPT_DIR, "ckpt_030000.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    wraps = install(model)

    rules = {"raw_char": sel, "probe_z": zhat}
    W = {r: np.full((len(K_LIST), N_PAIRS), np.nan) for r in rules}
    w_base = np.zeros(N_PAIRS)
    ep_worst = 0.0
    for i, (a, b) in enumerate(pairs):
        r0, _ = record_pair(model, wraps, seqs, a, b, ts, device)
        w_base[i], _ = width(ts, r0["d_logit"])
        for name, S in rules.items():
            order = np.argsort(-np.abs(S[a] - S[b]))
            for ki, k in enumerate(K_LIST):
                ra = ablate_pair(model, wraps, seqs, a, b, ts, device, order[:k], H)
                W[name][ki, i], _ = width(ts, ra["d_logit"])
                ep_worst = max(ep_worst, abs(ra["d0"]), abs(1 - ra["d1"]))
        if (i + 1) % 50 == 0:
            print(f"  pair {i+1}/{N_PAIRS} ({time.time()-t0:.0f}s)", flush=True)
    uninstall(model, wraps)

    wb, wi = float(np.median(w_base)), float(np.median(prior["w_init"]))
    gap = wi - wb
    probe = np.load(os.path.join(RES, "neuron_probe_raw.npz"))["w_probe"]
    psum = json.load(open(os.path.join(RES, "neuron_probe_summary.json")))["causal"]
    k32 = K_LIST.index(32)
    out = {
        "reproduction_check": {"median_w_baseline_here": round(wb, 4),
                               "max_abs_diff_per_pair":
                                   round(float(np.abs(w_base - prior["w_base"]).max()), 6)},
        "untrained_median_w": round(wi, 4), "worst_endpoint_error": round(ep_worst, 6),
        "k_list": K_LIST, "n_pairs": N_PAIRS,
        **{name: {"median_w": [round(float(np.median(W[name][ki])), 4) for ki in range(len(K_LIST))],
                  "recovered_frac": [round(float((np.median(W[name][ki]) - wb) / gap), 4)
                                     for ki in range(len(K_LIST))]} for name in rules},
        "reference": {"probe": psum["probe_selected"]["recovered_frac"],
                      "character_rule": psum["character_rule"]["recovered_frac"],
                      "pair_ranked": psum["reference_rules"]["pair"]["recovered_frac"]},
        "paired_at_k32": {
            "raw_char_vs_character_rule": float(wilcoxon(
                W["raw_char"][k32],
                np.load(os.path.join(RES, "neuron_feature_causal_raw.npz"))["w_tuning"][1]).pvalue),
            "probe_vs_raw_char": float(wilcoxon(probe[k32], W["raw_char"][k32]).pvalue),
            "probe_vs_probe_z": float(wilcoxon(probe[k32], W["probe_z"][k32]).pvalue),
            "probe_z_vs_character_rule": float(wilcoxon(
                W["probe_z"][k32],
                np.load(os.path.join(RES, "neuron_feature_causal_raw.npz"))["w_tuning"][1]).pvalue)},
    }
    with open(os.path.join(RES, "neuron_probe_control.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_probe_control_raw.npz"),
                        w_raw_char=W["raw_char"], w_probe_z=W["probe_z"], w_base=w_base,
                        k_list=np.array(K_LIST))
    print(json.dumps(out, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
