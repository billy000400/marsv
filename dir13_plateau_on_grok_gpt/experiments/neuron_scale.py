"""S24f: which SCALE does the intervention actually read -- activation units or residual displacement?

Where the series stands. Chord-linearizing a few dozen block-1-4 MLP units removes most of the
plateau (neuron_path.py). Ranking units by corpus character tuning selects them blind (28.9% of the
width gap at k=32), a fitted probe does better (56.5%), and the S24e control found the win comes from
DROPPING the per-unit standardization -- the plain character profile in raw activation units already
reaches 56.3%, the fitted context alone 34.8%. So the intervention cares how many activation units the
endpoint swap moves a unit, not which character it prefers.

But "activation units" is not the end of the story: a unit acts on the residual stream through its
write vector W_proj[:, j], so the displacement it causes is |a_j(1) - a_j(0)| * ||W_proj[:, j]||.
Every blind rule so far ignores that norm; the pair-fitted ranking includes it but measures curvature
(max_t |a_j(t) - chord_j(t)|) rather than endpoint displacement. This script closes the 2x2 by running
five more selection rules through the identical chord intervention, at the same k, on the same 150
pairs:

    raw_char_w   |sel_a - sel_b| * n_j        corpus character profile, in residual-displacement units
    probe_w      |yhat_a - yhat_b| * n_j      fitted probe, in residual-displacement units
    endpoint     |a_j(1) - a_j(0)|            the MEASURED swing the two rules above estimate
    endpoint_w   |a_j(1) - a_j(0)| * n_j      the same, in residual-displacement units
    wnorm        n_j                          write norm alone: pair-blind, data-free floor

with n_j = ||W_proj[:, j]||_2. `endpoint*` are oracles for the corpus rules (they read the assay's own
activations), which is the point: they separate "the score has the wrong form" from "the corpus
estimate of the score is noisy". `wnorm` is the same for every pair, so it says how much of any rule's
performance is just "edit the units that write hardest".

PRE-REGISTERED before running (references: raw_char 56.3%, probe 56.5%, pair-fitted 50.9%, corpus
character rule 28.9%, random 1.2%, all at k = 32):

 P1  Residual scale beats activation scale: raw_char_w and probe_w each land at least 2 points above
     their unweighted twins (>= 58.3% and >= 58.5%).
 P2  The corpus rules are estimation-limited, not form-limited: endpoint_w, the measured displacement,
     exceeds every corpus-estimated rule at k = 32 and also exceeds the pair-fitted 50.9%.
 P3  Selection must be pair-dependent: wnorm alone recovers < 20% -- above the 1.2% random control but
     far below every pair-dependent rule.

A failure of P1 says the intervention reads the activation swing itself and the write norm adds
nothing, which would also explain why the pair-fitted ranking (which uses the norm) is beaten by a
blind rule that does not. A failure of P3 would mean the whole ranking story is a statement about
which units write hardest.

Free checks, as in every script in this series: the unablated baseline must reproduce the stored
per-pair widths exactly, and both endpoints must stay exact under every selection.

Raw -> results/neuron_scale_raw.npz, stats -> results/neuron_scale_summary.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_path import install, uninstall, record_pair, ablate_pair, width, CKPT_DIR, EARLY, \
    N_PAIRS, SEED

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
    sel = np.load(os.path.join(RES, "neuron_feature_raw.npz"))["sel"]     # [V, U] raw corpus means
    yhat = np.load(os.path.join(RES, "neuron_probe_raw.npz"))["yhat"]     # [V, U] probe at context

    model, _ = load_ckpt(os.path.join(CKPT_DIR, "ckpt_030000.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    wraps = install(model)
    with torch.no_grad():
        wn = np.concatenate([torch.linalg.norm(wraps[l].mlp.c_proj.weight.float(),
                                               dim=0).cpu().numpy() for l in EARLY])   # [U]

    names = ["raw_char_w", "probe_w", "endpoint", "endpoint_w", "wnorm"]
    W = {r: np.full((len(K_LIST), N_PAIRS), np.nan) for r in names}
    w_base = np.zeros(N_PAIRS)
    ep_worst = 0.0
    overlap_pair = np.zeros(N_PAIRS)      # |top32(endpoint_w) & top32(pair-fitted)|
    for i, (a, b) in enumerate(pairs):
        r0, _ = record_pair(model, wraps, seqs, a, b, ts, device)
        w_base[i], _ = width(ts, r0["d_logit"])
        swing = np.concatenate([(wraps[l].rec[-1] - wraps[l].rec[0]).abs().float().cpu().numpy()
                                for l in EARLY])
        scores = {"raw_char_w": np.abs(sel[a] - sel[b]) * wn,
                  "probe_w": np.abs(yhat[a] - yhat[b]) * wn,
                  "endpoint": swing,
                  "endpoint_w": swing * wn,
                  "wnorm": wn}
        for name in names:
            order = np.argsort(-scores[name])
            if name == "endpoint_w":
                overlap_pair[i] = len(set(order[:32].tolist()) & set(prior["top32"][i].tolist()))
            for ki, k in enumerate(K_LIST):
                ra = ablate_pair(model, wraps, seqs, a, b, ts, device, order[:k], H)
                W[name][ki, i], _ = width(ts, ra["d_logit"])
                ep_worst = max(ep_worst, abs(ra["d0"]), abs(1 - ra["d1"]))
        if (i + 1) % 50 == 0:
            print(f"  pair {i+1}/{N_PAIRS} ({time.time()-t0:.0f}s)", flush=True)
    uninstall(model, wraps)

    wb, wi = float(np.median(w_base)), float(np.median(prior["w_init"]))
    gap = wi - wb
    k32 = K_LIST.index(32)
    ctrl = np.load(os.path.join(RES, "neuron_probe_control_raw.npz"))
    ref = {"raw_char": ctrl["w_raw_char"][k32],
           "probe": np.load(os.path.join(RES, "neuron_probe_raw.npz"))["w_probe"][k32],
           "pair_fitted": prior["w_pair"][list(prior["k_list"]).index(32)],
           "character_rule": np.load(os.path.join(RES,
                                                  "neuron_feature_causal_raw.npz"))["w_tuning"][1]}

    def rf(x):
        return round(float((np.median(x) - wb) / gap), 4)

    out = {
        "reproduction_check": {"median_w_baseline_here": round(wb, 4),
                               "max_abs_diff_per_pair":
                                   round(float(np.abs(w_base - prior["w_base"]).max()), 6)},
        "untrained_median_w": round(wi, 4), "worst_endpoint_error": round(ep_worst, 6),
        "k_list": K_LIST, "n_pairs": N_PAIRS, "n_units": len(wn),
        "wnorm_stats": {"median": round(float(np.median(wn)), 4),
                        "iqr": [round(float(np.percentile(wn, 25)), 4),
                                round(float(np.percentile(wn, 75)), 4)]},
        "rules": {n: {"median_w": [round(float(np.median(W[n][ki])), 4) for ki in range(len(K_LIST))],
                      "recovered_frac": [rf(W[n][ki]) for ki in range(len(K_LIST))]} for n in names},
        "reference_at_k32": {n: rf(v) for n, v in ref.items()},
        "overlap_endpoint_w_vs_pair_top32": {"median": float(np.median(overlap_pair)),
                                             "mean": round(float(overlap_pair.mean()), 2)},
        "paired_at_k32": {
            "raw_char_w_vs_raw_char": float(wilcoxon(W["raw_char_w"][k32], ref["raw_char"]).pvalue),
            "probe_w_vs_probe": float(wilcoxon(W["probe_w"][k32], ref["probe"]).pvalue),
            "endpoint_vs_probe": float(wilcoxon(W["endpoint"][k32], ref["probe"]).pvalue),
            "endpoint_w_vs_endpoint": float(wilcoxon(W["endpoint_w"][k32], W["endpoint"][k32]).pvalue),
            "endpoint_w_vs_pair_fitted": float(wilcoxon(W["endpoint_w"][k32],
                                                        ref["pair_fitted"]).pvalue),
            "wnorm_vs_raw_char": float(wilcoxon(W["wnorm"][k32], ref["raw_char"]).pvalue)},
    }
    with open(os.path.join(RES, "neuron_scale_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_scale_raw.npz"), w_base=w_base,
                        k_list=np.array(K_LIST), wnorm=wn, overlap_pair=overlap_pair,
                        **{f"w_{n}": W[n] for n in names})
    print(json.dumps(out, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
