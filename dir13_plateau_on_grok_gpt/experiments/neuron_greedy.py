"""S24g: is the selection limit a JOINT effect? -- residual-corrected (greedy) unit selection.

Where the series stands. Chord-linearizing k block-1-4 MLP units removes most of the plateau
(neuron_path.py). Every selection rule tried so far scores each unit ALONE and takes the top k:
the pair-fitted importance I_j = ||W_proj[:,j]|| * max_t |a_j(t) - chord_j(t)| (50.9% of the width
gap at k=32, 68.4% at k=128), the corpus character profile in raw activation units (56.3% at k=32),
the fitted probe (56.5%), and the measured endpoint swing (56.6%). S24f showed the corpus rules are
NOT estimation-limited -- an oracle for the same score ties them -- so what is left is the FORM of
the score: a per-unit ranking cannot see that a set of units removes more bend together than its
members do apart.

This script tests that directly. Selection stays fitted (it may read the pair's own activations) but
becomes sequential: build S in R equal rounds, and before each round RE-MEASURE the importance with
the units already chosen linearized, so a unit is scored by how much of the REMAINING bend it carries.

    S <- {}
    repeat R times:
        run the assay with S linearized, recording a_j(t) for every unit
        imp_j <- ||W_proj[:,j]|| * max_t |a_j(t) - chord_j(t)|      (0 for j in S by construction)
        S <- S + top-(k/R) units by imp_j

R = 1 is exactly the pair-fitted rule of neuron_path.py, so it doubles as a free reproduction check
and as the control: the ONLY difference between R = 1 and R > 1 is that later rounds see the network
after the earlier picks are already straightened. Any gain is therefore a joint effect, not a better
estimate of the same per-unit quantity.

PRE-REGISTERED before running (references at k=32: pair-fitted 50.9%, best per-unit rule 56.6%,
random 1.2%; at k=128: pair-fitted 68.4%, best per-unit rule 65.2%; all-units ceiling 86.7%):

 P1  Joint effects are real and large: greedy R=4 at k=32 recovers at least 5 points more than
     R=1 (>= 55.9%).
 P2  More rounds help monotonically: recovery is non-decreasing in R at both k=32 and k=128.
 P3  Joint selection breaks the per-unit ceiling: greedy R=8 at k=32 exceeds the best per-unit rule
     measured anywhere in this series (56.6%).

A failure of P1/P3 says the per-unit rankings are already near the best any size-k set can do, i.e.
the bend is spread over units that barely interact, and the remaining distance to the 86.7% ceiling
is simply the count of units, not the choice of them.

Free checks, as in every script in this series: R = 1 must reproduce neuron_path.py's per-pair
pair-fitted widths, the unablated baseline must reproduce the stored per-pair widths, and both
endpoints must stay exact under every selection.

Raw -> results/neuron_greedy_raw.npz, stats -> results/neuron_greedy_summary.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import run_pair, self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_path import ChordMLP, uninstall, record_pair, ablate_pair, width, CKPT_DIR, BLOCK, \
    EARLY, N_PAIRS, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
K_LIST = [32, 128]
R_LIST = [1, 2, 4, 8]


class GreedyMLP(ChordMLP):
    """ChordMLP plus mode 'ablate_record': linearize `idx` AND record the resulting activations,
    so one pass yields the residual curvature of every remaining unit."""

    def forward(self, x):
        a = F.gelu(self.mlp.c_fc(x))
        if self.mode == "ablate_record" and a.shape[0] == self.chord.shape[0]:
            a = a.clone()
            if self.idx is not None and self.idx.numel():
                a[:, -1, self.idx] = self.chord[:, self.idx].to(a.dtype)
            self.rec = a[:, -1, :].detach().clone()
            return self.mlp.drop(self.mlp.c_proj(a))
        return super().forward(x)


def install_greedy(model):
    wraps = {}
    for l in EARLY:
        w = GreedyMLP(model.blocks[l].mlp)
        model.blocks[l].mlp = w
        wraps[l] = w
    return wraps


def residual_imp(model, wraps, seqs, a, b, ts, device, sel, H):
    """Run the assay with `sel` linearized while recording; return per-unit residual importance."""
    for i, l in enumerate(EARLY):
        bs = sel[(sel >= i * H) & (sel < (i + 1) * H)] - i * H
        wraps[l].idx = torch.as_tensor(bs, dtype=torch.long, device=device)
        wraps[l].mode = "ablate_record"
    run_pair(model, seqs[a], seqs[b], BLOCK, ts, device, batch_k=len(ts))
    for w in wraps.values():
        w.mode = "off"
    imp = []
    for l in EARLY:
        w = wraps[l]
        with torch.no_grad():
            wn = torch.linalg.norm(w.mlp.c_proj.weight.float(), dim=0)
            imp.append((wn * (w.rec.float() - w.chord).abs().max(dim=0).values).cpu().numpy())
    return np.concatenate(imp)


def greedy_select(model, wraps, seqs, a, b, ts, device, k, R, H, imp0):
    """Build a size-k set in R rounds, re-measuring importance under the current set each round."""
    per = k // R
    sel = np.argsort(-imp0)[:per]
    for _ in range(R - 1):
        imp = residual_imp(model, wraps, seqs, a, b, ts, device, sel, H)
        imp[sel] = -np.inf
        sel = np.concatenate([sel, np.argsort(-imp)[:per]])
    return sel


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
    k_prior = list(prior["k_list"])
    w_pair_prior = {k: prior["w_pair"][k_prior.index(k)] for k in K_LIST}

    model, _ = load_ckpt(os.path.join(CKPT_DIR, "ckpt_030000.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    wraps = install_greedy(model)

    W = {(k, R): np.zeros(N_PAIRS) for k in K_LIST for R in R_LIST}
    ovl = {(k, R): np.zeros(N_PAIRS) for k in K_LIST for R in R_LIST}
    w_base = np.zeros(N_PAIRS)
    ep_worst = 0.0
    for i, (a, b) in enumerate(pairs):
        r0, imp0 = record_pair(model, wraps, seqs, a, b, ts, device)
        w_base[i], _ = width(ts, r0["d_logit"])
        ep_worst = max(ep_worst, abs(r0["d0"]), abs(1 - r0["d1"]))
        for k in K_LIST:
            base_set = set(np.argsort(-imp0)[:k].tolist())
            for R in R_LIST:
                sel = greedy_select(model, wraps, seqs, a, b, ts, device, k, R, H, imp0)
                ovl[(k, R)][i] = len(base_set & set(sel.tolist()))
                ra = ablate_pair(model, wraps, seqs, a, b, ts, device, sel, H)
                W[(k, R)][i], _ = width(ts, ra["d_logit"])
                ep_worst = max(ep_worst, abs(ra["d0"]), abs(1 - ra["d1"]))
        if (i + 1) % 25 == 0:
            print(f"  pair {i+1}/{N_PAIRS} ({time.time()-t0:.0f}s)", flush=True)
    uninstall(model, wraps)

    wb, wi = float(np.median(w_base)), float(np.median(prior["w_init"]))
    gap = wi - wb

    def rf(x):
        return round(float((np.median(x) - wb) / gap), 4)

    out = {
        "reproduction_check": {
            "median_w_baseline_here": round(wb, 4),
            "max_abs_diff_baseline": round(float(np.abs(w_base - prior["w_base"]).max()), 6),
            "max_abs_diff_R1_vs_pair_fitted":
                {str(k): round(float(np.abs(W[(k, 1)] - w_pair_prior[k]).max()), 6) for k in K_LIST}},
        "untrained_median_w": round(wi, 4), "worst_endpoint_error": round(ep_worst, 6),
        "n_pairs": N_PAIRS, "k_list": K_LIST, "r_list": R_LIST,
        "all_units_ceiling": round(float((np.median(prior["w_pair"][-1]) - wb) / gap), 4),
        "greedy": {str(k): {str(R): {"median_w": round(float(np.median(W[(k, R)])), 4),
                                     "recovered_frac": rf(W[(k, R)]),
                                     "median_overlap_with_R1": float(np.median(ovl[(k, R)]))}
                            for R in R_LIST} for k in K_LIST},
        "paired_vs_R1": {str(k): {str(R): float(wilcoxon(W[(k, R)], W[(k, 1)]).pvalue)
                                  for R in R_LIST[1:]} for k in K_LIST},
        "paired_R8_vs_R4": {str(k): float(wilcoxon(W[(k, 8)], W[(k, 4)]).pvalue) for k in K_LIST},
        "frac_pairs_R8_ge_R1": {str(k): round(float((W[(k, 8)] >= W[(k, 1)]).mean()), 4)
                                for k in K_LIST},
    }
    with open(os.path.join(RES, "neuron_greedy_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_greedy_raw.npz"), w_base=w_base,
                        k_list=np.array(K_LIST), r_list=np.array(R_LIST),
                        **{f"w_k{k}_R{R}": W[(k, R)] for k in K_LIST for R in R_LIST},
                        **{f"ovl_k{k}_R{R}": ovl[(k, R)] for k in K_LIST for R in R_LIST})
    print(json.dumps(out, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
