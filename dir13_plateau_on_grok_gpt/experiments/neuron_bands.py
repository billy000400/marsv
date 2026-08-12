"""Why does the TAIL of the unit ranking cost hundreds of units? -- band decomposition of rho(k).

Where the series stands (neuron_path.py, neuron_greedy.py): linearizing the per-pair top-k MLP
units of blocks 1-4 recovers rho = 0.51 of the trained->untrained width gap at k=32 and 0.87 at
k=3840 (all units), and sequential re-selection buys only 3.4 points at k=128 and nothing at k=32.
So a few dozen units carry the first half of the bend and the last third needs hundreds. The nested
prefix curve cannot say WHY: rho(k) mixes "the units are individually weaker out there" with "the
units out there only work together".

This script cuts the ranking into bands and linearizes each band ON ITS OWN, at the same 150 pairs /
context / block-0 interpolation / step-30000 checkpoint as everything else:

    B1=[0,8)  B2=[8,32)  B3=[32,128)  B4=[128,512)  B5=[512,2048)  B6=[2048,3840)

and measures, per pair,
  * rho(B) for each band alone;
  * rho of the nested prefix ending at each band edge (reproduces neuron_path at k=8..3840);
  * rho of a RANDOM set of the same size as each band, drawn from ALL 3840 units (the budget
    control: what an arbitrary edit of that size buys). Note this control is contaminated for the
    large bands -- a random 1536-unit set contains about 40% of the top-32 units -- so it is only a
    budget comparison, never evidence that the tail is weak;
  * rho of a random set of the same size drawn only from the units ranked at or below the band's own
    lower edge (the WITHIN-REGION control): for B5 that is 1536 units drawn from the 3328 ranked
    >=512, so it asks whether the fine ordering inside the tail carries any signal at all.

Two things then follow directly. (a) ADDITIVITY: if the tail is many small independent
contributions, the band-alone effects should add up to the all-units effect; a shortfall is a joint
effect, an excess is redundancy. (b) EFFICIENCY: rho(B)/|B| against the two controls says whether
the tail units are individually weak-but-real or indistinguishable from arbitrary units.

The "different character profile" half of the question needs no forward passes: neuron_probe.py
already fitted a text-statistics description to every one of the 3840 units, so we assign each unit
to a band by its BEST (minimum) rank over the 150 pairs -- a tail unit is then one that no pair ever
ranks highly -- and compare the fitted descriptions across bands.

Raw -> results/neuron_bands_raw.npz, stats -> results/neuron_bands_summary.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import run_pair, self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_path import (install, uninstall, record_pair, ablate_pair, width,
                         BLOCK, CKPT_DIR, N_PAIRS, SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
EDGES = [0, 8, 32, 128, 512, 2048]      # n_units appended at runtime
MIN_GAP = 0.10                          # per-pair rho needs a real trained->untrained gap


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

    model, _ = load_ckpt(os.path.join(CKPT_DIR, "ckpt_030000.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    n_units = 4 * H
    edges = EDGES + [n_units]
    nb = len(edges) - 1
    sizes = np.array([edges[i + 1] - edges[i] for i in range(nb)])
    wraps = install(model)
    t0 = time.time()

    w_base = np.zeros(N_PAIRS)
    w_band = np.zeros((nb, N_PAIRS))
    w_pref = np.zeros((nb, N_PAIRS))
    w_rand = np.zeros((nb, N_PAIRS))
    w_within = np.zeros((nb, N_PAIRS))
    min_rank = np.full(n_units, n_units, dtype=np.int64)   # best rank any pair gives each unit
    ep_worst = 0.0
    rng2 = np.random.default_rng(1234)
    for i, (a, b) in enumerate(pairs):
        r, imp = record_pair(model, wraps, seqs, a, b, ts, device)
        w_base[i], _ = width(ts, r["d_logit"])
        order = np.argsort(-imp)
        min_rank[order] = np.minimum(min_rank[order], np.arange(n_units))
        for bi in range(nb):
            lo, hi = edges[bi], edges[bi + 1]
            pool = order[lo:]                                  # units ranked at or below this band
            for name, sel in (("band", order[lo:hi]),
                              ("pref", order[:hi]),
                              ("rand", rng2.choice(n_units, sizes[bi], replace=False)),
                              ("within", rng2.choice(pool, sizes[bi], replace=False))):
                ra = ablate_pair(model, wraps, seqs, a, b, ts, device, sel, H)
                {"band": w_band, "pref": w_pref, "rand": w_rand,
                 "within": w_within}[name][bi, i], _ = width(ts, ra["d_logit"])
                ep_worst = max(ep_worst, abs(ra["d0"]), abs(1 - ra["d1"]))
        if (i + 1) % 25 == 0:
            print(f"  pair {i+1}/{N_PAIRS} ({time.time()-t0:.0f}s)", flush=True)

    uninstall(model, wraps)
    del model
    torch.cuda.empty_cache()
    m0, _ = load_ckpt(os.path.join(CKPT_DIR, "ckpt_000000.pt"), device)
    w_init = np.array([width(ts, run_pair(m0, seqs[a], seqs[b], BLOCK, ts, device,
                                          batch_k=len(ts))["d_logit"])[0] for a, b in pairs])
    del m0
    torch.cuda.empty_cache()

    # ---- per-pair recovered fraction ------------------------------------------------------
    gap = w_init - w_base
    ok = gap > MIN_GAP
    def rho(w):                                    # [nb, N_PAIRS] -> per-pair fraction of the gap
        return (w - w_base[None, :]) / gap[None, :]
    r_band, r_pref, r_rand, r_within = rho(w_band), rho(w_pref), rho(w_rand), rho(w_within)
    r_marg = np.vstack([r_pref[0], np.diff(r_pref, axis=0)])   # what the band adds inside the prefix
    add_sum = r_band[:, ok].sum(axis=0)                        # per pair: sum of band-alone effects
    total = r_pref[-1, ok]

    def med(x):
        return round(float(np.median(x)), 4)

    def p(x, y):
        return float(wilcoxon(x, y).pvalue)

    summary = {
        "context": CONTEXT, "step": 30000, "block": BLOCK, "n_t": N_T, "n_pairs": N_PAIRS,
        "ckpt_dir": CKPT_DIR, "n_units": n_units, "band_edges": edges,
        "band_sizes": sizes.tolist(), "min_gap": MIN_GAP,
        "n_pairs_used": int(ok.sum()),
        "baseline": {"median_w": med(w_base), "median_w_untrained": med(w_init)},
        "endpoint_check": {"worst_ablated": round(ep_worst, 6)},
        "bands": [{
            "edge_lo": edges[bi], "edge_hi": edges[bi + 1], "size": int(sizes[bi]),
            "median_w_alone": med(w_band[bi, ok]),
            "rho_alone": med(r_band[bi, ok]),
            "rho_marginal_in_prefix": med(r_marg[bi, ok]),
            "rho_random_same_size": med(r_rand[bi, ok]),
            "rho_within_region_same_size": med(r_within[bi, ok]),
            "rho_alone_per_1000_units": round(float(np.median(r_band[bi, ok]) / sizes[bi] * 1000), 4),
            "rho_random_per_1000_units": round(float(np.median(r_rand[bi, ok]) / sizes[bi] * 1000), 4),
            "p_alone_vs_random": p(r_band[bi, ok], r_rand[bi, ok]),
            "p_alone_vs_within_region": p(r_band[bi, ok], r_within[bi, ok]),
            "p_alone_vs_marginal": p(r_band[bi, ok], r_marg[bi, ok]),
            "frac_pairs_alone_gt_random": round(float((r_band[bi, ok] > r_rand[bi, ok]).mean()), 4),
            "frac_pairs_alone_gt_within": round(float((r_band[bi, ok] > r_within[bi, ok]).mean()), 4),
        } for bi in range(nb)],
        "prefix_rho": [med(r_pref[bi, ok]) for bi in range(nb)],
        "additivity": {
            "sum_of_band_alone": med(add_sum), "all_units": med(total),
            "ratio_sum_over_total": round(float(np.median(add_sum / total)), 4),
            "p_sum_vs_total": p(add_sum, total),
            "frac_pairs_sum_gt_total": round(float((add_sum > total).mean()), 4),
        },
    }

    # ---- character profile of each band (no forward passes; neuron_probe.py's fitted units) ----
    band_of = np.digitize(min_rank, edges[1:-1], right=False)      # 0..nb-1 by best rank
    pr = np.load(os.path.join(RES, "neuron_probe_raw.npz"))
    prof = []
    for bi in range(nb):
        m = band_of == bi
        prof.append({
            "edge_lo": edges[bi], "edge_hi": edges[bi + 1], "n_units_assigned": int(m.sum()),
            "r2_lag0_median": med(pr["r2_1"][m]), "r2_full_median": med(pr["r2_full"][m]),
            "gain_context_median": med(pr["r2_full"][m] - pr["r2_1"][m]),
        })
    head, tail = band_of <= 1, band_of >= 4        # mean rank < 32 vs >= 512
    summary["character_profile"] = {
        "assignment": "each unit assigned to the band containing its BEST (minimum) rank over the "
                      "150 pairs, so a tail unit is one no pair ever ranks highly",
        "bands": prof,
        "head_vs_tail": {
            "n_head": int(head.sum()), "n_tail": int(tail.sum()),
            "r2_full_head": med(pr["r2_full"][head]), "r2_full_tail": med(pr["r2_full"][tail]),
            "p_r2_full": float(mannwhitneyu(pr["r2_full"][head], pr["r2_full"][tail]).pvalue),
            "r2_lag0_head": med(pr["r2_1"][head]), "r2_lag0_tail": med(pr["r2_1"][tail]),
            "p_r2_lag0": float(mannwhitneyu(pr["r2_1"][head], pr["r2_1"][tail]).pvalue),
        },
    }

    with open(os.path.join(RES, "neuron_bands_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_bands_raw.npz"),
                        pairs=np.array(pairs), edges=np.array(edges), sizes=sizes,
                        w_base=w_base, w_init=w_init, w_band=w_band, w_pref=w_pref,
                        w_rand=w_rand, w_within=w_within, ok=ok, min_rank=min_rank,
                        band_of=band_of)
    print(json.dumps(summary["bands"], indent=2))
    print(json.dumps(summary["additivity"], indent=2))
    print(json.dumps(summary["character_profile"]["head_vs_tail"], indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
