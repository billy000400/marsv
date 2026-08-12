"""Does the band decomposition of the unit ranking hold in a second training run?

neuron_bands.py and neuron_bands_time.py measured, on ONE run (model seed 1337), that the six
importance-rank bands of the block-1..4 MLP units each bend the interpolation path on their own, that
their effects sum to more than the all-units effect (redundancy ratio Lambda = 1.29 at step 30,000),
that Lambda is already there at the first checkpoint and does not grow, that per-unit worth falls
roughly 500-fold from the top band to the 512-2,048 band while the ranked bands still beat a random
draw from their own region, and that a unit's best rank tracks how well an 8-character window predicts
it. neuron_seed2.py gave the head-turnover / promotion / describability results a second seed but left
these untouched, so the deliverables still carry them as one-run measurements.

This script re-measures them on the second training run of the reference recipe (model seed 2024,
checkpoints in /tmp/dir13_frozen/checkpoints_ref_pos_s2, trained by train_frozen.py --tag ref_pos_s2),
using the same code paths and the same settings as the seed-1337 runs: neuron_path.record_pair /
ablate_pair, the same 150 character pairs, the same context, block-0 interpolation, the same six bands
and the same five checkpoints. Three measurements, all with a published seed-1337 counterpart:

  * band-alone and all-units effects at every checkpoint -> Lambda(step)  (seed 1 in Figure 38);
  * at step 30,000, the within-region random control per band, per-unit worth, and additivity
    (seed 1 in Figure 37a,b);
  * at step 30,000, the character-window probe refitted on this run, read against the band of each
    unit's best rank (seed 1 in Figure 37c).

Unit indices are not comparable across initializations, so every quantity is defined inside a run from
that run's own ranking; only the summary statistics are compared.

Raw -> results/neuron_bands_seed2_raw.npz, stats -> results/neuron_bands_seed2_summary.json,
figure -> plots/neuron_bands_seed2.png.
"""
import os, sys, json, time, hashlib, itertools
import numpy as np
import torch
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import run_pair, self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_path import install, uninstall, record_pair, ablate_pair, width, BLOCK, N_PAIRS, SEED
from neuron_bands import EDGES, MIN_GAP
from neuron_feature import CORPUS, CORPUS_SHA, T
import neuron_probe_early as npe
from cvd_style import CVD, LINESTYLES, MARKERS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
S2_DIR = "/tmp/dir13_frozen/checkpoints_ref_pos_s2"    # reference recipe, model seed 2024
STEPS = [831, 2038, 5000, 12500, 30000]


def med(x):
    return round(float(np.median(x)), 4)


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    npe.CKPT_DIR = S2_DIR                 # fit_probe reads this global at call time
    self_test()
    t0 = time.time()

    stoi = load_vocab()
    V = len(stoi)
    chars = sorted(stoi, key=lambda c: stoi[c])
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    all_pairs = list(itertools.combinations(range(len(chars)), 2))
    rng = np.random.default_rng(SEED)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]

    # untrained reference, once: the straight-line end of the recovered-fraction scale
    m0, _ = load_ckpt(os.path.join(S2_DIR, "ckpt_000000.pt"), device)
    w_init = np.array([width(ts, run_pair(m0, seqs[a], seqs[b], BLOCK, ts, device,
                                          batch_k=len(ts))["d_logit"])[0] for a, b in pairs])
    del m0
    torch.cuda.empty_cache()

    rng2 = np.random.default_rng(1234)
    W_BASE, W_BAND, W_ALL = [], [], []
    w_within = None
    min_rank = None
    ep_worst = 0.0
    for step in STEPS:
        model, _ = load_ckpt(os.path.join(S2_DIR, f"ckpt_{step:06d}.pt"), device)
        H = model.blocks[1].mlp.c_fc.out_features
        n_units = 4 * H
        edges = EDGES + [n_units]
        nb = len(edges) - 1
        sizes = np.array([edges[i + 1] - edges[i] for i in range(nb)])
        last = step == STEPS[-1]
        if last:
            w_within = np.zeros((nb, N_PAIRS))
            min_rank = np.full(n_units, n_units, dtype=np.int64)
        wraps = install(model)
        w_base = np.zeros(N_PAIRS)
        w_band = np.zeros((nb, N_PAIRS))
        w_all = np.zeros(N_PAIRS)
        for i, (a, b) in enumerate(pairs):
            r, imp = record_pair(model, wraps, seqs, a, b, ts, device)
            w_base[i], _ = width(ts, r["d_logit"])
            order = np.argsort(-imp)
            if last:
                min_rank[order] = np.minimum(min_rank[order], np.arange(n_units))
            for bi in range(nb):
                ra = ablate_pair(model, wraps, seqs, a, b, ts, device,
                                 order[edges[bi]:edges[bi + 1]], H)
                w_band[bi, i], _ = width(ts, ra["d_logit"])
                if last:
                    rw = ablate_pair(model, wraps, seqs, a, b, ts, device,
                                     rng2.choice(order[edges[bi]:], sizes[bi], replace=False), H)
                    w_within[bi, i], _ = width(ts, rw["d_logit"])
                    ep_worst = max(ep_worst, abs(rw["d0"]), abs(1 - rw["d1"]))
            ra = ablate_pair(model, wraps, seqs, a, b, ts, device, order, H)
            w_all[i], _ = width(ts, ra["d_logit"])
            ep_worst = max(ep_worst, abs(ra["d0"]), abs(1 - ra["d1"]))
        uninstall(model, wraps)
        del model
        torch.cuda.empty_cache()
        W_BASE.append(w_base); W_BAND.append(w_band); W_ALL.append(w_all)
        print(f"  step {step}: median w {np.median(w_base):.4f} ({time.time()-t0:.0f}s)", flush=True)

    W_BASE, W_BAND, W_ALL = np.array(W_BASE), np.array(W_BAND), np.array(W_ALL)
    gap_all = w_init[None, :] - W_BASE
    keep = (gap_all > MIN_GAP).all(axis=0)          # pairs usable at EVERY checkpoint

    checkpoints, common = [], []
    for si, step in enumerate(STEPS):
        for rows, m in ((checkpoints, gap_all[si] > MIN_GAP), (common, keep)):
            g = gap_all[si, m]
            rb = (W_BAND[si][:, m] - W_BASE[si, m][None, :]) / g[None, :]
            ra = (W_ALL[si, m] - W_BASE[si, m]) / g
            ssum = rb.sum(axis=0)
            rows.append({
                "step": step, "n_pairs_used": int(m.sum()),
                "median_w_baseline": med(W_BASE[si, m]),
                "rho_all_units": med(ra),
                "rho_band_alone": [med(rb[bi]) for bi in range(rb.shape[0])],
                "sum_of_band_alone": med(ssum),
                "redundancy_ratio": med(ssum / ra),
                "frac_pairs_ratio_gt_1": round(float((ssum > ra).mean()), 4),
                "p_sum_vs_all": float(wilcoxon(ssum, ra).pvalue),
            })

    # ---- final checkpoint: within-region control, per-unit worth ---------------------------
    si = len(STEPS) - 1
    ok = gap_all[si] > MIN_GAP
    g = gap_all[si, ok]
    r_band = (W_BAND[si][:, ok] - W_BASE[si, ok][None, :]) / g[None, :]
    r_within = (w_within[:, ok] - W_BASE[si, ok][None, :]) / g[None, :]
    nb = r_band.shape[0]
    edges = EDGES + [n_units]
    sizes = np.array([edges[i + 1] - edges[i] for i in range(nb)])
    bands = [{
        "edge_lo": edges[bi], "edge_hi": edges[bi + 1], "size": int(sizes[bi]),
        "rho_alone": med(r_band[bi]),
        "rho_within_region_same_size": med(r_within[bi]),
        "rho_alone_per_1000_units": round(float(np.median(r_band[bi]) / sizes[bi] * 1000), 4),
        "p_alone_vs_within_region": float(wilcoxon(r_band[bi], r_within[bi]).pvalue),
        "frac_pairs_alone_gt_within": round(float((r_band[bi] > r_within[bi]).mean()), 4),
    } for bi in range(nb)]

    # ---- character-window probe refitted on THIS run, read by best-rank band ---------------
    raw = open(CORPUS, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == CORPUS_SHA, "corpus SHA mismatch"
    data = np.array([stoi[c] for c in raw.decode("utf-8")], dtype=np.int64)
    train_data = data[:int(0.9 * len(data))]
    n_win = len(train_data) // T
    windows = train_data[:n_win * T].reshape(n_win, T)
    split_of = np.where(np.arange(n_win) % 10 == 8, 1, np.where(np.arange(n_win) % 10 == 9, 2, 0))
    pr = npe.fit_probe(STEPS[-1], windows, split_of, V, device, t0)

    band_of = np.digitize(min_rank, edges[1:-1], right=False)
    prof = [{
        "edge_lo": edges[bi], "edge_hi": edges[bi + 1],
        "n_units_assigned": int((band_of == bi).sum()),
        "r2_lag0_median": med(pr["1"][band_of == bi]),
        "r2_full_median": med(pr["full"][band_of == bi]),
    } for bi in range(nb)]
    head, tail = band_of <= 1, band_of >= 4
    head_vs_tail = {
        "n_head": int(head.sum()), "n_tail": int(tail.sum()),
        "r2_full_head": med(pr["full"][head]), "r2_full_tail": med(pr["full"][tail]),
        "p_r2_full": float(mannwhitneyu(pr["full"][head], pr["full"][tail]).pvalue),
        "r2_lag0_head": med(pr["1"][head]), "r2_lag0_tail": med(pr["1"][tail]),
        "p_r2_lag0": float(mannwhitneyu(pr["1"][head], pr["1"][tail]).pvalue),
    }

    s1_time = json.load(open(os.path.join(RES, "neuron_bands_time_summary.json")))
    s1_final = json.load(open(os.path.join(RES, "neuron_bands_summary.json")))
    summary = {
        "context": CONTEXT, "block": BLOCK, "n_t": N_T, "n_pairs": N_PAIRS, "ckpt_dir": S2_DIR,
        "model_seed": 2024, "band_edges": edges, "min_gap": MIN_GAP,
        "median_w_untrained": med(w_init),
        "endpoint_check": {"worst_ablated": round(ep_worst, 6)},
        "seed2": {
            "checkpoints": checkpoints,
            "common_subset": {"n_pairs": int(keep.sum()), "checkpoints": common},
            "final_bands": bands,
            "character_profile": {"bands": prof, "head_vs_tail": head_vs_tail},
        },
        "seed1": {
            "checkpoints": s1_time["checkpoints"],
            "common_subset": s1_time["common_subset"],
            "final_bands": s1_final["bands"],
            "character_profile": s1_final["character_profile"],
        },
    }
    with open(os.path.join(RES, "neuron_bands_seed2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_bands_seed2_raw.npz"),
                        pairs=np.array(pairs), steps=np.array(STEPS), edges=np.array(edges),
                        w_init=w_init, w_base=W_BASE, w_band=W_BAND, w_all=W_ALL,
                        w_within=w_within, keep=keep, min_rank=min_rank, band_of=band_of,
                        r2_lag0=pr["1"], r2_full=pr["full"])

    # ---- figure: the three seed-1 measurements with the second run beside them --------------
    x = np.array(STEPS, dtype=float)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))

    a = ax[0]
    for s, (key, lbl) in enumerate([("seed1", "seed 1337"), ("seed2", "seed 2024")]):
        y = [c["redundancy_ratio"] for c in summary[key]["common_subset"]["checkpoints"]]
        a.plot(x, y, color=CVD[0], ls=LINESTYLES[s], marker=MARKERS[0],
               mfc="none" if s else CVD[0], label=lbl)
    a.axhline(1.0, color="0.45", ls="--", lw=1.4)
    a.text(x[0], 1.01, "exact additivity", fontsize=8, color="0.35")
    a.set_xscale("log"); a.set_xlabel("training step (log scale)")
    a.set_ylabel(r"redundancy ratio $\Lambda$ (median over pairs)")
    a.set_title("(a) redundancy over training:\nsum of band-alone effects / all-units effect")
    a.legend(fontsize=9); a.grid(alpha=0.3)

    a = ax[1]
    xb = np.arange(nb)
    for s, (key, lbl) in enumerate([("seed1", "seed 1337"), ("seed2", "seed 2024")]):
        b = summary[key]["final_bands"]
        a.plot(xb, [100 * v["rho_alone"] for v in b], color=CVD[0], ls=LINESTYLES[s],
               marker=MARKERS[0], mfc="none" if s else CVD[0], label=f"band alone, {lbl}")
        a.plot(xb, [100 * v["rho_within_region_same_size"] for v in b], color=CVD[1],
               ls=LINESTYLES[s], marker=MARKERS[1], mfc="none" if s else CVD[1],
               label=f"same-size random draw from the same region, {lbl}")
    a.set_xticks(xb)
    a.set_xticklabels([f"{edges[i]}-{edges[i+1]}\n({sizes[i]})" for i in range(nb)], fontsize=8)
    a.set_xlabel("importance-rank band (number of units)")
    a.set_ylabel("% of the width gap removed by that band alone")
    a.set_title("(b) per-band effect at step 30,000\nwith its within-region control")
    a.legend(fontsize=7.5); a.grid(alpha=0.3)

    a = ax[2]
    for s, (key, lbl) in enumerate([("seed1", "seed 1337"), ("seed2", "seed 2024")]):
        p = summary[key]["character_profile"]["bands"]
        a.plot(xb, [v["r2_full_median"] for v in p], color=CVD[0], ls=LINESTYLES[s],
               marker=MARKERS[0], mfc="none" if s else CVD[0], label=f"full window, {lbl}")
        a.plot(xb, [v["r2_lag0_median"] for v in p], color=CVD[1], ls=LINESTYLES[s],
               marker=MARKERS[1], mfc="none" if s else CVD[1],
               label=f"current character alone, {lbl}")
    a.set_xticks(xb)
    a.set_xticklabels([f"{edges[i]}-{edges[i+1]}" for i in range(nb)], fontsize=8)
    a.set_ylim(0, 1.05)
    a.set_xlabel("band of the unit's best rank over the 150 pairs")
    a.set_ylabel("median held-out $R^2$ of the character-window probe")
    a.set_title("(c) describability against band\nof a unit's best rank, step 30,000")
    a.legend(fontsize=7.5); a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "neuron_bands_seed2.png"), dpi=130)
    plt.close(fig)

    print(json.dumps(summary["seed2"]["common_subset"]["checkpoints"], indent=1))
    print(json.dumps(bands, indent=1))
    print(json.dumps(head_vs_tail, indent=1))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
