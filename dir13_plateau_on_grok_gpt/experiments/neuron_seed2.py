"""Do the developmental head results hold in a second training run?

Three findings in this direction describe how the block-1..4 MLP units that carry the plateau change
over training, and all three were measured on ONE run (model seed 1337):

  * head turnover      -- none of a pair's step-831 top-8 units are still in its top-8 at step 30,000
                          (neuron_head_identity.py);
  * promotion from just below the head -- those final top-8 units already sit at median rank ~114 of
                          3,840 at step 831, an order of magnitude above the chance rank 1919.5, and
                          climb from there (neuron_head_origin.py);
  * describability travels with head membership -- the network as a whole gets LESS predictable from an
                          8-character window as it trains, yet the units training promotes into the head
                          gain against that background while the ones it demotes fall faster than it
                          (neuron_probe_early.py).

A one-run result cannot distinguish a property of this training recipe from a property of this
particular initialization and data order. This script retrains the reference recipe with a different
model seed (2024, the seed every other second-seed run in this direction uses; data seed unchanged at
42, as in narrow192_s2 / frozen_early_s2) and re-measures all three, using the same code paths: the
importance recording of neuron_path.record_pair, the same 150 pairs, the same context, the same
block-0 interpolation, the same five checkpoints, and neuron_probe_early.fit_probe for the probe.

Nothing here is a new measurement -- it is the same three measurements on a second run, so every
number has a published seed-1337 counterpart, and both are written side by side into the summary.

Raw -> results/neuron_seed2_raw.npz, stats -> results/neuron_seed2_summary.json,
figure -> plots/neuron_seed2.png.
"""
import os, sys, json, time, hashlib, itertools
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_path import install, uninstall, record_pair, width, N_PAIRS, SEED
from neuron_feature import CORPUS, CORPUS_SHA, T
import neuron_probe_early as npe
from cvd_style import CVD, LINESTYLES, MARKERS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
S2_DIR = "/tmp/dir13_frozen/checkpoints_ref_pos_s2"    # reference recipe, model seed 2024
STEPS = npe.STEPS
GROUPS = npe.GROUPS
KS = [8, 32]
K_HEAD = 8
THRESH = [8, 32, 128, 512]


def q(x, p):
    return [round(float(np.percentile(x[si], p)), 1) for si in range(len(STEPS))]


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    npe.CKPT_DIR = S2_DIR                 # fit_probe reads this global at call time
    t0 = time.time()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    all_pairs = list(itertools.combinations(range(len(chars)), 2))
    rng = np.random.default_rng(SEED)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]

    # --- importance ranking, one unablated pass per pair per checkpoint -----------------------
    rank, n_units = None, None
    w_base = np.zeros((len(STEPS), N_PAIRS))
    for si, step in enumerate(STEPS):
        model, _ = load_ckpt(os.path.join(S2_DIR, f"ckpt_{step:06d}.pt"), device)
        wraps = install(model)
        for i, (a, b) in enumerate(pairs):
            r, imp = record_pair(model, wraps, seqs, a, b, ts, device)
            w_base[si, i], _ = width(ts, r["d_logit"])
            if rank is None:
                n_units = imp.size
                rank = np.zeros((len(STEPS), N_PAIRS, n_units), dtype=np.int16)
            rank[si, i, np.argsort(-imp)] = np.arange(n_units, dtype=np.int16)
        uninstall(model, wraps)
        del model
        torch.cuda.empty_cache()
        print(f"rank: step {step} done ({time.time()-t0:.0f}s)", flush=True)

    topk = {k: np.argsort(rank, axis=2)[:, :, :k] for k in KS}
    rows = np.arange(N_PAIRS)[:, None]
    r_final = np.stack([rank[si][rows, topk[K_HEAD][-1]] for si in range(len(STEPS))])
    r_early = np.stack([rank[si][rows, topk[K_HEAD][0]] for si in range(len(STEPS))])

    def overlap(x, y):
        return float(np.median([len(set(x[i]) & set(y[i])) for i in range(N_PAIRS)]))

    start, end = np.median(r_final[0]), np.median(r_final[-1])
    ident = {
        "median_w_baseline": [round(float(np.median(w_base[si])), 4) for si in range(len(STEPS))],
        "overlap_with_final": {str(k): [overlap(topk[k][si], topk[k][-1])
                                        for si in range(len(STEPS))] for k in KS},
        "overlap_with_previous": {str(k): [None] + [overlap(topk[k][si], topk[k][si - 1])
                                                    for si in range(1, len(STEPS))] for k in KS},
    }
    origin = {
        "random_unit_expected_rank": (n_units - 1) / 2,
        "final_head_rank": {"median": q(r_final, 50), "p25": q(r_final, 25), "p75": q(r_final, 75)},
        "early_head_rank": {"median": q(r_early, 50), "p25": q(r_early, 25), "p75": q(r_early, 75)},
        "frac_final_head_inside_top": {str(t): [round(float(np.mean(r_final[si] < t)), 4)
                                                for si in range(len(STEPS))] for t in THRESH},
        "frac_of_climb_completed": [round(float((start - np.median(r_final[si])) / (start - end)), 3)
                                    for si in range(len(STEPS))],
    }

    # --- character-window probe, refitted at every checkpoint of this run ---------------------
    V = len(stoi)
    raw = open(CORPUS, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == CORPUS_SHA, "corpus SHA mismatch"
    data = np.array([stoi[c] for c in raw.decode("utf-8")], dtype=np.int64)
    train_data = data[:int(0.9 * len(data))]
    n_win = len(train_data) // T
    windows = train_data[:n_win * T].reshape(n_win, T)
    split_of = np.where(np.arange(n_win) % 10 == 8, 1, np.where(np.arange(n_win) % 10 == 9, 2, 0))
    R2 = {step: npe.fit_probe(step, windows, split_of, V, device, t0) for step in STEPS}

    final, early = np.unique(topk[K_HEAD][-1]), np.unique(topk[K_HEAD][0])
    role = {"promoted": np.setdiff1d(final, early), "demoted": np.setdiff1d(early, final),
            "stable": np.intersect1d(final, early),
            "never_head": np.setdiff1d(np.arange(n_units), np.union1d(final, early))}
    probe = {"group_sizes": {g: int(role[g].size) for g in GROUPS}, "medians": {},
             "median_change": {}, "within_band": {}}
    for m in ["1", "full"]:
        probe["medians"][m] = {str(s): {**{g: round(float(np.median(R2[s][m][role[g]])), 4)
                                           for g in GROUPS},
                                        "all_units": round(float(np.median(R2[s][m])), 4)}
                               for s in STEPS}
        probe["median_change"][m] = {g: round(float(np.median(R2[STEPS[-1]][m][role[g]]
                                                             - R2[STEPS[0]][m][role[g]])), 4)
                                     for g in GROUPS}
    contrasts = {831: [("0-7", "stable", "demoted"), ("8-31", "promoted", "never_head")],
                 30000: [("0-7", "stable", "promoted"), ("8-31", "demoted", "never_head")]}
    for m in ["1", "full"]:
        probe["within_band"][m] = {}
        for s, cs in contrasts.items():
            si = STEPS.index(s)
            head = np.unique(topk[8][si])
            band = {"0-7": head, "8-31": np.setdiff1d(np.unique(topk[32][si]), head)}
            probe["within_band"][m][str(s)] = {
                f"{b}:{g1}_vs_{g2}": npe.test(R2[s][m], np.intersect1d(band[b], role[g1]),
                                              np.intersect1d(band[b], role[g2]))
                for b, g1, g2 in cs}

    seed1 = {k: json.load(open(os.path.join(RES, f"neuron_{k}_summary.json")))
             for k in ["head_identity", "head_origin", "probe_early"]}
    meta = json.load(open(os.path.join(RES, "train_meta_ref_pos_s2.json")))
    summary = {
        "context": CONTEXT, "block": 0, "n_t": N_T, "n_pairs": N_PAIRS, "steps": STEPS,
        "n_units": int(n_units), "ckpt_dir": S2_DIR,
        "seed2_run": {"model_seed": meta["model_seed"], "data_seed": meta["data_seed"],
                      "final_step": meta["final_step"], "final_val_acc": meta["final_val_acc"],
                      "params": meta["params"], "minutes": meta["minutes"]},
        "seed2": {**ident, **origin, **probe},
        "seed1": {
            "median_w_baseline": seed1["head_identity"]["median_w_baseline"],
            "overlap_with_final": seed1["head_identity"]["overlap_with_final"],
            "overlap_with_previous": seed1["head_identity"]["overlap_with_previous"],
            "final_head_rank": seed1["head_origin"]["final_head_rank"],
            "early_head_rank": seed1["head_origin"]["early_head_rank"],
            "frac_final_head_inside_top": seed1["head_origin"]["frac_final_head_inside_top"],
            "frac_of_climb_completed": seed1["head_origin"]["frac_of_climb_completed"],
            "group_sizes": seed1["probe_early"]["group_sizes"],
            "medians": seed1["probe_early"]["medians"],
            "median_change": seed1["probe_early"]["median_change"],
            "within_band": seed1["probe_early"]["within_band"],
        },
    }
    with open(os.path.join(RES, "neuron_seed2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_seed2_raw.npz"),
                        pairs=np.array(pairs), steps=np.array(STEPS), w_base=w_base,
                        top8=topk[8], top32=topk[32], r_final=r_final, r_early=r_early,
                        **{f"r2_{m}_{s}": R2[s][m] for s in STEPS for m in ["1", "full"]},
                        **{g: role[g] for g in GROUPS})

    # --- figure: the three developmental curves, one panel each, both seeds ------------------
    x = np.array(STEPS, dtype=float)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    a = ax[0]
    for j, k in enumerate(KS):
        for s, lbl in enumerate(["seed 1337", "seed 2024"]):
            y = (summary["seed1"] if s == 0 else summary["seed2"])["overlap_with_final"][str(k)]
            a.plot(x, np.array(y) / k, color=CVD[j], ls=LINESTYLES[s], marker=MARKERS[j],
                   mfc="none" if s else CVD[j], label=f"top-{k}, {lbl}")
    a.set_xscale("log"); a.set_ylim(0, 1.05)
    a.set_xlabel("training step (log scale)")
    a.set_ylabel("fraction of the final top-$k$ set already held")
    a.set_title("(a) head turnover:\noverlap of the top-$k$ units with the step-30,000 top-$k$")
    a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)

    a = ax[1]
    for j, key in enumerate(["final_head_rank", "early_head_rank"]):
        for s, lbl in enumerate(["seed 1337", "seed 2024"]):
            y = np.maximum((summary["seed1"] if s == 0 else summary["seed2"])[key]["median"], 0.5)
            a.plot(x, y, color=CVD[j], ls=LINESTYLES[s], marker=MARKERS[j], mfc="none" if s else CVD[j],
                   label=("step-30,000 top-8 units" if j == 0 else "step-831 top-8 units")
                         + f", {lbl}")
    a.axhline((n_units - 1) / 2, color="gray", ls=":", lw=1.2)
    a.text(x[0], (n_units - 1) / 2 * 1.1, "a random unit", fontsize=8, color="gray")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("training step (log scale)")
    a.set_ylabel("median importance rank of 3,840 (log scale)")
    a.set_title("(b) where the head units come from:\nmedian rank at each checkpoint")
    a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.3)

    a = ax[2]
    for j, g in enumerate(["promoted", "demoted", "all_units"]):
        for s, lbl in enumerate(["seed 1337", "seed 2024"]):
            src = (summary["seed1"] if s == 0 else summary["seed2"])["medians"]["1"]
            y = [src[str(st)][g] for st in STEPS]
            a.plot(x, y, color=CVD[j], ls=LINESTYLES[s], marker=MARKERS[j], mfc="none" if s else CVD[j],
                   label=f"{g.replace('_', ' ')}, {lbl}")
    a.set_xscale("log"); a.set_ylim(0, 1.35)
    a.set_xlabel("training step (log scale)")
    a.set_ylabel("median held-out $R^2$, current character alone")
    a.set_title("(c) describability of each role group\nat every checkpoint")
    a.legend(fontsize=8, loc="upper left", ncol=2); a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "neuron_seed2.png"), dpi=130)
    plt.close(fig)
    print(json.dumps(summary["seed2"]["overlap_with_final"]), flush=True)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
