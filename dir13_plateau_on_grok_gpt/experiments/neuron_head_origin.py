"""Where do the finished network's head units come from -- a smooth climb or a late arrival?

neuron_head_identity.py showed that the head is REWRITTEN as training proceeds: the median overlap
between a pair's top-8 units at step 831 and its top-8 at step 30,000 is 0 of 8. That says the final
head units were not in the head early, but not where they were. Two developmental stories fit:

  * smooth climb  -- the final head units are already high-ranked (say inside the top 128) early and
                     work their way up slot by slot;
  * late arrival  -- they sit in the anonymous middle of the ranking until late in training and then
                     jump into the head.

Both are consistent with the overlap numbers; they differ in the RANK of those units at earlier
checkpoints, which is what this script records. No ablations: one unablated pass per pair per
checkpoint gives the full importance vector

    I_j = ||W_proj[:,j]|| * max_t |a_j(t) - chord_j(t)|,

and the rank of every unit follows from it. For each pair we take its step-30,000 top-8 units and read
their rank (0 = most important, 3839 = least) at each earlier checkpoint; the mirror question -- what
becomes of the units that held the head early -- is answered by the step-831 top-8's rank at every
later checkpoint. A uniformly random unit sits at rank 1919.5 in expectation, which is the control
line both trajectories are read against.

Same 150 pairs, context, block-0 interpolation and checkpoints as neuron_bands_time.py /
neuron_head_identity.py.

Raw -> results/neuron_head_origin_raw.npz, stats -> results/neuron_head_origin_summary.json,
figure -> plots/neuron_head_origin.png.
"""
import os, sys, json, time, itertools
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_path import install, uninstall, record_pair, width, CKPT_DIR, N_PAIRS, SEED
from neuron_bands_time import STEPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
K_HEAD = 8
THRESH = [8, 32, 128, 512]          # "already inside the top-T" curves
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


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

    n_units = None
    rank = None                      # [n_steps, n_pairs, n_units]: rank of each unit (0 = best)
    for si, step in enumerate(STEPS):
        model, _ = load_ckpt(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"), device)
        wraps = install(model)
        for i, (a, b) in enumerate(pairs):
            _, imp = record_pair(model, wraps, seqs, a, b, ts, device)
            if rank is None:
                n_units = imp.size
                rank = np.zeros((len(STEPS), N_PAIRS, n_units), dtype=np.int16)
            order = np.argsort(-imp)
            rank[si, i, order] = np.arange(n_units, dtype=np.int16)
        uninstall(model, wraps)
        del model
        torch.cuda.empty_cache()
        print(f"step {step} done ({time.time()-t0:.0f}s)", flush=True)

    ns = len(STEPS)
    final_head = np.argsort(rank[-1], axis=1)[:, :K_HEAD]      # [n_pairs, 8] step-30000 top-8
    early_head = np.argsort(rank[0], axis=1)[:, :K_HEAD]       # [n_pairs, 8] step-831 top-8
    rows = np.arange(N_PAIRS)[:, None]
    r_final = np.stack([rank[si][rows, final_head] for si in range(ns)])   # [ns, n_pairs, 8]
    r_early = np.stack([rank[si][rows, early_head] for si in range(ns)])

    def q(x, p):
        return [round(float(np.percentile(x[si], p)), 1) for si in range(ns)]

    inside = {t: [round(float(np.mean(r_final[si] < t)), 4) for si in range(ns)] for t in THRESH}
    # how much of the climb from the first checkpoint's rank down to the head is done by each step
    start, end = np.median(r_final[0]), np.median(r_final[-1])
    climb = [round(float((start - np.median(r_final[si])) / (start - end)), 3) for si in range(ns)]

    summary = {
        "context": CONTEXT, "block": 0, "n_t": N_T, "n_pairs": N_PAIRS, "ckpt_dir": CKPT_DIR,
        "steps": STEPS, "n_units": n_units, "k_head": K_HEAD,
        "random_unit_expected_rank": (n_units - 1) / 2,
        "final_head_rank": {"median": q(r_final, 50), "p25": q(r_final, 25), "p75": q(r_final, 75)},
        "early_head_rank": {"median": q(r_early, 50), "p25": q(r_early, 25), "p75": q(r_early, 75)},
        "frac_final_head_inside_top": {str(t): inside[t] for t in THRESH},
        "frac_of_climb_completed": climb,
    }
    with open(os.path.join(RES, "neuron_head_origin_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_head_origin_raw.npz"),
                        pairs=np.array(pairs), steps=np.array(STEPS),
                        r_final=r_final, r_early=r_early)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.array(STEPS, dtype=float)
    a = ax[0]
    med_f, med_e = np.array(q(r_final, 50)), np.array(q(r_early, 50))
    a.fill_between(x, q(r_final, 25), q(r_final, 75), color=CVD[0], alpha=0.18, hatch="//", lw=0)
    a.plot(x, np.maximum(med_f, 0.5), "-o", color=CVD[0], label="the step-30,000 top-8 units")
    a.fill_between(x, q(r_early, 25), q(r_early, 75), color=CVD[1], alpha=0.18, hatch="\\\\", lw=0)
    a.plot(x, np.maximum(med_e, 0.5), "--s", color=CVD[1], label="the step-831 top-8 units")
    a.axhline((n_units - 1) / 2, color="gray", ls=":", lw=1.2)
    a.text(x[0], (n_units - 1) / 2 * 1.12, "a random unit (rank 1919.5)", color="gray", fontsize=8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("training step (log)"); a.set_ylabel("importance rank (log, 0 = most important)")
    a.set_title("(a) the final head units are mid-ranked early\nand climb; the early head falls away")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    b = ax[1]
    for t, c, ls, m in zip(THRESH, CVD, ["-", "--", ":", "-."], ["o", "s", "^", "D"]):
        b.plot(x, 100 * np.array(inside[t]), ls, marker=m, color=c, label=f"inside the top {t}")
    b.set_xscale("log")
    b.set_xlabel("training step (log)")
    b.set_ylabel("% of each pair's step-30,000 top-8 units\nalready ranked that highly")
    b.set_title("(b) they enter the top 128 early\nand the top 8 only at the end")
    b.legend(fontsize=8); b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "neuron_head_origin.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(summary["final_head_rank"], indent=2))
    print(json.dumps(summary["early_head_rank"], indent=2))
    print(json.dumps(summary["frac_final_head_inside_top"], indent=2))
    print("climb fraction:", climb)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
