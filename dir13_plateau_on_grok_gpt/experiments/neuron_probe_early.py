"""Were the early head units already describable when they held the head?

neuron_head_describe.py compared the units training promotes into the head with the ones it demotes,
using the character-window probe fitted on the step-30,000 model. Its limitation is that one checkpoint
carries the whole comparison: it says the finished network's early-selected units are the describable
ones, but not whether they were describable back when they were doing the work. This script removes that
limitation by refitting the SAME probe on the step-831 model and repeating the comparison there.

The probe is neuron_probe.py's, unchanged: for every block-1..4 unit, a ridge regression predicts the
unit's post-GeLU activation on held-out corpus positions from the characters in an 8-character window
(additive one-hot terms for lags 0..7 plus the full lag0 x lag1 interaction table), with lambda chosen
on a validation split and R^2 reported on a third split. Two readings are kept: R^2 from the current
character alone (r2_1) and from the full window (r2_full). Refitting it at step 831 makes describability
a per-checkpoint property, so each unit has a value at both ends of training.

Unit roles are the same four as neuron_head_describe.py, read from neuron_head_identity.py's per-pair
top-8 sets: promoted (final head only), demoted (early head only), stable (both), never-head (neither).
Present importance is conditioned on WITHIN each checkpoint, using that checkpoint's own per-pair top-8
and top-32 sets: band 0-7 holds the units some pair ranks in its top 8, band 8-31 the units some pair
ranks in its top 32 but none in its top 8. At step 831 that gives the two contrasts this script is for:
inside the early head, the units that keep the head (stable) against those that lose it (demoted); and
one band below it, the units that will be promoted against those that never hold the head at all.

The fit is repeated at all five checkpoints of the developmental series, so the two-point comparison
becomes a trajectory; the forward-looking band contrasts stay at step 831 and step 30,000, where the
roles are defined. Runtime is one corpus pass per checkpoint (~10 s each, no ablations). The step-30,000
fit is the pipeline check: its per-unit R^2 must reproduce results/neuron_probe_raw.npz.

Raw -> results/neuron_probe_early_raw.npz, stats -> results/neuron_probe_early_summary.json,
figure -> plots/neuron_probe_early.png.
"""
import os, sys, json, time, hashlib
import numpy as np
import torch
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from allpairs_sweep import load_vocab
from frozen_assay import load_ckpt
from neuron_feature import gelu_acts, CORPUS, CORPUS_SHA, T, BATCH
from neuron_path import CKPT_DIR, EARLY
from neuron_probe import blocks_of, solve, r2, L, MIN_POS, LAMBDAS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
STEPS = [831, 2038, 5000, 12500, 30000]
GROUPS = ["stable", "demoted", "promoted", "never_head"]
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9"]
LS = ["-", "--", "-.", ":"]


def fit_probe(step, windows, split_of, V, device, t0):
    """held-out R^2 per unit at one checkpoint: from the current character alone, and full window."""
    model, _ = load_ckpt(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    n_units = len(EARLY) * H
    grp = blocks_of(V)
    off = np.cumsum([0] + [d for _, d in grp])
    P = int(off[-1])
    G = [torch.zeros(P, P, dtype=torch.float64, device=device) for _ in range(3)]
    Xty = [torch.zeros(P, n_units, dtype=torch.float64, device=device) for _ in range(3)]
    yty = [torch.zeros(n_units, dtype=torch.float64, device=device) for _ in range(3)]
    sy = [torch.zeros(n_units, dtype=torch.float64, device=device) for _ in range(3)]
    npos = [0, 0, 0]
    n_win = windows.shape[0]

    for b0 in range(0, n_win, BATCH):
        idx = torch.as_tensor(windows[b0:b0 + BATCH], device=device)
        a = gelu_acts(model, idx)[:, MIN_POS:, :]
        B, Tp, _ = a.shape
        lag = [idx[:, MIN_POS - l:T - l].reshape(-1) for l in range(L)]
        feats = [torch.zeros(B * Tp, dtype=torch.long, device=device)] + lag + [lag[0] * V + lag[1]]
        af = a.reshape(-1, n_units).double()
        sp = torch.as_tensor(np.repeat(split_of[b0:b0 + B], Tp), device=device)
        for s in range(3):
            msk = sp == s
            if not bool(msk.any()):
                continue
            ys = af[msk]
            npos[s] += int(msk.sum())
            yty[s] += (ys * ys).sum(0)
            sy[s] += ys.sum(0)
            fs = [f[msk] for f in feats]
            for gi, (_, dg) in enumerate(grp):
                Xty[s][off[gi]:off[gi + 1]].index_add_(0, fs[gi], ys)
                for hi in range(gi, len(grp)):
                    dh = grp[hi][1]
                    cnt = torch.bincount(fs[gi] * dh + fs[hi], minlength=dg * dh).reshape(dg, dh)
                    G[s][off[gi]:off[gi + 1], off[hi]:off[hi + 1]] += cnt.double()
                    if hi != gi:
                        G[s][off[hi]:off[hi + 1], off[gi]:off[gi + 1]] += cnt.T.double()
        del a, af
    print(f"  step {step}: statistics done, {npos} positions ({time.time()-t0:.0f}s)", flush=True)

    out = {}
    for name, nc in [("1", int(off[2])), ("full", P)]:
        best, best_val = None, -np.inf
        for lam in LAMBDAS:
            bt = solve(G[0], Xty[0], nc, lam)
            val = float(np.median(r2(bt, G[1], Xty[1], yty[1], sy[1], npos[1])))
            if val > best_val:
                best, best_val = bt, val
        out[name] = r2(best, G[2], Xty[2], yty[2], sy[2], npos[2])
        print(f"  step {step}: r2_{name} median {np.median(out[name]):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    del G, Xty, yty, sy, model
    torch.cuda.empty_cache()
    return out


def test(x, g1, g2):
    u, p = mannwhitneyu(x[g1], x[g2], alternative="two-sided")
    return {"n1": int(g1.size), "n2": int(g2.size),
            "median1": round(float(np.median(x[g1])), 4), "median2": round(float(np.median(x[g2])), 4),
            "cles": round(float(u / (g1.size * g2.size)), 3), "p": float(p)}


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    t0 = time.time()

    stoi = load_vocab()
    V = len(stoi)
    raw = open(CORPUS, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == CORPUS_SHA, "corpus SHA mismatch"
    data = np.array([stoi[c] for c in raw.decode("utf-8")], dtype=np.int64)
    train_data = data[:int(0.9 * len(data))]
    n_win = len(train_data) // T
    windows = train_data[:n_win * T].reshape(n_win, T)
    split_of = np.where(np.arange(n_win) % 10 == 8, 1, np.where(np.arange(n_win) % 10 == 9, 2, 0))

    R2 = {step: fit_probe(step, windows, split_of, V, device, t0) for step in STEPS}

    # pipeline check: the step-30,000 fit must reproduce the published one
    prior = np.load(os.path.join(RES, "neuron_probe_raw.npz"))
    check = {f"max_abs_diff_r2_{k}_step30000": float(np.abs(R2[30000][k] - prior[f"r2_{k}"]).max())
             for k in ["1", "full"]}
    assert max(check.values()) < 1e-6, check

    ident = np.load(os.path.join(RES, "neuron_head_identity_raw.npz"))
    steps_id = list(ident["steps"])
    top8, top32 = ident["top8"], ident["top32"]
    n_units = R2[30000]["1"].size
    final, early = np.unique(top8[-1]), np.unique(top8[steps_id.index(831)])
    role = {"promoted": np.setdiff1d(final, early), "demoted": np.setdiff1d(early, final),
            "stable": np.intersect1d(final, early),
            "never_head": np.setdiff1d(np.arange(n_units), np.union1d(final, early))}

    summary = {"steps": STEPS, "n_units": int(n_units), "pipeline_check": check,
               "group_sizes": {g: int(role[g].size) for g in GROUPS},
               "medians": {}, "median_change": {}, "within_band": {}}
    for m in ["1", "full"]:
        summary["medians"][m] = {str(s): {**{g: round(float(np.median(R2[s][m][role[g]])), 4)
                                             for g in GROUPS},
                                          "all_units": round(float(np.median(R2[s][m])), 4)}
                                 for s in STEPS}
        summary["median_change"][m] = {g: round(float(np.median(R2[STEPS[-1]][m][role[g]]
                                                               - R2[STEPS[0]][m][role[g]])), 4)
                                       for g in GROUPS}
    # each checkpoint's own head and just-below-head bands, from that checkpoint's top-8 / top-32 sets
    contrasts = {831: [("0-7", "stable", "demoted"), ("8-31", "promoted", "never_head")],
                 30000: [("0-7", "stable", "promoted"), ("8-31", "demoted", "never_head")]}
    band = {}
    for s in contrasts:
        si = steps_id.index(s)
        head = np.unique(top8[si])
        band[s] = {"0-7": head, "8-31": np.setdiff1d(np.unique(top32[si]), head)}
    for m in ["1", "full"]:
        summary["within_band"][m] = {}
        for s in contrasts:
            summary["within_band"][m][str(s)] = {}
            for b, g1, g2 in contrasts[s]:
                a1 = np.intersect1d(band[s][b], role[g1])
                a2 = np.intersect1d(band[s][b], role[g2])
                summary["within_band"][m][str(s)][f"{b}:{g1}_vs_{g2}"] = test(R2[s][m], a1, a2)

    with open(os.path.join(RES, "neuron_probe_early_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_probe_early_raw.npz"),
                        steps=np.array(STEPS),
                        **{f"r2_{m}_{s}": R2[s][m] for s in STEPS for m in ["1", "full"]},
                        **{g: role[g] for g in GROUPS})

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    a = ax[0]
    for g, c, ls in zip(GROUPS, CVD, LS):
        x = np.sort(R2[831]["1"][role[g]])
        a.step(np.concatenate([[0.0], x]), np.arange(x.size + 1) / x.size, where="post",
               color=c, ls=ls, lw=1.8, label=f"{g.replace('_', '-')} (n={role[g].size})")
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_xlabel("held-out $R^2$ at step 831, from the current character alone")
    a.set_ylabel("fraction of units at or below")
    a.set_title("(a) describability at step 831,\nby the unit's role at the two ends of training")
    a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)

    b = ax[1]
    xs = np.array(STEPS, dtype=float)
    for g, c, ls, mk in zip(GROUPS, CVD, LS, ["o", "s", "^", "D"]):
        med = [np.median(R2[s]["1"][role[g]]) for s in STEPS]
        b.plot(xs, med, ls, marker=mk, color=c, label=g.replace("_", "-"))
        b.annotate(g.replace("_", "-"), (xs[-1] * 1.1, med[-1]), fontsize=8, color=c, va="center")
    b.plot(xs, [np.median(R2[s]["1"]) for s in STEPS], color="gray", ls=(0, (3, 1, 1, 1)),
           marker="x", label="all units")
    b.annotate("all units", (xs[-1] * 1.1, np.median(R2[STEPS[-1]]["1"])), fontsize=8,
               color="gray", va="center")
    b.set_xscale("log"); b.set_xlim(xs[0] * 0.8, xs[-1] * 3.2); b.set_ylim(0, 1)
    b.set_ylabel("median held-out $R^2$ from the current character alone")
    b.set_xlabel("checkpoint the probe was fitted on (training step, log)")
    b.set_title("(b) how each group's describability moves\nover training")
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "neuron_probe_early.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
