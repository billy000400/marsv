"""Is the describability decline a property of the units, or of the 8-character probe family?

neuron_probe_early.py fitted one probe family -- the characters in an 8-character window ending at the
position -- at five checkpoints, and found that units which lose head membership over training become
much harder to read: median held-out R^2 from the current character falls 0.92 -> 0.41 for the demoted
units, and the whole-network median falls 0.39 -> 0.15. Both deliverables already flag the alternative
reading: a unit that stops being readable from eight past characters may have started reading something
that family cannot express, in which case the decline measures the probe, not the unit.

This script tests that directly by widening the family and re-fitting at the two checkpoints where the
unit roles are defined (step 831, the early head; step 30,000, the final head). Five nested-or-disjoint
feature families are fitted from ONE corpus pass per checkpoint, using the same ridge estimator, the
same 80/10/10 window split, and the same lambda grid as neuron_probe.py:

    char1   intercept + the current character                          (the published headline family)
    past8   char1 + lags 1..7 + the full lag0 x lag1 table             (the published full family)
    past32  past8 + lags 8..31                                         (four times the history)
    next8   intercept + the eight characters AFTER the position        (a disjoint family: what comes next)
    all40   past32 + next8                                             (everything in a 40-character neighbourhood)

next8 is the family a "predictive" unit would score well under even though nothing in its past explains
it; past32 is the family a longer-range unit would score well under. If the decline is a probe artefact,
it should shrink or vanish as the family widens. If it survives all40, the units really did become
harder to describe from nearby characters.

Positions are restricted to 32..119 inside each 128-token window so every family sees the same rows (32
characters of history and 8 of lookahead available). That differs from neuron_probe.py's positions 8..127,
so the char1 and past8 numbers here are close to, not identical with, the published ones; their per-unit
correlation against results/neuron_probe_early_raw.npz is the pipeline check.

The reference run's checkpoints had been wiped from /tmp scratch, so the recipe was retrained from the
same seeds before this ran (train_frozen.py --tag ref_pos --seed 1337 --freeze ""). Unit roles are
therefore re-derived here from the retrained run's own top-8 importance sets at the two checkpoints, and
the Jaccard overlap with the published role pools is reported next to the probe correlations.

Raw -> results/neuron_probe_family_raw.npz, stats -> results/neuron_probe_family_summary.json,
figure -> plots/neuron_probe_family.png.
"""
import os, sys, json, time, hashlib, itertools
import numpy as np
import torch
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_feature import gelu_acts, CORPUS, CORPUS_SHA, T, BATCH
from neuron_path import CKPT_DIR, EARLY, install, uninstall, record_pair, N_PAIRS, SEED
from neuron_probe import solve, r2, LAMBDAS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
STEPS = [831, 30000]
NLAG = 32                 # characters of history in the widest past family
NFWD = 8                  # characters of lookahead in the forward family
GROUPS = ["stable", "demoted", "promoted", "never_head"]
FAMILIES = ["char1", "past8", "past32", "next8", "all40"]
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
MK = ["o", "s", "^", "D", "v"]


def groups_of(V):
    """(name, dim) of every one-hot feature group in the union design, in column order."""
    return ([("intercept", 1)] + [(f"lag{l}", V) for l in range(NLAG)]
            + [(f"next{k}", V) for k in range(1, NFWD + 1)] + [("lag0xlag1", V * V)])


def family_cols(V):
    """column indices of each feature family inside the union design."""
    off = np.cumsum([0] + [d for _, d in groups_of(V)])
    lag = lambda l: np.arange(off[1 + l], off[2 + l])
    nxt = lambda k: np.arange(off[1 + NLAG + k - 1], off[2 + NLAG + k - 1])
    inter = np.arange(off[-2], off[-1])
    char1 = np.concatenate([[0], lag(0)])
    past8 = np.concatenate([char1] + [lag(l) for l in range(1, 8)] + [inter])
    past32 = np.concatenate([past8] + [lag(l) for l in range(8, NLAG)])
    next8 = np.concatenate([[0]] + [nxt(k) for k in range(1, NFWD + 1)])
    return {"char1": char1, "past8": past8, "past32": past32, "next8": next8,
            "all40": np.union1d(past32, next8)}


def fit(step, windows, split_of, V, device, t0):
    """held-out R^2 per unit at one checkpoint, for every feature family."""
    model, _ = load_ckpt(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    n_units = len(EARLY) * H
    grp = groups_of(V)
    off = np.cumsum([0] + [d for _, d in grp])
    P = int(off[-1])
    G = [torch.zeros(P, P, dtype=torch.float64, device=device) for _ in range(3)]
    Xty = [torch.zeros(P, n_units, dtype=torch.float64, device=device) for _ in range(3)]
    yty = [torch.zeros(n_units, dtype=torch.float64, device=device) for _ in range(3)]
    sy = [torch.zeros(n_units, dtype=torch.float64, device=device) for _ in range(3)]
    npos = [0, 0, 0]

    for b0 in range(0, windows.shape[0], BATCH):
        idx = torch.as_tensor(windows[b0:b0 + BATCH], device=device)
        a = gelu_acts(model, idx)[:, NLAG:T - NFWD, :]
        B, Tp, _ = a.shape
        lag = [idx[:, NLAG - l:T - NFWD - l].reshape(-1) for l in range(NLAG)]
        fwd = [idx[:, NLAG + k:T - NFWD + k].reshape(-1) for k in range(1, NFWD + 1)]
        feats = [torch.zeros(B * Tp, dtype=torch.long, device=device)] + lag + fwd + [lag[0] * V + lag[1]]
        af = a.reshape(-1, n_units).double()
        # the design matrix of this batch, one column block per feature group (each row has one 1 per group)
        codes = torch.stack(feats, 1) + torch.as_tensor(off[:-1], device=device)
        X = torch.zeros(codes.shape[0], P, dtype=torch.float64, device=device)
        X.scatter_(1, codes, 1.0)
        sp = torch.as_tensor(np.repeat(split_of[b0:b0 + B], Tp), device=device)
        for s in range(3):
            msk = sp == s
            if not bool(msk.any()):
                continue
            ys, xs = af[msk], X[msk]
            npos[s] += int(msk.sum())
            yty[s] += (ys * ys).sum(0)
            sy[s] += ys.sum(0)
            G[s] += xs.T @ xs
            Xty[s] += xs.T @ ys
            del xs
        del a, af, X, codes
    print(f"  step {step}: statistics done, {npos} positions ({time.time()-t0:.0f}s)", flush=True)

    cols = family_cols(V)
    out = {}
    for fam in FAMILIES:
        c = torch.as_tensor(cols[fam], device=device)
        Gs = [g[c][:, c].contiguous() for g in G]
        Xs = [x[c].contiguous() for x in Xty]
        best, best_val = None, -np.inf
        for lam in LAMBDAS:
            bt = solve(Gs[0], Xs[0], c.numel(), lam)
            val = float(np.median(r2(bt, Gs[1], Xs[1], yty[1], sy[1], npos[1])))
            if val > best_val:
                best, best_val = bt, val
        out[fam] = r2(best, Gs[2], Xs[2], yty[2], sy[2], npos[2])
        print(f"  step {step}: {fam} ({c.numel()} cols) median R2 {np.median(out[fam]):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del Gs, Xs
    del G, Xty, yty, sy, model
    torch.cuda.empty_cache()
    return out


def roles(stoi, device, t0):
    """the four unit roles, from this run's own top-8 importance sets at the two checkpoints.

    The /tmp checkpoints of the original reference run were lost to a scratch wipe, so the recipe was
    retrained from the same seeds. Re-deriving the roles here keeps the probe and the roles on the same
    weights; the overlap with the published sets (returned alongside) says how far the retrain drifted.
    """
    chars = sorted(stoi, key=lambda c: stoi[c])
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    all_pairs = list(itertools.combinations(range(len(chars)), 2))
    rng = np.random.default_rng(SEED)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]

    top8 = {}
    for step in STEPS:
        model, _ = load_ckpt(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"), device)
        wraps = install(model)
        t8 = []
        for a, b in pairs:
            _, imp = record_pair(model, wraps, seqs, a, b, ts, device)
            t8.append(np.argsort(-imp)[:8])
        uninstall(model, wraps)
        n_units = imp.size
        top8[step] = np.unique(np.array(t8))
        del model
        torch.cuda.empty_cache()
        print(f"  roles: step {step} top-8 pool {top8[step].size} units ({time.time()-t0:.0f}s)", flush=True)

    early, final = top8[STEPS[0]], top8[STEPS[-1]]
    role = {"promoted": np.setdiff1d(final, early), "demoted": np.setdiff1d(early, final),
            "stable": np.intersect1d(final, early),
            "never_head": np.setdiff1d(np.arange(n_units), np.union1d(final, early))}

    ident = np.load(os.path.join(RES, "neuron_head_identity_raw.npz"))
    si = list(ident["steps"])
    pub = {"early": np.unique(ident["top8"][si.index(STEPS[0])]), "final": np.unique(ident["top8"][-1])}
    check = {f"jaccard_{k}_pool_vs_published": round(float(
        np.intersect1d(v, pub[k]).size / np.union1d(v, pub[k]).size), 3)
        for k, v in [("early", early), ("final", final)]}
    return role, check


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

    R2 = {s: fit(s, windows, split_of, V, device, t0) for s in STEPS}

    # pipeline check: char1 / past8 here vs the published fits of the same two families, at both
    # checkpoints (different position range, and checkpoints retrained from the same seeds)
    prior = np.load(os.path.join(RES, "neuron_probe_early_raw.npz"))
    check = {}
    for s in STEPS:
        for fam, pk in [("char1", "1"), ("past8", "full")]:
            p = prior[f"r2_{pk}_{s}"]
            check[f"pearson_r_{fam}_{s}"] = round(float(np.corrcoef(R2[s][fam], p)[0, 1]), 4)
            check[f"median_{fam}_{s}_here_vs_published"] = [round(float(np.median(R2[s][fam])), 4),
                                                           round(float(np.median(p)), 4)]
    print(json.dumps(check), flush=True)

    n_units = R2[30000]["char1"].size
    role, role_check = roles(stoi, device, t0)
    check.update(role_check)

    summary = {"steps": STEPS, "families": FAMILIES, "n_units": int(n_units),
               "n_cols": {f: int(family_cols(V)[f].size) for f in FAMILIES},
               "positions_per_window": int(T - NFWD - NLAG), "pipeline_check": check,
               "group_sizes": {g: int(role[g].size) for g in GROUPS},
               "medians": {}, "change": {}, "widening_gain_step30000": {}}
    for f in FAMILIES:
        summary["medians"][f] = {str(s): {**{g: round(float(np.median(R2[s][f][role[g]])), 4)
                                             for g in GROUPS},
                                          "all_units": round(float(np.median(R2[s][f])), 4)}
                                 for s in STEPS}
        summary["change"][f] = {}
        for g in GROUPS + ["all_units"]:
            u = np.arange(n_units) if g == "all_units" else role[g]
            d = R2[STEPS[-1]][f][u] - R2[STEPS[0]][f][u]
            summary["change"][f][g] = {"median": round(float(np.median(d)), 4),
                                       "frac_down": round(float((d < 0).mean()), 3),
                                       "n": int(u.size),
                                       "p_wilcoxon": float(wilcoxon(d).pvalue)}
    for g in GROUPS + ["all_units"]:
        u = np.arange(n_units) if g == "all_units" else role[g]
        summary["widening_gain_step30000"][g] = {
            f: round(float(np.median(R2[30000][f][u] - R2[30000]["char1"][u])), 4) for f in FAMILIES}

    with open(os.path.join(RES, "neuron_probe_family_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_probe_family_raw.npz"),
                        steps=np.array(STEPS),
                        **{f"r2_{f}_{s}": R2[s][f] for s in STEPS for f in FAMILIES},
                        **{g: role[g] for g in GROUPS})
    plot(R2, role, n_units)
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


def plot(R2, role, n_units):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))
    x = np.arange(len(FAMILIES))

    a = ax[0]
    for u, c, dx, lab in [(role["demoted"], CVD[0], -0.13, "demoted units"),
                          (np.arange(n_units), CVD[1], 0.13, "all 3,840 units")]:
        m = {s: np.array([np.median(R2[s][f][u]) for f in FAMILIES]) for s in STEPS}
        a.vlines(x + dx, m[STEPS[-1]], m[STEPS[0]], color=c, lw=1.4)
        for s, mk, fill in zip(STEPS, ["o", "s"], [True, False]):
            a.plot(x + dx, m[s], linestyle="none", marker=mk, color=c, ms=7,
                   mfc=c if fill else "white", label=f"{lab}, step {s:,}")
    a.set_xticks(x); a.set_xticklabels(FAMILIES, rotation=20)
    a.set_ylim(0, 1.3); a.set_xlabel("feature family the probe is fitted with")
    a.set_ylabel("median held-out $R^2$")
    a.set_title("(a) the demoted units' fall,\nunder five probe families")
    a.legend(fontsize=7, loc="upper left", ncol=2); a.grid(alpha=0.3)

    b = ax[1]
    for f, c, ls in zip(FAMILIES, CVD, LS):
        d = np.sort(R2[STEPS[-1]][f][role["demoted"]] - R2[STEPS[0]][f][role["demoted"]])
        b.step(d, np.arange(d.size) / d.size, where="post", color=c, ls=ls, lw=1.8, label=f)
    b.axvline(0.0, color="0.35", lw=1.0)
    b.set_xlabel("per-unit change in $R^2$, step 831 $\\to$ 30,000 (demoted units)")
    b.set_ylabel("fraction of units at or below")
    b.set_title("(b) the same fall unit by unit")
    b.legend(fontsize=8, loc="upper left"); b.grid(alpha=0.3)

    d = ax[2]
    xg = np.arange(len(GROUPS))
    for f, c, ls, mk in zip(FAMILIES, CVD, LS, MK):
        med = [np.median(R2[STEPS[-1]][f][role[g]]) for g in GROUPS]
        d.plot(xg, med, linestyle=ls, marker=mk, color=c, lw=1.8, label=f)
    d.set_xticks(xg); d.set_xticklabels([g.replace("_", "-") for g in GROUPS], rotation=20)
    d.set_ylim(0, 1); d.set_ylabel("median held-out $R^2$ at step 30,000")
    d.set_xlabel("the unit's role at the two ends of training")
    d.set_title("(c) how far each family closes\nthe gap between roles")
    d.legend(fontsize=8); d.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "neuron_probe_family.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:            # redraw from the saved fits, no GPU
        z = np.load(os.path.join(RES, "neuron_probe_family_raw.npz"))
        plot({s: {f: z[f"r2_{f}_{s}"] for f in FAMILIES} for s in STEPS},
             {g: z[g] for g in GROUPS}, z[f"r2_char1_{STEPS[0]}"].size)
    else:
        main()
