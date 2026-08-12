"""Is the part of a demoted unit that no character family explains a NONLINEAR function of the same
characters?

Where the series stands. neuron_probe_early.py fitted a ridge probe predicting each block-1..4 MLP
unit's post-GeLU activation from the characters around the position, and found that units which lose
head membership over training become harder to read. neuron_probe_family.py showed the decline is not
a window artefact (widening the window from 8 to 40 characters moves the demoted units' fall by
0.0002), and neuron_probe_struct.py showed the residual is not word/line position, word or trigram
identity, or the composition of the text before the window (all of those together close 9% of it, and
close as much for every other role). At step 30,000 the demoted units still sit at median held-out
R^2 0.78 under the widest character family (all40).

Every probe in that series is LINEAR in a one-hot design, so all of them share one blind spot: a unit
whose response is a nonlinear function of the very characters the design already contains -- e.g. "an
apostrophe, but only inside a capitalised word" -- would score low under every family fitted so far.
This script changes the ESTIMATOR rather than the feature set. It fits, on exactly the rows, split and
design all40 uses:

    mlp     one hidden layer: sum of per-feature embeddings -> GeLU -> linear read-out per unit
    linear  the same network with the GeLU removed (same parameters, same optimiser, same epochs)

`linear` is the control that separates nonlinearity from capacity and optimiser: it can represent any
linear map of the design (its width, 768, exceeds the 797 probed units), so `mlp` - `linear` is the
value of the nonlinearity alone. Ridge all40 is carried over from results/neuron_probe_struct_raw.npz
as the published linear baseline, and `linear` vs ridge is the pipeline check -- two estimators of the
same linear model on the same rows must agree closely.

The two questions the report consumes:
  1. LEVEL. Does the nonlinear probe close the demoted units' residual at step 30,000, and is any gain
     specific to them or shared with the units that stay readable?
  2. DECLINE. Does the fall from step 831 to step 30,000 shrink under the nonlinear probe? If the
     demoted units had merely switched to a nonlinear function of nearby characters, it should.

Units: all 75 stable, 141 demoted and 181 promoted units, plus a fixed random 400 of the 3,443
never-head units (activations for all 3,840 do not fit in the memory budget).

Raw -> results/neuron_probe_nonlinear_raw.npz, stats -> results/neuron_probe_nonlinear_summary.json,
figure -> plots/neuron_probe_nonlinear.png.
"""
import os, sys, json, time, hashlib
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from allpairs_sweep import load_vocab
from frozen_assay import load_ckpt
from neuron_feature import gelu_acts, CORPUS, CORPUS_SHA, T, BATCH
from neuron_path import CKPT_DIR, EARLY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
STEPS = [831, 30000]
NLAG = 32              # characters of history in the all40 design (lag 0 = current character)
NFWD = 8               # characters of lookahead in it
WIDTH = 768            # hidden width of both probes
EPOCHS = 600         # cap; training stops early when validation R^2 has not improved for PATIENCE
PATIENCE = 20
LR = 1e-3
WDS = [0.0, 1e-4]      # weight decay grid, selected on the validation split
RBATCH = 4096          # rows per optimiser step
N_NEVER = 400          # never-head units kept
SEED = 0
GROUPS = ["stable", "demoted", "promoted", "never_head"]
PROBES = ["linear", "mlp"]
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
MK = ["o", "s", "^", "D", "v"]


def n_cols(V):
    """columns of the all40 design: intercept, lags 0..31, next 1..8, and the lag0 x lag1 table."""
    return 1 + (NLAG + NFWD) * V + V * V


def batch_codes(idx, V, off):
    """one-hot group codes [rows, 1+NLAG+NFWD+1] for one batch of windows, rows = (window, position)
    with position in NLAG..T-NFWD-1, flattened in that order (the row order of the ridge runs)."""
    lo, hi = NLAG, idx.shape[1] - NFWD
    lag = [idx[:, lo - l:hi - l] for l in range(NLAG)]
    fwd = [idx[:, lo + k:hi + k] for k in range(1, NFWD + 1)]
    codes = torch.stack([torch.zeros_like(lag[0])] + lag + fwd + [lag[0] * V + lag[1]], dim=-1)
    return (codes.reshape(-1, codes.shape[-1]) + off).int()


class Probe(nn.Module):
    """sum of per-feature embeddings -> (GeLU) -> linear read-out, one output per probed unit."""

    def __init__(self, P, n_units, nonlinear):
        super().__init__()
        self.emb = nn.Embedding(P, WIDTH)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.out = nn.Linear(WIDTH, n_units)
        self.nonlinear = nonlinear

    def forward(self, codes):
        h = self.emb(codes).sum(1)
        return self.out(torch.nn.functional.gelu(h) if self.nonlinear else h)


def r2_of(pred, y, mean):
    """held-out R^2 per unit: 1 - SSE / sum((y - test mean)^2)."""
    sse = ((pred - y) ** 2).sum(0)
    tss = ((y - mean) ** 2).sum(0).clamp(min=1e-12)
    return (1.0 - sse / tss).cpu().numpy()


def evaluate(net, codes, Y, rows, mu, sd, chunk=16384):
    """R^2 on a row subset, in the units' original scale."""
    with torch.no_grad():
        pred = torch.cat([net(codes[rows[i:i + chunk]]) for i in range(0, rows.numel(), chunk)])
        y = Y[rows].float()
        return r2_of(pred * sd + mu, y, y.mean(0, keepdim=True))


def train_probe(codes, Y, split_rows, mu, sd, P, nonlinear, wd, t0, tag):
    torch.manual_seed(SEED)
    net = Probe(P, Y.shape[1], nonlinear).to(codes.device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=wd)
    tr = split_rows[0]
    best_val, best_state, best_ep = -np.inf, None, -1
    for ep in range(EPOCHS):
        perm = tr[torch.randperm(tr.numel(), device=tr.device)]
        for i in range(0, perm.numel(), RBATCH):
            r = perm[i:i + RBATCH]
            loss = ((net(codes[r]) - (Y[r].float() - mu) / sd) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = float(np.median(evaluate(net, codes, Y, split_rows[1], mu, sd)))
        if v > best_val:
            best_val, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        elif ep - best_ep >= PATIENCE:
            break
    net.load_state_dict(best_state)
    test = evaluate(net, codes, Y, split_rows[2], mu, sd)
    print(f"  {tag} wd={wd:g}: best epoch {best_ep} of {ep + 1} run, val median R2 {best_val:.4f}, "
          f"test median R2 {np.median(test):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return best_val, test, best_ep


def fit(step, windows, split_of, units, V, device, t0):
    """cache activations + codes for one checkpoint, then fit both probes over the wd grid."""
    model, _ = load_ckpt(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"), device)
    # column offset of each one-hot group: intercept, lags 0..31, next 1..8, lag0 x lag1
    off = torch.as_tensor(np.cumsum([0, 1] + [V] * (NLAG + NFWD)), device=device).int()
    Tp = T - NFWD - NLAG
    acts, codes = [], []
    for b0 in range(0, windows.shape[0], BATCH):
        idx = torch.as_tensor(windows[b0:b0 + BATCH], device=device)
        a = gelu_acts(model, idx)[:, NLAG:T - NFWD, :]
        acts.append(a.reshape(-1, a.shape[-1])[:, units].half())
        codes.append(batch_codes(idx, V, off))
        del a, idx
    del model
    torch.cuda.empty_cache()
    Y = torch.cat(acts); codes = torch.cat(codes)
    del acts
    spl = torch.as_tensor(np.repeat(split_of, Tp), device=device)
    split_rows = [torch.nonzero(spl == s, as_tuple=True)[0] for s in range(3)]
    print(f"  step {step}: activations cached {tuple(Y.shape)}, "
          f"rows {[int(r.numel()) for r in split_rows]} ({time.time()-t0:.0f}s)", flush=True)

    ytr = Y[split_rows[0]].float()
    mu, sd = ytr.mean(0, keepdim=True), ytr.std(0, keepdim=True).clamp(min=1e-6)
    del ytr
    out = {}
    for probe in PROBES:
        best = (-np.inf, None, None, None)
        for wd in WDS:
            v, test, ep = train_probe(codes, Y, split_rows, mu, sd, n_cols(V), probe == "mlp", wd,
                                      t0, f"step {step} {probe}")
            if v > best[0]:
                best = (v, test, wd, ep)
        out[probe] = (best[1], best[2], best[3])
    del Y, codes, split_rows
    torch.cuda.empty_cache()
    return out


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
    if "--smoke" in sys.argv:
        windows, split_of = windows[:400], split_of[:400]

    z = np.load(os.path.join(RES, "neuron_probe_struct_raw.npz"))
    rng = np.random.default_rng(SEED)
    never = np.sort(rng.choice(z["never_head"], size=min(N_NEVER, z["never_head"].size),
                               replace=False))
    units = np.sort(np.concatenate([z["stable"], z["demoted"], z["promoted"], never]))
    pos = {u: i for i, u in enumerate(units)}
    role = {g: np.array([pos[u] for u in (never if g == "never_head" else z[g])]) for g in GROUPS}
    ridge_units = {s: z[f"r2_all40_{s}"][units] for s in STEPS}

    R2, wd_pick, best_ep = {}, {}, {}
    for s in STEPS:
        r = fit(s, windows, split_of, torch.as_tensor(units, device=device), V, device, t0)
        R2[s] = {p: r[p][0] for p in PROBES}
        R2[s]["ridge_all40"] = ridge_units[s]
        wd_pick[s] = {p: r[p][1] for p in PROBES}
        best_ep[s] = {p: r[p][2] for p in PROBES}

    methods = ["ridge_all40"] + PROBES
    summary = {"steps": STEPS, "probes": methods, "width": WIDTH, "epochs_max": EPOCHS, "patience": PATIENCE, "lr": LR,
               "weight_decay_grid": WDS, "weight_decay_selected": {str(s): wd_pick[s] for s in STEPS},
               "best_epoch_selected": {str(s): best_ep[s] for s in STEPS},
               "n_design_columns": int(n_cols(V)), "n_units_probed": int(units.size),
               "n_windows": int(windows.shape[0]), "positions_per_window": int(T - NFWD - NLAG),
               "group_sizes": {g: int(role[g].size) for g in GROUPS},
               "medians": {}, "change": {}, "gain_over_ridge_step30000": {},
               "gain_over_linear_step30000": {}, "pipeline_check": {}}
    for s in STEPS:
        d = R2[s]["linear"] - R2[s]["ridge_all40"]
        summary["pipeline_check"][f"linear_minus_ridge_{s}"] = {
            "median": round(float(np.median(d)), 4),
            "max_abs": round(float(np.max(np.abs(d))), 4),
            "pearson_r": round(float(np.corrcoef(R2[s]["linear"], R2[s]["ridge_all40"])[0, 1]), 4)}
    for m in methods:
        summary["medians"][m] = {str(s): {**{g: round(float(np.median(R2[s][m][role[g]])), 4)
                                             for g in GROUPS},
                                          "all_probed": round(float(np.median(R2[s][m])), 4)}
                                 for s in STEPS}
        summary["change"][m] = {}
        for g in GROUPS + ["all_probed"]:
            u = np.arange(units.size) if g == "all_probed" else role[g]
            d = R2[STEPS[-1]][m][u] - R2[STEPS[0]][m][u]
            summary["change"][m][g] = {"median": round(float(np.median(d)), 4),
                                       "frac_down": round(float((d < 0).mean()), 3),
                                       "n": int(u.size),
                                       "p_wilcoxon": float(wilcoxon(d).pvalue)}
    for base, key in [("ridge_all40", "gain_over_ridge_step30000"),
                      ("linear", "gain_over_linear_step30000")]:
        for g in GROUPS + ["all_probed"]:
            u = np.arange(units.size) if g == "all_probed" else role[g]
            d = R2[30000]["mlp"][u] - R2[30000][base][u]
            share = d / np.clip(1 - R2[30000][base][u], 1e-9, None)
            summary[key][g] = {"median_gain": round(float(np.median(d)), 4),
                               "median_per_unit_share_of_residual_closed": round(float(np.median(share)), 4),
                               "frac_up": round(float((d > 0).mean()), 3),
                               "median_residual_base": round(float(np.median(1 - R2[30000][base][u])), 4),
                               "median_residual_mlp": round(float(np.median(1 - R2[30000]["mlp"][u])), 4),
                               "frac_of_residual_closed": round(float(
                                   np.median(d) / max(np.median(1 - R2[30000][base][u]), 1e-9)), 4),
                               "p_wilcoxon": float(wilcoxon(d).pvalue)}
    with open(os.path.join(RES, "neuron_probe_nonlinear_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_probe_nonlinear_raw.npz"),
                        steps=np.array(STEPS), units=units,
                        **{f"r2_{m}_{s}": R2[s][m] for s in STEPS for m in methods},
                        **{g: role[g] for g in GROUPS})
    plot(R2, role, units.size)
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


def plot(R2, role, n_units):
    methods = ["ridge_all40", "linear", "mlp"]
    lab = {"ridge_all40": "ridge (published)", "linear": "linear, same net", "mlp": "one hidden layer"}
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

    a = ax[0]
    x = np.arange(len(methods))
    for u, c, dx, name in [(role["demoted"], CVD[0], -0.13, "demoted units"),
                           (np.arange(n_units), CVD[1], 0.13, "all probed units")]:
        m = {s: np.array([np.median(R2[s][p][u]) for p in methods]) for s in STEPS}
        a.vlines(x + dx, m[STEPS[-1]], m[STEPS[0]], color=c, lw=1.4)
        for s, mk, fill in zip(STEPS, ["o", "s"], [True, False]):
            a.plot(x + dx, m[s], linestyle="none", marker=mk, color=c, ms=7,
                   mfc=c if fill else "white", label=f"{name}, step {s:,}")
    a.set_xticks(x); a.set_xticklabels([lab[p] for p in methods], rotation=15)
    a.set_ylim(0, 1.25); a.set_xlabel("probe fitted on the all40 character design")
    a.set_ylabel("median held-out $R^2$")
    a.set_title("(a) a nonlinear read-out of the same characters")
    a.legend(fontsize=7, loc="upper left", ncol=2); a.grid(alpha=0.3)

    b = ax[1]
    for g, c, mk in zip(GROUPS, CVD, MK):
        d = np.sort(R2[30000]["mlp"][role[g]] - R2[30000]["linear"][role[g]])
        b.step(d, np.arange(d.size) / d.size, where="post", color=c, lw=1.8,
               marker=mk, markevery=max(1, d.size // 8), ms=5,
               label=f"{g.replace('_','-')} (n={d.size})")
    b.axvline(0.0, color="0.35", lw=1.0)
    b.set_xlim(-0.02, 0.60)
    b.set_xlabel("per-unit $R^2$ gain of the hidden layer over the linear net,\n"
                 "step 30,000 (x clipped at 0.60)")
    b.set_ylabel("fraction of units at or below")
    b.set_title("(b) who the nonlinearity helps")
    b.legend(fontsize=8, loc="lower right"); b.grid(alpha=0.3)

    d = ax[2]
    xg = np.arange(len(GROUPS))
    for p, c, mk in zip(methods, [CVD[0], CVD[2], CVD[1]], [MK[0], MK[2], MK[1]]):
        ch = [np.median(R2[STEPS[-1]][p][role[g]] - R2[STEPS[0]][p][role[g]]) for g in GROUPS]
        d.plot(xg, ch, linestyle="none", marker=mk, color=c, ms=8, label=lab[p])
    d.axhline(0.0, color="0.35", lw=1.0)
    d.set_xticks(xg); d.set_xticklabels([g.replace("_", "-") for g in GROUPS], rotation=15)
    d.set_ylabel("median per-unit change in $R^2$,\nstep 831 $\\rightarrow$ 30,000")
    d.set_xlabel("unit role over training")
    d.set_title("(c) does the fall shrink?")
    d.legend(fontsize=8, loc="lower left"); d.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "neuron_probe_nonlinear.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        z = np.load(os.path.join(RES, "neuron_probe_nonlinear_raw.npz"))
        plot({s: {m: z[f"r2_{m}_{s}"] for m in ["ridge_all40"] + PROBES} for s in STEPS},
             {g: z[g] for g in GROUPS}, z["units"].size)
    else:
        main()
