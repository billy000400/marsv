"""Is the part of a unit that the nonlinear read-out still misses a limit of the PROBE's capacity?

neuron_probe_nonlinear.py replaced the ridge estimator of the character-probe series with a
one-hidden-layer network (width 768) on the same all40 design, and most of what the linear probes miss
turned out to be a nonlinear function of the same characters: at step 30,000 the demoted units go
0.778 -> 0.921, and network-wide the developmental fall reverses. Two things survive that test -- about
40% of the demoted units' fall (-0.064 -> -0.026), and 8% of a demoted unit's response.

That leaves an estimator question before any claim about the units: a *bigger* nonlinear probe might
close the rest. This script varies ONLY the probe's capacity, holding the rows, the split, the design,
the optimiser, the stopping rule and the seed at the values neuron_probe_nonlinear.py used:

    linear   width 768, no hidden nonlinearity   (published, re-used)
    w768     width 768, one hidden layer         (published, re-used)
    w1536    width 1536, one hidden layer        (fitted here)
    w768x2   width 768, two hidden layers        (fitted here)

Weight decay is fixed at 0 because the validation split selected 0 for every fit in the previous run.
The two questions the report consumes:
  1. LEVEL. Does the median held-out R^2 keep climbing with capacity at step 30,000, or has it flattened?
     A flat curve bounds the leftover as something the 40-character neighbourhood does not contain; a
     climbing curve says the previous run was capacity-limited and the leftover is not yet located.
  2. DECLINE. Does the surviving part of the demoted units' fall keep shrinking with capacity?

Raw -> results/neuron_probe_capacity_raw.npz, stats -> results/neuron_probe_capacity_summary.json,
figure -> plots/neuron_probe_capacity.png.
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
from neuron_path import CKPT_DIR
from neuron_probe_nonlinear import (NLAG, NFWD, PATIENCE, LR, RBATCH, N_NEVER, SEED, GROUPS,
                                    batch_codes, n_cols, r2_of)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
STEPS = [831, 30000]
EPOCHS = 600
WD = 0.0
# name -> (hidden width, number of GeLU hidden layers)
NEW = {"w1536": (1536, 1), "w768x2": (768, 2)}
OLD = {"linear": (768, 0), "w768": (768, 1)}       # re-used from neuron_probe_nonlinear_raw.npz
ORDER = ["linear", "w768", "w1536", "w768x2"]
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
MK = ["o", "s", "^", "D", "v"]


class Probe(nn.Module):
    """sum of per-feature embeddings -> (GeLU -> Linear) x depth -> linear read-out per unit."""

    def __init__(self, P, n_units, width, depth):
        super().__init__()
        self.emb = nn.Embedding(P, width)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, n_units)
        self.depth = depth

    def forward(self, codes):
        h = self.emb(codes).sum(1)
        if self.depth:
            h = torch.nn.functional.gelu(h)
        for lin in self.hidden:
            h = torch.nn.functional.gelu(lin(h))
        return self.out(h)


def evaluate(net, codes, Y, rows, mu, sd, chunk):
    with torch.no_grad():
        pred = torch.cat([net(codes[rows[i:i + chunk]]) for i in range(0, rows.numel(), chunk)])
        y = Y[rows].float()
        return r2_of(pred * sd + mu, y, y.mean(0, keepdim=True))


def train_probe(codes, Y, split_rows, mu, sd, P, width, depth, t0, tag):
    chunk = max(2048, 16384 * 768 // width)
    torch.manual_seed(SEED)
    net = Probe(P, Y.shape[1], width, depth).to(codes.device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
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
        v = float(np.median(evaluate(net, codes, Y, split_rows[1], mu, sd, chunk)))
        if v > best_val:
            best_val, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        elif ep - best_ep >= PATIENCE:
            break
    net.load_state_dict(best_state)
    test = evaluate(net, codes, Y, split_rows[2], mu, sd, chunk)
    print(f"  {tag}: best epoch {best_ep} of {ep + 1} run, val median R2 {best_val:.4f}, "
          f"test median R2 {np.median(test):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return test, best_ep


def fit(step, windows, split_of, units, V, device, t0):
    model, _ = load_ckpt(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"), device)
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
    for name, (width, depth) in NEW.items():
        out[name] = train_probe(codes, Y, split_rows, mu, sd, n_cols(V), width, depth, t0,
                                f"step {step} {name}")
        torch.cuda.empty_cache()
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

    # same units as the previous run: all stable/demoted/promoted plus the same random never-head 400
    z = np.load(os.path.join(RES, "neuron_probe_struct_raw.npz"))
    rng = np.random.default_rng(SEED)
    never = np.sort(rng.choice(z["never_head"], size=min(N_NEVER, z["never_head"].size),
                               replace=False))
    units = np.sort(np.concatenate([z["stable"], z["demoted"], z["promoted"], never]))
    prev = np.load(os.path.join(RES, "neuron_probe_nonlinear_raw.npz"))
    assert np.array_equal(prev["units"], units), "unit set differs from the previous run"
    role = {g: prev[g] for g in GROUPS}

    R2 = {s: {"linear": prev[f"r2_linear_{s}"], "w768": prev[f"r2_mlp_{s}"]} for s in STEPS}
    best_ep = {s: {} for s in STEPS}
    for s in STEPS:
        for name, (test, ep) in fit(s, windows, split_of, torch.as_tensor(units, device=device),
                                    V, device, t0).items():
            R2[s][name] = test
            best_ep[s][name] = ep

    summary = {"steps": STEPS, "probes": ORDER,
               "capacity": {n: {"width": w, "hidden_layers": d} for n, (w, d) in
                            {**OLD, **NEW}.items()},
               "reused_from_neuron_probe_nonlinear": list(OLD),
               "epochs_max": EPOCHS, "patience": PATIENCE, "lr": LR, "weight_decay": WD,
               "best_epoch_selected": {str(s): best_ep[s] for s in STEPS},
               "n_design_columns": int(n_cols(V)), "n_units_probed": int(units.size),
               "n_windows": int(windows.shape[0]), "positions_per_window": int(T - NFWD - NLAG),
               "group_sizes": {g: int(role[g].size) for g in GROUPS},
               "medians": {}, "change": {}, "gain_over_w768_step30000": {}}
    for m in ORDER:
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
    for m in ["w1536", "w768x2"]:
        summary["gain_over_w768_step30000"][m] = {}
        for g in GROUPS + ["all_probed"]:
            u = np.arange(units.size) if g == "all_probed" else role[g]
            d = R2[30000][m][u] - R2[30000]["w768"][u]
            summary["gain_over_w768_step30000"][m][g] = {
                "median_gain": round(float(np.median(d)), 4),
                "frac_up": round(float((d > 0).mean()), 3),
                "median_residual_w768": round(float(np.median(1 - R2[30000]["w768"][u])), 4),
                "median_residual": round(float(np.median(1 - R2[30000][m][u])), 4),
                "p_wilcoxon": float(wilcoxon(d).pvalue)}
    with open(os.path.join(RES, "neuron_probe_capacity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_probe_capacity_raw.npz"),
                        steps=np.array(STEPS), units=units,
                        **{f"r2_{m}_{s}": R2[s][m] for s in STEPS for m in ORDER},
                        **{g: role[g] for g in GROUPS})
    plot(R2, role)
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


def plot(R2, role):
    lab = {"linear": "linear\n(768, 0 hidden)", "w768": "768, 1 hidden",
           "w1536": "1536, 1 hidden", "w768x2": "768, 2 hidden"}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.7))
    x = np.arange(len(ORDER))

    a = ax[0]
    for g, c, mk in zip(GROUPS, CVD, MK):
        y = [np.median(R2[30000][m][role[g]]) for m in ORDER]
        a.plot(x, y, marker=mk, color=c, lw=1.8, ms=7, label=f"{g.replace('_','-')} (n={role[g].size})")
    a.set_xticks(x); a.set_xticklabels([lab[m] for m in ORDER], fontsize=8)
    a.set_xlabel("probe capacity (hidden width, hidden layers)")
    a.set_ylabel("median held-out $R^2$, step 30,000")
    a.set_ylim(0.55, 1.0)
    a.set_title("(a) does more probe capacity keep helping?")
    a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.3)

    b = ax[1]
    for g, c, mk in zip(GROUPS, CVD, MK):
        y = [np.median(R2[STEPS[-1]][m][role[g]] - R2[STEPS[0]][m][role[g]]) for m in ORDER]
        b.plot(x, y, marker=mk, color=c, lw=1.8, ms=7, label=g.replace("_", "-"))
    b.axhline(0.0, color="0.35", lw=1.0)
    b.set_xticks(x); b.set_xticklabels([lab[m] for m in ORDER], fontsize=8)
    b.set_xlabel("probe capacity (hidden width, hidden layers)")
    b.set_ylabel("median per-unit change in $R^2$,\nstep 831 $\\rightarrow$ 30,000")
    b.set_title("(b) does the fall keep shrinking?")
    b.legend(fontsize=8, loc="lower right"); b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "neuron_probe_capacity.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        z = np.load(os.path.join(RES, "neuron_probe_capacity_raw.npz"))
        plot({s: {m: z[f"r2_{m}_{s}"] for m in ORDER} for s in STEPS},
             {g: z[g] for g in GROUPS})
    else:
        main()
