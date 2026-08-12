"""What does GPT-2's static embedding hold -- the curve's shape, or its crossing width?

Iteration 15 established that GPT-2's embedding probe is above chance (test rho +0.295, permutation
p = 0.020) but reaches only 0.30 of its noise ceiling against the plateau-filtered width, against
0.81 of ceiling inside Pythia-1.4B. Two readings survive that number, and they are different claims:

  (i) the embedding holds the same trait Pythia's does, only weakly; or
 (ii) the embedding holds a DIFFERENT property of the interpolation curve -- how plateau-shaped it is
      -- and the width probe only works to the extent that shape and width are correlated.

Edge drift E = d(0.1) + (1 - d(0.9)) is the shape measurement already stored for every curve: E ~ 0
means the curve leaves both endpoints alone (a plateau), E = 0.2 is the straight line d(t) = t.
Across the 123 tokens the two properties are correlated -- rho(E, w) = +0.770 in GPT-2 and +0.967 in
Pythia-1.4B -- so the two readings cannot be told apart by looking at the width probe alone.

This runs four probes per model on the SAME features, tokens and 50 train/test splits, so every pair
is comparable per split:

  1. shape           target = median edge drift E_u over the token's 18 curves
  2. width           target = median plateau-filtered envelope width (the iteration-15 target)
  3. width | shape   target = rank(width) with rank(E) regressed out
  4. shape | width   target = rank(E) with rank(width) regressed out

Probes 3 and 4 are what separate the two readings: if the embedding holds shape, probe 4 survives and
probe 3 collapses. Each target gets its own split-half reliability (Spearman-Brown over the six
anchors) so accuracies are read against the ceiling that target's own measurement noise allows, and
its own 50-permutation null. Pythia-1.4B runs the identical four probes as the contrast case, the
model where the width probe reaches its ceiling.

Everything is a lookup over stored curves plus the two embedding matrices: zero forward passes, CPU.

Writes results/gpt2_shape.json.
"""
import json

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr, wilcoxon
from transformers import AutoTokenizer, GPT2LMHeadModel, GPTNeoXForCausalLM

from basin_probe import MODEL, REVISION
from common import RESULTS
from edgedrift_analysis import rel, widths
from mlp_read import probe
from xmodel_analysis import half_split

EDGE_CUT = 0.1
N_TRAIN = 80
N_PERM = 50
N_BOOT = 2000

torch.set_num_threads(2)


def resid(y, x):
    """Rank of y with the rank of x linearly regressed out; z-scored so probes are comparable."""
    ry, rx = rankdata(y), rankdata(x)
    rx = (rx - rx.mean()) / rx.std()
    r = ry - ry.mean() - float((ry * rx).mean() / (rx * rx).mean()) * rx
    return r / r.std()


def rel_resid(wc, ec):
    """Split-half reliability of the residual target, residualised inside each half separately."""
    hw = np.array([half_split(row) for row in wc])
    he = np.array([half_split(row) for row in ec])
    ok = np.isfinite(hw).all(1) & np.isfinite(he).all(1)
    a = resid(hw[ok, 0], he[ok, 0])
    b = resid(hw[ok, 1], he[ok, 1])
    r = float(spearmanr(a, b).statistic)
    return 2 * r / (1 + r), int(ok.sum())


def rel_ci(fn, n, n_boot=N_BOOT):
    """Percentile interval for a reliability, resampling the 123 tokens with replacement.

    Every "fraction of ceiling" below divides by sqrt(reliability), and the shape target's
    reliability is small enough that the ratio is worth reading with its uncertainty attached.
    """
    rng = np.random.default_rng(7)
    b = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    b = np.array([x for x in b if np.isfinite(x)])
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def null_dist(F, y, n_perm=N_PERM):
    out = []
    for j in range(n_perm):
        yp = np.random.default_rng(1000 + j).permutation(y)
        r, _, _ = probe(F, yp, np.random.default_rng(1), N_TRAIN)
        out.append(float(r.mean()))
    return np.array(out)


def run(tag, name, F, y, reliability, rel_interval, res):
    """One probe: 50 shared splits, a 50-permutation null, and the ceiling for this target."""
    rho, r2, lam = probe(F, y, np.random.default_rng(1), N_TRAIN)
    null = null_dist(F, y)
    ceil = float(np.sqrt(max(reliability, 0)))
    frac = float(rho.mean() / ceil) if ceil > 0 else None
    res[tag][name] = dict(
        reliability=reliability, reliability_ci=rel_interval, ceiling=ceil,
        rho_mean=float(rho.mean()), rho_sd=float(rho.std()), disattenuated=frac,
        disattenuated_ci=[float(rho.mean() / np.sqrt(c)) if c > 0 else None
                          for c in rel_interval[::-1]],
        r2_mean=float(r2.mean()), median_alpha=lam,
        null_mean=float(null.mean()), null_sd=float(null.std()), null_max=float(null.max()),
        perm_p=float(((null >= rho.mean()).sum() + 1) / (len(null) + 1)),
        rho_per_split=[float(x) for x in rho], null_draws=[float(x) for x in null])
    r = res[tag][name]
    print(f"[{tag}] {name:17s} rho {rho.mean():+.3f} +- {rho.std():.3f}; reliability "
          f"{reliability:+.3f} [{rel_interval[0]:+.3f}, {rel_interval[1]:+.3f}] (ceiling {ceil:.3f})"
          f" -> {'undefined' if frac is None else f'{frac:+.3f}'} of ceiling; null "
          f"{null.mean():+.3f} +- {null.std():.3f}, permutation p {r['perm_p']:.3f}", flush=True)
    return rho


def paired(res, tag, a, b):
    d = np.array(res[tag][a]["rho_per_split"]) - np.array(res[tag][b]["rho_per_split"])
    res[tag][f"paired_{a}_minus_{b}"] = dict(
        mean=float(d.mean()), sd=float(d.std()), frac_a_higher=float((d > 0).mean()),
        p=float(wilcoxon(d).pvalue))
    print(f"[{tag}] paired {a} - {b}: {d.mean():+.3f} +- {d.std():.3f}, {a} higher in "
          f"{(d > 0).mean():.0%} of 50 shared splits (Wilcoxon p "
          f"{res[tag][f'paired_{a}_minus_{b}']['p']:.3g})", flush=True)


def gpt2_features(names):
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
    E = model.transformer.wte.weight.detach().float().numpy()
    del model
    assert all(len(tok(s).input_ids) == 1 for s in names)
    return np.array([E[tok(s).input_ids[0]] for s in names])


def pythia_features(names):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16)
    E = model.gpt_neox.embed_in.weight.detach().float().numpy()
    del model
    assert all(len(tok(s).input_ids) == 1 for s in names)
    return np.array([E[tok(s).input_ids[0]] for s in names])


def model_block(tag, row, raw, names, F, res):
    """The four probes for one model, plus the paired comparisons between them."""
    e_curves = np.array(row["edge_curves"], float).reshape(len(names), -1)
    w_curves = widths(raw, names)
    assert e_curves.shape == w_curves.shape
    w_pl = np.where(e_curves <= EDGE_CUT, w_curves, np.nan)
    assert np.isfinite(w_pl).any(1).all()

    y_e, y_w = np.median(e_curves, 1), np.nanmedian(w_pl, 1)
    res[tag] = {"n_tokens": len(names), "n_features": F.shape[1], "edge_cut": EDGE_CUT,
                "frac_curves_kept": float(np.mean(e_curves <= EDGE_CUT)),
                "rho_shape_width": [float(x) for x in spearmanr(y_e, y_w)[:2]],
                "edge_median": float(np.median(y_e)), "width_median": float(np.median(y_w))}
    print(f"[{tag}] shape and width rank the 123 tokens at rho "
          f"{res[tag]['rho_shape_width'][0]:+.3f}", flush=True)

    n = len(names)
    run(tag, "shape", F, y_e, rel(e_curves)[0],
        rel_ci(lambda i: rel(e_curves[i])[0], n), res)
    run(tag, "width", F, y_w, rel(w_pl)[0],
        rel_ci(lambda i: rel(w_pl[i])[0], n), res)
    run(tag, "width_given_shape", F, resid(y_w, y_e), rel_resid(w_pl, e_curves)[0],
        rel_ci(lambda i: rel_resid(w_pl[i], e_curves[i])[0], n), res)
    run(tag, "shape_given_width", F, resid(y_e, y_w), rel_resid(e_curves, w_pl)[0],
        rel_ci(lambda i: rel_resid(e_curves[i], w_pl[i])[0], n), res)
    paired(res, tag, "shape", "width")
    paired(res, tag, "shape_given_width", "width_given_shape")


def main():
    rows = json.load(open(f"{RESULTS}/edgedrift.json"))["rows"]
    names = rows["gpt2_block0"]["tokens"]
    res = {"tokens": names, "n_perm": N_PERM, "n_train": N_TRAIN}

    model_block("gpt2", rows["gpt2_block0"],
                json.load(open(f"{RESULTS}/xwidth_gpt2.json"))["raw"], names,
                gpt2_features(names), res)
    model_block("pythia_1.4b", rows["1.4b_block0"],
                json.load(open(f"{RESULTS}/xwidth_1.4b.json"))["raw"], names,
                pythia_features(names), res)

    json.dump(res, open(f"{RESULTS}/gpt2_shape.json", "w"), indent=1)
    print(f"wrote {RESULTS}/gpt2_shape.json")


if __name__ == "__main__":
    main()
