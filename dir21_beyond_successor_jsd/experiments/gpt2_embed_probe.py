"""Does GPT-2's static embedding carry a width trait, once the target is a reliable one?

The embedding probe inside Pythia-1.4B recovers the per-token anchor width from the token's static
embedding row (test rho +0.76). The same probe inside GPT-2 returned +0.295 against a shuffled-target
control of +0.275 -- read as a null. Two things were wrong with that test:

1. THE TARGET WAS UNRELIABLE. It was the median envelope width over all 18 of a token's curves, whose
   split-half reliability in GPT-2 is 0.319. Keeping only the plateau-shaped curves (edge drift
   E <= 0.1, 56% of them) raises reliability to 0.661 -- so a probe can now be asked to reach a
   ceiling of sqrt(0.661) = 0.813 rather than sqrt(0.319) = 0.565.
2. THE CONTROL WAS ONE SHUFFLE. The shuffled-target rho was computed from a SINGLE permutation of y,
   reused across the 50 train/test splits, so it estimates one draw of the null, not the null. Across
   models those single draws range from -0.201 (Pythia-1.4B) to +0.275 (GPT-2), which is exactly the
   size of the effect being tested. Here the null is 50 independent permutations, each with the same
   50 splits, giving a permutation p-value.

Everything is a lookup over stored curves plus GPT-2's embedding matrix: zero forward passes, CPU only.

Also asks whether the free corpus statistics (log unigram count, successor entropy) predict the
filtered widths as well as the embedding does, and whether either model's embedding lookup ranks the
other model's measured widths.

Writes results/gpt2_probe.json.
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, GPT2LMHeadModel

from checkpoints_analysis import corpus_stats
from common import RESULTS
from edgedrift_analysis import rel, widths
from embed_probe import oof
from mlp_read import probe

EDGE_CUT = 0.1
N_TRAIN = 80
N_PERM = 50

torch.set_num_threads(2)


def null_dist(F, y, n_train, n_perm=N_PERM):
    """Mean test rho of the probe under `n_perm` independent target shuffles, same splits each time."""
    out = []
    for j in range(n_perm):
        yp = np.random.default_rng(1000 + j).permutation(y)
        r, _, _ = probe(F, yp, np.random.default_rng(1), n_train)
        out.append(float(r.mean()))
        if j % 10 == 0:
            print(f"    null shuffle {j}/{n_perm}: {out[-1]:+.3f}", flush=True)
    return np.array(out)


def run(name, F, y, res, n_train=N_TRAIN):
    rho, r2, lam = probe(F, y, np.random.default_rng(1), n_train)
    null = null_dist(F, y, n_train)
    p = float(((null >= rho.mean()).sum() + 1) / (len(null) + 1))
    res[name] = dict(n=len(y), n_train=n_train, n_features=F.shape[1],
                     rho_mean=float(rho.mean()), rho_sd=float(rho.std()),
                     r2_mean=float(r2.mean()), r2_sd=float(r2.std()), median_alpha=lam,
                     null_mean=float(null.mean()), null_sd=float(null.std()),
                     null_min=float(null.min()), null_max=float(null.max()),
                     n_perm=len(null), perm_p=p, rho_per_split=[float(x) for x in rho],
                     null_draws=[float(x) for x in null])
    print(f"[probe] {name}: test rho {rho.mean():+.3f} +- {rho.std():.3f}, R2 {r2.mean():+.3f}; "
          f"null {null.mean():+.3f} +- {null.std():.3f} (range {null.min():+.3f}..{null.max():+.3f}), "
          f"permutation p {p:.3f}", flush=True)
    return rho


def main():
    d = json.load(open(f"{RESULTS}/edgedrift.json"))["rows"]["gpt2_block0"]
    names = d["tokens"]
    raw = json.load(open(f"{RESULTS}/xwidth_gpt2.json"))["raw"]
    w_all = widths(raw, names)
    edge = np.array(d["edge_curves"], float).reshape(len(names), -1)
    assert edge.shape == w_all.shape
    w_pl = np.where(edge <= EDGE_CUT, w_all, np.nan)

    y_all, y_pl = np.nanmedian(w_all, 1), np.nanmedian(w_pl, 1)
    assert np.isfinite(y_pl).all()                       # every token keeps at least one curve
    rel_all, rel_pl = rel(w_all)[0], rel(w_pl)[0]

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
    E = model.transformer.wte.weight.detach().float().numpy()
    del model
    ids = [tok(s).input_ids[0] for s in names]
    assert all(len(tok(s).input_ids) == 1 for s in names)
    F = np.array([E[i] for i in ids])

    res = {"model": "gpt2", "n_tokens": len(names), "tokens": names, "edge_cut": EDGE_CUT,
           "frac_curves_kept": float(np.mean(edge <= EDGE_CUT)),
           "reliability": {"all_curves": rel_all, "plateau_curves": rel_pl},
           "ceiling": {"all_curves": float(np.sqrt(max(rel_all, 0))),
                       "plateau_curves": float(np.sqrt(max(rel_pl, 0)))},
           "rho_targets": float(spearmanr(y_all, y_pl).statistic)}
    print(f"targets: reliability all {rel_all:+.3f} -> plateau {rel_pl:+.3f}; the two targets "
          f"rank tokens at rho {res['rho_targets']:+.3f}", flush=True)

    # the two probes: same features, same protocol, unreliable vs reliable target
    rho_all = run("probe_all_curves", F, y_all, res)
    rho_pl = run("probe_plateau_curves", F, y_pl, res)
    for k in ("probe_all_curves", "probe_plateau_curves"):
        lab = k.replace("probe_", "")
        res[k]["disattenuated"] = float(res[k]["rho_mean"] / res["ceiling"][lab]) \
            if res["ceiling"][lab] > 0 else None

    # the two probes share their 50 train/test splits, so the difference can be read per split
    diff = rho_pl - rho_all
    res["paired_plateau_minus_all"] = dict(
        mean=float(diff.mean()), sd=float(diff.std()),
        frac_splits_plateau_higher=float((diff > 0).mean()),
        p=float(wilcoxon(diff).pvalue))
    print(f"[paired] plateau target minus all-curve target: {diff.mean():+.3f} +- {diff.std():.3f} "
          f"over 50 shared splits, plateau higher in {(diff > 0).mean():.0%} "
          f"(Wilcoxon p {res['paired_plateau_minus_all']['p']:.3g})", flush=True)

    # where the stored single-shuffle control sat in that null: it used one draw, rng(0)
    stored = json.load(open(f"{RESULTS}/xwidth_gpt2.json"))["probe"]["null_rho_mean"]
    one, _, _ = probe(F, np.random.default_rng(0).permutation(y_all), np.random.default_rng(1), N_TRAIN)
    nd = np.array(res["probe_all_curves"]["null_draws"])
    res["stored_single_shuffle"] = dict(
        stored_null_rho=stored, reproduced=float(one.mean()),
        percentile_in_null=float((nd <= one.mean()).mean()),
        n_of_50_draws_at_least_as_high=int((nd >= one.mean()).sum()))
    print(f"[control] stored single-shuffle null {stored:+.3f} (reproduced {one.mean():+.3f}) sits at "
          f"the {res['stored_single_shuffle']['percentile_in_null']:.0%} point of the 50-draw null",
          flush=True)

    # free-lookup baseline: do the two corpus statistics do the same job as 768 embedding dimensions?
    st = corpus_stats(names)
    logc = np.log10([st[s]["count"] for s in names])
    ent = np.array([st[s]["ent"] for s in names])
    res["corpus_stats"] = {
        k: dict(rho_all=[float(x) for x in spearmanr(v, y_all)[:2]],
                rho_plateau=[float(x) for x in spearmanr(v, y_pl)[:2]])
        for k, v in (("log_count", logc), ("successor_entropy", ent))}
    for k, v in res["corpus_stats"].items():
        print(f"[baseline] {k}: rho with plateau widths {v['rho_plateau'][0]:+.3f} "
              f"(p {v['rho_plateau'][1]:.3g})", flush=True)
    rho_cs = run("probe_corpus_stats", np.column_stack([logc, ent]), y_pl, res)
    dc = rho_pl - rho_cs                                 # same target, same 50 splits
    res["paired_embedding_minus_corpus"] = dict(
        mean=float(dc.mean()), sd=float(dc.std()),
        frac_splits_embedding_higher=float((dc > 0).mean()), p=float(wilcoxon(dc).pvalue))
    print(f"[paired] embedding minus corpus statistics on the plateau target: {dc.mean():+.3f} "
          f"+- {dc.std():.3f}, embedding higher in {(dc > 0).mean():.0%} of 50 shared splits "
          f"(Wilcoxon p {res['paired_embedding_minus_corpus']['p']:.3g})", flush=True)

    # does either model's embedding lookup rank the OTHER model's measured widths?
    ref = np.nanmedian(widths(json.load(open(f"{RESULTS}/xwidth_1.4b.json"))["raw"], names), 1)
    p_gpt2 = oof(F, y_pl, np.random.default_rng(2))
    p_14b = np.array([json.load(open(f"{RESULTS}/embed.json"))["probe_pred"][s] for s in names])
    res["lookup"] = dict(
        gpt2_lookup_vs_gpt2_plateau=[float(x) for x in spearmanr(p_gpt2, y_pl)[:2]],
        gpt2_lookup_vs_1_4b_width=[float(x) for x in spearmanr(p_gpt2, ref)[:2]],
        p14b_lookup_vs_gpt2_plateau=[float(x) for x in spearmanr(p_14b, y_pl)[:2]],
        p14b_lookup_vs_gpt2_all=[float(x) for x in spearmanr(p_14b, y_all)[:2]],
        rho_between_lookups=[float(x) for x in spearmanr(p_gpt2, p_14b)[:2]])
    res["probe_pred_gpt2"] = {s: float(v) for s, v in zip(names, p_gpt2)}
    for k, v in res["lookup"].items():
        print(f"[lookup] {k}: rho {v[0]:+.3f} (p {v[1]:.3g})", flush=True)

    json.dump(res, open(f"{RESULTS}/gpt2_probe.json", "w"), indent=1)
    print(f"wrote {RESULTS}/gpt2_probe.json")


if __name__ == "__main__":
    main()
