"""S14: where between f = 1 and f ~ 0.5 does the fixed-set head effect die?

Experiment 9 measured the held-out fixed-set ablation at two patch sites in gpt2-large (block 0,
f = 1: paired delta +0.187; block 18, f = 0.486: -0.002) and could not say where in between the
effect disappears. This sweeps three intermediate sites (blocks 4, 9, 13) plus both endpoints on one
subsample of the low-JSD bank, holding EVERYTHING else fixed: the head set is the stored held-out
fixed set from Experiment 8 (ranked on the other half of the prefixes over the full bank), so the
only thing that varies along the curve is the patch site.

Conditions per pair and site: base (no ablation), ctrl (per-pair engagement-matched control set),
glob (the fixed cross-pair set). Writes results/depth_curve.json and plots/depth_curve.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon

from ablate_heads import Ablation, N_MEAN, head_means
from analyze import tv_width
from analyze_bank import cluster_boot
from common import PLOTS, RESULTS, load
from depth_gap import DOSE, per_pair_sets, run_at
from mine_lowjsd import N_PREFIX, SEED, get_prefixes

MKEY = os.environ.get("MKEY", "gpt2-large")
# gpt2-large default: f = 1.000 / 0.886 / 0.743 / 0.629 / 0.486 on 36 blocks. S15 re-runs the
# top-of-stack part of the curve (blocks 0-4) in the other two GPT-2 models.
SITES = [int(x) for x in os.environ.get("SITES", "0,4,9,13,18").split(",")]
N_PAIRS = int(os.environ.get("N_PAIRS", 60))
N_ALPHA = 101
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)


def stats(rows, rng_boot):
    """Median w_TV per condition and the paired fixed-set-minus-control effect, clustered."""
    cl = {}
    for r in rows:
        cl.setdefault(r["prefix_idx"], []).append(r)
    g = np.array([r["wtv_glob"] for r in rows])
    c = np.array([r["wtv_ctrl"] for r in rows])
    lo, hi = cluster_boot(
        cl, lambda rr: float(np.median([q["wtv_glob"] - q["wtv_ctrl"] for q in rr])), rng_boot)
    med = {k: float(np.median([r[f"wtv_{k}"] for r in rows])) for k in ("base", "ctrl", "glob")}
    # Headroom to the linear response. Below ~0.05 the ratio is a small number over a small number
    # and stops being informative, so it is reported as undefined rather than as a large percentage.
    head = 0.5 - med["ctrl"]
    return dict(
        n=len(rows), n_prefix=len(cl), median_wtv=med, headroom=float(head),
        median_delta=float(np.median(g - c)), ci_delta=[lo, hi],
        norm_delta=float(np.median(g - c) / head) if head >= 0.05 else None,
        wilcoxon_p=float(wilcoxon(g, c)[1]), frac_glob_gt_ctrl=float(np.mean(g > c)),
        max_endpoint_err=float(max(max(r[f"err_{k}"] for k in ("base", "ctrl", "glob"))
                                   for r in rows)))


def main():
    alphas = np.linspace(0, 1, N_ALPHA)
    rng_boot = np.random.default_rng(0)
    suffix = "" if MKEY == "gpt2-large" else f"_{MKEY}"
    path = os.path.join(RESULTS, f"depth_curve{suffix}.json")
    out = json.load(open(path)) if os.path.exists(path) else {}

    bank = json.load(open(os.path.join(RESULTS, f"lowjsd_{MKEY}.json")))
    idx = np.linspace(0, len(bank) - 1, min(N_PAIRS, len(bank))).round().astype(int)
    rows = [bank[i] for i in sorted(set(idx.tolist()))]

    tok, m = load(MKEY)
    prefixes = get_prefixes(tok, np.random.default_rng(SEED), N_PREFIX)
    nh, n_block = m.config.n_head, len(m.transformer.h)
    dh = m.config.n_embd // nh
    k = max(4, int(round(DOSE * nh * n_block)))

    # The held-out fixed sets from Experiment 8, ranked on the FULL bank's opposite prefix fold.
    # gpt2-small's were produced by the Experiment 9 block-0 run instead, so fall back to those.
    src = json.load(open(os.path.join(RESULTS, "localize_heads.json")))[MKEY].get("global_sets")
    if src is None:
        src = json.load(open(os.path.join(RESULTS, "depth_gap.json")))[MKEY]["block0"]["sets"]
    gsets = {int(f): [int(g) for g in G] for f, G in src.items()}
    print(f"{MKEY}: k={k} of {nh * n_block} heads, {len(rows)} pairs, sites {SITES}", flush=True)

    _, ctrl, prefix_of = per_pair_sets(m, rows, prefixes, k, nh)
    seen, mean_prompts = set(), []
    for r in rows:
        if r["prefix_idx"] not in seen and len(mean_prompts) < N_MEAN:
            seen.add(r["prefix_idx"])
            mean_prompts.append(prefixes[r["prefix_idx"]] + [r["id_a"]])
    means = head_means(m, mean_prompts)

    out.setdefault("meta", {}).update(
        model=MKEY, n_block=n_block, k=k, n_heads=nh * n_block, dose=DOSE, n_pairs=len(rows),
        sites=SITES, f={str(L): (n_block - 1 - L) / (n_block - 1) for L in SITES},
        global_set_block0={str(f): float(np.mean([g // nh == 0 for g in G]))
                           for f, G in gsets.items()})

    with Ablation(m, means, dh) as abl:
        for L in SITES:
            if str(L) in out.get("sites", {}):
                continue
            recs = []
            for i, r in enumerate(rows):
                pre = prefixes[r["prefix_idx"]]
                ida, idb = pre + [r["id_a"]], pre + [r["id_b"]]
                fold = prefix_of[i] % 2
                G = gsets[fold]
                rec = dict(key=r["key"], prefix_idx=r["prefix_idx"], fold=fold, jsd=r["jsd"],
                           overlap=len(set(G) & set(ctrl[i])) / k)
                for cond, heads in (("base", []), ("ctrl", sorted(ctrl[i])), ("glob", G)):
                    abl.set([(g // nh, g % nh) for g in heads])
                    d = run_at(m, ida, idb, alphas, L)
                    rec[f"wtv_{cond}"] = tv_width(alphas, d)
                    rec[f"err_{cond}"] = float(max(abs(d[0]), abs(d[-1] - 1)))
                    abl.remove()
                recs.append(rec)
                if i % 20 == 0:
                    print(f"    L{L} pair {i}/{len(rows)}", flush=True)
            s = stats(recs, rng_boot)
            s["f"] = (n_block - 1 - L) / (n_block - 1)
            out.setdefault("sites", {})[str(L)] = dict(stats=s, rows=recs)
            print(L, json.dumps(s), flush=True)
            json.dump(out, open(path, "w"), indent=1)

    del m
    torch.cuda.empty_cache()
    if MKEY == "gpt2-large":
        figure(out)
    return out


def figure(out):
    sites = sorted((int(L) for L in out["sites"]), reverse=False)
    st = [out["sites"][str(L)]["stats"] for L in sites]
    f = np.array([s["f"] for s in st])

    fig, ax = plt.subplots(1, 3, figsize=(15.6, 4.7))

    style = [("base", "no ablation", "0.55", "o", "-"),
             ("ctrl", "matched control set", CVD[0], "s", "--"),
             ("glob", "fixed cross-pair set (held out)", CVD[3], "^", ":")]
    for kk, lab, col, mk, ls in style:
        y = [s["median_wtv"][kk] for s in st]
        ax[0].plot(f, y, color=col, marker=mk, ls=ls, lw=2.2, ms=8, mec="k", mew=0.6, label=lab)
    ax[0].axhline(0.5, color="0.35", ls=(0, (4, 3)), lw=1.4)
    ax[0].set_ylim(0, 0.62)
    ax[0].set_ylabel("median transition width $w_{TV}$")
    ax[0].set_title("A. The switch and the ablations, site by site\n"
                    "(dashed = linear response 0.5)", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower left")

    y = np.array([s["median_delta"] for s in st])
    ci = np.array([s["ci_delta"] for s in st]).T
    ax[1].plot(f, y, color=CVD[1], marker="D", ls="-", lw=2.4, ms=8, mec="k", mew=0.6)
    ax[1].errorbar(f, y, yerr=np.abs(ci - y), color=CVD[1], capsize=4, lw=1.6, ls="")
    ax[1].axhline(0.0, color="0.35", ls=(0, (4, 3)), lw=1.4)
    ax[1].set_ylabel("paired median $w_{TV}$(fixed set) $-$ $w_{TV}$(control)")
    ax[1].set_title("B. The causal effect of the head circuit\n"
                    "(bars = 95% cluster bootstrap over prefixes)", fontsize=10)

    nd = [100 * s["norm_delta"] if s["norm_delta"] is not None else np.nan for s in st]
    ax[2].bar(np.arange(len(sites)), nd, 0.6, color=CVD[2], hatch="//", edgecolor="k", lw=0.7)
    for i, v in enumerate(nd):
        if np.isnan(v):
            ax[2].text(i, 2, "undefined\n(no headroom)", ha="center", va="bottom", fontsize=7.5,
                       color="0.3", rotation=90)
    ax[2].axhline(0.0, color="0.35", lw=1.0)
    ax[2].set_xticks(np.arange(len(sites)))
    ax[2].set_xticklabels([f"block {L}\n$f$={s['f']:.2f}" for L, s in zip(sites, st)], fontsize=8.5)
    ax[2].set_ylabel("headroom-normalised effect $\\hat\\Delta$ (%)")
    ax[2].set_title("C. Share of the available compression the heads supply\n"
                    "($\\hat\\Delta=\\Delta/(0.5-\\tilde w_{TV}$(control)$)$)", fontsize=10)

    for a in ax[:2]:
        a.set_xlim(1.06, 0.42)
        a.set_xlabel("fraction of the stack below the patch, $f = (N-1-L)/(N-1)$")
        a.grid(alpha=0.3)
    ax[2].grid(alpha=0.3, axis="y")
    fig.suptitle(f"gpt2-large, {out['meta']['n_pairs']} low-JSD pairs, one fixed 22-head set at "
                 f"every patch site", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(PLOTS, "depth_curve.png"), dpi=140)
    plt.close(fig)
    print("wrote plots/depth_curve.png")


if __name__ == "__main__":
    main()
