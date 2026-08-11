"""S12: does the cross-model gap in the ablation effect close when relative depth is matched?

Experiment 8 ablated a held-out fixed head set with the interpolation patched at block 0, and the
effect was ~10x larger in gpt2-large than in gpt2-medium. Neither model size nor the block-0 share
of the selected heads predicts that ordering. Experiment 5's organising variable is *relative*
depth f = (N-1-L)/(N-1), the fraction of the stack below the patch; at block 0 the three GPT-2
models sit at f = 1 but with very different numbers of blocks below.

This script repeats the identical held-out protocol with the patch moved to the middle block, where
f ~ 0.5 in all three models, and adds the missing block-0 fixed-set run for gpt2-small so the
comparison is 3 models x 2 patch sites.

Conditions per pair (all re-run inside the condition, so d(0)=0 and d(1)=1 hold within it):
  base - no ablation
  ctrl - the per-pair engagement-matched control set (Experiment 7's control)
  glob - a single fixed head set, ranked by selection frequency on the other half of the prefixes

Writes results/depth_gap.json and plots/depth_gap.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon

from ablate_heads import Ablation, N_MEAN, head_means, select_heads
from analyze import tv_width
from analyze_bank import cluster_boot
from circuit_features import probe
from common import PLOTS, RESULTS, load
from mine_lowjsd import N_PREFIX, SEED, get_prefixes
from run_interp import clean_run, rel_dist, slerp_lerp_norm, sweep

MODELS = ["gpt2-small", "gpt2-medium", "gpt2-large"]
# Middle block of each stack: f = (N-1-L)/(N-1) = 0.455 / 0.478 / 0.486, matched to within 0.03.
MID = {"gpt2-small": 6, "gpt2-medium": 12, "gpt2-large": 18}
DOSE = 0.03
N_ALPHA = 101
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
MSTYLE = {"gpt2-small": dict(color=CVD[0], marker="o", ls="-"),
          "gpt2-medium": dict(color=CVD[1], marker="s", ls="--"),
          "gpt2-large": dict(color=CVD[2], marker="^", ls=":")}


def run_at(m, ids_a, ids_b, alphas, layer):
    """Endpoints + 101-point sweep patched at `layer`, under whatever hooks are installed."""
    reca, lga = clean_run(m, ids_a)
    recb, lgb = clean_run(m, ids_b)
    vecs, _, _ = slerp_lerp_norm(reca[layer], recb[layer], alphas)
    _, lgs = sweep(m, ids_a, vecs, layer=layer)
    return rel_dist(lgs, lga.cpu().unsqueeze(0), lgb.cpu().unsqueeze(0))


def per_pair_sets(m, rows, prefixes, k, nh):
    """Per-pair differential and engagement-matched control sets (flat head indices)."""
    diff, ctrl, prefix_of = [], [], []
    for n, r in enumerate(rows):
        pre = prefixes[r["prefix_idx"]]
        pa = probe(m, pre + [r["id_a"]])
        pb = probe(m, pre + [r["id_b"]])
        d_idx, c_idx, _, _ = select_heads(pa["contrib"], pb["contrib"], k)
        diff.append({b * nh + h for b, h in d_idx})
        ctrl.append({b * nh + h for b, h in c_idx})
        prefix_of.append(r["prefix_idx"])
        if n % 100 == 0:
            print(f"    probe {n}/{len(rows)}", flush=True)
    return diff, ctrl, prefix_of


def heldout(m, rows, prefixes, diff, ctrl, prefix_of, means, dh, nh, k, alphas, layer, conds):
    """Rank heads on one prefix fold, ablate that fixed set on the other; plus base and ctrl."""
    folds = {0: [i for i, p in enumerate(prefix_of) if p % 2 == 0],
             1: [i for i, p in enumerate(prefix_of) if p % 2 == 1]}
    n_block = len(m.transformer.h)
    recs, gsets = {}, {}
    with Ablation(m, means, dh) as abl:
        for f in (0, 1):
            train, test = folds[1 - f], folds[f]
            freq = np.zeros(nh * n_block)
            for i in train:
                for h in diff[i]:
                    freq[h] += 1
            G = [int(i) for i in np.argsort(-freq)[:k]]
            gsets[f] = G
            for n, i in enumerate(test):
                r = rows[i]
                pre = prefixes[r["prefix_idx"]]
                ida, idb = pre + [r["id_a"]], pre + [r["id_b"]]
                rec = recs.setdefault(r["key"], dict(key=r["key"], prefix_idx=r["prefix_idx"],
                                                     fold=f, jsd=r["jsd"]))
                rec["overlap"] = len(set(G) & diff[i]) / k
                for cond in conds:
                    heads = {"base": [], "ctrl": sorted(ctrl[i]), "glob": G}[cond]
                    abl.set([(g // nh, g % nh) for g in heads])
                    d = run_at(m, ida, idb, alphas, layer)
                    rec[f"wtv_{cond}"] = tv_width(alphas, d)
                    rec[f"err_{cond}"] = float(max(abs(d[0]), abs(d[-1] - 1)))
                    abl.remove()
                if n % 50 == 0:
                    print(f"    L{layer} fold {f} pair {n}/{len(test)}", flush=True)
    below = lambda G: float(np.mean([g // nh <= layer for g in G]))
    return list(recs.values()), {str(f): g for f, g in gsets.items()}, \
        float(np.mean([below(g) for g in gsets.values()]))


def stats(rows, rng_boot):
    """Median w_TV per condition and the paired fixed-set-minus-control effect, clustered."""
    cl = {}
    for r in rows:
        cl.setdefault(r["prefix_idx"], []).append(r)
    med = lambda c: float(np.median([r[f"wtv_{c}"] for r in rows]))
    g = np.array([r["wtv_glob"] for r in rows])
    c = np.array([r["wtv_ctrl"] for r in rows])
    lo, hi = cluster_boot(
        cl, lambda rr: float(np.median([q["wtv_glob"] - q["wtv_ctrl"] for q in rr])), rng_boot)
    return dict(
        n=len(rows), n_prefix=len(cl),
        median_wtv={c_: med(c_) for c_ in ("base", "ctrl", "glob")},
        median_delta=float(np.median(g - c)), ci_delta=[lo, hi],
        wilcoxon_p=float(wilcoxon(g, c)[1]), frac_glob_gt_ctrl=float(np.mean(g > c)),
        mean_overlap=float(np.mean([r["overlap"] for r in rows])),
        max_endpoint_err=float(max(max(r[f"err_{c_}"] for c_ in ("base", "ctrl", "glob"))
                                   for r in rows)))


def block0_small(m, rows, prefixes, diff, ctrl, prefix_of, means, dh, nh, k, alphas, rng_boot):
    """The missing block-0 fixed-set run for gpt2-small; base/ctrl come from the stored sweep."""
    recs, gsets, share = heldout(m, rows, prefixes, diff, ctrl, prefix_of, means, dh, nh, k,
                                 alphas, 0, ["glob"])
    stored = json.load(open(os.path.join(RESULTS, "ablate_gpt2-small.json")))
    by_key = {r["key"]: r for r in stored["doses"][f"{DOSE:.2f}"]["rows"]}
    merged = [dict(r, wtv_base=by_key[r["key"]]["wtv_base"], wtv_ctrl=by_key[r["key"]]["wtv_ctrl"],
                   err_base=0.0, err_ctrl=by_key[r["key"]]["err_ctrl"])
              for r in recs if r["key"] in by_key]
    s = stats(merged, rng_boot)
    s["below_share"] = share
    return s, merged, gsets


def main():
    alphas = np.linspace(0, 1, N_ALPHA)
    rng_boot = np.random.default_rng(0)
    path = os.path.join(RESULTS, "depth_gap.json")
    out = json.load(open(path)) if os.path.exists(path) else {}

    for mkey in MODELS:
        rows = json.load(open(os.path.join(RESULTS, f"lowjsd_{mkey}.json")))
        rows = rows[:int(os.environ.get("LIMIT", len(rows)))]
        tok, m = load(mkey)
        prefixes = get_prefixes(tok, np.random.default_rng(SEED), N_PREFIX)
        nh, n_block = m.config.n_head, len(m.transformer.h)
        dh = m.config.n_embd // nh
        k = max(4, int(round(DOSE * nh * n_block)))
        L = MID[mkey]
        print(f"{mkey}: k={k} of {nh * n_block} heads, {len(rows)} pairs, mid block {L} "
              f"(f={(n_block - 1 - L) / (n_block - 1):.3f})", flush=True)

        diff, ctrl, prefix_of = per_pair_sets(m, rows, prefixes, k, nh)
        seen, mean_prompts = set(), []
        for r in rows:
            if r["prefix_idx"] not in seen and len(mean_prompts) < N_MEAN:
                seen.add(r["prefix_idx"])
                mean_prompts.append(prefixes[r["prefix_idx"]] + [r["id_a"]])
        means = head_means(m, mean_prompts)

        entry = out.setdefault(mkey, {})
        entry.update(k=k, n_heads=nh * n_block, n_block=n_block, mid_block=L,
                     f_mid=(n_block - 1 - L) / (n_block - 1))
        if mkey == "gpt2-small" and "block0" not in entry:
            s, merged, gs = block0_small(m, rows, prefixes, diff, ctrl, prefix_of, means, dh, nh,
                                         k, alphas, rng_boot)
            entry["block0"] = dict(stats=s, sets=gs, rows=merged)
            print(mkey, "block0", json.dumps(s), flush=True)
            json.dump(out, open(path, "w"), indent=1)

        if "mid" not in entry:
            recs, gs, share = heldout(m, rows, prefixes, diff, ctrl, prefix_of, means, dh, nh, k,
                                      alphas, L, ["base", "ctrl", "glob"])
            s = stats(recs, rng_boot)
            s["below_share"] = share
            entry["mid"] = dict(stats=s, sets=gs, rows=recs)
            print(mkey, "mid", json.dumps(s), flush=True)
            json.dump(out, open(path, "w"), indent=1)

        del m
        torch.cuda.empty_cache()

    figure(out)
    return out


def block0_stats(out):
    """The block-0 held-out fixed-set run for each model, scored exactly like the mid-patch one.

    gpt2-large and gpt2-medium were run in Experiment 8 (localize_heads.json, whose merged rows keep
    the base/ctrl conditions under different key names); gpt2-small is run here.
    """
    lh = json.load(open(os.path.join(RESULTS, "localize_heads.json")))
    rng_boot = np.random.default_rng(0)
    b0 = {}
    for mkey in MODELS:
        if mkey in lh and "rows" in lh[mkey]:
            rows = [dict(r, wtv_base=r["base"], wtv_ctrl=r["ctrl"],
                         err_base=0.0, err_ctrl=0.0, err_glob=r["err"]) for r in lh[mkey]["rows"]]
            sets = lh[mkey]["global_sets"]
        elif mkey in out and "block0" in out[mkey]:
            rows, sets = out[mkey]["block0"]["rows"], out[mkey]["block0"]["sets"]
        else:
            continue
        b0[mkey] = stats(rows, rng_boot)
        nh = out[mkey]["n_heads"] // out[mkey]["n_block"]
        b0[mkey]["below_share"] = float(np.mean([np.mean([g // nh == 0 for g in G])
                                                 for G in sets.values()]))
    return b0


def figure(out):
    """Three panels: the switch under each condition, the paired effect, and how it moves."""
    b0 = block0_stats(out)
    shown = [k for k in MODELS if k in out and "mid" in out[k]]

    fig, ax = plt.subplots(1, 3, figsize=(15.6, 4.9))
    bars = [("base", "no ablation", "0.55", ""),
            ("ctrl", "matched control set", CVD[0], ".."),
            ("glob", "fixed cross-pair set (held out)", CVD[3], "//")]
    x = np.arange(len(shown))
    for j, (site, src, title) in enumerate(
            [("block 0", b0, "A. Patch at block 0 ($f = 1$)"),
             ("mid", {k: out[k]["mid"]["stats"] for k in shown},
              "B. Patch at the middle block ($f \\approx 0.47$)")]):
        w = 0.8 / len(bars)
        for i, (kk, lab, col, hat) in enumerate(bars):
            vals = [src[mk]["median_wtv"][kk] if mk in src else np.nan for mk in shown]
            ax[j].bar(x + (i - (len(bars) - 1) / 2) * w, vals, w, color=col, hatch=hat,
                      edgecolor="k", lw=0.7, label=lab)
        ax[j].axhline(0.5, color="0.35", ls=(0, (4, 3)), lw=1.4)
        ax[j].set_xticks(x)
        ax[j].set_xticklabels([f"{mk}\n({out[mk]['n_block']} blocks)" for mk in shown], fontsize=9)
        ax[j].set_ylabel("median transition width $w_{TV}$")
        ax[j].set_ylim(0, 0.62)
        ax[j].set_title(title + "\n(dashed = linear response 0.5)", fontsize=10)
        ax[j].legend(fontsize=8, loc="upper left")
        ax[j].grid(alpha=0.3, axis="y")

    c = ax[2]
    for mk in shown:
        st = [b0[mk], out[mk]["mid"]["stats"]]
        xs = [1.0, out[mk]["f_mid"]]
        y = np.array([s["median_delta"] for s in st])
        ci = np.array([s["ci_delta"] for s in st]).T
        c.plot(xs, y, lw=2.4, ms=9, mec="k", mew=0.6, label=mk, **MSTYLE[mk])
        c.errorbar(xs, y, yerr=np.abs(ci - y), color=MSTYLE[mk]["color"], capsize=4, lw=1.6, ls="")
    c.axhline(0.0, color="0.35", ls=(0, (4, 3)), lw=1.4)
    c.set_yscale("symlog", linthresh=0.01)
    c.set_xlim(1.06, 0.36)
    c.set_xlabel("fraction of the stack below the patch, $f = (N-1-L)/(N-1)$")
    c.set_ylabel("paired median $w_{TV}$(fixed set) $-$ $w_{TV}$(control)")
    c.set_title("C. The head circuit only has an effect where the plateau is\n"
                "(symlog y; bars = 95% cluster bootstrap over prefixes)", fontsize=10)
    c.legend(fontsize=8.5)
    c.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "depth_gap.png"), dpi=140)
    plt.close(fig)
    print("wrote plots/depth_gap.png")


if __name__ == "__main__":
    main()
