"""S11: is the differential head set a named circuit, and does the effect track depth or family?

Experiment 7 ablated a *per-pair* top-k head set, so the causal claim is about a construct
(differential engagement), not about a fixed circuit. This script asks three things.

  A. Recurrence - do the same heads keep coming top-k across pairs? Measured as the pairwise
     Jaccard overlap of per-pair sets (within-prefix and, the meaningful one, across-prefix),
     against a random-set null and against the top-k-by-magnitude set (which recurs trivially).

  B. Localisation - split the bank by prefix parity, rank heads by how often they are selected in
     one fold, and ablate that single FIXED set on the held-out fold. If a fixed set reproduces the
     per-pair effect, the differential machinery is a circuit that can be named once; if it does
     not, the effect is genuinely pair-specific.

  C. Model scope - run the Experiment 7 dose sweep in gpt2-small, the third GPT-2-family member, so
     the gpt2-large/gpt2-medium gap can be read against model size within one family.

Writes results/localize_heads.json, results/ablate_gpt2-small.json, plots/localization.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon

from ablate_heads import (Ablation, DOSES, N_MEAN, head_means, one_dose, run_condition,
                          select_heads)
from analyze import tv_width
from analyze_bank import cluster_boot
from circuit_features import probe
from common import PLOTS, RESULTS, load
from mine_lowjsd import N_PREFIX, SEED, get_prefixes

MODELS = ["gpt2-large", "gpt2-medium"]
DOSE = 0.03            # localisation is run at the smallest pre-specified dose
N_JPAIR = 4000         # random pair-of-pairs samples for the Jaccard estimates
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
MSTYLE = {"gpt2-small": dict(color=CVD[0], marker="o", ls="-"),
          "gpt2-medium": dict(color=CVD[1], marker="s", ls="--"),
          "gpt2-large": dict(color=CVD[2], marker="^", ls=":")}


def jaccard(sets, prefix_of, rng, same_prefix):
    """Mean Jaccard overlap over random pairs of pairs, split by shared/different prefix."""
    n = len(sets)
    out = []
    for _ in range(N_JPAIR * (4 if same_prefix else 1)):
        i, j = rng.integers(0, n, 2)
        if i == j or (prefix_of[i] == prefix_of[j]) != same_prefix:
            continue
        a, b = sets[i], sets[j]
        out.append(len(a & b) / len(a | b))
        if len(out) >= N_JPAIR:
            break
    return float(np.mean(out)) if out else float("nan")


def recurrence(model_sets, prefix_of, k, n_heads, rng):
    """Selection frequency, overlap statistics and the random-set null."""
    freq = np.zeros(n_heads)
    for s in model_sets["diff"]:
        for h in s:
            freq[h] += 1
    freq /= len(model_sets["diff"])
    rand_sets = [set(rng.choice(n_heads, k, replace=False).tolist())
                 for _ in range(len(model_sets["diff"]))]
    return dict(
        freq=freq.tolist(),
        jaccard={f"{nm}_{sp}": jaccard(model_sets[nm] if nm != "rand" else rand_sets,
                                       prefix_of, rng, sp == "within")
                 for nm in ("diff", "mag", "rand") for sp in ("within", "across")},
        top_heads=[[int(i), float(freq[i])] for i in np.argsort(-freq)[:10]],
        freq_topk_mass=float(np.sort(freq)[::-1][:k].sum() / freq.sum()))


def collect_sets(m, rows, prefixes, k, nh):
    """Per-pair differential and magnitude head sets (flat head indices)."""
    diff_sets, mag_sets, prefix_of = [], [], []
    for n, r in enumerate(rows):
        pre = prefixes[r["prefix_idx"]]
        pa = probe(m, pre + [r["id_a"]])
        pb = probe(m, pre + [r["id_b"]])
        d_idx, _, delta, mag = select_heads(pa["contrib"], pb["contrib"], k)
        diff_sets.append({b * nh + h for b, h in d_idx})
        mag_sets.append(set(torch.topk(mag, k).indices.tolist()))
        prefix_of.append(r["prefix_idx"])
        if n % 100 == 0:
            print(f"    probe {n}/{len(rows)}", flush=True)
    return dict(diff=diff_sets, mag=mag_sets), prefix_of


def heldout_global(m, rows, prefixes, sets, prefix_of, means, dh, nh, k, alphas):
    """Fold-wise: rank heads on one half of the prefixes, ablate that fixed set on the other."""
    folds = {0: [i for i, p in enumerate(prefix_of) if p % 2 == 0],
             1: [i for i, p in enumerate(prefix_of) if p % 2 == 1]}
    recs, gsets = [], {}
    with Ablation(m, means, dh) as abl:
        for f in (0, 1):
            train, test = folds[1 - f], folds[f]
            freq = np.zeros(nh * len(m.transformer.h))
            for i in train:
                for h in sets["diff"][i]:
                    freq[h] += 1
            G = [int(i) for i in np.argsort(-freq)[:k]]
            gsets[f] = G
            abl.set([(g // nh, g % nh) for g in G])
            for n, i in enumerate(test):
                r = rows[i]
                pre = prefixes[r["prefix_idx"]]
                d = run_condition(m, pre + [r["id_a"]], pre + [r["id_b"]], alphas)
                recs.append(dict(key=r["key"], prefix_idx=r["prefix_idx"], fold=f,
                                 wtv_glob=tv_width(alphas, d),
                                 overlap=len(set(G) & sets["diff"][i]) / k,
                                 err=float(max(abs(d[0]), abs(d[-1] - 1)))))
                if n % 100 == 0:
                    print(f"    fold {f} pair {n}/{len(test)}", flush=True)
            abl.remove()
    return recs, gsets


def join_stats(recs, abl_rows, rng_boot):
    """Merge the held-out global condition with the stored base/ctrl/diff rows for those pairs."""
    by_key = {r["key"]: r for r in abl_rows}
    rows = [dict(g, **{c: by_key[g["key"]][f"wtv_{c}"] for c in ("base", "ctrl", "diff")})
            for g in recs if g["key"] in by_key]
    cl = {}
    for r in rows:
        cl.setdefault(r["prefix_idx"], []).append(r)
    med = lambda c: float(np.median([r[c] for r in rows]))
    delta = lambda c: float(np.median([r[c] - r["base"] for r in rows]))
    ci = lambda c: list(cluster_boot(
        cl, lambda rr, c=c: float(np.median([q[c] - q["base"] for q in rr])), rng_boot))
    stats = dict(
        n=len(rows), n_prefix=len(cl),
        median_wtv={c: med(c) for c in ("base", "ctrl", "diff", "wtv_glob")},
        delta_vs_base={c: delta(c) for c in ("ctrl", "diff", "wtv_glob")},
        ci_vs_base={c: ci(c) for c in ("ctrl", "diff", "wtv_glob")},
        p_glob_vs_ctrl=float(wilcoxon([r["wtv_glob"] for r in rows], [r["ctrl"] for r in rows])[1]),
        p_glob_vs_diff=float(wilcoxon([r["wtv_glob"] for r in rows], [r["diff"] for r in rows])[1]),
        mean_overlap=float(np.mean([r["overlap"] for r in rows])),
        max_endpoint_err=float(max(r["err"] for r in rows)))
    stats["recovery"] = stats["delta_vs_base"]["wtv_glob"] / stats["delta_vs_base"]["diff"]
    return stats, rows


def main():
    alphas = np.linspace(0, 1, 101)
    rng, rng_boot = np.random.default_rng(0), np.random.default_rng(0)
    abl = json.load(open(os.path.join(RESULTS, "ablate_heads.json")))
    out = {}

    for mkey in MODELS:
        rows = json.load(open(os.path.join(RESULTS, f"lowjsd_{mkey}.json")))
        rows = rows[:int(os.environ.get("LIMIT", len(rows)))]
        tok, m = load(mkey)
        prefixes = get_prefixes(tok, np.random.default_rng(SEED), N_PREFIX)
        nh, n_block = m.config.n_head, len(m.transformer.h)
        dh = m.config.n_embd // nh
        n_heads = nh * n_block
        k = max(4, int(round(DOSE * n_heads)))
        print(f"{mkey}: k={k} of {n_heads} heads, {len(rows)} pairs", flush=True)

        sets, prefix_of = collect_sets(m, rows, prefixes, k, nh)
        rec = recurrence(sets, prefix_of, k, n_heads, rng)
        rec["layer_mass"] = np.bincount(
            [h // nh for s in sets["diff"] for h in s], minlength=n_block).astype(float).tolist()
        rec["n_block"], rec["nh"], rec["k"], rec["n_heads"] = n_block, nh, k, n_heads

        seen, mean_prompts = set(), []
        for r in rows:
            if r["prefix_idx"] not in seen and len(mean_prompts) < N_MEAN:
                seen.add(r["prefix_idx"])
                mean_prompts.append(prefixes[r["prefix_idx"]] + [r["id_a"]])
        means = head_means(m, mean_prompts)

        recs, gsets = heldout_global(m, rows, prefixes, sets, prefix_of, means, dh, nh, k, alphas)
        stats, joined = join_stats(recs, abl[mkey]["doses"][f"{DOSE:.2f}"]["rows"], rng_boot)
        out[mkey] = dict(recurrence=rec, heldout=stats, global_sets={str(f): g
                                                                    for f, g in gsets.items()},
                         rows=joined)
        print(mkey, json.dumps({kk: vv for kk, vv in stats.items() if kk != "rows"}), flush=True)
        with open(os.path.join(RESULTS, "localize_heads.json"), "w") as f:
            json.dump(out, f, indent=1)
        del m
        torch.cuda.empty_cache()

    # C: the third family member, same pre-specified dose sweep as Experiment 7.
    small = small_sweep(alphas, rng_boot)
    figure(out, abl, small)
    return out


def small_sweep(alphas, rng_boot, mkey="gpt2-small"):
    path = os.path.join(RESULTS, f"ablate_{mkey}.json")
    if os.path.exists(path):
        return json.load(open(path))
    rows = json.load(open(os.path.join(RESULTS, f"lowjsd_{mkey}.json")))
    rows = rows[:int(os.environ.get("LIMIT", len(rows)))]
    tok, m = load(mkey)
    prefixes = get_prefixes(tok, np.random.default_rng(SEED), N_PREFIX)
    nh = m.config.n_head
    dh = m.config.n_embd // nh
    n_heads = nh * len(m.transformer.h)
    seen, mean_prompts = set(), []
    for r in rows:
        if r["prefix_idx"] not in seen and len(mean_prompts) < N_MEAN:
            seen.add(r["prefix_idx"])
            mean_prompts.append(prefixes[r["prefix_idx"]] + [r["id_a"]])
    means = head_means(m, mean_prompts)
    out = dict(n_heads=n_heads, doses={})
    for frac in DOSES:
        k = max(4, int(round(frac * n_heads)))
        print(f"{mkey}: dose {frac:.0%} -> k={k} of {n_heads} heads, {len(rows)} pairs", flush=True)
        info, recs = one_dose(m, rows, prefixes, means, dh, nh, k, alphas, rng_boot)
        info["frac"], info["n_heads"] = frac, n_heads
        out["doses"][f"{frac:.2f}"] = dict(info=info, rows=recs)
        print(mkey, json.dumps(info), flush=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
    del m
    torch.cuda.empty_cache()
    return out


BARS = [("base", "no ablation", "0.55", ""),
        ("ctrl", "matched control set", CVD[0], ".."),
        ("wtv_glob", "fixed cross-pair set (held out)", CVD[3], "//"),
        ("diff", "per-pair differential set", CVD[1], "\\\\")]


def figure(out, abl, small, extra=None):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.0))
    ax = axes.ravel()

    shown = [k for k in ("gpt2-small", "gpt2-medium", "gpt2-large") if k in out]
    a = ax[0]
    for mkey in shown:
        r = out[mkey]["recurrence"]
        f = np.sort(np.array(r["freq"]))[::-1]
        a.plot(np.arange(1, len(f) + 1), f, lw=2.2, label=f"{mkey} (k={r['k']})", **MSTYLE[mkey],
               markevery=max(1, len(f) // 12), ms=7, mec="k", mew=0.6)
        a.axhline(r["k"] / r["n_heads"], color=MSTYLE[mkey]["color"], lw=1.2, ls=(0, (1, 3)))
    a.set_xscale("log")
    a.set_xlabel("head rank (sorted by selection frequency)")
    a.set_ylabel("fraction of pairs selecting this head")
    a.set_title("A. Do the same heads recur across pairs?\n"
                "(dotted horizontal = random-selection rate $k/H$)", fontsize=10)
    a.legend(fontsize=8.5)
    a.grid(alpha=0.3)

    b = ax[1]
    for mkey in shown:
        r = out[mkey]["recurrence"]
        lm = np.array(r["layer_mass"])
        b.plot(np.arange(r["n_block"]) / (r["n_block"] - 1), lm / lm.sum(), lw=2.2,
               label=f"{mkey} (block 0: {100 * lm[0] / lm.sum():.0f}%)", **MSTYLE[mkey], ms=7,
               mec="k", mew=0.6)
    b.set_xlabel("block index / (blocks $-$ 1)   (relative depth)")
    b.set_ylabel("share of selected heads")
    b.set_title("B. Where in the stack the differential heads sit", fontsize=10)
    b.legend(fontsize=8.5)
    b.grid(alpha=0.3)

    c = ax[2]
    bars = BARS + ([("noL0", "fixed set, block-0 heads excluded", CVD[2], "xx")] if extra else [])
    w = 0.8 / len(bars)
    for i, (kk, lab, col, hat) in enumerate(bars):
        vals = [(extra or {}).get(mk, np.nan) if kk == "noL0"
                else out[mk]["heldout"]["median_wtv"][kk] for mk in MODELS]
        c.bar(np.arange(len(MODELS)) + (i - (len(bars) - 1) / 2) * w, vals, w, color=col,
              hatch=hat, edgecolor="k", lw=0.7, label=lab)
    c.axhline(0.5, color="0.35", ls=(0, (4, 3)), lw=1.4)
    c.set_xticks(np.arange(len(MODELS)))
    c.set_xticklabels([f"{mk}\n(n={out[mk]['heldout']['n']})" for mk in MODELS])
    c.set_ylabel("median transition width $w_{TV}$")
    c.set_title("C. A single fixed head set, ablated on held-out pairs\n"
                "(dashed = linear response 0.5)", fontsize=10)
    c.legend(fontsize=8, loc="upper left")
    c.grid(alpha=0.3, axis="y")

    d = ax[3]
    src = {"gpt2-small": small, "gpt2-medium": abl["gpt2-medium"], "gpt2-large": abl["gpt2-large"]}
    for mkey, s in src.items():
        doses = s["doses"]
        x = np.array([doses[k]["info"]["frac"] for k in sorted(doses)]) * 100
        y = np.array([doses[k]["info"]["median_delta"] for k in sorted(doses)])
        ci = np.array([doses[k]["info"]["ci_delta"] for k in sorted(doses)]).T
        d.errorbar(x, y, yerr=np.abs(ci - y), lw=2.2, capsize=4, ms=8, mec="k", mew=0.6,
                   label=f"{mkey} ({len(doses[sorted(doses)[0]]['rows'])} pairs)", **MSTYLE[mkey])
    d.axhline(0.0, color="0.35", ls=(0, (4, 3)), lw=1.4)
    d.set_yscale("symlog", linthresh=0.01)
    d.set_xticks(x)
    d.set_xlabel("heads mean-ablated (% of all heads)")
    d.set_ylabel("paired median $w_{TV}$(differential) $-$ $w_{TV}$(control)")
    d.set_title("D. Same intervention across three GPT-2 sizes\n"
                "(symlog y; bars = 95% cluster bootstrap over prefixes)", fontsize=10)
    d.legend(fontsize=8)
    d.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "localization.png"), dpi=140)
    plt.close(fig)
    print("wrote plots/localization.png")


if __name__ == "__main__":
    main()
