"""S4: powered test of the JSD-vs-sharpness association on the mined pair bank.

Replaces the 5-point scatter with a regression over ~200 corpus-derived pairs per model.
Confidence intervals use a cluster bootstrap over prefixes, since the pairs sharing a prefix
are not independent.

Writes results/bank_analysis.json, plots/bank_prevalence.png, plots/bank_regression.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import PAIRS, PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
MODELS = ["gpt2-medium", "pythia-410m", "opt-350m"]
MK = ["o", "s", "^", "D", "v"]
NBOOT = 2000
LIN = {"w": 0.8, "wtv": 0.5}          # linear-response value of each statistic
THR = {"w": 0.5, "wtv": 0.25}         # plateau threshold


def cluster_boot(clusters, fn, rng):
    """Bootstrap fn over resampled prefix clusters; returns (lo, hi) 95% interval."""
    ks = list(clusters)
    vals = []
    for _ in range(NBOOT):
        pick = rng.choice(len(ks), size=len(ks), replace=True)
        rows = [r for i in pick for r in clusters[ks[i]]]
        v = fn(rows)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rho_of(rows, stat):
    x = [r["jsd"] for r in rows if r[stat] is not None]
    y = [r[stat] for r in rows if r[stat] is not None]
    return None if len(set(x)) < 3 else float(spearmanr(x, y)[0])


def slope_of(rows, stat):
    x = np.array([r["jsd"] for r in rows if r[stat] is not None])
    y = np.array([r[stat] for r in rows if r[stat] is not None])
    return None if len(x) < 3 else float(np.polyfit(x, y, 1)[0])


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after linearly removing z, all on ranks."""
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    rx, ry, rz = rk(x), rk(y), rk(z)
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)
    return float(spearmanr(res(rx), res(ry))[0])


def binned(rows, stat, nb=5):
    """Equal-count JSD bins: (bin centre, mean stat, SE)."""
    ok = [r for r in rows if r[stat] is not None]
    ok.sort(key=lambda r: r["jsd"])
    out = []
    for c in np.array_split(np.arange(len(ok)), nb):
        js = [ok[i]["jsd"] for i in c]
        ys = np.array([ok[i][stat] for i in c])
        out.append((float(np.mean(js)), float(ys.mean()),
                    float(ys.std(ddof=1) / np.sqrt(len(ys)))))
    return out


JSD_BINS = [0.0, 0.2, 0.4, 0.65, np.log(2) + 1e-9]


def jsd_matched(rows):
    """Median w_TV and % sharp within fixed JSD bins, so models are compared at equal divergence."""
    out = []
    for lo, hi in zip(JSD_BINS[:-1], JSD_BINS[1:]):
        v = np.array([r["wtv"] for r in rows if lo <= r["jsd"] < hi])
        out.append(dict(lo=lo, hi=hi, n=len(v),
                        median_wtv=None if not len(v) else float(np.median(v)),
                        frac_sharp=None if not len(v) else float(np.mean(v < THR["wtv"]))))
    return out


def main():
    rng = np.random.default_rng(0)
    bank = {m: json.load(open(os.path.join(RESULTS, f"bank_{m}.json"))) for m in MODELS}
    hand = json.load(open(os.path.join(RESULTS, "analysis.json")))["rows"]
    labels = {p[0]: p[1] for p in PAIRS}
    keys = [p[0] for p in PAIRS]

    stats = {}
    for m in MODELS:
        rows = bank[m]
        clusters = {}
        for r in rows:
            clusters.setdefault(r["prefix_idx"], []).append(r)
        ent = dict(n=len(rows), n_prefix=len(clusters),
                   n_missing_w=sum(r["w"] is None for r in rows),
                   jsd_min=min(r["jsd"] for r in rows), jsd_max=max(r["jsd"] for r in rows),
                   jsd_med=float(np.median([r["jsd"] for r in rows])),
                   max_endpoint_err=max(max(r["endpoint_err"]) for r in rows),
                   frac_sharp_wtv=float(np.mean([r["wtv"] < THR["wtv"] for r in rows])),
                   frac_sharp_w=float(np.mean([r["w"] is not None and r["w"] < THR["w"]
                                               for r in rows])),
                   median_wtv=float(np.median([r["wtv"] for r in rows])),
                   median_w=float(np.median([r["w"] for r in rows if r["w"] is not None])),
                   frac_mono=float(np.mean([r["mono"] for r in rows])))
        for st in ["w", "wtv", "pf"]:
            x = [r["jsd"] for r in rows if r[st] is not None]
            y = [r[st] for r in rows if r[st] is not None]
            rho, p = spearmanr(x, y)
            lo, hi = cluster_boot(clusters, lambda rr, s=st: rho_of(rr, s), rng)
            ent[st] = dict(rho=float(rho), p=float(p), n=len(x), ci=[lo, hi])
            if st != "pf":
                sl = slope_of(rows, st)
                slo, shi = cluster_boot(clusters, lambda rr, s=st: slope_of(rr, s), rng)
                ent[st]["slope"] = sl
                ent[st]["slope_ci"] = [slo, shi]
                ent[st]["binned"] = binned(rows, st)
        # smallest |rho| a two-sided test at this n could have called significant (Fisher z)
        ent["rho_detectable"] = float(np.tanh(1.96 / np.sqrt(len(rows) - 3)))

        # JSD saturates at ln 2 for disjoint predictions: re-test away from the ceiling.
        unsat = [r for r in rows if r["jsd"] < 0.65]
        ent["unsat"] = dict(n=len(unsat), rho_wtv=rho_of(unsat, "wtv"),
                            p=float(spearmanr([r["jsd"] for r in unsat],
                                              [r["wtv"] for r in unsat])[1]),
                            rho_w=rho_of(unsat, "w"),
                            p_w=float(spearmanr([r["jsd"] for r in unsat],
                                                [r["w"] for r in unsat])[1]))
        # Is the association just endpoint geometry? Partial Spearman controlling for the
        # block-0 angle Omega between the two patched activations.
        ent["omega"] = dict(
            rho_jsd=float(spearmanr([r["jsd"] for r in rows], [r["omega"] for r in rows])[0]),
            rho_wtv=float(spearmanr([r["omega"] for r in rows], [r["wtv"] for r in rows])[0]),
            partial_jsd_wtv=partial_spearman([r["jsd"] for r in rows], [r["wtv"] for r in rows],
                                             [r["omega"] for r in rows]))
        ent["jsd_matched"] = jsd_matched(rows)
        stats[m] = ent

    # ---- Figure: how common is a plateau in the bank? -------------------------------------
    fig, ax = plt.subplots(1, len(MODELS), figsize=(14, 4.4), sharey=True)
    ymax = max(np.histogram([r["wtv"] for r in bank[m]], bins=np.linspace(0, 0.6, 31))[0].max()
               for m in MODELS)
    for c, m in enumerate(MODELS):
        v = np.array([r["wtv"] for r in bank[m]])
        ax[c].hist(v, bins=np.linspace(0, 0.6, 31), color="0.62", edgecolor="k",
                   linewidth=0.4, hatch="//", label=f"mined bank (n={len(v)})")
        ax[c].set_ylim(0, ymax * 1.55)
        ax[c].axvline(LIN["wtv"], color="0.35", ls=(0, (4, 3)), lw=1.6)
        ax[c].text(LIN["wtv"] - 0.008, ymax * 0.55, "linear response (0.5)",
                   rotation=90, va="center", ha="right", fontsize=7, color="0.3")
        ax[c].axvline(THR["wtv"], color="0.1", ls=":", lw=1.6)
        ax[c].text(THR["wtv"] - 0.008, ymax * 0.55, "sharp threshold (0.25)",
                   rotation=90, va="center", ha="right", fontsize=7, color="0.1")
        ax[c].axhline(ymax * 1.06, color="0.75", lw=0.8)
        ax[c].text(0.585, ymax * 1.50, "hand-picked pairs (Table 1)", fontsize=7,
                   color="0.3", ha="right", va="top")
        for i, k in enumerate(keys):
            h = [q for q in hand if q["model"] == m and q["pair"] == k][0]
            ax[c].plot([h["wtv"]], [ymax * (1.13 + 0.07 * i)], marker=MK[i],
                       color=CVD[i], ms=9, mec="k",
                       mew=1.8 if h["control"] else 0.6, ls="none", label=labels[k])
        ax[c].set_title(f"{m}: {stats[m]['frac_sharp_wtv']*100:.0f}% of mined pairs sharp",
                        fontsize=10)
        ax[c].set_xlabel("$w_{\\mathrm{TV}}$ at final logits (smaller = sharper)")
        ax[c].grid(alpha=0.3)
    ax[0].set_ylabel("number of prompt pairs")
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=7.5, ncol=3, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(PLOTS, "bank_prevalence.png"), dpi=140)
    plt.close(fig)

    # ---- Figure: is the cross-model prevalence gap just a difference in JSD spread? --------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = np.arange(len(JSD_BINS) - 1)
    for i, m in enumerate(MODELS):
        b = stats[m]["jsd_matched"]
        ax.plot(xs, [q["median_wtv"] for q in b], color=CVD[i], ls=["-", "--", ":"][i],
                marker=MK[i], ms=9, lw=2, mec="k", mew=0.6, label=m)
    ax.axhline(LIN["wtv"], color="0.35", ls=(0, (4, 3)), lw=1.4)
    ax.text(0.02, LIN["wtv"] + 0.008, "linear response (0.5)", fontsize=7.5, color="0.3")
    ax.axhline(THR["wtv"], color="0.1", ls=":", lw=1.4)
    ax.text(0.02, THR["wtv"] - 0.01, "sharp threshold (0.25)", fontsize=7.5, color="0.1", va="top")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%.2f-%.2f\n(n=%s)" % (
        q["lo"], q["hi"], "/".join(str(stats[m]["jsd_matched"][k]["n"]) for m in MODELS))
        for k, q in enumerate(stats[MODELS[0]]["jsd_matched"])], fontsize=7.5)
    ax.set_xlabel("endpoint JSD bin (nats); n per bin = gpt2-medium / pythia-410m / opt-350m")
    ax.set_ylabel("median $w_{\\mathrm{TV}}$ at final logits")
    ax.set_ylim(0, 0.56)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "jsd_matched.png"), dpi=140)
    plt.close(fig)

    # ---- Figure: the powered association test ---------------------------------------------
    fig, ax = plt.subplots(2, len(MODELS), figsize=(14, 8), sharex=True)
    for r, st in enumerate(["w", "wtv"]):
        for c, m in enumerate(MODELS):
            a = ax[r, c]
            rows = [q for q in bank[m] if q[st] is not None]
            x = np.array([q["jsd"] for q in rows])
            y = np.array([q[st] for q in rows])
            a.scatter(x, y, s=14, color=CVD[3], alpha=0.55, edgecolor="none",
                      label=f"mined pair (n={len(x)})")
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            a.plot(xs, b0 + b1 * xs, color=CVD[0], lw=2, ls="-", label="OLS fit")
            bm = np.array(stats[m][st]["binned"])
            a.errorbar(bm[:, 0], bm[:, 1], yerr=bm[:, 2], color=CVD[1], ls="--", lw=1.8,
                       marker="s", ms=6, capsize=3, label="quintile mean $\\pm$ SE")
            for i, k in enumerate(keys):
                h = [q for q in hand if q["model"] == m and q["pair"] == k][0]
                a.plot([h["jsd"]], [h[st]], marker="*", color=CVD[2],
                       ms=17 if h["control"] else 14, ls="none", mec="k",
                       mew=1.8 if h["control"] else 0.7,
                       label=("control pair" if h["control"] else
                              ("hand-picked pair (Table 1)" if i == 0 else None)))
            a.axvline(np.log(2), color="0.2", ls="-.", lw=1)
            a.text(np.log(2) - 0.008, 0.5, "JSD ceiling ($\\ln 2$)", rotation=90, fontsize=7,
                   color="0.2", ha="right", va="center", transform=a.get_xaxis_transform())
            a.axhline(LIN[st], color="0.55", ls=(0, (4, 3)), lw=1)
            a.text(0.99, LIN[st] * 1.01, "linear response", fontsize=7, color="0.35",
                   ha="right", transform=a.get_yaxis_transform(which="grid"))
            a.axhline(THR[st], color="0.2", ls=":", lw=1)
            s = stats[m][st]
            a.set_title(f"{m} - Spearman $\\rho$={s['rho']:+.2f} "
                        f"[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}], n={s['n']}", fontsize=9)
            a.grid(alpha=0.3)
            a.set_ylim(0, max(LIN[st] * 1.2, float(y.max()) * 1.05))
            if r == 1:
                a.set_xlabel("endpoint JSD (nats)")
        ax[r, 0].set_ylabel("$w_{10\\text{-}90}$" if st == "w" else "$w_{\\mathrm{TV}}$")
    ax[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "bank_regression.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(RESULTS, "bank_analysis.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({m: {k: v for k, v in s.items() if k != "binned"}
                      for m, s in stats.items()}, indent=2)[:3000])


if __name__ == "__main__":
    main()
