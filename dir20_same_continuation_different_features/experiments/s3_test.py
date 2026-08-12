"""S3 (fresh confirmatory plan): run the locked contrasts and test the matched prediction.

Reads results/matched_pairs.json (locked by S2, hash recorded), runs the identical block-0
final-token interpolation for both members of every contrast, and tests whether the high-F
member has the smaller transition width w_TV.

Writes results/matched_sweeps.npz, results/matched_metrics.json, and three figures.
"""
import hashlib
import json
import os

import numpy as np
import torch

from common import PLOTS, RESULTS, load
from run_interp import clean_run, rel_dist, slerp_lerp_norm, sweep, width_10_90
from s1_sanity import nonmono, w_tv

N_ALPHA = 101
N_BOOT = 10000
SEED = 31
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
CONFOUNDS = ["jsd", "logit_dist", "angle0", "lognorm_ratio", "mean_surprisal"]
CLABEL = {"jsd": "successor JSD", "logit_dist": "final-logit distance",
          "angle0": "block-0 angle (rad)", "lognorm_ratio": "block-0 |log norm ratio|",
          "mean_surprisal": "mean final-token surprisal", "F": "feature difference $F$"}


def run_pair(m, ids_prefix, tok_a, tok_b, alphas):
    ida, idb = ids_prefix + [tok_a], ids_prefix + [tok_b]
    reca, lga = clean_run(m, ida)
    recb, lgb = clean_run(m, idb)
    vecs, _, _ = slerp_lerp_norm(reca[0], recb[0], alphas)
    _, lg = sweep(m, ida, vecs)
    err = max(float((lg[0] - lga.cpu()).norm() / lga.cpu().norm()),
              float((lg[-1] - lgb.cpu()).norm() / lgb.cpu().norm()))
    d = rel_dist(lg, lga.cpu().unsqueeze(0), lgb.cpu().unsqueeze(0))
    return d, err


def main():
    alphas = np.linspace(0, 1, N_ALPHA)
    lock = json.load(open(os.path.join(RESULTS, "matched_pairs.json")))
    h = hashlib.sha256(open(os.path.join(RESULTS, "matched_pairs.json"), "rb").read()).hexdigest()
    print(f"lock: {lock['n_contrasts']} contrasts, rule={lock['matching_version']}, sha256={h[:16]}")

    tok, m = load("gpt2-large")
    curves, rows, worst = {}, [], 0.0
    for k, c in enumerate(lock["contrasts"]):
        r = dict(prefix=c["prefix"], dF=c["dF"], conf_dist=c["conf_dist"])
        for lab in ("high", "low"):
            e = c[lab]
            d, err = run_pair(m, c["ids"], e["tok_a"], e["tok_b"], alphas)
            curves[f"{c['prefix']}__{lab}"] = d
            worst = max(worst, err)
            r[f"w_tv_{lab}"] = w_tv(alphas, d)
            r[f"w1090_{lab}"] = width_10_90(alphas, d)
            r[f"nonmono_{lab}"] = nonmono(d)
            for f in CONFOUNDS + ["F"]:
                r[f"{f}_{lab}"] = e[f]
        r["dw"] = r["w_tv_high"] - r["w_tv_low"]
        rows.append(r)
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{lock['n_contrasts']} contrasts swept")
    del m
    torch.cuda.empty_cache()

    dw = np.array([r["dw"] for r in rows])
    rng = np.random.default_rng(SEED)
    n = len(dw)
    boot = np.array([np.median(dw[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    # Paired permutation: flip the sign of each contrast's Delta w independently.
    signs = rng.choice([-1.0, 1.0], size=(N_BOOT, n))
    perm = np.median(dw[None, :] * signs, axis=1)
    med = float(np.median(dw))
    p = float((np.abs(perm) >= abs(med)).mean())

    summary = dict(
        n_contrasts=n, matching_version=lock["matching_version"],
        lock_sha256=h, worst_endpoint_rel_err=worst,
        median_dw=med, ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        frac_predicted_sign=float((dw < 0).mean()), perm_p=p,
        mean_dw=float(dw.mean()),
        w_tv_high=[float(np.median([r["w_tv_high"] for r in rows]))],
        w_tv_low=[float(np.median([r["w_tv_low"] for r in rows]))],
        balance={f: dict(
            high=float(np.mean([r[f"{f}_high"] for r in rows])),
            low=float(np.mean([r[f"{f}_low"] for r in rows])),
            smd=float((np.mean([r[f"{f}_high"] for r in rows])
                       - np.mean([r[f"{f}_low"] for r in rows]))
                      / (np.std([r[f"{f}_high"] for r in rows] + [r[f"{f}_low"] for r in rows]) + 1e-12)))
            for f in CONFOUNDS + ["F"]},
        contrasts=rows,
    )
    summary["verdict"] = (
        "supported" if (n >= 80 and med <= -0.05 and summary["frac_predicted_sign"] >= 0.60
                        and summary["ci95"][1] < 0)
        else ("underpowered" if n < 80 else "not supported"))
    print(f"n={n} median dw={med:+.4f} CI={summary['ci95']} "
          f"frac<0={summary['frac_predicted_sign']:.3f} p={p:.4f} -> {summary['verdict']}")

    with open(os.path.join(RESULTS, "matched_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RESULTS, "matched_sweeps.npz"), alphas=alphas, **curves)
    plots(alphas, curves, rows, summary, lock)
    return summary


def plots(alphas, curves, rows, summary, lock):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- balance -----------------------------------------------------------------
    keys = CONFOUNDS + ["F"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.6))
    for a, f in zip(ax.ravel(), keys):
        hi = [r[f"{f}_high"] for r in rows]
        lo = [r[f"{f}_low"] for r in rows]
        a.scatter(lo, hi, s=16, color=CVD[0], marker="o", alpha=0.75)
        lim = [min(lo + hi), max(lo + hi)]
        a.plot(lim, lim, ":", color="0.45", lw=1.3)
        a.set_xlabel(f"low-F member"), a.set_ylabel("high-F member")
        a.set_title(f"{CLABEL[f]}\nSMD = {summary['balance'][f]['smd']:+.3f}", fontsize=9)
    fig.suptitle("Matched-contrast balance: every variable except $F$ sits on the diagonal", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "matching_balance.png"), dpi=150)
    plt.close(fig)

    # --- paired widths -----------------------------------------------------------
    dw = np.array([r["dw"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for r in rows:
        c = CVD[0] if r["dw"] < 0 else CVD[1]
        ls = "-" if r["dw"] < 0 else "--"
        ax[0].plot([0, 1], [r["w_tv_low"], r["w_tv_high"]], ls, color=c, lw=0.8, alpha=0.55)
    ax[0].plot([0, 1], [np.median([r["w_tv_low"] for r in rows]),
                        np.median([r["w_tv_high"] for r in rows])],
               "-", color="k", lw=2.6, marker="o", label="median")
    ax[0].set_xticks([0, 1]), ax[0].set_xticklabels(["low-$F$", "high-$F$"])
    ax[0].set_ylabel(r"transition width $w_{TV}$")
    ax[0].set_title("Paired widths within matched contrasts")
    ax[0].legend(fontsize=8)
    ax[1].hist(dw, bins=25, color=CVD[0], edgecolor="k", lw=0.4)
    ax[1].axvline(0, color="0.35", ls=":", lw=1.5)
    ax[1].axvline(summary["median_dw"], color=CVD[1], ls="--", lw=2,
                  label=f"median = {summary['median_dw']:+.3f}")
    ax[1].axvspan(*summary["ci95"], color=CVD[1], alpha=0.18, label="95% CI")
    ax[1].set_xlabel(r"$\Delta w = w_{TV}(\mathrm{high}\,F) - w_{TV}(\mathrm{low}\,F)$")
    ax[1].set_ylabel("contrasts")
    ax[1].set_title(f"n={summary['n_contrasts']}, "
                    f"{100 * summary['frac_predicted_sign']:.0f}% predicted sign")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "matched_widths.png"), dpi=150)
    plt.close(fig)

    # --- example curves ----------------------------------------------------------
    order = np.argsort(dw)
    picks = list(order[:5]) + list(order[-5:])
    byp = {c["prefix"]: c for c in lock["contrasts"]}
    fig, ax = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True, sharey=True)
    for a, k in zip(ax.ravel(), picks):
        r = rows[k]
        c = byp[r["prefix"]]
        for lab, ls, mk, col in (("high", "-", "o", CVD[0]), ("low", "--", "s", CVD[1])):
            a.plot(alphas, curves[f"{r['prefix']}__{lab}"], ls, color=col, marker=mk,
                   markevery=15, ms=3.5,
                   label=f"{lab}-$F$: {c[lab]['str_a'].strip()}/{c[lab]['str_b'].strip()}  "
                         f"$F$={r[f'F_{lab}']:.2f}, $w$={r[f'w_tv_{lab}']:.3f}")
        a.plot(alphas, alphas, ":", color="0.45", lw=1.0)
        a.set_title(f"prefix {r['prefix']}  JSD$\\approx${r['jsd_high']:.3f}  "
                    f"$\\Delta w$={r['dw']:+.3f}", fontsize=8)
        a.legend(fontsize=6, loc="upper left")
    for a in ax[-1]:
        a.set_xlabel(r"$\alpha$")
    for a in ax[:, 0]:
        a.set_ylabel(r"$d(\alpha)$")
    fig.suptitle("Top row: five strongest supporting contrasts. "
                 "Bottom row: five strongest counterexamples.", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "example_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
