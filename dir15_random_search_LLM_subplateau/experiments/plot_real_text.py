"""Figures for the real-language-path screen (operator feedback human_feedback_2).

1) real_text_prevalence.png — does the third-token region / the sub-plateau survive when every point
   of the path is a real token sequence?  Rates side by side with the activation screen, path
   sharpness w(10->90), C-window flatness rho, and where the B prediction first appears.
2) real_text_examples.png   — d(t) for the 3 highest-scoring real-text candidates in each bank.
"""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ALPHAS, PLOTS, RESULTS, rle, wilson
from cvd_style import CVD, REF_DIAG, REF_RULE, use_cvd
from real_text_paths import K_STEPS, T_GRID, w10_90

use_cvd()
RHO_FLAT = 0.5
BANKS = [("random_pairs", "real text · random pairs"),
         ("final_token_matched", "real text · final-token-matched pairs")]


def rho_of(d, kin, kout, grid):
    seg = d[kin:kout + 1]
    return float(seg.max() - seg.min()) / float(grid[kout] - grid[kin])


def show(t):
    return repr(t)[1:-1].replace(" ", "␣")


def main():
    o = json.load(open(os.path.join(RESULTS, "real_text_paths.json")))
    z = np.load(os.path.join(RESULTS, "real_text_curves.npz"))
    mj = json.load(open(os.path.join(RESULTS, "matthew_examples.json")))
    mz = np.load(os.path.join(RESULTS, "matthew_d_curves.npz"))
    by_layer = json.load(open(os.path.join(RESULTS, "analysis.json")))["by_layer"]
    pri = pickle.load(open(os.path.join(RESULTS, "screen_primary.pkl"), "rb"))
    prows, ppaths = pri["rows"], pri["paths"]

    # activation-screen reference curves (candidates + matched non-candidates, 50-alpha grid)
    a_cand, a_kin, a_kout = mz["d_cand"], mz["kin"], mz["kout"]
    a_rho = np.array([rho_of(a_cand[i], a_kin[i], a_kout[i], ALPHAS) for i in range(len(a_cand))])
    a_w = np.array([w10_90_alpha(a_cand[i]) for i in range(len(a_cand))])
    a_w_ctrl = np.array([w10_90_alpha(mz["d_ctrl"][i]) for i in range(len(mz["d_ctrl"]))])

    # ------------------------------------------------------------------ figure 1: six panels
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.0))
    a1, a2, a3, a4, a5, a6 = axes.ravel()

    # (A) prevalence under the SAME symmetric rule for every path family
    lay = np.array([prows[i]["layer"] for i in mz["cand_idx"]])
    rho_by_path = {int(mz["cand_idx"][i]): float(a_rho[i]) for i in range(len(a_rho))}

    def flanks_ok(i):
        rr = rle(ppaths[i]["top1"])
        return (rr[0][2] - rr[0][1] + 1) >= 3 and (rr[-1][2] - rr[-1][1] + 1) >= 3

    def act_rates(sel):
        elig = [i for i in sel if prows[i]["eligible"]]
        st = [i for i in elig if prows[i]["is_candidate"] and flanks_ok(i)]
        sub = [i for i in st if rho_by_path[i] < RHO_FLAT]
        n = len(elig)
        return len(st) / n, wilson(len(st), n), len(sub) / n, wilson(len(sub), n)

    labels, c_rate, c_ci, s_rate, s_ci = [], [], [], [], []
    for sel, lab in ((list(range(len(prows))), "activation interp.\nblocks 0–6 (pooled)"),
                     ([i for i in range(len(prows)) if prows[i]["layer"] == 6],
                      "activation interp.\nblock 6 only")):
        cr, cc, sr, sc = act_rates(sel)
        labels.append(lab)
        c_rate.append(cr); c_ci.append(cc); s_rate.append(sr); s_ci.append(sc)
    act_pooled, act_b6 = (c_rate[0], c_ci[0], s_rate[0], s_ci[0]), (c_rate[1], c_ci[1],
                                                                    s_rate[1], s_ci[1])
    n_e6, k6 = by_layer["6"]["n_eligible"], int((a_rho[lay == 6] < RHO_FLAT).sum())
    for key, lab in BANKS:
        r = o[key]
        labels.append(lab.replace(" · ", "\n"))
        c_rate.append(r["strict_rate"]); c_ci.append(tuple(r["strict_rate_ci"]))
        s_rate.append(r["strict_sub_rate"]); s_ci.append(tuple(r["strict_sub_rate_ci"]))
    x = np.arange(len(labels))
    for off, vals, cis, col, hat, lab in (
            (-0.18, c_rate, c_ci, CVD[0], "//", "persistent third token (A|C|B rule)"),
            (0.18, s_rate, s_ci, CVD[1], "\\\\", "true sub-plateau (ρ < 0.5)")):
        v = np.array(vals) * 100
        err = np.abs(np.array(cis).T * 100 - v)
        a1.bar(x + off, v, width=0.34, color=col, hatch=hat, ec="k", lw=0.8, label=lab)
        a1.errorbar(x + off, v, yerr=err, fmt="none", ecolor="k", capsize=3, lw=1.1)
        for xi, vi in zip(x + off, v):
            a1.annotate(f"{vi:.1f}%", (xi, vi), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=8)
    a1.set_xticks(x)
    a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel("% of eligible paths")
    a1.set_title("Does the third region survive when every path point is real text?\n"
                 "(symmetric rule: the A, C and B runs must each last ≥ 3 grid points)",
                 fontsize=10)
    a1.legend(fontsize=8)

    # (B) path sharpness
    bins = np.linspace(0, 1, 26)
    styles = [(CVD[0], "-"), (CVD[1], "--")]
    for (key, lab), (col, ls) in zip(BANKS, styles):
        w = np.array([w10_90(d) for d in z[key + "_d"]])
        a2.hist(w[np.isfinite(w)], bins=bins, histtype="step", lw=2.2, color=col, ls=ls,
                density=True, label=f"{lab} ({ls})")
    a2.hist(a_w_ctrl[np.isfinite(a_w_ctrl)], bins=bins, histtype="step", lw=2.2, color=CVD[2],
            ls=":", density=True, label="activation interp. · ordinary paths (dotted)")
    a2.hist(a_w[np.isfinite(a_w)], bins=bins, histtype="step", lw=2.2, color=CVD[3], ls="-.",
            density=True, label="activation interp. · A|C|B candidates (dash-dot)")
    a2.set_xlabel("transition width  w(10→90)   (fraction of the path)")
    a2.set_ylabel("density")
    a2.set_title("Are there plateaus at all? Sharpness of the A→B transition", fontsize=10)
    a2.legend(fontsize=7.5)

    # (C) flatness of the C window
    bins = np.linspace(0, 6, 31)
    for (key, lab), (col, ls) in zip(BANKS, styles):
        d, kin, kout, isc = z[key + "_d"], z[key + "_kin"], z[key + "_kout"], z[key + "_iscand"]
        idx = np.where(isc)[0]
        rho = np.array([rho_of(d[i], kin[i], kout[i], T_GRID) for i in idx])
        a3.hist(np.clip(rho, 0, 6), bins=bins, histtype="step", lw=2.2, color=col, ls=ls,
                density=True, label=f"{lab}, n={len(rho)} ({ls})")
    a3.hist(np.clip(a_rho, 0, 6), bins=bins, histtype="step", lw=2.2, color=CVD[3], ls="-.",
            density=True, label=f"activation interp. candidates, n={len(a_rho)} (dash-dot)")
    a3.axvline(1.0, label="ρ = 1: as steep as the diagonal", **REF_DIAG)
    a3.axvline(RHO_FLAT, label=f"ρ = {RHO_FLAT}: sub-plateau line", **REF_RULE)
    a3.set_xlabel("flatness ρ of the C window  (range of d ÷ width in t)")
    a3.set_ylabel("density")
    a3.set_title("When a third token appears, is it a shelf?", fontsize=10)
    a3.legend(fontsize=7.5)

    # (D) where the B prediction first appears along the text path
    for (key, lab), (col, ls) in zip(BANKS, styles):
        t1 = z[key + "_top1"]
        first = []
        for row in t1:
            B = row[-1]
            first.append(int(np.argmax(row == B)))
        a4.hist(np.array(first), bins=np.arange(0, K_STEPS + 1) - 0.5, histtype="step", lw=2.2,
                color=col, ls=ls, density=True, label=f"{lab} ({ls})")
    a4.set_xlabel("step k at which the B prediction first becomes top-1  (0 = A's text, 32 = B's)")
    a4.set_ylabel("density")
    a4.set_title("How abruptly does real text hand over to B?", fontsize=10)
    a4.legend(fontsize=7.5)

    # (E) is the output motion concentrated in a few sharp boundaries, or spread out?
    def top_decile_share(d):
        s = np.abs(np.diff(d))
        if s.sum() <= 0:
            return np.nan
        k = max(1, int(round(0.1 * len(s))))
        return float(np.sort(s)[-k:].sum() / s.sum())
    bins = np.linspace(0, 1, 26)
    conc = {}
    for (key, lab), (col, ls) in zip(BANKS, styles):
        v = np.array([top_decile_share(d) for d in z[key + "_d"]])
        conc[key] = v
        a5.hist(v[np.isfinite(v)], bins=bins, histtype="step", lw=2.2, color=col, ls=ls,
                density=True, label=f"{lab} ({ls})")
    v_act = np.array([top_decile_share(d) for d in np.concatenate([a_cand, mz["d_ctrl"]])])
    conc["activation"] = v_act
    a5.hist(v_act[np.isfinite(v_act)], bins=bins, histtype="step", lw=2.2, color=CVD[3], ls="-.",
            density=True, label="activation interp. paths (dash-dot)")
    a5.axvline(0.1, label="0.1 = motion spread evenly (no boundary)", **REF_DIAG)
    a5.set_xlabel("share of total output motion Σ|Δd| in the sharpest 10% of steps")
    a5.set_ylabel("density")
    a5.set_title("Sharp boundaries or a smooth ramp?", fontsize=10)
    a5.legend(fontsize=7.5)

    # (F) how many distinct top-1 runs does a path pass through?
    bins = np.arange(0.5, 16.5, 1.0)
    for (key, lab), (col, ls) in zip(BANKS, styles):
        n = np.array([len(rle(t)) for t in z[key + "_top1"]])
        a6.hist(np.clip(n, 1, 15), bins=bins, histtype="step", lw=2.2, color=col, ls=ls,
                density=True, label=f"{lab}, median {int(np.median(n))} ({ls})")
    n_act = np.array([prows[i]["n_runs"] for i in range(len(prows)) if prows[i]["eligible"]])
    a6.hist(np.clip(n_act, 1, 15), bins=bins, histtype="step", lw=2.2, color=CVD[3], ls="-.",
            density=True, label=f"activation interp., median {int(np.median(n_act))} (dash-dot)")
    a6.set_xlabel("number of top-1 runs on the path (clipped at 15)")
    a6.set_ylabel("density")
    a6.set_title("How many predictions does one path pass through?", fontsize=10)
    a6.legend(fontsize=7.5)

    for ax in (a1, a2, a3, a4, a5, a6):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "real_text_prevalence.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    extra = {"top_decile_share": {k: {"median": float(np.nanmedian(v)),
                                      "q25": float(np.nanpercentile(v, 25)),
                                      "q75": float(np.nanpercentile(v, 75))}
                                  for k, v in conc.items()},
             "n_runs_median": {k: float(np.median([len(rle(t)) for t in z[k + "_top1"]]))
                               for k, _ in BANKS},
             "n_runs_median_activation": float(np.median(n_act)),
             "w10_90_activation_candidates": float(np.nanmedian(a_w)),
             "w10_90_activation_ordinary": float(np.nanmedian(a_w_ctrl)),
             "block6_sub_rate_frozen_rule": k6 / n_e6,
             "block6_sub_ci_frozen_rule": list(wilson(k6, n_e6)),
             "activation_symmetric_pooled": {"rate": act_pooled[0], "ci": list(act_pooled[1]),
                                             "sub_rate": act_pooled[2],
                                             "sub_ci": list(act_pooled[3])},
             "activation_symmetric_block6": {"rate": act_b6[0], "ci": list(act_b6[1]),
                                             "sub_rate": act_b6[2], "sub_ci": list(act_b6[3])}}
    with open(os.path.join(RESULTS, "real_text_extra.json"), "w") as f:
        json.dump(extra, f, indent=1)
    print(json.dumps(extra, indent=1))

    # ------------------------------------------------------------------ figure 2: examples
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), sharey=True)
    for r, (key, lab) in enumerate(BANKS):
        d = z[key + "_d"]
        for c, ex in enumerate(o[key]["examples"][:3]):
            ax = axes[r, c]
            i = ex["path"]
            ax.plot([0, 1], [0, 1], label="no plateau (d = t)", **REF_DIAG)
            ax.axvspan(T_GRID[ex["k_in"]], T_GRID[ex["k_out"]], color="0.82", hatch="//",
                       ec="0.55", lw=0, zorder=0)
            ax.plot(T_GRID, d[i], color=CVD[0], ls="-", lw=2, marker="o", ms=3,
                    label="d(t), final logits")
            for s in ex["sequence"][1:]:
                ax.axvline(T_GRID[s["k_lo"]] - 0.5 / (K_STEPS - 1), color="0.75", lw=0.6, zorder=0)
            for tt, labl in ((T_GRID[max(ex["k_in"] // 2, 0)], f"A = {show(ex['A'])}"),
                             (0.5 * (T_GRID[ex["k_in"]] + T_GRID[ex["k_out"]]),
                              f"C = {show(ex['C'])}"),
                             (0.5 * (T_GRID[ex["k_out"]] + 1.0), f"B = {show(ex['B'])}")):
                ax.annotate(labl, (tt, 1.04), xycoords=("data", "axes fraction"), ha="center",
                            fontsize=8, annotation_clip=False)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{lab} · rank {ex['rank']} · C run k {ex['k_in']}–{ex['k_out']}"
                         f" · ρ = {ex['rho']:.2f}", fontsize=8.5, pad=16)
            ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("path position  t = k / 32   (k tokens replaced by context B's)")
    for ax in axes[:, 0]:
        ax.set_ylabel("d(t): normalized logit distance\n0 = looks like A,  1 = looks like B")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Highest-scoring real-language A|C|B paths: every point is a real 32-token "
                 "sequence run through the unmodified model (no patching).", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "real_text_examples.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/real_text_prevalence.png, plots/real_text_examples.png")


def w10_90_alpha(d):
    """w(10->90) on the 50-point alpha grid of the activation screen."""
    def cross(y):
        k = int(np.argmax(d >= y))
        if d[k] < y:
            return np.nan
        if k == 0:
            return float(ALPHAS[0])
        x0, x1, y0, y1 = ALPHAS[k - 1], ALPHAS[k], d[k - 1], d[k]
        return float(x0 + (y - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(x1)
    return cross(0.9) - cross(0.1)


if __name__ == "__main__":
    main()
