"""Figures + summary numbers for the same-context token-to-token screen (operator feedback #3).

Two figures:
  plots/token_prevalence.png  -- six panels: rates by hook point against the context-to-context
                                 screen, C-window flatness, the flat-AND-intermediate criterion,
                                 path sharpness, controls, and path complexity;
  plots/token_examples.png    -- output-distance curves d(alpha) for the flattest token paths.

Writes results/token_examples.json (decoded context, both final tokens, full top-1 sequence per
example) and results/token_summary.json (every number quoted in RESULTS.md / REPORT.md).

usage: python plot_token.py
"""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2TokenizerFast

from common import ALPHAS, MODEL, PLOTS, RESULTS, REVISION, rle, wilson
from cvd_style import CVD, REF_DIAG, REF_RULE, use_cvd
from token_interp import kappa, rho_of, w10_90

use_cvd()
HOOKS = ["embed", "0", "2", "4", "6"]
HOOK_LABEL = {"embed": "token\nembedding", "0": "block 0", "2": "block 2", "4": "block 4",
              "6": "block 6"}
RHO_FLAT, D_LO, D_HI = 0.5, 0.2, 0.8      # "true sub-plateau": flat AND at intermediate height


def show(t):
    return repr(t)[1:-1].replace(" ", "␣")


def shelf(d, kin, kout):
    """(flatness rho, mean height d_bar) of one C window."""
    return rho_of(d, kin, kout), float(d[kin:kout + 1].mean())


def context_screen():
    """rho, d_bar, block and the matched-control equivalents for the context-to-context screen."""
    z = np.load(os.path.join(RESULTS, "matthew_d_curves.npz"))
    pri = pickle.load(open(os.path.join(RESULTS, "screen_primary.pkl"), "rb"))
    kin, kout, ci = z["kin"], z["kout"], z["cand_idx"]
    cand = np.array([shelf(z["d_cand"][i], kin[i], kout[i]) for i in range(len(ci))])
    ctrl = np.array([shelf(z["d_ctrl"][i], kin[i], kout[i]) for i in range(len(ci))])
    lay = np.array([pri["rows"][i]["layer"] for i in ci])
    w = np.array([w10_90(z["d_cand"][i]) for i in range(len(ci))])
    return {"rho": cand[:, 0], "dbar": cand[:, 1], "rho_ctrl": ctrl[:, 0],
            "dbar_ctrl": ctrl[:, 1], "layer": lay, "w": w,
            "n_eligible_by_layer": {l: json.load(open(os.path.join(RESULTS, "analysis.json")))
                                    ["by_layer"][str(l)]["n_eligible"] for l in (0, 2, 4, 6)}}


def strict_counts(rows, D):
    """Candidates whose C window is flat AND sits at an intermediate output height, plus the same
    criterion scored on the matched non-candidate control windows."""
    cand = [n for n, r in enumerate(rows) if r["is_candidate"]]
    non = [n for n, r in enumerate(rows) if r["eligible"] and not r["is_candidate"]]
    rng = np.random.default_rng(13)
    ctrl = rng.choice(non, size=min(len(cand), len(non)), replace=False) if cand and non else []
    sc = np.array([shelf(D[n], rows[n]["k_in"], rows[n]["k_out"]) for n in cand]) \
        if cand else np.zeros((0, 2))
    kc = np.array([shelf(D[int(m)], rows[n]["k_in"], rows[n]["k_out"])
                   for n, m in zip(cand, ctrl)]) if len(ctrl) else np.zeros((0, 2))
    ok = lambda a: (a[:, 0] < RHO_FLAT) & (a[:, 1] > D_LO) & (a[:, 1] < D_HI)
    return {"cand": sc, "ctrl": kc, "n_strict": int(ok(sc).sum()) if len(sc) else 0,
            "n_strict_ctrl": int(ok(kc).sum()) if len(kc) else 0,
            "n_eligible": sum(r["eligible"] for r in rows), "idx": cand}


def main():
    tok = GPT2TokenizerFast.from_pretrained(MODEL, revision=REVISION)
    o = json.load(open(os.path.join(RESULTS, "token_interp.json")))
    z = np.load(os.path.join(RESULTS, "token_interp_curves.npz"))
    R = pickle.load(open(os.path.join(RESULTS, "token_interp_rows.pkl"), "rb"))["rows"]
    W = np.load(os.path.join(RESULTS, "ctx.npz"))["windows"]
    A = json.load(open(os.path.join(RESULTS, "analysis.json")))
    ctxs = context_screen()

    S = {h: strict_counts(R[f"token_{h}"], z[f"token_{h}_d"]) for h in HOOKS}
    S["discrete"] = strict_counts(R["discrete_real_token"], z["discrete_real_token_d"])

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.6))
    (a1, a2, a3), (a4, a5, a6) = ax

    # (A) third-token rate and true sub-plateau rate per hook point, token path vs context path
    x = np.arange(len(HOOKS))
    tr = np.array([100 * o[f"token_{h}"]["rate"] for h in HOOKS])
    tlo = np.array([100 * o[f"token_{h}"]["rate_ci"][0] for h in HOOKS])
    thi = np.array([100 * o[f"token_{h}"]["rate_ci"][1] for h in HOOKS])
    ts = np.array([100 * S[h]["n_strict"] / S[h]["n_eligible"] for h in HOOKS])
    tslo = np.array([100 * wilson(S[h]["n_strict"], S[h]["n_eligible"])[0] for h in HOOKS])
    tshi = np.array([100 * wilson(S[h]["n_strict"], S[h]["n_eligible"])[1] for h in HOOKS])
    cr, cs = [], []
    for h in HOOKS:
        if h == "embed":
            cr.append(np.nan)
            cs.append(np.nan)
            continue
        n_el = ctxs["n_eligible_by_layer"][int(h)]
        m = ctxs["layer"] == int(h)
        cr.append(100 * A["by_layer"][h]["rate"])
        k = int(((ctxs["rho"][m] < RHO_FLAT) & (ctxs["dbar"][m] > D_LO)
                 & (ctxs["dbar"][m] < D_HI)).sum())
        cs.append(100 * k / n_el)
    a1.bar(x - 0.3, tr, 0.19, yerr=[tr - tlo, thi - tr], capsize=3, color=CVD[0], hatch="//",
           edgecolor="k", lw=0.5, label="token path · third token (//)")
    a1.bar(x - 0.1, ts, 0.19, yerr=[ts - tslo, tshi - ts], capsize=3, color=CVD[1], hatch="\\\\",
           edgecolor="k", lw=0.5, label="token path · true sub-plateau (\\\\)")
    a1.bar(x + 0.1, cr, 0.19, color=CVD[3], hatch="..", edgecolor="k", lw=0.5,
           label="context path · third token (..)")
    a1.bar(x + 0.3, cs, 0.19, color=CVD[4], hatch="xx", edgecolor="k", lw=0.5,
           label="context path · true sub-plateau (xx)")
    val = json.load(open(os.path.join(RESULTS, "token_validation.json")))
    xv = len(HOOKS)
    vr, vs = 100 * val["rate"], 100 * val["strict_sub_rate"]
    a1.bar(xv - 0.3, vr, 0.19, yerr=[[vr - 100 * val["rate_ci"][0]],
                                     [100 * val["rate_ci"][1] - vr]], capsize=3, color=CVD[0],
           hatch="//", edgecolor="k", lw=0.5)
    a1.bar(xv - 0.1, vs, 0.19, yerr=[[vs - 100 * val["strict_sub_ci"][0]],
                                     [100 * val["strict_sub_ci"][1] - vs]], capsize=3, color=CVD[1],
           hatch="\\\\", edgecolor="k", lw=0.5)
    x = np.append(x, xv)
    a1.set_xticks(x)
    a1.set_xticklabels([HOOK_LABEL[h] for h in HOOKS] + ["disjoint\nvalidation\n(embedding)"],
                       fontsize=8.5)
    a1.set_ylabel("% of eligible paths")
    a1.set_xlabel("hook point where the two endpoints are interpolated")
    a1.set_title("(A) One token swapped vs the whole context swapped", fontsize=10)
    a1.legend(fontsize=7.0)

    # (B) how sharp is the whole path?
    bins = np.linspace(0, 1, 41)
    for lab, v, c, ls in (("token path (embedding)",
                           np.array([r["w10_90"] for r in R["token_embed"]]), CVD[0], "-"),
                          ("token path (block 6)",
                           np.array([r["w10_90"] for r in R["token_6"]]), CVD[1], "--"),
                          ("nearest real token, no patch",
                           np.array([r["w10_90"] for r in R["discrete_real_token"]]), CVD[2], ":"),
                          ("context path (candidates)", ctxs["w"], CVD[3], "-.")):
        v = v[np.isfinite(v)]
        a2.hist(v, bins=bins, histtype="step", lw=2.2, color=c, ls=ls, density=True,
                label=f"{lab} ({ls})")
    a2.set_xlabel("transition width w(10→90) as a fraction of the path")
    a2.set_ylabel("density")
    a2.set_title("(B) A token swap is a near-step function", fontsize=10)
    a2.legend(fontsize=7.0)

    # (C) flatness of the C window
    bins = np.linspace(0, 6, 31)
    for lab, v, c, ls in (("token path (embedding), C windows", S["embed"]["cand"][:, 0], CVD[0], "-"),
                          ("token path (embedding), control windows",
                           S["embed"]["ctrl"][:, 0], CVD[1], "--"),
                          ("context path, C windows", ctxs["rho"], CVD[3], "-.")):
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        if v.size:
            a3.hist(np.clip(v, 0, 6), bins=bins, histtype="step", lw=2.2, color=c, ls=ls,
                    density=True, label=f"{lab} ({ls})")
    a3.axvline(1.0, label="ρ = 1: no plateau (dashed grey)", **REF_DIAG)
    a3.axvline(RHO_FLAT, label="ρ = 0.5: flatness cut (dotted)", **REF_RULE)
    a3.set_xlabel("flatness ρ of the C window (clipped at 6)")
    a3.set_ylabel("density")
    a3.set_title("(C) Is the third region flat?", fontsize=10)
    a3.legend(fontsize=7.0)

    # (D) flat AND intermediate: the two-dimensional criterion
    for lab, arr, c, mk in (("C windows (candidates)", S["embed"]["cand"], CVD[0], "o"),
                            ("matched control windows", S["embed"]["ctrl"], CVD[1], "s")):
        if len(arr):
            a4.scatter(np.clip(arr[:, 0], 0, 6), arr[:, 1], s=18, marker=mk, alpha=0.6,
                       edgecolor="k", linewidth=0.3, color=c, label=f"{lab} ({mk})")
    a4.axvline(RHO_FLAT, **REF_RULE)
    a4.axhline(D_LO, **REF_DIAG)
    a4.axhline(D_HI, **REF_DIAG)
    a4.set_xlim(0, 6)
    a4.set_ylim(-0.02, 1.02)
    a4.set_xlabel("flatness ρ of the window (clipped at 6); dotted rule = 0.5")
    a4.set_ylabel("mean output height d̄ of the window")
    a4.set_title("(D) A true sub-plateau is flat AND intermediate\n(bottom-left box, token-embedding"
                 " paths)", fontsize=10)
    a4.legend(fontsize=7.0, loc="upper right")

    # (E) controls
    names, vals, los, his = [], [], [], []
    for key, lab in (("token_embed", "token pairs\n(primary)"),
                     ("control_lerp_embed", "linear\ninterpolation"),
                     ("control_same_prediction_embed", "same-prediction\npairs"),
                     ("control_self_token_embed", "same token\n(self-pair)"),
                     ("discrete_real_token", "nearest real\ntoken (no patch)")):
        s = o[key]
        names.append(lab)
        vals.append(100 * s["detour_rate"])
        los.append(100 * s["detour_rate_ci"][0])
        his.append(100 * s["detour_rate_ci"][1])
    xc = np.arange(len(names))
    vals = np.array(vals)
    a5.bar(xc, vals, 0.55, yerr=[vals - los, np.array(his) - vals], capsize=3,
           color=CVD, edgecolor="k", lw=0.5, hatch=["//", "\\\\", "..", "xx", "++"])
    a5.set_xticks(xc)
    a5.set_xticklabels(names, fontsize=7.6)
    a5.set_ylabel("% of paths with a third-token detour")
    a5.set_title("(E) Controls, all at the token-embedding hook", fontsize=10)

    # (F) how many distinct predictions does the path visit?
    bins = np.arange(0.5, 12.5, 1.0)
    for lab, key, c, ls in (("token path (embedding)", "token_embed", CVD[0], "-"),
                            ("token path (block 6)", "token_6", CVD[1], "--"),
                            ("nearest real token, no patch", "discrete_real_token", CVD[2], ":")):
        v = np.array([r["n_runs_total"] for r in R[key]])
        a6.hist(np.clip(v, 1, 12), bins=bins, histtype="step", lw=2.2, color=c, ls=ls,
                density=True, label=f"{lab} ({ls})")
    a6.set_xlabel("number of top-1 runs along the path (clipped at 12)")
    a6.set_ylabel("density")
    a6.set_title("(F) How many predictions does the path visit?", fontsize=10)
    a6.legend(fontsize=7.0)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "token_prevalence.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------- Figure: examples
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    examples = []
    for row, h in enumerate(["embed", "6"]):
        rows, D = R[f"token_{h}"], z[f"token_{h}_d"]
        cand = [n for n, r in enumerate(rows) if r["is_candidate"]]
        good = [n for n in cand
                if shelf(D[n], rows[n]["k_in"], rows[n]["k_out"])[0] < RHO_FLAT
                and D_LO < shelf(D[n], rows[n]["k_in"], rows[n]["k_out"])[1] < D_HI]
        order = sorted(good or cand, key=lambda n: -rows[n]["score"])
        for col, n in enumerate(order[:3]):
            r = rows[n]
            rho, dbar = shelf(D[n], r["k_in"], r["k_out"])
            axx = axes[row, col]
            axx.plot(ALPHAS, D[n], color=CVD[row], ls=["-", "--"][row], lw=2.0, marker="o", ms=3)
            axx.plot([0, 1], [0, 1], **REF_DIAG)
            axx.axvspan(ALPHAS[r["k_in"]], ALPHAS[r["k_out"]], color="0.85", zorder=0)
            for _, s0, _ in rle(np.array(r["top1_seq"]))[1:]:
                axx.axvline(ALPHAS[s0], color="0.6", lw=0.7)
            axx.set_title(f"{HOOK_LABEL[h].replace(chr(10), ' ')} · "
                          f"'{show(tok.decode([r['t_A']]))}' → '{show(tok.decode([r['t_B']]))}'\n"
                          f"A='{show(tok.decode([r['A']]))}'  C='{show(tok.decode([r['C']]))}'  "
                          f"B='{show(tok.decode([r['B']]))}'  ρ={rho:.2f}  d̄={dbar:.2f}",
                          fontsize=8.0)
            axx.set_xlabel("interpolation coefficient α")
            axx.set_ylabel("relative output distance d(α)")
            examples.append({"hook": h, "rank": col + 1, "path": int(n),
                             "context": tok.decode(W[r["i"], :-1]),
                             "t_A": tok.decode([r["t_A"]]), "t_B": tok.decode([r["t_B"]]),
                             "A": tok.decode([r["A"]]), "C": tok.decode([r["C"]]),
                             "B": tok.decode([r["B"]]), "rho": float(rho), "d_mean_C": float(dbar),
                             "alpha_in": float(ALPHAS[r["k_in"]]),
                             "alpha_out": float(ALPHAS[r["k_out"]]),
                             "run_len": int(r["run_len"]), "clean": bool(r["clean"]),
                             "score": float(r["score"]),
                             "sequence": [{"token": tok.decode([t]), "a_lo": float(ALPHAS[a]),
                                           "a_hi": float(ALPHAS[b])}
                                          for t, a, b in rle(np.array(r["top1_seq"]))]})
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "token_examples.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(RESULTS, "token_examples.json"), "w") as f:
        json.dump(examples, f, indent=1)

    # ------------------------------------------------------------------------- numbers
    out = {"criterion": {"rho_flat": RHO_FLAT, "d_lo": D_LO, "d_hi": D_HI}}
    for h in HOOKS + ["discrete"]:
        key = f"token_{h}" if h in HOOKS else "discrete_real_token"
        s, d = S[h], o[key]
        out[key] = {
            "n_eligible": s["n_eligible"], "n_candidates": d["n_candidates"],
            "rate": d["rate"], "rate_ci": d["rate_ci"], "n_clean": d["n_clean"],
            "n_strict_subplateau": s["n_strict"],
            "strict_sub_rate": s["n_strict"] / s["n_eligible"],
            "strict_sub_ci": list(wilson(s["n_strict"], s["n_eligible"])),
            "n_strict_control": s["n_strict_ctrl"],
            "strict_ctrl_rate": s["n_strict_ctrl"] / s["n_eligible"],
            "median_rho": float(np.median(s["cand"][:, 0])) if len(s["cand"]) else None,
            "median_rho_ctrl": float(np.median(s["ctrl"][:, 0])) if len(s["ctrl"]) else None,
            "median_dbar": float(np.median(s["cand"][:, 1])) if len(s["cand"]) else None,
            "median_w10_90": d["w10_90"]["median"], "median_kappa": d["kappa"]["median"],
            "median_runs": d["n_runs_total"]["median"], "detour_rate": d["detour_rate"],
            "entropy_mid": d["entropy_mid"]["median"],
            "min_run_sensitivity": d.get("min_run_sensitivity"),
        }
    ctx_strict = {}
    for l in (0, 2, 4, 6):
        m = ctxs["layer"] == l
        n_el = ctxs["n_eligible_by_layer"][l]
        k = int(((ctxs["rho"][m] < RHO_FLAT) & (ctxs["dbar"][m] > D_LO)
                 & (ctxs["dbar"][m] < D_HI)).sum())
        ctx_strict[str(l)] = {"n_eligible": n_el, "rate": A["by_layer"][str(l)]["rate"],
                              "n_strict": k, "strict_rate": k / n_el,
                              "strict_ci": list(wilson(k, n_el)),
                              "median_rho": float(np.median(ctxs["rho"][m]))}
    k_all = int(((ctxs["rho"] < RHO_FLAT) & (ctxs["dbar"] > D_LO) & (ctxs["dbar"] < D_HI)).sum())
    kc_all = int(((ctxs["rho_ctrl"] < RHO_FLAT) & (ctxs["dbar_ctrl"] > D_LO)
                  & (ctxs["dbar_ctrl"] < D_HI)).sum())
    ctx_strict["all"] = {"n_eligible": 7611, "n_strict": k_all, "strict_rate": k_all / 7611,
                         "strict_ci": list(wilson(k_all, 7611)), "n_strict_control": kc_all,
                         "median_w10_90": float(np.median(ctxs["w"]))}
    out["context_screen"] = ctx_strict
    out["controls"] = {k: {"n_paths": o[k]["n_paths"], "detour_rate": o[k]["detour_rate"],
                           "detour_ci": o[k]["detour_rate_ci"],
                           "n_candidates": o[k]["n_candidates"], "rate": o[k]["rate"]}
                       for k in ("control_lerp_embed", "control_same_prediction_embed",
                                 "control_self_token_embed", "discrete_real_token")}
    out["discrete_n_distinct_tokens"] = o["discrete_real_token"].get("n_distinct_tokens")
    out["endpoint_fidelity"] = o["endpoint_fidelity"]
    with open(os.path.join(RESULTS, "token_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("wrote plots/token_prevalence.png, plots/token_examples.png, "
          "results/token_examples.json, results/token_summary.json")


if __name__ == "__main__":
    main()
