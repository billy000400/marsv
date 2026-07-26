"""S3 — census, prevalence, threshold sensitivity, validation-bank confirmation, controls.

Reads results/screen_*.pkl, applies the frozen A|C|B rule (already stored) plus the preregistered
sensitivity variants, and writes results/analysis.json + the aggregate figures.
"""
import json
import os
import pickle
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast

from common import ALPHAS, LAYERS, MODEL, PLOTS, RESULTS, REVISION, detect, wilson
from cvd_style import CVD, LINESTYLES, MARKERS, use_cvd

use_cvd()
COND = {0: "context A", 1: "context B"}


def load(name):
    p = os.path.join(RESULTS, f"{name}.pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def bank_stats(d, min_run=3, margin_thr=0.0):
    """Path- and pair-level candidate counts under a (possibly non-default) threshold."""
    rows, paths = d["rows"], d["paths"]
    if min_run == 3 and margin_thr == 0.0:
        flags = [r["is_candidate"] for r in rows]
    else:
        flags = []
        for r, s in zip(rows, paths):
            dd = detect(s, min_run=min_run)
            flags.append(bool(dd["is_candidate"] and dd["margin_min"] > margin_thr))
    elig = np.array([r["eligible"] for r in rows])
    flags = np.array(flags)
    pair_ids = np.array([r["pair"] for r in rows])
    pairs_el = set(pair_ids[elig])
    pairs_cd = set(pair_ids[flags])
    return {"n_paths": len(rows), "n_eligible": int(elig.sum()), "n_cand": int(flags.sum()),
            "path_rate": float(flags.sum() / max(elig.sum(), 1)),
            "path_ci": wilson(int(flags.sum()), int(elig.sum())),
            "n_pairs_eligible": len(pairs_el), "n_pairs_cand": len(pairs_cd),
            "pair_rate": len(pairs_cd) / max(len(pairs_el), 1),
            "pair_ci": wilson(len(pairs_cd), len(pairs_el)),
            "flags": flags, "elig": elig}


def by_group(d, key):
    out = {}
    rows = d["rows"]
    for r in rows:
        out.setdefault(r[key], [0, 0])
        out[r[key]][0] += int(r["eligible"])
        out[r[key]][1] += int(r["is_candidate"])
    return {k: {"n_eligible": v[0], "n_cand": v[1], "rate": v[1] / max(v[0], 1),
                "ci": wilson(v[1], v[0])} for k, v in sorted(out.items())}


def main():
    os.makedirs(PLOTS, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained(MODEL, revision=REVISION)
    man = json.load(open(os.path.join(RESULTS, "manifest.json")))
    pri = load("screen_primary")
    val = load("screen_validation")
    sp = load("screen_primary_same_pred")
    slf = load("screen_primary_self")
    lin = load("screen_primary_lerp")

    A = {"manifest": {k: man[k] for k in ("model", "revision", "ctx_len", "layers", "n_alpha",
                                          "seed", "n_windows", "n_primary_pairs",
                                          "n_validation_pairs", "n_reference", "corpus",
                                          "rejected_same_prediction")}}
    st = bank_stats(pri)
    A["primary"] = {k: v for k, v in st.items() if k not in ("flags", "elig")}
    A["primary"]["endpoint_fidelity_maxabs_logit"] = float(max(pri["endpoint_fidelity_maxabs_logit"]))
    cand_rows = [r for r in pri["rows"] if r["is_candidate"]]
    A["primary"]["n_clean"] = sum(r["clean"] for r in cand_rows)
    A["primary"]["n_complex"] = len(cand_rows) - A["primary"]["n_clean"]
    # own end = the alpha whose activation belongs to the conditioning context (exact no-op);
    # foreign end = the other one, which need not reproduce its home context's prediction.
    own, tr = [], []
    for r, s in zip(pri["rows"], pri["paths"]):
        if r["cond"] == 0:                       # conditioned on context A: own end is alpha=0
            own.append(int(s["top1"][0]) == r["unpatched_A"])
            tr.append(int(s["top1"][-1]) == r["unpatched_B"])
        else:                                    # conditioned on context B: own end is alpha=1
            own.append(int(s["top1"][-1]) == r["unpatched_B"])
            tr.append(int(s["top1"][0]) == r["unpatched_A"])
    A["primary"]["own_endpoint_match_rate"] = float(np.mean(own))
    A["primary"]["foreign_endpoint_transfer_rate"] = float(np.mean(tr))
    # prevalence restricted to paths whose foreign endpoint also reproduces its own prediction
    tr = np.array(tr, bool)
    el, fl = st["elig"], st["flags"]
    A["primary"]["transfer_consistent"] = {
        "n_eligible": int((el & tr).sum()), "n_cand": int((fl & tr).sum()),
        "rate": float((fl & tr).sum() / max((el & tr).sum(), 1)),
        "ci": wilson(int((fl & tr).sum()), int((el & tr).sum()))}

    # is the C region a confident state or a flat "no prediction" zone?
    conf = {"p_C": [], "p_end": [], "H_C": [], "H_end": []}
    for i, r in enumerate(pri["rows"]):
        if not r["is_candidate"]:
            continue
        s = pri["paths"][i]
        kc = (r["k_in"] + r["k_out"]) // 2
        conf["p_C"].append(float(s["top1_p"][kc]))
        conf["p_end"].append(float(0.5 * (s["top1_p"][0] + s["top1_p"][-1])))
        conf["H_C"].append(float(s["entropy"][kc]))
        conf["H_end"].append(float(0.5 * (s["entropy"][0] + s["entropy"][-1])))
    A["c_region_confidence"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                                for k, v in conf.items()}
    A["c_region_confidence"]["frac_C_sharper_than_endpoints"] = float(
        np.mean(np.array(conf["H_C"]) < np.array(conf["H_end"])))
    A["c_region_confidence"]["frac_C_more_confident_than_endpoints"] = float(
        np.mean(np.array(conf["p_C"]) > np.array(conf["p_end"])))
    marg = np.array([r["margin_min"] for r in cand_rows])
    A["c_region_confidence"]["frac_margin_gt_0.05"] = float(np.mean(marg > 0.05))
    A["c_region_confidence"]["frac_margin_gt_0.2"] = float(np.mean(marg > 0.2))
    common10 = {c["id"] for c in
                [{"id": t} for t, _ in Counter(int(r["A"]) for r in pri["rows"]
                                               if r["eligible"]).most_common(10)]}
    A["C_in_top10_common_endpoint_frac"] = float(
        np.mean([r["C"] in common10 for r in cand_rows])) if cand_rows else float("nan")

    A["by_layer"] = {str(k): v for k, v in by_group(pri, "layer").items()}
    A["by_cond"] = {COND[k]: v for k, v in by_group(pri, "cond").items()}

    # threshold sensitivity
    A["sensitivity"] = {}
    for mr in (2, 3, 5):
        for mt in (0.0, 0.02, 0.05):
            s = bank_stats(pri, min_run=mr, margin_thr=mt)
            A["sensitivity"][f"min_run={mr},margin>{mt}"] = {
                "path_rate": s["path_rate"], "pair_rate": s["pair_rate"], "n_cand": s["n_cand"]}

    # validation bank (rule applied unchanged)
    if val:
        sv = bank_stats(val)
        A["validation"] = {k: v for k, v in sv.items() if k not in ("flags", "elig")}
        A["validation"]["endpoint_fidelity_maxabs_logit"] = float(
            max(val["endpoint_fidelity_maxabs_logit"]))

    # controls
    A["controls"] = {}
    for nm, d in (("same_prediction_pairs", sp), ("self_pairs", slf), ("linear_interp", lin)):
        if d:
            s = bank_stats(d)
            A["controls"][nm] = {k: v for k, v in s.items() if k not in ("flags", "elig")}

    # intermediate-token census
    C = Counter(int(r["C"]) for r in cand_rows)
    A["C_census"] = [{"id": t, "token": tok.decode([t]), "n": n} for t, n in C.most_common(25)]
    endp = Counter(int(r["A"]) for r in pri["rows"] if r["eligible"])
    A["endpoint_census"] = [{"id": t, "token": tok.decode([t]), "n": n} for t, n in endp.most_common(15)]

    # candidate table
    keys = ("pair", "ia", "ib", "layer", "cond", "A", "B", "C", "run_len", "width_C",
            "margin_min", "sep", "score", "k_in", "k_out", "clean", "n_transient",
            "jsd_in", "jsd_out", "ep1_match_foreign")
    tab = sorted(({k: (float(r[k]) if isinstance(r[k], (float, np.floating)) else int(r[k]))
                   for k in keys} for r in cand_rows), key=lambda r: -r["score"])
    for r in tab:
        r["A_tok"], r["B_tok"], r["C_tok"] = (tok.decode([r["A"]]), tok.decode([r["B"]]),
                                              tok.decode([r["C"]]))
    with open(os.path.join(RESULTS, "candidates.json"), "w") as f:
        json.dump(tab, f, indent=1)
    A["top_candidates"] = tab[:10]

    with open(os.path.join(RESULTS, "analysis.json"), "w") as f:
        json.dump(A, f, indent=1, default=float)

    # ---------------------------------------------------------------------------- figures
    # 1. prevalence by layer (and by conditioning context)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ls = sorted(A["by_layer"])
    r = [A["by_layer"][k]["rate"] for k in ls]
    lo = [A["by_layer"][k]["rate"] - A["by_layer"][k]["ci"][0] for k in ls]
    hi = [A["by_layer"][k]["ci"][1] - A["by_layer"][k]["rate"] for k in ls]
    ax[0].bar(range(len(ls)), r, color=CVD[0], hatch="//", edgecolor="k")
    ax[0].errorbar(range(len(ls)), r, yerr=[lo, hi], fmt="none", ecolor="k", capsize=4)
    ax[0].set_xticks(range(len(ls)))
    ax[0].set_xticklabels([f"block {k}" for k in ls])
    ax[0].set_ylabel("A|C|B rate per eligible path")
    ax[0].set_title("by interpolation layer")
    names = list(A["by_cond"]) + list(A["controls"])
    vals = [A["by_cond"][k]["rate"] for k in A["by_cond"]] + \
           [A["controls"][k]["path_rate"] for k in A["controls"]]
    cis = [A["by_cond"][k]["ci"] for k in A["by_cond"]] + \
          [A["controls"][k]["path_ci"] for k in A["controls"]]
    hatches = ["//", "\\\\", "..", "xx", "oo"]
    for i, (n, v, c) in enumerate(zip(names, vals, cis)):
        ax[1].bar(i, v, color=CVD[i % len(CVD)], hatch=hatches[i % len(hatches)], edgecolor="k")
        ax[1].errorbar(i, v, yerr=[[v - c[0]], [c[1] - v]], fmt="none", ecolor="k", capsize=4)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax[1].set_title("conditioning context and controls")
    ax[1].set_ylabel("A|C|B rate per eligible path")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "candidate_prevalence_by_layer.png"), dpi=150)
    plt.close(fig)

    # 2. width / margin distributions
    if cand_rows:
        w = np.array([r["width_C"] for r in cand_rows])
        m = np.array([r["margin_min"] for r in cand_rows])
        kin = np.array([ALPHAS[r["k_in"]] for r in cand_rows])
        kout = np.array([ALPHAS[r["k_out"]] for r in cand_rows])
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        ax[0].hist(w, bins=np.arange(0.02, 1.02, 0.04), color=CVD[0], edgecolor="k", hatch="//")
        ax[0].set_xlabel("C-segment width (fraction of alpha grid)")
        ax[0].set_ylabel("candidate paths")
        ax[1].hist(m, bins=30, color=CVD[1], edgecolor="k", hatch="\\\\")
        ax[1].set_xlabel(r"minimum margin  $\min_k\,[p_C-\max(p_A,p_B)]$")
        ax[1].set_ylabel("candidate paths")
        ax[2].scatter(kin, kout, s=12, c=CVD[2], marker="o", alpha=0.6, edgecolor="none")
        ax[2].plot([0, 1], [0, 1], color="0.45", ls="--", lw=1.2)
        ax[2].set_xlabel(r"entry alpha $\alpha_{in}$")
        ax[2].set_ylabel(r"exit alpha $\alpha_{out}$")
        ax[2].set_title("transition locations")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "segment_width_margin_distribution.png"), dpi=150)
        plt.close(fig)

        # 3. intermediate-token census
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        cc = A["C_census"][:15]
        ax[0].barh(range(len(cc))[::-1], [c["n"] for c in cc], color=CVD[0], hatch="//",
                   edgecolor="k")
        ax[0].set_yticks(range(len(cc))[::-1])
        ax[0].set_yticklabels([repr(c["token"]) for c in cc], fontsize=8)
        ax[0].set_xlabel("candidate paths with this C")
        ax[0].set_title("intermediate (C) token census")
        ec = A["endpoint_census"][:15]
        ax[1].barh(range(len(ec))[::-1], [c["n"] for c in ec], color=CVD[1], hatch="\\\\",
                   edgecolor="k")
        ax[1].set_yticks(range(len(ec))[::-1])
        ax[1].set_yticklabels([repr(c["token"]) for c in ec], fontsize=8)
        ax[1].set_xlabel("eligible paths with this endpoint A")
        ax[1].set_title("endpoint (A) token census")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "intermediate_token_census.png"), dpi=150)
        plt.close(fig)

        # 4. probability paths for top-ranked and random candidates
        rng = np.random.default_rng(7)
        idx_c = [i for i, r in enumerate(pri["rows"]) if r["is_candidate"]]
        order = sorted(idx_c, key=lambda i: -pri["rows"][i]["score"])
        pick = order[:3] + list(rng.choice([i for i in idx_c if i not in order[:3]],
                                           size=min(3, max(len(idx_c) - 3, 0)), replace=False))
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, sharey=True)
        for ax_, i in zip(axes.ravel(), pick):
            r, s = pri["rows"][i], pri["paths"][i]
            pid = {int(t): j for j, t in enumerate(s["tok_ids"])}
            for n, (t, lbl) in enumerate([(r["A"], "A"), (r["C"], "C"), (r["B"], "B")]):
                ax_.plot(ALPHAS, s["tok_probs"][:, pid[t]], color=CVD[n], ls=LINESTYLES[n],
                         marker=MARKERS[n], ms=3, lw=1.6,
                         label=f"{lbl}={tok.decode([t])!r}")
            ax_.axvspan(ALPHAS[r["k_in"]], ALPHAS[r["k_out"]], color="0.85", zorder=0)
            ax_.set_title(f"block {r['layer']}, cond {COND[r['cond']]}, score={r['score']:.3f}",
                          fontsize=8)
            ax_.legend(fontsize=7, loc="upper left")
            tw = ax_.twinx()
            tw.plot(0.5 * (ALPHAS[:-1] + ALPHAS[1:]), s["jsd"], color="0.45", ls=":", lw=1.4)
            tw.set_ylim(0, max(0.05, float(s["jsd"].max()) * 1.05))
            tw.tick_params(labelsize=6)
            tw.set_ylabel("adjacent JSD (bits, dotted)", fontsize=6)
        for ax_ in axes[-1]:
            ax_.set_xlabel(r"interpolation coefficient $\alpha$")
        for ax_ in axes[:, 0]:
            ax_.set_ylabel("next-token probability")
        fig.suptitle("top-3 ranked (upper row) and 3 randomly drawn (lower row) A|C|B candidates",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "top_candidate_probability_paths.png"), dpi=150)
        plt.close(fig)

    # 5. threshold sensitivity
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for n, mt in enumerate((0.0, 0.02, 0.05)):
        ys = [A["sensitivity"][f"min_run={mr},margin>{mt}"]["path_rate"] for mr in (2, 3, 5)]
        ax.plot([2, 3, 5], ys, color=CVD[n], ls=LINESTYLES[n], marker=MARKERS[n],
                label=f"margin > {mt}")
    ax.axvline(3, color="k", ls=":", lw=1.6)
    ax.text(3.05, ax.get_ylim()[1] * 0.95, "frozen default", fontsize=7, va="top")
    ax.set_xlabel("persistence threshold (consecutive alpha points C must stay top-1)")
    ax.set_ylabel("A|C|B rate per eligible path")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "threshold_sensitivity.png"), dpi=150)
    plt.close(fig)

    # 6. is C confident? top-1 probability and entropy at the C centre vs at the endpoints
    if cand_rows:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for n, (k1, k2, xl) in enumerate([("p_C", "p_end", "top-1 probability"),
                                          ("H_C", "H_end", "predictive entropy (bits)")]):
            b = np.linspace(0, max(max(conf[k1]), max(conf[k2])), 40)
            ax[n].hist(conf[k1], bins=b, histtype="step", lw=2, color=CVD[0], ls="-",
                       label="C-region centre", density=True)
            ax[n].hist(conf[k2], bins=b, histtype="step", lw=2, color=CVD[1], ls="--",
                       label="path endpoints (mean of $\\alpha$=0,1)", density=True)
            ax[n].set_xlabel(xl)
            ax[n].set_ylabel("density")
            ax[n].legend(fontsize=8)
        ax[0].set_title("how confident is the third region?")
        ax[1].set_title("how sharp is the distribution there?")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "c_region_confidence.png"), dpi=150)
        plt.close(fig)

    print(json.dumps({k: A[k] for k in ("primary", "by_layer", "by_cond", "controls",
                                        "validation") if k in A}, indent=1, default=float)[:2500])
    print("C census:", [(c["token"], c["n"]) for c in A["C_census"][:10]])


if __name__ == "__main__":
    main()
