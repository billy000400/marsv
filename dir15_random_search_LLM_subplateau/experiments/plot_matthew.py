"""Figures for the operator-feedback analysis (human_feedback_1): Matthew-style d(t) curves.

1) matthew_dt_frozen.png    — d(t) for the 6 FROZEN inspection paths (3 top-scoring + 3 random).
2) matthew_dt_gallery.png   — d(t) for the 6 clearest sub-plateaus (post-hoc selection, labelled).
3) subplateau_dwell.png     — how flat the C window is (rho) vs a matched control, rho vs the frozen
                              score decile, and where the C window sits between the endpoints.

Also writes results/matthew_gallery.json with the gallery paths' contexts and top-1 sequences.
"""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2TokenizerFast

from common import ALPHAS, MODEL, PLOTS, RESULTS, REVISION, rle
from cvd_style import CVD, REF_DIAG, REF_RULE, use_cvd

use_cvd()
RHO_FLAT = 0.5          # post-hoc "flat enough to be a sub-plateau" line (descriptive only)


def rho_of(d, kin, kout):
    seg = d[kin:kout + 1]
    return float(seg.max() - seg.min()) / float(ALPHAS[kout] - ALPHAS[kin])


def show(t):
    """Printable token: make whitespace visible."""
    return repr(t)[1:-1].replace(" ", "␣")


def panel(ax, d, seq, kin, kout, A, C, B, title):
    ax.plot([0, 1], [0, 1], label="no plateau (d = t)", **REF_DIAG)
    ax.axvspan(ALPHAS[kin], ALPHAS[kout], color="0.82", hatch="//", ec="0.55", lw=0, zorder=0)
    ax.plot(ALPHAS, d, color=CVD[0], ls="-", lw=2, marker="o", ms=2.6, label="d(t), final logits")
    for s in seq[1:]:
        ax.axvline(s["alpha_lo"] - 0.5 / (len(ALPHAS) - 1), color="0.75", lw=0.6, zorder=0)
    for a, lab in ((ALPHAS[max(kin // 2, 0)], f"A = {show(A)}"),
                   (0.5 * (ALPHAS[kin] + ALPHAS[kout]), f"C = {show(C)}"),
                   (0.5 * (ALPHAS[kout] + 1.0), f"B = {show(B)}")):
        ax.annotate(lab, (a, 1.04), xycoords=("data", "axes fraction"), ha="center",
                    fontsize=8, annotation_clip=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, fontsize=9, pad=16)
    ax.grid(alpha=0.25)


def make_fig(items, fname, suptitle):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2), sharey=True)
    for ax, it in zip(axes.ravel(), items):
        panel(ax, np.asarray(it["d"]), it["sequence"], it["k_in"], it["k_out"],
              it["A"], it["C"], it["B"],
              f"{it['tag']} · block {it['layer']} · C run α {it['alpha_in']:.2f}–{it['alpha_out']:.2f}"
              f" · ρ = {it['rho']:.2f}")
    for ax in axes[-1]:
        ax.set_xlabel("interpolation coefficient  t")
    for ax in axes[:, 0]:
        ax.set_ylabel("d(t): normalized logit distance\n0 = looks like A,  1 = looks like B")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(suptitle, y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    tok = GPT2TokenizerFast.from_pretrained(MODEL, revision=REVISION)
    o = json.load(open(os.path.join(RESULTS, "matthew_examples.json")))
    z = np.load(os.path.join(RESULTS, "matthew_d_curves.npz"))
    pri = pickle.load(open(os.path.join(RESULTS, "screen_primary.pkl"), "rb"))
    rows, paths = pri["rows"], pri["paths"]
    ctx = np.load(os.path.join(RESULTS, "ctx.npz"))
    W = ctx["windows"]

    cand_idx, d_cand, kin, kout = z["cand_idx"], z["d_cand"], z["kin"], z["kout"]
    d_ctrl, ctrl_idx = z["d_ctrl"], z["ctrl_idx"]
    rho_c = np.array([rho_of(d_cand[i], kin[i], kout[i]) for i in range(len(cand_idx))])
    rho_x = np.array([rho_of(d_ctrl[i], kin[i], kout[i]) for i in range(len(ctrl_idx))])
    dbar = np.array([d_cand[i, kin[i]:kout[i] + 1].mean() for i in range(len(cand_idx))])
    score = np.array([rows[i]["score"] for i in cand_idx])

    # ---------------------------------------------------------------- figure 1: frozen examples
    frozen = []
    for e in o["examples"]:
        e = dict(e)
        e["tag"] = {"top-1": "frozen top-1", "top-2": "frozen top-2", "top-3": "frozen top-3"}.get(
            e["rank"], "frozen random")
        frozen.append(e)
    make_fig(frozen, "matthew_dt_frozen.png",
             "Matthew-style plateau curves for the 6 pre-frozen inspection paths "
             "(top row: 3 highest-scoring candidates; bottom row: 3 drawn at random). "
             "Shaded band = the detected third-token (C) run; ρ = its d-range ÷ its t-width "
             "(ρ < 1 flatter than the diagonal = a sub-plateau).")

    # ---------------------------------------------------------------- figure 2: sub-plateau gallery
    ok = np.where((rho_c < RHO_FLAT) & ((kout - kin + 1) >= 5))[0]
    pick = ok[np.argsort(rho_c[ok])][:6]
    gallery = []
    for j in pick:
        pi = int(cand_idx[j])
        r, s = rows[pi], paths[pi]
        seq = [{"token": tok.decode([t]), "alpha_lo": float(ALPHAS[a]), "alpha_hi": float(ALPHAS[b]),
                "n_alpha": b - a + 1} for t, a, b in rle(s["top1"])]
        gallery.append({
            "path": pi, "layer": r["layer"], "cond": "A" if r["cond"] == 0 else "B",
            "A": tok.decode([r["A"]]), "C": tok.decode([r["C"]]), "B": tok.decode([r["B"]]),
            "k_in": int(r["k_in"]), "k_out": int(r["k_out"]),
            "alpha_in": float(ALPHAS[r["k_in"]]), "alpha_out": float(ALPHAS[r["k_out"]]),
            "run_len": int(r["run_len"]), "margin_min": float(r["margin_min"]),
            "score": float(r["score"]), "clean": bool(r["clean"]),
            "rho": float(rho_c[j]), "d_mean": float(dbar[j]),
            "ctx_A": tok.decode(W[r["ia"]]), "ctx_B": tok.decode(W[r["ib"]]),
            "sequence": seq, "d": d_cand[j].tolist(), "tag": "sub-plateau"})
    make_fig(gallery, "matthew_dt_gallery.png",
             f"The clearest sub-plateaus: the 6 candidates with the flattest C window "
             f"(ρ < {RHO_FLAT}, C run ≥ 5 grid points) out of 1,290. Selected post hoc for "
             "illustration — they are the tail, not the typical candidate.")
    with open(os.path.join(RESULTS, "matthew_gallery.json"), "w") as f:
        json.dump({"rho_flat": RHO_FLAT, "n_flat": int((rho_c < RHO_FLAT).sum()),
                   "n_flat_wide": int(len(ok)), "gallery": gallery}, f, indent=1)

    # ---------------------------------------------------------------- figure 3: dwell statistics
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.linspace(0, 6, 31)
    a1.hist(np.clip(rho_c, 0, 6), bins=bins, histtype="step", lw=2.2, color=CVD[0], ls="-",
            density=True, label=f"C runs of A|C|B candidates (solid), n={len(rho_c)}")
    a1.hist(np.clip(rho_x, 0, 6), bins=bins, histtype="step", lw=2.2, color=CVD[1], ls="--",
            density=True, label=f"same α window, non-candidate paths (dashed), n={len(rho_x)}")
    a1.axvline(1.0, label="ρ = 1: as steep as the diagonal", **REF_DIAG)
    a1.axvline(RHO_FLAT, label=f"ρ = {RHO_FLAT}: sub-plateau line", **REF_RULE)
    a1.set_xlabel("flatness ρ of the C window  (range of d ÷ width in t)")
    a1.set_ylabel("density")
    a1.set_title("Is the third-token region flat in output space?", fontsize=10)
    a1.legend(fontsize=7.5)

    q = np.quantile(score, np.linspace(0, 1, 11))
    mids, meds, los, his = [], [], [], []
    for i in range(10):
        m = (score >= q[i]) & (score <= q[i + 1])
        if m.sum() < 5:
            continue
        mids.append(i + 1)
        v = rho_c[m]
        meds.append(np.median(v))
        los.append(np.percentile(v, 25))
        his.append(np.percentile(v, 75))
    a2.plot(mids, meds, color=CVD[0], ls="-", marker="o", lw=2, label="median ρ (solid, circles)")
    a2.fill_between(mids, los, his, color=CVD[0], alpha=0.18, hatch="//", ec=CVD[0], lw=0,
                    label="inter-quartile range (hatched)")
    a2.axhline(1.0, label="ρ = 1 (diagonal)", **REF_DIAG)
    a2.axhline(RHO_FLAT, label=f"ρ = {RHO_FLAT}", **REF_RULE)
    a2.set_xlabel("decile of the frozen candidate score (10 = highest-scoring)")
    a2.set_ylabel("flatness ρ")
    a2.set_title("Does the pre-frozen score find the flat ones?", fontsize=10)
    a2.legend(fontsize=7.5)

    a3.hist(dbar, bins=np.linspace(0, 1, 26), histtype="stepfilled", color=CVD[0], alpha=0.35,
            hatch="//", ec=CVD[0], lw=1.6, label="mean d over the C run (hatched)")
    a3.axvline(0.0, color="0.3", ls="-", lw=1.2)
    a3.axvline(1.0, color="0.3", ls="-", lw=1.2)
    a3.annotate("endpoint A", (0.02, 0.92), xycoords="axes fraction", fontsize=8)
    a3.annotate("endpoint B", (0.98, 0.92), xycoords="axes fraction", fontsize=8, ha="right")
    a3.set_xlabel("mean d(t) across the C run")
    a3.set_ylabel("candidate paths")
    a3.set_title("Where does the C region sit between the endpoints?", fontsize=10)
    a3.legend(fontsize=7.5)
    for ax in (a1, a2, a3):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "subplateau_dwell.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"n_flat": int((rho_c < RHO_FLAT).sum()),
                      "n_flat_wide": int(len(ok)),
                      "median_rho_top_decile": float(meds[-1]),
                      "median_rho_bottom_decile": float(meds[0]),
                      "gallery_paths": [g["path"] for g in gallery],
                      "gallery_rho": [round(g["rho"], 3) for g in gallery]}, indent=1))


if __name__ == "__main__":
    main()
