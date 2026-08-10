"""S3: figures + the JSD-vs-plateau comparison table."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import PAIRS, PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00", "#555555"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
MODELS = ["gpt2-large", "gpt2-medium", "gpt2-small", "opt-350m", "pythia-410m"]
LS = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1))]
MK = ["o", "s", "^", "D", "v", "P"]


def plateau_fraction(d):
    """Fraction of the alpha grid where d sits within 0.1 of an endpoint value."""
    return float(np.mean((d < 0.1) | (d > 0.9)))


def tv_width(alphas, d, lo=0.25, hi=0.75):
    """Alpha span carrying the middle 50% of the curve's total variation.

    Threshold-free and well defined for non-monotonic curves: 0.5 for a linear response,
    -> 0 for a step. Complements w10-90, which a dip below 0.1 can inflate.
    """
    c = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(d)))])
    c = c / c[-1]
    return float(np.interp(hi, c, alphas) - np.interp(lo, c, alphas))


def main():
    S = json.load(open(os.path.join(RESULTS, "summary.json")))
    Z = {m: np.load(os.path.join(RESULTS, f"{m}.npz")) for m in MODELS}
    alphas = Z[MODELS[0]]["alphas"]
    keys = [p[0] for p in PAIRS]
    labels = {p[0]: p[1] for p in PAIRS}
    ctrl = {p[0]: p[5] for p in PAIRS}

    # ---- Figure 1: raw final-logit curves -------------------------------------------------
    fig, ax = plt.subplots(len(keys), len(MODELS), figsize=(19, 15), sharex=True, sharey=True)
    for r, k in enumerate(keys):
        for c, m in enumerate(MODELS):
            a = ax[r, c]
            d = Z[m][f"{k}__logits"]
            a.plot(alphas, alphas, color="0.55", lw=1, ls=(0, (4, 3)), label="linear $d=\\alpha$")
            a.plot(alphas, d, color=CVD[0], lw=2, marker=MK[0], markevery=10, ms=4,
                   label="measured $d(\\alpha)$")
            e = S[m][k]
            w = e["w10_90_logits"]
            tag = "" if e["monotonic"] else "  [non-monotonic]"
            a.set_title(f"{m} - {labels[k]}\nJSD={e['jsd']:.3f} nats, "
                        f"$w_{{10\\text{{-}}90}}$={w:.2f}{tag}", fontsize=8)
            a.set_ylim(-0.03, 1.03)
            a.grid(alpha=0.3)
            if ctrl[k]:
                for s in a.spines.values():
                    s.set_linewidth(2.0)
            if r == len(keys) - 1:
                a.set_xlabel("interpolation position $\\alpha$")
            if c == 0:
                a.set_ylabel("relative distance $d$")
    ax[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "final_logit_curves.png"), dpi=140)
    plt.close(fig)

    # ---- Figure 2: layerwise widths -------------------------------------------------------
    fig, ax = plt.subplots(1, len(MODELS), figsize=(19, 4.2), sharey=True)
    for c, m in enumerate(MODELS):
        for i, k in enumerate(keys):
            e = S[m][k]
            lay = e["layers"] + [e["layers"][-1] + 1]
            w = [np.nan if x is None else x for x in e["w10_90_layers"]] + [e["w10_90_logits"]]
            ax[c].plot(lay, w, color=CVD[i], ls=LS[i], marker=MK[i], ms=3.5, lw=1.6,
                       label=labels[k])
        ax[c].axhline(0.8, color="0.55", ls=(0, (4, 3)), lw=1)
        ax[c].text(1.2, 0.81, "linear response (0.8)", fontsize=7, color="0.35")
        ax[c].axhline(0.5, color="0.2", ls=":", lw=1)
        ax[c].text(1.2, 0.51, "plateau threshold (0.5)", fontsize=7, color="0.2")
        ax[c].set_title(m, fontsize=10)
        ax[c].set_xlabel("recording block (last = final logits)")
        ax[c].grid(alpha=0.3)
    ax[0].set_ylabel("$w_{10\\text{-}90}$")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "layerwise_widths.png"), dpi=140)
    plt.close(fig)

    # ---- Figure 3: the test -- endpoint JSD vs transition sharpness -----------------------
    rows = []
    for m in MODELS:
        for k in keys:
            e = S[m][k]
            d = Z[m][f"{k}__logits"]
            rows.append(dict(model=m, pair=k, label=labels[k], control=ctrl[k], jsd=e["jsd"],
                             w=e["w10_90_logits"], wtv=tv_width(alphas, d),
                             pf=plateau_fraction(d), mono=e["monotonic"], cos=e["cos_h0"]))

    METS = [("w", "$w_{10\\text{-}90}$ at final logits", 0.8, 0.5),
            ("wtv", "$w_{\\mathrm{TV}}$ at final logits", 0.5, 0.25)]
    fig, ax = plt.subplots(2, len(MODELS), figsize=(19, 8), sharex=True)
    for r, (mt, ylab, lin, thr) in enumerate(METS):
        for c, m in enumerate(MODELS):
            sub = [x for x in rows if x["model"] == m]
            for i, k in enumerate(keys):
                x = [q for q in sub if q["pair"] == k][0]
                ax[r, c].scatter([x["jsd"]], [x[mt]], color=CVD[i], marker=MK[i], s=75,
                                 edgecolor="k", linewidth=1.8 if x["control"] else 0.6,
                                 zorder=3, label=labels[k])
            rho, p = spearmanr([q["jsd"] for q in sub], [q[mt] for q in sub])
            ax[r, c].axhline(lin, color="0.55", ls=(0, (4, 3)), lw=1)
            ax[r, c].text(0.70, lin * 1.02, "linear response", fontsize=7, color="0.35",
                          ha="right")
            ax[r, c].axhline(thr, color="0.2", ls=":", lw=1)
            ax[r, c].text(0.70, thr * 1.03, "plateau threshold", fontsize=7, color="0.2",
                          ha="right")
            ax[r, c].set_title(f"{m} - Spearman $\\rho$={rho:+.2f} (p={p:.2f}, n=6)", fontsize=9)
            ax[r, c].set_xlim(-0.02, 0.72)
            ax[r, c].set_ylim(0, lin * 1.15)
            ax[r, c].grid(alpha=0.3)
            if r == 1:
                ax[r, c].set_xlabel("endpoint JSD (nats)")
        ax[r, 0].set_ylabel(ylab)
    ax[0, 1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "jsd_vs_width.png"), dpi=140)
    plt.close(fig)

    # pooled correlations over all 10 (pair, model) cells
    aj = [r["jsd"] for r in rows]
    pooled = {f"pooled_jsd_vs_{n}": dict(rho=float(spearmanr(aj, [r[n] for r in rows])[0]),
                                         p=float(spearmanr(aj, [r[n] for r in rows])[1]),
                                         n=len(rows))
              for n in ["w", "wtv", "pf"]}
    per_model = {}
    for m in MODELS:
        sub = [x for x in rows if x["model"] == m]
        per_model[m] = {n: dict(rho=float(spearmanr([q["jsd"] for q in sub], [q[n] for q in sub])[0]),
                                p=float(spearmanr([q["jsd"] for q in sub], [q[n] for q in sub])[1]))
                        for n in ["w", "wtv", "pf"]}
    out = dict(rows=rows, per_model=per_model, **pooled)
    with open(os.path.join(RESULTS, "analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    for r in rows:
        print(f"{r['model']:12s} {r['pair']:14s} JSD={r['jsd']:.3f} w={r['w']:.3f} "
              f"wTV={r['wtv']:.3f} PF={r['pf']:.3f} mono={r['mono']} cos={r['cos']:.3f}")
    print(json.dumps(pooled, indent=2))
    print(json.dumps(per_model, indent=2))


if __name__ == "__main__":
    main()
