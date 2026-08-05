"""Feedback #4 figure: do the named reference pairs plateau in Pythia?

Reads results/reference_house.json (+ results/reference_jsd.json) and writes
plots/house_reference.png and results/reference_summary.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS
from curve_metrics import E_LINEAR

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
STYLE = {" big| large": dict(color=CVD[0], ls="-", marker="o", label="` big` / ` large`"),
         " big| in": dict(color=CVD[1], ls="--", marker="s", label="` big` / ` in`")}
HOUSE = "My house is"


def curve(ref, run, pair, carrier):
    return ref["runs"][run]["curves"][f"{pair}@@{carrier}"]


if __name__ == "__main__":
    ref = json.load(open(os.path.join(RESULTS, "reference_house.json")))
    cj = json.load(open(os.path.join(RESULTS, "reference_jsd.json")))
    t = np.array(ref["grid"])
    pairs = list(STYLE)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))

    # (a) relative-logit coordinate, trained 1.4B, the operator's carrier
    a = ax[0, 0]
    a.plot(t, t, color="0.55", ls=":", lw=1.5, label="no-plateau reference $d(t)=t$")
    for p in pairs:
        r = curve(ref, "1.4B trained", p, HOUSE)
        a.plot(t, r["d"], lw=2, ms=4, markevery=4,
               **{**STYLE[p], "label": f"{STYLE[p]['label']}  w={r['w']:.2f}"})
    for lv in (0.1, 0.9):
        a.axhline(lv, color="0.75", lw=0.8, ls=(0, (1, 3)))
    a.set_xlabel("interpolation position $t$")
    a.set_ylabel("relative-logit coordinate $d(t)$")
    a.set_title("(a) 1.4B trained — “My house is …”")
    a.legend(fontsize=8, loc="upper left")

    # (b) absolute movement of the output distribution
    b = ax[0, 1]
    for p in pairs:
        r = curve(ref, "1.4B trained", p, HOUSE)
        b.plot(t, r["m_abs"], lw=2, ms=4, markevery=4,
               **{**STYLE[p], "label": f"{STYLE[p]['label']}  M(1)={r['out_jsd']:.2f} bits"})
    b.set_xlabel("interpolation position $t$")
    b.set_ylabel("output movement $M(t)$ [bits]")
    b.set_title("(b) 1.4B trained — absolute output movement")
    b.legend(fontsize=8, loc="upper left")

    # (c) untrained control, same carrier
    c = ax[1, 0]
    c.plot(t, t, color="0.55", ls=":", lw=1.5, label="no-plateau reference $d(t)=t$")
    for p in pairs:
        r = curve(ref, "1.4B untrained", p, HOUSE)
        c.plot(t, r["d"], lw=2, ms=4, markevery=4,
               **{**STYLE[p], "label": f"{STYLE[p]['label']}  w={r['w']:.2f}"})
    for lv in (0.1, 0.9):
        c.axhline(lv, color="0.75", lw=0.8, ls=(0, (1, 3)))
    c.set_xlabel("interpolation position $t$")
    c.set_ylabel("relative-logit coordinate $d(t)$")
    c.set_title("(c) 1.4B untrained (step 0) — same prompts")
    c.legend(fontsize=8, loc="upper left")

    # (d) the two reference pairs against the 60-pair bank trend, trained 1.4B
    d = ax[1, 1]
    bank = json.load(open(os.path.join(RESULTS, "assay_step143000_t256.json")))["rows"]
    bj = np.array([r["jsd_B"] for r in bank])
    bw = np.array([r["w"] for r in bank])
    d.scatter(bj, bw, s=18, c="0.6", marker="o", label="60-pair bank (1 dot = 1 pair)")
    edges = np.quantile(bj, np.linspace(0, 1, 6))
    ctr, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (bj >= lo) & (bj <= hi)
        ctr.append(float(np.median(bj[m])))
        med.append(float(np.median(bw[m])))
    d.plot(ctr, med, color="0.25", ls="-.", marker="x", ms=8, lw=1.6, label="bank: binned medians")
    d.axhline(0.8, color="0.55", ls=":", lw=1.5)
    d.text(0.97, 0.808, "linear-response ceiling  w ≈ 0.8", fontsize=8, color="0.35", ha="right")
    for p, key in zip(pairs, [" big| large", " big| in"]):
        j = cj["pairs"][key.replace("|", "|")]["jsd_holdout"]
        ws = [curve(ref, "1.4B trained", p, ct)["w"] for ct in ref["carriers"]]
        d.plot([j, j], [min(ws), max(ws)], color=STYLE[p]["color"], lw=2.5, alpha=0.6)
        d.plot([j], [curve(ref, "1.4B trained", p, HOUSE)["w"]], ms=13, mew=2, mfc="none",
               ls="none", **{k: v for k, v in STYLE[p].items() if k != "ls"})
    d.set_xlabel(r"held-out corpus next-token JSD $\widehat J_{\mathrm{hold}}(u,v)$ [bits]")
    d.set_ylabel("transition width $w$")
    d.set_ylim(0, 0.95)
    d.set_title("(d) 1.4B trained — reference pairs vs the 60-pair bank")
    d.legend(fontsize=7.5, loc="lower left")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "house_reference.png"), dpi=150)
    plt.close(fig)

    summary = {"E_linear": E_LINEAR, "corpus_jsd": cj["pairs"],
               "splithalf_noise": cj["splithalf_noise"],
               "counts_holdout": cj["counts_holdout"], "rows": []}
    for run in ref["runs"]:
        for p in pairs:
            for ct in ref["carriers"]:
                r = curve(ref, run, p, ct)
                summary["rows"].append(dict(run=run, pair=p, carrier=ct, w=r["w"],
                                            edge_drift=r["edge_drift"], out_jsd=r["out_jsd"],
                                            m_mid=r["m_abs"][len(r["m_abs"]) // 2],
                                            valid=r["valid"], err=r["err"]))
    json.dump(summary, open(os.path.join(RESULTS, "reference_summary.json"), "w"), indent=2)
    for row in summary["rows"]:
        print(f"{row['run']:15s} {row['pair']:12s} {row['carrier']:18s} "
              f"w={row['w']:.3f} E={row['edge_drift']:.3f} M(1)={row['out_jsd']:.3f}")
    print("corpus JSD:", json.dumps(summary["corpus_jsd"]))
