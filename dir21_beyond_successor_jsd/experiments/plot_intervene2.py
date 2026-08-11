"""Figure for the behaviour-calibrated intervention: does the probe direction beat a random one?"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CVD, PLOTS, RESULTS

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9


def summarise(rows, bits):
    """Per-direction, per-bit-level summaries, computed from the raw records."""
    out = {}
    for tag in ("probe", "random"):
        recs = [x for s in rows for x in rows[s][tag]]
        out[tag] = {"by_bits": {str(b): dict(
            mean_abs_dw=float(np.mean([abs(x["dw"]) for x in recs if x["bits_target"] == b])),
            mean_dw_plus=float(np.mean([x["dw"] for x in recs
                                        if x["bits_target"] == b and x["sign"] > 0])),
            mean_dw_minus=float(np.mean([x["dw"] for x in recs
                                         if x["bits_target"] == b and x["sign"] < 0])),
            step_norm=float(np.median([x["step_norm"] for x in recs if x["bits_target"] == b])),
            bits_got=float(np.median([x["bits_got"] for x in recs if x["bits_target"] == b])))
            for b in bits}, "mean_abs_dw": float(np.mean([abs(x["dw"]) for x in recs]))}
    return out


def main():
    d = json.load(open(f"{RESULTS}/intervene2.json"))
    rows = d["tokens"]
    bits = d.get("bits", [0.05, 0.1, 0.2])
    d.update(summarise(rows, bits))
    pa_ = np.array([abs(x["dw"]) for s in rows for x in rows[s]["probe"]])
    ra_ = np.array([abs(x["dw"]) for s in rows for x in rows[s]["random"]])
    d["ratio_probe_over_random"] = float(pa_.mean() / ra_.mean())
    d["frac_probe_larger"] = float(np.mean(pa_ > ra_))

    fig, axg = plt.subplots(2, 2, figsize=(10.6, 8.0))
    ax = axg.ravel()

    # (1) how much width moves, against how much the model's output was made to move
    for k, (tag, mk, ls, lab) in enumerate([("probe", "o", "-", "probe direction"),
                                            ("random", "s", "--", "random direction")]):
        m = [d[tag]["by_bits"][str(b)]["mean_abs_dw"] for b in bits]
        pts_x, pts_y = [], []
        for s in rows:
            for x in rows[s][tag]:
                pts_x.append(x["bits_target"] * (1 + (k - .5) * .07))
                pts_y.append(abs(x["dw"]))
        ax[0].scatter(pts_x, pts_y, s=12, marker=mk, color=CVD[k], alpha=.35, edgecolor="none")
        ax[0].plot(bits, m, ls=ls, marker=mk, color=CVD[k], lw=1.8, ms=7, label=lab + " (mean)")
    ax[0].set_xscale("log")
    ax[0].set_xticks(bits)
    ax[0].set_xticklabels([f"{b:g}" for b in bits])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_xlabel("output movement the edit was calibrated to produce (bits)")
    ax[0].set_ylabel(r"$|\Delta \hat w_u|$, measured")
    ax[0].set_title("Width does move once the edit is big enough —\n"
                    "but a random direction moves it just as much", fontsize=9)
    ax[0].legend(fontsize=7, loc="upper left")

    # (2) the direct matched comparison: probe vs random at the same behavioural size
    pa = np.array([abs(x["dw"]) for s in rows for x in rows[s]["probe"]])
    ra = np.array([abs(x["dw"]) for s in rows for x in rows[s]["random"]])
    lo, hi = 0, max(pa.max(), ra.max()) * 1.08
    ax[1].plot([lo, hi], [lo, hi], ls="--", color="0.5", lw=1.0)
    for k, b in enumerate(bits):
        sel = [j for j, x in enumerate([x for s in rows for x in rows[s]["probe"]])
               if x["bits_target"] == b]
        ax[1].scatter(ra[sel], pa[sel], s=26, marker=["o", "s", "^"][k], color=CVD[k],
                      alpha=.8, edgecolor="none", label=f"{b:g} bits")
    ax[1].set_xlim(lo, hi)
    ax[1].set_ylim(lo, hi)
    ax[1].set_xlabel(r"$|\Delta \hat w_u|$ along a random direction")
    ax[1].set_ylabel(r"$|\Delta \hat w_u|$ along the probe direction")
    ax[1].set_title(f"Matched on output movement: ratio "
                    f"{d['ratio_probe_over_random']:.2f}, probe larger in "
                    f"{d['frac_probe_larger']:.0%}", fontsize=9)
    ax[1].legend(fontsize=7, loc="upper left", title="calibrated to", title_fontsize=7)

    # (3) sign: the probe says +direction widens and -direction narrows
    w = 0.21
    for k, tag in enumerate(("probe", "random")):
        for j, (key, hatch) in enumerate([("mean_dw_plus", "//"), ("mean_dw_minus", "\\\\")]):
            vals = [d[tag]["by_bits"][str(b)][key] for b in bits]
            xs = np.arange(len(bits)) + (k - .5) * 2 * w + (j - .5) * w
            ax[2].bar(xs, vals, width=w * .92, color=CVD[k], hatch=hatch, edgecolor="white",
                      lw=.6, alpha=.9,
                      label=f"{tag}, {'+' if j == 0 else '-'} direction" if True else None)
    ax[2].axhline(0, color="0.4", lw=.9)
    ax[2].set_xticks(np.arange(len(bits)))
    ax[2].set_xticklabels([f"{b:g}" for b in bits])
    ax[2].set_xlabel("output movement the edit was calibrated to (bits)")
    ax[2].set_ylabel(r"mean signed $\Delta \hat w_u$")
    ax[2].set_title("The probe predicts opposite signs for + and -;\nboth widen instead", fontsize=9)
    ax[2].legend(fontsize=6.5, loc="upper left", ncol=2)

    # (4) where the edited tokens end up: a common width, whatever the token and whatever the direction
    base = np.array([rows[s]["base_w"] for s in rows])
    big = bits[-1]
    for k, (tag, mk, lab) in enumerate([("probe", "o", "probe direction"),
                                        ("random", "s", "random direction")]):
        wn = np.array([[x["w"] for x in rows[s][tag] if x["bits_target"] == big] for s in rows])
        ax[3].scatter(np.repeat(base, wn.shape[1]), wn.ravel(), s=26, marker=mk, color=CVD[k],
                      alpha=.75, edgecolor="none", label=lab)
        ax[3].axhline(wn.mean(), ls=["-", "--"][k], color=CVD[k], lw=1.2)
    allw = [x["w"] for s in rows for x in rows[s]["probe"] + rows[s]["random"]
            if x["bits_target"] == big]
    lo2, hi2 = base.min() - .03, max(allw) + .03
    ax[3].plot([lo2, hi2], [lo2, hi2], ls=":", color="0.45", lw=1.1)
    ax[3].annotate("no change", (hi2 * .99, hi2 * .99), fontsize=6.5, ha="right", va="top",
                   color="0.35", rotation=34)
    ax[3].set_xlim(lo2, hi2)
    ax[3].set_ylim(lo2, hi2)
    ax[3].set_xlabel(r"$\hat w_u$ before the edit")
    ax[3].set_ylabel(rf"$\hat w_u$ after a {big:g}-bit edit")
    ax[3].set_title("Every edit lands the token near the same width:\n"
                    "the trait is destroyed, not shifted", fontsize=9)
    ax[3].legend(fontsize=7, loc="upper left")

    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    fig.savefig(os.path.join(PLOTS, "intervene2.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/intervene2.png")


if __name__ == "__main__":
    main()
