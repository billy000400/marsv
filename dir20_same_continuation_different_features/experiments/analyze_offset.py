"""S13 analysis: plateau strength when the differing token is no longer the last token."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
MSTYLE = {"gpt2-small": dict(color=CVD[0], marker="o", ls="-"),
          "gpt2-medium": dict(color=CVD[1], marker="s", ls="--"),
          "gpt2-large": dict(color=CVD[2], marker="^", ls=":")}
MODELS = ["gpt2-small", "gpt2-medium", "gpt2-large"]
SUFFIX_LENS = [0, 1, 2, 4]
BOOT = 5000
RNG = np.random.default_rng(0)


def boot_ci(x, stat=np.median):
    idx = RNG.integers(0, len(x), size=(BOOT, len(x)))
    b = np.sort(stat(np.asarray(x)[idx], axis=1))
    return float(b[int(0.025 * BOOT)]), float(b[int(0.975 * BOOT)])


def main():
    out, curves = {}, {}
    for mk in MODELS:
        p = os.path.join(RESULTS, f"offset_{mk}.json")
        if not os.path.exists(p):
            print("missing", p)
            continue
        rows = json.load(open(p))
        Z = np.load(os.path.join(RESULTS, f"offset_{mk}.npz"))
        alphas = Z["alphas"]
        by = {s: {r["pair"]: r for r in rows if r["s"] == s} for s in SUFFIX_LENS}
        pairs = sorted(set(by[0]) & set(by[SUFFIX_LENS[-1]]))
        m = {}
        for s in SUFFIX_LENS:
            w = np.array([by[s][k]["wtv"] for k in pairs])
            w0 = np.array([by[0][k]["wtv"] for k in pairs])
            sharp = np.array([by[s][k]["w"] is not None and by[s][k]["w"] < 0.5 for k in pairs])
            jf = np.array([by[s][k]["jsd_final"] for k in pairs])
            sep = np.array([by[s][k]["sep"] for k in pairs])
            err = np.array([max(by[s][k]["endpoint_err"]) for k in pairs])
            dlt = w - w0
            m[s] = dict(
                n=len(pairs), wtv_med=float(np.median(w)), wtv_ci=boot_ci(w),
                sharp_pct=float(100 * sharp.mean()),
                jsd_med=float(np.median(jf)), jsd_ci=boot_ci(jf),
                sep_med=float(np.median(sep)),
                delta_med=float(np.median(dlt)), delta_ci=boot_ci(dlt),
                p=None if s == 0 else float(wilcoxon(dlt).pvalue),
                max_endpoint_err=float(err.max()),
                mono_pct=float(100 * np.mean([by[s][k]["mono"] for k in pairs])))
            curves[(mk, s)] = np.stack([Z[f"{k}_s{s}"] for k in pairs]).mean(0)
        out[mk] = m
        print(mk, {s: (round(m[s]["wtv_med"], 3), round(m[s]["sharp_pct"], 1),
                       round(m[s]["jsd_med"], 4)) for s in SUFFIX_LENS})

    with open(os.path.join(RESULTS, "offset_analysis.json"), "w") as f:
        json.dump({k: {str(s): v for s, v in d.items()} for k, d in out.items()}, f, indent=1)

    have = [mk for mk in MODELS if mk in out]
    fig, ax = plt.subplots(1, 3, figsize=(15.6, 4.6))

    for mk in have:
        st = MSTYLE[mk]
        y = [out[mk][s]["wtv_med"] for s in SUFFIX_LENS]
        lo = [out[mk][s]["wtv_ci"][0] for s in SUFFIX_LENS]
        hi = [out[mk][s]["wtv_ci"][1] for s in SUFFIX_LENS]
        ax[0].errorbar(SUFFIX_LENS, y, yerr=[np.array(y) - lo, np.array(hi) - y],
                       capsize=3, lw=2, ms=7, label=mk, **st)
        ax[1].plot(SUFFIX_LENS, [out[mk][s]["jsd_med"] for s in SUFFIX_LENS],
                   lw=2, ms=7, label=mk, **st)
    ax[0].axhline(0.5, color="0.45", ls=(0, (4, 3)), lw=1.4)
    ax[0].text(0.05, 0.505, "linear response $w_{TV}=0.5$", color="0.3", fontsize=8, va="bottom")
    ax[0].set_xlabel("suffix length $s$ (tokens after the differing token)")
    ax[0].set_ylabel("median $w_{TV}$  (smaller = sharper switch)")
    ax[0].set_title("A. Plateau strength is unchanged by the suffix")
    ax[0].set_ylim(0, 0.56)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("suffix length $s$")
    ax[1].set_ylabel("median endpoint JSD at the final position (nats)")
    ax[1].set_title("B. ...although the endpoints nearly merge")

    for mk in have:
        st = MSTYLE[mk]
        for s, a in [(0, 1.0), (4, 0.45)]:
            ax[2].plot(np.linspace(0, 1, len(curves[(mk, s)])), curves[(mk, s)],
                       color=st["color"], ls="-" if s == 0 else (0, (1, 1)), lw=2, alpha=a,
                       label=f"{mk}, $s$={s}")
    ax[2].plot([0, 1], [0, 1], color="0.45", ls=(0, (4, 3)), lw=1.4, label="linear $d=\\alpha$")
    ax[2].set_xlabel("interpolation position $\\alpha$")
    ax[2].set_ylabel("mean relative distance $d(\\alpha)$")
    ax[2].set_title("C. Mean switch curve, $s=0$ (solid) vs $s=4$ (dotted)")

    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7.5)
    for a in ax[:2]:
        a.set_xticks(SUFFIX_LENS)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "offset_position.png"), dpi=140)
    plt.close(fig)
    print("wrote plots/offset_position.png")

    hdr = f"{'model':13s} {'s':>2s} {'n':>4s} {'med w_TV':>9s} {'%sharp':>7s} " \
          f"{'JSD_final':>10s} {'paired d':>9s} {'95% CI':>18s} {'p':>9s} {'max err':>8s}"
    print(hdr)
    for mk in have:
        for s in SUFFIX_LENS:
            e = out[mk][s]
            ci = f"[{e['delta_ci'][0]:+.3f},{e['delta_ci'][1]:+.3f}]"
            pp = "-" if e["p"] is None else f"{e['p']:.1e}"
            print(f"{mk:13s} {s:2d} {e['n']:4d} {e['wtv_med']:9.3f} {e['sharp_pct']:7.1f} "
                  f"{e['jsd_med']:10.4f} {e['delta_med']:+9.3f} {ci:>18s} {pp:>9s} "
                  f"{e['max_endpoint_err']:8.1e}")


if __name__ == "__main__":
    main()
