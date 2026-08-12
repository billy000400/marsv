"""Figure: the one open cell refitted on 250 tokens, and the checks that license reading it.

Top row, one panel per target: held-out accuracy of the two readouts at block 6 in each sample, each
against the permutation null drawn at that sample's own test-half size. Bottom row: whether the 127
new tokens land in the same range as the original 123 on each measured target, and whether the null
width still follows the 0.572/sqrt(n_test) rule fitted on the 123-token learning curve.

Writes plots/pool.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00"]
STYLE = [("ridge", "-", "o"), ("krr", "--", "s")]
LABEL = {"ridge": "linear ridge", "krr": "RBF kernel ridge"}
ROWS = ["width_given_shape", "width", "shape"]
TITLE = {"width_given_shape": "width, shape removed\n(the open question)",
         "width": "width  $w_u$  (control)", "shape": "shape  $E_u$  (control)"}
CFG = [("old", "80"), ("new", "80"), ("all", "80"), ("all", "125"), ("group_control", None)]
TICK = {("old", "80"): "original 123\n80 train\n43 test",
        ("new", "80"): "new 127\n80 train\n47 test",
        ("all", "80"): "all 250\n80 train\n170 test",
        ("all", "125"): "all 250\n125 train\n125 test",
        ("group_control", None): "all 250\n125 train, 125 test\ntarget z-scored\nwithin each sample"}


def cell(d, t, s, r, key):
    return d[t][r][key] if s is None else d[t][s][r][key]


def panel(ax, d, key):
    x = np.arange(len(CFG))
    lo = [min(cell(d, t, s, r, key)["null_mean"] - 2 * cell(d, t, s, r, key)["null_sd"]
              for r, _, _ in STYLE) for t, s in CFG]
    hi = [max(cell(d, t, s, r, key)["null_mean"] + 2 * cell(d, t, s, r, key)["null_sd"]
              for r, _, _ in STYLE) for t, s in CFG]
    ax.fill_between(x, lo, hi, color="0.85", edgecolor="0.45", hatch="//", lw=0.8, zorder=1,
                    label="permutation null, $\\pm$2 s.d.")
    for (r, ls, mk), c in zip(STYLE, CVD):
        m = [cell(d, t, s, r, key)["rho_mean"] for t, s in CFG]
        e = [cell(d, t, s, r, key)["rho_sd"] for t, s in CFG]
        ax.errorbar(x, m, yerr=e, ls=ls, marker=mk, ms=5, lw=1.6, capsize=3, color=c, zorder=3,
                    label=LABEL[r])
    ceil = [cell(d, t, s, "ridge", key)["ceiling"] for t, s in CFG[:-1]]
    ax.plot(x[:-1], ceil, ls="-.", lw=1.1, color="0.2", marker="_", ms=9, zorder=2,
            label="ceiling $\\sqrt{R}$")
    ax.axhline(0, color="0.3", lw=0.8, ls=":")
    ax.set_xticks(x, [TICK[c] for c in CFG], fontsize=7)
    ax.set_title(TITLE[key], fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)


def ecdf(ax, d, y_old, y_new, key, xlabel):
    for v, lab, ls, c in ((y_old, f"original {len(y_old)} tokens", "-", CVD[0]),
                          (y_new, f"new {len(y_new)} tokens", "--", CVD[1])):
        s = np.sort(v)
        ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", ls=ls, lw=1.8, color=c, label=lab)
    p = d["sample_comparison"][key]["mannwhitney_p"]
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("fraction of tokens at or below", fontsize=8.5)
    ax.set_title(f"{key} in the two samples (Mann-Whitney $p$ = {p:.3f})", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)


def price(ax, d):
    p = d["price_check"]
    n = np.array([r["n_test"] for r in p], float)
    sd = np.array([r["null_sd"] for r in p])
    grid = np.linspace(n.min() * 0.9, n.max() * 1.05, 60)
    ax.plot(grid, d["price_c"] / np.sqrt(grid), color="0.25", ls="-.", lw=1.3,
            label=f"$c/\\sqrt{{n_{{\\rm test}}}}$, $c$ = {d['price_c']},\nfitted on the 123-token run")
    for (r, _, mk), c in zip(STYLE, CVD):
        k = [i for i, row in enumerate(p) if row["readout"] == r]
        ax.plot(n[k], sd[k], mk, ms=6, color=c, ls="none", label=LABEL[r])
    ax.set_xlabel("test-half size  $n_{\\rm test}$  (tokens)", fontsize=9)
    ax.set_ylabel("s.d. of the permutation null", fontsize=8.5)
    ax.set_title("does the null shrink as the 123-token run predicted?", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)


def main():
    d = json.load(open(f"{RESULTS}/pool.json"))
    y = {k: (np.array([d["target_by_token"][k][s] for s in d["old"]["tokens"]]),
             np.array([d["target_by_token"][k][s] for s in d["new"]["tokens"]]))
         for k in ("shape", "width")}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    for j, key in enumerate(ROWS):
        panel(axes[0, j], d, key)
    axes[0, 0].set_ylabel("held-out Spearman $\\rho$", fontsize=9)
    axes[0, 0].legend(loc="upper left", fontsize=7, framealpha=0.95)
    ecdf(axes[1, 0], d, *y["shape"], "shape", "shape  $E_u$  (median edge drift)")
    ecdf(axes[1, 1], d, *y["width"], "width", "width  $w_u$")
    price(axes[1, 2], d)
    fig.suptitle("Pythia-1.4B, residual stream after block 6: the one open cell refitted on 250 "
                 "endpoint tokens", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(f"{PLOTS}/pool.png", dpi=170)
    plt.close(fig)
    print(f"wrote {PLOTS}/pool.png")


if __name__ == "__main__":
    main()
