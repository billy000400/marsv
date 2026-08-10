"""Figure for the forward screen on unseen tokens."""
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


def main():
    f = json.load(open(f"{RESULTS}/forward.json"))
    ok = [r for r in f["rows"] if not np.isnan(r["w"]) and r["n_valid"] >= 2
          and r["out_jsd_min"] >= 0.2]
    y = np.array([r["w"] for r in ok])
    p = np.array([r["pred"] for r in ok])

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9),
                           gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax[0].scatter(p, y, s=11, marker="o", c=CVD[0], alpha=.4, linewidths=0)
    lim = [min(p.min(), y.min()) - .02, max(p.max(), y.max()) + .02]
    ax[0].plot(lim, lim, ls="--", c="0.3", lw=1)
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    ax[0].set_xlabel(r"predicted $w$ from the two tokens' anchor widths")
    ax[0].set_ylabel("observed $w$ of the pair")
    ax[0].set_title(f"{len(ok)} pairs of {f['n_new_tokens']} unseen tokens\n"
                    f"$R^2$ = {f['r2_forward']:.3f}   Spearman $\\rho$ = "
                    f"{f['rho_forward'][0]:+.2f}   MAE = {f['mean_abs_err']:.3f}", fontsize=9)

    q = np.quantile(p, [1 / 3, 2 / 3])
    groups = [y[p <= q[0]], y[(p > q[0]) & (p <= q[1])], y[p > q[1]]]
    bp = ax[1].boxplot(groups, patch_artist=True, widths=.6,
                       tick_labels=["narrowest", "middle", "widest"])
    for patch, col, h in zip(bp["boxes"], [CVD[0], CVD[3], CVD[1]], ["//", "..", "\\\\"]):
        patch.set_facecolor(col)
        patch.set_alpha(.75)
        patch.set_hatch(h)
    for med in bp["medians"]:
        med.set_color("0.1")
    ax[1].set_xlabel("tercile of the screen's prediction (before running the pair)")
    ax[1].set_ylabel("observed $w$")
    ax[1].set_title("The screen separates narrow from wide\n"
                    f"median $w$: {np.median(groups[0]):.2f} / {np.median(groups[1]):.2f} / "
                    f"{np.median(groups[2]):.2f}", fontsize=9)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(PLOTS, "forward_screen.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/forward_screen.png")


if __name__ == "__main__":
    main()
