"""Figure for the embedding-site measurement and the static-embedding probe."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import CVD, PLOTS, RESULTS

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9


def main():
    d = json.load(open(f"{RESULTS}/embed.json"))
    w0 = {s: float(np.nanmedian(v["w"]))
          for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    names = sorted(d["w_emb"])
    ye = np.array([d["w_emb"][s] for s in names])
    y0 = np.array([w0[s] for s in names])
    yp = np.array([d["probe_pred"][s] for s in names])

    fw = json.load(open(f"{RESULTS}/embed_forward.json"))
    fwd_rows = [r for r in json.load(open(f"{RESULTS}/forward.json"))["rows"]
                if r["n_valid"] == 3 and r["out_jsd_min"] >= 0.2]

    fig, ax = plt.subplots(1, 4, figsize=(16.0, 3.8))

    ax[0].scatter(y0, ye, s=16, marker="o", color=CVD[0], alpha=.75, edgecolor="none")
    lim = [min(y0.min(), ye.min()) - .01, max(y0.max(), ye.max()) + .01]
    ax[0].plot(lim, lim, ls="--", color="0.5", lw=.9)
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    ax[0].set_xlabel(r"$\hat w_u$ measured at block 0")
    ax[0].set_ylabel(r"$\hat w_u$ measured at the input embedding")
    ax[0].set_title(f"Moving the site below block 0\n"
                    fr"$\rho = {d['emb_site']['rho_vs_block0'][0]:+.2f}$", fontsize=9)

    r = spearmanr(yp, y0)
    ax[1].scatter(yp, y0, s=16, marker="s", color=CVD[1], alpha=.75, edgecolor="none")
    ax[1].set_xlabel(r"$\hat w_u$ predicted from the static embedding (out-of-fold)")
    ax[1].set_ylabel(r"$\hat w_u$ measured at block 0")
    ax[1].set_title(f"Lookup instead of a forward pass\n"
                    fr"out-of-fold $\rho = {r[0]:+.2f}$", fontsize=9)

    keys = ["w_block0", "w_emb_site", "token_effect"]
    lab = [r"$\hat w_u$ (block 0)", r"$\hat w_u$ (embedding site)", r"fitted $a_u$"]
    x = np.arange(len(keys))
    real = [d[f"probe_{k}"]["rho_mean"] for k in keys]
    rerr = [d[f"probe_{k}"]["rho_sd"] for k in keys]
    null = [d[f"probe_{k}"]["null_rho_mean"] for k in keys]
    nerr = [d[f"probe_{k}"]["null_rho_sd"] for k in keys]
    ax[2].bar(x - .19, real, .36, yerr=rerr, capsize=3, color=CVD[0], hatch="//",
              edgecolor="white", label="probe")
    ax[2].bar(x + .19, null, .36, yerr=nerr, capsize=3, color=CVD[2], hatch="..",
              edgecolor="white", label="shuffled-target control")
    ax[2].axhline(0, color="0.5", lw=.8)
    for xi, v, e in zip(x, real, rerr):
        ax[2].annotate(f"{v:+.2f}", (xi - .19, v + e), fontsize=7, ha="center",
                       xytext=(0, 4), textcoords="offset points")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(lab, fontsize=8)
    ax[2].set_ylabel(r"held-out Spearman $\rho$ (43 test tokens)")
    ax[2].set_title(f"Probe from $W_E$ row, {d['n_split']} random splits", fontsize=9)
    ax[2].axhline(fw_norm := fw["probe_norm_only"]["rho_mean"], ls=":", color="0.35", lw=1.2)
    ax[2].annotate(f"embedding norm only ({fw_norm:+.2f})", (0.52, fw_norm), fontsize=6.5,
                   ha="center", va="bottom", color="0.25")
    ax[2].legend(fontsize=7, loc="lower left")

    obs = np.array([r["w"] for r in fwd_rows])
    pred = np.array([fw["beta0"] + fw["beta1"] * (fw["w_lookup"][r["a"]] + fw["w_lookup"][r["b"]])
                     for r in fwd_rows])
    ax[3].scatter(pred, obs, s=9, marker="^", color=CVD[3], alpha=.55, edgecolor="none")
    lim = [min(pred.min(), obs.min()) - .02, max(pred.max(), obs.max()) + .02]
    ax[3].plot(lim, lim, ls="--", color="0.5", lw=.9)
    ax[3].set_xlim(lim)
    ax[3].set_ylim(lim)
    ax[3].set_xlabel(r"$\hat w$ from static embeddings only (no forward pass)")
    ax[3].set_ylabel("observed transition width $w$ of the unseen pair")
    ax[3].set_title(f"{fw['n_scored']} pairs of {fw['n_new_tokens']} unseen tokens\n"
                    fr"$R^2 = {fw['r2_forward']:.2f}$, $\rho = {fw['rho_forward'][0]:+.2f}$ "
                    f"(measured screen {fw['measured_screen']['r2']:.2f})", fontsize=9)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(PLOTS, "embed_probe.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/embed_probe.png")


if __name__ == "__main__":
    main()
