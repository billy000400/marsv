"""The advisor's hypothesis: holding output JSD LOW, do prompts that engage different internal
features rest on different (intermediate) plateaus?

Everything else in this direction correlated endpoint JSD with transition WIDTH, which tests a
different claim. Here output divergence is a control held low, the independent variable is a measure
of how differently the two prompts are represented inside the model, and the dependent variable is
whether the interpolation rests at an intermediate level rather than switching straight from A to B.

Writes results/feature_plateau.json and plots/feature_plateau.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from analyze_bank import cluster_boot
from common import PLOTS, RESULTS, blocks, load
from mine_pairs import MAX_RANK, N_PARTNER, SEED, get_prefixes
from run_interp import clean_run

MODELS = ["gpt2-large", "gpt2-medium"]
JSD_LOW = 0.10          # "holding output JSD low"
FLAT_TOL = 0.10         # a plateau is a window whose d varies by at most this
MID_LO, MID_HI = 0.15, 0.85   # intermediate = away from both endpoints
IPW_LINEAR = FLAT_TOL   # a linear response d = alpha gives a window of exactly this width
IPW_THR = 2 * IPW_LINEAR      # call it an intermediate plateau at twice the linear baseline
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
STYLE = {"gpt2-large": dict(color=CVD[0], marker="o"),
         "gpt2-medium": dict(color=CVD[1], marker="s")}


def intermediate_plateau_width(alphas, d):
    """Longest alpha-span over which d stays flat at a level away from both endpoints.

    Returns (width, level). A linear response gives IPW_LINEAR by construction.
    """
    best, lvl = 0.0, np.nan
    n = len(d)
    for i in range(n):
        for j in range(i + 1, n + 1):
            seg = d[i:j]
            if seg.max() - seg.min() > FLAT_TOL:
                break
            m = seg.mean()
            if MID_LO <= m <= MID_HI:
                w = alphas[j - 1] - alphas[i]
                if w > best:
                    best, lvl = w, float(m)
    return float(best), lvl


def internal_distance(reca, recb):
    """Mean over blocks of 1 - cos(h_A, h_B) at the final token: how differently the two prompts
    are represented inside the model, independently of what they predict."""
    vals = []
    for i in sorted(reca):
        a, b = reca[i], recb[i]
        vals.append(1.0 - float((a * b).sum() / (a.norm() * b.norm())))
    return float(np.mean(vals)), float(np.max(vals))


def main():
    rng_boot = np.random.default_rng(0)
    out = {}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    for mi, mkey in enumerate(MODELS):
        tok, m = load(mkey)
        dev = next(m.parameters()).device
        rng = np.random.default_rng(SEED)
        prefixes = get_prefixes(tok, rng)
        bank = {r["key"]: r for r in
                json.load(open(os.path.join(RESULTS, f"bank_{mkey}.json")))}
        Z = np.load(os.path.join(RESULTS, f"bank_{mkey}.npz"))
        alphas = Z["alphas"]

        rows = []
        for pi, pre in enumerate(prefixes):
            with torch.no_grad():
                lg = m(torch.tensor([pre], device=dev), use_cache=False).logits[0, -1, :].float()
            order = torch.argsort(lg, descending=True)
            ranks = np.unique(np.round(
                np.exp(rng.uniform(np.log(1), np.log(MAX_RANK), size=N_PARTNER * 2))).astype(int))
            ranks = ranks[:N_PARTNER]
            ta = int(order[0])
            for r in ranks:
                key = f"p{pi}_r{int(r)}"
                if key not in bank or bank[key]["jsd"] >= JSD_LOW:
                    continue
                tb = int(order[int(r)])
                reca, _ = clean_run(m, pre + [ta])
                recb, _ = clean_run(m, pre + [tb])
                ird, ird_max = internal_distance(reca, recb)
                ipw, lvl = intermediate_plateau_width(alphas, Z[key])
                rows.append(dict(key=key, prefix_idx=pi, jsd=bank[key]["jsd"],
                                 wtv=bank[key]["wtv"], ird=ird, ird_max=ird_max,
                                 ipw=ipw, level=lvl))
        del m
        torch.cuda.empty_cache()

        ipw = np.array([r["ipw"] for r in rows])
        ird = np.array([r["ird"] for r in rows])
        rho, p = spearmanr(ird, ipw)
        cl = {}
        for r in rows:
            cl.setdefault(r["prefix_idx"], []).append(r)
        lo, hi = cluster_boot(
            cl, lambda rr: (spearmanr([q["ird"] for q in rr], [q["ipw"] for q in rr])[0]
                            if len({q["ird"] for q in rr}) > 2 else None), rng_boot)
        out[mkey] = dict(n=len(rows), n_prefix=len(cl),
                         jsd_max=float(max(r["jsd"] for r in rows)),
                         frac_intermediate=float(np.mean(ipw > IPW_THR)),
                         median_ipw=float(np.median(ipw)), ipw_linear=IPW_LINEAR,
                         rho_ird_ipw=float(rho), p_ird_ipw=float(p), ci=[lo, hi],
                         median_ird=float(np.median(ird)), rows=rows)
        print(mkey, json.dumps({k: v for k, v in out[mkey].items() if k != "rows"}))

        ax[mi].scatter(ird, ipw, s=42, alpha=0.75, edgecolor="k", linewidth=0.5,
                       **STYLE[mkey])
        ax[mi].axhline(IPW_LINEAR, color="0.35", ls=(0, (4, 3)), lw=1.4)
        ax[mi].axhline(IPW_THR, color="0.1", ls=":", lw=1.4)
        ax[mi].margins(y=0.18)
        ax[mi].text(0.99, 0.03, "dashed = linear response (0.10)\ndotted = intermediate-plateau"
                    " threshold (0.20)", fontsize=7.5, color="0.15", ha="right", va="bottom",
                    transform=ax[mi].transAxes)
        ax[mi].set_title(f"{mkey} — JSD < {JSD_LOW} pairs, "
                         f"$\\rho$={rho:+.2f} (p={p:.2f}, n={len(rows)})", fontsize=9.5)
        ax[mi].set_xlabel("internal representational distance $\\mathrm{IRD}$")
        ax[mi].set_ylabel("intermediate-plateau width $\\mathrm{IPW}$")
        ax[mi].grid(alpha=0.3)

    # third panel: the two Matthew pairs in gpt2-large, as reference curves
    S = json.load(open(os.path.join(RESULTS, "summary.json")))
    Zl = np.load(os.path.join(RESULTS, "gpt2-large.npz"))
    al = Zl["alphas"]
    for i, (k, lab, ls, mk) in enumerate([
            ("house_big_in", "big / in (Matthew's plateau example)", "-", "o"),
            ("house_big_large", "big / large (Matthew's smooth comparison)", "--", "s")]):
        ax[2].plot(al, Zl[f"{k}__logits"], color=CVD[i], ls=ls, marker=mk, markevery=10,
                   ms=5, lw=2, label=f"{lab}\nJSD={S['gpt2-large'][k]['jsd']:.2f}, "
                                     f"$w_{{10\\text{{-}}90}}$={S['gpt2-large'][k]['w10_90_logits']:.2f}")
    ax[2].plot(al, al, color="0.55", lw=1.2, ls=(0, (4, 3)), label="linear $d=\\alpha$")
    ax[2].set_title("gpt2-large — the two pairs Matthew contrasts", fontsize=9.5)
    ax[2].set_xlabel("interpolation position $\\alpha$")
    ax[2].set_ylabel("relative distance $d$")
    ax[2].legend(fontsize=7, loc="center right")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "feature_plateau.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(RESULTS, "feature_plateau.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote feature_plateau.json + plots/feature_plateau.png")


if __name__ == "__main__":
    main()
