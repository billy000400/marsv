"""S5 — analysis + figures for the pilot pair sweep.

Reads results/sweep_<pair>.npy + weekday_setup.npz. Computes the activation-path
recovery metric (mean nearest-spline distance in PCA-32), and produces the four
required figures: recovery vs lambda, energy trade-off, d(t) curves, PCA geometry.
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import common as C


def dense_arc(centroids, a_idx, n=600):
    t = np.arange(7)
    cs = CubicSpline(np.append(t, 7), np.vstack([centroids, centroids[0]]),
                     bc_type="periodic", axis=0)
    ts = np.linspace(a_idx, a_idx + 1, n)
    return cs(ts)[:, :32]                       # [n, 32]


def recovery(W, arc):
    """Mean over waypoints of nearest-point L2 distance to the dense spline arc."""
    d = np.linalg.norm(W[:, None, :] - arc[None, :, :], axis=2)   # [n_wp, n]
    return float(d.min(axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="Tuesday-Wednesday")
    args = ap.parse_args()
    sw = np.load(C.RESULTS + f"/sweep_{args.pair}.npy", allow_pickle=True).item()
    setup = np.load(C.RESULTS + "/weekday_setup.npz")
    centroids = setup["centroids"]
    a_idx, b_idx = sw["a_idx"], sw["b_idx"]
    arc = dense_arc(centroids, a_idx)
    lambdas = sw["lambdas"]
    res = sw["results"]

    lin = sw["linear"]; spl = sw["spline_ref"]
    rec_lin = recovery(np.array(lin["W"]), arc)
    rec_spl = recovery(np.array(spl["W"]), arc)

    # seed-0 (linear-init) primary curve
    rows = []
    for lam in lambdas:
        r = res[f"linear_lam{lam}"]
        rows.append(dict(lam=lam, rec=recovery(np.array(r["W"]), arc),
                         E_act=r["E_act"], E_out=r["E_out"],
                         endpoint_err=r["endpoint_err"], steps=r["steps"]))
    oo = res["linear_outonly"]
    rec_oo = recovery(np.array(oo["W"]), arc)

    # initialization sensitivity: perturbed seeds if present
    seed_recs = {}
    for k, r in res.items():
        if r["init_type"].startswith("perturbed") and r["kind"].startswith("lambda"):
            seed_recs.setdefault(r["lam"], []).append(recovery(np.array(r["W"]), arc))

    summary = dict(pair=args.pair, rec_linear=rec_lin, rec_spline_ref=rec_spl,
                   rec_output_only=rec_oo,
                   lambda_rows=rows,
                   spline_E_act=spl["E_act"], spline_E_out=spl["E_out"],
                   linear_E_act=lin["E_act"], linear_E_out=lin["E_out"])
    with open(C.RESULTS + f"/analysis_{args.pair}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Fig 1: recovery vs lambda ----
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = [max(l, 1e-2) for l in lambdas]     # place lambda=0 at 1e-2 on log axis
    ys = [r["rec"] for r in rows]
    ax.plot(xs, ys, "o-", color="#2166ac", label="combined objective (linear init)")
    for lam, rl in seed_recs.items():
        xl = max(lam, 1e-2)
        ax.scatter([xl] * len(rl), rl, color="#92c5de", s=30, zorder=3,
                   label="perturbed-init seeds" if lam == lambdas[0] else None)
    ax.axhline(rec_lin, ls="--", color="gray", label=f"linear chord ({rec_lin:.2f})")
    ax.axhline(rec_oo, ls=":", color="#b2182b", label=f"output-only ({rec_oo:.2f})")
    ax.axhline(rec_spl, ls="-.", color="green", label=f"centroid spline ({rec_spl:.2f})")
    ax.set_xscale("log"); ax.set_xlabel(r"$\lambda$ (output-energy weight; 0 shown at 0.01)")
    ax.set_ylabel("mean nearest-spline dist (PCA-32)")
    ax.set_title(f"Activation-manifold recovery vs $\\lambda$ — {args.pair}")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(C.PLOTS + "/s4_recovery_vs_lambda.png", dpi=130); plt.close(fig)

    # ---- Fig 2: energy trade-off ----
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for r in rows:
        ax.scatter(r["E_act"], r["E_out"], s=45, zorder=4)
        ax.annotate(f"λ={r['lam']:g}", (r["E_act"], r["E_out"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.scatter(lin["E_act"], lin["E_out"], marker="s", s=90, color="gray",
               label="linear chord", zorder=5)
    ax.scatter(oo["E_act"], oo["E_out"], marker="v", s=90, color="#b2182b",
               label="output-only", zorder=5)
    ax.scatter(spl["E_act"], spl["E_out"], marker="*", s=200, color="green",
               label="centroid spline", zorder=5)
    ax.set_xlabel(r"$E_{\mathrm{act}}$ (activation KE, PCA-32)")
    ax.set_ylabel(r"$E_{\mathrm{out}}$ (behavior KE, Hellinger)")
    ax.set_title(f"Energy trade-off — {args.pair}")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(C.PLOTS + "/s4_energy_tradeoff.png", dpi=130); plt.close(fig)

    # ---- Fig 3: d(t) curves ----
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    tt = np.linspace(0, 1, sw["n_wp"])
    ax.plot(tt, lin["dt"], "s--", color="gray", label="linear chord")
    ax.plot(tt, oo["dt"], "v:", color="#b2182b", label="output-only")
    ax.plot(tt, spl["dt"], "*-.", color="green", label="centroid spline")
    cmap = plt.get_cmap("viridis")
    for i, lam in enumerate(lambdas):
        r = res[f"linear_lam{lam}"]
        ax.plot(tt, r["dt"], "-", color=cmap(i / len(lambdas)), label=f"λ={lam:g}")
    ax.set_xlabel("path position $t$")
    ax.set_ylabel("$d(t)$: mean Hellinger dist from start behavior")
    ax.set_title(f"Downstream behavior displacement $d(t)$ — {args.pair}")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(C.PLOTS + "/s4_dt_curves.png", dpi=130); plt.close(fig)

    # ---- Fig 4: PCA geometry ----
    Z = setup["Z"]; gt = setup["gt_idx"]; sp = setup["spline_pts"]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    cmap2 = plt.get_cmap("hsv")
    for d in range(7):
        m = gt == d
        ax.scatter(Z[m, 0], Z[m, 1], color=cmap2(d / 7), s=16, alpha=0.4)
        ax.scatter(centroids[d, 0], centroids[d, 1], color=cmap2(d / 7),
                   s=140, marker="*", edgecolor="k", linewidth=0.7, zorder=6)
    ax.plot(sp[:, 0], sp[:, 1], "k-", lw=1.0, alpha=0.6, label="periodic spline")
    def pl(W, **kw): ax.plot(W[:, 0], W[:, 1], **kw)
    pl(np.array(lin["W"]), color="gray", ls="--", lw=2, marker="s", ms=4,
       label="linear chord")
    pl(np.array(oo["W"]), color="#b2182b", ls=":", lw=2, marker="v", ms=4,
       label="output-only")
    pl(np.array(spl["W"]), color="green", ls="-.", lw=2, marker="*", ms=6,
       label="centroid spline")
    for lam in [1.0, 100.0]:
        pl(np.array(res[f"linear_lam{lam}"]["W"]), lw=1.8, marker="o", ms=3,
           label=f"λ={lam:g}")
    ax.set_xlim(centroids[[a_idx, b_idx], 0].min() - 6,
                centroids[[a_idx, b_idx], 0].max() + 6)
    ax.set_ylim(centroids[[a_idx, b_idx], 1].min() - 6,
                centroids[[a_idx, b_idx], 1].max() + 6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Paths in PCA space (illustrative) — {args.pair}")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(C.PLOTS + "/s4_pca_geometry.png", dpi=130); plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
