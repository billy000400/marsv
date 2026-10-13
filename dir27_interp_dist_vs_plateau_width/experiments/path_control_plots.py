"""Path control: widths, tables and figures from results/block0_resid_post/path_control/pilot.npz (no model).

Usage: DIR27_SITE=block0_resid_post python3 path_control_plots.py
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from common import CVD, MODEL, PLOTS, RESULTS, TS, width
from s3_plots import BACKSTEP

OUT = os.path.join(RESULTS, "path_control")
PL = os.path.join(PLOTS, "path_control")
os.makedirs(PL, exist_ok=True)
METHODS = ("linear", "matthew_slerp_rescale")
NAME = {"linear": "linear path", "matthew_slerp_rescale": "Matthew's path"}
STYLE = {"linear": dict(color=CVD[0], marker="o"), "matthew_slerp_rescale": dict(color=CVD[1], marker="^")}


def widths(y):
    """y (N, T) -> W (N,), flag (N,), valid (N,). Same first-crossing rule and flag as the main sweep."""
    W, flag, valid = np.full(len(y), np.nan), np.zeros(len(y), bool), np.isfinite(y).all(1)
    for i in np.where(valid)[0]:
        _, _, W[i], n1, n9, back = width(TS, y[i])
        flag[i] = n1 > 1 or n9 > 1 or back > BACKSTEP
    return W, flag, valid


def main():
    z = np.load(os.path.join(OUT, "pilot.npz"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids, rand, D = z["ids"], z["is_random"], z["D_endpoint"]
    R = {}
    for me in METHODS:
        L = z[f"{me}__L_path_n2000"]
        R[me] = dict(L=L)
        for blk in ("mid", "fin"):
            dA, dB = z[f"{me}__dA_{blk}"].astype(np.float64), z[f"{me}__dB_{blk}"].astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                ys = dict(fixed=dA / dA[:, -1:], relative=dA / (dA + dB))
            for norm, y in ys.items():
                W, flag, valid = widths(y)
                R[me][f"y_{norm}_{blk}"], R[me][f"W_{norm}_{blk}"] = y, W
                R[me][f"flag_{norm}_{blk}"], R[me][f"valid_{norm}_{blk}"] = flag, valid
        for norm in ("fixed", "relative"):   # as in the main sweep, a token is flagged if either block's curve is
            R[me][f"flag_{norm}"] = R[me][f"flag_{norm}_mid"] | R[me][f"flag_{norm}_fin"]
            R[me][f"valid_{norm}"] = R[me][f"valid_{norm}_mid"] & R[me][f"valid_{norm}_fin"]

    # reuse check: rerun linear widths vs the saved full-vocabulary block-0 sweep
    sh = [np.load(p) for p in sorted(glob.glob(os.path.join(RESULTS, "sweep", "shard_*.npz")))]
    sid, sst = np.concatenate([s["ids"] for s in sh]), np.concatenate([s["stats"] for s in sh])
    sD = np.concatenate([s["D"] for s in sh])
    pos = {int(i): k for k, i in enumerate(sid)}
    k = np.array([pos[int(i)] for i in ids])
    S = dict(n_random=int(rand.sum()), n_illustrative=int((~rand).sum()),
             rerun_linear_vs_saved_sweep_max_abs=dict(
                 W_mid=float(np.abs(R["linear"]["W_fixed_mid"] - sst[k, 2]).max()),
                 W_final=float(np.abs(R["linear"]["W_fixed_fin"] - sst[k, 8]).max()),
                 D=float(np.abs(D - sD[k]).max())))

    S["cosine"] = dict(min=float(z["cosine"][rand].min()), median=float(np.median(z["cosine"][rand])),
                       max=float(z["cosine"][rand].max()), n_negative=int((z["cosine"] < 0).sum()))
    S["path_length_over_D_endpoint"] = {
        me: dict(min=float((R[me]["L"] / D)[rand].min()), median=float(np.median((R[me]["L"] / D)[rand])),
                 max=float((R[me]["L"] / D)[rand].max()),
                 doubling_max_rel_change=float(np.abs(z[f"{me}__L_path_n2000"] / z[f"{me}__L_path_n1000"] - 1).max()))
        for me in METHODS}
    S["norm_realized_over_nominal"] = {me: [float(z[f"{me}__norm_ratio"].min()), float(z[f"{me}__norm_ratio"].max())]
                                       for me in METHODS}
    prog = z["matthew_slerp_rescale__progress"]
    S["matthew_path_progress_minus_t"] = dict(max_abs_random=float(np.abs(prog[rand] - TS).max()),
                                              median_of_per_pair_max=float(np.median(np.abs(prog[rand] - TS).max(1))))

    fifth_edges = np.quantile(D[rand], [0.2, 0.4, 0.6, 0.8])
    grp = np.digitize(D, fifth_edges)
    rows = []
    for norm in ("fixed", "relative"):
        both = rand & np.logical_and.reduce([R[me][f"valid_{norm}"] & ~R[me][f"flag_{norm}"] for me in METHODS])
        s = dict(n_valid={me: int((rand & R[me][f"valid_{norm}"]).sum()) for me in METHODS},
                 n_flagged={me: int((rand & R[me][f"flag_{norm}"]).sum()) for me in METHODS},
                 n_unflagged_both=int(both.sum()), n_excluded=int(rand.sum() - both.sum()))
        for blk in ("mid", "fin"):
            wl, wm = R["linear"][f"W_{norm}_{blk}"], R["matthew_slerp_rescale"][f"W_{norm}_{blk}"]
            diff = (wm - wl)[both]
            s[blk] = dict(
                median_W={"linear": float(np.median(wl[both])), "matthew_slerp_rescale": float(np.median(wm[both]))},
                median_W_by_D_fifth={me: [float(np.median(R[me][f"W_{norm}_{blk}"][both & (grp == g)])) for g in range(5)]
                                     for me in METHODS},
                n_by_D_fifth=[int((both & (grp == g)).sum()) for g in range(5)],
                paired_diff_matthew_minus_linear=dict(median=float(np.median(diff)), min=float(diff.min()),
                                                      max=float(diff.max()),
                                                      median_abs=float(np.median(np.abs(diff))),
                                                      n_abs_gt_0p02=int((np.abs(diff) > 0.02).sum()),
                                                      n_abs_gt_0p05=int((np.abs(diff) > 0.05).sum()),
                                                      n_matthew_wider=int((diff > 0).sum())),
                median_W_by_L_path_fifth_matthew=[float(np.median(wm[both & (np.digitize(
                    R["matthew_slerp_rescale"]["L"], np.quantile(R["matthew_slerp_rescale"]["L"][rand],
                                                                 [0.2, 0.4, 0.6, 0.8])) == g)])) for g in range(5)])
        S[f"normalization_{norm}"] = s
    S["D_fifth_edges"] = fifth_edges.tolist()

    with open(os.path.join(OUT, "pilot_widths.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = [f"{q}_{norm}_{blk}__{me}" for me in METHODS for norm in ("fixed", "relative") for blk in ("mid", "fin")
                for q in ("W", "flag", "valid")]
        w.writerow(["token_id", "token", "set", "D_endpoint", "cosine", "L_path__linear", "L_path__matthew_slerp_rescale"] + cols)
        for i in range(len(ids)):
            vals = []
            for c in cols:
                key, me = c.split("__")
                v = R[me][key][i]
                vals.append(f"{v:.4f}" if key.startswith("W") else int(v))
            w.writerow([ids[i], repr(tok.decode([int(ids[i])])), "random" if rand[i] else "illustrative", f"{D[i]:.4f}",
                        f"{z['cosine'][i]:.4f}", f"{R['linear']['L'][i]:.4f}", f"{R['matthew_slerp_rescale']['L'][i]:.4f}"] + vals)
    ill = np.where(~rand)[0]
    S["illustrative"] = [dict(token=tok.decode([int(ids[i])]), D=float(D[i]), cosine=float(z["cosine"][i]),
                              L_matthew=float(R["matthew_slerp_rescale"]["L"][i]),
                              **{f"W_{norm}_{blk}_{me}": float(R[me][f"W_{norm}_{blk}"][i]) for me in METHODS
                                 for norm in ("fixed", "relative") for blk in ("mid", "fin")}) for i in ill]
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))

    def scatter_dw(norm, fname):
        """Rows: middle / final block. Columns: x = endpoint distance; x = sampled path length."""
        both = rand & np.logical_and.reduce([R[me][f"valid_{norm}"] & ~R[me][f"flag_{norm}"] for me in METHODS])
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="row")
        for i, (blk, bname) in enumerate((("mid", "block 18 (middle)"), ("fin", "block 35 (final)"))):
            for j, xname in enumerate(("endpoint distance D_endpoint", "injected path length L_path")):
                ax = axes[i, j]
                for me in METHODS:
                    x = D if j == 0 else R[me]["L"]
                    W = R[me][f"W_{norm}_{blk}"]
                    ax.scatter(x[both], W[both], s=16, alpha=0.7, facecolors="none" if me == "linear" else STYLE[me]["color"],
                               edgecolors=STYLE[me]["color"], marker=STYLE[me]["marker"], lw=0.8, label=NAME[me])
                ax.set_title(f"{bname}; x = {xname.split(' D_')[0].split(' L_')[0]}", fontsize=10)
                ax.set_xlabel(f"block-0 resid_post {xname}")
                if j == 0:
                    ax.set_ylabel("transition width W" + (" (relative normalization)" if norm == "relative" else ""))
        axes[0, 0].legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(PL, fname), dpi=130)
        plt.close(fig)

    def paired(norm, fname):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        ok = {me: rand & R[me][f"valid_{norm}"] for me in METHODS}
        both = np.logical_and.reduce([ok[me] & ~R[me][f"flag_{norm}"] for me in METHODS])
        other = ok["linear"] & ok["matthew_slerp_rescale"] & ~both
        for ax, blk, bname in ((axes[0], "mid", "(a) block 18 (middle)"), (axes[1], "fin", "(b) block 35 (final)")):
            wl, wm = R["linear"][f"W_{norm}_{blk}"], R["matthew_slerp_rescale"][f"W_{norm}_{blk}"]
            ax.scatter(wl[both], wm[both], s=16, alpha=0.7, color=CVD[0], marker="o", lw=0, label="unflagged under both paths")
            ax.scatter(wl[other], wm[other], s=30, color=CVD[1], marker="x", lw=1, label="flagged under at least one path")
            lim = [0, 1.05 * np.nanmax([wl[rand], wm[rand]])]
            ax.plot(lim, lim, color="0.4", ls="--", lw=1, label="equal widths")
            ax.set_xlim(lim), ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.set_xlabel("width W, linear path")
            ax.set_ylabel("width W, Matthew's path")
            ax.set_title(bname)
        axes[0].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PL, fname), dpi=130)
        plt.close(fig)

    scatter_dw("fixed", "pc1_distance_vs_width.png")
    paired("fixed", "pc2_paired_width.png")
    paired("relative", "pc4_paired_width_relative_normalization.png")
    scatter_dw("relative", "pc5_distance_vs_width_relative_normalization.png")

    # matched curves for the illustrative pairs: rows = middle / final block
    fig, axes = plt.subplots(2, len(ill), figsize=(3.1 * len(ill), 6), sharex=True, sharey=True)
    for c, i in enumerate(ill):
        for rw, blk in enumerate(("mid", "fin")):
            ax = axes[rw, c]
            ax.plot(TS, R["linear"][f"y_fixed_{blk}"][i], color=CVD[0], ls="-", lw=1.6, label="y(t), linear path")
            ax.plot(TS, R["matthew_slerp_rescale"][f"y_fixed_{blk}"][i], color=CVD[1], ls="--", lw=1.6,
                    label="y(t), Matthew's path")
            ax.plot(TS, prog[i], color="0.35", ls=":", lw=1.4, label="progress of Matthew's injected path")
            ax.axhline(0.1, color="0.75", lw=0.6)
            ax.axhline(0.9, color="0.75", lw=0.6)
            wl, wm = R["linear"][f"W_fixed_{blk}"][i], R["matthew_slerp_rescale"][f"W_fixed_{blk}"][i]
            ax.set_title(f"{tok.decode([int(ids[i])])!r}, {'middle' if blk == 'mid' else 'final'}\n"
                         f"W {wl:.3f} / {wm:.3f}", fontsize=9)
            if rw == 1:
                ax.set_xlabel("t")
        axes[0, c].text(0.97, 0.03, f"D={D[i]:.1f}, cos={z['cosine'][i]:.2f}", transform=axes[0, c].transAxes,
                        ha="right", fontsize=8)
    axes[0, 0].set_ylabel("progress (block 18)")
    axes[1, 0].set_ylabel("progress (block 35)")
    axes[1, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PL, "pc3_matched_curves.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
