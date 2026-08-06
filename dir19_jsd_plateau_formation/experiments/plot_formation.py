"""Figures for the formation scan. Headless Agg; CVD-safe palette, every series also coded by
linestyle/marker so the figures survive grayscale printing."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
LS = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]
MK = ["o", "s", "^", "D", "v"]


def xaxis(ax, steps):
    """Training step on a symlog axis so step 0 is visible next to step 143000."""
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.3, max(steps) * 1.6)
    ax.set_xlabel("training step (symlog; 0 shown at left)")


def band(ax, x, lo, hi, color, hatch):
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, hatch=hatch, edgecolor=color, linewidth=0)


def overview(M):
    s = np.array(M["steps"])
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6))

    a = ax[0]
    band(a, s, M["rho_sim_lo"], M["rho_sim_hi"], CVD[0], "//")
    a.plot(s, M["rho"], ls=LS[0], marker=MK[0], ms=3.5, color=CVD[0], label=r"$\rho$(J, w)")
    a.axhline(0, color="0.35", ls=LS[1], lw=1)
    if M["onset_ordering"] and M["onset_ordering"]["after"] is not None:
        a.axvspan(M["onset_ordering"]["after"], M["onset_ordering"]["by"], alpha=0.16,
                  hatch="\\\\", facecolor=CVD[1], edgecolor=CVD[1], lw=0)
    xaxis(a, s)
    a.set_ylabel(r"Spearman $\rho$(corpus JSD, width $w$)")
    a.set_title("A. JSD-selective ordering")
    a.legend(loc="lower left", fontsize=8)

    a = ax[1]
    band(a, s, M["w_sim_lo"], M["w_sim_hi"], CVD[0], "//")
    a.plot(s, M["median_w"], ls=LS[0], marker=MK[0], ms=3.5, color=CVD[0], label="median $w$")
    a.plot(s, M["iqr_w"], ls=LS[2], marker=MK[2], ms=3.5, color=CVD[2], label="IQR($w$)")
    a.axhline(M["w_linear"], color="0.35", ls=LS[1], lw=1)
    a.text(2.5e3, M["w_linear"] + 0.01, "straight-line reference $w=0.8$", fontsize=7, color="0.25")
    xaxis(a, s)
    a.set_ylabel("transition width $w$")
    a.set_title("B. Global sharpening")
    a.legend(loc="center left", fontsize=8)

    a = ax[2]
    band(a, s, M["e_sim_lo"], M["e_sim_hi"], CVD[1], "\\\\")
    a.plot(s, M["edge_drift"], ls=LS[1], marker=MK[1], ms=3.5, color=CVD[1], label="median $E$")
    a.axhline(M["e_linear"], color="0.35", ls=LS[1], lw=1)
    a.text(2.0e3, M["e_linear"] + 0.006, f"straight-line reference $E$={M['e_linear']:.3f}",
           fontsize=7, color="0.25")
    xaxis(a, s)
    a.set_ylabel("edge drift $E$")
    a.set_title("C. Flatness of the endpoint regions")
    a.legend(loc="center left", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "formation_overview.png"))
    plt.close(fig)


def interval(M):
    s = np.array(M["steps"])
    gj = M["group_jsd"]
    gw = np.array(M["group_w"])
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6))

    a = ax[0]
    for g in range(5):
        a.plot(s, gw[:, g], ls=LS[g], marker=MK[g], ms=3.2, color=CVD[g],
               label=f"Q{g+1}: J={gj[g]:.2f}")
    xaxis(a, s)
    a.set_ylabel("median $w$ within quintile")
    a.set_title("A. Width by corpus-JSD quintile")
    a.legend(fontsize=7, loc="lower left")

    a = ax[1]
    x = [i["s1"] for i in M["intervals"]]
    r = [i["rho"] for i in M["intervals"]]
    lo = [i["rho"] - i["rho_ci"][0] for i in M["intervals"]]
    hi = [i["rho_ci"][1] - i["rho"] for i in M["intervals"]]
    a.errorbar(x, r, yerr=[lo, hi], ls=LS[0], marker=MK[0], ms=3.5, color=CVD[0], capsize=2, lw=1)
    a.axhline(0, color="0.35", ls=LS[1], lw=1)
    xaxis(a, s)
    a.set_ylabel(r"$\rho$(corpus JSD, $\Delta w$ in interval)")
    a.set_title("B. Does JSD predict WHERE sharpening happens?")

    a = ax[2]
    a.plot(s, M["rho_dw_from0"], ls=LS[0], marker=MK[0], ms=3.5, color=CVD[0],
           label=r"$\rho$(J, $w_s - w_0$)")
    a.plot(s, M["rho_outjsd"], ls=LS[1], marker=MK[1], ms=3.5, color=CVD[1],
           label=r"$\rho$(J, model output JSD)")
    a.axhline(0, color="0.35", ls=LS[1], lw=1)
    xaxis(a, s)
    a.set_ylabel(r"Spearman $\rho$")
    a.set_title("C. Cumulative change and learned separation")
    a.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "interval_sharpening.png"))
    plt.close(fig)


def movement_fig(M):
    s = np.array(M["steps"])
    z = np.load(os.path.join(RESULTS, "per_pair_trajectories.npz"))
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6))

    a = ax[0]
    band(a, s, M["h_sim_lo"], M["h_sim_hi"], CVD[0], "//")
    a.plot(s, M["move_entropy"], ls=LS[0], marker=MK[0], ms=3.5, color=CVD[0])
    a.axhline(1.0, color="0.35", ls=LS[1], lw=1)
    a.text(6e3, 1.01, "uniform movement (=1)", fontsize=7, color="0.25")
    xaxis(a, s)
    a.set_ylabel(r"normalised entropy $H(r)/\log 49$")
    a.set_title("A. Movement concentration (lower = more concentrated)")

    a = ax[1]
    band(a, s, M["m_sim_lo"], M["m_sim_hi"], CVD[1], "\\\\")
    a.plot(s, M["move_window"], ls=LS[1], marker=MK[1], ms=3.5, color=CVD[1])
    a.axhline(0.2, color="0.35", ls=LS[1], lw=1)
    a.text(6e3, 0.15, "uniform expectation (=0.2)", fontsize=7, color="0.25")
    xaxis(a, s)
    a.set_ylabel("movement mass in fixed 0.2 window")
    a.set_title("B. Movement location vs the $d=0.5$ crossing")

    a = ax[2]
    a.plot(s, M["move_total"], ls=LS[0], marker=MK[0], ms=3.5, color=CVD[0],
           label="median total movement $T$ (bits)")
    a.set_yscale("log")
    a2 = a.twinx()
    a2.plot(s, M["loss"], ls=LS[2], marker=MK[2], ms=3.5, color=CVD[2],
            label="held-out next-token loss")
    a2.set_ylabel("held-out loss (nats)", color=CVD[2])
    a2.grid(False)
    xaxis(a, s)
    a.set_ylabel("total movement $T$ (bits, log)", color=CVD[0])
    a.set_title("C. Total movement and training loss")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "output_movement_formation.png"))
    plt.close(fig)
    return z


def profiles(M):
    """Median movement profile r(t) at a few checkpoints, aligned on the d=0.5 crossing."""
    import curve_metrics
    grid = np.linspace(0, 1, 50)
    mid = 0.5 * (grid[:-1] + grid[1:])
    show = [st for st in [0, 128, 1000, 8000, 143000] if st in M["steps"]][:5]
    fig, a = plt.subplots(figsize=(5.4, 3.8))
    for k, st in enumerate(show):
        mv = np.load(os.path.join(RESULTS, f"moves_step{st}.npy"))
        cv = np.load(os.path.join(RESULTS, f"curves_step{st}.npy"))
        prof = []
        for i in range(mv.shape[0]):
            for c in range(mv.shape[1]):
                T = mv[i, c].sum()
                t50 = curve_metrics._first_up(np.asarray(cv[i, c], float), grid, 0.5)
                if T < 1e-8 or t50 is None:
                    continue
                prof.append(np.interp(mid, mid - t50 + 0.5, mv[i, c] / T))
        a.plot(mid - 0.5, np.median(prof, 0), ls=LS[k], marker=MK[k], ms=2.5, color=CVD[k],
               label=f"step {st}")
    a.set_xlabel(r"position relative to the $d=0.5$ crossing,  $t - t_{50}$")
    a.set_ylabel("median normalised movement $r_j$")
    a.set_title("Where along the path the output moves")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "movement_profiles.png"))
    plt.close(fig)


def large_onset_fig():
    """Ordering-onset bracket re-tested on the 1,000-pair bank with endpoint-clustered CIs."""
    r = json.load(open(os.path.join(RESULTS, "large_late.json")))
    M = json.load(open(os.path.join(RESULTS, "checkpoint_metrics.json")))
    steps = [0, 8, 32, 64000, 143000]
    tags = [f"large_step{s}" for s in steps]
    if not all(t in r["rho"] for t in tags):
        return
    fig, a = plt.subplots(figsize=(6.2, 3.9))
    x = np.arange(len(steps))
    lr = np.array([r["rho"][t] for t in tags])
    lo = np.array([r["rho_endpoint_ci"][t][0] for t in tags])
    hi = np.array([r["rho_endpoint_ci"][t][1] for t in tags])
    a.errorbar(x - 0.07, lr, yerr=[lr - lo, hi - lr], fmt=MK[0], color=CVD[0], capsize=4, ms=7,
               lw=1.6, ls="none", label="1,000-pair set (bootstrap over 123 endpoint tokens)")
    k = [M["steps"].index(s) for s in steps]
    sr = np.array([M["rho"][i] for i in k])
    slo = np.array([M["rho_ci"][i][0] for i in k])
    shi = np.array([M["rho_ci"][i][1] for i in k])
    a.errorbar(x + 0.07, sr, yerr=[sr - slo, shi - sr], fmt=MK[1], color=CVD[1], capsize=4, ms=7,
               lw=1.6, ls="none", label="60-pair controlled set (bootstrap over pairs)")
    a.axhline(0, color="0.35", ls=LS[1], lw=1)
    a.axvspan(1.0, 2.0, alpha=0.14, hatch="\\\\", facecolor=CVD[2], edgecolor=CVD[2], lw=0)
    a.text(1.5, 0.09, "onset bracket", ha="center", fontsize=7.5, color="0.25")
    a.set_xticks(x)
    a.set_xticklabels([f"step {s}" for s in steps], fontsize=8)
    a.set_ylabel(r"Spearman $\rho$(corpus JSD, width $w$), 95% CI")
    a.set_title("Ordering onset replicates on the 1,000-pair bank")
    a.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "large_bank_onset.png"))
    plt.close(fig)


def qc_fig():
    """Evidence that the released `step16` artefact is not a step-16 model."""
    q = json.load(open(os.path.join(RESULTS, "ckpt_qc.json")))
    s = np.array(q["steps"], float)
    L = np.array(q["heldout_loss"])
    bad = s == 16
    fig, a = plt.subplots(figsize=(6.0, 3.9))
    a.plot(s[~bad], L[~bad], ls=LS[0], marker=MK[0], ms=4, color=CVD[0],
           label="released checkpoints (kept)")
    a.plot(s[bad], L[bad], ls="none", marker="X", ms=11, color=CVD[1], mew=1.6,
           label="revision 'step16' (excluded)")
    a.annotate("its 9,000 d(t) curve values are\nbit-identical to step143000",
               xy=(16, L[bad][0]), xytext=(60, 5.6), fontsize=7.5, color=CVD[1],
               arrowprops=dict(arrowstyle="->", color=CVD[1], lw=1.1))
    xaxis(a, s)
    a.set_ylabel("held-out next-token loss (nats)")
    a.set_title("Checkpoint QC: one released revision is mislabelled")
    a.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "checkpoint_qc.png"))
    plt.close(fig)


def large_fig():
    """60-pair vs 1,000-pair test of the late 64k -> final widening."""
    r = json.load(open(os.path.join(RESULTS, "large_late.json")))
    M = json.load(open(os.path.join(RESULTS, "checkpoint_metrics.json")))
    k64, kf = M["steps"].index(64000), M["steps"].index(143000)
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))

    a = ax[0]
    for k, (nm, m64, mf) in enumerate([
            ("60-pair controlled set", M["median_w"][k64], M["median_w"][kf]),
            ("1,000-pair set", r["median_w"]["large_step64000"],
             r["median_w"]["large_step143000"])]):
        a.plot([64000, 143000], [m64, mf], ls=LS[k], marker=MK[k], ms=6, color=CVD[k], label=nm)
    a.set_xticks([64000, 143000])
    a.set_xticklabels(["step 64000", "step 143000"])
    a.set_ylabel("median transition width $w$")
    a.set_title("A. Both banks widen late")
    a.legend(fontsize=7.5)

    a = ax[1]
    est = [(r["small_median_dw"], r["small_median_dw_ci"],
            f"{r['small_n']}-pair controlled set\n(bootstrap over pairs)"),
           (r["median_dw_64k_to_final"], r["median_dw_ci"],
            f"{r['n_pairs']}-pair set\n(bootstrap over the {r['n_endpoints']} endpoint tokens)")]
    for k, (v, ci, nm) in enumerate(est):
        a.errorbar([v], [1 - k], xerr=[[v - ci[0]], [ci[1] - v]], fmt=MK[k], color=CVD[k],
                   capsize=4, ms=7, lw=1.6)
    a.axvline(0, color="0.4", ls=LS[1], lw=1)
    a.set_yticks([1, 0])
    a.set_yticklabels([e[2] for e in est], fontsize=7.5)
    a.set_ylim(-0.6, 1.6)
    a.set_xlabel(r"median $\Delta w$ from step 64000 to step 143000, 95% CI")
    a.set_title("B. Positive means blunter at the end of training")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "large_bank_confirmation.png"))
    plt.close(fig)


if __name__ == "__main__":
    M = json.load(open(os.path.join(RESULTS, "checkpoint_metrics.json")))
    overview(M)
    interval(M)
    movement_fig(M)
    profiles(M)
    qc_fig()
    if os.path.exists(os.path.join(RESULTS, "large_late.json")):
        large_fig()
        large_onset_fig()
    print("plots written")
