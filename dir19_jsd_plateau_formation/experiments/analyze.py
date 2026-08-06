"""Formation analysis: when do JSD-selective ordering and global plateau shape appear?

Reads every results/assay_step*.json + moves_step*.npy produced by scan.py and writes
results/checkpoint_metrics.json with, per checkpoint:

  rho          Spearman(J_hold, w) across the 60 pairs, with a SIMULTANEOUS 95% band built from
               the trajectory bootstrap (the same resampled pairs at every checkpoint, band width
               set by the max deviation across checkpoints so the whole curve is covered at 95%);
  median_w, iqr_w, edge_drift          global curve shape;
  group_w[5]                           median w inside each corpus-JSD quintile;
  dw_from0, rho_J_dw0                  change since step 0 and whether J predicts it;
  interval rho(J, dw)                  the same, restricted to each adjacent checkpoint interval;
  rho_outjsd                           Spearman(J_hold, model output JSD);
  move_entropy, move_window, move_total   full-output movement concentration along the path;
  loss, lr                             timing context.

Bootstrap resampling is over PAIRS (the unit of independence), paired across checkpoints.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

import curve_metrics
from common import RESULTS

B = 4000
RNG = np.random.default_rng(0)
GRID = np.linspace(0.0, 1.0, 50)
MID = 0.5 * (GRID[:-1] + GRID[1:])   # each movement step m_j sits between two grid positions
WIN = 0.1                            # half-width of the fixed 0.2-wide movement window in t
TINY = 1e-8                          # total-movement floor: below this a path is not normalised


# step16's released weights fail the monotone-loss check by 6.5 nats and reproduce the FINAL
# model's curves; ckpt_qc.py documents the evidence. It is excluded from every trajectory.
EXCLUDE_STEPS = {16}


def load_checkpoints():
    """Every assayed checkpoint, ordered by training step."""
    out = []
    for f in glob.glob(os.path.join(RESULTS, "assay_step*.json")):
        if f.endswith("_t256.json"):
            continue                                  # inherited dir18 copy, superseded by our run
        a = json.load(open(f))
        if a["step"] in EXCLUDE_STEPS:
            continue
        a["moves"] = np.load(f.replace("assay_", "moves_").replace(".json", ".npy"))
        a["curves"] = np.load(f.replace("assay_", "curves_").replace(".json", ".npy"))
        out.append(a)
    return sorted(out, key=lambda a: a["step"])


def curve_shape(curves):
    """Per-pair median transition width w and median edge drift E over the three contexts."""
    n_p, n_c, _ = curves.shape
    w = np.full((n_p, n_c), np.nan)
    e = np.full((n_p, n_c), np.nan)
    for i in range(n_p):
        for c in range(n_c):
            m = curve_metrics.metrics(curves[i, c], GRID)
            w[i, c], e[i, c] = m["w"], m["edge_drift"]
    return np.nanmedian(w, 1), np.nanmedian(e, 1)


def movement(moves):
    """Per-pair movement concentration: normalised entropy of the profile, and total movement.

    m_j is the JSD in bits between the model's full next-token distributions at neighbouring
    interpolation positions. Total movement T = sum_j m_j is reported before normalisation; paths
    with T < TINY are not normalised (they carry no movement to concentrate). Normalised entropy
    H(r)/log(49) is 1 when movement is spread evenly along the path and falls towards 0 as it
    concentrates into a few steps.
    """
    n_p, n_c, _ = moves.shape
    H = np.full((n_p, n_c), np.nan)
    T = np.nansum(moves, -1)
    for i in range(n_p):
        for c in range(n_c):
            if not np.isfinite(T[i, c]) or T[i, c] < TINY:
                continue
            r = moves[i, c] / T[i, c]
            p = r[r > 0]
            H[i, c] = -(p * np.log(p)).sum() / np.log(len(r))
    return H, T


def window_mass(moves, curves):
    """Fraction of movement inside the fixed 0.2-wide window in t centred on the d(t)=0.5 crossing.

    The window is a LOCATION check with a width that does not depend on w, so it cannot be inflated
    by the transition itself getting narrower.
    """
    n_p, n_c, _ = moves.shape
    out = np.full((n_p, n_c), np.nan)
    for i in range(n_p):
        for c in range(n_c):
            T = np.nansum(moves[i, c])
            if not np.isfinite(T) or T < TINY:
                continue
            t50 = curve_metrics._first_up(np.asarray(curves[i, c], float), GRID, 0.5)
            if t50 is None:
                continue
            sel = np.abs(MID - t50) <= WIN
            out[i, c] = float(moves[i, c][sel].sum() / T)
    return out


def boot_idx(n):
    return RNG.integers(0, n, size=(B, n))


def simultaneous_band(stat_boot, stat):
    """95% band covering the WHOLE trajectory: stat +/- c, c = 95th pct of max_s |boot - stat|."""
    dev = np.nanmax(np.abs(stat_boot - stat[None, :]), axis=1)
    c = float(np.nanpercentile(dev, 95))
    return stat - c, stat + c, c


def pct_ci(x):
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


def main():
    cps = load_checkpoints()
    steps = np.array([a["step"] for a in cps])
    J = np.array([r["jsd_B"] for r in cps[0]["rows"]])        # holdout-split corpus next-token JSD
    JA = np.array([r["jsd_A"] for r in cps[0]["rows"]])
    bins = np.array([r["bin"] for r in cps[0]["rows"]])
    n = len(J)
    print(f"{len(cps)} checkpoints, {n} pairs")

    W, E, H, M, T, OJ = [], [], [], [], [], []
    for a in cps:
        w, e = curve_shape(a["curves"])
        h, t = movement(a["moves"])
        W.append(w)
        E.append(e)
        H.append(np.nanmedian(h, 1))
        M.append(np.nanmedian(window_mass(a["moves"], a["curves"]), 1))
        T.append(np.nanmedian(t, 1))
        OJ.append(np.array([r["out_jsd_med"] for r in a["rows"]]))
    W, E, H, M, T, OJ = map(np.array, (W, E, H, M, T, OJ))     # each (n_ckpt, n_pairs)

    idx = boot_idx(n)
    rho = np.array([spearmanr(J, w, nan_policy="omit").statistic for w in W])
    rho_b = np.array([[spearmanr(J[i], w[i], nan_policy="omit").statistic for w in W] for i in idx])
    lo, hi, c_sim = simultaneous_band(rho_b, rho)
    rho_ci = [pct_ci(rho_b[:, k]) for k in range(len(cps))]

    rho_oj = np.array([spearmanr(J, o, nan_policy="omit").statistic for o in OJ])
    rho_oj_ci = [pct_ci(np.array([spearmanr(J[i], o[i], nan_policy="omit").statistic for i in idx]))
                 for o in OJ]

    # --- change since step 0, and interval-specific change ---
    dw0 = W - W[0][None, :]
    rho_dw0 = np.array([spearmanr(J, d, nan_policy="omit").statistic for d in dw0])
    rho_dw0_ci = [pct_ci(np.array([spearmanr(J[i], d[i], nan_policy="omit").statistic for i in idx]))
                  for d in dw0]
    intervals = []
    for k in range(1, len(cps)):
        d = W[k] - W[k - 1]
        ok = np.isfinite(d)
        r = spearmanr(J[ok], d[ok]).statistic
        rb = np.array([spearmanr(J[i][np.isfinite(d[i])], d[i][np.isfinite(d[i])]).statistic
                       for i in idx])
        med_b = np.array([np.nanmedian(d[i]) for i in idx])
        intervals.append(dict(s0=int(steps[k - 1]), s1=int(steps[k]), rho=float(r),
                              rho_ci=pct_ci(rb), median_dw=float(np.nanmedian(d)),
                              median_dw_ci=pct_ci(med_b),
                              wilcoxon_p=float(wilcoxon(d[ok]).pvalue) if ok.sum() > 5 else None))

    # --- global shape trajectories with simultaneous bands ---
    def traj(X):
        s = np.nanmedian(X, 1)
        sb = np.array([[np.nanmedian(x[i]) for x in X] for i in idx])
        l, h_, _ = simultaneous_band(sb, s)
        return s, l, h_, [pct_ci(sb[:, k]) for k in range(len(cps))]

    med_w, w_lo, w_hi, w_ci = traj(W)
    med_e, e_lo, e_hi, e_ci = traj(E)
    med_h, h_lo, h_hi, h_ci = traj(H)
    med_m, m_lo, m_hi, m_ci = traj(M)

    # --- onset brackets ---
    below = (hi < 0)
    onset = None
    for k in range(len(cps) - 1):
        if below[k] and below[k + 1]:
            prev = [j for j in range(k) if not below[j]]
            onset = dict(after=int(steps[max(prev)]) if prev else None, by=int(steps[k]))
            break
    shape_onset = None
    for k in range(len(cps) - 1):
        if all(med_w[j] < 0.8 and w_hi[j] < 0.8 and med_e[j] < curve_metrics.E_LINEAR
               and e_hi[j] < curve_metrics.E_LINEAR for j in (k, k + 1)):
            prev = [j for j in range(k) if not (med_w[j] < 0.8 and w_hi[j] < 0.8)]
            shape_onset = dict(after=int(steps[max(prev)]) if prev else None, by=int(steps[k]))
            break

    out = dict(
        n_pairs=n, n_boot=B, steps=[int(s) for s in steps],
        e_linear=float(curve_metrics.E_LINEAR), w_linear=0.8, simultaneous_c_rho=c_sim,
        rho=[float(x) for x in rho], rho_sim_lo=[float(x) for x in lo],
        rho_sim_hi=[float(x) for x in hi], rho_ci=rho_ci,
        rho_outjsd=[float(x) for x in rho_oj], rho_outjsd_ci=rho_oj_ci,
        rho_dw_from0=[float(x) for x in rho_dw0], rho_dw_from0_ci=rho_dw0_ci,
        intervals=intervals,
        median_w=[float(x) for x in med_w], w_sim_lo=[float(x) for x in w_lo],
        w_sim_hi=[float(x) for x in w_hi], w_ci=w_ci,
        iqr_w=[float(np.nanpercentile(w, 75) - np.nanpercentile(w, 25)) for w in W],
        edge_drift=[float(x) for x in med_e], e_sim_lo=[float(x) for x in e_lo],
        e_sim_hi=[float(x) for x in e_hi], e_ci=e_ci,
        move_entropy=[float(x) for x in med_h], h_sim_lo=[float(x) for x in h_lo],
        h_sim_hi=[float(x) for x in h_hi], h_ci=h_ci,
        move_window=[float(x) for x in med_m], m_sim_lo=[float(x) for x in m_lo],
        m_sim_hi=[float(x) for x in m_hi], m_ci=m_ci,
        move_total=[float(np.nanmedian(t)) for t in T],
        n_zero_movement=[int(np.sum(t < TINY)) for t in T],
        group_w=[[float(np.nanmedian(w[bins == g])) for g in range(5)] for w in W],
        group_jsd=[float(np.median(J[bins == g])) for g in range(5)],
        valid_rate=[float(np.isfinite(w).mean()) for w in W],
        loss=[float(a["heldout_loss"]) for a in cps], lr=[float(a["lr"]) for a in cps],
        max_endpoint_relerr=[float(a["max_endpoint_relerr"]) for a in cps],
        rho_JA_JB=float(spearmanr(JA, J).statistic),
        onset_ordering=onset, onset_shape=shape_onset,
    )
    json.dump(out, open(os.path.join(RESULTS, "checkpoint_metrics.json"), "w"), indent=2)
    np.savez(os.path.join(RESULTS, "per_pair_trajectories.npz"), steps=steps, J=J, JA=JA,
             bins=bins, W=W, E=E, H=H, M=M, T=T, OJ=OJ)
    for k, s in enumerate(steps):
        print(f"step {s:>6}  rho {rho[k]:+.3f} [{lo[k]:+.3f},{hi[k]:+.3f}]  med_w {med_w[k]:.3f}  "
              f"E {med_e[k]:.3f}  Hmove {med_h[k]:.3f}  win {med_m[k]:.3f}  loss {out['loss'][k]:.3f}")
    print("ordering onset:", onset, " shape onset:", shape_onset)


if __name__ == "__main__":
    main()
