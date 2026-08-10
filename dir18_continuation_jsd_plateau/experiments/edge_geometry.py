"""Why edge drift E tracks width w: the endpoints are pinned, so what E adds is WHERE the move sits.

Operator feedback #7 asked why a narrow transition could ever be "a steeper straight line". It
cannot: d(0)=0 and d(1)=1 by construction, so the only straight line available has slope 1 and gives
w = 0.8 exactly. This script replaces that wrong dichotomy with the correct one and measures it:

  * take a MEASURED curve and slide the whole transition along the path (holding the endpoint values
    outside it). The width w is unchanged by construction; the edge drift E is not,
  * so at fixed w, E is free to vary over a wide range -- E is NOT determined by w,
  * empirically our transitions sit in the middle of the path, which puts every measured curve at
    the bottom of its own range and is why E and w agree at rho ~ +0.97.

Also reports how much information E carries about corpus JSD once w is accounted for.

Reads committed curves only; no GPU, no model.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from common import PLOTS, RESULTS
from curve_metrics import E_LINEAR, edge_drift, metrics

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams["figure.dpi"] = 130
GRID = np.linspace(0.0, 1.0, 50)
RNG = np.random.default_rng(0)

SETTINGS = [("step143000_t256", "curves_step143000_t256.npy", "trained 1.4B"),
            ("step0_t256", "curves_step0_t256.npy", "untrained step 0"),
            ("step143000_410m_t256", "curves_step143000_410m_t256.npy", "410M trained")]


def place(w, A, grid=GRID):
    """A monotone curve with d(0)=0, d(1)=1 whose 0.1 -> 0.9 transition has width `w` and starts at `A`.

    Three straight segments through the knots (0,0), (A,0.1), (A+w,0.9), (1,1): the same transition,
    parked at different points of the path. Every member is a curve the experiment could have
    produced; they differ only in where the move happens.
    """
    return np.interp(grid, [0.0, A, A + w, 1.0], [0.0, 0.1, 0.9, 1.0])


def placement_range(w, n=201):
    """(min E, max E) over every placement of a width-`w` transition inside the path."""
    eps = 1.0 / (len(GRID) - 1)
    if w >= 1 - 2 * eps:
        e = edge_drift(place(w, (1 - w) / 2), GRID)
        return e, e
    es = np.array([edge_drift(place(w, A), GRID) for A in np.linspace(eps, 1 - w - eps, n)])
    return float(es.min()), float(es.max())


def midpoint(d, grid=GRID):
    """Position of the first upward crossing of d = 0.5, linearly interpolated."""
    for i in range(len(d) - 1):
        if d[i] <= 0.5 <= d[i + 1] and d[i + 1] > d[i]:
            f = (0.5 - d[i]) / (d[i + 1] - d[i])
            return float(grid[i] + f * (grid[i + 1] - grid[i]))
    return float("nan")


def per_pair(curves, rows):
    """Median-over-frames w, E, midpoint and the placement range of E, for every valid pair."""
    w, e, m, j, emin, emax = [], [], [], [], [], []
    for k, r in enumerate(rows):
        if not r["valid"]:
            continue
        ws, es, ms, lo, hi = [], [], [], [], []
        for c in range(curves.shape[1]):
            d = np.asarray(curves[k, c], float)
            mm = metrics(d, GRID)
            if not mm["valid"]:
                continue
            ws.append(mm["w"]); es.append(mm["edge_drift"]); ms.append(midpoint(d))
            a, b = placement_range(mm["w"])
            lo.append(a); hi.append(b)
        w.append(np.median(ws)); e.append(np.median(es)); m.append(np.median(ms))
        emin.append(np.median(lo)); emax.append(np.median(hi)); j.append(r["jsd_B"])
    return tuple(np.array(v, float) for v in (w, e, m, j, emin, emax))


def partial_spearman(x, y, cov):
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(cov)
    C = np.column_stack([rc, np.ones(len(rc))])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    return float(spearmanr(ex, ey).statistic)


def boot_partial(x, y, cov, n=10_000):
    r = partial_spearman(x, y, cov)
    bs = np.array([partial_spearman(*(v[k] for v in (x, y, cov)))
                   for k in (RNG.integers(0, len(x), len(x)) for _ in range(n))])
    return r, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    out = {"edge_drift_linear_reference": E_LINEAR, "settings": {}}
    D = {}
    for tag, fn, lab in SETTINGS:
        rows = json.load(open(os.path.join(RESULTS, f"qc_{tag}.json")))["rows"]
        curves = np.load(os.path.join(RESULTS, fn))
        w, e, m, j, emin, emax = per_pair(curves, rows)
        D[tag] = dict(w=w, e=e, m=m, j=j, emin=emin, emax=emax, lab=lab)
        out["settings"][tag] = dict(
            label=lab, n=len(w), median_w=float(np.median(w)), median_E=float(np.median(e)),
            spearman_w_E=float(spearmanr(w, e).statistic),
            median_midpoint=float(np.median(m)),
            iqr_midpoint=float(np.subtract(*np.percentile(m, [75, 25]))),
            frac_centred_within_0p1=float(np.mean(np.abs(m - 0.5) <= 0.1)),
            median_abs_offset=float(np.median(np.abs(m - 0.5))),
            median_E_min_over_placements=float(np.median(emin)),
            median_E_max_over_placements=float(np.median(emax)),
            median_ratio_Emax_over_E=float(np.median(emax / e)),
            frac_at_placement_minimum_within_0p01=float(np.mean(e - emin <= 0.01)),
            spearman_J_E=float(spearmanr(j, e).statistic))

    # How much does E know about corpus JSD that w does not, and vice versa?
    T = D["step143000_t256"]
    r, lo, hi = boot_partial(T["j"], T["e"], T["w"])
    out["partial_J_E_given_w"] = dict(rho=r, lo=lo, hi=hi, n=len(T["w"]))
    r2, lo2, hi2 = boot_partial(T["j"], T["w"], T["e"])
    out["partial_J_w_given_E"] = dict(rho=r2, lo=lo2, hi=hi2, n=len(T["w"]))

    # 1,000-pair replication of the w-E redundancy.
    rows = json.load(open(os.path.join(RESULTS, "qc_large_step143000.json")))["rows"]
    wl, el, ml, jl, _, _ = per_pair(np.load(os.path.join(RESULTS, "curves_large_step143000.npy")), rows)
    out["large_1000"] = dict(n=len(wl), spearman_w_E=float(spearmanr(wl, el).statistic),
                             median_midpoint=float(np.median(ml)),
                             frac_centred_within_0p1=float(np.mean(np.abs(ml - 0.5) <= 0.1)),
                             partial_J_E_given_w=partial_spearman(jl, el, wl))

    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 2, figsize=(11.4, 8.4))
    hatch = ["//", "\\\\", ".."]
    mk = ["o", "s", "^"]

    a = ax[0, 0]
    bins = np.linspace(0, 0.28, 40)
    for k, (tag, _, lab) in enumerate(SETTINGS):
        e_k = D[tag]["e"]
        a.hist(e_k, bins=bins, color=CVD[k], histtype="stepfilled", alpha=0.5, hatch=hatch[k],
               label=f"{lab} (median {np.median(e_k):.3f})")
    a.axvline(E_LINEAR, color="0.15", ls="--", lw=1.4)
    a.text(E_LINEAR - 0.004, a.get_ylim()[1] * 0.75, "straight line\n$d(t)=t$", ha="right", fontsize=7.5)
    a.set_xlabel("edge drift $E$  (0 = perfectly flat endpoint regions)")
    a.set_ylabel("number of pairs")
    a.set_title("(a) Trained curves have flat ends")
    a.legend(frameon=False, fontsize=7.5, loc="upper left")

    # (b) the same transition parked at two places: identical w, very different E.
    T = D["step143000_t256"]
    wstar = float(np.median(T["w"]))
    eps = 1.0 / (len(GRID) - 1)
    As = np.linspace(eps, 1 - wstar - eps, 201)
    Es = np.array([edge_drift(place(wstar, A), GRID) for A in As])
    A_lo, A_hi = float(As[int(np.argmin(Es))]), float(As[int(np.argmax(Es))])
    out["width_fidelity_of_constructed_curves"] = float(max(
        abs(metrics(place(wstar, A), GRID)["w"] - wstar) for A in (A_lo, A_hi)))
    out["band_at_median_w"] = dict(w=wstar, E_min=float(Es.min()), E_max=float(Es.max()),
                                   A_at_E_min=A_lo, A_at_E_max=A_hi,
                                   E_measured_median=float(np.median(T["e"])))
    # the measured curve whose width is closest to the median, for scale
    curves = np.load(os.path.join(RESULTS, "curves_step143000_t256.npy"))
    rows = json.load(open(os.path.join(RESULTS, "qc_step143000_t256.json")))["rows"]
    cand = [(abs(metrics(curves[i, c], GRID)["w"] - wstar), i, c) for i in range(curves.shape[0])
            for c in range(curves.shape[1]) if rows[i]["valid"] and metrics(curves[i, c], GRID)["valid"]]
    _, isel, csel = min(cand)
    d0 = np.asarray(curves[isel, csel], float)

    a = ax[0, 1]
    a.axvspan(0, 0.2, color="0.9", zorder=0); a.axvspan(0.8, 1, color="0.9", zorder=0)
    a.plot(GRID, GRID, color="0.35", ls=":", lw=1.6, label="straight line: $w=0.800$, $E=0.184$")
    a.plot(GRID, place(wstar, A_lo), color=CVD[0], ls="-", lw=1.8, marker="o", markevery=5, ms=4,
           label=f"transition centred: $E={Es.min():.3f}$")
    a.plot(GRID, place(wstar, A_hi), color=CVD[1], ls="--", lw=1.8, marker="s", markevery=5, ms=4,
           label=f"same transition parked {'late' if A_hi > A_lo else 'early'}: $E={Es.max():.3f}$")
    a.plot(GRID, d0, color=CVD[2], ls="-.", lw=1.6, marker="^", markevery=5, ms=4,
           label=f"a measured curve: $E={edge_drift(d0, GRID):.3f}$")
    a.axhline(0.1, color="0.7", lw=0.7); a.axhline(0.9, color="0.7", lw=0.7)
    a.text(0.1, 1.03, "edge window", fontsize=7, ha="center")
    a.text(0.9, 1.03, "edge window", fontsize=7, ha="center")
    a.set_xlabel("interpolation position $t$"); a.set_ylabel("output-distance score $d(t)$")
    a.set_title(f"(b) All three have width $w={wstar:.3f}$")
    a.legend(frameon=False, fontsize=7.2, loc="lower right")

    a = ax[1, 0]
    T = D["step143000_t256"]
    order = np.argsort(T["w"])
    a.vlines(T["w"][order], T["emin"][order], T["emax"][order], color="0.8", lw=1.6, zorder=0,
             label="range of $E$ over placements of the same transition")
    for k in (0, 1):
        Dk = D[SETTINGS[k][0]]
        a.scatter(Dk["w"], Dk["e"], s=26, color=CVD[k], marker=mk[k], alpha=0.85, edgecolors="none",
                  label=f"{Dk['lab']} as run (Spearman $w$ vs $E$ "
                        f"{spearmanr(Dk['w'], Dk['e']).statistic:+.3f})")
    a.axhline(E_LINEAR, color="0.15", ls="--", lw=1.2)
    a.set_xlabel("transition width $w$"); a.set_ylabel("edge drift $E$")
    a.set_title("(c) Every measured curve sits at the bottom of its own range")
    a.legend(frameon=False, fontsize=7.2, loc="upper left")

    a = ax[1, 1]
    bins = np.linspace(0.2, 0.8, 40)
    for k, (tag, _, lab) in enumerate(SETTINGS):
        m_k = D[tag]["m"]
        a.hist(m_k, bins=bins, color=CVD[k], histtype="stepfilled", alpha=0.5, hatch=hatch[k],
               label=f"{lab} (median {np.median(m_k):.3f})")
    a.axvline(0.5, color="0.15", ls="--", lw=1.4)
    a.set_xlabel("transition midpoint $t(d=0.5)$")
    a.set_ylabel("number of pairs")
    a.set_title("(d) Transitions sit in the middle of the path")
    a.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "edge_drift.png"))
    plt.close(fig)

    json.dump(out, open(os.path.join(RESULTS, "edge_geometry.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
