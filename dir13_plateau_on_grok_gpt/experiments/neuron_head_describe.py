"""Are the units training promotes into the head describable in a way the ones it demotes are not?

neuron_head_origin.py showed that a pair's step-30,000 top-8 units are already mid-ranked at step 831
(median rank 113.5 of 3,840) and climb, while the step-831 top-8 are demoted but not discarded (median
rank 100.5 at the end). That is a statement about ranks only: it says nothing about what the promoted
and the demoted units COMPUTE. neuron_probe.py already measured, for every block-1..4 unit, how well a
ridge regression on the characters in an 8-character window predicts its post-GeLU activation on
held-out corpus positions -- R^2 from the current character alone (r2_1) and from the full window with
the lag0 x lag1 interaction (r2_full). This script joins the two: it labels every unit by its role at
the two ends of training and compares those R^2 distributions. No model is loaded and nothing is
recomputed; both inputs are on disk.

Units are labelled from neuron_head_identity.py's per-pair top-8 sets:

  * final head   F  -- in some pair's step-30,000 top 8
  * early head   E  -- in some pair's step-831 top 8
  * promoted   F\\E  -- in the finished head, never in the early head
  * demoted    E\\F  -- held the head early, not in the finished head
  * stable     F&E  -- in both
  * never-head      -- in neither

The raw promoted-vs-demoted contrast that the question asks for is confounded by construction: a
promoted unit has best rank <= 7 at step 30,000 and a demoted unit >= 8, and importance rank at step
30,000 is itself associated with describability (neuron_bands.py, Figure 37c). So the comparison is
also run WITHIN a band of current importance, using neuron_bands.py's per-unit best rank across the
150 pairs at step 30,000:

  * band 0-7   (the finished head)        -- stable vs promoted
  * band 8-31  (just below the head)      -- demoted vs never-head

Within a band the units share a current importance level, so any surviving difference is carried by
the early-checkpoint role rather than by present rank. Every test is a two-sided Mann-Whitney U over
DISTINCT units, reported with the median of each group and the common-language effect size
U / (n1 * n2) -- the chance that a randomly drawn unit of the first group scores above one of the
second, 0.5 being no difference.

Stats -> results/neuron_head_describe_summary.json, figure -> plots/neuron_head_describe.png.
"""
import os, sys, json
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
METRICS = [("r2_1", "the current character alone"), ("r2_full", "the full 8-character window")]
BANDS = {"0-7": (0, 7), "8-31": (8, 31)}
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def test(r, g1, g2):
    u, p = mannwhitneyu(r[g1], r[g2], alternative="two-sided")
    return {"n1": int(g1.size), "n2": int(g2.size),
            "median1": round(float(np.median(r[g1])), 4), "median2": round(float(np.median(r[g2])), 4),
            "cles": round(float(u / (g1.size * g2.size)), 3), "p": float(p)}


def ecdf(ax, r, g, color, ls, label):
    x = np.sort(r[g])
    ax.step(np.concatenate([[0.0], x]), np.arange(x.size + 1) / x.size,
            where="post", color=color, ls=ls, lw=1.8, label=f"{label} (n={g.size})")


def main():
    ident = np.load(os.path.join(RES, "neuron_head_identity_raw.npz"))
    bands = np.load(os.path.join(RES, "neuron_bands_raw.npz"))
    probe = np.load(os.path.join(RES, "neuron_probe_raw.npz"))
    assert np.array_equal(ident["pairs"], bands["pairs"]), "different pair sets"
    steps = list(ident["steps"])
    top8 = ident["top8"]                                  # [n_steps, n_pairs, 8] unit ids
    min_rank = bands["min_rank"]                          # best rank any of the 150 pairs gives a unit
    n_units = min_rank.size

    final = np.unique(top8[-1])
    early = np.unique(top8[0])
    groups = {"promoted": np.setdiff1d(final, early),
              "demoted": np.setdiff1d(early, final),
              "stable": np.intersect1d(final, early),
              "never_head": np.setdiff1d(np.arange(n_units), np.union1d(final, early)),
              "all_units": np.arange(n_units)}
    in_band = {b: np.where((min_rank >= lo) & (min_rank <= hi))[0] for b, (lo, hi) in BANDS.items()}
    # the two within-band comparisons: same current importance, different early-checkpoint role
    band_groups = {
        "0-7": ("stable", np.intersect1d(in_band["0-7"], groups["stable"]),
                "promoted", np.intersect1d(in_band["0-7"], groups["promoted"])),
        "8-31": ("demoted", np.intersect1d(in_band["8-31"], groups["demoted"]),
                 "never_head", np.intersect1d(in_band["8-31"], groups["never_head"])),
    }

    summary = {"steps": [int(s) for s in steps], "n_units": int(n_units),
               "k_head": int(top8.shape[2]), "n_pairs": int(top8.shape[1]),
               "group_sizes": {k: int(v.size) for k, v in groups.items()},
               "band_sizes": {b: {"n_band": int(v.size), band_groups[b][0]: int(band_groups[b][1].size),
                                  band_groups[b][2]: int(band_groups[b][3].size)}
                              for b, v in in_band.items()},
               "medians": {}, "raw_contrast": {}, "within_band": {}}
    for m, _ in METRICS:
        r = probe[m]
        summary["medians"][m] = {k: round(float(np.median(r[g])), 4) for k, g in groups.items()}
        summary["raw_contrast"][m] = test(r, groups["promoted"], groups["demoted"])
        summary["within_band"][m] = {b: test(r, band_groups[b][1], band_groups[b][3]) for b in BANDS}

    with open(os.path.join(RES, "neuron_head_describe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6))
    for row, (m, mlabel) in enumerate(METRICS):
        r = probe[m]
        a = axes[row][0]
        ecdf(a, r, groups["promoted"], CVD[0], "-", "promoted")
        ecdf(a, r, groups["demoted"], CVD[1], "--", "demoted")
        ecdf(a, r, groups["all_units"], "gray", ":", "all units")
        s = summary["raw_contrast"][m]
        a.set_title(f"(a{row + 1}) promoted vs demoted, unconditional\n"
                    f"CLES {s['cles']:.2f}, p = {s['p']:.1g}")
        for col, b in enumerate(BANDS):
            a2 = axes[row][col + 1]
            n1, g1, n2, g2 = band_groups[b]
            ecdf(a2, r, g1, CVD[2], "-", n1.replace("_", "-"))
            ecdf(a2, r, g2, CVD[3], "-.", n2.replace("_", "-"))
            s2 = summary["within_band"][m][b]
            a2.set_title(f"({'abc'[col + 1]}{row + 1}) best rank {b} at step 30,000\n"
                         f"CLES {s2['cles']:.2f}, p = {s2['p']:.1g}")
        for col in range(3):
            ax = axes[row][col]
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel(f"held-out $R^2$ from {mlabel}")
            ax.set_ylabel("fraction of units at or below")
            ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "neuron_head_describe.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
