"""Figure for the token-path continuation analysis (results/token_continuations.json).

usage: python plot_token_cont.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS
from cvd_style import CVD, use_cvd

use_cvd()


def main():
    d = json.load(open(os.path.join(RESULTS, "token_continuations.json")))
    C = d["candidates"]
    sub = [c for c in C if c["subplateau"]]
    rest = [c for c in C if not c["subplateau"]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.3))

    bins = np.arange(-0.5, 21.5, 1.0)
    for lab, v, col, ls, hatch in (("true sub-plateaus (n=%d)" % len(sub),
                                    [c["prefix_C_run"] for c in sub], CVD[0], "-", "//"),
                                   ("other candidates (n=%d)" % len(rest),
                                    [c["prefix_C_run"] for c in rest], CVD[1], "--", "\\\\")):
        a1.hist(v, bins=bins, histtype="step", lw=2.4, color=col, ls=ls, density=True,
                label=f"{lab} ({ls})")
    a1.set_xlabel("identical leading tokens across the C run (of 20)")
    a1.set_ylabel("density")
    a1.set_title("(A) Is the third region one state across its whole run?", fontsize=10)
    a1.legend(fontsize=8)

    keys = [("prefix_A_region_vs_A_end", "A region\nvs endpoint A"),
            ("prefix_B_region_vs_B_end", "B region\nvs endpoint B"),
            ("prefix_C_run", "C run\nfirst vs mid vs last"),
            ("prefix_C_vs_A_end", "C centre\nvs endpoint A"),
            ("prefix_C_vs_B_end", "C centre\nvs endpoint B")]
    x = np.arange(len(keys))
    med = [np.median([c[k] for c in C]) for k, _ in keys]
    q1 = [np.percentile([c[k] for c in C], 25) for k, _ in keys]
    q3 = [np.percentile([c[k] for c in C], 75) for k, _ in keys]
    a2.bar(x, med, 0.6, yerr=[np.array(med) - q1, np.array(q3) - np.array(med)], capsize=4,
           color=CVD[:5], edgecolor="k", lw=0.5, hatch=["//", "\\\\", "..", "xx", "++"])
    a2.set_xticks(x)
    a2.set_xticklabels([l for _, l in keys], fontsize=8)
    a2.set_ylabel("identical leading tokens (of 20)")
    a2.set_title("(B) Median common greedy prefix, all 72 candidates", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "token_continuation_stability.png"), dpi=150)
    plt.close(fig)
    print("wrote plots/token_continuation_stability.png")


if __name__ == "__main__":
    main()
