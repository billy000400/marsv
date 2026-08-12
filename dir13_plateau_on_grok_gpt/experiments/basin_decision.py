"""Feedback #7: are the basins individuated by the ENDPOINT CHARACTER or by the DECISION CLASS?

The all-pairs sweep established that 59 of 65 characters hold a basin against most partners, and that
91% of next-character prediction changes along a path fall inside the transition window. Those two
facts were read together as "one next-character decision basin per character". That reading conflates
two labellings of the same path: the endpoint CHARACTER that is patched in, and the model's argmax
next-character PREDICTION at that position. This script separates them, using the argmax already
stored for every one of the 2,080 pairs by allpairs_sweep.py -- no new forward passes.

Three counts decide it:
  * how many DISTINCT endpoint predictions the 65 endpoint characters produce (if this is far below
    65, predictions cannot label basins one-to-one);
  * the fraction of paths that visit exactly TWO predictions (what "A-basin -> boundary -> B-basin in
    decision space" requires);
  * the fraction of pairs whose two ENDPOINTS already share one prediction (for those, no decision
    distinguishes the endpoints at all, yet the plateau is still there).

Out -> results/basin_decision.json, plots/basin_decision.png.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")


def label(i, itos):
    c = itos[i]
    return {" ": "␣", "\n": "\\n"}.get(c, c)


def main():
    d = np.load(os.path.join(RES, "allpairs_raw.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(RES, "allpairs_summary.json")))
    itos = meta["itos"] if "itos" in meta else None
    if itos is None:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from allpairs_sweep import load_vocab
        stoi = load_vocab()
        itos = [c for c, _ in sorted(stoi.items(), key=lambda kv: kv[1])]

    keys = [k for k in d.keys() if k.startswith("final|L0|am|")]
    ep, n_visited, same_ep = {}, [], 0
    for k in keys:
        a, b = (int(x) for x in k.split("|")[-1].split("_"))
        am = d[k]
        ep[a], ep[b] = int(am[0]), int(am[-1])
        n_visited.append(len(set(am.tolist())))
        same_ep += int(am[0] == am[-1])
    n_visited = np.array(n_visited)
    classes = Counter(ep.values())

    out = {
        "n_pairs": len(keys), "n_characters": len(ep),
        "n_distinct_endpoint_predictions": len(classes),
        "class_sizes": sorted(classes.values(), reverse=True),
        "class_labels": {label(i, itos): n for i, n in
                         sorted(classes.items(), key=lambda kv: -kv[1])},
        "frac_paths_exactly_two_predictions": round(float((n_visited == 2).mean()), 4),
        "median_predictions_visited": float(np.median(n_visited)),
        "mean_predictions_visited": round(float(n_visited.mean()), 2),
        "iqr_predictions_visited": [float(np.percentile(n_visited, 25)),
                                    float(np.percentile(n_visited, 75))],
        "max_predictions_visited": int(n_visited.max()),
        "frac_pairs_endpoints_share_prediction": round(float(same_ep / len(keys)), 4),
    }
    with open(os.path.join(RES, "basin_decision.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    vals, cnts = np.unique(n_visited, return_counts=True)
    bars = ax[0].bar(vals, 100 * cnts / len(n_visited),
                     color=[CVD[1] if v == 2 else CVD[0] for v in vals],
                     hatch=["" if v == 2 else "//" for v in vals], edgecolor="black", lw=0.7)
    for v, c, b in zip(vals, cnts, bars):
        ax[0].text(v, 100 * c / len(n_visited) + 0.8, f"{100*c/len(n_visited):.1f}%",
                   ha="center", fontsize=9)
    ax[0].set_xticks(vals)
    ax[0].set_xlabel("distinct argmax next-character predictions visited along the path")
    ax[0].set_ylabel("% of the 2,080 character pairs")
    ax[0].set_ylim(0, 48)
    ax[0].set_title("(a) only 31.6% of paths visit two predictions\n(solid bar); the median is three")
    ax[0].grid(alpha=0.25, axis="y")

    items = sorted(classes.items(), key=lambda kv: -kv[1])
    ax[1].bar(range(len(items)), [n for _, n in items], color=CVD[0], hatch="\\\\",
              edgecolor="black", lw=0.7)
    ax[1].set_xticks(range(len(items)))
    ax[1].set_xticklabels([label(i, itos) for i, _ in items], fontsize=10)
    ax[1].set_xlabel("the 15 distinct endpoint predictions (predicted next character)")
    ax[1].set_ylabel("endpoint characters mapping to it (of 65)")
    ax[1].set_title("(b) 65 endpoint characters collapse onto\n15 predictions, one of them 13-to-1")
    ax[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "basin_decision.png"), dpi=150)
    plt.close(fig)
    print("wrote plots/basin_decision.png")


if __name__ == "__main__":
    main()
