"""Feedback 7: the summary_distance_vs_width figure for the SLERP-path sweeps (slerp_sweep.py), five models.

Same layout, style, y-range rule, monotonic-curve filter and correlations as summary_distance_vs_width.py
(its functions are reused; only its configuration is replaced here). Two figures:
  slerp_distance_vs_width.{png,pdf}            x = L_slerp, the length of the injected SLERP path
  slerp_endpoint_distance_vs_width.png         x = D, the straight-line distance between the two endpoints
y = final-block transition width measured along the SLERP path, in both.
Llama 3.1-8B is cropped to its bulk as in the linear summary figure (D from 1.0 to 2.5; path length from 1.0 to
2.75, the same range stretched by the ~1.1 median ratio of path length to D). Any other panel is cropped when a few
far points stretch its x range (AUTO_CROP rule of summary_distance_vs_width.py). A cropped panel prints how many
points are not drawn.
Also writes results/block0_resid_post/slerp/summary.json. Reads CSVs only; runs no model.

Usage: python experiments/summary_slerp_distance_vs_width.py
"""
import csv
import json
import os

import numpy as np

import summary_distance_vs_width as S

SLERP_DIR = os.path.join(S.DEFAULT_RESULTS_DIR, "slerp")
PLOT_DIR = os.path.dirname(S.DEFAULT_OUT)
KEYS = [("GPT-2 Large", "gpt2-large"), ("GPT-2 XL", "gpt2-xl"), ("Pythia 1.4B", "pythia-1.4b"),
        ("Llama 3.1-8B", "llama-3.1-8b"), ("Qwen3-8B", "qwen3-8b-base")]

S.MODELS = [(name, f"{key}.csv") for name, key in KEYS]
# The SLERP CSVs carry the final-block monotonic test themselves (column mono_final), so no shard files are read.
S.MONOTONIC_RULE, S.COL_FLAG, S.MONOTONIC_FLAG_VALUE = "csv_flag", "mono_final", "1"
S.FILTER_NOTE["csv_flag"] = (
    "SLERP path injected at the block-0 output. Monotonic curve: the final-block readout curve crosses the 10% "
    "and the 90% level once each and never drops by more than 0.05 between neighbouring interpolation "
    "positions. r, rho and n count every monotonic curve, including the ones outside a cropped x range.")
S.Y_LABEL = "Transition width W (final block, SLERP path)"
S.AUTO_CROP = True
LLAMA_CROP = {"L_slerp": (1.0, 2.75), "D": (1.0, 2.5)}

FIGURES = [  # (x column, x label, summary-panel title, output path without extension, formats)
    ("L_slerp", "SLERP path length (L2, block-0 output)", "Spearman rank correlation, path length vs W",
     os.path.join(PLOT_DIR, "slerp_distance_vs_width"), ["png", "pdf"]),
    ("D", "Endpoint distance D (L2, block-0 output)", "Spearman rank correlation, D vs W (SLERP path)",
     os.path.join(PLOT_DIR, "slerp_endpoint_distance_vs_width"), ["png"]),
]


def linear_records():
    """r, rho, n of the linear-path sweeps under the same final-block filter, for the side-by-side table."""
    import importlib
    L = importlib.reload(S)   # restores that script's own configuration; call only after the SLERP figures
    out = {}
    for (name, rel), (_, key) in zip(L.MODELS, KEYS):
        path = os.path.join(L.DEFAULT_RESULTS_DIR, rel)
        dist, width = L.select_monotonic(path, *L.load_csv(path))
        r, rho = L.correlations(dist, width)
        out[key] = dict(n_monotonic=len(dist), pearson_r=r, spearman_rho=rho, W_final_median=float(np.median(width)))
    return out


def main():
    summary = {key: {} for _, key in KEYS}
    for col, xlabel, title, out, formats in FIGURES:
        S.COL_DISTANCE, S.X_LABEL, S.SUMMARY_TITLE = col, xlabel, title
        S.X_CROP = {"Llama 3.1-8B": LLAMA_CROP[col]}
        for rec, (_, key) in zip(S.main(SLERP_DIR, out, formats), KEYS):
            summary[key].update({"n_tokens": rec["rows"], "n_monotonic": rec["n"],
                                 f"pearson_r_W_final_vs_{col}": rec["pearson"],
                                 f"spearman_rho_W_final_vs_{col}": rec["spearman"],
                                 f"plot_xlim_{col}": rec["xlim"], f"n_not_drawn_{col}": rec["n_not_drawn"],
                                 "W_final_median": float(np.median(rec["width"]))})
    for _, key in KEYS:   # path facts from every row of the CSV
        rows = list(csv.DictReader(open(os.path.join(SLERP_DIR, f"{key}.csv"), newline="")))
        D, L, c = (np.array([float(r[k]) for r in rows]) for k in ("D", "L_slerp", "cos"))
        summary[key].update(n_negative_cosine=int((c < 0).sum()), n_nonfinite_width=int(sum(r["W_final"] == "nan" for r in rows)),
                            L_over_D_pct_0_50_100=[float(x) for x in np.nanpercentile(L / D, [0, 50, 100])],
                            pearson_r_L_vs_D=float(np.corrcoef(L[np.isfinite(L)], D[np.isfinite(L)])[0, 1]))
    lin = linear_records()
    for _, key in KEYS:
        summary[key]["linear_path_same_filter"] = lin[key]
    json.dump(summary, open(os.path.join(SLERP_DIR, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
