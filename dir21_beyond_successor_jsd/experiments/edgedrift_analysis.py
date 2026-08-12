"""Which of the two negatives is about missing plateau structure, and which is about a different order?

Two separate negatives are on the table. GPT-2 ranks the 123 tokens by width with no agreement with
Pythia; Pythia-160M carries almost no width ordering of its own. Edge drift E = d(0.1) + (1 - d(0.9))
tells the two possible causes apart on curves that are already stored: E ~ 0 means the curve leaves
both endpoints alone (a plateau), E = 0.2 is the straight line d(t) = t, on which a width is close to
1 by construction and carries nothing.

Reads results/edgedrift.json (E and envelope width per token, GPT-2 blocks 0/4/8 and Pythia
160M/410M/1.4B at block 0) plus the per-curve widths in results/xwidth_*.json, and answers:

1. How plateau-shaped is each model, against the straight-line reference?
2. Does the E ordering travel between models the way the width ordering does?
3. Inside GPT-2, does the ordering appear once the least plateau-shaped measurements are thrown away?
   Keep only the curves with E <= 0.1 (half the straight-line value), recompute each token's width
   from what survives, and redo the reliability and transfer tests. Filtering curves rather than
   tokens keeps all 123 tokens, so the correlations are not restricted to a narrower range of widths.

Writes results/edgedrift_summary.json.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS
from xmodel_analysis import half_split

LINE_E = 0.2


def widths(raw, names, mask=None):
    """The 18 envelope widths of each token, with the curves excluded by `mask` set to NaN."""
    w = np.array([raw[s]["w_env"] for s in names], float)
    return w if mask is None else np.where(mask, w, np.nan)


def rel(w):
    """Split-half reliability of the per-token width over the 6 anchors (Spearman-Brown)."""
    h = np.array([half_split(row) for row in w])
    ok = np.isfinite(h).all(1)
    r = float(spearmanr(h[ok, 0], h[ok, 1]).statistic)
    return 2 * r / (1 + r), int(ok.sum())


def main():
    d = json.load(open(f"{RESULTS}/edgedrift.json"))
    rows = d["rows"]
    keys = list(rows)
    names = rows["gpt2_block0"]["tokens"]
    E = {k: np.array(rows[k]["edge_by_token"]) for k in keys}
    W = {k: np.array(rows[k]["w_by_token"]) for k in keys}
    res = {"configs": keys, "tokens": names, "line_reference": LINE_E,
           "frames": d["frames"], "anchors": d["anchors"]}

    # 1. how plateau-shaped is each model?
    res["shape"] = {k: dict(edge_median=rows[k]["edge_median"], edge_q10_q90=rows[k]["edge_q"],
                            frac_curves_above_half_line=rows[k]["edge_frac_above_half_line"],
                            w_median=rows[k]["w_median"], w_sd_across_tokens=float(W[k].std()),
                            rho_edge_width=rows[k]["rho_edge_w"]) for k in keys}
    for k in keys:
        s = res["shape"][k]
        print(f"[1] {k:12s} median E {s['edge_median']:.4f} (10-90% {s['edge_q10_q90'][0]:.4f}-"
              f"{s['edge_q10_q90'][1]:.4f}), {s['frac_curves_above_half_line']:.3f} of curves above "
              f"{0.5 * LINE_E:.2f}, median w_env {s['w_median']:.3f}, "
              f"rho(E, w) {s['rho_edge_width'][0]:+.3f}", flush=True)

    # 2. does the E ordering travel the way the width ordering does?
    pairs = [("410m_block0", "1.4b_block0"), ("160m_block0", "1.4b_block0"),
             ("gpt2_block0", "1.4b_block0"), ("gpt2_block0", "410m_block0")]
    res["transfer"] = {}
    for a, b in pairs:
        re_, we = spearmanr(E[a], E[b]), spearmanr(W[a], W[b])
        res["transfer"][f"{a}|{b}"] = dict(rho_edge=float(re_.statistic), p_edge=float(re_.pvalue),
                                           rho_width=float(we.statistic), p_width=float(we.pvalue),
                                           n=len(names))
        print(f"[2] {a} vs {b}: E rho {re_.statistic:+.3f} (p {re_.pvalue:.3g}), "
              f"width rho {we.statistic:+.3f} (p {we.pvalue:.3g})", flush=True)

    # 3. does either model's ordering appear once the least plateau-shaped curves are discarded?
    ref_w = widths(json.load(open(f"{RESULTS}/xwidth_1.4b.json"))["raw"], names)
    ref, rel_ref = np.nanmedian(ref_w, 1), rel(ref_w)[0]
    gpt2_w = widths(json.load(open(f"{RESULTS}/xwidth_gpt2.json"))["raw"], names)
    assert np.allclose(np.median(gpt2_w, 1), W["gpt2_block0"])  # same tokens, same curve order
    res["filter"] = {"edge_cut": 0.5 * LINE_E, "reference": "1.4b_block0",
                     "reference_reliability": rel_ref}
    for tag, wc_all in [("gpt2_block0", gpt2_w),
                        ("160m_block0", np.array(rows["160m_block0"]["w_curves"], float)
                         .reshape(len(names), -1))]:
        edge = np.array(rows[tag]["edge_curves"], float).reshape(len(names), -1)
        res["filter"][tag] = {}
        for label, mask in {"all_curves": None, "plateau_curves": edge <= 0.5 * LINE_E}.items():
            wc = wc_all if mask is None else np.where(mask, wc_all, np.nan)
            w = np.nanmedian(wc, 1)
            r_m, n_rel = rel(wc)
            ok = np.isfinite(w)
            rho = spearmanr(w[ok], ref[ok])
            ceil = float(np.sqrt(max(r_m, 0) * max(rel_ref, 0)))
            res["filter"][tag][label] = dict(
                frac_curves_kept=1.0 if mask is None else float(np.mean(mask)),
                n_tokens=int(ok.sum()), n_reliability=n_rel, reliability=r_m, ceiling=ceil,
                w_median=float(np.nanmedian(w)), rho_vs_1_4b=float(rho.statistic),
                p=float(rho.pvalue),
                disattenuated=float(rho.statistic / ceil) if ceil > 0 else None)
            r = res["filter"][tag][label]
            print(f"[3] {tag:12s} {label:15s} ({r['frac_curves_kept']:.3f} of curves kept, "
                  f"{r['n_tokens']} tokens): reliability {r['reliability']:+.3f}, "
                  f"ceiling {r['ceiling']:.3f}, rho with 1.4B {r['rho_vs_1_4b']:+.3f} "
                  f"(p {r['p']:.3g})", flush=True)

    json.dump(res, open(f"{RESULTS}/edgedrift_summary.json", "w"), indent=1)
    print(f"wrote {RESULTS}/edgedrift_summary.json")


if __name__ == "__main__":
    main()
