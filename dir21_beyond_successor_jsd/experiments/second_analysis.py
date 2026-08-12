"""Pool the second-model runs: how much of the width trait is the token, and how much is the network?

Three questions the per-model runs cannot answer on their own:

A. RELIABILITY. A weak cross-model agreement means nothing if the measurement itself is noisy in that
   model. For every model we split the 6 anchors into two halves, recompute w_hat_u from each half and
   Spearman-Brown correct the agreement between them. That gives each model's measurement reliability
   and hence a CEILING on any correlation it can show.
B. AGREEMENT. All pairwise Spearman correlations of the measured per-token widths across the four
   sizes, raw and divided by the geometric-mean reliability of the pair (disattenuated).
C. TRANSFER OF THE FREE LOOKUP. The deliverable is a probe read off Pythia-1.4B's embedding matrix.
   Do its out-of-fold predictions rank the OTHER models' measured widths?

Writes results/second_summary.json.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS

TAGS = ["160m", "410m", "1b", "1.4b"]
N_ANCHOR = 6


def load_raw():
    """Per-token list of the 18 (frame, anchor) width measurements, for each model."""
    out = {}
    for tag in TAGS[:-1]:
        d = json.load(open(f"{RESULTS}/second_{tag}.json"))
        out[tag] = {s: np.array(v, float) for s, v in d["w_raw"].items()}
    d = json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"]
    out["1.4b"] = {s: np.array(v["w"], float) for s, v in d.items()}
    return out


def half_split(v):
    """Median width over the even-indexed anchors and over the odd-indexed ones."""
    i = np.arange(len(v))
    a = np.nanmedian(v[(i % N_ANCHOR) % 2 == 0])
    b = np.nanmedian(v[(i % N_ANCHOR) % 2 == 1])
    return a, b


def main():
    raw = load_raw()
    names = sorted(set.intersection(*[set(r) for r in raw.values()]))
    res = {"tokens": names, "n_tokens": len(names), "tags": TAGS}

    # A. split-half reliability of the per-token width, per model
    rel = {}
    for tag in TAGS:
        h = np.array([half_split(raw[tag][s]) for s in names])
        ok = np.isfinite(h).all(1)
        r = float(spearmanr(h[ok, 0], h[ok, 1]).statistic)
        rel[tag] = dict(half_rho=r, spearman_brown=float(2 * r / (1 + r)),
                        n=int(ok.sum()),
                        sd=float(np.nanstd([np.nanmedian(raw[tag][s]) for s in names], ddof=1)),
                        median=float(np.nanmedian([np.nanmedian(raw[tag][s]) for s in names])))
        print(f"{tag}: split-half rho {r:+.3f} -> reliability {rel[tag]['spearman_brown']:+.3f}, "
              f"median w {rel[tag]['median']:.3f}, sd {rel[tag]['sd']:.3f}", flush=True)
    res["reliability"] = rel

    # B. pairwise agreement of the measured widths, raw and disattenuated
    w = {tag: np.array([np.nanmedian(raw[tag][s]) for s in names]) for tag in TAGS}
    pair = {}
    for i, a in enumerate(TAGS):
        for b in TAGS[i + 1:]:
            ok = np.isfinite(w[a]) & np.isfinite(w[b])
            r = spearmanr(w[a][ok], w[b][ok])
            ceil = float(np.sqrt(max(rel[a]["spearman_brown"], 0) * max(rel[b]["spearman_brown"], 0)))
            pair[f"{a}|{b}"] = dict(rho=float(r.statistic), p=float(r.pvalue), n=int(ok.sum()),
                                    ceiling=ceil, disattenuated=float(r.statistic / ceil))
            print(f"{a} vs {b}: rho {r.statistic:+.3f} (ceiling {ceil:.3f}, "
                  f"disattenuated {r.statistic / ceil:+.3f})", flush=True)
    res["pairwise"] = pair

    # C. does the 1.4B embedding lookup rank the other models' measured widths?
    pred = json.load(open(f"{RESULTS}/embed.json"))["probe_pred"]
    xfer = {}
    for tag in TAGS:
        y = np.array([np.nanmedian(raw[tag][s]) for s in names])
        p = np.array([pred[s] for s in names])
        ok = np.isfinite(y)
        r = spearmanr(p[ok], y[ok])
        xfer[tag] = dict(rho=float(r.statistic), p=float(r.pvalue), n=int(ok.sum()))
        print(f"1.4B lookup -> {tag} measured width: rho {r.statistic:+.3f} (p {r.pvalue:.2g})",
              flush=True)
    res["lookup_transfer"] = xfer

    # per-model headline numbers from the individual runs, for the figure and the tables
    per = {}
    for tag in TAGS[:-1]:
        d = json.load(open(f"{RESULTS}/second_{tag}.json"))
        abl = d["ablate"]
        base = np.array(d["ablate_base"]["w"])
        per[tag] = dict(
            config=d["config"], valid=d["valid_frac"],
            probe=dict(rho=d["probe"]["rho_mean"], sd=d["probe"]["rho_sd"],
                       r2=d["probe"]["r2_mean"], null=d["probe"]["null_rho_mean"]),
            base_sd=float(base.std(ddof=1)), base_mean=float(base.mean()),
            ablate={f"{r['block']}:{r['comp']}": dict(sd=r["sd"], rho=r["rho"], bits=r["bits"],
                                                      mean=r["mean"]) for r in abl})
    e = json.load(open(f"{RESULTS}/embed.json"))
    a14 = json.load(open(f"{RESULTS}/ablate.json"))
    b14 = np.array(a14["base_w"])
    mlp0 = [r for r in a14["rows"] if r["block"] == 0 and r["comp"] == "mlp"][0]
    per["1.4b"] = dict(config=dict(n_layer=24, d_model=2048, n_head=16), valid=None,
                       probe=dict(rho=e["probe_w_block0"]["rho_mean"],
                                  sd=e["probe_w_block0"]["rho_sd"],
                                  r2=e["probe_w_block0"]["r2_mean"],
                                  null=e["probe_w_block0"]["null_rho_mean"]),
                       base_sd=float(b14.std(ddof=1)), base_mean=float(b14.mean()),
                       ablate={"0:mlp": dict(sd=mlp0["sd"], bits=mlp0["bits"], mean=mlp0["mean"],
                                             rho=float(spearmanr(b14, mlp0["w"]).statistic))})
    res["per_model"] = per

    # D. the matched-control rerun on the second model, summarised next to the 1.4B one
    c = json.load(open(f"{RESULTS}/second_ctrl_410m.json"))
    arms = []
    for a in c["alphas"]:
        m = [r for r in c["rows"] if r["seed"] is None and r["alpha"] == a][0]
        cs = [r for r in c["rows"] if r["seed"] is not None and r["alpha"] == a]
        arms.append(dict(alpha=a, bits=m["bits"], mlp_rho=m["rho"],
                         ctrl_rho=[r["rho"] for r in cs],
                         ctrl_rho_mean=float(np.mean([r["rho"] for r in cs])),
                         mlp_sd=m["sd"], ctrl_sd_mean=float(np.mean([r["sd"] for r in cs]))))
    res["ctrl_410m"] = dict(arms=arms, paired=c["paired"],
                            base_sd=float(np.std(c["base_w"], ddof=1)),
                            base_mean=float(np.mean(c["base_w"])))

    json.dump(res, open(f"{RESULTS}/second_summary.json", "w"), indent=1)
    print("wrote results/second_summary.json")


if __name__ == "__main__":
    main()
