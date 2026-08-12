"""Does the per-token width ordering survive a change of tokenizer, corpus and architecture family?

Four questions, in the order they have to be answered:

0. IS THE PROTOCOL EVEN DEFINED THERE? Report dir18's strict validity rate per model. GPT-2's curves
   wander, so the strict statistic exists for almost none of them; the envelope width does.
1. IS THE SUBSTITUTION FAIR? Inside Pythia, where both statistics exist, how well does the envelope
   width reproduce the strict one -- per measurement and per token?
2. HOW RELIABLE IS EACH MODEL'S MEASUREMENT? Split-half over the 6 anchors, Spearman-Brown corrected.
   This is the ceiling any cross-model correlation can reach.
3. DOES THE ORDERING TRANSFER? Spearman agreement of the per-token widths between GPT-2 and each
   Pythia, raw and disattenuated; the same agreement after partialling out the two free corpus
   statistics (unigram count, successor entropy); and whether the free lookup read off Pythia-1.4B's
   embedding matrix ranks GPT-2's measured widths.

Writes results/xmodel_summary.json.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from checkpoints_analysis import corpus_stats, partial_rho
from common import RESULTS

TAGS = ["gpt2", "410m", "1.4b"]
N_ANCHOR = 6


def half_split(v):
    """Median over the even-indexed anchors and over the odd-indexed ones."""
    i = np.arange(len(v))
    return np.nanmedian(v[(i % N_ANCHOR) % 2 == 0]), np.nanmedian(v[(i % N_ANCHOR) % 2 == 1])


def reliability(raw, names, key):
    h = np.array([half_split(np.array(raw[s][key], float)) for s in names])
    ok = np.isfinite(h).all(1)
    r = float(spearmanr(h[ok, 0], h[ok, 1]).statistic)
    return r, float(2 * r / (1 + r)), int(ok.sum())


def main():
    d = {t: json.load(open(f"{RESULTS}/xwidth_{t}.json")) for t in TAGS}
    raw = {t: d[t]["raw"] for t in TAGS}
    names = sorted(set.intersection(*[set(raw[t]) for t in TAGS]))
    res = {"tags": TAGS, "tokens": names, "n_tokens": len(names),
           "anchors": d["gpt2"]["anchors"], "frames": d["gpt2"]["frames"]}

    # 0. is dir18's strict width defined at all in each model?
    val = {}
    for t in TAGS:
        f = np.array([raw[t][s]["valid"] for s in names], float)
        b = np.array([raw[t][s]["backslide"] for s in names], float)
        mono = np.array([raw[t][s]["mono"] for s in names], float)
        single = np.array([raw[t][s]["single"] for s in names], float)
        val[t] = dict(valid=float(f.mean()), mono=float(mono.mean()), single=float(single.mean()),
                      backslide_med=float(np.median(b)), backslide_p90=float(np.quantile(b, 0.9)),
                      n_curves=int(f.size))
        print(f"[0] {t}: strict-valid {val[t]['valid']:.3f} (monotone {val[t]['mono']:.3f}, "
              f"single-crossing {val[t]['single']:.3f}), median backslide {val[t]['backslide_med']:.3f}",
              flush=True)
    res["validity"] = val

    # 1. does the envelope width reproduce the strict width where both exist?
    agree = {}
    for t in TAGS:
        a = np.concatenate([np.array(raw[t][s]["w"], float) for s in names])
        b = np.concatenate([np.array(raw[t][s]["w_env"], float) for s in names])
        ok = np.isfinite(a)
        per_tok = [(np.nanmedian(np.array(raw[t][s]["w"], float)),
                    np.median(np.array(raw[t][s]["w_env"], float))) for s in names]
        p = np.array([x for x in per_tok if np.isfinite(x[0])])
        agree[t] = dict(n_curves=int(ok.sum()),
                        curve_rho=float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 3 else None,
                        curve_max_abs_diff=float(np.max(np.abs(a[ok] - b[ok]))) if ok.sum() else None,
                        n_tokens=len(p),
                        token_rho=float(spearmanr(p[:, 0], p[:, 1]).statistic) if len(p) > 3 else None)
        print(f"[1] {t}: envelope vs strict width -- per curve rho {agree[t]['curve_rho']}, "
              f"per token rho {agree[t]['token_rho']} (n {agree[t]['n_tokens']})", flush=True)
    res["env_vs_strict"] = agree

    # 2. reliability of each model's per-token envelope width
    rel = {t: dict(zip(("half_rho", "spearman_brown", "n"), reliability(raw[t], names, "w_env")))
           for t in TAGS}
    W = {t: np.array([np.median(np.array(raw[t][s]["w_env"], float)) for s in names]) for t in TAGS}
    for t in TAGS:
        rel[t].update(median=float(np.median(W[t])), sd=float(W[t].std(ddof=1)))
        print(f"[2] {t}: reliability {rel[t]['spearman_brown']:+.3f}, median w_env "
              f"{rel[t]['median']:.3f}, sd {rel[t]['sd']:.3f}", flush=True)
    res["reliability"] = rel

    # 3. cross-model agreement, raw / disattenuated / partialled on the free corpus statistics
    stats = corpus_stats(names)
    logc = np.log10([stats[s]["count"] for s in names])
    ent = np.array([stats[s]["ent"] for s in names])
    pair = {}
    for i, a in enumerate(TAGS):
        for b in TAGS[i + 1:]:
            r = spearmanr(W[a], W[b])
            ceil = float(np.sqrt(max(rel[a]["spearman_brown"], 0) * max(rel[b]["spearman_brown"], 0)))
            pr, pp = partial_rho(W[a], W[b], [logc, ent])
            pair[f"{a}|{b}"] = dict(rho=float(r.statistic), p=float(r.pvalue), n=len(names),
                                    ceiling=ceil, disattenuated=float(r.statistic / ceil),
                                    partial=pr, partial_p=pp)
            print(f"[3] {a} vs {b}: rho {r.statistic:+.3f} (p {r.pvalue:.1e}), ceiling {ceil:.3f}, "
                  f"disattenuated {r.statistic / ceil:+.3f}, partial {pr:+.3f}", flush=True)
    res["pairwise"] = pair

    # the free lookup, and the two corpus statistics, against each model's widths
    pred = json.load(open(f"{RESULTS}/embed.json"))["probe_pred"]
    p14 = np.array([pred[s] for s in names])
    look = {}
    for t in TAGS:
        r = spearmanr(p14, W[t])
        look[t] = dict(rho=float(r.statistic), p=float(r.pvalue), n=len(names),
                       disattenuated=float(r.statistic / np.sqrt(max(rel[t]["spearman_brown"], 0))),
                       rho_logcount=float(spearmanr(W[t], logc).statistic),
                       rho_entropy=float(spearmanr(W[t], ent).statistic),
                       probe_own=d[t]["probe"]["rho_mean"], probe_own_sd=d[t]["probe"]["rho_sd"],
                       probe_own_null=d[t]["probe"]["null_rho_mean"])
        print(f"[3] 1.4B lookup -> {t}: rho {r.statistic:+.3f} (p {r.pvalue:.1e}); own probe "
              f"{look[t]['probe_own']:+.3f}; logN {look[t]['rho_logcount']:+.3f}", flush=True)
    res["lookup"] = look

    # GPT-2's early-component ablation, next to its own baseline
    g = json.load(open(f"{RESULTS}/xwidth_gpt2.json"))
    res["gpt2_ablate"] = dict(base=g["ablate_base"], rows=g["ablate"])

    json.dump(res, open(f"{RESULTS}/xmodel_summary.json", "w"), indent=1)
    print("wrote results/xmodel_summary.json")


if __name__ == "__main__":
    main()
