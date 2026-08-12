"""Turn the checkpoint sweep into the two numbers that decide where the width trait comes from.

Question 1 -- WHEN. For every checkpoint, how well does its per-token width ranking agree with the
final checkpoint's, raw and divided by the geometric-mean split-half reliability of the two (so an
early checkpoint is not penalised for being noisy). Early-and-high => the ordering is set before the
model has learned much; late-and-climbing => it is something training builds.

Question 2 -- WHAT. Two statistics of the corpus that need no model at all are available per token in
dir18's manifest: the token's unigram count in the sampled corpus and the entropy of its successor
distribution. If the width ranking is a repackaged corpus statistic, these should predict it, and the
agreement between an early checkpoint and the final one should vanish once they are partialled out.

Writes results/checkpoints_summary.json.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import D18, RESULTS

FINAL = "step143000"
N_ANCHOR = 6


def half_split(v):
    """Median width over the even-indexed anchors and over the odd-indexed ones."""
    i = np.arange(len(v))
    return np.nanmedian(v[(i % N_ANCHOR) % 2 == 0]), np.nanmedian(v[(i % N_ANCHOR) % 2 == 1])


def reliability(raw, names):
    h = np.array([half_split(np.array(raw[s], float)) for s in names])
    ok = np.isfinite(h).all(1)
    r = float(spearmanr(h[ok, 0], h[ok, 1]).statistic)
    return r, float(2 * r / (1 + r))


def corpus_stats(names):
    """Per-token unigram count and successor entropy, pooled over dir18's 1,000 pairs."""
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    acc = {}
    for p in man:
        for side in ("a", "b"):
            s = p[f"{side}_str"]
            acc.setdefault(s, []).append((p[f"count_{side}"], p[f"ent_{side}"], p[f"surp_{side}"]))
    out = {}
    for s in names:
        v = np.array(acc[s], float)
        out[s] = dict(count=float(np.median(v[:, 0])), ent=float(np.median(v[:, 1])),
                      surp=float(np.median(v[:, 2])))
    return out


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def partial_rho(a, b, C):
    """Spearman partial correlation of a and b given the columns of C (all rank-transformed)."""
    A, B = rank(a), rank(b)
    X = np.hstack([np.ones((len(A), 1)), np.column_stack([rank(c) for c in C])])
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r = spearmanr(res(A), res(B))
    return float(r.statistic), float(r.pvalue)


def main():
    d = json.load(open(f"{RESULTS}/checkpoints.json"))
    ck = d["ckpt"]
    steps = sorted(ck, key=lambda s: int(s[4:]))
    names = sorted(set.intersection(*[{s for s, x in ck[c]["w"].items() if np.isfinite(x)}
                                      for c in ck]))
    stats = corpus_stats(names)
    logc = np.log10([stats[s]["count"] for s in names])
    ent = np.array([stats[s]["ent"] for s in names])
    surp = np.array([stats[s]["surp"] for s in names])
    W = {c: np.array([ck[c]["w"][s] for s in names]) for c in steps}
    rel = {c: reliability(ck[c]["w_raw"], names) for c in steps}
    pred14 = json.load(open(f"{RESULTS}/embed.json"))["probe_pred"]
    p14 = np.array([pred14[s] for s in names])

    rows = []
    for c in steps:
        r_fin = spearmanr(W[c], W[FINAL])
        ceil = float(np.sqrt(max(rel[c][1], 0) * max(rel[FINAL][1], 0)))
        pr, pp = partial_rho(W[c], W[FINAL], [logc, ent])
        rows.append(dict(
            step=int(c[4:]), name=c, n=len(names),
            median_w=float(np.median(W[c])), sd_w=float(W[c].std(ddof=1)),
            valid=ck[c]["valid_frac"],
            half_rho=rel[c][0], reliability=rel[c][1],
            rho_final=float(r_fin.statistic), p_final=float(r_fin.pvalue),
            ceiling=ceil, disattenuated=float(r_fin.statistic / ceil) if ceil > 0 else float("nan"),
            partial_final=pr, partial_final_p=pp,
            rho_logcount=float(spearmanr(W[c], logc).statistic),
            rho_entropy=float(spearmanr(W[c], ent).statistic),
            rho_surp=float(spearmanr(W[c], surp).statistic),
            rho_lookup14=float(spearmanr(W[c], p14).statistic),
            probe_rho=ck[c]["probe"]["rho_mean"], probe_sd=ck[c]["probe"]["rho_sd"],
            probe_null=ck[c]["probe"]["null_rho_mean"]))
        print(f"{c:>10} med {rows[-1]['median_w']:.3f} sd {rows[-1]['sd_w']:.3f} "
              f"rel {rows[-1]['reliability']:.3f} | vs final {rows[-1]['rho_final']:+.3f} "
              f"(dis {rows[-1]['disattenuated']:+.3f}, partial {pr:+.3f}) | "
              f"logN {rows[-1]['rho_logcount']:+.3f} ent {rows[-1]['rho_entropy']:+.3f} | "
              f"probe {rows[-1]['probe_rho']:+.3f} lookup14 {rows[-1]['rho_lookup14']:+.3f}",
              flush=True)

    # how much of the FINAL ranking do the two free corpus statistics explain on their own?
    X = np.column_stack([np.ones(len(names)), rank(logc), rank(ent)])
    yr = rank(W[FINAL])
    beta = np.linalg.lstsq(X, yr, rcond=None)[0]
    r2 = float(1 - ((yr - X @ beta) ** 2).sum() / ((yr - yr.mean()) ** 2).sum())

    # consistency: the same model+revision measured in iteration 11's separate run
    ref = json.load(open(f"{RESULTS}/second_410m.json"))["w"]
    same = [s for s in names if np.isfinite(ref.get(s, np.nan))]
    rep = float(spearmanr([W[FINAL][names.index(s)] for s in same],
                          [ref[s] for s in same]).statistic)

    out = dict(model=d["model"], final=FINAL, tokens=names, rows=rows,
               corpus=dict(rho_logcount_ent=float(spearmanr(logc, ent).statistic),
                           final_rank_r2_from_corpus=r2),
               rerun_agreement=dict(rho=rep, n=len(same)))
    json.dump(out, open(f"{RESULTS}/checkpoints_summary.json", "w"), indent=1)
    print(f"\nfinal ranking R2 from (log count, successor entropy) ranks = {r2:.3f}")
    print(f"step143000 here vs iteration-11 410M run: rho {rep:+.4f} (n {len(same)})")
    print("wrote results/checkpoints_summary.json")


if __name__ == "__main__":
    main()
