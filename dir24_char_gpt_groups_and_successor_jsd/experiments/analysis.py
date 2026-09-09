"""S1-S4: validate dir13's stored step-30000 widths, test character-class grouping from letter
anchors, compute corpus successor JSD, and relate JSD to transition width.

Reads (read-only) ../dir13_plateau_on_grok_gpt/results/{allpairs_summary.json,allpairs_raw.npz}
and the verified tinyshakespeare corpus. Writes results/analysis.json and plots/*.png.
CPU only; no model is loaded and no interpolation is re-run.
"""
import os, sys, json, hashlib, itertools, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "..", "dir13_plateau_on_grok_gpt", "results")
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
CORPUS = "/tmp/tinyshakespeare.txt"
CORPUS_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
FREQ_MIN = 1000
N_PERM = 5000
SEED = 0

CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

CLASSES = ["lower vowel", "lower cons.", "upper vowel", "upper cons.", "punct. & digits", "space / \\n"]
CCOLOR = {"lower vowel": CVD[0], "lower cons.": CVD[1], "upper vowel": CVD[3],
          "upper cons.": CVD[4], "punct. & digits": CVD[2], "space / \\n": "0.45"}
CHATCH = {"lower vowel": "//", "lower cons.": "\\\\", "upper vowel": "xx",
          "upper cons.": "..", "punct. & digits": "++", "space / \\n": "oo"}
CMARK = {"lower vowel": "o", "lower cons.": "s", "upper vowel": "^",
         "upper cons.": "D", "punct. & digits": "P", "space / \\n": "X"}


def disp(c):
    return {"\n": "\\n", " ": "space"}.get(c, c)


def class_of(c):
    if c in " \n":
        return "space / \\n"
    if c.isupper():
        return "upper vowel" if c in "AEIOU" else "upper cons."
    if c.islower():
        return "lower vowel" if c in "aeiou" else "lower cons."
    return "punct. & digits"


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/" + name, flush=True)


def jsd_matrix(P):
    """Base-2 Jensen-Shannon divergence between every pair of rows of P (rows sum to 1)."""
    def ent(Q):
        return -np.sum(np.where(Q > 0, Q * np.log2(np.where(Q > 0, Q, 1.0)), 0.0), axis=-1)
    n = P.shape[0]
    H = ent(P)
    J = np.zeros((n, n))
    for i in range(n):
        M = 0.5 * (P[i][None, :] + P)
        J[i] = ent(M) - 0.5 * (H[i] + H)
    return np.clip(J, 0.0, None)


def successor_probs(text, chars):
    """P(next char | c) from raw bigram counts, no smoothing. Rows follow `chars` order."""
    idx = {c: k for k, c in enumerate(chars)}
    C = np.zeros((len(chars), len(chars)))
    for a, b in zip(text, text[1:]):
        C[idx[a], idx[b]] += 1
    return C, C / np.maximum(C.sum(1, keepdims=True), 1.0)


def main():
    rng = np.random.default_rng(SEED)
    out = {}

    # ---------------- S1: validate the source data ------------------------------------------
    raw = open(CORPUS, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == CORPUS_SHA, f"corpus SHA mismatch: {sha}"
    text = raw.decode("utf-8")
    train = text[:int(0.9 * len(text))]

    S = json.load(open(os.path.join(SRC, "allpairs_summary.json")))
    chars, V = S["chars"], S["vocab_size"]
    fin = S["final_block0"]
    diag = S["analysis"]["diagnostics"]
    assert chars == sorted(set(text)), "vocab order != sorted(set(corpus))"
    assert V == 65 and S["n_pairs"] == 2080 and len(fin) == 2080
    assert S["context"] == "The house was " and S["n_t"] == 50 and S["final_step"] == 30000

    freq = collections.Counter(train)
    f = np.array([freq[c] for c in chars], float)
    keep = f >= FREQ_MIN
    kept = [i for i in range(V) if keep[i]]
    letters = [i for i in kept if chars[i].isalpha()]

    W = np.full((V, V), np.nan)
    for p in fin:
        if p["w"] is not None:
            W[p["i"], p["j"]] = W[p["j"], p["i"]] = p["w"]
    kp = [(a, b) for a, b in itertools.combinations(kept, 2)]
    w_kept = np.array([W[a, b] for a, b in kp])
    w_rare = np.array([W[p["i"], p["j"]] for p in fin
                       if p["w"] is not None and not (keep[p["i"]] and keep[p["j"]])])

    out["s1_validation"] = {
        "source_dir": "../dir13_plateau_on_grok_gpt",
        "source_files": ["results/allpairs_summary.json", "results/allpairs_raw.npz"],
        "source_git_commit": "01d2501db74cd829db2d9efeada16246b8ec1efe",
        "corpus_sha256": sha, "n_chars_corpus": len(text), "n_chars_train_split": len(train),
        "vocab_size": V, "vocab_order_matches_sorted_set": True,
        "context": S["context"], "n_interp_points": S["n_t"],
        "checkpoint_step": S["final_step"], "hook": "resid_post of block 0, final position only",
        "n_pairs": S["n_pairs"], "n_width_undefined": diag["n_width_undefined"],
        "max_endpoint_err": diag["max_endpoint_err"], "max_prefix_err": diag["max_prefix_err"],
        "max_d0": diag["max_d0"], "min_d1": diag["min_d1"],
        "median_w_all_2080": float(np.median([p["w"] for p in fin])),
        "freq_min": FREQ_MIN, "n_well_trained": int(keep.sum()), "n_letter_anchors": len(letters),
        "n_well_trained_pairs": len(kp), "median_w_well_trained": float(np.median(w_kept)),
        "n_rare_pairs": int(w_rare.size), "median_w_rare": float(np.median(w_rare)),
        "dropped_chars": [disp(chars[i]) for i in range(V) if not keep[i]],
        "median_w_init_ckpt": float(np.median([p["w"] for p in S["init_block0"]])),
    }
    print("S1 ok:", out["s1_validation"]["n_well_trained"], "well-trained chars,",
          len(kp), "pairs, median w", round(float(np.median(w_kept)), 4), flush=True)

    # ---------------- S2: character classes seen from letter anchors -------------------------
    partner_class = {}
    med = {}          # (anchor, class) -> median w
    for a in letters:
        for g in CLASSES:
            ps = [b for b in kept if b != a and class_of(chars[b]) == g]
            partner_class[(a, g)] = ps
            med[(a, g)] = float(np.median([W[a, b] for b in ps]))
    M = np.array([[med[(a, g)] for g in CLASSES] for a in letters])       # 43 x 6
    ranks = np.array([stats.rankdata(r) for r in M])
    fried = stats.friedmanchisquare(*M.T)
    kw = fried.statistic / (len(letters) * (len(CLASSES) - 1))            # Kendall's W

    # same agreement after removing each anchor's partner-frequency trend (class confounds with
    # how common its members are): rank w within an anchor, regress out rank log-frequency.
    Mr = np.zeros_like(M)
    for k, a in enumerate(letters):
        ps = [b for b in kept if b != a]
        ry = stats.rankdata([W[a, b] for b in ps])
        rz = stats.rankdata(np.log10([f[b] for b in ps]))
        Z = np.stack([np.ones_like(rz), rz], 1)
        res = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
        rmap = dict(zip(ps, res))
        Mr[k] = [np.median([rmap[b] for b in partner_class[(a, g)]]) for g in CLASSES]
    fried_r = stats.friedmanchisquare(*Mr.T)
    kw_r = fried_r.statistic / (len(letters) * (len(CLASSES) - 1))

    out["s2_classes"] = {
        "n_anchors": len(letters), "anchors": [chars[i] for i in letters], "classes": CLASSES,
        "n_partners_per_anchor": 52,
        "mean_class_median_w": [float(x) for x in M.mean(0)],
        "median_class_median_w": [float(x) for x in np.median(M, 0)],
        "iqr_class_median_w": [[float(np.percentile(M[:, k], 25)), float(np.percentile(M[:, k], 75))]
                               for k in range(len(CLASSES))],
        "mean_rank": [float(x) for x in ranks.mean(0)],
        "kendall_w": float(kw), "friedman_chi2": float(fried.statistic),
        "friedman_p": float(fried.pvalue),
        "freq_residualized": {"mean_rank": [float(x) for x in
                                            np.array([stats.rankdata(r) for r in Mr]).mean(0)],
                              "kendall_w": float(kw_r), "friedman_chi2": float(fried_r.statistic),
                              "friedman_p": float(fried_r.pvalue)},
        "frac_anchors_lower_vowel_below_upper_cons": float(np.mean(M[:, 0] < M[:, 3])),
        "anchor_overall_median_w": {
            "min": float(min(np.median([W[a, b] for b in kept if b != a]) for a in letters)),
            "max": float(max(np.median([W[a, b] for b in kept if b != a]) for a in letters))},
    }
    # raw pair-to-pair spread inside each class block, for judging whether the class median is
    # representative of the cells it summarises (descriptive spread of the raw widths, no new score)
    blk = {}
    for g in CLASSES:
        iq, rg = [], []
        for a in letters:
            v = np.array([W[a, b] for b in partner_class[(a, g)]])
            iq.append(np.percentile(v, 75) - np.percentile(v, 25))
            rg.append(v.max() - v.min())
        blk[g] = {"mean_within_anchor_iqr": float(np.mean(iq)),
                  "mean_within_anchor_range": float(np.mean(rg)),
                  "n_members": int(sum(class_of(chars[b]) == g for b in kept))}
    out["s2_classes"]["raw_block_spread"] = blk
    out["s2_classes"]["between_class_spread_of_mean_medians"] = float(M.mean(0).max() - M.mean(0).min())
    out["s2_classes"]["partner_mean_w_over_anchors"] = {
        disp(chars[b]): float(np.mean([W[a, b] for a in letters if a != b])) for b in kept}

    # four non-cherry-picked example anchors: most frequent eligible member of each letter class
    ex_anchors = []
    for g in ["lower vowel", "lower cons.", "upper vowel", "upper cons."]:
        cand = [i for i in letters if class_of(chars[i]) == g]
        ex_anchors.append(max(cand, key=lambda i: f[i]))
    out["s2_classes"]["example_anchors"] = [chars[i] for i in ex_anchors]

    # ---------------- S3: successor JSD and its reliability ----------------------------------
    Cnt, P = successor_probs(train, chars)
    J = jsd_matrix(P)
    h1, h2 = train[:len(train) // 2], train[len(train) // 2:]
    _, P1 = successor_probs(h1, chars)
    _, P2 = successor_probs(h2, chars)
    J1, J2 = jsd_matrix(P1), jsd_matrix(P2)
    j_pairs = np.array([J[a, b] for a, b in kp])
    r_half = stats.spearmanr([J1[a, b] for a, b in kp], [J2[a, b] for a, b in kp])
    self_j = np.array([jsd_matrix(np.stack([P1[i], P2[i]]))[0, 1] for i in kept])

    out["s3_jsd"] = {
        "n_bigrams_train": int(Cnt.sum()), "smoothing": "none",
        "n_pairs": len(kp), "median_J": float(np.median(j_pairs)),
        "min_J": float(j_pairs.min()), "max_J": float(j_pairs.max()),
        "splithalf_spearman_rho": float(r_half.statistic), "splithalf_spearman_p": float(r_half.pvalue),
        "median_same_char_splithalf_J": float(np.median(self_j)),
        "max_same_char_splithalf_J": float(self_j.max()),
        "noise_floor_vs_median_ratio": float(np.median(self_j) / np.median(j_pairs)),
        "min_train_count_well_trained": int(Cnt.sum(1)[kept].min()),
    }
    print("S3 ok: split-half rho", round(float(r_half.statistic), 3),
          "noise floor", round(float(np.median(self_j)), 4), flush=True)

    # ---------------- S4: does successor JSD predict width? ----------------------------------
    rho_pool = stats.spearmanr(j_pairs, w_kept)
    # permutation null that preserves both pairwise matrices: relabel the 53 characters
    idx_of = {c: k for k, c in enumerate(kept)}
    ii = np.array([idx_of[a] for a, b in kp]); jj = np.array([idx_of[b] for a, b in kp])
    Jk = J[np.ix_(kept, kept)]
    rw = stats.rankdata(w_kept)
    null = np.empty(N_PERM)
    for t in range(N_PERM):
        pm = rng.permutation(len(kept))
        jp = Jk[np.ix_(pm, pm)][ii, jj]
        null[t] = stats.spearmanr(jp, rw).statistic
    p_perm = float((np.sum(np.abs(null) >= abs(rho_pool.statistic)) + 1) / (N_PERM + 1))

    def anchor_rhos(Jm):
        return np.array([stats.spearmanr([Jm[a, b] for b in kept if b != a],
                                         [W[a, b] for b in kept if b != a]).statistic
                         for a in letters])

    per_anchor = []
    for a in letters:
        ps = [b for b in kept if b != a]
        r = stats.spearmanr([J[a, b] for b in ps], [W[a, b] for b in ps])
        per_anchor.append({"char": chars[a], "rho": float(r.statistic), "p": float(r.pvalue),
                           "n": len(ps)})
    pr = np.array([d["rho"] for d in per_anchor])
    # same character-relabel null, applied to the within-anchor statistic (anchors share partners,
    # so a sign test over the 43 anchors would treat dependent numbers as independent)
    Jfull = np.zeros((V, V))
    Jfull[np.ix_(kept, kept)] = Jk
    null_pa = np.empty(N_PERM // 5)
    for t in range(null_pa.size):
        pm = rng.permutation(len(kept))
        Jp = np.zeros((V, V))
        Jp[np.ix_(kept, kept)] = Jk[np.ix_(pm, pm)]
        null_pa[t] = np.median(anchor_rhos(Jp))
    p_perm_pa = float((np.sum(null_pa <= np.median(pr)) + 1) / (null_pa.size + 1))

    # frequency confound: within each anchor, partial the partner's log training frequency out of
    # both J and w by rank residualisation, then correlate the residuals
    def resid(y, z):
        ry, rz = stats.rankdata(y), stats.rankdata(z)
        Z = np.stack([np.ones_like(rz), rz], 1)
        return ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    pr_part = []
    for a in letters:
        ps = [b for b in kept if b != a]
        lf = np.log10([f[b] for b in ps])
        pr_part.append(stats.spearmanr(resid([J[a, b] for b in ps], lf),
                                       resid([W[a, b] for b in ps], lf)).statistic)
    pr_part = np.array(pr_part)

    rare_pairs = [(p["i"], p["j"]) for p in fin
                  if p["w"] is not None and not (keep[p["i"]] and keep[p["j"]])]
    rho_rare = stats.spearmanr([J[a, b] for a, b in rare_pairs], [W[a, b] for a, b in rare_pairs])

    # equal-count binned medians (superseded by the fixed-width bins below; kept for the record)
    o = np.argsort(j_pairs)
    bins = np.array_split(o, 10)
    binned = [[float(np.median(j_pairs[b])), float(np.median(w_kept[b])), int(b.size)] for b in bins]

    # fixed-width JSD bins [0.0,0.1) ... [0.9,1.0]: mean width, count, and the same excluding
    # punctuation-punctuation pairs (both endpoints in the punctuation-and-digits class)
    is_pp = np.array([class_of(chars[a]) == "punct. & digits" and
                      class_of(chars[b]) == "punct. & digits" for a, b in kp])
    edges = np.round(np.arange(0, 1.0001, 0.1), 3)
    bidx = np.clip(np.digitize(j_pairs, edges[1:-1]), 0, 9)
    fixed = []
    for k in range(10):
        m = bidx == k
        mn = m & ~is_pp
        fixed.append({"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": int(m.sum()),
                      "mean_w": float(w_kept[m].mean()) if m.any() else None,
                      "median_w": float(np.median(w_kept[m])) if m.any() else None,
                      "n_punct_punct": int((m & is_pp).sum()),
                      "n_excl_pp": int(mn.sum()),
                      "mean_w_excl_pp": float(w_kept[mn].mean()) if mn.any() else None})
    low = bidx == 0
    low_pairs = [{"pair": [disp(chars[a]), disp(chars[b])], "J": float(J[a, b]), "w": float(W[a, b]),
                  "both_punct": bool(class_of(chars[a]) == "punct. & digits" and
                                     class_of(chars[b]) == "punct. & digits")}
                 for (a, b), m in zip(kp, low) if m]
    low_pairs.sort(key=lambda r: r["J"])

    out["s4_jsd_vs_width"] = {
        "pooled": {"n": len(kp), "spearman_rho": float(rho_pool.statistic),
                   "naive_p": float(rho_pool.pvalue), "perm_p": p_perm, "n_perm": N_PERM,
                   "perm_null_sd": float(null.std()),
                   "perm_null_q025_q975": [float(np.percentile(null, 2.5)),
                                           float(np.percentile(null, 97.5))]},
        "binned_medians_J_w_n": binned,
        "fixed_width_bins": fixed,
        "low_jsd_bin_pairs": low_pairs,
        "punct_punct": {"n": int(is_pp.sum()), "mean_w": float(w_kept[is_pp].mean()),
                        "median_w": float(np.median(w_kept[is_pp])),
                        "mean_w_other": float(w_kept[~is_pp].mean()),
                        "median_w_other": float(np.median(w_kept[~is_pp])),
                        "members": [disp(chars[i]) for i in kept
                                    if class_of(chars[i]) == "punct. & digits"]},
        "per_anchor": {"n_anchors": len(letters), "median_rho": float(np.median(pr)),
                       "iqr": [float(np.percentile(pr, 25)), float(np.percentile(pr, 75))],
                       "n_negative": int((pr < 0).sum()), "min_rho": float(pr.min()),
                       "max_rho": float(pr.max()),
                       "n_p_below_05": int(sum(d["p"] < 0.05 for d in per_anchor)),
                       "perm_p_median_rho": p_perm_pa, "n_perm": int(null_pa.size),
                       "perm_null_median_rho_q025_q975": [float(np.percentile(null_pa, 2.5)),
                                                          float(np.percentile(null_pa, 97.5))],
                       "rows": per_anchor},
        "per_anchor_partial_out_partner_freq": {
            "median_rho": float(np.median(pr_part)),
            "iqr": [float(np.percentile(pr_part, 25)), float(np.percentile(pr_part, 75))],
            "n_negative": int((pr_part < 0).sum()), "n_anchors": len(letters)},
        "rare_sensitivity": {"n": len(rare_pairs), "spearman_rho": float(rho_rare.statistic),
                             "naive_p": float(rho_rare.pvalue)},
        "between_anchor": {
            "spearman_medJ_vs_medW_over_anchors": [
                float(stats.spearmanr(
                    [np.median([J[a, b] for b in kept if b != a]) for a in letters],
                    [np.median([W[a, b] for b in kept if b != a]) for a in letters]).statistic),
                float(stats.spearmanr(
                    [np.median([J[a, b] for b in kept if b != a]) for a in letters],
                    [np.median([W[a, b] for b in kept if b != a]) for a in letters]).pvalue)],
            "n_anchors": len(letters)},
        "class_mean_J_from_letter_anchors": [
            float(np.mean([np.median([J[a, b] for b in partner_class[(a, g)]]) for a in letters]))
            for g in CLASSES],
    }
    print("S4 ok: pooled rho", round(float(rho_pool.statistic), 3), "perm p", p_perm,
          "| per-anchor median rho", round(float(np.median(pr)), 3),
          "negatives", int((pr < 0).sum()), flush=True)

    json.dump(out, open(os.path.join(RES, "analysis.json"), "w"), indent=1)

    # ================= FIGURES ==============================================================
    R = np.load(os.path.join(SRC, "allpairs_raw.npz"))
    ts = R["ts"]

    # Fig 1 - what d(t) and w mean
    wk = {(a, b): W[a, b] for a, b in kp}
    srt = sorted(wk, key=lambda k: wk[k])
    picks = [srt[0], srt[len(srt) // 2], srt[-1]]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    curves = np.stack([R[f"final|L0|d|{a}_{b}"] for a, b in kp])
    ax.plot(ts, np.median(curves, 0), color="0.5", lw=3, alpha=.6, zorder=1,
            label=f"median of all {len(kp)} pairs")
    for k, ((a, b), ls, mk) in enumerate(zip(picks, ["-", "--", "-."], ["o", "s", "^"])):
        d = R[f"final|L0|d|{a}_{b}"]
        ax.plot(ts, d, ls=ls, marker=mk, ms=3.5, color=CVD[k], zorder=3,
                label=f"'{disp(chars[a])}' to '{disp(chars[b])}'   $w$ = {W[a, b]:.2f}")
    ax.plot([0, 1], [0, 1], ls=":", color="0.3", lw=1.2, zorder=2, label="straight line ($w$ = 0.80)")
    for y in (0.1, 0.9):
        ax.axhline(y, color="0.75", lw=.8, zorder=0)
    ax.text(0.02, 0.93, "$d$ = 0.9", fontsize=7.5, color="0.4")
    ax.text(0.02, 0.045, "$d$ = 0.1", fontsize=7.5, color="0.4")
    ax.set_xlabel("interpolation position $t$   (0 = endpoint $a$, 1 = endpoint $b$)")
    ax.set_ylabel("relative output distance $d(t)$")
    ax.set_title("Transition width $w_{10\\to90}$ = the span of $t$ over which\n"
                 "the output moves from 10% to 90% of the way to endpoint $b$", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=.95)
    fig.text(0.5, -0.03, 'context "The house was ", block-0 residual stream, step-30,000 character GPT',
             ha="center", fontsize=7.5)
    save(fig, "fig1_width_definition.png")

    # Fig 2 - raw 43 x 53 pairwise width matrix, with the class-median summary as a small panel
    row_ord, row_sep, row_lab = [], [], []
    for g in CLASSES:
        grp = sorted([a for a in letters if class_of(chars[a]) == g], key=lambda i: chars[i].lower())
        if grp:
            row_lab.append((len(row_ord) + len(grp) / 2 - 0.5, g))
            row_ord += grp
            row_sep.append(len(row_ord) - 0.5)
    col_ord, col_sep, col_lab = [], [], []
    for g in CLASSES:
        grp = sorted([b for b in kept if class_of(chars[b]) == g], key=lambda i: disp(chars[i]))
        col_lab.append((len(col_ord) + len(grp) / 2 - 0.5, g))
        col_ord += grp
        col_sep.append(len(col_ord) - 0.5)
    H = np.array([[W[a, b] if a != b else np.nan for b in col_ord] for a in row_ord])

    fig = plt.figure(figsize=(13.4, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("white")
    im = ax.imshow(H, cmap=cmap, aspect="auto", interpolation="nearest")
    for x in col_sep[:-1]:
        ax.axvline(x, color="k", lw=1.4)
    for y in row_sep[:-1]:
        ax.axhline(y, color="k", lw=1.0)
    ax.set_xticks(range(len(col_ord)))
    ax.set_xticklabels([disp(chars[b]) for b in col_ord], fontsize=6.5, rotation=90)
    ax.set_yticks(range(len(row_ord)))
    ax.set_yticklabels([chars[a] for a in row_ord], fontsize=6.5)
    for x, g in col_lab:
        ax.text(x, -1.6, g, ha="center", va="bottom",
                fontsize=7.5 if g not in ("punct. & digits", "space / \\n") else 6.3)
    for y, g in row_lab:
        ax.text(-4.2, y, g, ha="center", va="center", fontsize=7.5, rotation=90)
    ax.set_xlabel("partner character $c_{\\mathrm{partner}}$   (53 well-trained characters, "
                  "grouped by class, alphabetical within class)", fontsize=8.5)
    ax.set_ylabel("anchor character $c_{\\mathrm{anchor}}$   (43 letters)", fontsize=8.5, labelpad=34)
    ax.set_title("Every measured pair: transition width $w(c_{\\mathrm{anchor}},c_{\\mathrm{partner}})$"
                 "   (white = self-pair, not measured)", fontsize=9.5, pad=18)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("$w_{10\\to90}$  (small = abrupt switch)", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ypos = np.arange(len(CLASSES))[::-1]
    bp = ax2.boxplot([M[:, k] for k in range(len(CLASSES))], positions=ypos, vert=False, widths=.6,
                     patch_artist=True, medianprops=dict(color="k", lw=1.6),
                     flierprops=dict(marker=".", ms=3))
    for k, box in enumerate(bp["boxes"]):
        box.set(facecolor=CCOLOR[CLASSES[k]], alpha=.55, hatch=CHATCH[CLASSES[k]], edgecolor="0.2")
    for k in range(len(CLASSES)):
        ax2.plot(M[:, k], np.full(len(letters), ypos[k]) + rng.uniform(-.14, .14, len(letters)),
                 ls="none", marker=CMARK[CLASSES[k]], ms=2.6, color="0.25", alpha=.65)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(CLASSES, fontsize=7.5)
    ax2.set_xlabel("median $w$ over that anchor's\npartners in the class", fontsize=8)
    ax2.set_title("Class median, one point per anchor\n(same class order as rows and columns)",
                  fontsize=9)
    save(fig, "fig2_width_heatmap.png")

    # kept for RESULTS.md: the four-anchor view of the same class medians
    ordc = [CLASSES[k] for k in np.argsort(ranks.mean(0))]   # display order: narrowest first
    Mo = np.array([[med[(a, g)] for g in ordc] for a in letters])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axes[0]
    x = np.arange(len(ordc))
    for k, a in enumerate(ex_anchors):
        ax.plot(x + (k - 1.5) * 0.05, [med[(a, g)] for g in ordc], ls=["-", "--", "-.", ":"][k],
                marker=["o", "s", "^", "D"][k], ms=6, color=CVD[k],
                label=f"anchor '{chars[a]}' ({class_of(chars[a])})")
    ax.set_xticks(x)
    ax.set_xticklabels(ordc, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("median $w_{10\\to90}$ over the anchor's partners in that class")
    ax.set_xlabel("character class of the partner (classes ordered by mean rank over all anchors)")
    ax.set_title("Four anchors: the most frequent letter of each letter class\n"
                 "single anchors are noisy; no one letter settles the question", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    bp = ax.boxplot([Mo[:, k] for k in range(len(ordc))], widths=.6, patch_artist=True,
                    medianprops=dict(color="k", lw=1.6), flierprops=dict(marker=".", ms=3))
    for k, box in enumerate(bp["boxes"]):
        box.set(facecolor=CCOLOR[ordc[k]], alpha=.55, hatch=CHATCH[ordc[k]], edgecolor="0.2")
    for k in range(len(ordc)):
        ax.plot(np.full(len(letters), k + 1) + rng.uniform(-.13, .13, len(letters)), Mo[:, k],
                ls="none", marker=CMARK[ordc[k]], ms=2.6, color="0.25", alpha=.65)
    ax.set_xticklabels(ordc, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("median $w_{10\\to90}$ per anchor")
    ax.set_xlabel("character class of the partner (same order as the left panel)")
    ax.set_title(f"All {len(letters)} letter anchors (one point per anchor)\n"
                 f"Kendall's $W$ = {kw:.2f}, Friedman $p$ = {fried.pvalue:.1e}", fontsize=9.5)
    save(fig, "fig2b_class_widths.png")

    # Fig 3 - successor JSD vs width in fixed-width JSD bins
    ctr = np.array([r["lo"] + 0.05 for r in fixed])
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.plot(j_pairs[~is_pp], w_kept[~is_pp], ls="none", marker="o", ms=2.4, alpha=.22,
            color=CVD[0],
            label=f"one pair, at least one non-punctuation endpoint (n = {int((~is_pp).sum())})")
    ax.plot(j_pairs[is_pp], w_kept[is_pp], ls="none", marker="D", ms=5.5, alpha=.95,
            color=CVD[1], markeredgecolor="0.2", markeredgewidth=.4,
            label=f"one pair, both endpoints punctuation (n = {int(is_pp.sum())})")
    ax.plot(ctr, [r["mean_w"] for r in fixed], ls="-", marker="s", ms=7, lw=2, color="k",
            zorder=5, label="bin mean width, all pairs")
    ax.plot(ctr, [r["mean_w_excl_pp"] for r in fixed], ls="--", marker="^", ms=6, lw=1.8,
            color=CVD[2], zorder=6, label="bin mean width, punctuation-punctuation pairs removed")
    for c, r in zip(ctr, fixed):
        lab = f"n = {r['n']}" + (f"  ({r['n_punct_punct']} p-p)" if r["n_punct_punct"] else "")
        ax.text(c, 0.755, lab, ha="center", va="bottom", fontsize=6.8, rotation=90, color="0.25")
    ax.set_xticks(np.round(np.arange(0, 1.001, 0.1), 1))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.10, 1.00)
    ax.set_xlabel("successor JSD $J(c_{\\mathrm{anchor}},c_{\\mathrm{partner}})$ between the two "
                  "endpoints (bits), fixed-width bins of 0.1")
    ax.set_ylabel("transition width $w_{10\\to90}$")
    ax.set_title("Width against successor divergence, all 1,378 well-trained pairs\n"
                 "the lowest-JSD bin is wide because it is almost entirely punctuation pairs",
                 fontsize=9.5)
    ax.legend(fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              framealpha=.95)
    save(fig, "fig3_jsd_vs_width.png")

    # supporting figure for RESULTS.md: the per-anchor Spearman check
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    lo95, hi95 = np.percentile(null_pa, 2.5), np.percentile(null_pa, 97.5)
    ax.axvspan(lo95, hi95, color="0.75", alpha=.5, zorder=0,
               label="95% range of the median under\ncharacter-relabel permutation")
    ax.hist(pr, bins=np.linspace(-0.8, 0.6, 29), color=CVD[0], alpha=.65, hatch="//",
            edgecolor="0.2", zorder=2)
    ax.axvline(0, color="0.3", lw=1.2, ls=":")
    ax.axvline(np.median(pr), color=CVD[1], lw=2, ls="--", label=f"median = {np.median(pr):.2f}")
    ax.axvline(rho_pool.statistic, color=CVD[2], lw=2, ls="-.",
               label=f"pooled = {rho_pool.statistic:.2f}")
    ax.set_xlim(-0.8, 0.6)
    ax.set_xlabel("Spearman $\\rho$ between $J$ and $w$ across one anchor's 52 partners")
    ax.set_ylabel("number of letter anchors")
    ax.set_title(f"Within a fixed letter anchor ({len(letters)} anchors)\n"
                 f"{int((pr < 0).sum())} of {len(letters)} anchors have $\\rho$ < 0", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "fig_s2_per_anchor_rho.png")

    # supporting figure for RESULTS.md: successor-JSD reliability
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot([J1[a, b] for a, b in kp], [J2[a, b] for a, b in kp], ls="none", marker="o",
            ms=2.2, alpha=.3, color=CVD[0], label=f"pair JSD (n = {len(kp)})")
    lim = [0, max(np.max(J1[np.ix_(kept, kept)]), np.max(J2[np.ix_(kept, kept)])) * 1.05]
    ax.plot(lim, lim, ls=":", color="0.3", lw=1.2, label="equality")
    ax.plot(self_j, self_j, ls="none", marker="^", ms=5, color=CVD[1],
            label=f"same character, two halves\n(median {np.median(self_j):.3f} bits)")
    ax.set_xlabel("$J$ from the first half of the training split (bits)")
    ax.set_ylabel("$J$ from the second half (bits)")
    ax.set_title(f"Successor JSD is stable across halves\nSpearman $\\rho$ = {r_half.statistic:.3f}",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "fig_s1_jsd_reliability.png")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
