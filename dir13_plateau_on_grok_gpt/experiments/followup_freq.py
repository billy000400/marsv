"""Operator feedback #5 — training-frequency and semantic-group analyses of the all-pairs sweep.

Items addressed (see human_feedback_5.txt):
  1. pairwise width matrix with the undertrained characters (< 1000 occurrences in the training
     split) removed, plus the width-vs-frequency trend;
  2. example d(t) curves for well-trained endpoints, plotted against their own mirror image so the
     shape asymmetry can be seen (no metrics reported);
  4. every char-level figure's context and per-cell sample count, stated on the figures;
  6. one well-trained letter against all other well-trained characters, grouped by semantic class,
     with a cross-letter concordance test.

Reads results/allpairs_summary.json + results/allpairs_raw.npz + the corpus; writes
results/followup_summary.json and plots/followup_*.png.
"""
import os, sys, json, collections, itertools
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD, MARKERS, HATCHES, REF_DIAG, REF_RULE, use_cvd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
CORPUS = "/tmp/tinyshakespeare.txt"
FREQ_MIN = 1000                       # operator's "undertrained" cut
CONTEXT_NOTE = ('context = "The house was " (14 characters), patch at the final position only; '
                'step-30000 character GPT, interpolation block 0')
GROUPS = ["lower vowel", "lower cons.", "upper vowel", "upper cons.", "punct. & digits", "space / \\n"]
# 6 groups > 5 palette hues (CLAUDE.md rule 13): hue carries vowel / consonant / punctuation, the
# tail group (space) is gray, and every group also gets its own hatch so the figures survive
# grayscale printing.
GCOLOR = {"lower vowel": CVD[0], "upper vowel": CVD[0], "lower cons.": CVD[1], "upper cons.": CVD[1],
          "punct. & digits": CVD[2], "space / \\n": "0.55"}
GHATCH = {"lower vowel": "//", "upper vowel": "xx", "lower cons.": "\\\\", "upper cons.": "..",
          "punct. & digits": "++", "space / \\n": "oo"}
GMARK = {"lower vowel": "o", "upper vowel": "^", "lower cons.": "s", "upper cons.": "D",
         "punct. & digits": "P", "space / \\n": "X"}
LINEAR_W = 0.80
STRICT_W = 0.25


def disp(c):
    return {"\n": "\\n", " ": "␣"}.get(c, c)


def group_of(c):
    if c in " \n":
        return "space / \\n"
    if c.isupper():
        return "upper vowel" if c in "AEIOU" else "upper cons."
    if c.islower():
        return "lower vowel" if c in "aeiou" else "lower cons."
    return "punct. & digits"


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


def residual_ranks(y, z):
    """rank(y) with rank(z) linearly regressed out — a Spearman partialling."""
    ry, rz = stats.rankdata(y), stats.rankdata(z)
    Z = np.stack([np.ones_like(rz), rz], 1)
    return ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]


def main():
    use_cvd()
    S = json.load(open(os.path.join(RES, "allpairs_summary.json")))
    A = S["analysis"]
    R = np.load(os.path.join(RES, "allpairs_raw.npz"))
    ts = R["ts"]
    chars, V = S["chars"], S["vocab_size"]
    fin = S["final_block0"]
    pc = {p["idx"]: p for p in A["per_char"]}

    text = open(CORPUS, "rb").read().decode("utf-8")
    freq = collections.Counter(text[:int(0.9 * len(text))])
    f = np.array([freq[c] for c in chars], dtype=float)
    keep = f >= FREQ_MIN
    kept = [i for i in range(V) if keep[i]]
    W = {}
    for p in fin:
        if p["w"] is not None:
            W[(p["i"], p["j"])] = W[(p["j"], p["i"])] = p["w"]

    out = {"freq_min": FREQ_MIN, "n_kept": int(keep.sum()), "n_dropped": int((~keep).sum()),
           "dropped": [{"char": disp(chars[i]), "freq": int(f[i])} for i in range(V) if not keep[i]],
           "context": S["context"], "n_t": S["n_t"]}

    # ---- item 1: widths with undertrained characters removed, and the frequency trend ----------
    w_kept = np.array([W[(a, b)] for a, b in itertools.combinations(kept, 2) if (a, b) in W])
    w_low = np.array([p["w"] for p in fin if p["w"] is not None and not (keep[p["i"]] and keep[p["j"]])])
    w_all = np.array([p["w"] for p in fin if p["w"] is not None])
    med_w = np.array([pc[i]["med_w"] for i in range(V)])
    med_w_kept = np.array([np.median([W[(i, j)] for j in kept if j != i and (i, j) in W]) for i in kept])
    rho_all = stats.spearmanr(np.log10(f), med_w)
    rho_kept = stats.spearmanr(np.log10(f[keep]), med_w_kept)
    u = stats.mannwhitneyu(w_kept, w_low, alternative="less")
    out["item1"] = {
        "median_w_all_pairs": float(np.median(w_all)), "n_all": int(w_all.size),
        "median_w_kept_pairs": float(np.median(w_kept)), "n_kept_pairs": int(w_kept.size),
        "median_w_pairs_with_undertrained": float(np.median(w_low)), "n_low_pairs": int(w_low.size),
        "mannwhitney_p": float(u.pvalue),
        "spearman_medw_vs_log_freq_all65": [float(rho_all.statistic), float(rho_all.pvalue)],
        "spearman_medw_vs_log_freq_kept53": [float(rho_kept.statistic), float(rho_kept.pvalue)],
        "strict_frac_all": float(np.mean([p["plateau"] for p in fin])),
        "strict_frac_kept": float(np.mean([p["plateau"] for p in fin if keep[p["i"]] and keep[p["j"]]])),
    }

    # Figure F1 — 53x53 width matrix, class blocks, frequency-sorted inside each block
    order = [i for g in GROUPS for i in sorted((k for k in kept if group_of(chars[k]) == g),
                                               key=lambda k: -f[k])]
    pos = {c: k for k, c in enumerate(order)}
    n = len(order)
    M = np.full((n, n), np.nan)
    for a, b in itertools.combinations(kept, 2):
        if (a, b) in W:
            M[pos[a], pos[b]] = M[pos[b], pos[a]] = W[(a, b)]
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(M, cmap="viridis", vmin=np.nanmin(w_all), vmax=np.nanmax(w_all))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([disp(chars[i]) for i in order], fontsize=6, rotation=90)
    ax.set_yticklabels([disp(chars[i]) for i in order], fontsize=6)
    b = 0
    for g in GROUPS[:-1]:
        b += sum(group_of(chars[i]) == g for i in order)
        ax.axhline(b - .5, color="w", lw=1.2); ax.axvline(b - .5, color="w", lw=1.2)
    start = 0
    for gk, g in enumerate(GROUPS):
        k = sum(group_of(chars[i]) == g for i in order)
        ax.text(start + k / 2 - .5, -1.2 - 2.2 * (gk % 2), g, ha="center", fontsize=7.5)
        start += k
    ax.set_xlabel("character B (within each block: most frequent on the left)")
    ax.set_ylabel("character A (within each block: most frequent at the top)")
    ax.set_title(f"Transition width $w_{{10\\to90}}$, {n} well-trained characters only "
                 f"($\\geq${FREQ_MIN} in train split)\n"
                 f"{int(w_kept.size)} pairs; each cell = 1 pair = 1 $d(t)$ curve of "
                 f"{S['n_t']} interpolation points", fontsize=9.5, pad=34)
    fig.text(0.5, -0.02, CONTEXT_NOTE, ha="center", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("$w_{10\\to90}$  (0 = instant switch, 0.80 = straight line)")
    save(fig, "followup_width_matrix_trained.png")

    # Figure F2 — the frequency trend
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    ax = axes[0]
    for g in GROUPS:
        idx = [i for i in range(V) if group_of(chars[i]) == g]
        hi = [i for i in idx if keep[i]]
        lo = [i for i in idx if not keep[i]]
        ax.plot(f[hi], med_w[hi], ls="none", marker=GMARK[g], ms=6,
                color=GCOLOR[g], label=f"{g} (well trained)")
        if lo:
            ax.plot(f[lo], med_w[lo], ls="none", marker=GMARK[g], ms=6, mfc="none",
                    mew=1.4, color=GCOLOR[g], label=f"{g} (undertrained, open)")
    for i in range(V):
        if not keep[i]:
            ax.annotate(disp(chars[i]), (f[i], med_w[i]), fontsize=7,
                        textcoords="offset points", xytext=(4, 3))
    ax.axvline(FREQ_MIN, **REF_RULE)
    ax.text(FREQ_MIN * 1.6, 0.575, f"dotted line: undertrained cut\nat {FREQ_MIN} occurrences",
            fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0.25, 0.63)
    ax.set_xlabel("training-split occurrences of character $c$ (log scale)")
    ax.set_ylabel("median $w_{10\\to90}$ over $c$'s 64 partners")
    ax.set_title(f"Rarer characters have wider transitions\nSpearman $\\rho$ = "
                 f"{rho_all.statistic:.2f} (p = {rho_all.pvalue:.1e}, n = 65 characters); "
                 f"well-trained only: $\\rho$ = {rho_kept.statistic:.2f} "
                 f"(p = {rho_kept.pvalue:.0e}, n = {int(keep.sum())})", fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False)

    ax = axes[1]
    bins = np.linspace(0.1, 1.0, 46)
    ax.hist(w_low, bins=bins, color=CVD[1], alpha=.6, hatch=HATCHES[1], edgecolor="0.25",
            label=f"pairs touching an undertrained char (n = {w_low.size}), median {np.median(w_low):.3f}")
    ax.hist(w_kept, bins=bins, color=CVD[0], alpha=.6, hatch=HATCHES[0], edgecolor="0.25",
            label=f"both endpoints well trained (n = {w_kept.size}), median {np.median(w_kept):.3f}")
    ax.axvline(LINEAR_W, **REF_DIAG); ax.axvline(STRICT_W, **REF_RULE)
    ymax = ax.get_ylim()[1]
    ax.text(LINEAR_W - .015, ymax * .55, "straight-line\nreference 0.80", fontsize=8,
            ha="right", color="0.35")
    ax.text(STRICT_W - .015, ymax * .55, "strict\nrule 0.25", fontsize=8, ha="right")
    ax.set_xlabel("$w_{10\\to90}$"); ax.set_ylabel("number of pairs")
    ax.set_title(f"Dropping the 12 undertrained characters removes the wide tail\n"
                 f"Mann–Whitney one-sided p = {u.pvalue:.1e}", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(CONTEXT_NOTE, fontsize=8.5, y=1.02)
    save(fig, "followup_width_vs_freq.png")

    # ---- item 2: example curves, well-trained endpoints, against their own mirror --------------
    ex = [("e", "o"), ("t", "s"), (" ", "e"), (".", ","), ("T", "A"), ("a", ".")]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.8), sharex=True, sharey=True)
    for ax, (ca, cb) in zip(axes.ravel(), ex):
        i, j = chars.index(ca), chars.index(cb)
        key = f"final|L0|d|{min(i,j)}_{max(i,j)}"
        d = R[key]
        if i > j:                                        # orient so ca sits at t = 0
            d = 1 - d[::-1]
        ax.plot(ts, d, color=CVD[0], lw=1.9, marker="o", ms=2.6,
                label=f"$d(t)$: '{disp(ca)}' $\\to$ '{disp(cb)}'")
        ax.plot(ts, 1 - d[::-1], color=CVD[1], lw=1.6, ls="--",
                label="its mirror $1-d(1-t)$")
        ax.plot([0, 1], [0, 1], **REF_DIAG)
        ax.axhline(0.5, color="0.8", lw=.8)
        ax.set_title(f"'{disp(ca)}' ({int(f[i])} occ.)  $\\to$  '{disp(cb)}' ({int(f[j])} occ.)",
                     fontsize=9)
        ax.set_ylim(-.03, 1.03)
        ax.legend(fontsize=7.5, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("interpolation position $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("relative distance $d(t)$")
    fig.suptitle("Six example pairs with BOTH endpoints well trained: solid = measured curve, "
                 "dashed = its own mirror image\n"
                 "(a curve that lies on its mirror is symmetric about $t=0.5$; gaps between the two "
                 "are the asymmetry)\n" + CONTEXT_NOTE + "; 1 curve per panel", fontsize=9)
    save(fig, "followup_asymmetry_examples.png")

    # ---- item 6: one well-trained letter against the other well-trained characters -------------
    letters = [c for c in chars if c.isalpha() and freq[c] >= FREQ_MIN]
    Mg, rows = [], []
    for L in letters:
        i = chars.index(L)
        js = [j for j in kept if j != i and (i, j) in W]
        w = np.array([W[(i, j)] for j in js])
        res = residual_ranks(w, np.log10(f[js]))
        Mg.append([np.median([v for j, v in zip(js, w) if group_of(chars[j]) == g]) for g in GROUPS])
        rows.append([np.median([v for j, v in zip(js, res) if group_of(chars[j]) == g]) for g in GROUPS])
    Mg, rows = np.array(Mg), np.array(rows)

    def concordance(X):
        Rk = np.array([stats.rankdata(r) for r in X])
        nn, kk = X.shape
        s = ((Rk.sum(0) - Rk.sum() / kk) ** 2).sum()
        fr = stats.friedmanchisquare(*[X[:, c] for c in range(kk)])
        return {"mean_rank": [float(v) for v in Rk.mean(0)],
                "kendall_w": float(12 * s / (nn ** 2 * (kk ** 3 - kk))),
                "friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue),
                "n_letters": int(nn)}

    conc_raw, conc_res = concordance(Mg), concordance(rows)
    focus = "e"
    fi = chars.index(focus)
    fjs = [j for j in kept if j != fi and (fi, j) in W]
    fw = [W[(fi, j)] for j in fjs]
    kw = stats.kruskal(*[[v for j, v in zip(fjs, fw) if group_of(chars[j]) == g]
                         for g in GROUPS if sum(group_of(chars[j]) == g for j in fjs) >= 3])
    out["item6"] = {"focus_letter": focus, "n_partners": len(fjs),
                    "focus_group_medians": {g: float(np.median([v for j, v in zip(fjs, fw)
                                                                if group_of(chars[j]) == g]))
                                            for g in GROUPS},
                    "focus_group_n": {g: int(sum(group_of(chars[j]) == g for j in fjs)) for g in GROUPS},
                    "focus_kruskal_p": float(kw.pvalue),
                    "groups": GROUPS, "concordance_raw": conc_raw,
                    "concordance_freq_residualized": conc_res,
                    "mean_group_median_over_letters": [float(v) for v in Mg.mean(0)]}

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    srt = sorted(range(len(fjs)), key=lambda k: fw[k])
    for k, s_ in enumerate(srt):
        g = group_of(chars[fjs[s_]])
        ax.bar(k, fw[s_], color=GCOLOR[g], hatch=GHATCH[g], edgecolor="0.25", width=.8)
    ax.set_xticks(range(len(srt)))
    ax.set_xticklabels([disp(chars[fjs[s_]]) for s_ in srt], fontsize=7)
    ax.axhline(STRICT_W, **REF_RULE)
    ax.text(0.4, 0.335, "strict plateau rule 0.25 (dotted)", fontsize=8, ha="left")
    ax.set_ylim(0, .62)
    ax.set_xlabel(f"partner character (all {len(fjs)} other well-trained characters), sorted by width")
    ax.set_ylabel("$w_{10\\to90}$ for the pair ('e', partner)")
    ax.set_title(f"'{focus}' against every other well-trained character; 1 bar = 1 pair\n"
                 f"= 1 $d(t)$ curve of {S['n_t']} points. Kruskal–Wallis across groups "
                 f"p = {kw.pvalue:.1e}", fontsize=9)
    hs = [plt.Rectangle((0, 0), 1, 1, facecolor=GCOLOR[g], hatch=GHATCH[g],
                        edgecolor="0.25", label=g) for g in GROUPS]
    ax.legend(handles=hs, fontsize=7.5, ncol=3, loc="upper left", frameon=False,
              title="partner group (bar colour + hatch)")

    ax = axes[1]
    bp = ax.boxplot([Mg[:, k] for k in range(len(GROUPS))], widths=.6, showfliers=False,
                    patch_artist=True, medianprops=dict(color="k", lw=1.3))
    for k, g in enumerate(GROUPS):
        bp["boxes"][k].set(facecolor=GCOLOR[g], alpha=.5, hatch=GHATCH[g], edgecolor="0.25")
        ax.plot(np.full(len(letters), k + 1) + np.linspace(-.12, .12, len(letters)),
                Mg[:, k], ls="none", marker=".", ms=3, color="0.3", alpha=.65)
    ax.set_xticks(range(1, len(GROUPS) + 1))
    ax.set_xticklabels(GROUPS, fontsize=7.5, rotation=18, ha="right")
    ax.set_ylabel("median $w_{10\\to90}$ of one letter against that group")
    ax.set_title(f"The ordering repeats across all {len(letters)} well-trained letters\n"
                 f"Friedman p = {conc_raw['friedman_p']:.0e}, Kendall $W$ = {conc_raw['kendall_w']:.2f}; "
                 f"partner frequency removed: $W$ = {conc_res['kendall_w']:.2f} "
                 f"(p = {conc_res['friedman_p']:.0e})", fontsize=9)
    fig.suptitle(CONTEXT_NOTE, fontsize=8.5, y=1.03)
    save(fig, "followup_letter_groups.png")

    json.dump(out, open(os.path.join(RES, "followup_summary.json"), "w"), indent=1)
    print(json.dumps(out["item1"], indent=1))
    print(json.dumps(out["item6"]["concordance_raw"], indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
