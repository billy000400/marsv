# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

---

## 2026-09-09 — first and final deliverables for this direction

- **RESULTS.md**: created (was a template). Records S1 source validation of the reused dir13 step-30,000
  sweep (commit `01d2501`; 2,080 pairs, 0 undefined widths, corpus SHA verified), the S2 letter-anchor
  class table (mean per-anchor median `w` from 0.270 for lower vowels to 0.356 for upper consonants;
  Kendall's W = 0.424, p = 3.8e-18; frequency-residualised W = 0.267, p = 4.3e-11), the S3 successor-JSD
  reliability check (split-half Spearman 0.951, same-character noise floor 0.0072 bits vs median pair
  0.667 bits), and the S4 width-vs-JSD results (pooled rho = -0.064, permutation p = 0.32; per-anchor
  median rho = -0.205, IQR -0.401 to -0.081, 36/43 negative, permutation p = 0.002; -0.320 with partner
  log-frequency partialled out; rare-pair sensitivity +0.081 kept visibly separate; between-anchor
  rho = +0.255, p = 0.10).
- **REPORT.md**: created (was a template). Answers both plan questions in two Results subsections, with
  Figures 1–3 (width definition; class widths from anchors; JSD vs width) and one table. Reports the
  pooled/fixed-anchor difference rather than selecting the favourable one, and states that no result is
  causal.
- No superseded numbers: this is the direction's first set of results. The two concordance statistics
  reproduce dir13's stored values exactly (W = 0.4242, p = 3.76e-18), which is why the interpolation
  sweep was reused rather than rerun.

## 2026-09-09 — revision for `human_feedback.txt`: raw structure before summaries

**REPORT.md** (rewritten, 1,600 words, 3 figures, 1 table).
- Methods: added the 65 -> 53 -> 43 vocabulary accounting and the worked `t` class-median example;
  renamed all endpoint variables `a`/`b` -> `c_anchor`/`c_partner` (and `z_a`/`z_b` -> `z(0)`/`z(1)`).
- Figure 2 replaced: class-median line/box figure -> raw 43x53 pairwise width heatmap with the
  class-median summary as a smaller side panel (`plots/fig2_class_widths.png` ->
  `plots/fig2_width_heatmap.png`; the old figure survives as `plots/fig2b_class_widths.png` in
  RESULTS.md).
- Figure 3 replaced: ten equal-count JSD bins -> ten fixed-width 0.1 bins with per-bin mean width, per-bin
  pair counts, faint per-pair points and punctuation-punctuation pairs marked; the per-anchor Spearman
  histogram moved out to `plots/fig_s2_per_anchor_rho.png` (RESULTS.md).
- New result reported: the 0.0-0.1 JSD bin has n = 12 and mean width 0.581 versus 0.317-0.360 for every
  other bin; 10 of those 12 pairs are punctuation-punctuation, punctuation-punctuation pairs average
  0.531 across the whole JSD range against 0.327 for the other 1,350 pairs, and removing them flattens
  the lowest bin to 0.340. Previously this bin was invisible: equal-count binning put it inside a
  138-pair bin whose median was 0.333.
- New result reported: mean within-anchor IQR of the *raw* widths inside a class block is 0.021 (lower
  vowels) and 0.020 (whitespace) but 0.051 (lower consonants) and 0.063 (upper consonants), against a
  0.087 spread between the extreme class averages — so the class median is representative only for the
  small classes.
- Conclusion narrowed: "no relationship between successor JSD and transition width" -> "no *general
  monotonic* relationship, with a wider low-JSD group dominated by punctuation-punctuation pairs"; the
  fixed-letter result restated in plain language. Kendall's W = 0.42, Friedman p = 3.8e-18, pooled
  rho = -0.06 and per-anchor median rho = -0.21 removed from the report body and retained in RESULTS.md
  after the visual results.

**RESULTS.md** (rewritten to current best).
- Section 2 gains the raw within-class spread columns (IQR and range per class) and per-partner means.
- Section 4 restructured: 4.1 fixed-width JSD bins (all 10 bins, with and without punctuation-punctuation
  pairs) and the 12 lowest-JSD pairs listed individually; 4.2 the rank statistics as a secondary check;
  4.3 class divergences. Figure numbering updated (now Figures 1-4).

**results/analysis.json**: added `s2_classes.raw_block_spread`, `s2_classes.partner_mean_w_over_anchors`,
`s4_jsd_vs_width.fixed_width_bins`, `.low_jsd_bin_pairs` and `.punct_punct`. All previously reported
numbers are unchanged.
