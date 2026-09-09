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
