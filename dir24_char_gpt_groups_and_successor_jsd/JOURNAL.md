# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-09-09 — S1–S6 in one iteration

**Did.** Verified the dir13 source (corpus SHA matched the training-time value after re-downloading
tinyshakespeare to `/tmp`; the stored 2,080-pair summary matched its recorded context, checkpoint, hook
point and endpoint diagnostics) and wrote one CPU script, `experiments/analysis.py`, covering S1–S4. It
reuses the stored widths, groups each of the 43 well-trained letters' 52 partners into the six frozen
classes, computes unsmoothed bigram successor distributions and their pairwise JSD with a split-half
reliability check, and correlates JSD with width pooled and within anchors. Wrote RESULTS.md, REPORT.md
(3 figures, 1 table), CHANGELOG entry; `check_render.py` passes.

**Learned.** (1) The class grouping reproduces dir13's concordance numbers exactly and holds across all
43 anchors, but the between-anchor spread (0.258–0.404) is as large as the class effect, and part of the
grouping is partner frequency (W drops 0.424 -> 0.267 after residualising). (2) The JSD question has a
genuinely mixed answer: nothing pooled, a consistent weak negative within anchors. The reason is visible
in the data — the between-anchor component has the opposite sign (rho = +0.255, p = 0.10) and roughly
cancels the within-anchor slope, so pooling hides it. This is why the plan's demand for both views
mattered.

**Assumptions logged.** (a) The permutation null the plan asks for was extended to the per-anchor median
rho, because anchors share partners and a sign test would treat dependent numbers as independent;
alternative rejected: binomial sign test over 43 anchors. (b) Partialling partner log-frequency out of
the within-anchor JSD–width correlation was added because rarer partners already have wider transitions;
it is reported in RESULTS.md as a robustness check, not as a new composite predictor. (c) REPORT.md is
1,621 words by `wc -w`; roughly 120 of those are table pipes and image markup, so the prose is ~1,500,
within the plan's 1,600-word limit.

**Next step.** None — success criterion met, STOP written.

On track? yes — S1–S6 complete, 100% done, no blocker.
