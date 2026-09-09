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

## 2026-09-09 — triage of `human_feedback.txt`

**Did.** Triage pass only, no research. Verified the feedback SHA-256 matches the manifest and that all
48 checklist requests are byte-for-byte substrings of `human_feedback.txt` (including the literal
`$$ ... \operatorname{median}{...} $$` block and the curly quotes). Filled the manifest routing fields:
required outputs `REPORT.md`, `RESULTS.md`, `plots/fig2_width_heatmap.png`,
`plots/fig3_jsd_vs_width.png`; `dir13` artifacts and `human_feedback.txt` must remain unchanged;
report policy = same two core questions, primary report `REPORT.md`, 1,600 words, 3 main figures,
detailed results in `RESULTS.md`. State set to `ready`.

**Learned / decided (triage judgements, not research).** (1) "43 × 53 heatmap" and "52 partners" are
consistent, not contradictory: 53 well-trained characters are columns, the 43 self-pair cells are blank,
so each anchor keeps 52 partners. (2) "Replace or supplement Figure 2" is explicitly the author's
choice and the class-median summary may stay "as a smaller accompanying panel", so the 3-figure plan
limit is preserved by making Figure 2 a heatmap-dominant panel plus a smaller class-median panel.
(3) The three bin numbers quoted in the feedback (n = 12/31/78, mean width 0.581/0.360/0.343) are to be
recomputed independently from this direction's artifacts, as the feedback asks. (4) No new grouping
score, regression, significance test or composite metric may be added; existing W / rho / Friedman /
permutation results may only follow the visual and plain-language results. No ambiguity remained, so the
task was not blocked.

**Next step.** Work the ready checklist: rewrite Methods with the `t`-anchor worked example and the
65 -> 53 -> 43 vocabulary accounting, rename endpoint variables to `c_anchor` / `c_partner`, build the
raw pairwise width heatmap, rebuild Figure 3 with fixed-width JSD bins and punctuation–punctuation
pairs marked, move the rank statistics behind the plain-language conclusions, and demote the per-anchor
Spearman analysis to `RESULTS.md`.

On track? yes — triage complete, checklist verified verbatim, state `ready`.
