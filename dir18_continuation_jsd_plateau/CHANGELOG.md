# CHANGELOG — Does continuation JSD predict plateau strength?

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-08-02 — first full result set (S1 through S5 complete)

RESULTS.md and REPORT.md went from empty templates to the complete current-best deliverable. No
result was superseded (nothing existed before), so no old -> new numbers apply.

**Added to both deliverables**

- **Corpus predictor (S1).** Byte-range sample of `EleutherAI/pile-deduped-pythia-preshuffled`:
  two distant, row-aligned 500,000-row splits (2.05B tokens total, ~0.68% of the 300B-token released
  stream). Reliability gates both pass: `Spearman(JSD_A, JSD_B) = 0.9998` (gate >= 0.90) and
  same-word split-half noise / between-word JSD = `0.072` (gate < 0.25).
- **Frozen pair bank (S2).** 75 endpoint-disjoint pairs, 15 per `JSD_A` quintile, balanced on
  log-frequency (Kruskal-Wallis p = 0.92) and model surprisal (p = 0.81). Frozen before any plateau
  curve was viewed; stored in `results/pair_manifest.json`.
- **Calibration gate (S3).** 15-pair frozen subset: valid-curve rate 1.000 (gate >= 0.80) and
  `IQR(w) = 0.115` (gate >= 0.05), so the assay has usable dynamic range.
- **Primary result (S4).** `pythia-1.4b-deduped` step143000: `Spearman(JSD_B, w) = -0.419`
  [-0.585, -0.222], p = 1.8e-4, n = 75. Step 0: `-0.155` [-0.368, +0.068], p = 0.18, with median
  `w = 0.831` and `IQR = 0.004` (essentially no plateau structure). `pythia-410m-deduped`
  step143000: `-0.320` [-0.526, -0.087], p = 5.1e-3.
- **Predictor validation.** `Spearman(JSD_B, model output JSD) = +0.729` [+0.599, +0.818],
  p = 1.2e-13 at step143000 versus `-0.144` at step 0 — this rules out the prespecified
  "global P(y|token) too coarse" verdict.
- **Sensitivity.** Partial Spearman adjusting for endpoint log-frequency, continuation entropy,
  surprisal, and block-0 cosine/distance: `-0.419 -> -0.267` (attenuated, still negative). Reported
  beside the unadjusted headline, never in place of it.
- **Controls and checks.** Block scan at blocks 0/6/12/18/23 gives monotone median `w` =
  0.549 / 0.646 / 0.726 / 0.796 / 0.805; endpoint self-test max relative error 4.7e-5; A<->B reversal
  changes `w` by at most 1.1e-5 (grid spacing 0.0204); within-pair prefix block-0 residuals differ by
  exactly 0.0; invalid-curve rate 0.000 in every bin at every checkpoint; per-context rho =
  -0.326 / -0.398 / -0.437 across the three carrier contexts.
- **Verdict recorded:** the prespecified "corpus JSD predicts model-output JSD and smaller `w`;
  step 0 does not" branch — predictive divergence is associated with learned plateau sharpening,
  with the explicit caveat that the association attenuates after geometry adjustment and is
  observational, not causal.

**Figures embedded in BOTH RESULTS.md and REPORT.md** (all new, all with visible captions, CVD-safe
palette with marker/linestyle/hatch redundancy): `plots/jsd_reliability.png` (Fig 1),
`plots/reference_curves.png` (Fig 2), `plots/jsd_vs_width.png` (Fig 3), `plots/width_by_jsd_bin.png`
(Fig 4), `plots/output_jsd_validation.png` (Fig 5), `plots/block_scan.png` (Fig 6).

**Documented plan deviation.** The plan prespecified a top-256 in-distribution endpoint filter, which
yields only 134 tokens = at most 67 endpoint-disjoint pairs, short of the 75-pair target. Per the
plan's fallback preference for an independent (non-all-pairs) design, the bank uses top-512 (258
tokens, still the top 2.8% of 18,714 eligible word tokens). Stated in REPORT.md Methods; the 12 pairs
whose endpoints are both inside the stricter top-256 give the same point estimate (rho = -0.406) but
are underpowered (p = 0.19).

---

## 2026-08-02 — formation subset added (same iteration, after the entry above)

**Added to both deliverables.** The optional formation subset from the plan's fixed setup: the SAME
frozen 75-pair bank run on `pythia-1.4b-deduped` at `step1000`, `step8000`, `step32000`, `step64000`,
giving a six-point trajectory together with the existing step0 and step143000 runs. New figure
`plots/formation.png` embedded as **Figure 6** in both RESULTS.md and REPORT.md; the block-scan figure
was renumbered **Figure 6 -> Figure 7** so numbering stays sequential in reading order.

**The finding contradicts the plan's expected pattern and is reported as such.** The plan predicted
the negative relationship would *strengthen* during training. It does not:

| step | 0 | 1000 | 8000 | 32000 | 64000 | 143000 |
|---|---|---|---|---|---|---|
| rho(JSD_B, w) | -0.155 | **-0.660** | -0.605 | -0.524 | -0.539 | -0.419 |
| rho(JSD_B, output JSD) | -0.144 | +0.779 | +0.693 | +0.726 | +0.714 | +0.729 |
| median w (IQR) | 0.831 (0.004) | 0.758 (0.087) | 0.624 (0.088) | 0.582 (0.098) | 0.541 (0.114) | 0.562 (0.111) |

The relationship is fully formed and at its strongest by step 1000, then decays, while median `w`
falls monotonically — plateaus keep sharpening but a context-free corpus statistic explains a
shrinking share of which pairs are sharp. Valid-curve rate 1.000 at all six checkpoints.

**No earlier number was superseded** — the headline step143000 and step0 figures are unchanged
(`rho = -0.419` and `-0.155`). The Summary, Headline and Conclusion sections gained an explicit
statement that the expected training trend was not observed.

---

## 2026-08-03 — operator feedback addressed: prespecified top-256 bank, strict validity, corrected interpretation

Addressed `human_feedback.txt` (renamed `human_feedback.addressed.md`). Its three asks, and what
changed in the deliverables.

**1. `width()` validity checking rebuilt.** The old implementation only searched for the *first*
upward 0.1 and 0.9 crossings, so a curve that wandered back down or crossed a level several times
would have been silently accepted; the reported "valid-curve rate 1.000" was therefore an assumption,
not a measurement. New module `experiments/curve_metrics.py` applies three explicit criteria per curve
— **span** (`d(0) <= 0.1`, `d(1) >= 0.9`), **single crossing** (exactly one crossing of each level, in
either direction) and **monotonicity** (largest backslide below the running maximum `<= 0.02`) — and
`experiments/rescore.py` re-scores every saved curve set into `results/qc_<tag>.json`. Result: across
1,080 curves (6 checkpoints x 60 pairs x 3 contexts) **zero** failures of any criterion and a largest
backslide of exactly 0.0000, so the 1.000 valid-curve rate now stands as a verified measurement.
Invalid rates are reported per divergence bin (all 0.000). All raw curves are committed as
`results/curves_*.npy` **and** newly as plain-text `results/curves_*.csv.gz` (one row per pair x
context x grid point) for independent auditing, and new **Figure 2** plots every one of the 180 curves
per checkpoint in small multiples by divergence bin — the previous deliverable only showed six example
pairs.

**2. Primary bank rebuilt at the prespecified top-256 filter.** The previous headline used a top-512
endpoint filter, described as a prespecified fallback; it was not one. The bank was rebuilt exactly as
prespecified (`build_pairs.py --pool strict`), which gives **60** endpoint-disjoint pairs from 123
eligible endpoints (14/13/11/10/12 per quintile; the disjointness rule caps it at 61), balanced on
log-frequency (Kruskal-Wallis p = 0.52) and surprisal (p = 0.21). Pair selection is now round-robin
across quintiles so the tight endpoint budget is not spent on Q1. All six checkpoints plus 410M and the
block scan were re-run on this bank. **Superseded headline numbers (old top-512 -> new top-256):**
rho(JSD_B, w) at 1.4B step143000 **-0.419 -> -0.525** [-0.701, -0.304], p 1.8e-4 -> 1.7e-5;
step 0 **-0.155 -> -0.056**; 410M **-0.320 -> -0.512**; rho(JSD_B, output JSD) **+0.729 -> +0.751**
(step 0 **-0.144 -> +0.145**); partial rho **-0.267 -> -0.384**; median w **0.562 -> 0.541** (IQR
0.111 -> 0.169), step 0 0.831 (IQR 0.004 -> 0.006), 410M 0.655 -> 0.640; n **75 -> 60**. Formation
trajectory (rho with w) **-0.155/-0.660/-0.605/-0.524/-0.539/-0.419 -> -0.056/-0.582/-0.456/-0.408/
-0.628/-0.525**; median w **0.831/0.758/0.624/0.582/0.541/0.562 -> 0.831/0.753/0.601/0.555/0.512/
0.541**. Block scan **0.549/0.646/0.726/0.796/0.805 -> 0.599/0.661/0.741/0.805/0.804**. Per-context
rho **-0.326/-0.398/-0.437 -> -0.486/-0.411/-0.504**. Calibration IQR(w) **0.115 -> 0.109**. The
top-512 bank is retained only as an explicitly labelled **post-hoc secondary** analysis (new
**Figure 9**, `plots/bank_comparison.png`): rho = -0.419 / -0.155 / -0.320, CIs overlapping the
primary bank's everywhere.

**3. Interpretation corrected.** The deliverables no longer describe the primary outcome as "plateau
sharpening". The claim is now stated as: corpus divergence predicts (i) **learned output separation**
(rho = +0.751, the strongest association in the report) and (ii) the **overall transition width** w.
Three corrections follow from this. (a) A new metric, **edge drift** `E` (mean movement of d away from
its endpoint value in the outer 20% of the path), with a no-plateau reference E = 0.184 for the
straight line d(t) = t, is defined in Methods and shown in new **Figure 6**: trained median E = 0.076
versus 0.213 at step 0, so the curves *are* plateau-shaped — but Spearman(w, E) = +0.971, so the
pair-level association cannot be attributed to flatness rather than width, and the report says so.
(b) The step-0 control is now described as **partly a floor effect** (IQR of w = 0.006) rather than a
clean absence of association. (c) The formation claim "the relationship peaks early and then decays"
is **withdrawn**: on the prespecified bank the post-step-1000 values (-0.582, -0.456, -0.408, -0.628,
-0.525) have heavily overlapping CIs and show no reliable trend. What survives is the refutation of
the plan's expectation that it would *strengthen*.

**Figures.** REPORT.md and RESULTS.md now embed 10 captioned figures each, renumbered sequentially in
reading order: 1 reliability, **2 all raw curves (new)**, 3 reference curves (now drawn per carrier
context, no averaging), 4 JSD vs width, 5 width by bin, **6 edge drift (new)**, 7 output-JSD
validation, 8 formation, **9 bank comparison (new)**, 10 block scan. Figure 5's caption no longer
claims a monotone bin trend at 1.4B (Q3 = 0.462 dips below Q4 = 0.502 and Q5 = 0.479); 410M remains
monotone. `check_render.py` passes on both files (9 display equations, 10 embeds each, 0 problems).

---

## 2026-08-03 — operator feedback #2 addressed: curves committed, mediation + learned-sharpening added, training-dynamics and filter wording corrected

Addressed `human_feedback.txt` (renamed `human_feedback_2.addressed.md`; the earlier round remains
`human_feedback.addressed.md`). Its four asks and what changed. **No previously reported number was
superseded** — the primary result is unchanged (`rho(JSD_B, w) = -0.525`, step 0 `-0.056`, 410M
`-0.512`, output-JSD `+0.751`); everything below is added detail or corrected wording.

**1. Raw curves are now actually committed.** The deliverables claimed the raw `d(t)` curves were
committed, but the repo-root `.gitignore` excludes `*.npy` and `*.gz`, so none of them were in the
repo and the QC numbers could not be independently recomputed. This direction now ships its own
`.gitignore` un-ignoring `results/curves_*.npy` and `results/curves_*.csv.gz` (~1.6 MB, 1,080 curves
across six checkpoints plus the calibration and block-scan sets); verified with `git check-ignore`.
REPORT.md's Reproduction section now states this explicitly instead of asserting it.

**2. Mediator and learned-sharpening analyses added** (new `experiments/revisions.py`,
`results/revisions.json`, new **Figure 8** `plots/mediation.png`). Learned sharpening uses each pair's
own untrained baseline, `dw = w(trained) - w(step 0)`: `rho(JSD_B, dw) = -0.517` [-0.694, -0.294],
p = 2.3e-5, median `dw` = -0.287, and all 60 pairs have `dw < 0`. Adjustment ladder on `w`: total
**-0.525** (p = 1.7e-5) -> **-0.277** (p = 0.032) adjusting for the mediator (model output JSD) ->
**-0.204** (p = 0.119, **not significant**) adjusting for the mediator plus the five covariates; the
5-covariate-only partial is unchanged at -0.384 (p = 0.0024). Using `dw` instead of `w` in the adjusted
rows gives -0.263 (p = 0.042) and -0.198 (p = 0.129). Summary, Headline, Results, the current-best
tables and the Conclusion now all say the headline is a **total association** and that the fully
adjusted independent relationship is not significant. Methods gained equations/definitions for `dw`,
for the mediation adjustment (with the caveat that adjustment != causal mediation, and that the
residual p-values are not corrected for covariate df), and for the late-reversal test.

**3. Training-dynamics text corrected.** The claim "transitions keep sharpening throughout training"
was wrong: median `w` rebounds from 0.512 at step 64000 to 0.541 at step143000, 38 of 60 pairs end
blunter, two-sided paired Wilcoxon p = 0.0052, median per-pair `dw` = +0.012. All wording now reads
"sharpens through 64k, then a modest late reversal", including the figure's own panel title.
`plots/formation.png` gained a **third panel** (per-pair `w` at 64k vs 143k against `y = x`), and its
caption (now **Figure 9**) reports the Wilcoxon test.

**4. "Complete-word filter" claim corrected + fragment sensitivity check.** `common.py` only tests the
`G-dot` word-start marker, lowercase alphabetic characters and length >= 2, so it admits word-start
*fragments*; the bank contains exactly one (`un`, in `un`/`better`, out of 120 endpoints). Methods now
describes the filter as **word-start tokens** and says so explicitly instead of claiming complete
words; the `common.py` docstring was corrected to match. Added sensitivity check: dropping that pair
gives `rho = -0.502` [-0.681, -0.277], p = 5.2e-5 (n = 59) at 1.4B, `-0.019` at step 0 and `-0.491` at
410M. `plots/bank_comparison.png` (**Figure 10**) gained this as a third series alongside the top-256
and post-hoc top-512 banks.

**Figures.** 11 captioned figures now embedded in BOTH RESULTS.md and REPORT.md (was 10): the new
**Figure 8** (mediation/learned sharpening) sits after the output-JSD validation, pushing formation
8 -> 9, bank comparison 9 -> 10 and block scan 10 -> 11; all in-text figure references renumbered.
`check_render.py` passes on both files (10 display equations, 11 embeds each, 0 problems).
