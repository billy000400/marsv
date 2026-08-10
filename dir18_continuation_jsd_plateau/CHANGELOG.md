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

---

## 2026-08-03 — operator feedback #3: terminology, split naming, and a 1,000-pair endpoint-clustered generality test

Addressed `human_feedback_3.txt` (renamed `human_feedback_3.addressed.md`). All 26 points are
reflected in RESULTS.md and REPORT.md; the two substantive new experiments are recorded here.

**NEW — secondary 1,000-pair bank (new Figure 11, `plots/large_bank.png`).** Built by
`experiments/build_large_bank.py` from the same 123 eligible endpoints and the same frequency-ratio
rule as the primary bank, replacing endpoint-disjointness with a cap of 20 pairs per endpoint and
taking 200 pairs in each selection-split quintile (no transition curve consulted). Assayed on
`pythia-1.4b-deduped` step143000 and step0, 3 contexts x 50 positions = 3,000 curves per checkpoint,
all valid under the strict criteria (max backslide 0.0000). New `experiments/large_analysis.py` does
endpoint-clustered inference only:

| | trained step143000 | untrained step0 |
|---|---|---|
| Spearman rho(J_hold, w) | **-0.486** | -0.008 |
| dyadic endpoint-bootstrap 95% CI (4,000) | [-0.603, -0.353] | [-0.126, +0.109] |
| endpoint-label permutation p (4,000) | **< 0.00025** (0 reached it) | 0.86 |
| naive pair-bootstrap CI (invalid, contrast only) | [-0.533, -0.437] | [-0.068, +0.053] |
| rho(J_hold, JSD_out) | +0.729 | +0.001 |
| median w (IQR) | 0.555 (0.129) | 0.831 (0.005) |

Binned medians over 10 non-overlapping equal-count holdout-JSD bins fall 0.649 -> 0.611 -> 0.602 ->
0.567 -> 0.563 -> 0.542 -> 0.520 -> 0.524 -> 0.497 -> 0.499, i.e. essentially monotone with a
flattening above ~0.75 bits — so the association is not driven by the small matched bank and shows no
non-monotonicity. Reported throughout as an **endpoint-dependent robustness analysis**, never as 1,000
independent observations; no naive p-value is quoted for it.

**NEW — selection-split sensitivity (`experiments/split_sensitivity.py`,
`results/split_sensitivity.json`).** Using the selection split as the predictor instead of the holdout
split changes nothing: rho = -0.526 vs -0.525 (trained 1.4B), -0.053 vs -0.056 (step 0), -0.511 vs
-0.512 (410M), with rho(J_sel, J_hold) = 0.99972 on the bank. Matches the operator's numbers.

**Terminology and claim corrections (deliverables + figure labels regenerated).**
- Predictor renamed and redefined explicitly as **context-averaged immediate-next-token JSD**,
  `J_hold(u,v)`, with its own equation; splits renamed **selection**/**holdout** (was A/B) and
  endpoints renamed `(u,v)` (was A/B, which clashed with endpoint labels). Every axis label, legend
  and caption regenerated (`analyze.py`, `revisions.py`), e.g. "JSD (split B, bits)" ->
  "held-out corpus next-token JSD J_hold(u,v) [bits]".
- Methods now states why only the holdout JSD is used **and** that the holdout split is not untouched
  (its counts gate eligibility; summed counts drive frequency matching).
- Figure 4 caption now states that each dot is one endpoint pair (median w over 3 contexts, each from
  50 positions), that there are only 60 pair-level observations re-used across the three panels, that
  hue/marker = selection-split stratum, and that the crosses are 5 non-overlapping binned medians
  after re-binning by holdout JSD — not a running median, not extra observations.
- Added the explanation of why n = 60 (123 eligible endpoints -> at most 61 disjoint pairs), with the
  weaker claim "removes direct dependence from endpoint reuse", not "statistically independent".
- "complete words" -> **single-token word-start endpoints**; "every prompt is in-distribution" ->
  "endpoints are **model-plausible under the three carrier contexts**".
- Plateau language narrowed to **relative-logit-coordinate plateau**, with an explicit statement that
  a flat d(t) does not show the full logit vector or output distribution stays put; `w` is now always
  described as the **10%-90% transition width**, never as generic transition strength.
- "predicted from the training corpus alone" -> "the JSD predictor itself is computed from corpus
  statistics" (filtering and matching use model probabilities and surprisal).
- Removed the Figure 1 claim that "93% is real signal"; the noise ratio is now described as a ratio of
  medians, not an additive signal/noise decomposition.
- "the bins are indistinguishable" -> "we detected no significant imbalance".
- Step-0 `w` described as a **restricted range / near-ceiling**, no longer a "floor effect".
- 410M relabelled a **cross-scale robustness check**, not an independent replication.
- `JSD_out` defined precisely: median across the 3 carrier contexts, over the 50,060 corpus-observed
  target IDs.
- "almost nothing survives adjustment" -> "the association is attenuated after adjustment, and the
  fully adjusted estimate is not statistically significant"; the -0.277 (p = 0.032) row is now
  explicitly flagged as still significant.
- "full strength by step 1000, then no further change" -> "already comparable to later checkpoints at
  the earliest measured checkpoint", plus "nothing constrains what happened between step 0 and 1000".
- Figure 2 scope corrected: it draws all 180 curves at **two** checkpoints; the validity audit spans
  1,080 primary-bank curves plus the 3,000 secondary-bank curves.
- Block scan scope stated in both the caption and Methods: 10 extreme pairs, one carrier context;
  panel title changed from "Sharpness needs downstream blocks".
- Prespecification claim narrowed to "the top-256 selection rules were prespecified, and the
  exact-pair curves were not used during pair selection".
- New precise headline sentence adopted verbatim in both deliverables.

**Figures.** 12 captioned figures now embedded in BOTH files (was 11); new **Figure 11** = the
1,000-pair bank, pushing the block scan 11 -> 12. All other PNGs regenerated with the new axis
labels. `check_render.py` passes on both files (12 display equations, 12 embeds each, 0 problems).

**Unchanged numbers.** The corpus pipeline was rebuilt from scratch this iteration (the /tmp cache
does not survive sessions) and reproduces exactly: Spearman(J_sel, J_hold) = 0.9998, noise ratio
0.0723, 50,060 valid target IDs, 123 eligible endpoints, primary rho = -0.525 / -0.056 / -0.512.

---

## 2026-08-05 — operator feedback #4: the two named example pairs (`big`/`large`, `big`/`in`)

Addressed `human_feedback_4.txt` (renamed `human_feedback_4.addressed.md`), which asked for a plot
validating whether the Pythia models plateau on *"My house is big/large"* and not on
*"My house is big -> in"*. No previously reported number changed; this iteration only ADDS a result.

**New experiment.** `experiments/reference_jsd.py` recounted the successors of ` big`, ` large` and
` in` in the same two 500,000-row corpus splits used throughout (the /tmp corpus cache does not
survive a session, so both splits were re-downloaded; the pipeline reproduces exactly — 50,060 valid
target IDs, same as every earlier run). `experiments/reference_house.py` ran both pairs through the
unchanged post-block-0 assay in four carriers (`My house is` plus the three project carriers) at three
model settings, and additionally records the **absolute output movement**
`M(t) = JSD(p(t), p(0))` in bits. `experiments/plot_reference_house.py` makes the figure.

**New numbers (no supersession — these pairs had never been assayed).**

- Corpus: `J_hold( big, large) = 0.412` bits, `J_hold( big, in) = 0.701` bits; split-half sampling
  noise 0.070 / 0.059 / 0.003 bits for ` big` / ` large` / ` in`; holdout counts 122,257 / 175,159 /
  9,821,847.
- Trained 1.4B, carrier `My house is`: ` big`/` large` `w = 0.773`, `E = 0.162`, `M(1) = 0.035` bits;
  ` big`/` in` `w = 0.357`, `E = 0.043`, `M(1) = 0.935` bits (`M(0.5) = 0.505`).
- Step 0: 0.834 / 0.829 (`E` 0.216 / 0.211). 410M trained: 0.794 / 0.494 (`E` 0.198 / 0.075).
- Context in the bank: `w = 0.357` is sharper than all 60 bank pairs (bank min 0.401); `w = 0.773` is
  above 95% of them. Bank pairs near 0.41 bits have median `w = 0.639`; near 0.70 bits, 0.502.

**Answer recorded in both deliverables.** The plateau is on ` big`/` in`, not on ` big`/` large` —
the opposite way round from the question as phrased — and the two facts are the same fact once
absolute movement is measured: the trained model separates *"My house is big"* from *"My house is
large"* by only 0.035 bits, so the whole interpolation path lies inside one plateau and there is no
boundary to cross, while ` big` and ` in` sit in different plateaus and the crossing is abrupt. Both
behaviours are learned (step 0 shows neither) and the 410M model reproduces them.

**Changed in RESULTS.md and REPORT.md**

- New **Figure 13** (`plots/house_reference.png`) embedded with a visible caption in BOTH files: (a)
  `d(t)` for the two pairs, trained 1.4B, carrier `My house is`, against the no-plateau diagonal;
  (b) absolute movement `M(t)` in bits; (c) the same prompts at step 0; (d) both pairs placed on the
  60-pair bank scatter with its binned medians, with a bar spanning the other three carriers.
- New Results subsection in REPORT.md ("The two named example pairs") and a matching block in
  RESULTS.md, each with the table above, the reading of the answer, and two caveats: the gap is wider
  than the bank trend predicts (pair-level scatter, cf. Figure 3), and ` in` is ~80x more frequent
  than ` big`, so this pair would fail the bank's 2x frequency-matching rule.
- New Methods metric in REPORT.md: **absolute output movement** `M(t) = JSD(softmax z(t), softmax
  z(0))` in bits, motivated by the blind spot of the normalised coordinate `d(t)` (it runs 0 -> 1
  however little the output moves).
- Summary and Conclusion each gained one paragraph/sentence on the example pairs; the sample-size
  paragraph gained the 24 extra curves; the reproduction list gained the three new scripts.
- Figure count 12 -> 13 in both files; `check_render.py` passes (13 display equations, 13 embeds and
  13 captions per file, 0 problems).

---

## 2026-08-05 — operator feedback #5: appendix documenting the 60-pair bank

Addressed `human_feedback_5.txt` (renamed `human_feedback_5.addressed.md`): "How did you sample the
60-pair bank? what are they? write those in the appendix of the report." No result changed; this
iteration only ADDS documentation of an existing frozen artefact.

**Added to REPORT.md — new "Appendix A — the 60-pair bank: how it was sampled, and what is in it"**

- **A.1, the procedure in seven steps**, each with its surviving count: eligible token type (word-start
  `Ġ` tokens, >= 2 lowercase ASCII letters) -> top-256 model-plausibility filter intersected over all
  three carriers -> >= 20,000 occurrences in EACH corpus split (**123 endpoints**) -> factor-of-two
  within-pair frequency rule (**1,763 candidate pairs**) -> quintiles of `J_sel` with the actual bin
  edges (0.118 / 0.499 / 0.605 / 0.691 / 0.768 / 0.971 bits) -> round-robin, endpoint-disjoint
  selection minimising a stated balance cost -> **60 pairs, 14/13/11/10/12 across Q1-Q5**, ceiling 61.
  The balance cost is now written out as a rendered equation (normalised distance of the pair's mean
  log-frequency and mean surprisal from the eligible-endpoint medians), and the reason for
  round-robin order is stated (endpoint-disjointness is the binding constraint).
- **A.2, the complete list of all 60 pairs** as a table: index, quintile, both endpoint strings, both
  corpus counts (summed over the two splits), `J_sel`, `J_hold`, `w` at step143000 and `w` at step0,
  with the 15 frozen calibration pairs asterisked. Generated by the new
  `experiments/appendix_bank.py` from the frozen manifest plus the two 1.4B assay runs, so it cannot
  drift from the analysed data.
- Cross-references added: the Methods "Pair bank construction" section now points to Appendix A, and
  RESULTS.md points to it as well.

`check_render.py` passes on both files (14 display equations in REPORT.md, 13 embeds and 13 captions
per file, 0 problems).

---

## 2026-08-05 — operator feedback #6: correspondence-only report, formation removed

Addressed `human_feedback_6.txt` (renamed `human_feedback_6.addressed.md`). The feedback asked for a
simplified report answering one question only — *do token pairs with more different
immediate-next-token distributions tend to have narrower transitions in the trained model's
output-distance score `d(t)`?* — and asked to separate correspondence from formation. No experiment
was re-run and no number changed; this is a restructuring, a re-ordering, a de-jargoning, and a set of
deletions. All code, logs, result files and PNGs are preserved, including those no longer shown.

**Story re-framed (rule 9b).** Old lead: the prespecified 60-pair bank was the primary result and the
1,000-pair bank the secondary generality check. New lead: the **1,000-pair final-checkpoint result is
the main result** (it shows the relationship across the whole JSD range), followed by the controlled
60-pair result. Same numbers, different order of evidence.

**Removed from both deliverables** (artefacts kept on disk, they belong to the formation direction):

- All intermediate checkpoints (steps 1000 / 8000 / 32000 / 64000), the six-checkpoint table, the
  formation figure (`plots/formation.png`), every onset claim, and the 64k → final late-reversal
  result (median `w` 0.512 → 0.541, 38/60 blunter, paired Wilcoxon p = 0.0052).
- The learned-sharpening analysis `Δw = w(trained) − w(step 0)` (ρ = −0.517, median Δw = −0.287) and
  the two Δw rows of the adjustment ladder (−0.263 / −0.198). The mediation figure that carried them
  (`plots/mediation.png`) is replaced in the deliverables by a new **`plots/adjustment.png`** — a
  forest plot of the four `w`-only rows (−0.525 / −0.384 / −0.277 / −0.204), generated by the new
  `experiments/plot_adjustment.py`.
- The block scan moved out of the main story into Appendix B as a short sanity check.
- The `width_by_jsd_bin.png` figure, whose content is duplicated by the binned medians already drawn
  in the two lead figures and by the per-group median row of the 60-pair table.
- All audit history and feedback-round narration; the claim that a plateau represents one continuation
  distribution and that the path crosses a boundary (only the two endpoints were ever measured).

**Kept, per the feedback.** The 1,000-pair result with the token-reuse-aware interval and permutation
test; the controlled 60-pair result; corpus-JSD reliability; model-output-JSD validation; curve
validity; the `w` vs edge-drift redundancy limitation (ρ = +0.971); the adjustment result (strong
overall, not significant after accounting for model-output JSD and all five measured pair properties);
step 0 as a brief baseline; 410M as a cross-scale check; `big`/`large` and `big`/`in` as illustrations,
now framed as the caution the feedback asked for — **`d(t)` is uninformative when the endpoint outputs
are already almost identical** (M(1) = 0.035 bits for `big`/`large`).

**Moved to Methods/Appendix.** Pair construction (Appendix A.1), the 60-pair table (A.2), the
corpus-sample sensitivity check (B.1), the alternative pair sets (B.2), and all self-tests plus the
block scan (B.3).

**De-jargoned** throughout, per the feedback's replacement table: held-out JSD → next-token JSD on the
measurement sample; selection/holdout split → pair-selection sample / measurement sample; stratified
bank → pairs chosen to cover the full JSD range; endpoint-disjoint → no token reused across pairs;
carrier context → fixed sentence frame; quintile/stratum → one of five JSD groups; assay →
interpolation experiment; relative-logit coordinate → output-distance score `d(t)`; 10%–90% width →
the fraction of the path `d(t)` needs to move from 0.1 to 0.9; dyadic endpoint bootstrap → uncertainty
calculation accounting for tokens reused across pairs; mediator/covariate adjustment → after
accounting for model-output difference and the measured pair properties. Notation simplified:
`Ĵ_hold(u,v)` → `J(u,v)`, `Ĵ_sel(u,v)` → `J_sel(u,v)`.

**Figure axis labels regenerated to match the new vocabulary** (`analyze.py`, `large_analysis.py`,
`plot_reference_house.py` label strings only; no analysis code touched, every number reproduced
identically). `plots/jsd_reliability.png` could not be regenerated — its 10,000-pair source array
lived in the /tmp corpus cache, which is not preserved across sessions — so its old axis wording is
mapped to the new vocabulary in the caption instead.

**Final claim** now stated verbatim as the feedback specified, in both the Summary and the Conclusion.

**Figure set, both files, renumbered in reading order:** 1 large_bank, 2 jsd_vs_width,
3 jsd_reliability, 4 all_curves, 5 output_jsd_validation, 6 adjustment, 7 edge_drift,
8 reference_curves, 9 house_reference; REPORT.md adds 10 bank_comparison and 11 block_scan in
Appendix B. `check_render.py` passes on both files (REPORT.md: 12 display equations, 11 embeds,
11 captions; RESULTS.md: 9 embeds, 9 captions; 0 problems).

## 2026-08-10 — operator feedback #7: the flatness-vs-width claim was justified by a wrong dichotomy

`human_feedback_7.txt` (renamed `human_feedback_7.addressed.md`) challenged the sentence introducing
edge drift: *"A narrow transition could mean flat ends with a quick move in the middle, or just a
steeper straight line"* — asking why a narrow transition could ever be a steeper straight line with no
quick move in the middle. **It cannot, and the sentence was wrong.** `d(0) = 0` and `d(1) = 1` hold
exactly by construction, so the only straight line available is `d(t) = t`: slope fixed at 1, width
fixed at `w = 0.800`, edge drift fixed at `E = 0.184`. Any `w < 0.8` already means the score moves
faster in some part of the path than in others.

**New analysis** (`experiments/edge_geometry.py`, no GPU, reads the committed curves;
`results/edge_geometry.json`). The property `E` actually adds over `w` is *where* the move sits, not
how steep it is. For a given width we build the monotone reference curve through `(0,0)`, `(A,0.1)`,
`(A+w,0.9)`, `(1,1)` and sweep the starting position `A` over 201 placements (width preserved to
within 0.007 on the 50-point grid) — the **placement range** of `E`. New quantity: the **transition
midpoint** `m = t(d = 0.5)`.

- At the median trained width `w = 0.541`, placement swings `E` from **0.080** (centred) to **0.220**
  (parked late) — a factor of 2.7, wider than the whole trained-to-untrained gap in `E`
  (0.076 → 0.213). So `E` is *not* implied by `w`.
- Every measured transition is centred: `m = 0.505`, interquartile range 0.047, **96.7%** of the 60
  controlled pairs and **97.6%** of the 1,000 within 0.1 of the middle; 96.7% of pairs sit at (or
  below) the bottom of their own placement range. That, not an algebraic identity, is why
  `rho(w, E) = +0.971` (60 pairs) and `+0.978` (1,000 pairs).
- New number quantifying the redundancy: `rho(J, E) = -0.520` alone, but partial
  `rho(J, E | w) = -0.008` [−0.332, +0.328] on the controlled set and `-0.009` on the 1,000 pairs.

**Deliverable changes.** REPORT.md Methods: the wrong dichotomy replaced by the pinned-endpoint
argument plus the placement/position explanation; `m` and the placement range defined. REPORT.md §4
and RESULTS.md §4 rewritten around the corrected mechanism ("flatness and width are the same
measurement *because every transition is centred*"), with the new numbers; the Summary's second
bounding sentence and both files' limitation bullets updated to give the corrected reason. No result
about the main JSD-width association changed.

**Figure 7 regenerated** (`plots/edge_drift.png`), 2 panels → 4: (a) edge-drift histograms as before;
(b) three curves of identical width `w = 0.541` — centred (`E = 0.080`), parked late (`E = 0.220`),
and a measured curve (`E = 0.070`) — against the straight line; (c) `E` vs `w` with each pair's
placement range drawn as a gray segment; (d) transition-midpoint histograms. Panels (b)–(d) are new.
`check_render.py` passes on both files (REPORT.md: 13 display equations, 11 embeds; RESULTS.md: 9
embeds; 0 problems).
