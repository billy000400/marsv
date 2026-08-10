# PLAN — What explains transition-width variation beyond successor JSD?

> Working folder: create a new direction. Treat `dir18_continuation_jsd_plateau` as read-only source material.

## Core question

Why can two token pairs with similar successor JSD have very different transition widths (w)?

Corpus successor JSD describes how different the tokens’ context-averaged next-token distributions are. Model-output JSD describes how different the model’s endpoint output distributions are in the experimental frames. Neither directly specifies how the model moves between those endpoints.

Transition width (w) instead describes the shape of that movement: the fraction of the interpolation path over which (d(t)) rises from 0.1 to 0.9.

The existing correlation therefore leaves an important question unanswered:

> What additional properties distinguish narrow and wide transitions after endpoint differences are held approximately constant?

This is an exploratory investigation. Do not begin by assuming that layer alignment, boundary angle, spline density, context, semantics, or any other proposed mechanism is correct. Inspect the local evidence first and propose hypotheses afterward.

## Success criterion

Produce a concise exploratory report that:

* identifies several reproducible narrow-vs-wide contrasts with similar corpus successor JSD;
* separates genuine transitions from cases where (w) is unreliable because endpoint output movement is tiny;
* describes the concrete differences observed between the contrast cases;
* proposes a small number of hypotheses ranked by how well they fit the evidence;
* gives one simple discriminating experiment for each serious hypothesis.

A null result is complete: it is acceptable to conclude that the current artifacts do not reveal a consistent second factor.

## Existing evidence to inherit

Read the local artifacts in `dir18_continuation_jsd_plateau`, including `REPORT.md`, the per-pair JSON files, raw curves, checkpoint results, and block-scan results.

Use the following findings as background, not as conclusions for this investigation:

* Corpus successor JSD predicts (w) on average, but there is substantial scatter.
* Corpus successor JSD also predicts model-output JSD.
* Width variation is almost absent at initialization and develops during training.
* Previous work finds that plateaus sharpen across downstream layers and are primarily produced by MLPs.
* Plateau boundaries have been associated with high MLP sensitivity, Jacobian peaks, and increased spline-crossing density.
* Humayun et al. report that training can concentrate nonlinear-region boundaries near learned decision boundaries.

References:

* [Activation Plateaus: Where and How They Emerge](https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge)
* [Deep Networks Always Grok and Here Is Why](https://arxiv.org/abs/2402.15555)

## Current status (2026-08-10)

S1-S5 all have a pass, and the experiment S5 recommended has been run. **Main finding:** the leftover
after corpus successor JSD is a per-token additive trait — held-out R^2 0.149 (JSD) -> 0.578 (JSD + one
fitted number per token, ceiling 0.934) — and that per-token number is directly **measurable**: a
token's width against 6 anchor tokens used in no pair predicts its fitted effect at rho = +0.70, and 2
measured parameters match the 123 fitted ones (0.350 vs 0.365). The basin-radius mechanism was tested
and NOT supported (random directions rho = -0.02; anchor directions +0.39, wrong sign). Path-length
normalisation refuted. A real pair-specific remainder survives (across-frame residual r = 0.67), partly
block-0 geometry.

The forward screen has since been run: 40 tokens outside the bank, all 780 pairs predicted from anchor
widths with bank-frozen coefficients, **forward R^2 = 0.397, rho = +0.66, MAE = 0.047** on the 718 gated
pairs, beating model-output JSD (rho = -0.51) on the same pairs.

The anchor-set swap has since been run: two disjoint anchor sets (6 function words, 6 rare content
words) rank the 123 tokens at rho = 0.46 with each other while each recovers the fitted token effect at
rho = 0.57 / 0.61 (mixed set: 0.70). A common trait exists; the anchor set is part of the method.

The layer sweep has since been run: the token ranking survives moving the interpolation site (rho =
+0.72 with block 0 even at block 18) while the median anchor width climbs 0.553 -> 0.800 and the spread
across tokens collapses fivefold. Which tokens are narrow is set early; the sharpening is produced by
the blocks below the site.

**Next step:** the embedding-level probe — measure anchor widths with the site at the input embedding,
and fit a linear probe from a token's static embedding to w_hat_u on 80 of the 123 tokens, testing on
43. A positive result turns the screen into a lookup with no forward pass.

## Stages

* [x] **S1 — Audit the available evidence.** Determine exactly which quantities are already stored per pair and per context. Verify the meanings of corpus JSD, model-output JSD, `w`, `w_ctx`, block-0 geometry, and the raw curves before analyzing them.

* [x] **S2 — Construct matched contrasts.** Find token pairs with approximately similar corpus successor JSD but substantially different (w). Prefer contrasts that are also reasonably similar in model-output JSD, so differences in total endpoint separation are not mistaken for differences in transition shape. Include multiple contrasts rather than selecting one visually appealing example.

* [x] **S3 — Remove normalization artifacts.** Label or exclude cases whose endpoint model-output JSD, (M(1)), or equivalent absolute output movement is too small for normalized (d(t)) and (w) to be informative. Show the absolute movement alongside (d(t)) for every main example.

* [x] **S4 — Perform contrastive case studies.** For each matched narrow-vs-wide contrast, inspect whatever can be recovered or cheaply measured from the local setup, including:

  * the complete (d(t)) and absolute-output-movement curves;
  * the top successor tokens at both endpoints and around the transition;
  * shared versus differing successors in the corpus distributions;
  * variation across the three sentence frames;
  * token frequency, continuation entropy, surprisal, and block-0 geometry;
  * when the width difference emerges during training;
  * whether existing block-level results localize the difference.

  Do not require every candidate variable to be analyzed. Follow the differences that appear repeatedly across several contrasts.

* [x] **S5 — Propose evidence-led hypotheses.** Only after inspecting the cases, propose at most three candidate explanations. For each hypothesis, state:

  1. the observations that motivated it;
  2. which cases it explains and fails to explain;
  3. at least one plausible alternative explanation;
  4. the cheapest experiment that could distinguish them.

  If the observations point toward the MLP/Jacobian/spline mechanism from previous research, explain specifically what varies between narrow and wide cases. Do not merely restate that MLPs create plateaus.

## Required outputs

* A table of the strongest matched narrow-vs-wide contrasts and the criteria used to select them.
* Individual plots for the most informative cases, rather than only aggregate scatter plots.
* A short section titled **Observed patterns** containing only direct evidence.
* A separate section titled **Candidate hypotheses** clearly marked as interpretation.
* A final recommendation for the single most informative next experiment.

## Out of scope

* Do not make quantile regression or lower-envelope significance testing the main project.
* Do not fit a large predictive model over many hand-selected features.
* Do not assume the scatter must have one universal explanation.
* Do not claim a causal or semantic mechanism from endpoint correlations alone.
* Do not modify the conclusions or artifacts in `dir18`.
* Do not launch expensive mechanistic experiments until the local-data exploration identifies a concrete pattern worth testing.

## First step

Load the 1,000-pair final-checkpoint results and produce a reproducible list of pairs that are close in corpus successor JSD but far apart in (w). Add model-output JSD and absolute endpoint movement to this table, remove uninterpretable near-zero-movement cases, and inspect the resulting token pairs before proposing any mechanism.
