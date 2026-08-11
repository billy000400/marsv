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

## Current status (2026-08-11, iteration 7)

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

The embedding probe has since been run: the trait is already in the static embedding. Anchor widths
measured at the input embedding agree with block 0 at rho = +0.79, and a ridge probe from W_E[u]
predicts a held-out token's measured width at **rho = +0.764 +- 0.045, R^2 = 0.514** (embedding-norm
baseline +0.597, shuffled-target control -0.201). End to end, a screen built from embeddings alone
predicts the 718 unseen pairs at **R^2 = 0.213, rho = +0.53** with no forward pass, against 0.397 for
the measured screen — a two-tier screen: free table for triage, measured widths when accuracy matters.

The vocabulary-wide test has since been run: the probe applied to all 50,304 embedding rows and 32
tokens measured from four classes the pool excludes (ordinary words outside the pool, subword
fragments, punctuation/numerals, capitalised names) give rho = +0.60 (p = 3.0e-4) between predicted and
measured width, MAE 0.046, with measured widths spanning 0.367-0.686 against the pool's 0.361-0.660.
The lookup is not confined to the token class it was built on.

The frame-shape control has since been run: the same measurement in four differently shaped contexts
keeps the token ranking at rho = +0.844 (mid-sentence), +0.770 (question), +0.735 (list) and +0.501
(code) against the original, where two frames of the ORIGINAL shape agree at +0.822. The level moves
(median w_hat 0.530 -> 0.705). The ordering is carried by the token; the scale is set by the context.

The embedding intervention has since been run, and is a NULL: editing a token's embedding row along the
probe's gradient by a step the probe says should move width by +-0.05 moves the measured width by
0.0027 on average (slope -0.023, sign agreement 0.39); a matched-norm random direction moves it 0.0008.
The direction that predicts width does not set it. Caveat that defined the follow-up: the edits shifted
the model's output by only 0.0001 bits, so that test never reached a behaviourally meaningful regime.

The behaviour-calibrated intervention has since been run, and it converts the null into the direction's
clearest causal statement. With the step grown until the token's output moves 0.05 / 0.1 / 0.2 bits,
width moves 0.10-0.15 units (fifty times more, so the earlier null was a step-size null) — but a random
direction matched on output movement moves it just as much (0.123 vs 0.127, Wilcoxon p = 0.47), the
probe's signed prediction has slope -0.002, and **all 144 edits widen** where the probe predicts
opposite signs. The edits do not steer width, they **destroy the trait**: after a 0.2-bit edit the 12
tokens land at mean w_hat 0.68 with sd 0.02 across tokens, against 0.543 +- 0.083 before, narrowest
tokens moving furthest (rho = -0.78 / -0.94). The probe direction is behaviourally special (1.5-1.8x
smaller step for a given output movement) but does not carry width. The compression is strong but not
total: the post-edit ranking still agrees with the original at rho = +0.73 / +0.85 after a 0.05-bit
edit and +0.57 / +0.36 after a 0.2-bit one.

The displacement-norm ladder has since been run, and it settles the question the fixed-displacement
test could not (that test is superseded: its quiet direction was no quieter than a random one, because
the linear response at a 0.05 step does not survive a step of norm 1.84). Rebuilding the quiet and loud
directions at each rung by MEASURING what 24 random directions do at that norm gives an 8x separation
at norm 1.8 (0.049 vs 0.402 bits), and with a real contrast the earlier reading **reverses**: at the
same displacement the quiet edit keeps the token ordering (rho(before, after) = +0.94, p = 4e-6) and
the loud edit erases it (+0.08, p = 0.80); the quiet direction widens less in the paired test at every
rung (p = 5e-4 at norm 0.9). The level still follows displacement (0.543 -> 0.656 even for the quietest
direction, sd across tokens 0.083 -> 0.038). Displacement compresses the level; behaviour destroys the
ordering — so the free vocabulary-wide lookup reads a behavioural property, not a geometric accident.

The mode split has since been run. Partitioning each edit's output change by successor token shows the
damage is **tail-weighted**: the token's 32 most likely successors hold 0.707 of its probability mass
but absorb only 0.389 of the divergence a loud edit produces, and louder directions are more
tail-weighted still (rho(bits, top-share) = -0.36). So the behaviour whose disruption coincides with the
trait's collapse is not mainly what corpus successor JSD scores. The steering half is a bounded null:
random directions span only S = 0.36-0.56, and at matched output movement (0.410 vs 0.453 bits) the
most top-heavy and most tail-heavy both flatten the ordering (rho = -0.08 and -0.37, n = 12) and widen
every token toward ~0.66, with one surviving paired difference (top-heavy widens more, +0.124 vs
+0.108, p = 0.009, while moving the output less).

The construction test has since been run, and it closes the mode-split line. Building the top-heavy and
tail-heavy directions from a 24x24 generalised eigenproblem (the top-mass share $S$ is a Rayleigh
quotient in the mixing coefficients) separates $S$ 0.856 vs 0.179 in the small-step regime — three
times the random-draw spread — but once both are rescaled to the 0.4 bits at which width responds they
land at **0.369 vs 0.390**, indistinguishable and both below the base mass 0.71, with the paired
difference running backwards ($p = 0.09$). Both widen every token to $\hat w_u \approx 0.67$ and both
erase the ordering ($\rho = -0.16$, $-0.28$). The tail-weighting of a large embedding edit is set by
the step size, not the direction; the top-mass hypothesis is untestable by embedding edits; the trait
belongs to the token's whole output map.

The component ablation has since been run, and it is the first positive mechanistic localisation.
Mean-ablating each of the 102 attention heads and MLPs in blocks 0-5 one at a time leaves the token
ordering untouched in 101 cases (median rho = +0.99; every head >= +0.97, every MLP above block 0 >=
+0.90). Only the **block-0 MLP** collapses the across-token spread (sd 0.084 -> 0.018), lifts every
token to w_hat ~ 0.82 and erases the ordering (rho = -0.10). Confound stated in the deliverables: it is
also the only early component whose removal the model feels (0.451 bits vs <= 0.007 for every other),
and 0.4 bits is the rung at which the ladder showed any disturbance flattens the ordering.

**Next step:** break that confound with a dose-response. Blend the block-0 MLP's final-position output
toward its mean with weight alpha = 0.1 ... 1, and at each alpha run a control that perturbs the same
residual stream with a random vector rescaled to the same output movement in bits; plot rho(before,
after) against bits for both arms. Separated curves place the trait in the block-0 MLP; coincident
curves say disturbance as such kills it, closing the mechanistic line and leaving the static-embedding
lookup as the practical deliverable.

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
