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

## Current status (2026-08-12, iteration 12 — COMPLETE; training-checkpoint sweep added)

**Iteration 12 ran the experiment both deliverables named as next, on 17 checkpoints rather than four**
(`experiments/checkpoints.py`, `checkpoints_analysis.py`, `plot_checkpoints.py`; Figures 25–26;
deliverables now carry 26 figures each and pass `check_render.py`).

* **The ordering is learned in the first 512 of 143,000 steps.** Pythia-410M has no width ordering at
  initialisation (across-token sd 0.003 vs 0.060 trained; rho +0.015 with the final ranking) or at
  `step16`; agreement then runs +0.17 / +0.29 / +0.44 / +0.66 / +0.79 at `step32/64/128/256/512`
  (+0.87 after the noise-ceiling correction), +0.94 by `step2000`, and does not change for the
  remaining 98.6% of training. The **level** keeps sharpening until `step64000` (median 0.833 → 0.595):
  the ordering/level split now holds across contexts, model sizes AND training time.
* **Two stages, and only the first is a corpus statistic.** Through `step128` the ranking is entirely
  unigram frequency (rho −0.72 with log10 count, stronger than the finished model's −0.53; zero
  agreement with the final ranking once frequency and successor entropy are partialled out). From
  `step256` a component those statistics do not contain appears (partial agreement +0.45 → +0.60 →
  +0.75 → +0.80). The two corpus statistics explain only $R^2 = 0.375$ of the final ranking.
* **A mature model's lookup reads a young checkpoint first.** The fixed 1.4B embedding lookup ranks
  `step128` at +0.54 and `step512` at +0.81, while a probe refitted inside those checkpoints is at its
  shuffled-control level until `step512` and only matures by `step4000`.
* **Consistency:** this sweep's `step143000` reproduces iteration 11's independent 410M run at
  rho = +1.0000 (n = 123).

**Next step (only if reopened):** the last untested generalisation — a different tokenizer and training
corpus. Measure anchor widths in `gpt2` for the token strings that are single tokens in both
vocabularies, same three frames and six anchors chosen the same way, and compare the ranking with
Pythia's against each model's split-half reliability. ~10 min GPU (`gpt2` is cached). If positive,
follow with a cross-checkpoint transplant: write the final checkpoint's $m_u$ into the `step128` model
and ask whether the ordering appears.

## Previous status (2026-08-12, iteration 11 — COMPLETE; cross-model replication added)

**Iteration 11 ran the experiment both deliverables named as next, on three extra models rather than
one** (`experiments/second_model.py`, `second_ctrl.py`, `second_analysis.py`, `plot_second.py`;
Figures 22–24; deliverables now carry 24 figures each and pass `check_render.py`).

* **The ordering is the token's, not the network's.** Pythia-410M, 1B and 1.4B rank the same 123 tokens
  at rho +0.884 / +0.898 (410M–1B +0.890), and at **+0.98 to +1.00 after dividing by each model's own
  split-half reliability** (0.891 / 0.932 / 0.885) — identical to within measurement noise. The level is
  the network's: median $\hat w_u$ falls 0.749 → 0.658 → 0.620 → 0.549 with size.
* **The free lookup transfers.** The probe read off 1.4B's embedding matrix ranks 410M's measured widths
  at +0.760 and 1B's at +0.745, against +0.765 on 1.4B itself.
* **160M does not have the trait** (+0.207 against a 0.806 ceiling; refitted probe +0.233 ± 0.104;
  lookup transfer +0.043, p = 0.63), so it is acquired between 160M and 410M.
* **Localisation replicates, its matched-control margin does not.** The block-0 MLP is again the only
  early component whose ablation collapses the spread and erases the ordering in all three models, but
  the 410M rerun of the per-token movement-matched dose–response is null on the level-free statistic at
  all nine rungs (9/18 rung × seed comparisons, chance; rho = 0.6 crossing ratio 0.66× against 1.3× at
  1.4B). Deliverables now claim the site, not the per-bit specificity, and point at the transplant.

**Next step named at iteration 11 — DONE in iteration 12, see the current status above.** Where does the trait come from? Measure anchor widths for the 123
tokens in Pythia-410M at `step1000`, `step8000`, `step32000`, `step143000`; correlate each checkpoint's
ranking with the final one and with the token's unigram frequency and successor entropy (both in
dir18's manifest). Early-and-sharpening ⇒ the lookup reads a corpus statistic computable with no model;
late-and-gradual ⇒ it reads what the network learned. ~15 min GPU.

## Previous status (2026-08-12, iteration 10 — COMPLETE, feedback addressed)

**The direction is finished.** All stages S1–S5 are done, every required output is delivered, both
deliverables pass `check_render.py`, and no unaddressed `human_feedback*` / `*REVIEW*` file remains
(`human_feedback.addressed.md`).

Iteration 10 addressed the operator's control-matching objection: the dose–response's random control
was matched to the block-0 MLP dose on the *mean* output JSD over the 12 tokens, while the conclusion
ranks those tokens individually. `experiments/dose2.py` reruns it with the control's scale
binary-searched **per endpoint prompt** (three seeds), plus a paired per-token Wilcoxon test.
The objection was material: the old control mis-dosed individual tokens by 0.08×–8.5×, and correcting
it cuts the headline margin from "a random disturbance needs ~3.5× more output movement" to **1.3×**,
narrows the claim's band from 0.007–0.103 bits to below 0.03 bits, and retracts "the spread collapses
identically in both arms" above 0.014 bits. The localisation survives: MLP below its matched control
in 15/15 rung × seed comparisons in the live band, and the level-free paired per-token test gives ~2×
more width movement under the dose (p = 0.034 / 0.016 at the two live doses).

With budget left after the feedback, iteration 10 also ran the follow-up both deliverables named
(`experiments/mlp_read.py`, `mlp_geom.py`; Figure 20): transplanting the block-0 MLP's output vector
transports the width almost completely (slope +0.913 on the donor, nothing from the recipient's
remaining state), while a probe from that vector is no more accurate than one from the static
embedding, and no low-dimensional part of it carries the trait (`mlp_rank.py`, Figure 21). Deliverables
now carry 21 figures each and pass `check_render.py`. No `STOP` written: budget remained and the
direction has a concrete next experiment (a second model).

That follow-up has since been run too (iteration 10, part 2): the probe/transplant experiment both
deliverables named. **Transplant = strong positive**: overwriting a token's block-0 MLP output $m_u$
with another token's transports the width (per-recipient rho = +0.968, slope +0.913 on the donor;
recipient's remaining state rho = −0.104; self-transplant reproduces the baseline exactly), and $m_u$
is context-free (cosine 1.0000 across frames — Pythia's parallel residual means block 0's MLP reads the
embedding before attention writes), which is why the static-embedding lookup works at all.
**Probe = null**: rho = +0.748 from $m_u$ against +0.764 from the embedding row and +0.772 from the
full post-block-0 state — the component carries the trait without making it more readable. Scale caveat
stated: a transplant moves the output 0.738 bits and $m_u$ is 0.79 of the state's norm / 0.76 of its
across-token spread.

The rank sweep has since been run too (iteration 10, part 3; `experiments/mlp_rank.py`, Figure 21) and
is a clean **negative**: transplanting only the top $k$ principal components of $m_d - m_r$ gives slope
+0.256 / +0.298 / +0.274 at $k$ = 8 / 32 / 64 (0.24 / 0.55 / 0.79 of the across-token variance) against
+0.913 for the intact vector, while a top-64 transplant already causes 95% of the full transplant's
output movement; the discarded tail transfers nothing (−0.022) and a random 64-dim subspace gives
+0.000. Partial transplants disturb (mean $\hat w$ 0.565 → 0.61–0.65, spread compressed) where the
complete one exchanges widths cleanly (mean 0.573, sd 0.076). The trait is a property of the whole
vector, not a low-dimensional feature.

**Next step named at iteration 10 — DONE in iteration 11, see the current status above.** Leave this
model. Repeat the cheap end of the pipeline on a second
Pythia size (410M or 2.8B) — anchor widths for ~60 tokens, the embedding probe, the block-0 MLP
ablation — and compare the probe's held-out rho (+0.76 here), the cross-model rank agreement of
measured widths on shared tokens, and whether the block-0 MLP is again the single early carrier. That
decides whether the free screen is a property of tokens or a per-model calibration. Cost ~20 min GPU.

### Record of what was established (iteration 8 status, retained)

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

The dose-response has since been run — and rerun in iteration 10 with the control matched PER TOKEN
after operator feedback, which is the version that stands. Softening the ablation to alpha = 0.1 ... 1
and matching every dose to a random perturbation of the same residual stream that moves EACH TOKEN's
output by the same number of bits (three seeds), the MLP arm is below its matched control at all five
rungs up to 0.03 bits and in 15/15 rung x seed comparisons (rho +0.84/+0.98, +0.64/+0.91,
+0.62/+0.76): it crosses rho = 0.6 at 0.031 bits, the control at 0.041, so **a random disturbance needs
~1.3x more output movement for the same damage**. Above 0.1 bits the arms cross and are noise
(SE(rho) ~ 0.3 at n = 12). The load-bearing statistic is paired and per-token: the dose moves a token's
width ~2x as far as that token's own matched control (0.074 vs 0.036 at 0.0068 bits, p = 0.0010),
still so after each arm's mean level shift is removed (p = 0.034, 0.016 at the two live doses). The
across-token sd collapses identically only through 0.014 bits; beyond that the dose compresses harder.
Positive mechanistic localisation, modest in size: the trait is realised in the block-0 MLP's
contribution to the final-position residual stream.

**Recommended follow-up (carried into the status block above):** stop destroying, start reading. Fit a ridge probe from the block-0 MLP's final-position
output m_u to the measured anchor width (held-out, shuffled-target control) over the 123 endpoint
tokens, and transplant m_u from a narrow token onto a wide token's forward pass. Probe beats the
embedding probe's rho = +0.76 and the transplant moves the recipient toward the donor -> the trait is a
readable vector; both null -> the block-0 MLP is a necessary stage rather than the store, and the
static-embedding lookup stands as the practical deliverable.

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
