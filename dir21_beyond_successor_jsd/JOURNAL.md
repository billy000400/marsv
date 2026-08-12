# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-08-10 — iteration 1: audit dir18, then decompose the residual scatter

**Did.** Feedback check first: no `human_feedback*`/`*REVIEW*` files in this direction. Audited the
inherited artifacts (S1): `dir18/results/assay_large_step143000.json` stores per pair `jsd_A/jsd_B`
(selection/measurement corpus JSD), `w_ctx` and `out_jsd` per frame, `cos0`/`dist0` block-0 geometry,
and `curves_large_*.npy` holds the raw (1000, 3, 50) `d(t)` curves; `pair_manifest_large.json` adds
per-token corpus counts, continuation entropy and in-frame model surprisal. `data/` (raw corpus counts)
is gone, so token-level corpus statistics are only available through the manifest — enough for this
work. Wrote `experiments/common.py` (merge into one table) and `experiments/explore1.py`
(gate + matched contrasts + cross-validated decomposition + interaction test + path-length test), plus
`plot_explore1.py`. All CPU, ~2 s, no GPU.

**Learned.**
1. The single biggest thing corpus JSD misses is **additive in the tokens**. One free parameter per
   token, no interaction, takes held-out R^2 from 0.149 to 0.578 (ceiling 0.934) and alone beats every
   pair-level predictor tried. That reframes the direction's question from "what pair property is
   missing" to "what per-token property is this".
2. The S3 concern turned out differently than PLAN assumed: low-movement pairs are **not** unreliable
   (their across-frame `w` spread is the smallest in the sample) — they are uninterpretable in a
   different way, sitting at w ~ 0.7 because a normalised score has nothing to normalise. So the gate is
   an interpretability filter, not a noise filter, and the deliverables say so.
3. The deflationary path-length explanation is refuted: converting `w` into residual-distance units
   makes it *more* dispersed (CV 0.158 -> 0.216) and dist0 correlates positively with `w`.
4. Real pair-specific structure survives (residual across-frame r = 0.67), so additivity is dominant
   but not complete; block-0 geometry recovers part of it (held-out R^2 0.648 -> 0.723).

**Assumptions logged (loop mode, no human to ask).** (a) Gate at 0.2 bits of min-frame endpoint
movement — chosen because it is the point below which `w` saturates toward 0.8 in Figure 1 right;
alternatives rejected: median split (throws away 500 usable pairs) and no gate (lets "model barely
distinguishes them" masquerade as "wide transition"). (b) Contrast tolerances 0.02 bits / 0.05 bits /
dw >= 0.15 with all-frame consistency — rejected a looser 0.05-bit JSD tolerance that admitted
contrasts explainable by the JSD trend itself. (c) 5-fold CV with identical folds and ridge 1e-3 rather
than a mixed-effects fit — the additive model has 123 parameters and an in-sample R^2 would flatter it;
rejected leave-one-token-out because token effects are the estimand.

**Next step.** Run the recommended experiment: the partner-free basin-radius probe. For each of the 123
endpoint tokens x 3 frames, step from x_u along random directions and toward anchor tokens used in no
pair, and record the distance at which output JSD from z_u first crosses a threshold. Test whether that
radius predicts the fitted a_u out of sample (H1) or whether a_u instead tracks endpoint logit norm
(the nuisance reading). GPU: pythia-1.4b fp32 fits in the 7.2 GB share at small batch.

On track? yes — S1-S3 complete, S4/S5 first pass done (contrast case studies + three ranked
hypotheses), ~55% done, no blocker.

## 2026-08-10 — iteration 1, second step: ran the experiment the first step recommended

**Did.** With ~4 h of the budget left after persisting the first deliverables, I ran the recommended
experiment rather than stopping at the hypothesis. Two probes, both partner-free, both using 6 anchor
tokens that appear in none of the 1,000 pairs: `basin_probe.py` (great-circle sweep from a token's
block-0 state, output movement in bits, radius = first crossing of tau) and `anchor_width.py` (dir18's
full interpolation protocol against the anchors). `transfer.py` joins them to the fitted token effects;
`plot_transfer.py` makes Figure 6. GPU: fp32, memory fraction 0.225, ~5 min each, no OOM.

**Learned.**
1. **The transfer test passed decisively.** A token's width measured against strangers predicts its
   fitted effect at rho = +0.70, and two measured numbers match 123 fitted ones at the pair level
   (0.350 vs 0.365). That upgrades the finding from "the residual is additive" to "width is a
   measurable per-token trait", which is the version an auditor could use.
2. **My own mechanistic hypothesis was wrong.** The basin picture predicted that a token which holds
   its output longer contributes NARROWER transitions. Random-direction radius is unrelated to the
   token effect (rho = -0.02) and anchor-direction radius correlates with the opposite sign (+0.39).
   The likely reason the absolute-movement radius fails: it measures JSD movement, which a peaked
   output distribution racks up quickly, whereas w measures relative logit displacement along
   z_u -> z_v. I dropped the basin hypothesis rather than reinterpreting it into survival.
3. Output entropy (-0.30) and endpoint logit norm (-0.23) join frequency/entropy/surprisal as weak
   correlates — five weak per-token proxies, none of them the trait itself.

**Assumption logged.** 6 anchors, chosen deterministically as every ~87th token of the sorted eligible
pool minus the 123 endpoints, so the anchor set is not hand-picked; rejected using random endpoint
tokens as anchors (would break the disjointness that makes the transfer test meaningful) and using more
anchors (18 curves/token already gave 100% curve validity and a rho of 0.70).

**Next step.** The forward screen: 40 endpoint tokens outside the 123, measure anchor widths only,
predict 200 fresh pairs before running them, then run and score. Also worth pairing with H1's control —
recompute anchor width with two disjoint anchor sets (function words vs rare content words) and check
the token rankings agree, which separates a per-token trait from a similarity-to-typical-token statistic.

On track? yes — S1-S5 all have a pass, the recommended experiment is done and changed the story
(reported), ~75% done, no blocker.

## 2026-08-10 — iteration 1, third step: forward screen on unseen tokens

**Did.** Ran `forward_screen.py`: 40 tokens outside the bank, anchor widths measured, all 780 pairs
predicted with bank-frozen coefficients, then all 780 pairs actually run and scored (~14 min GPU).
Wrote `plot_forward.py` (Figure 7) and folded the result through both deliverables.

**Learned.** The screen generalises: R^2 = 0.397, rho = +0.66, MAE = 0.047 on 718 gated pairs with zero
parameters fitted on the new tokens, beating model-output JSD (rho = -0.51) on the same pairs. The new
tokens are a wider mix than the bank's 123 (rarer content words like ' delicious', ' extraordinary'),
and the bank's slope transferred without adjustment — evidence the additive coefficient is a property
of the measurement, not of the original token sample. Note the honest gap: corpus JSD cannot be scored
on the new tokens because dir18's count arrays were deleted from disk, so the only baseline available
there is the model-side one.

**Assumption logged.** Scored all 780 pairs rather than a 200-pair sample (the plan's sketch) — running
everything removes a selection choice and cost only ~4 extra minutes. Terciles rather than a threshold
for the "screen separates narrow from wide" panel: no operating point has been justified yet.

**Next step.** The anchor-set swap: recompute anchor widths against six function words and six rare
content words, correlate the two rankings of the 123 tokens. That decides whether w_hat_u is a token
trait or a similarity-to-anchor statistic, which is the main thing the current write-up cannot say.
Follow-on: repeat anchor widths at blocks 6/12/18 to locate where the trait is established.

On track? yes — S1-S5 complete and the two experiments those stages recommended have both been run and
reported, ~85% done, no blocker.

## 2026-08-10 — iteration 1, fourth step: the anchor-set swap

**Did.** Ran `anchor_swap.py` (two extra anchor sets x 123 tokens x 3 frames), plus `plot_swap.py`
(Figure 8), and folded the result into both deliverables. Also ran a five-seed stability check on the
cross-validated R^2 numbers (sd <= 0.008; noted in Methods).

**Learned.**
1. The answer is a mixed one and I reported it as such: every anchor set recovers the same per-token
   trait (rho 0.57-0.70 with the fitted effect), but two disjoint sets agree with each other only at
   rho = 0.46, and their usefulness as pair-level predictors differs a lot (held-out R^2 0.146 vs 0.265
   vs 0.350). So w_hat_u is not an anchor-free constant. The honest framing is "the anchor set is part
   of the method", which also justifies the mixed set the forward screen used.
2. Function-word anchors are the weakest measuring stick. Plausibly because the 123 endpoint tokens are
   mostly content-ish, so function-word anchors probe a less relevant direction — untested.
3. **Process failure worth recording:** the first run of this control silently used ONE function-word
   anchor, because my hand-written list of function words barely intersected the eligible pool. The
   printout caught it. Lesson applied: the rerun asserts the arm has exactly 6 anchors before using it,
   and only recomputes arms that are missing or malformed.

**Next step.** Layer sweep of the anchor width (blocks 6, 12, 18) to see whether the token ranking is
already fixed at block 0 or is reordered downstream. That decides whether to look at the unembedding
geometry or at the MLPs.

On track? yes — S1-S5 complete, four experiments run and reported (probes, forward screen, anchor-set
swap), deliverables current-best and render-checked, ~95% done, no blocker.

## 2026-08-10 — iteration 1, fifth step: layer sweep

**Did.** Parameterised the block-0 hook (`Patcher(model, layer=L)`, default unchanged so the earlier
scripts still reproduce), ran `layer_sweep.py` at blocks 6/12/18 (~8 min — later sites are cheap because
fewer blocks run below them), plotted Figure 9, and folded the result into both deliverables.

**Learned.** Two things separate cleanly, which I had not expected to see so sharply:
1. *Which* token is narrow is essentially fixed at block 0 — the ranking still agrees at rho = +0.72
   when the state is interpolated 18 blocks in.
2. *How narrow anything is* depends entirely on the remaining depth: median anchor width climbs
   0.553 -> 0.800 (0.8 = proportional response, i.e. no transition at all) and the interquartile range
   across tokens falls 0.102 -> 0.020. Interpolate five blocks from the output and there is nothing left
   to measure in any token.
   This is the cleanest evidence in the direction for the sharpening-happens-downstream picture, and it
   also explains why the correlations with a_u decay with depth: there is little dynamic range left, a
   caveat now stated in Limitations rather than read as "the trait vanishes".

**Next step (left for the next iteration).** The embedding-level probe: measure anchor widths with the
site at the input embedding, and fit a linear probe from a token's static embedding to w_hat_u on 80 of
the 123 tokens, testing on 43. A positive result makes the screen a lookup instead of a forward pass.
I stopped adding experiments here deliberately — the report now carries 9 figures and 12 observed
patterns, and the direction's success criterion asks for a concise exploratory report, so the remaining
budget went to a coherence pass rather than a sixth experiment.

On track? yes — S1-S5 complete, five experiments run and reported, deliverables current-best and
render-checked, no blocker.

## 2026-08-10 — iteration 2: finished the embedding probe and made the screen a lookup

**Did.** Feedback check first: no `human_feedback*` / `*REVIEW*` files in this direction. Found that
the previous session died right after `embed_probe.py` wrote `results/embed.json` — the experiment had
run but was never plotted or reported, so this iteration completed it rather than starting anything
new. Plotted it (`plot_embed.py`, Figure 10), added the norm-only baseline, and wrote
`experiments/embed_forward.py`: fit the probe on the 123 bank tokens, look up w_hat for the 40
forward-screen tokens from their embedding rows, and score the same 718 gated pairs with bank-frozen
coefficients. Folded everything into both deliverables; `check_render.py` passes.

**Learned.**
1. **The trait is in the embedding.** A ridge probe on W_E[u] predicts a held-out token's measured
   anchor width at rho = +0.76 (R^2 = 0.51), 50/50 splits positive. Measuring the same quantity with
   the interpolation site moved *below* block 0 agrees with block 0 at rho = +0.79. Combined with last
   iteration's layer sweep, the picture is now sharp at both ends: the ranking is present in the input
   embedding and unchanged by 18 blocks of computation, while the sharpness itself is manufactured by
   the blocks below the interpolation site.
2. **It is not just embedding norm.** Norm alone gives rho = +0.60 under the same splits — a large
   share, and the honest way to report it, since norm tracks frequency in Pythia. The full embedding
   adds a clear increment (0.76 vs 0.60), and shuffled targets give -0.20, so the probe is not fitting
   noise with 2048 features on 80 points.
3. **The end-to-end lookup works but loses half the R^2**: 0.213 vs the measured screen's 0.397 on the
   same 718 unseen pairs (rho +0.53 vs +0.66). The mechanism is visible in Figure 10 far right — ridge
   shrinks the predicted range, so the lookup under-disperses. I reported this as a two-tier screen (free
   table for triage, 18 curves per token when accuracy matters) rather than presenting the lookup as a
   replacement.

**Assumptions logged (loop mode).** (a) The pair-level mapping for the lookup screen is re-estimated on
bank pairs using *out-of-fold* probe features rather than reusing the measured-width slope — the two
feature scales differ because of shrinkage, and reusing the old slope would have miscalibrated the
lookup; no information from the 40 new tokens enters either way. (b) The norm-only baseline was moved
out of `embed_probe.py` into `embed_forward.py` after the rerun proved too slow (six 2048-feature ridge
probes, >15 min under 4-way CPU contention); rejected the alternative of leaving the number
uncomputed, since "is the probe just reading frequency off the norm?" is the first question a reader
asks. (c) Kept the whole embedding story in one 4-panel figure instead of adding two — the report
already carries ten figures and the direction asks for a concise exploratory report.

**Next step.** Test the lookup outside the curated pool: apply the probe to all 50,304 embedding rows,
take ~30 tokens spanning the predicted range including subword fragments, punctuation, numerals and
capitalised names, and measure their anchor widths at block 0. Everything reported so far uses
`dir18`'s common-alphabetic-word pool, so this decides whether the vocabulary-wide table is publishable
as an auditing artifact or whether the screen's scope is common English words.

On track? yes — S1-S5 complete, six experiments run and reported, deliverables current-best and
render-checked, no blocker.

## 2026-08-10 — iteration 2, second step: the lookup outside the curated pool

**Did.** Ran the experiment the first step recommended (`vocab_probe.py` + `plot_vocab.py`, Figure 11):
probe applied to all 50,304 embedding rows, 32 tokens selected from four classes the pool excludes
(ordinary words outside the pool, subword fragments, punctuation/numerals, capitalised names), anchor
widths measured at block 0. Folded into both deliverables; `check_render.py` passes.

**Learned.** The ranking survives leaving the pool: rho = +0.60 (p = 3.0e-4) over the 32 tokens,
MAE 0.046, no class inverting the relation (per-class rho +0.24 to +0.83, but n = 8 each, so those are
indicative). Two things worth carrying forward: (1) the classes sit at systematically different levels
— rarer words wider (0.632), capitalised names and punctuation narrower (~0.53) — so a vocabulary table
would be predicting a real spread, not a constant; (2) measured widths outside the pool span
0.367-0.686, essentially the pool's own range, which is the argument that the table is useful rather
than merely correlated. Shrinkage is confirmed as the lookup's main weakness (predicted sd 0.047 vs
measured 0.073).

**Assumption logged.** Eight tokens per class spaced over each class's own predicted quantiles, rather
than 32 tokens spaced over the global predicted range — the latter would have been dominated by
whichever class occupies the extremes and could not have shown per-class behaviour at all. The cost is
that each per-class correlation rests on 8 points, stated in the report rather than glossed.

**Next step.** The frame-shape control: every measurement in this direction uses three short
declarative frames ending in `was`, with the token final. Re-measure anchor widths for the 123 tokens
in structurally different contexts (mid-sentence continuation, interrogative, list, code-like) and
correlate each context's ranking with the current one. That decides whether w_hat_u is a token property
or a token-in-this-slot property, which is the largest remaining scope caveat on the screen.

On track? yes — S1-S5 complete, seven experiments run and reported, deliverables current-best and
render-checked, no blocker.

## 2026-08-10 — iteration 2, third step: the frame-shape control

**Did.** Ran the experiment the second step recommended (`frame_control.py` + `plot_frames.py`,
Figure 12): anchor widths for the 123 tokens re-measured in four contexts of different shape
(mid-sentence continuation, interrogative, colon-list, code prefix), each context's ranking correlated
with the original, and — the part that makes the numbers interpretable — the agreement among the three
original frames computed the same way as a reference. Folded into both deliverables; `check_render.py`
passes.

**Learned.** The cleanest generalisation result in the direction so far. rho with the original ranking:
+0.844 (mid-sentence), +0.770 (question), +0.735 (list), +0.501 (code), against +0.822 for two frames
of the original shape. So in a nearby context the measurement transfers with no measurable loss, and it
degrades gracefully with contextual distance rather than collapsing. Meanwhile the level moves a lot
(median 0.530 list -> 0.705 code) and code compresses the token spread fivefold (IQR 0.049 vs
0.107-0.123). The same rank-survives / level-moves split now appears three times — across anchor sets,
across depth, across context — which is a stronger statement about what w_hat_u is than any single one
of them: the ordering is carried by the token, the scale by whatever else is in the picture.

**Assumption logged.** Compared each new context against the 3-frame median ranking, using the
frame-to-frame agreement of the original three as the reference ceiling, rather than comparing single
frame to single frame throughout — the 3-frame median is the quantity the screen actually uses, and the
reference is reported alongside so the comparison is not read as if 1.0 were achievable. Rejected
adding more contexts: four shapes at 123 tokens each already cost ~7 min of GPU and the pattern was
unambiguous.

**Next step.** The first causal test: take the probe's weight vector, add +-epsilon times its unit
direction to a token's embedding row, re-measure that token's anchor width, and compare against a
matched-norm random direction, over ~20 tokens at three step sizes. Everything in the direction so far
is correlational; this asks whether the embedding direction the probe found is a handle on transition
sharpness. ~20 min of GPU.

On track? yes — S1-S5 complete, eight experiments run and reported, deliverables current-best and
render-checked, no blocker.

## 2026-08-10 — iteration 2, fourth step: the embedding intervention (null)

**Did.** Ran the causal test the third step recommended (`embed_intervene.py` + `plot_intervene.py`,
Figure 13): edited 16 tokens' embedding rows along the probe's prediction gradient by steps sized to
move the probe's own prediction by +-0.025 and +-0.05 width units, re-measured each token's anchor
width, and compared against matched-norm random directions. Folded into both deliverables;
`check_render.py` passes.

**Learned.** A clean null, and I reported it as one. Measured width moved by 0.0027 on average against
0.0375 requested, with slope -0.023 and sign agreement 0.39 — no effect, and if anything slightly
against the probe. Random directions moved it by 0.0008. I checked one salvage reading (that the
perturbation compresses width toward the middle rather than translating it) and dropped it: the
per-token response slope is unrelated to the token's base width (rho = -0.115, p = 0.67). The
diagnostic that matters for what comes next is the output shift: 0.0001 bits at a 5%-of-row-norm step,
so the edit was functionally invisible to the model and the null is at least partly a step-size null.
That is stated in the report rather than buried, and it defines the follow-up.

**Assumption logged.** Sized the steps by the probe's own prediction (delta = Delta g / ||g||^2) rather
than by a fixed fraction of the embedding norm — that makes "the probe says +0.05, the model gives
+0.003" a directly interpretable statement, which a norm-based step would not be. The cost is that the
resulting steps turned out to be behaviourally tiny; the fix is the next experiment's
behaviour-calibrated step, not a reinterpretation of this one.

**Next step.** Repeat the intervention with the step grown along the probe direction until the token's
next-token distribution moves by 0.05 / 0.1 / 0.2 bits (the basin sweep's thresholds), keeping the
matched-norm random control. If width then moves along the probe direction and not the random one, the
direction is a lever and this iteration's null is a step-size artifact; if both move width, no single
embedding direction carries the trait and the search moves to block 0's attention pattern and MLP
response, which the layer sweep says is where the ordering is already fixed.

On track? yes — S1-S5 complete, nine experiments run and reported (two of them nulls, both reported as
such), deliverables current-best and render-checked, no blocker.

## 2026-08-11 — iteration 3: the behaviour-calibrated intervention (the null was a step-size null)

**Did.** Ran the experiment the previous iteration recommended (`embed_intervene2.py` +
`plot_intervene2.py`, Figure 14): 12 tokens, embedding edits grown along the probe direction until the
token's next-token distribution moves 0.05 / 0.1 / 0.2 bits, both signs, with a random direction
calibrated to the SAME output movement as the control (144 re-measurements). Folded into both
deliverables; `check_render.py` passes.

**Learned.** Three things, and the third is the one worth keeping. (1) The previous null WAS a step-size
null: width moves 0.10-0.15 width units once the edit is behaviourally real, against 0.003 before —
fifty times more. (2) The specificity test nonetheless fails in every way available: random directions
matched on output movement move width just as much (0.123 vs 0.127, Wilcoxon p = 0.47), the probe's
signed prediction has slope -0.002, and all 144 edits widen although the probe predicts opposite signs
for opposite steps. (3) The positive finding: the edits do not slide tokens along a width axis, they
COLLAPSE the trait — after a 0.2-bit edit the 12 tokens sit at mean w_hat 0.68 with sd 0.02 across
tokens, against 0.543 +- 0.083 before, and the narrowest tokens move furthest (rho = -0.78 / -0.94).
Narrowness is a fragile property of the exact trained embedding. Two side facts sharpen this: the probe
direction needs a 1.5-1.8x smaller step to reach a given output movement (so it is behaviourally
special, just not for width), and the larger random displacements (edited row norm 1.90 vs 1.40) land
at the same width, so the collapse is indexed by output movement rather than by geometric displacement
— which is what makes the next experiment the right one.

**Process note.** The first launch of the run silently started a second copy (a `cd X && nohup ... &`
compound backgrounded the `cd` too, so my follow-up `tail` looked in the wrong directory and I assumed
nothing had started). Two processes then raced on the same output JSON. Caught it by comparing the
per-token count in the JSON against the log, killed both, and re-ran clean — the reported numbers come
from a single clean run.

**Assumption logged (loop mode).** Calibrated by scanning a geometric ladder of step norms and
interpolating in log-log rather than bisecting each target separately: one scan serves all three
budgets per direction and the achieved / requested ratio came out at median 1.00 (IQR 0.91-1.05), so
the extra accuracy of per-target bisection would have bought nothing for ~3x the calibration cost.
Rejected also: dropping the probe-calibrated intervention from the deliverables as superseded. It tests
a different claim (the probe's own quantitative prediction, off by 20x) and it is what motivates this
experiment's design, so it stays, framed as the loophole this run closes rather than as history.

**Next step.** Displace the embedding row a long way while keeping the model's output fixed: for each
token search directions whose output shift stays under 0.005 bits, take a step of the same norm a
0.2-bit edit needed (a displacement pattern 14 says should collapse width), and re-measure w_hat. Width
surviving => the trait is a function of the behaviour the embedding induces, and the search moves to
which output modes the token activates; width collapsing anyway => the trait is tied to the embedding's
exact location, and the free vocabulary-wide lookup is reading a geometric accident rather than a
behavioural property, which is a caveat an auditor using the table would need.

On track? yes — S1-S5 complete, ten experiments run and reported (three of them nulls, all reported as
such, and this one turned its null into the direction's clearest causal statement), deliverables
current-best and render-checked, no blocker.

## 2026-08-11 — iteration 3, second step: the fixed-displacement test (and a correction)

**Did.** Ran the experiment the first step recommended, in the form the budget allowed
(`embed_quiet.py` + `plot_quiet.py`, Figure 15): hold the displacement norm fixed at the value each
token's 0.2-bit edit needed, and vary how loudly the model responds to the direction taken — quietest
and loudest combinations of 48 probed directions (from the SVD of their logit responses at a step of
0.05), plus a plain random direction.

**Learned.** Two things, one of which corrects yesterday's write-up. (1) The construction fails at this
scale: the "quiet" direction moves the output by 0.181 bits against a random direction's 0.165, because
the linear response measured at a 0.05 step does not survive a step of norm 1.84. I reported that as a
limitation of the test rather than quietly dropping it. (2) The decoupling still works as an
observation: at fixed displacement norm the width change is flat in the output movement actually
produced (rho = +0.07, p = 0.67, over 0.03-0.77 bits), while the collapse reproduces in all three
directions (mean w_hat 0.648-0.675, sd 0.019-0.039 against 0.543 +- 0.083). That **withdraws** the
claim I wrote in the first step, that the landing point is indexed by output movement rather than
displacement — it was inferred from probe-vs-random step norms in a run where norm and bits moved
together, and this run decouples them and finds no bits effect. Corrected in REPORT.md, RESULTS.md and
CHANGELOG.md. I also checked the post-edit ranking, which I should have checked yesterday: it partly
survives (rho = +0.62 here; +0.73/+0.85 after a 0.05-bit edit and +0.57/+0.36 after a 0.2-bit one in
the calibrated run), so "destroys the trait" became "compresses the trait, leaving a residual ordering"
everywhere it appeared.

**Assumption logged (loop mode).** Built the quiet/loud directions from a 48-direction random subspace
rather than the full 2048-dimensional Jacobian: the full Jacobian needs 2048 forward passes per token
per frame, roughly 40x the budget, and the subspace version was expected to give a usable contrast. It
did give a contrast in the linear regime but not at the displacement used, which is exactly the
limitation now reported and what makes the norm ladder the right next experiment. Rejected: rerunning
immediately at smaller norms this iteration — that is a full experiment, not a patch, and the time left
was better spent making the correction propagate cleanly through both deliverables.

**Next step.** Run the quiet-versus-loud contrast on a ladder of displacement norms (0.1 to 1.0),
rebuilding the two combinations at each norm, and plot w_hat against norm for both. The separation
point (if any) decides whether width is a function of the behaviour the embedding induces or of the
embedding's exact location — and therefore whether the free vocabulary-wide lookup is reading a
behavioural property or a geometric accident, which is the caveat that matters for using it as an
auditing table.

On track? yes — S1-S5 complete, eleven experiments run and reported, one earlier claim withdrawn and
corrected in place, deliverables current-best and render-checked, no blocker.

## 2026-08-11 — iteration 4: the displacement-norm ladder (and a reversal)

**Did.** Ran the experiment the previous iteration recommended, with one design change:
`experiments/norm_ladder.py` walks four displacement norms (0.15 / 0.4 / 0.9 / 1.8) and rebuilds the
quiet and loud directions at each rung by **measuring** what 24 random directions actually do to the
token's output at that norm and taking the argmin / argmax, instead of extrapolating from the SVD of
linear-regime responses at a step of 0.05. 12 tokens x 4 rungs x 3 directions = 144 anchor-width
re-measurements, ~11 min on the shared GPU. Plot: `plots/ladder.png` (Figure 15), which replaces
`plots/quiet.png` in both deliverables.

**Learned.** The design change was the whole experiment. Direct selection separates quiet from loud by
8x in output movement at norm 1.8 (0.049 vs 0.402 bits) where the SVD construction separated them not
at all (0.181 vs 0.165) — and with a real contrast the previous conclusion reverses. At *identical*
displacement of norm 1.8, the quiet edit keeps the token ordering (rho(before, after) = +0.94,
p = 4e-6) and the loud edit destroys it (+0.08, p = 0.80). The quiet direction also widens less than
the loud one in the paired test at every rung (p = 5e-4 at norm 0.9, p = 0.02 at 0.4). What does NOT
reverse: the level. Every direction, quiet included, raises the mean width (0.543 -> 0.656 quiet,
0.683 loud) and shrinks the spread across tokens (0.083 -> 0.038 / 0.022). So the correct split is
displacement compresses the level, behaviour destroys the ordering — and since the ordering is what the
screen consumes, the vocabulary-wide lookup is reading a behavioural property. That closes the caveat
the previous iteration left open, and in the favourable direction.

**Assumption logged (loop mode).** Chose argmin/argmax over 24 measured directions rather than
rebuilding the SVD construction at each rung, as PLAN.md's wording implied. Reason: the SVD version is
the thing that failed, and its failure mode (linear response not surviving to large steps) is exactly
what a per-rung measurement avoids; direct selection is also cheaper, which paid for the fourth rung.
Cost: "quietest of 24 draws" is a weaker quiet direction than an optimised one, so the result bounds
from below how much a behaviour-preserving edit can keep — reported as a limitation. Rejected: 48
directions with 3 rungs (same cost, less range, and the range is where the effect lives); rejected also
keeping the superseded fixed-displacement table alongside the ladder, since it is the same experiment
done worse (rule 6) — its withdrawal and the reversal are recorded in CHANGELOG.md instead.

**Next step.** Ask which part of the token's output behaviour carries the trait. Take the loud
direction at norm 1.8, decompose the JSD it produces by successor token into the token's high-mass
successors versus the tail, then edit along directions restricted to each subspace at matched total
output movement and re-measure w_hat. Top-mass-driven would tie the per-token trait back to corpus
successor JSD, the statistic this direction started from, and suggest a corpus-side estimator;
tail-driven would say the embedding probe sees something corpus statistics cannot.

On track? yes — S1-S5 complete, twelve experiments run and reported, one earlier conclusion reversed
with a stronger instrument and the reversal recorded, deliverables current-best and render-checked, no
blocker.

## 2026-08-11 — iteration 5: the mode split

**Did.** Wrote and ran `experiments/mode_split.py`: partition each embedding edit's output change by
successor token and score the share $S$ on the token's top-32 successors, for 24 random directions per
token at the ladder's top rung (norm 1.8), then rescale the most top-heavy and most tail-heavy of them
to a matched 0.4 bits and re-measure anchor width. 12 tokens, ~16 min on the shared GPU. Plot:
`plots/mode_split.png` (Figure 16). Deliverables curated, pattern numbering fixed, render check passes.

**Learned.** Two things, one clean and one bounded. Clean: the damage a large embedding edit does is
**tail-weighted** — the top 32 successors hold 0.707 of the mass but absorb only 0.389 of the
divergence, and louder directions are more tail-weighted still (rho = -0.36). So the behaviour whose
disruption coincides with the trait's collapse is not mainly the behaviour corpus successor JSD scores,
which is mildly favourable for the claim that the embedding lookup carries information corpus counts do
not. Bounded: the steering test is a null with a real limit. Random directions span only S = 0.36-0.56,
never reaching the mass-proportional 0.71, and at matched movement both extremes flatten the ordering
(rho = -0.08 top-heavy, -0.37 tail-heavy, neither significant at n = 12) and widen every token toward
~0.66. The one surviving paired difference — top-heavy widens more (+0.124 vs +0.108, p = 0.009) while
moving the output less — hints that top-mass disturbance inflates the *level* more efficiently, but it
says nothing about the ordering, which is what a screen consumes.

**Assumption logged (loop mode).** Selected the two contrast directions from 24 random draws rather
than constructing them from the unembedding rows of the top successors. Reason: construction needs the
output-logit Jacobian with respect to the embedding row (2048 forward passes per token per frame,
~40x the time available this iteration), and the draw-and-select design had already worked for the
ladder. Cost, now visible in the result: random draws barely vary in S, so the causal half is
underpowered — reported as a limitation and turned into the recommended next experiment rather than
being presented as a settled null. Rejected: fewer tokens with a bigger direction pool (the pool's
spread in S is the binding constraint, not the count of draws, and n = 12 is already small for the
paired test); rejected also running the calibration to 0.2 bits to save time, since 0.4 bits is the
rung at which the ladder showed the ordering die and matching it keeps the two experiments comparable.

**Next step.** Build the two directions instead of drawing them: project the output response onto the
span of the top-32 successors' unembedding rows and onto its complement, or search a modest random
subspace for the combination maximising/minimising S, aiming for S > 0.8 versus S < 0.2 at a matched
0.4 bits, then re-measure w_hat. A top-heavy edit that preserves the ordering where a tail-heavy one
destroys it would place the trait in the tail of the next-token distribution and rule out any
corpus-side estimator built on high-mass successors; both erasing it would say the trait belongs to the
token's whole output map, making the static-embedding lookup the right level of description.

On track? yes — S1-S5 complete, thirteen experiments run and reported, deliverables current-best and
render-checked, no unaddressed feedback, no blocker.

## 2026-08-11 — iteration 6: constructing the directions the mode split could only draw

**Did.** Wrote and ran `experiments/mode_construct.py`. Instead of picking the most top-heavy and most
tail-heavy of 24 random directions, I built them: for small displacements the per-successor JSD is
quadratic in the centred logit response, so within the span of 24 probe directions the top-mass share
$S$ is a Rayleigh quotient $c'Ac/c'Bc$, and the generalised eigenvectors of $(A,B)$ give the extremes in
closed form. 24 probe forwards per token at norm 0.6, then both constructed directions rescaled to
0.4 bits and anchor width re-measured. 12 tokens, ~12 min. Plot: `plots/mode_construct.png`
(Figure 17). Deliverables curated, patterns renumbered, render check passes.

**Learned.** The construction does exactly what it should in the regime it was derived for — predicted
$S$ 0.856 vs 0.179, three times the random-draw spread, the top end past the mass-proportional 0.71 —
and then buys nothing. Rescaled to the 0.4 bits at which width actually responds, both edits land at
$S = 0.369$ / $0.390$: no separation, both inside the random-direction band, paired difference
backwards ($p = 0.09$). Both widen every token to $\hat w_u \approx 0.67$ and both erase the ordering
($\rho = -0.16$, $-0.28$). Two readings, both useful. Mechanistic: the tail-weighting of a large
embedding edit is a property of the *step size*, not of the direction — the linear picture of an
embedding edit expires well before the displacement width responds to, the same expiry that made the
fixed-displacement test misleading in iteration 4. For the direction's question: $S$ cannot be held
apart at a step the model feels, so the tail-vs-top-mass hypothesis is untestable by embedding edits,
and both arms agree that any disturbance the model registers erases the ordering wherever it lands. The
trait is a property of the token's whole output map, which is the reading that makes the
vocabulary-wide static-embedding lookup the right level of description.

**Assumption logged (loop mode).** Used the subspace-search form of the construction (24 probe
directions + generalised eigenproblem) rather than the unembedding-row projection PLAN.md also listed.
Reason: the projection needs the output-logit Jacobian w.r.t. the embedding row (2048 forwards per
token per frame), ~40x the time available; the eigen form is exact for the quadratic approximation
within the probed subspace and its predicted-$S$ eigenvalues double as a check that the method worked
before the step is grown. Cost, now visible: the construction is only valid where the quadratic
approximation is, and that is precisely the regime the calibrated step leaves — reported as the
result, not as a defect. Rejected: probing at a larger norm (would make the eigenvalues meaningless
without making the achieved $S$ any better); rejected also raising the probe subspace to 48 directions
(the binding constraint is nonlinearity at the calibrated step, not the span).

**Next step.** Stop perturbing embeddings. Ablate one attention head or MLP at a time in blocks 0–5,
re-measure $\hat w_u$ against the six anchors for the same 12 tokens, and score each component by how
much of the across-token spread in $\hat w_u$ it destroys. The layer sweep already says the ordering is
fixed at the input and the sharpening comes from the blocks below the site, so a component whose
removal collapses the spread while leaving the output intact would localise the trait for the first
time; a flat profile would close the mechanistic search on a negative and leave the static-embedding
lookup as the deliverable.

On track? yes — S1-S5 complete, fourteen experiments run and reported, the recommended experiment from
the last iteration executed and answered (negatively, with the instrument validated), deliverables
current-best and render-checked, no unaddressed feedback, no blocker.

## 2026-08-11 — iteration 7: out of embedding space, into the computation

**Did.** Wrote and ran `experiments/ablate.py`: mean-ablate one early component at a time (16 attention
heads + the MLP, blocks 0–5 = 102 components) at the final token position, recompute endpoints and the
interpolation bank with the ablation live, re-measure $\hat w_u$ for the 12 intervention tokens against
the 6 anchors in frame 1. Replacement vector = the component's mean final-position output over the 18
endpoint prompts. Scored each component by across-token sd, rank agreement with the unablated ordering,
and how many bits of output movement the ablation costs. 470 s. Plot `plots/ablate.png` (Figure 18);
deliverables curated, render check passes.

**Learned.** The profile is flat with exactly one spike. 101 of 102 components leave the ordering
untouched (median $\rho = +0.99$; every head $\ge +0.97$, every MLP above block 0 $\ge +0.90$). The
block-0 MLP collapses the spread 0.084 -> 0.018, pushes every token to $\hat w_u \approx 0.82$ and
leaves $\rho = -0.10$. Two things follow. The negative half is clean and useful: the trait is not spread
thinly over early attention and is not re-derived layer by layer, which narrows the search from 102
components to one. The positive half is confounded, and I reported it that way — the block-0 MLP moves
the output 0.451 bits, sixty times any other component here and almost exactly the rung at which the
displacement ladder (pattern 15) showed *any* disturbance flattens the ordering. Ablation alone cannot
tell "this component computes the trait" from "this is the only early component big enough to reach the
lethal regime".

**Assumption logged (loop mode).** Ran one frame rather than three and 6 anchors rather than the full
set, to fit 102 components into the time available; the baseline was re-measured under the identical
1-frame protocol so every comparison is internal. Mean-ablated at the final token position only (a
query-position ablation) rather than at all positions — cheaper, and the readout is the final position.
Rejected: ablating whole blocks (too coarse to distinguish head from MLP); rejected restricting to ~40
components as PLAN.md suggested, since the per-component cost turned out to be ~4.6 s and all 102 fit.

**Next step.** Break the confound with a dose–response: blend the block-0 MLP's final-position output
toward its mean with weight $\alpha = 0.1 \dots 1$, and at each $\alpha$ run a control that perturbs
the same residual stream with a random vector rescaled to the same output movement in bits. Plot
$\rho$(before, after) against bits for both arms. Separated curves = the block-0 MLP computes the
trait; coincident curves = disturbance as such kills it, the mechanistic line closes on a negative and
the static-embedding lookup stands as the deliverable.

On track? yes — S1-S5 complete, fifteen experiments run and reported, the recommended experiment from
the last iteration executed and answered (positively, with its confound stated), deliverables
current-best and render-checked, no unaddressed feedback, no blocker.

## 2026-08-11 — iteration 8: the confound breaks in the MLP's favour

**Did.** Wrote and ran `experiments/dose.py` (56 s): blend the block-0 MLP's final-position output
toward its mean at alpha = 0.1 … 1, and at each dose bisection-scale a fixed random direction added to
the same residual stream until the model's output moves the same bits. Re-measured w_hat_u for the same
12 tokens / 6 anchors / 1 frame as the ablation sweep, scored both arms by rho(before, after) and
across-token sd, plotted both against bits (`plots/dose.png`, Figure 19). Curated both deliverables,
render check passes.

**Learned.** The curves separate in the survivable band and coincide in the destroyed one. From 0.007
to 0.103 bits the MLP dose is below its matched random control at every rung (+0.84/+0.99, +0.64/+0.91,
+0.62/+0.79, +0.25/+0.61); crossing rho = 0.6 costs the MLP ~0.03 bits and the control ~0.10, so
~3.5x. Above 0.25 bits both arms are noise (SE(rho) ≈ 0.3 at n = 12) and I reported those two rungs
without interpreting them. The unexpected half is the second panel: the across-token sd collapses along
an identical trajectory in the two arms, so LEVEL compression is a pure disturbance effect (as pattern
15 found for embeddings) while ORDERING is what singles out the component. That distinction retro-fits
every earlier "the edit widens everything" null — those experiments were reading the level channel.

**Assumption logged (loop mode).** One random-control seed and one random direction, not an ensemble:
the effect at the informative rungs is a factor ~3.5 in bits, far larger than the seed-to-seed spread
the norm ladder saw across 24 directions, and time allowed 16 measurements. Matched on bits rather than
on displacement norm, because the ladder already showed bits is the variable that governs damage.
Rejected: matching by norm (would have reproduced the ladder's confound); rejected running three frames
(the ablation baseline this is compared against is 1-frame, so the comparison must be too).

**Next step.** Stop destroying and start reading: fit a ridge probe from the block-0 MLP's
final-position output m_u to the measured anchor width (held-out, shuffled-target control) and
transplant m_u from a narrow token onto a wide token's forward pass. Probe beats the embedding probe's
rho = +0.76 and transplant moves the recipient → the trait is a readable vector; both null → the
block-0 MLP is a necessary stage, not the store, and the static-embedding lookup stands as the
deliverable.

On track? yes — S1-S5 complete, sixteen experiments run and reported, the recommended experiment from
the last iteration executed and answered positively with the confound broken, deliverables current-best
and render-checked, no unaddressed feedback, no blocker.

## 2026-08-11 — iteration 9 (final): finalization, Conclusion repair, STOP

**Feedback check first (CLAUDE.md Part C).** `ls` of the direction root for `human_feedback*.md` and
`*REVIEW*` without the `.addressed.md` suffix: none present. So rule 11's bar for writing `STOP` is
met, and no feedback work was owed this iteration.

**What I did.** ~23 minutes of wall-clock remained at iteration start, which is the finalization band,
so I ran no new experiment. I read PLAN.md, RESULTS.md and the CHANGELOG tail, then audited both
deliverables against the current evidence rather than assuming iteration 8 had left them consistent.

RESULTS.md was already current-best end to end — headline, all thirteen result tables, all 19 embedded
figures with visible numbered captions, and a "Next experiment" section already rewritten to the
probe/transplant ask. I changed nothing in it, which is the right call under rule 6: curating does not
mean rewriting a file that is already correct.

REPORT.md was not consistent. Its Summary and Results carried the iteration 7–8 localisation (component
ablation, then the matched-bits dose–response), but the Conclusion still ended the mechanism discussion
with "we cannot yet say what in the network produces it" — a sentence that was true through iteration 6
and false afterwards. A reader who read only the Conclusion would have taken away the opposite of the
report's strongest finding. I cut that clause, added a Conclusion paragraph carrying the localisation
with its numbers (101/102 components leave the ordering intact, median rho = +0.99; block-0 MLP alone
collapses spread 0.084 → 0.018 and ordering to rho = −0.10; dose–response separates carrier from
loudness at +0.64 vs +0.91 at 0.014 bits, ~3.5x more movement needed by the control), and stated the
level-vs-ordering split plus the probe/transplant follow-up. I also added the matching caveat to
Limitations (12 tokens, one frame, one control seed, both arms at noise above 0.25 bits).

**Verification.** `check_render.py REPORT.md RESULTS.md` passes: REPORT 23 display equations, 522
inline, 19 embeds, 0 problems; RESULTS 319 inline, 19 embeds, 0 problems. That covers rules 8a–8c,
rule 12's embed check, rule 9a's prose-above-table check and rule 9d's contrast budget.

**What I learned this iteration.** The specific failure mode worth recording: when a direction adds
results iteration by iteration, the Summary gets updated (it is where the new finding feels like it
belongs) and the Conclusion silently rots, because nothing forces a diff between them. A stale
Conclusion is worse than a missing one — it actively contradicts the report. Checking Summary against
Conclusion for contradictions should be a standing finalization step, not something noticed by luck.

**Final state.** The direction answers its core question. The leftover after corpus successor JSD is
mostly a per-token additive trait; that trait is measurable from unpaired anchors, transfers to unseen
tokens as a frozen screen (R^2 = 0.397), is largely readable free from the static embedding
(R^2 = 0.213, no forward pass), survives changes of token class and context shape in its ordering
though not its level, is behavioural rather than positional, and is realised in the block-0 MLP.
Four mechanisms were tested and refuted (basin radius, path-length normalisation, the probe direction
as a width lever, top-mass steering), which is recorded as evidence rather than dropped.

**Next step if the direction were reopened.** The single experiment named in both deliverables: fit a
ridge probe from the block-0 MLP's final-position output m_u to the measured anchor width with the
embedding probe's held-out protocol, and transplant m_u from a narrow token onto a wide token's forward
pass. Probe beating rho = +0.76 plus a transplant that moves the recipient toward the donor would make
the trait a readable vector; both null would mark the block-0 MLP a necessary stage rather than the
store, leaving the static-embedding lookup as the practical deliverable. Cost is ~123 forwards plus
~12 transplants — cheaper than iteration 8.

**On track?** Yes — complete. All stages S1–S5 done, every required output delivered, deliverables
verified rendering-clean and mutually consistent, no unaddressed feedback, `STOP` written.

---

## 2026-08-12 — iteration 10: operator feedback, per-token-matched dose–response

**Context on re-entry.** The direction had written `STOP` at iteration 9; the operator removed it and
dropped `human_feedback.txt` into the folder. Per CLAUDE.md Part C that file is the whole iteration.

**The feedback.** The dose–response's random control was matched to the block-0 MLP dose only on the
*mean* output JSD over the 12 tokens, while the conclusion is about the *ordering* of those tokens —
so per-token perturbation magnitudes could be badly mismatched, and the control could be too weak on
exactly the tokens that carry the ranking. Match or normalise per token before concluding.

**What I did.** Wrote `experiments/dose2.py`. For each dose alpha, one batched forward gives the MLP
arm's output movement B_p for each of the 18 endpoint prompts (12 tokens + 6 anchors); a vectorised
bisection then finds a separate scale c_p per prompt so the random direction moves *that prompt's*
output by exactly B_p bits; widths are re-measured with each prompt carrying its own c_p. Three seeds
instead of one. The old mean-matched control is rerun alongside solely as a diagnostic. Added a paired
per-token Wilcoxon on |Δŵ_u| and on the level-free |Δŵ_u − mean Δŵ|, since a Spearman rho over 12
tokens has SE ≈ 0.3 and cannot carry this claim on its own. Runtime ~3 min on the shared GPU.

**Implementation note worth keeping.** The perturbation only enters through the endpoint states and
endpoint logits: during the interpolation the block-0 Patcher overwrites the final-position hidden
state, so the MLP hook is inert inside `run_pair`. That is why a per-prompt scale is well defined at
all — each endpoint forward is a separate call, so the hook can carry a different c per prompt — and
why per-token matching costs almost nothing (only the 6 anchor endpoints must be recomputed per token).

**What I found — the feedback was correct and material.** The mean-matched control was giving
individual tokens 0.08× to 8.5× the movement the dose gave them, because the dose is itself uneven
(per-token movement spans 0.254–0.710 bits at full ablation). Fixing it:
- the control's rho at the 0.103-bit rung falls from +0.61 (at an actual 0.078 bits) to +0.15 — below
  the MLP arm, i.e. the old figure's most eye-catching separation was an under-dosed control;
- the headline margin drops from "~3.5× more output movement needed by the control" to **1.3×**;
- the claim's band narrows from 0.007–0.103 bits to **below 0.03 bits**; above 0.1 bits the arms cross;
- the localisation itself survives: MLP below its matched control in **15/15** rung × seed comparisons
  in the live band, and the paired per-token test gives ~2× more width movement under the dose than
  under each token's exactly matched control (p = 0.0010 at 0.0068 bits), still significant after each
  arm's mean level shift is subtracted (p = 0.034 / 0.016 at the two live doses);
- one earlier claim is retracted: the across-token spread does *not* collapse identically in both arms
  above 0.014 bits — the dose compresses harder (0.027 vs 0.055 at 0.103 bits).

**What I learned.** Matching an intervention against a control on an *aggregate* of the very units the
conclusion ranks is not a control at all: the aggregate can be satisfied while the per-unit doses are
off by two orders of magnitude, and the mismatch flatters whichever arm is measured on the units it
under-doses. The general rule for this project: match the control on the same index the claim is
indexed by. Second, when the headline statistic is a correlation over n = 12, a paired per-unit test
should be reported next to it — here it is the statistic that survived the re-matching, and it also
made it possible to separate the level shift from the reordering (subtract each arm's mean change).

**Assumptions logged.** (a) Anchors are perturbed with their own matched scale c_p rather than the
token's, so every prompt in the measurement is matched to its own MLP-arm movement — the strictest
reading of the feedback. (b) The rho = 0.6 crossing is read by log-linear interpolation between rungs,
the same rule for both arms. (c) Rejected alternative: normalising rho by per-token movement post hoc
instead of re-running matched — rejected because the mismatch is up to 8.5× and no post-hoc
normalisation of a rank statistic can undo an under-dosed control.

**Next step.** Unchanged and still the recommended experiment in both deliverables: ridge-probe the
block-0 MLP's final-position output m_u for the measured anchor width (held-out, shuffled-target
control) over the 123 endpoint tokens, and transplant m_u from a narrow token onto a wide token's
forward pass. Given that the localisation's margin is now known to be modest, the probe half matters
more than before: a probe beating the embedding probe's rho = +0.76 would establish the component's
role positively rather than by damage.

**On track?** Yes — the operator's point was addressed in full, the affected numbers were corrected
downward in both deliverables rather than defended, both files pass `check_render.py`, and the
direction's conclusion survives in weakened, honestly stated form.

### 2026-08-12 — iteration 10, part 2: the recommended follow-up (block-0 MLP probe + transplant)

The feedback was fully addressed with ~2.5 h of budget left, so I ran the experiment both deliverables
had named as the single most informative next step rather than writing STOP on a direction that had a
concrete, cheap question outstanding.

**What I ran.** `experiments/mlp_read.py`: (A) ridge probes to the measured anchor width from the
block-0 MLP final-position output m_u, the static embedding row, and the full post-block-0 state, all
on the embedding probe's protocol; (B) transplants — every ordered pair of the 12 tokens, overwriting
the recipient's m_u with the donor's. Plus `experiments/mlp_geom.py` to size the intervention.

**Two bugs caught before they became results.**
1. The first transplant run applied the write to the ANCHOR forwards too, so every transplant — self
   included — pushed all widths to ~0.81 and the effect looked like a null (per-recipient rho +0.32,
   slope +0.036). The self-transplant sanity check is what exposed it: it must return the baseline
   exactly, and it did not. Fixed by computing the anchor endpoints once, unedited. Lesson worth
   keeping: build the identity case into every intervention experiment; it is the cheapest possible
   test that the harness edits only what it claims to.
2. The probe was going to take ~70 min because `embed_probe.probe` solves a 2049x2049 system per ridge
   strength (0.16 s each, 3300 per probe, 8 probes) — over the script's own timeout. Replaced with the
   dual form (X'X + lam I)^-1 X' = X'(XX' + lam I)^-1, an 80x80 solve, with a run-time assertion that
   it matches the primal on the same splits (it does, to 1e-12).

**Result — the transplant is the direction's cleanest positive.** Width follows the donor: per-recipient
rho = +0.968 (min +0.95, Wilcoxon p = 5e-4), slope +0.913 on the donor's own width; the recipient's
remaining state contributes nothing (per-donor rho = −0.104, p = 0.64, between-donor variance 66x
between-recipient). Self-transplant reproduces the baseline to 4 dp. One vector carries the trait.

**And m_u is context-free.** Its cosine across the three sentence frames is 1.0000 — exact, because
Pythia is parallel-residual, so block 0's MLP reads ln(W_E[u]) before attention writes anything into
the stream. This retroactively explains three earlier results: why a per-token width exists at all, why
it survives a change of frame (only the level moves), and why a static-embedding lookup works.

**The probe half is a null, and I reported it as one.** rho = +0.748 ± 0.049 from m_u against +0.764
from the embedding row and +0.772 from the full post-block-0 state — all within 1 sd. The first MLP
transports the trait without making it more linearly explicit, which is consistent with the earlier
failure of edits along the probe direction to steer width.

**Honesty check I ran on myself.** A transplant that swaps 79% of the state's norm and 76% of its
across-token spread could make "m_u carries the width" near-tautological, so `mlp_geom.py` measured
exactly that, and both deliverables now state it: the hybrid state sits ~0.75 of the way to the donor
and the edit costs a median 0.738 bits. What rescues the claim from tautology is the control — the
untouched quarter of the state carries zero width information (rho = −0.104), and the transfer slope
(0.913) exceeds the state-displacement fraction (0.75).

**Next step.** How compressible is m_u? Project the donor–recipient difference onto the top k principal
components of m across the 123 tokens, transplant only that projection, sweep k. A handful of
directions reproducing slope +0.913 would turn the trait into a low-dimensional feature an auditor
could monitor or edit — the first thing in this direction a steering method could act on. Cost is ~1
minute of GPU per k.

**On track?** Yes — feedback addressed in full and the named follow-up delivered, with its positive
half (transplant), its null half (probe) and its scale caveat all in the deliverables; both files pass
check_render.py at 20 figures.

### 2026-08-12 — iteration 10, part 3: the rank sweep (how compressible is m_u?)

Ran the experiment part 2 had just named, since it costs ~1 min of GPU per value of k.

**What I ran.** `experiments/mlp_rank.py`: transplant only P_k(m_d − m_r), the projection of the
donor–recipient difference onto the top k principal components of m across the 123 endpoint tokens, for
k = 1 … 122; controls = the bottom k components and a random k-dimensional subspace.

**Result — a clean negative, and a more interesting one than a positive would have been.** The top 64
components carry 79% of the across-token variance of m and buy only 30% of the transfer (slope +0.274
vs +0.913 for the intact vector), while already causing 95% of the full transplant's output movement.
The tail they discard transfers nothing (−0.022) and barely moves the model (0.016 bits). Random
subspaces do nothing. Top-64 + bottom-58 would give +0.25 if the effects added; together they give
+0.913. So the trait needs the exact vector.

**The panel I did not expect.** Partial transplants behave exactly like every disturbance in this
report — mean w_hat rises from 0.565 toward 0.65, spread compresses — while the COMPLETE transplant
returns mean 0.573, sd 0.076, i.e. the unedited distribution with the tokens' widths swapped. Truncation
keeps all the damage and loses the transfer. That is a much sharper way to say "this is not a
low-dimensional feature" than the slope curve alone.

**What I learned.** Two things worth carrying. (a) When an intervention transfers a trait, sweep its
rank before calling the trait a feature — the transfer slope and the disruption it causes come apart,
and only their combination distinguishes "carries the trait" from "breaks the model". (b) The
off-manifold caveat is real and I stated it in both deliverables: a truncated m is a vector no token
produces, so this bounds the code's compressibility only in so far as the model's response to
off-manifold states is informative.

**Next step.** Leave this model. The direction's practical deliverable is the free static-embedding
screen, and its untested assumption is that the width trait belongs to the token's representation
rather than to this network. Repeat the cheap end of the pipeline on Pythia 410M or 2.8B (~60 tokens'
anchor widths, the embedding probe, the block-0 MLP ablation) and compare three numbers: the probe's
held-out rho (+0.76 here), the cross-model rank agreement of measured widths, and whether the block-0
MLP is again the single early carrier. ~20 min of GPU.

**On track?** Yes — the operator's feedback was addressed in full, and the two experiments the
deliverables named as "next" were both run and written up in the same iteration, one positive
(transplant transfers the trait) and one negative (no low-dimensional part of it does). Both files pass
check_render.py at 21 figures. No STOP written: the direction has a concrete, cheap next experiment and
budget remains.

### 2026-08-12 — iteration 11: the second model, times three

No unaddressed feedback (`human_feedback.addressed.md` only), so this iteration ran the experiment both
deliverables named: repeat the cheap end of the pipeline on another Pythia size.

**Scope decision.** The plan said "410M or 2.8B". 2.8B in fp32 is 11 GB against my 7.2 GB share, so it
was out without switching precision (rejected: changing dtype would confound a cross-model comparison
with a numerics change). 410M was already in the HF cache; 160M and 1B downloaded in seconds, and the
whole per-model run costs 20–130 s, so I ran **three** models instead of one. That turned a
yes/no replication into a size trend, which is what produced the interesting part.

**What I ran.** `second_model.py` (anchor widths for the same 123 tokens × 6 anchors × 3 frames at
block 0; embedding probe refitted inside the model; mean-ablation of every MLP and whole attention
block in blocks 0–5), `second_ctrl.py` (dose2's per-token movement-matched control, code reused
unchanged, on 410M), `second_analysis.py` (split-half reliability, disattenuated agreement, transfer of
the 1.4B lookup), `plot_second.py`.

**Result 1 — the strongest generalisation result in the direction.** 410M, 1B and 1.4B rank the 123
tokens at rho +0.88 to +0.90, and at +0.98 to +1.00 after dividing by each model's own split-half
reliability. Three networks differing in depth and width contain the same ranking, to the limit of what
six anchors resolve. The level is the network's (median w 0.749 → 0.658 → 0.620 → 0.549 with size:
transitions sharpen as models grow) — the same level/ordering split the frame-shape control found for
context. The 1.4B embedding lookup predicts 410M's and 1B's measured widths (+0.760, +0.745) as well as
it predicts 1.4B's own (+0.765).

**Result 2 — a floor, and it was worth the extra 20 s of GPU.** 160M does NOT have the trait: rho
+0.207 against a ceiling of 0.806, refitted probe +0.233 ± 0.104 (R² = −0.02), lookup transfer +0.043.
Had I run only 410M I would have reported "the trait generalises" and missed that it is *acquired*
between 160M and 410M — which is what makes the next experiment (checkpoints) the obvious one.

**Method note worth keeping: report the noise ceiling with every cross-model correlation.** 160M's raw
+0.21 is ambiguous on its own (weak trait, or noisy measurement?). Splitting the six anchors into halves
and Spearman–Brown correcting settles it in one line — and it also converts 410M/1B/1.4B's "+0.89, quite
high" into "+0.99, i.e. identical", which is a much stronger and more accurate statement than the raw
number supports.

**Result 3 — half the dose–response replicates, and I reported the failing half plainly.** The 410M
rerun reproduces the raw per-token effect (dose moves widths ~2× its matched control, p ≤ 0.012 at the
low rungs) and the harder spread compression, but the level-free paired test is null at all nine rungs
and the MLP arm is below its matched control in exactly 9/18 rung × seed comparisons in the live band,
with the rho = 0.6 crossing ratio running backwards (0.66× vs 1.3× at 1.4B). I first ran a 5-dose grid,
noticed only two rungs landed in the live band (< 0.05 bits) where 1.4B had five, and reran with nine
doses so the comparison was fair before drawing a conclusion; the finer grid did not rescue it. Both
deliverables now say the *site* replicates and the *per-bit specificity* does not, and point at the
transplant as the component's durable evidence.

**Assumptions logged.** (a) Same token ids across models — asserted in code via the tokenizer, since all
Pythia sizes share one. (b) 6 anchors and 3 frames kept identical rather than re-selected per model, so
that any disagreement is about the models and not the measuring stick. (c) Ablation at block rather than
head resolution in the new models (12 components instead of 102) to stay inside budget; the fine sweep
stays a 1.4B-only result. (d) Reliability estimated by anchor split-half, not frame split-half, because
frames were already shown to shift the level.

**Next step.** Where does the trait come from? It is learned (absent at 160M) and shared across shapes,
so measure anchor widths for the 123 tokens in Pythia-410M at step1000 / step8000 / step32000 /
step143000 and correlate each checkpoint's ranking with the final one and with the token's unigram
frequency and successor entropy (both already in dir18's manifest). Early-and-sharpening ⇒ the lookup is
a corpus statistic an auditor could compute with no model at all; late-and-gradual ⇒ it reads what the
network learned. ~15 min of GPU.

**On track?** Yes — the named next experiment was run and widened to three models, it produced the
direction's strongest generalisation result plus an honest partial-replication failure, both
deliverables are curated to current-best with Figures 22–24 embedded, and `check_render.py` passes. No
STOP written: budget remains and there is a concrete cheap next experiment.

### 2026-08-12 — iteration 12: seventeen checkpoints, and a two-stage answer

No unaddressed feedback (`human_feedback.addressed.md` only), so this iteration ran the experiment both
deliverables named: when during training does the per-token width trait appear, and is it a corpus
statistic?

**What I ran.** `checkpoints.py` (anchor widths for the same 123 tokens × 6 anchors × 3 frames at
block 0, plus an embedding probe refitted inside the checkpoint) on `pythia-410m-deduped`, and
`checkpoints_analysis.py` / `plot_checkpoints.py`. The plan named four checkpoints; each run costs ~50 s
including the download, so I ran the four, saw the ordering was already near-final at the earliest of
them (`step1000`, +0.80), and added `step0/512/2000/4000/16000/64000` and then a dense early grid
(`step2/8/16/32/64/128/256`) — 17 in total, ~8 min of GPU. That is what turned a one-line answer into
the two-stage result.

**Result 1 — the ordering is fixed in the first 512 of 143,000 steps.** Nothing at initialisation
(across-token sd 0.003 vs 0.060 trained; rho +0.015 with the final ranking; reliability 0.570, i.e. the
measurement is pure noise there), nothing at `step16`, then +0.17/+0.29/+0.44/+0.66/+0.79 at
`step32/64/128/256/512` (+0.87 disattenuated), +0.94 by `step2000`, flat for the remaining 98.6% of
training. Meanwhile the LEVEL keeps falling until `step64000` (median 0.833 → 0.595). The
ordering/level split this direction found for context (frames) and for scale (model sizes) also holds
for training time. Three independent instances of the same dissociation is now the report's most
repeated structural fact.

**Result 2 — the two-stage answer, which neither hypothesis predicted.** The experiment was designed to
separate "early and sharpening ⇒ corpus statistic" from "late and gradual ⇒ learned". The truth is
early AND mostly not a corpus statistic: up to `step128` the ranking is *entirely* unigram frequency
(rho = −0.72 with log10 count, stronger than the finished model's −0.53, and zero agreement with the
final ranking once frequency and successor entropy are partialled out), and from `step256` a second
component appears that those statistics do not contain (partial agreement +0.45 → +0.60 → +0.75 →
+0.80). In the finished model the two corpus statistics explain only 0.375 of the ranking's rank
variance. Lesson worth keeping: the dense early grid is what made this visible — with only the four
planned checkpoints, stage one is entirely below the first measurement and the answer reads as a flat
"in place by step1000, partial +0.65", which would have been a materially weaker and slightly wrong
conclusion.

**Result 3 — an asymmetry I did not expect.** The fixed lookup read off 1.4B's embedding matrix ranks
`step128`'s measured widths at +0.54 and `step512`'s at +0.81, while a probe refitted inside those same
checkpoints is at its shuffled-control level (+0.03, +0.25). A mature model's table detects the trait
in a young model several hundred steps before that model's own embeddings encode it linearly.

**An inconsistency I found and fixed in the deliverables.** The Summary has always said the trait "is
not a simple corpus statistic: frequency rho = −0.33", which is the *fitted* token effect a_u against
log-frequency. The *measured* anchor width tracks log-frequency at −0.52 (1.4B) / −0.53 (410M). Both are
true and they are different quantities, but printed side by side without that note they read as a
contradiction, so I stated the distinction in pattern 30 and added the measured rows to RESULTS.md's
supporting-quantities table.

**Sanity checks that passed.** This sweep's `step143000` reproduces iteration 11's separate 410M run at
rho = +1.0000 over 123 tokens (independent process, independent model load) — the cheapest possible
check that the harness is measuring the same thing. Validity of the width curves is 1.000 at every
checkpoint including `step0`, so the near-zero spread at initialisation is a real absence of variation,
not a mass of failed fits.

**Assumptions logged.** (a) 410M rather than 1.4B for the sweep, because the cross-model section
established that 410M and 1.4B rank tokens identically and 410M is 3.5× cheaper per checkpoint;
rejected running 1.4B checkpoints, which would have bought ~4 checkpoints instead of 17. (b) Corpus
statistics are dir18's sampled counts, not exact Pile statistics — stated as a limitation, since it
bounds what *these* statistics explain. (c) Reliability again estimated by anchor split-half, for
comparability with the cross-model section.

**Next step.** The last untested generalisation: a different tokenizer and training corpus. Measure
anchor widths in `gpt2` for the token strings that are single tokens in both vocabularies, same three
frames, six anchors chosen the same way, and compare the ranking with Pythia's against each model's
split-half reliability. `gpt2` is cached; ~10 min GPU. If positive, follow with a cross-checkpoint
transplant (write the final checkpoint's m_u into the `step128` model and ask whether the ordering
appears), which would tie the transplant result to the emergence result.

**On track?** Yes — the named next experiment was run, widened from 4 checkpoints to 17, and returned a
result that changed the answer rather than confirming a guess; both deliverables are curated to
current-best with Figures 25–26 embedded and `check_render.py` passes. No STOP written: budget remains
and there is a concrete cheap next experiment.

### 2026-08-12 — iteration 13: GPT-2, and the first result that takes a claim back

No unaddressed feedback (`human_feedback.addressed.md` only), so this iteration ran the experiment both
deliverables named: does the per-token width ordering survive a different tokenizer and training corpus?

**What I ran.** All 123 endpoint strings and all 6 anchor strings turn out to be single tokens in
GPT-2's vocabulary, so the transfer needed no substitution at all — same strings, anchors, frames,
block-0 site. `xmodel_width.py` measures widths in `gpt2`, `pythia-410m` and `pythia-1.4b`;
`xmodel_analysis.py` pools them; `gpt2_sites.py` repeats GPT-2 at six depths; `xcurve_examples.py`
saves raw curves; `plot_xmodel.py` makes Figures 27–28. ~5 min of GPU in total.

**The experiment broke on first contact, and the break was the finding.** My first run reported valid
fraction 0.112 and a median width of NaN. The cause is not a bug: 88.8% of GPT-2's block-0 `d(t)`
curves are non-monotone (median backslide 0.107), so dir18's `w` is undefined for them, where every
Pythia curve in this project passes. Rather than loosen the validity criteria (which would have made
the two models incomparable in an unstated way), I defined an **envelope width** on the running maximum
of `d`, which exists for every curve and coincides with `w` exactly on monotone ones, and validated the
substitution inside Pythia before using it: rank correlation with `w` is 1.0000 per curve and per token
in both Pythia models, and re-measuring 1.4B/410M with it reproduces every previously reported number
(median 0.549/0.658, probe +0.764/+0.774, cross-model +0.884). That validation is what makes the
negative interpretable instead of a metric artifact.

**Result — the ordering does not port.** GPT-2 vs Pythia-1.4B $\rho = -0.219$ (ceiling 0.53) against
$+0.884$ between two Pythias; the free lookup transfers at $-0.200$ against $+0.76$; a probe refitted
inside GPT-2 sits on its shuffled control (+0.295 vs +0.275); GPT-2's widths do not even track unigram
frequency ($-0.038$ vs $-0.52$). Partialling out frequency and successor entropy changes nothing.

**The control that makes it a result rather than a shrug.** GPT-2's own reliability is 0.319, so I
could not have concluded much from a small correlation — the ceiling is 0.53. Two things settle it.
First, $|\rho| \le 0.22$ is well inside that ceiling. Second, the site sweep: moving the interpolation
site down to block 8 repairs the curves (validity 0.112 → 0.801, backslide → 0) and raises the level
(0.442 → 0.671) exactly as depth does in Pythia, yet reliability peaks at 0.462 and agreement with
Pythia never exceeds $+0.141$ ($p = 0.12$). The block-0 negative sign does not reproduce anywhere else,
so the honest statement is *no relationship*, not a reversed one.

**What I changed in the story (rule 9b).** The report has claimed since iteration 11 that "the ordering
belongs to the token, the level to the network". That is now qualified everywhere: the ordering belongs
to a token *as trained in a particular corpus*, the screen is per-model, and the split-half reliability
check is the go/no-go test an auditor can run with no reference model (0.89 where the screen works,
0.32 where it does not). The 160M floor is re-read the same way — a fact about that training run, not
about parameter count. I did not delete or soften any Pythia result; the cross-model and checkpoint
sections stand as measured.

**Assumptions logged.** (a) GPT-2 small rather than GPT-2 medium/large — cheapest and the plan named
`gpt2`; a size sweep would confound corpus with scale, and the Pythia scale sweep already covers scale.
(b) Envelope width on the running maximum rather than relaxing MONO_TOL or fitting a sigmoid: it is the
only choice that is exactly `w` on the curves where `w` exists, which the other two are not. (c) The
ablation comparison in GPT-2 uses the same 12 tokens as every intervention here, so it inherits their
n = 12 and is reported as suggestive. (d) Corpus statistics are dir18's Pile-sampled counts, so they
are the *wrong* corpus for GPT-2 — used only to show the mismatch is not a frequency artifact, and the
partial correlation is reported alongside the raw one for that reason.

**Next step.** Separate the two ways GPT-2 could fail, at zero GPU cost: edge drift `E` on the 2,214
curves already stored per model and per site. Straight-line-like `E` in GPT-2 means it has no plateau
to measure and the negative is about plateau structure (and predicts Pythia-160M looks the same);
Pythia-like `E` means genuine plateaus in a different token order, and the next question is what that
order correlates with.

**On track?** Yes — the named next experiment was run, returned a clean negative with the control that
makes it interpretable, and forced a scope correction that both deliverables now carry; RESULTS.md and
REPORT.md are curated to current-best with Figures 27–28 embedded and `check_render.py` passes. No STOP
written: budget remains and the next experiment costs no GPU.
