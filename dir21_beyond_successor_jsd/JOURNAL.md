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
