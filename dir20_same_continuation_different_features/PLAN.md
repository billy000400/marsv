# PLAN - Do these last-token interpolations induce plateaus?

## Question

When two prompts differ only in the final token but have similar next-token distributions, does interpolating that token's activation produce a plateau?

## Success criterion

`RESULTS.md` reports all 4 prompt pairs x 2 models with endpoint JSD, final-logit transition width, plots, and a clear plateau/no-plateau verdict. A null result is complete.

## Setup (fixed)

- Models: final pretrained checkpoints of `gpt2-medium` and `EleutherAI/pythia-410m-deduped` (`revision="step143000"`), both in evaluation mode. `facebook/opt-350m` was added in S6 as a third family (24 blocks, d_model 1024, GPT-2's tokenizer).
- Prompt pairs:

  1. `Mary and John went to the store. John gave a book to Mary` / `... to her`
  2. `Two plus two is four` / `Two plus two is 4`
  3. `The answer is four` / `The answer is Four`
  4. `Which chemical element does this clue identify? Au` / `... identify? 79`

- First verify that each pair has an identical tokenized prefix and exactly one different, single final token in each model. Mark a pair invalid for a model if this fails; do not perform multi-token interpolation.
- Use 101 evenly spaced interpolation values, fixed seeds, `torch.no_grad()`, and float32 analysis.
- `The house was big` / `The house was in` is **Matthew's positive plateau example**, not a control
  (corrected per `human_feedback.txt`); his smooth comparison `The house was big` / `large` was added
  in S8. `gpt2-large` (36 blocks, Matthew's model) and `gpt2` (12 blocks) were added in S7/S8.

## Measurements

### 1. Endpoint similarity

For each complete prompt, obtain the full-vocabulary next-token distribution during inference. Report Jensen-Shannon divergence (natural-log units) between the two endpoint distributions. Do not use training-set statistics.

### 2. Plateau

At the final token's `resid_post` after block 0, use Matthew's interpolation: SLERP the vector direction and linearly interpolate its L2 norm. Patch each interpolated vector forward through the remaining model.

At every downstream `resid_post` and at the final logits, compute

\[
d(\alpha)=\frac{\|x_\alpha-x_A\|_2}{\|x_\alpha-x_A\|_2+\|x_\alpha-x_B\|_2}.
\]

The primary plateau-strength metric is the final-logit transition width

\[
w_{10\text{-}90}=\alpha(d=0.9)-\alpha(d=0.1).
\]

A linear response has width 0.8; call `w10-90 < 0.5` a clear plateau. Always show the raw curve and flag non-monotonic curves rather than forcing a verdict from the width alone.

## Stages

- [x] S1 - Validate tokenization; compute endpoint predictions and inference JSD.
- [x] S2 - Run interpolation; save final-logit curves and layerwise transition widths.
- [x] S3 - Compare prompt pairs and models; write the verdict in `RESULTS.md`.
- [x] S4 (added) - Mine 200 corpus-derived pairs per model spanning the JSD range; re-run the same
  sweep and replace the 5-point scatter with a powered regression.
- [x] S5 (added, optional) - Repeat the mined-bank sweep with the patch at a middle and a late block,
  to test the winner-take-all/depth mechanism directly.
- [x] S6 (added, optional) - Add a third model family (`facebook/opt-350m`: 24 blocks, d_model 1024,
  GPT-2's exact tokenizer) at all three patch sites, to test whether the cross-model prevalence gap
  tracks the tokenizer.
- [x] S7 (added, optional) - Add depth-mismatched members of the GPT-2 family (`gpt2`, 12 blocks;
  `gpt2-large`, 36 blocks) at patch sites matched either on blocks-below or on fraction-of-stack-below,
  to test whether the S5 depth curve is about absolute block count or relative depth.
- [x] S8 (operator feedback, `human_feedback.txt`) - Reproduce in GPT-2 Large (Matthew's model), add
  his smooth comparison pair `big`/`large`, correct the prevalence counts to the predefined criterion,
  soften the depth claim, and give the advisor's actual hypothesis its first direct test.
- [x] S9 - Sharper hypothesis test: replace the IRD proxy with feature-level measurements (SAE feature
  sets in gpt2-small; attention-head and MLP-neuron sets in three models) and mine specifically for
  low-JSD pairs, taking n from 38 to 365/399/356.
- [x] S10 (added) - Make the circuit-difference result causal: mean-ablate the differentially-engaged
  heads against an engagement-matched control set, at three pre-specified doses, in gpt2-medium and
  gpt2-large.
- [x] S11 - Localise the differential heads: recurrence statistics, a held-out fixed cross-pair set,
  the same set with block 0 excluded, and the Experiment 7 dose sweep extended to gpt2-small.
- [x] S12 - Why is gpt2-large's intervention effect ~10x the other two GPT-2 models', when neither
  model size nor the block-0 share of the selected heads predicts it? Re-run the held-out fixed-set
  ablation with the patch at a relative-depth-matched middle block in all three models, and add the
  block-0 fixed-set run for gpt2-small that S11 skipped.

- [x] S13 - Vary the one design choice the whole report shares: append a shared continuation after the
  differing token so the readout sits 1/2/4 positions downstream, and re-sweep.

- [x] S14 - Turn Experiment 9's two patch sites into a curve: the identical held-out fixed head set
  ablated at five sites in gpt2-large, to locate where the effect (and the plateau) dies.

- [x] S16 - Confound check on S15: re-run the blocks 0-4 unablated sweep on the S4 corpus-mined
  wide-JSD bank, to test whether C(b) is a model property or a property of the low-JSD banks.

- [x] S17 - Extend S16's wide-JSD blocks 0-4 sweep to gpt2-small (its S4 bank already existed), closing
  the three-model comparison and retiring the "no wide bank for Small" limitation.

- [x] S15 - Does the top-of-stack collapse reproduce outside gpt2-large? Blocks 0-4 in gpt2-small and
  gpt2-medium, with the unablated w_TV curve as the primary readout (the ablation delta is under-powered
  in those models).

## Required outputs

- `plots/final_logit_curves.png`: 4 x 2 grid of raw \(d(\alpha)\) curves, plus \(d=\alpha\).
- `plots/layerwise_widths.png`: transition width versus recording layer.
- `RESULTS.md`: tokenization validity, endpoint top predictions, JSD, `w10-90`, and verdict.

## Fallback

If time is short, complete both models using only block-0-to-final-logits interpolation and omit the layerwise plot.

## Out of scope

Training models, checkpoint sweeps, full-sequence interpolation, training-corpus statistics, and circuit localization.

## Current status

**S1-S17 complete (2026-08-11). The success criterion is met and exceeded.** All 6
hand-picked pairs validate in all five models, a 200-pair-per-model corpus-mined bank carries the
association test in five models, that bank has been re-run at three patch sites in the 24-block models
and at four/five sites in the depth-mismatched GPT-2 models, a dedicated low-JSD bank (365/399/356
pairs) carries the hypothesis test, that bank has been re-swept under six ablation conditions in all
three GPT-2 models, and a held-out fixed head set has been ablated at two patch sites per model on 4836
more sweeps, a shared-continuation re-sweep at four readout offsets adds 900, and a five-site patch
curve in gpt2-large adds 1728, and the blocks 0-4 re-run in the two smaller GPT-2 models adds
1800, and the same blocks 0-4 sweep on the wide-JSD bank in all three GPT-2 models adds 900 — 21634
sweeps in total, endpoint identity error <= 3.6e-4 throughout
except in S13, where the manipulation drives the two endpoints to near-coincidence and the bound is
2.1e-3.

**Verdict.** Matthew's contrast reproduces in his own model (gpt2-large) and not in gpt2-medium;
relative depth governs plateau strength; the base rate of plateaus among arbitrary pairs is high
(83.5% in gpt2-large under the predefined w10-90 < 0.5 criterion), so a single sharp curve is weak
evidence; and the advisor's hypothesis is still open after a first, under-powered null test.
`w_TV` and `PF` were added beyond the planned measurements because many curves are non-monotonic;
`w10-90` remains primary and all prevalence counts are quoted under it.

**S5 outcome.** Moving the patch from block 0 to 12 to 20 (23 / 11 / 3 blocks below) walks the plateau
away: % of pairs sharp 82 -> 50.5 -> 10 (gpt2-medium), 61 -> 36.5 -> 1 (opt-350m) and 47.5 -> 2.5 -> 0
(pythia-410m), the last landing on the linear baseline (median w_TV 0.509 vs 0.5). The depth mechanism
is therefore causal, not just correlational. The second half of the prediction failed: the
JSD-sharpness correlation does *not* decay with depth (gpt2-medium -0.61 / -0.53 / -0.53; opt-350m
-0.57 / -0.54 / -0.55), so depth sets how much the response is compressed while endpoint divergence
sets which pairs compress more.

**S6 outcome.** The cross-model prevalence gap is a model property, not a tokenizer or bank-composition
artifact: at matched endpoint divergence gpt2-medium is the sharpest model in all four JSD bins (median
w_TV 2-4x smaller), and opt-350m — which tokenizes identically to gpt2-medium — plateaus 21 points less
often and swaps rank with pythia-410m across the range. Architecture, corpus and pretraining length
remain confounded; that is now the stated limitation.

**S7 outcome.** The depth effect is set by the *fraction* of the stack below the patch, not the number
of blocks. gpt2-large patched at block 12 and gpt2-medium patched at block 0 have the same 23 blocks
below and differ 3.2x in median w_TV (0.255 vs 0.080; 47% vs 82% sharp); matching on f = (N-1-L)/(N-1)
instead halves the mean across-model spread (0.212 -> 0.104). At 11 blocks below, the ordering inverts
the absolute reading entirely (12-block model 0.153, 36-block model 0.444). The inverted JSD
association replicates in both new models (rho = -0.44 and -0.64 at block 0), making it 5/5 models.
Residual confound: width rises with depth inside the GPT-2 family.

**S8 outcome (operator feedback).** Matthew's contrast reproduces in his own model: gpt2-large gives
`big`/`in` w10-90 = 0.044 and `big`/`large` 0.592, a 13-fold gap. It does NOT reproduce in gpt2-medium
(big/in 0.516, failing the predefined criterion), so the earlier report's primary model was wrong for
this question. `big`/`in` is Matthew's positive example, not a negative control — that mislabelling is
removed everywhere. Prevalence corrected to the predefined w10-90 < 0.5 criterion: 11/30 hand-picked
cells, and 83.5% / 73.0% / 60.5% / 47.0% / 30.0% of mined pairs in gpt2-large / -medium / -small /
opt-350m / pythia-410m. Depth is now stated as necessary but not sufficient (big/large stays smooth
with 35 blocks below the patch). The JSD-vs-width correlation is demoted to a descriptive regularity
that does not test the advisor's hypothesis.

**S9 outcome (supersedes the S8 hypothesis test).** With feature-level instruments (SAE feature sets,
attention-head sets and contributions, MLP neuron sets) on banks mined specifically for JSD < 0.1, the
hypothesis splits. The intermediate-plateau reading fails with power: 14/14 instrument-model tests give
rho in [-0.11, +0.12] against IPW, nothing survives Holm correction, rho_min = 0.10 at n ~ 370, and
only 2.0% of gpt2-large's low-JSD pairs pause anywhere in the middle. The endpoint-plateau reading
holds: 14/14 tests against w_TV are negative (to -0.36), and the head-level instruments beat the old
IRD proxy roughly threefold (-0.36 vs -0.13 in gpt2-medium).

**S10 outcome.** The endpoint-plateau association is causal in gpt2-large. Mean-ablating the top-3% of
heads by differential engagement moves median w_TV 0.198 -> 0.358 (+81%); 6% -> 0.441; 10% -> 0.484,
within 3% of the linear response. The engagement-matched control set does nothing (0.198 -> 0.200).
Paired deltas +0.097/+0.145/+0.199, all CIs excluding zero, p ~ 1e-43 to 1e-48, 83-87% of pairs.
gpt2-medium replicates the sign at every dose (+0.009/+0.009/+0.010, p <= 0.019) but ~15x smaller,
even though its correlation was the stronger of the two.

**S11 outcome.** The differential heads are a shared circuit, and most of their causal effect is
upstream of the patch. Per-pair sets overlap across prefixes at J = 0.090 / 0.064 / 0.280 (gpt2-large /
-medium / -small) against a 0.016 random null, and gpt2-large's most-selected head enters 78.9% of
pairs. A single fixed 22-head set ranked on half the prefixes and ablated on the other half moves
gpt2-large's median w_TV 0.198 -> 0.485 (p = 4e-51 vs the matched control), beating the per-pair sets'
0.358 — recovery 198% at 29.4% head overlap; gpt2-medium recovers 70%. But the top heads sit in block 0,
which the patch overwrites, so they act on the interpolated endpoints, not on the computation below:
excluding block 0 leaves 0.198 -> 0.217 (+0.012, p = 5e-24 vs control), 6% of the full effect. Extending
the dose sweep to gpt2-small (+0.014 / +0.019 / +0.025) shows the effect is not ordered by model size
and is not explained by block-0 share (62.6% in gpt2-small vs 16.7% in gpt2-large), so the cross-model
gap stays described rather than attributed.

**S12 outcome.** The head circuit's causal effect is contingent on depth below the patch, and the
relative-depth explanation of the cross-model gap is withdrawn. Repeating the held-out fixed-set
ablation with the patch at each model's middle block (f = 0.455 / 0.478 / 0.486) takes the paired effect
from +0.187 to -0.002 (gpt2-large), +0.015 to +0.003 (gpt2-small) and +0.005 to +0.002 (gpt2-medium),
because the unablated switch is already at the linear response there (median w_TV 0.501 / 0.448 /
0.420). The headroom-normalised effect rules out a pure ceiling artifact (gpt2-large 61.9% -> none;
gpt2-small 8.1% -> 5.0%; gpt2-medium 2.0% -> 2.0%). Relative depth cannot explain the cross-model gap
because all three models were already matched at f = 1 in the block-0 comparison; what is now known is
that gpt2-large's advantage belongs to that patch site, not to the model. gpt2-small's block-0 fixed-set
run (+0.015, p = 1.6e-3) completes the three-model comparison, and recounting the stored sets corrected
gpt2-large's fixed-set block-0 membership from five heads to seven.

**S13 outcome.** The plateau does not depend on the interpolated token being the last token. Appending
the model's own greedy continuation to both prompts (s = 0/1/2/4) moves the readout downstream, where the
patched activation is reachable only through attention, and the median w_TV is unchanged in all three
GPT-2 models: 0.148 -> 0.193 (gpt2-large, paired delta +0.001, p = 0.65), 0.252 -> 0.284 (medium,
p = 0.11), 0.311 -> 0.303 (small, p = 0.60). The same suffix collapses endpoint JSD 15-16x
(0.0499 -> 0.0034 in gpt2-large), so S3's across-pair divergence-sharpness correlation does NOT hold
within a pair: divergence is a marker of feature disjointness, not the driver of sharpness.

**S14 outcome.** The whole phenomenon is built in the top ~4 of gpt2-large's 36 blocks, and half of it
in one block. With the same held-out 22-head set ablated at eight patch sites (blocks 0/1/2/3/4/9/13/18,
f = 1.00 down to 0.49), the paired effect goes +0.250 -> +0.120 -> +0.062 -> +0.057 -> +0.017 -> +0.002
-> +0.003 -> +0.000 and the unablated switch goes 0.189 -> 0.262 -> 0.307 -> 0.350 -> 0.378 -> 0.450 ->
0.479 -> 0.496. Removing a single block from the post-interpolation path, with 34 of 36 still
downstream, halves the head effect; by block 4 it is 93% gone and by block 9 it is at chance (p = 0.34).
So the relative-depth law of S7 holds but is steeply concave — block 1 alone is 24% of the widening
between f = 1 and f = 0.49, blocks 1-4 are 62% — and an interpolation probe is evidence about the few
blocks immediately below the patch, not the network. `hat_Delta` is now reported only where the control
retains >= 0.05 of headroom (blocks 13 and 18 otherwise read 19% and 14% off headrooms of 0.017/0.001).
Caveat logged: the four top-of-stack sites were chosen after seeing the block-4 drop.

**S15 outcome.** The shape generalises, the rate does not. In all three GPT-2 models the unablated switch
widens monotonically as blocks are removed from below the patch and the first block removed is the
largest single step, but the share of each model's own block-0 headroom closed after four blocks is
60.7% (gpt2-large), 51.1% (gpt2-small) and only 18.6% (gpt2-medium). "Four blocks build the plateau" is
therefore a gpt2-large statement; the general claim is that the top blocks matter most by a model-specific
amount. The fixed-set ablation delta was a null at 9 of 10 model-sites as predicted (only gpt2-medium
block 0, +0.011, CI [+0.004, +0.017]), which is why the unablated curve carries the claim.

**S16 outcome.** S15's "rate is model-specific" claim is narrowed: it holds among low-JSD pairs only.
On the wide-JSD corpus-mined bank the two models close the same share of headroom after four blocks
(gpt2-medium 17.7%, gpt2-large 16.9%, against 18.6% and 60.7% on the low-JSD banks), and it is
gpt2-large that moves, not gpt2-medium. High-divergence pairs start much sharper (median w_TV 0.042 /
0.094 at block 0), so they carry more headroom and give up a smaller share per block. The direction
(monotone widening) and the front-loading (first block removed is the largest step) hold on both banks
in both models, so those carry the mechanism claim.

**S17 outcome.** Extending the wide-JSD blocks 0-4 sweep to gpt2-small (its S4 bank already existed;
the "no wide bank for Small" limitation was false) closes the three-model comparison and strengthens
S16: wide-bank C(4) = 16.9% / 17.7% / 24.4% (large / medium / small) against 60.7% / 18.6% / 51.1% on
the low-JSD banks, so the cross-model spread collapses from 42 points to 7.5. Two models move a long
way with the pair population and gpt2-medium does not, making C(b) a joint model-by-population
quantity. Front-loading is now three models on two banks, always monotone and always largest at b = 1
(gpt2-small on the wide bank: 15.1 of its 24.4 points from the first block alone).

## Next step

The untouched design question is pairs that differ at an *earlier*
position rather than the final token (S13 moved the readout downstream but kept the differing token
last). The longer-suffix extension of S13 stays blocked on conditioning,
not GPU: past s ~ 4 the endpoints are too close for d(alpha) to be well defined and it needs a different
readout (e.g. KL to each endpoint distribution). A smaller open item from S17: the residual 7-point
wide-bank C(4) gap (small 24.4% vs 16.9-17.7%) is unresolved at 60 pairs and would need the full
200-pair banks. Both deliverables are current, pass `experiments/check_render.py`, and embed all
sixteen figures with visible captions cited by number.
