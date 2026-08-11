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
- [ ] S12 (open) - Why is gpt2-large's intervention effect ~10x the other two GPT-2 models', when
  neither model size nor the block-0 share of the selected heads predicts it? Re-run the held-out
  fixed-set ablation with the patch at a relative-depth-matched middle block in all three models.

## Required outputs

- `plots/final_logit_curves.png`: 4 x 2 grid of raw \(d(\alpha)\) curves, plus \(d=\alpha\).
- `plots/layerwise_widths.png`: transition width versus recording layer.
- `RESULTS.md`: tokenization validity, endpoint top predictions, JSD, `w10-90`, and verdict.

## Fallback

If time is short, complete both models using only block-0-to-final-logits interpolation and omit the layerwise plot.

## Out of scope

Training models, checkpoint sweeps, full-sequence interpolation, training-corpus statistics, and circuit localization.

## Current status

**S1-S11 complete (2026-08-11); S12 open. The success criterion is met and exceeded.** All 6
hand-picked pairs validate in all five models, a 200-pair-per-model corpus-mined bank carries the
association test in five models, that bank has been re-run at three patch sites in the 24-block models
and at four/five sites in the depth-mismatched GPT-2 models, a dedicated low-JSD bank (365/399/356
pairs) carries the hypothesis test, that bank has been re-swept under six ablation conditions in all
three GPT-2 models, and a held-out fixed head set has been ablated on 1111 more sweeps — 12581 sweeps
in total, endpoint identity error <= 3.6e-4 throughout.

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

## Next step

S12: explain the cross-model gap. Re-run the held-out fixed-set ablation with the patch at a
relative-depth-matched middle block in all three GPT-2 models — if the gap tracks relative depth
(Experiment 5's organising variable) it should close there. Secondary, still open: pairs differing at
an *earlier* position rather than the final token. Both deliverables are current, pass
`experiments/check_render.py`, and embed all eleven figures with visible captions cited by number.
