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
- [ ] S9 (open) - Sharper hypothesis test: replace the IRD proxy with a feature-level measurement
  (SAE feature sets or path patching) and mine specifically for low-JSD pairs to raise power.

## Required outputs

- `plots/final_logit_curves.png`: 4 x 2 grid of raw \(d(\alpha)\) curves, plus \(d=\alpha\).
- `plots/layerwise_widths.png`: transition width versus recording layer.
- `RESULTS.md`: tokenization validity, endpoint top predictions, JSD, `w10-90`, and verdict.

## Fallback

If time is short, complete both models using only block-0-to-final-logits interpolation and omit the layerwise plot.

## Out of scope

Training models, checkpoint sweeps, full-sequence interpolation, training-corpus statistics, and circuit localization.

## Current status

**S1-S8 complete (2026-08-10); S9 open. The success criterion is met and exceeded.** All 5 hand-picked
pairs validate in all three 24-block models (gpt2-medium, pythia-410m, opt-350m), a 200-pair-per-model
corpus-mined bank carries the association test in five models, that bank has been re-run at three
patch sites in the 24-block models and at four/five sites in the depth-mismatched GPT-2 models
— 3645 sweeps in total, endpoint identity error <= 3.5e-4 throughout.

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

**S8 hypothesis test.** First direct test of "holding output JSD low, different circuits/features may
occupy different plateaus": JSD held < 0.1, IRD (internal representational distance) as IV, IPW
(intermediate-plateau width) as DV. Null in both models — gpt2-large rho = +0.17 (p=0.31, n=38, with
0.0% of pairs showing any intermediate plateau), gpt2-medium rho = -0.00 (p=0.99, n=32). Under-powered
(rho_min = 0.32 at n=38) and proxy-based, so this is a first datapoint, not a refutation.

## Next step

S9: the sharper hypothesis test. Replace IRD with a feature-level measurement of circuit difference
(SAE feature sets on GPT-2 Large residuals, or path patching over attention heads for a subset of
low-JSD pairs) and mine specifically for low-JSD pairs to take n from 38 to a few hundred. Secondary,
still open from before: pairs differing at an *earlier* position rather than the final token. Both
deliverables are current, pass `experiments/check_render.py`, and embed all eight figures with visible
captions cited by number.
