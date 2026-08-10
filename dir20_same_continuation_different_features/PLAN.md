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
- Use `The house was big` / `The house was in` as an implementation control.

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

## Required outputs

- `plots/final_logit_curves.png`: 4 x 2 grid of raw \(d(\alpha)\) curves, plus \(d=\alpha\).
- `plots/layerwise_widths.png`: transition width versus recording layer.
- `RESULTS.md`: tokenization validity, endpoint top predictions, JSD, `w10-90`, and verdict.

## Fallback

If time is short, complete both models using only block-0-to-final-logits interpolation and omit the layerwise plot.

## Out of scope

Training models, checkpoint sweeps, full-sequence interpolation, training-corpus statistics, and circuit localization.

## Current status

**S1-S6 all complete (2026-08-10); the success criterion is met and exceeded.** All 5 hand-picked
pairs validate in all three models (gpt2-medium, pythia-410m, opt-350m), a 200-pair-per-model
corpus-mined bank carries the association test, and that bank has been re-run at three patch sites in
every model — 1815 sweeps, endpoint identity error <= 3.5e-4 throughout.

**Verdict: plateaus yes, hypothesis inverted, shape caused by depth, and all three replicate across
model families.** Plateaus are the default response — 82% (gpt2-medium) / 61% (opt-350m) / 48%
(pythia-410m) of arbitrary mined pairs are sharp, and the dissimilar-continuation control plateaus as
hard as the test pairs (in opt-350m harder than all of them). At n=200 the association is significant
with the sign *opposite* to the hypothesis in every model: below the ln 2 JSD ceiling Spearman
rho(JSD, `w_TV`) = -0.61 / -0.57 / -0.45, all p<1e-7. More divergent continuations give sharper
plateaus. The iteration-1 null (rho=-0.37, p=0.29, n=10) was underpowered. `w_TV` and `PF` were added
beyond the planned measurements because most curves are non-monotonic (only 7.5% monotonic in
gpt2-medium's bank); `w10-90` remains primary and agrees.

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

## Next step

The plan is complete and the last open question it named is answered. Remaining work is optional:
(a) a depth-mismatched model (12- or 36-block) to test whether Figure 6's curve is about absolute block
count or fraction of the stack, and (b) pairs differing at an earlier position rather than the final
token. Both deliverables are current, pass `experiments/check_render.py`, and embed all six figures
with visible captions cited by number.
