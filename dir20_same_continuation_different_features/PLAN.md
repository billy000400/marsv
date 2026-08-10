# PLAN - Do these last-token interpolations induce plateaus?

## Question

When two prompts differ only in the final token but have similar next-token distributions, does interpolating that token's activation produce a plateau?

## Success criterion

`RESULTS.md` reports all 4 prompt pairs x 2 models with endpoint JSD, final-logit transition width, plots, and a clear plateau/no-plateau verdict. A null result is complete.

## Setup (fixed)

- Models: final pretrained checkpoints of `gpt2-medium` and `EleutherAI/pythia-410m-deduped` (`revision="step143000"`), both in evaluation mode.
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

## Required outputs

- `plots/final_logit_curves.png`: 4 x 2 grid of raw \(d(\alpha)\) curves, plus \(d=\alpha\).
- `plots/layerwise_widths.png`: transition width versus recording layer.
- `RESULTS.md`: tokenization validity, endpoint top predictions, JSD, `w10-90`, and verdict.

## Fallback

If time is short, complete both models using only block-0-to-final-logits interpolation and omit the layerwise plot.

## Out of scope

Training models, checkpoint sweeps, full-sequence interpolation, training-corpus statistics, and circuit localization.

## Current status

**S1-S3 complete (2026-08-10); the success criterion is met.** All 5 pairs (4 test + 1 control)
validate in both models; endpoint JSD, final-logit `w10-90`, layerwise widths, all three required
plots, and the verdict are in `RESULTS.md` / `REPORT.md`.

**Verdict: plateaus yes, hypothesis no.** 9/10 model-pair cells plateau (`w_TV` <= 0.27 vs 0.5 for a
linear response), but endpoint similarity does not predict sharpness (pooled Spearman
rho = -0.37, p = 0.29, n = 10; sign flips across models and statistics), and the dissimilar-continuation
control plateaus as sharply as the test pairs. `w_TV` and `PF` were added beyond the measurements
above because 4/10 curves are non-monotonic; `w10-90` remains primary.

## Next step

Optional strengthening only — the plan itself is finished. Highest value: mine ~100-200 prompt pairs
programmatically (shared prefix, one differing final token) spanning a wide JSD range and re-run the
same pipeline, replacing Figure 2's 5-point scatter with a powered regression. Secondary: repeat the
sweep with the patch at a middle block to test the depth explanation directly.
