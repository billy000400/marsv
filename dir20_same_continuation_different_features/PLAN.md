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
- [x] S4 (added) - Mine 200 corpus-derived pairs per model spanning the JSD range; re-run the same
  sweep and replace the 5-point scatter with a powered regression.
- [x] S5 (added, optional) - Repeat the mined-bank sweep with the patch at a middle and a late block,
  to test the winner-take-all/depth mechanism directly.

## Required outputs

- `plots/final_logit_curves.png`: 4 x 2 grid of raw \(d(\alpha)\) curves, plus \(d=\alpha\).
- `plots/layerwise_widths.png`: transition width versus recording layer.
- `RESULTS.md`: tokenization validity, endpoint top predictions, JSD, `w10-90`, and verdict.

## Fallback

If time is short, complete both models using only block-0-to-final-logits interpolation and omit the layerwise plot.

## Out of scope

Training models, checkpoint sweeps, full-sequence interpolation, training-corpus statistics, and circuit localization.

## Current status

**S1-S5 all complete (2026-08-10); the success criterion is met and exceeded.** All 5 hand-picked
pairs validate in both models, a 200-pair-per-model corpus-mined bank carries the association test,
and that bank has been re-run at three patch sites.

**Verdict: plateaus yes, hypothesis inverted, shape caused by depth.** Plateaus are the default
response — 82% (gpt2-medium) / 48% (pythia-410m) of arbitrary mined pairs are sharp, and the
dissimilar-continuation control plateaus as hard as the test pairs. At n=200 the association is
significant with the sign *opposite* to the hypothesis: Spearman rho(JSD, `w_TV`) = -0.55 (p=6.2e-17)
in gpt2-medium, and -0.61 / -0.45 (p<1e-7) in the two models once pairs at the ln 2 JSD ceiling are
excluded. More divergent continuations give sharper plateaus. The iteration-1 null (rho=-0.37, p=0.29,
n=10) was underpowered. `w_TV` and `PF` were added beyond the planned measurements because most curves
are non-monotonic (only 7.5% monotonic in gpt2-medium's bank); `w10-90` remains primary and agrees.

**S5 outcome.** Moving the patch from block 0 to 12 to 20 (23 / 11 / 3 blocks below) walks the plateau
away: % of pairs sharp 82 -> 50.5 -> 10 (gpt2-medium) and 47.5 -> 2.5 -> 0 (pythia-410m), the latter
landing on the linear baseline (median w_TV 0.509 vs 0.5). The depth mechanism is therefore causal,
not just correlational. The second half of the prediction failed: the JSD-sharpness correlation does
*not* decay with depth (gpt2-medium -0.61 / -0.53 / -0.53), so depth sets how much the response is
compressed while endpoint divergence sets which pairs compress more.

## Next step

The plan is complete; what remains is optional generalization. Highest value: a third model family
with a different tokenizer and similar depth (e.g. OPT-350m) at block 0 and block 20, to test whether
the 82% vs 48% block-0 prevalence gap is tokenizer or architecture — the one open question the report
names and does not answer. Otherwise finalize: re-read both deliverables for newcomer readability and
confirm every figure is cited by number in the prose.
