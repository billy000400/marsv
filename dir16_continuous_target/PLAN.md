# PLAN â Do continuous MNIST targets reduce activation plateaus?

> Working folder: this one . The agent rewrites **Current status** and **Next step**, ticks completed stages, and records each iteration in `JOURNAL.md`. Disk files are the only memory.

## Question

Do activation plateaus arise partly because classification uses discrete targets? Compare two otherwise matched 4-layer MLPs trained on the same MNIST inputs:

1. **Classifier:** predict the digit label.
2. **Regressor:** reconstruct a clean, downsampled grayscale image.

If the regression model changes more smoothly under the same activation-interpolation probe, continuous supervision may discourage plateau-like representations.

## Success criterion (definition of âdoneâ)

`REPORT.md` gives a clear positive, negative, or inconclusive verdict based on:

- successful training of both models;
- consistent with the 60K images training in /workspace/marsv_agent_haoyang/dir12_plateau_during_training
- the usual endpoint-relative `d(alpha)` plots for the same image pairs and layers;
- an aggregate paired comparison across a fixed, digit-balanced test-pair set;
- example reconstructed images along the interpolation path;
- results from at least 3 random seeds.

A null result is complete if both models are trained adequately and the comparison is reported clearly. When complete, write an empty `STOP` file.

## Fallback (if time runs short)

Use one seed and the existing hand-selected digit transitions, including `6 -> 7`, plus at least 20 fixed random cross-digit pairs. Produce the main `d(alpha)` comparison and a short verdict. The final 20 minutes are reserved for cleaning `REPORT.md` and writing `STOP`.

## Setup (fixed)

### Data

- MNIST train/test split.
- Normalize pixels to `[0, 1]`.
- Use the **same input images and data order** for both models.
- Add one fixed mild Gaussian corruption to the model input, then clip to `[0, 1]`.
- The classifier target is the digit label.
- The regressor target is the **clean image average-pooled to `7 x 7`**, flattened to 49 continuous values.
- Use a fixed corruption seed so both models see identical inputs.

The corruption prevents reconstruction from being a trivial identity map. Do not sweep corruption levels unless the fixed setting makes either task fail.

### Models

Train two 4-layer ReLU MLPs with identical hidden widths and the same initialization seeds, optimizer, batch size, training steps, and weight decay. Reuse the current MNIST plateau model configuration.

- Classifier output: 10 logits; mean-squared-error loss.
- Regressor output: 49 values; mean-squared-error loss.
- No batch normalization or dropout.

Training is adequate when:

- trianing shows slightly overfitting (val loss achieved minimum first, and then increased);
- trianing loss converged to a smooth minimum, not overshooting in the end;
- classifier test accuracy is close to the existing MNIST baseline;
- regressor test MSE beats predicting the average clean target and beats directly pooling the corrupted input;
- reconstructed examples visibly preserve digit shape.

### Activation-interpolation probe

For every fixed test-image pair `(x_a, x_b)`:

1. Record first-hidden-layer activations `h1_a` and `h1_b`.
2. Spherically interpolate between them for 101 evenly spaced `alpha` values, exactly as in the existing plateau experiment.
3. Pass each interpolated activation through the remaining layers.
4. At each later layer, calculate

   `d_l(alpha) = ||h_l(alpha) - h_l(0)|| / ||h_l(1) - h_l(0)||`.

5. Save classifier predictions and regression outputs along the same path.

Use the same image pairs, alpha grid, layers, and plotting code for both models.

### Pair set

- Preserve the existing hand-selected examples for direct comparison with prior results.
- For aggregation, use 2 fixed test-image pairs for each unordered digit pair: `45 x 2 = 90` pairs total.
- Keep this exact pair list fixed across models and seeds.

### Main comparison

For each pair, layer, and seed, calculate the mean deviation from a straight transition:

`linearity_deviation = mean_alpha |d_l(alpha) - alpha|`.

Report the paired difference:

`classifier deviation - regressor deviation`.

Use bootstrap confidence intervals over image pairs. Also show the raw `d(alpha)` curves; the aggregate number must not replace visual inspection.

## Stages (checklist)

- [x] **S1 â Train matched models.** Implement the shared corrupted-MNIST dataset, train both MLPs for 3 seeds, verify task quality, and save checkpoints plus training curves.
- [x] **S2 â Run the matched interpolation probe.** Reuse the existing slerp and `d(alpha)` code, verify it on the hand-selected transitions, and save side-by-side classifier/regressor plots plus regression reconstructions.
- [x] **S3 â Aggregate and report.** Run the fixed 90-pair set, plot layerwise paired differences with confidence intervals, write the verdict and limitations in `REPORT.md`, update `RESULTS.md`, and write `STOP`.

Every reported metric must be defined in `REPORT.md` and have a corresponding saved figure in `plots/`.

## Interpretation

- **Regressor is consistently smoother:** evidence that continuous, information-preserving supervision reduces plateaus.
- **Both models show similar plateaus:** discrete targets are not necessary for plateau formation.
- **Regressor plateaus only in deeper layers:** the output objective may matter, but shared architectural compression may also create plateaus.
- **Results vary strongly by pair or seed:** report the dependence rather than averaging it away.

## Limitations

The endpoints are in-distribution activations, but the interpolated path is not guaranteed to be in-distribution. This experiment tests whether continuous supervision changes plateau behavior under the same interpolation procedure; it does not prove that every intermediate activation lies on the data manifold.

## Out of scope (do NOT)

- Do not add target-quantization sweeps.
- Do not compare additional architectures, optimizers, or datasets.
- Do not introduce new manifold, density, or local-complexity metrics.
- Do not tune many corruption strengths.
- Do not replace the matched classifier/regressor comparison with pixel-sum regression.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> â <stage, % done, blocker if any>`

## Current status

**Complete.** S1-S3 all done in one iteration (2026-07-26). Both models trained and verified on 3
seeds; the frozen 90-pair probe run for both; aggregate paired comparison with bootstrap CIs, the
hand-selected `d(alpha)` curves, reconstructions along the path, and an early-stopped-classifier
control are all in RESULTS.md / REPORT.md with 8 embedded figures.

**Verdict: POSITIVE.** The regressor's representation is 4.4-5.9x closer to a constant-rate
transition than the classifier's (linearity deviation, all bootstrap CIs exclude 0, 90/90 pairs,
3 seeds), and the gap survives matching the classifier to its best-generalizing checkpoint.

Deviation from the setup, logged in JOURNAL.md: under matched training the regressor does not
overfit (its validation loss flattens rather than rising), so the "slight overfitting" adequacy
criterion is met by the classifier only. Matched step count was kept in preference to forcing
regressor overfitting; reported as Limitation 2 in REPORT.md.

## Next step

None — success criterion met and no unaddressed operator feedback, so `STOP` is written. If new
feedback arrives, delete `STOP`, address it, and re-write `STOP` when clean.
