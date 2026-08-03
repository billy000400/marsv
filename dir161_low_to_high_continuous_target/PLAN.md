# PLAN — Do continuous low-to-high-resolution targets reduce activation plateaus?

> Working folder: `dir161_low_to_high_continuous_target`. The agent rewrites **Current status** and
> **Next step**, ticks completed stages, and records every iteration in `JOURNAL.md`. Disk
> (`PLAN.md`, `JOURNAL.md`, `RESULTS.md`, `REPORT.md`, `CHANGELOG.md`, `../BUDGET.md`, and
> `../CLAUDE.md`) is the only memory.

This is a task-changed replication of `../dir16_continuous_target/`, the repository direction that
compared MNIST classification with clean 7x7 reconstruction from a corrupted 28x28 input. Preserve
that direction's matched-model, training, pair-bank, interpolation, statistical, and reporting
protocols except where this plan explicitly changes the prediction task.

## Motivation and research question

The earlier regressor predicted a lower-resolution image whose contents were already present in its
higher-resolution input. Its smooth activations could therefore reflect an information-discarding
reconstruction task, rather than continuous supervision itself.

Here both models receive only a clean 7x7 average-pooled MNIST image. Compare two otherwise matched
4-layer multilayer perceptrons (MLPs):

1. **Classifier:** predict the discrete digit label.
2. **Low-to-high predictor:** predict the original clean 28x28 grayscale image.

The 7x7 input fixes only 49 block means. The 28x28 target also contains within-block spatial detail
that was removed by downsampling. The predictor must use regularities learned from the training set
to predict that missing detail. The primary question is: **does continuous supervision still reduce
activation plateaus when the continuous task predicts high-resolution structure rather than
discarding information?**

This is “information-adding” in the operational prediction sense: the output contains target detail
not explicitly present in the input. A deterministic network cannot recreate information-theoretically
unknowable pixels; under mean-squared error it learns the conditionally predictable component. The
report must preserve that distinction.

## Success criterion (definition of “done”)

`RESULTS.md` and a self-contained `REPORT.md` give a clear positive, negative, reverse, mixed, or
inconclusive verdict based on:

- training both matched models on all 60,000 MNIST training images for 3 fixed seeds;
- showing that the classifier learned the low-resolution classification task and that the predictor
  learned held-out high-resolution detail, not merely a mean image or fixed resize;
- probing every model at its best-validation-loss checkpoint, with the final training checkpoint as
  a preregistered robustness control;
- the usual endpoint-relative `d(alpha)` curves for the same fixed image pairs, alpha grid, and
  equal-width hidden layers in both models;
- a paired aggregate comparison over the frozen, digit-balanced 90-pair test bank, with bootstrap
  confidence intervals and seed-level results;
- high-resolution predictions for held-out inputs and along representative interpolation paths;
- all task-validity, endpoint-fidelity, and reproducibility checks below.

The plateau verdict is valid only if both task-adequacy gates pass. A null or reverse result is
complete when the protocol and controls pass and the result is reported plainly. When all criteria
are satisfied and no unaddressed feedback remains, write an empty `STOP` file.

## Fallback (if time runs short)

Run one matched seed, preserve the fixed preprocessing and task-validity audit, and probe the existing
hand-selected transitions (including `6 -> 7`) plus at least 20 frozen random cross-digit pairs. Show
the main hidden-layer `d(alpha)` curves, paired linearity-deviation comparison, high-resolution path
predictions, missing-detail metric, and fixed-upsample baselines. Do not relax the task-validity gate
or select pairs after viewing results. Reserve the final 20 minutes for current-best figures,
`RESULTS.md`, `REPORT.md`, `CHANGELOG.md`, render checks, the feedback check, and `STOP` if warranted.

## Setup (fixed before viewing interpolation results)

### Data and the low-to-high target

- Use the official MNIST train/test split with clean pixels scaled to `[0, 1]`.
- Let `y` be the clean 28x28 image. Form the only model input `z = D(y)` using non-overlapping 4x4
  average pooling, producing a 7x7 image flattened to 49 values.
- Give **exactly the same `z` tensors, examples, order, and batches** to both models.
- The classifier target is the 10-dimensional one-hot digit label.
- The low-to-high target is the original clean `y`, flattened to 784 values.
- Use all 60,000 training images. Keep test indices `0:2000` as the untouched evaluation and
  interpolation-endpoint pool, and test indices `2000:10000` as the checkpoint-validation set,
  matching direction 16. Save these indices and the preprocessing definition in a frozen manifest.
- Do not add Gaussian corruption in the primary assay. Downsampling is already the intended lossy
  transformation; extra noise would mix super-resolution with denoising.

Define a block-repetition operator `U` that copies each 7x7 value into its corresponding 4x4 block,
so `D(U(z)) = z`. Define `P = I - U D`. The removed-detail target is `r = P(y)`; it has zero mean in
every 4x4 block and satisfies `D(r) = 0`. Before training, test these identities numerically and
record the nonzero held-out energy of `r`. This is the audit that the new target contains a component
not explicitly available in the 49-value input.

### Matched models and training

Train two 4-linear-layer ReLU MLPs with a shared architecture:

- shared trunk: `49 -> 200 -> 200 -> 200`, with ReLU after each hidden linear layer;
- classifier head: `200 -> 10`, trained with per-output mean-squared error against one-hot labels;
- low-to-high head: `200 -> 784`, trained with per-pixel mean-squared error against the clean image;
- no batch normalization, dropout, skip connection, perceptual loss, or auxiliary loss.

For each seed, construct or clone the shared trunk so its initial parameters are bit-identical across
the two models and assert equality before training. Reuse direction 16's frozen optimization protocol:
seeds `0, 1, 2`; AdamW; learning rate `1e-3`; weight decay `0.01`; batch size 200; 30,000 steps
(100 epochs); identical epoch shuffles; and cosine decay to `1e-6`. Losses are averaged per output
unit so their scale does not depend on head dimension. Evaluate and record training/validation
histories every 100 steps; retain the running best-validation checkpoint and the final checkpoint.

The primary probe checkpoint for each model and seed is the checkpoint with minimum validation loss
for that model's own task. Freeze this rule before probing; never choose a checkpoint based on
activation smoothness. Also retain step 30,000 for a complete final-checkpoint control. Do not force
the predictor to overfit if its validation loss reaches a flat minimum.

If the fixed schedule clearly underfits, diagnose it before running any interpolation. At most one
documented global training-schedule change may be applied identically to both tasks and all seeds;
then discard the pilot and rerun the complete frozen protocol. Do not tune the two models separately.

### Task-adequacy gates and baselines

Evaluate task quality on the untouched 2,000-image endpoint pool. The classifier gate is smooth
training convergence and at least 95% top-1 accuracy. For the low-to-high predictor, report all of:

- full-image per-pixel MSE;
- removed-detail MSE between `P(y_hat)` and `P(y)`;
- removed-detail recovery
  `R2_detail = 1 - sum ||P(y_hat)-P(y)||^2 / sum ||P(y)||^2`, where zero means no recovery beyond a
  block-constant image and higher is better;
- low-resolution consistency MSE between `D(y_hat)` and the supplied `z`;
- held-out predictions and residual images shown on a common fixed grayscale scale.

Freeze these evaluation-only baselines before training:

1. the mean 28x28 training image;
2. exact 4x4 block repetition `U(z)`;
3. 7x7-to-28x28 bicubic interpolation using PyTorch `interpolate`,
   `mode="bicubic"`, `align_corners=False`, clipped to `[0, 1]`;
4. a privileged digit-template diagnostic `U(z) + mean_train[P(y) | digit]`, using the true test
   label and training data only. This preserves the observed block means while adding the average
   missing detail for that digit.

The predictor passes its primary gate only if it beats both block repetition and bicubic interpolation
in full-image MSE and removed-detail MSE with paired 95% bootstrap intervals excluding zero, and the
lower 95% bootstrap bound of `R2_detail` is above zero. The digit-template result is a qualification,
not a pass/fail gate: beating it supports instance/style-specific detail prediction; failing to beat
it means the learned detail is only prototype- or conditional-mean-level and must be described that
way. Beating a resize baseline does not justify saying that unknowable sample-specific information
was recovered.

### Activation-interpolation probe

Reuse direction 16's probe without task-dependent tuning. For every fixed endpoint pair `(x_a, x_b)`
and every model/seed:

1. Record post-ReLU first-hidden-layer activations `h1_a` and `h1_b`.
2. Apply the same norm-rescaled spherical interpolation (SLERP) at 101 evenly spaced `alpha` values,
   including both endpoints.
3. Pass every interpolated `h1(alpha)` through the remaining layers.
4. Record the equal-width post-ReLU activations `h2(alpha)` and `h3(alpha)`, plus the task-specific
   raw output.
5. Verify that alpha 0 and 1 reproduce the unpatched endpoint activations and outputs within a stated
   numerical tolerance, and rerun a subset to verify determinism.

At each equal-width hidden layer `l`, calculate the inherited endpoint-relative path coordinate:

`d_l(alpha) = ||h_l(alpha) - h_l(0)|| / ||h_l(1) - h_l(0)||`.

Use identical endpoint images, alpha values, code paths, layer definitions, batching, and tolerances
for both tasks. Hidden layers 2 and 3 are the primary comparison because their dimensions and
architectures match. The 10-dimensional classifier output and 784-dimensional image output may be
shown as task-specific descriptions, but output-space geometry must not determine the cross-task
verdict.

For the low-to-high model, save 28x28 outputs at all alpha values for numeric analysis and render a
fixed subset (for example every tenth point) along each selected path. Show both the full prediction
and its removed-detail component `P(y_hat(alpha))`.

### Frozen pair bank

- Reuse direction 16's exact 90 endpoint pairs from the first 2,000 test images: two replicas for
  each of the 45 unordered digit pairs.
- Preserve the existing hand-selected examples, including `6 -> 7`, for direct visual comparison.
- Save pair IDs, source indices, endpoint labels, replicas, and a manifest checksum before probing.
- Keep the pair list identical across models, checkpoints, and seeds. Never remove a difficult pair
  or promote a pair into the aggregate after seeing its curve.

### Primary plateau metrics and decision rule

For each pair, seed, and hidden layer calculate:

- **linearity deviation (primary):**
  `LD_l = mean_alpha |d_l(alpha) - alpha|`; lower means a more constant-rate transition;
- **maximum normalized jump (secondary):**
  `MJ_l = 100 * max_i |d_l(alpha_(i+1)) - d_l(alpha_i)|`; 1 is the step size of a perfectly
  constant-rate 101-point path and larger values indicate a sharper lurch;
- direction 16's alternative endpoint-fraction normalization as a robustness check, not a replacement
  for the primary `d_l` definition.

The paired effect is always `classifier metric - low-to-high metric` on the same pair and seed. For
each layer, average the paired effect over the 3 seeds, then form a 95% percentile bootstrap interval
from 10,000 resamples of the 90 pair IDs. Also report every seed separately, per-pair scatter, and raw
mean `d(alpha)` curves; the aggregate scalar must not replace curve inspection.

Preregister the verdict:

- **robust positive:** both task gates pass; the 95% interval for classifier-minus-predictor `LD` is
  above zero at both hidden layers 2 and 3; and the effect is positive at every seed;
- **no evidence / negative:** both tasks are adequate but the primary intervals include zero at both
  hidden layers, without a consistent reverse effect;
- **reverse:** both primary intervals are below zero and every seed favors the classifier;
- **mixed:** layer, seed, `LD`, or jump evidence materially disagrees; report the disagreement rather
  than selecting the favorable view;
- **inconclusive:** either task gate, endpoint reproduction, deterministic rerun, or frozen-protocol
  check fails.

`MJ` and the alternative normalization must be reported as robustness evidence. A positive `LD`
verdict with a strong opposing jump result is **mixed**, not robust positive.

## Required artifacts and figures

- frozen data/preprocessing/pair manifest with seeds, indices, operator definitions, and checksums;
- scripts under `experiments/`, checkpoints/histories and machine-readable metrics under `results/`;
- raw per-pair, per-seed, per-layer `d(alpha)`, `LD`, and `MJ` arrays sufficient to regenerate every
  aggregate;
- a data/detail audit figure demonstrating downsampling, block repetition, and the removed component;
- training/validation curves with selected checkpoints;
- a held-out super-resolution panel comparing input, fixed baselines, model prediction, target, and
  removed-detail residuals;
- task-adequacy and baseline comparisons with uncertainty;
- hand-selected and aggregate hidden-layer `d(alpha)` curves;
- high-resolution predictions along frozen interpolation paths;
- layerwise paired differences, per-pair scatter, seed results, and final-checkpoint control;
- current-best `RESULTS.md` and a self-contained `REPORT.md` structured as
  `Summary -> Methods -> Results -> Conclusion`.

Every reported quantitative metric must be defined in `REPORT.md`, used by a stated claim, and have a
corresponding saved figure under `plots/`. Every used plot must be embedded in both curated
deliverables where relevant, motivated in prose, and followed by a visible numbered caption. Use a
color-vision-deficiency-safe palette plus non-color encodings. Run the repository render checker so
all equations, images, and captions render correctly.

## Stages (checklist)

- [x] **S1 — Freeze and validate the protocol.** Reuse direction 16's code where practical; freeze
  splits, `D/U/P`, baselines, seeds, optimization, checkpoints, pair IDs, metrics, and decision rule;
  test the operator identities, removed-detail energy, manifests, and endpoint pool.
- [x] **S2 — Train matched models and establish task adequacy.** Train both models for all 3 seeds,
  assert bit-identical trunk initialization and shared batch order, select validation checkpoints,
  evaluate classifier accuracy and low-to-high baselines/detail recovery, and save checkpoints,
  histories, numeric outputs, and figures.
- [x] **S3 — Run the frozen interpolation probe.** Probe all 90 pairs, both models, all seeds, and
  both checkpoint conditions; verify endpoints and deterministic reruns; save raw arrays, selected
  `d(alpha)` curves, and full/detail high-resolution path predictions.
- [x] **S4 — Aggregate, interpret, and finalize.** Bootstrap the paired hidden-layer effects, run
  normalization and checkpoint checks, apply the preregistered verdict, curate current-best figures
  and deliverables, update history, run render checks, address feedback, and write `STOP` only when
  every success criterion is satisfied.

## Interpretation

- **Robustly smoother low-to-high predictor:** continuous supervision reduces plateaus even when the
  target requires prediction of spatial detail absent from the input; the earlier result is not
  explained solely by downsampling the output.
- **No gap after adequate super-resolution:** continuous targets alone are not sufficient here; task
  complexity or detail prediction may restore plateau-like behavior. This does not by itself isolate
  which factor caused the change from direction 16.
- **Reverse effect:** the information-predicting continuous task is more plateau-like than
  classification under this assay; report it directly.
- **Layer, seed, or metric dependence:** call the result mixed and show the dependence rather than
  averaging it away.

## Limitations

- Endpoint activations are in distribution, but their interpolated path need not be. The experiment
  compares objectives under a common off-manifold probe; it does not prove that intermediate states
  occur on natural inputs.
- Low-to-high MNIST prediction is one-to-many. An MSE model estimates conditional means and may blur
  genuinely ambiguous detail. Positive detail recovery demonstrates learned statistical prediction,
  not literal recovery of every discarded pixel.
- The task heads have different dimensions and parameter counts (`10` versus `784`). The shared
  hidden layers are therefore the primary causal comparison; output-space curves are descriptive.
- Direction 16 and this direction use different inputs as required by their tasks. A numerical
  difference between directions is historical context, not a controlled cross-direction effect.

## Out of scope (do NOT)

- Do not add Gaussian corruption, sweep input resolutions, or tune the downsampling operator.
- Do not add convolutional networks, U-Nets, generative adversarial networks, diffusion models,
  perceptual losses, or alternative architectures/datasets before the primary MLP assay is complete.
- Do not add labels as predictor inputs or train the privileged digit-template diagnostic.
- Do not add target-quantization, manifold, density, or local-complexity studies.
- Do not tune models separately, choose checkpoints by smoothness, or change pairs/metrics after
  viewing interpolation results.
- Do not use the unequal-dimensional output layers as the primary smoothness comparison.
- Do not claim the model creates information or recovers unknowable sample-specific detail.
- Do not install or replace torch, torchvision, TransformerLens, JAX, Flax, or Cupbearer.

Read `../BUDGET.md` and `../CLAUDE.md` at the start of every iteration and obey the dynamically assigned
GPU, CPU, RAM, and time limits. Operator feedback or review files take priority over advancing a
stage. Keep `RESULTS.md` and `REPORT.md` current-best only; append history only to `CHANGELOG.md` and
working notes only to `JOURNAL.md`. Never write `STOP` while unaddressed feedback remains.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> — <stage, % done, blocker if any>`

## Current status

**Complete — S1 through S4 all done (2026-07-28), verdict robust positive.** Full frozen protocol
ran end to end: operator audit (removed detail = 39.1% of pixel energy, identities to <=2.4e-07),
3 matched seeds trained with bit-identical trunk initialization, both task gates passed (classifier
95.8/96.3/96.8% top-1; predictor `R2_detail` 0.660 [0.654, 0.666], beating bicubic 0.195 and the
privileged digit template 0.165), and the 90-pair SLERP probe run at both checkpoint conditions.
Classifier − predictor linearity deviation: hidden 2 0.0968 [0.0873, 0.1063], hidden 3
0.1335 [0.1197, 0.1474], output 0.1403 [0.1300, 0.1510]; predictor smoother on 90/90 pairs at every
layer with every seed positive, max normalized jump agreeing at every layer. Fraction-normalization
and final-checkpoint controls both keep every interval above zero. `RESULTS.md` and `REPORT.md` are
curated current-best with all 10 figures embedded and captioned. No feedback file has ever been
present, so `STOP` is written.

Re-entered 2026-08-03: `STOP` is gitignored repo-wide and did not survive the workspace checkout, and
the shared `check_render.py` had meanwhile gained the rule-9a / rule-9d checks, which the deliverables
failed on five tables sitting under a bare label or heading. Each now carries a claim-stating prose
paragraph above it. No experiment was rerun; every published number was re-verified against
`results/aggregate.json` and is unchanged, `check_render.py REPORT.md RESULTS.md` exits 0, and `STOP`
is rewritten.

## Next step

None while `STOP` stands. If an operator drops a `human_feedback*.md` / `*REVIEW*` file, the next
iteration must delete `STOP`, address every point in the deliverables, rename the file
`.addressed.md`, log it in `CHANGELOG.md` + `JOURNAL.md`, and only then re-write `STOP`.
