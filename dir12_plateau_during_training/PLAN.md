# PLAN — Direction: Animate plateau formation through training in the MNIST MLP

> Working folder: `plateau_during_training`. Agent REWRITES "Current status"/"Next step" + ticks stages each
> iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.

## Research question

How do activation plateaus emerge and sharpen as the 4-layer ReLU MNIST MLP is trained?

**Reopened extension.** If the same MLP is trained from random initialization on the full 60,000-image MNIST
training set, do its sub-plateau structures form, disappear, or merge differently from the existing run trained
on a fixed 1,000-example draw, especially on 3-to-5 interpolation paths? The full-data experiment starts fresh;
it does not fine-tune or continue from the 1,000-example-trained model.

The central experiment is longitudinal: at many fixed training checkpoints, run exactly the same early-layer
activation interpolation, save the resulting downstream activations, and render the plateau curve as one frame
of an animation. The movie should show whether plateau -> boundary -> plateau structure appears gradually,
suddenly, or at different times for different digit pairs.

## Authoritative definition and implementation anchor

The target phenomenon is defined by Matthew Shinkle and StefanHex's post
[*Activation Plateaus: Where and How They Emerge*](https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge):
interpolate between two early activations, patch every interpolation point into the model, and plot its relative
downstream distance to the two endpoint outputs. A plateau -> boundary -> plateau appears when `d(t)` stays near
0, transitions rapidly, then stays near 1.

The canonical implementation in this branch is `interpolate_digits.py`, including its `slerp_path` routine.
**Read and reuse that script before writing any new plateau code.** Extend it to load training checkpoints and
save per-checkpoint records; do not replace it with the radial-perturbation pipeline. If any wording in this PLAN
is ambiguous, the post's interpolation experiment and `interpolate_digits.py` take precedence.

Random-direction perturbation measures local robustness around one activation. It is related evidence, but it is
not the definition of a plateau in this direction.

## Success criterion (definition of "done")

`REPORT.md` and `RESULTS.md` contain:

- a checkpointed training run from initialization to 100,000 optimization steps;
- the same fixed activation-interpolation experiment evaluated at every saved checkpoint;
- raw endpoint, interpolated, downstream-layer, and logit activations saved so every frame can be regenerated
  without rerunning training;
- a clear animation showing plateau evolution through training;
- a static heatmap or selected frames that remain understandable without playing the animation;
- train/test accuracy and confidence shown alongside the plateau evolution;
- confirmation on two additional seeds after the primary animation works;
- a concise verdict on when plateaus emerge and whether the evolution is consistent across pairs and seeds;
- a new run trained from an untrained step-0 initialization on all 60,000 MNIST training images, shuffled
  without replacement each epoch;
- a side-by-side animation and fixed-path comparison against the existing 1,000-example run that answer whether
  3-to-5 sub-plateau evolution differs under full-data training.

Expected, null, and non-monotonic results are all COMPLETE. When the question is answered, write an empty
`STOP` file.

## Core plateau protocol

Match the basic protocol in *Activation Plateaus: Where and How They Emerge* as closely as the MLP architecture
allows:

1. Choose two fixed MNIST test images, `A` and `B`.
2. Run both through the checkpointed model and record their post-ReLU first-hidden-layer activations,
   `h1_A` and `h1_B`.
3. Generate 50 interpolation points using spherical interpolation with linearly interpolated endpoint norms,
   matching the post's `slerp_rescale` convention.
4. Patch each interpolated activation at the output of hidden layer 1.
5. Propagate through the remainder of the model and record hidden layers 2, 3, the final hidden layer, and logits.
6. At every recorded layer, compute the post's relative endpoint distance:

   `d(alpha) = ||x(alpha) - A|| / (||x(alpha) - A|| + ||x(alpha) - B||)`.

7. Plot `d(alpha)` against interpolation position `alpha`. A plateau-boundary-plateau curve stays close to 0,
   changes rapidly over a narrow interval, and stays close to 1.

The primary animation uses logit-space `d(alpha)`, which is the closest analogue to the post. Layerwise curves
are saved and shown at selected checkpoints to reveal how the plateau is sharpened by successive MLP layers.

## What "sub-plateau merge" means in the extension

The raw Matthew-style `d(alpha)` curve remains the definition of plateau phenomenology. Predicted-class runs
are a behavioral annotation of that curve, not a replacement plateau definition.

For one fixed interpolation path, run-length encode the 50 predicted classes into maximal contiguous segments.
For example, `2,2,3,...,3,5,...,5` becomes `2 | 3 | 5`.

- A **segment disappearance/simplification** occurs when an intervening predicted-class segment vanishes and
  the run-length-encoded path becomes shorter.
- A **merge** is the narrower case in which two non-adjacent segments with the same predicted class become one
  contiguous segment because the segment between them disappears.
- An **endpoint correction** occurs when training changes the prediction of one of the two endpoint images.
  This must be reported separately from a merge.

The current seed-0 3-to-5 path is `2 | 3 | 5` because the fixed 3 endpoint is misclassified as 2. If the fresh
full-data run produces `3 | 5`, call that endpoint correction plus removal of a third-class detour, not a
merge of the global 3 and 5 regions. A one-dimensional path cannot establish global connected-component
topology, so the report must say "segments along the measured paths merge/disappear," not "the class regions
globally merge."

## Full-MNIST from-scratch experiment (new primary task)

For each seed, start from the saved **step-0 untrained weights**, not the step-100,000 weights. Loading the old
run's step-0 checkpoint preserves the exact random initialization for comparison while still training the
full-data model entirely from scratch. Verify the step-0 weights have not received an optimizer update.

Train on all 60,000 MNIST training images, shuffled without replacement every epoch. One epoch is 300 steps at
batch size 200. Assert that all 60,000 training indices are seen exactly once before the first reshuffle. Use
AdamW with learning rate `1e-3`, weight decay `0.01`, and a fixed cosine schedule to `1e-6` over 30,000 steps
(100 full-data epochs). If this schedule is numerically unstable, lower the learning rate using loss traces
only, record the decision in `JOURNAL.md`, and restart the full-data run from step 0. Do not choose a schedule by
looking at plateau curves.

Use seed 0 for the full-resolution animation, then repeat the frozen experiment for seeds 1 and 2. Keep MSE,
the architecture, intervention layer, test pool, and all original pair IDs fixed. The completed 1,000-example
run is the reference trajectory; do not resume or fine-tune it.

### Evaluation paths

- Preserve the original 55-pair bank so the new frames are directly comparable with the existing report.
- Add 50 deterministic 3-to-5 test-image pairs selected before viewing full-data results. Do not filter them
  for correct endpoint predictions. Save endpoint correctness at training step 0 and every checkpoint, and
  report all paths plus the subset whose endpoints were already correct at step 0.
- Use the exact same endpoint image IDs, SLERP coefficients, axes, and colors in the full-data and existing
  1,000-example animations.

For every path and checkpoint, save the raw `d(alpha)` curves, logits, probabilities, predicted classes,
run-length-encoded class sequence, endpoint predictions, and endpoint confidence. The primary evidence is a
side-by-side animation of the existing 1,000-example run and the fresh full-60k run. The compact summary is the
per-path evolution of segment count and third-class detour presence. Do not infer a merge from a change seen in
only one hand-picked path.

## Intervention layer

Use the **post-ReLU output of hidden layer 1** as the primary intervention point.

This is the earliest hidden representation in the 4-layer MLP and is the direct analogue of interpolating at an
early `resid_post` layer in the post. Intervening earlier would mean interpolating raw MNIST pixels. That asks a
different question about input-space image morphing and is not part of the primary experiment.

Do not vary the intervention layer in the main training animation. Holding it fixed ensures that changes between
frames are caused by training, not by changing the number of downstream layers.

## Fixed image-pair bank

Select all pairs before inspecting intermediate-checkpoint plateau curves and save their test-set indices:

- one deterministic pair for each unordered pair of different true digits: 45 cross-class pairs;
- one deterministic within-class pair for each digit: 10 within-class controls.

Use a fixed seed and record the exact image IDs. Do not replace a pair because its animation looks uninteresting.
At each checkpoint, save both true labels and current predictions/confidences, since early checkpoints may not
yet classify the endpoints correctly.

For the main animation, show a fixed readable subset of ten cross-class pairs chosen by digit identity before
viewing results. Save individual animations/heatmaps for all 55 pairs so heterogeneous behavior is not hidden.

## Checkpoint schedule

The model is small, so use a simple high-resolution schedule rather than trying to guess the transition time:

- save steps `0, 10, 30, 100, 300` to capture very early learning;
- save every 500 optimization steps from `500` through `100,000`.

This gives 205 frames and avoids cherry-picking checkpoints after seeing the result. Save model `state_dict`,
training step, optimizer/config metadata, train/test loss, train/test accuracy, and mean prediction confidence.
Optimizer state is needed only for resumability and may be stored less frequently if disk limits require it.

After the primary-seed movie is rendered, identify any adjacent 500-step frames between which the plateau changes
abruptly. If the timing cannot be resolved, rerun that seed deterministically and save every 50 steps only within
the relevant interval. Do not globally increase checkpoint density unless the transition genuinely requires it.

For the reopened full-data run, define step 0 as the untrained initialization. Save steps `0, 10, 30, 100`, then
every 300 steps through `30,000`. This yields early frames plus one frame per full-MNIST epoch. For comparison
with the existing 1,000-example animation, align frames by optimizer step and also show train/test accuracy so
differences in fitting time are visible rather than hidden.

## What to save at every checkpoint

For every fixed pair, save one self-contained record containing:

- checkpoint step and seed;
- endpoint image IDs, true labels, predictions, and confidences;
- endpoint activations at every hidden layer and endpoint logits;
- the 50 interpolation coefficients;
- the 50 interpolated hidden-layer-1 activations;
- downstream activations at every remaining hidden layer for all interpolation points;
- logits for all interpolation points;
- relative-distance curves `d(alpha)` at every recorded layer;
- predicted class and maximum softmax probability at every interpolation point.

Use a documented stable schema such as one `.pt` or `.npz` file per checkpoint and seed. Include a manifest that
lists every expected checkpoint and flags missing/corrupt files. Save numeric arrays, not only PNG frames.

## Optional perturbation control (after the interpolation deliverable)

The existing random-direction perturbation experiment may be rerun at a small number of selected checkpoints
only after the interpolation animation is complete:

- choose a fixed class-balanced set of natural first-hidden-layer activations;
- use the same perturbation directions and radius grid at the selected checkpoints;
- measure last-hidden and logit displacement;
- compare the aggregate natural-activation response with norm-and-sparsity-matched random activations.

Its role is only to ask whether local robustness changes consistently with the interpolation curves. It is not a
required animation, must not determine checkpoint selection, and must not be used to define or count plateaus.

## Minimal reported quantities

Keep the report readable. The primary evidence is the saved `d(alpha)` curves, predicted-class trajectories,
and their animation. Report only:

1. train/test accuracy and mean confidence versus training step;
2. the raw relative-distance curves `d(alpha)`;
3. predicted class along the interpolation path.

For the reopened extension, also report the run-length-encoded predicted-class sequence and the paired change
in segment count / third-class detour presence. These quantities answer the new merge question directly; they
must remain annotations beside the raw curves rather than a thresholded definition of plateau.

Do not impose a scalar threshold that decides whether a curve "is a plateau" in the primary analysis. First show
the curves and animation using the post's phenomenology. If the visual transition is too ambiguous to describe,
add at most one transparent curve-derived summary and explain why it is necessary.

Do not add Jacobian norms, silhouette scores, cluster counts, manifold distances, AUC variants, or additional
sharpness metrics unless the primary experiment reveals a specific ambiguity they are required to resolve.

## Fallback (if time runs short)

Run one seed and save every 2,000 steps plus `0, 10, 30, 100, 300`, giving at least 55 frames. Evaluate ten fixed
cross-class pairs and the ten within-class controls, save all downstream activations/logits, and produce one
logit-relative-distance animation plus a static early/middle/late comparison. State clearly that seed and pair
coverage are limited. The wrapper reserves the last 20 minutes to finalize + STOP.

For the reopened extension, the minimum acceptable deliverable is one fresh seed-0 full-MNIST run, all 50
3-to-5 paths, and steps `0, 10, 30, 100, 300, 1,500, 3,000, 6,000, 15,000, 30,000`, with a comparison against
the existing 1,000-example animation and a verdict that distinguishes endpoint correction, detour disappearance,
and a true same-class segment merge. A null result is complete.

## Setup (fixed)

- Build on the existing `image-models` branch and reuse its model and training configuration.
- Model: 4-layer ReLU MLP, hidden width 200.
- Data for the completed reference: the same fixed 1,000-example MNIST draw and train/test split used in the
  existing experiment. Data for the reopened run: the full 60,000-image MNIST training set, shuffled without
  replacement.
- Training: initialize from the saved untrained step-0 weights and train the full-data model from scratch for
  30,000 steps with batch size 200 and the fixed schedule above. Never load the step-100,000 trained weights.
- Seeds: one primary seed and two confirmation seeds.
- Primary intervention: first hidden-layer post-ReLU activation.
- Interpolation: 50-point norm-rescaled SLERP from `alpha=0` to `alpha=1`.
- Primary recording: logits. Also record every downstream hidden-layer activation.
- Fix pair IDs, interpolation coefficients, plot limits, colors, and animation layout across checkpoints.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` — read both every iteration.**
- **Deliverable hygiene (see CLAUDE.md):** RESULTS.md/REPORT.md = current-best only, no history; CHANGELOG.md
  = the history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, flax** — they break the CUDA
  build.

## Stages (checklist)

- [x] **S1 — Match the post at the final checkpoint.** Read `../BUDGET.md` and `../CLAUDE.md`. Read
  `interpolate_digits.py` end to end and use its `slerp_path` and relative-distance convention as the starting
  point. Extend/test it for norm-rescaled SLERP at hidden layer 1, downstream recording,
  and relative endpoint distance. At the 100,000-step checkpoint, reproduce plateau and non-plateau examples
  with 50 interpolation points and predicted class along the path. Verify that `alpha=0/1` exactly reproduce the
  two unpatched endpoint outputs.
- [x] **S2 — Lock pairs, schema, and checkpoints.** Freeze the 55 image pairs, checkpoint schedule, plot axes,
  and saved-record schema. Write a manifest test that checks every file contains endpoints, 50 interpolation
  points, all downstream layers, logits, per-point predictions, and metadata.
- [x] **S3 — Run and record the primary training movie.** Train the primary seed, save the scheduled state dicts
  and training metrics, then evaluate the exact same `interpolate_digits.py`-based protocol at every checkpoint.
  Evaluation may run online or offline, but it must use the checkpointed weights and fixed pair IDs. Do not save
  figures without the underlying numeric arrays. Do not substitute radial perturbations for this stage.
- [x] **S4 — Render the animation and static summary.** Produce a main MP4/GIF with training step visible in every
  frame, fixed axes, logit `d(alpha)` curves for the fixed ten-pair subset, and a compact accuracy/confidence
  inset. Produce one static training-step-by-interpolation heatmap for representative pairs and one layerwise
  early/middle/late plot. Avoid a wall of checkpoint figures in `REPORT.md`.
- [x] **S5 — Resolve timing and confirm.** Inspect adjacent animation frames to identify the transition interval.
  If 500-step resolution is insufficient, rerun densely at 50-step spacing inside that interval. Repeat the
  frozen main protocol for two additional seeds and compare whether the emergence time and qualitative movie
  are stable.
- [x] **S6 — Verdict and stop.** State whether plateaus appear gradually or abruptly, whether different digit
  pairs synchronize, and whether sharpening continues after test accuracy stabilizes. Keep only the main
  animation, minimal static figures, and required tables in REPORT/RESULTS; move development history to
  CHANGELOG; write empty `STOP`.
- [x] **S7 — Implement and verify fresh full-data training.** Remove the stale `STOP` if present. Add the smallest
  training script that loads each original step-0 untrained MSE checkpoint and trains a new model on all 60,000
  MNIST training images. Test that no trained checkpoint is loaded, all 60,000 indices appear exactly once in
  the first epoch, and the checkpoint manifest is complete before launching long runs.
- [x] **S8 — Record full-resolution seed 0.** Train seed 0 from step 0 and evaluate the unchanged Matthew-style
  protocol at every scheduled checkpoint. Preserve the original pair bank and add the frozen 50-path 3-to-5
  bank. Save numeric arrays before rendering figures.
- [x] **S9 — Render, compare, and replicate.** Produce a synchronized comparison of the existing 1,000-example
  animation and the fresh full-60k animation, plus a static 3-to-5 panel showing the original path and the
  distribution over 50 paths. Repeat the full-data run for seeds 1 and 2 at fallback checkpoint density. Do not
  call endpoint correction a merge.
- [x] **S10 — Rewrite the verdict and stop.** Curate `REPORT.md` and `RESULTS.md` to answer whether plateau and
  sub-plateau evolution differs when a fresh model is trained on full MNIST. Include the full-60k sampling
  method, from-scratch initialization, operational merge definition, raw-curve animation, segment summary,
  endpoint-correction audit, and limitations. Append history to `CHANGELOG.md`/`JOURNAL.md`, verify artifacts,
  then write an empty `STOP`.

## Required deliverables

- `results/checkpoint_manifest.*`
- `results/plateau_records/seed_<n>/step_<step>.*`
- `plots/plateau_evolution.mp4` or `.gif`
- `plots/plateau_training_heatmap.*`
- `plots/layerwise_selected_steps.*`
- `plots/training_context.*`
- `results/full_mnist_from_scratch/seed_<n>/manifest.*`
- `results/full_mnist_from_scratch/seed_<n>/step_<step>.*`
- `plots/full_mnist_3v5_training.gif`
- `plots/full_mnist_3v5_summary.*`
- `REPORT.md` with a short Methods section and direct verdict

## Out of scope (do NOT)

- Using matched-random activations to label individual points as "plateau-positive."
- Treating random-direction perturbation response as the definition of a plateau.
- Claiming that the animation alone proves exactly ten connected stable regions.
- Interpolating raw pixels as part of the primary experiment.
- Varying intervention layer, architecture, width, loss, or optimizer family.
- Additional dataset-size sweeps or curricula beyond the existing 1k reference versus fresh full-60k run.
- Larger ResNet or small-GPT experiments.
- Manifold-connectivity, steering, Jacobian, spline, or clustering analyses.
- Claiming global region topology, connectedness, or class-region merging from one-dimensional interpolation
  paths.

The saved activation records should make later region-counting or mechanism analyses possible without retraining,
but those analyses must not delay completion of this training-evolution experiment.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status

COMPLETE (2026-07-17, iter 10). Operator feedback `human_feedback_1` addressed: the 60k run now uses a
smooth `ReduceLROnPlateau(0.5, 100)` schedule (chosen by a 60k scheduler search — the previous cosine was
noisy, spike 101×), every 1k-only experiment was rerun on 60k (MSE seeds 0/1/2 + CE seed 0, 258 records +
201 early-zoom, manifest-verified), and REPORT/RESULTS are refocused on the 60k results with the 1k run
kept only in a dedicated "effect of training-set size" section. Headline (60k, smooth): plateaus form early
and keep sharpening to PF 0.674/0.663/0.668 while the loss converges smoothly (test acc 0.9775/0.9795/0.9785,
late curve motion 7.6e-4); CE logit PF stays at the 0.25 floor while probability PF reaches 0.90. Correction
surfaced by the refocus: 3v5 is NOT the hardest pair on 60k (AUROC 0.9993, rank 4/45) — it was rank 1/45
(0.977) only on 1k, so pair difficulty was a small-data effect; finding 4 rewritten. 3→5 verdict unchanged:
endpoint correction (49/50 vs 36/50), not merging. Render checks clean (14/14 display math, all 19 figures
embedded in both files); STOP written.

## Next step

None — plan complete, feedback human_feedback_1 addressed, zero unaddressed feedback, STOP written.