# PLAN — Direction: Animate plateau formation through training in the MNIST MLP

> Working folder: `plateau_during_training`. Agent REWRITES "Current status"/"Next step" + ticks stages each
> iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.

## Research question

How do activation plateaus emerge and sharpen as the 4-layer ReLU MNIST MLP is trained?

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
- a concise verdict on when plateaus emerge and whether the evolution is consistent across pairs and seeds.

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

## Setup (fixed)

- Build on the existing `image-models` branch and reuse its model and training configuration.
- Model: 4-layer ReLU MLP, hidden width 200.
- Data: the same fixed 1,000-example MNIST training subset and train/test split used in the existing experiment.
- Training: batch size 200 for 100,000 optimization steps. Recover and record the exact optimizer, learning rate,
  weight decay, initialization, preprocessing, and data-order conventions from the branch.
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

- [ ] **S1 — Match the post at the final checkpoint.** Read `../BUDGET.md` and `../CLAUDE.md`. Read
  `interpolate_digits.py` end to end and use its `slerp_path` and relative-distance convention as the starting
  point. Extend/test it for norm-rescaled SLERP at hidden layer 1, downstream recording,
  and relative endpoint distance. At the 100,000-step checkpoint, reproduce plateau and non-plateau examples
  with 50 interpolation points and predicted class along the path. Verify that `alpha=0/1` exactly reproduce the
  two unpatched endpoint outputs.
- [ ] **S2 — Lock pairs, schema, and checkpoints.** Freeze the 55 image pairs, checkpoint schedule, plot axes,
  and saved-record schema. Write a manifest test that checks every file contains endpoints, 50 interpolation
  points, all downstream layers, logits, per-point predictions, and metadata.
- [ ] **S3 — Run and record the primary training movie.** Train the primary seed, save the scheduled state dicts
  and training metrics, then evaluate the exact same `interpolate_digits.py`-based protocol at every checkpoint.
  Evaluation may run online or offline, but it must use the checkpointed weights and fixed pair IDs. Do not save
  figures without the underlying numeric arrays. Do not substitute radial perturbations for this stage.
- [ ] **S4 — Render the animation and static summary.** Produce a main MP4/GIF with training step visible in every
  frame, fixed axes, logit `d(alpha)` curves for the fixed ten-pair subset, and a compact accuracy/confidence
  inset. Produce one static training-step-by-interpolation heatmap for representative pairs and one layerwise
  early/middle/late plot. Avoid a wall of checkpoint figures in `REPORT.md`.
- [ ] **S5 — Resolve timing and confirm.** Inspect adjacent animation frames to identify the transition interval.
  If 500-step resolution is insufficient, rerun densely at 50-step spacing inside that interval. Repeat the
  frozen main protocol for two additional seeds and compare whether the emergence time and qualitative movie
  are stable.
- [ ] **S6 — Verdict and stop.** State whether plateaus appear gradually or abruptly, whether different digit
  pairs synchronize, and whether sharpening continues after test accuracy stabilizes. Keep only the main
  animation, minimal static figures, and required tables in REPORT/RESULTS; move development history to
  CHANGELOG; write empty `STOP`.

## Required deliverables

- `results/checkpoint_manifest.*`
- `results/plateau_records/seed_<n>/step_<step>.*`
- `plots/plateau_evolution.mp4` or `.gif`
- `plots/plateau_training_heatmap.*`
- `plots/layerwise_selected_steps.*`
- `plots/training_context.*`
- `REPORT.md` with a short Methods section and direct verdict

## Out of scope (do NOT)

- Using matched-random activations to label individual points as "plateau-positive."
- Treating random-direction perturbation response as the definition of a plateau.
- Claiming that the animation alone proves exactly ten connected stable regions.
- Interpolating raw pixels as part of the primary experiment.
- Varying intervention layer, architecture, width, optimizer, or dataset size.
- Larger ResNet or small-GPT experiments.
- Manifold-connectivity, steering, Jacobian, spline, or clustering analyses.

The saved activation records should make later region-counting or mechanism analyses possible without retraining,
but those analyses must not delay completion of this training-evolution experiment.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status

The branch already contains `interpolate_digits.py`, which implements the relevant endpoint-activation SLERP,
relative-distance `d(t)` curve, and predicted-class trajectory at the final trained model. What is missing is an
extension of that exact experiment across fixed training checkpoints, with raw activations saved for animation.

## Next step

Read `../BUDGET.md`, `../CLAUDE.md`, Matthew's post, and `interpolate_digits.py`. Run that script unchanged on its
existing final-checkpoint example first. Then make the smallest extension needed to emit one checkpoint record
containing endpoint activations, 50 SLERP points, downstream activations, logits, `d(alpha)`, and predicted class
along the path. Confirm exact endpoint reproduction before locking the pair bank and launching checkpointed
training.