# PLAN — Continuous-to-Discrete Target Sweep on MNIST

> Working folder: `continuous_target_plateau`. Agent REWRITES "Current status"/"Next step" + ticks stages each
> iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.

## Research question

Can an activation plateau emerge in a regression task whose target is always continuous, as the target function is gradually changed from approximately linear to approximately discrete?

The experiment keeps the inputs, model architecture, loss, and training procedure fixed. Only one target-sharpness parameter \(k\) changes.

## Hypothesis

A continuous target with nearly constant slope should produce relatively uniform activation movement along the brightness direction.

As the target becomes more switch-like, activation movement should become concentrated near the target transition, producing:

`plateau -> rapid transition -> plateau`

A null result is also informative: it would show that a switch-like continuous target is not sufficient to produce the activation plateaus observed in classification.

## Success criterion (definition of "done")

The direction is complete when:

1. Ten target-sharpness settings have been trained with identical inputs and training settings.
2. Every reported training run satisfies the training-adequacy requirements below.
3. `REPORT.md` contains:
   - the ten target functions;
   - train and validation loss curves;
   - target and prediction curves over brightness;
   - layerwise activation-movement curves;
   - a quantitative activation-concentration score versus \(k\);
   - a clear verdict on whether plateau strength increases as the continuous target becomes more switch-like.
4. Primary results use three random seeds and report the mean and uncertainty across held-out images and seeds.
5. Every reported metric is defined in the Methods section and has a saved figure in `plots/`.

A null or non-monotonic result is complete if the experiment is valid and the result is clearly documented. When complete, write an empty `STOP` file.

## Fallback (if time runs short)

Train all ten \(k\) values with one seed, evaluate the deepest hidden layer on at least 50 held-out MNIST images, and produce:

- target and prediction curves;
- train and validation loss curves;
- normalized activation-movement curves;
- activation-concentration score versus \(k\);
- a short verdict in `REPORT.md`.

The wrapper reserves the final 20 minutes to finalize the report and write `STOP`.

## Setup (fixed)

### Data

Use MNIST images, but do not use digit labels as learning targets.

For every image \(x\), first normalize its \(L_2\) norm:

\[
\tilde{x} = \frac{x}{\lVert x\rVert_2 + \epsilon}.
\]

Sample a continuous brightness value:

\[
b \sim U(0.4, 1.0),
\]

and construct the model input:

\[
x_b = b\tilde{x}.
\]

Because \(\lVert\tilde{x}\rVert_2 \approx 1\), brightness is controlled by \(b\), while digit identity and writing style are nuisance variation.

Use fixed train, validation, and test splits based on the original MNIST images. Brightness assignments must also be fixed and shared across all ten \(k\) settings within each seed. This ensures that the models receive exactly the same inputs.

Recommended initial split:

- all training images;
- all validation images;
- all test images.

Digit labels may only be used to make the splits and evaluation samples digit-balanced.

### Continuous target family

Set the transition center to:

\[
b_0 = 0.7.
\]

Train the model to predict:

\[
y_k(b)
=
\frac{\tanh(k(b-b_0))}
{\tanh(0.3k)}.
\]

The denominator keeps the endpoint target range approximately fixed at \([-1,1]\) for every \(k\). This prevents target amplitude from becoming a confounding variable.

Use ten values (extended from five after operator feedback #1, 2026-07-29, which noted that the
original grid never reached a genuinely discrete target):

\[
k \in \{0.5,\ 1,\ 2,\ 5,\ 10,\ 20,\ 40,\ 80,\ 160,\ 320\}.
\]

Interpretation:

- \(k=0.5\): approximately linear continuous target;
- \(k=1,2\): weak to moderate saturation;
- \(k=5\): strong saturation;
- \(k=10\): sigmoid, but its transition still spans half the brightness range;
- \(k=20,40\): switch-like;
- \(k=80,160,320\): a step function at the resolution of the 201-point probe grid
  (transition width 0.0046 at \(k=320\) vs grid spacing 0.003).

Before training, save a single plot containing all ten target curves.

### Model

Reuse the exact 4-layer ReLU MLP architecture from the existing MNIST plateau experiment whenever possible.

Only change the output head to produce one scalar regression value.

Record post-ReLU activations from every hidden layer.

### Training

- Loss: mean squared error.
- Optimizer, initialization, batch size, and regularization: identical across all ten \(k\) values.
- Train for a fixed maximum number of epochs that is long enough to observe slight validation overfitting.
- Do not tune learning rate, model size, or regularization separately for individual \(k\) values.
- If the initial training schedule is inadequate, adjust the global schedule and rerun all ten \(k\) values.
- Save both the final checkpoint and the minimum-validation-loss checkpoint.
- Use the final converged checkpoint for the primary analysis.
- Use the minimum-validation-loss checkpoint only as a robustness check if the conclusions differ materially.

### Training adequacy requirements

Training is adequate only when both conditions hold:

1. **Slight validation overfitting**
   - Validation loss reaches its minimum before the end of training.
   - Validation loss subsequently increases modestly.
   - The final validation loss should not be more than approximately 20% above its minimum; larger increases indicate excessive overfitting.

2. **Smooth training-loss convergence**
   - Training loss converges to a smooth minimum.
   - The final training loss is within approximately 5% of the minimum training loss.
   - There is no clear upward trend, instability, large oscillation, or final overshoot.

Save training and validation loss curves for every \(k\). Do not report plateau results from an inadequately trained model.

- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` — read both every iteration.**
- **Deliverable hygiene:** `RESULTS.md`/`REPORT.md` contain current-best results only; `CHANGELOG.md` contains experiment history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax** — they may break the CUDA build.

## Plateau evaluation

### Brightness sweep

Select at least 100 digit-balanced held-out test images.

For each image, evaluate:

\[
b \in [0.4,1.0]
\]

using 201 evenly spaced brightness values.

For every brightness value, record:

- true target \(y_k(b)\);
- model prediction \(\hat y_k(b)\);
- activation \(h_l(b)\) from every hidden layer \(l\).

### Normalized local activation movement

For adjacent brightness points, compute:

\[
m_l(b_i)
=
\left\|
h_l(b_{i+1}) - h_l(b_i)
\right\|_2.
\]

Normalize movement along each image path:

\[
s_l(b_i)
=
\frac{m_l(b_i)}
{\sum_j m_l(b_j) + \epsilon}.
\]

Interpretation:

- approximately uniform \(s_l(b)\): no activation plateau;
- low movement at both ends and concentrated movement near \(b_0\): plateau-transition-plateau structure.

Average the curves across held-out images and seeds. Report uncertainty.

### Activation-concentration score

Define the central transition interval as the middle 20% of the brightness range:

\[
b \in [0.64,0.76].
\]

Compute:

\[
C_l(k)
=
\sum_{b_i \in [0.64,0.76]} s_l(b_i).
\]

For perfectly uniform movement:

\[
C_l(k) \approx 0.2.
\]

Report the normalized concentration score. NOTE (operator feedback #1): this is written
\(\Gamma_l(k)\) in the deliverables, not \(R_l(k)\), because \(R_2\) reads as \(R^2\).

\[
\Gamma_l(k)=\frac{C_l(k)}{0.2}.
\]

Interpretation:

- \(\Gamma_l(k)\approx1\): activation movement is approximately uniform;
- \(\Gamma_l(k)>1\): movement is concentrated near the target transition;
- \(\Gamma_l(k)=5\): the maximum — all movement inside the central window;
- increasing \(\Gamma_l(k)\) with \(k\): evidence that plateau strength increases as the continuous target becomes more discrete-like.

Compute the same concentration score for the target curve using \(|y_k(b_{i+1})-y_k(b_i)|\). This provides a reference for how target sharpness changes with \(k\).

## Required figures

Save at least the following:

1. `plots/target_functions.png`
   - All ten continuous target functions.

2. `plots/training_curves.png`
   - Training and validation loss for all ten \(k\) values.
   - Mark the minimum validation-loss epoch.

3. `plots/prediction_sweeps.png`
   - True target and mean model prediction versus brightness for every \(k\).

4. `plots/activation_movement_by_k.png`
   - Normalized activation movement versus brightness.
   - Show at least the first, middle, and deepest hidden layers.

5. `plots/concentration_vs_k.png`
   - Target concentration and activation-concentration score versus \(k\).
   - Include all hidden layers and uncertainty across seeds.

The main presentation figure should combine the target curves, prediction curves, deepest-layer activation movement, and layerwise concentration score into one compact figure.

## Stages (checklist)

- [x] **S1 — Implement and validate the dataset**
  - Normalize MNIST images.
  - Generate fixed brightness values.
  - Verify numerically that \(\lVert x_b\rVert_2 \approx b\).
  - Implement the ten normalized target functions.
  - Save `plots/target_functions.png`.

- [x] **S2 — Train all ten target settings**
  - Train the ten models with identical data and hyperparameters.
  - Save checkpoints and complete loss histories.
  - Check both training-adequacy requirements.
  - If training is inadequate, change only global training settings and rerun all ten models.
  - Save `plots/training_curves.png`.

- [x] **S3 — Measure activation plateaus**
  - Run brightness sweeps on held-out images.
  - Record predictions and hidden-layer activations.
  - Compute normalized local activation movement.
  - Compute target and activation-concentration scores.
  - Save all required plots.
  - Define every metric in `REPORT.md` Methods.

- [x] **S4 — Robustness and final verdict**
  - Repeat the complete experiment for three seeds.
  - Aggregate results across images and seeds.
  - Check whether the conclusion changes at the minimum-validation-loss checkpoints.
  - Write the final Methods, Results, Limitations, and Verdict sections.
  - Update `RESULTS.md`, finalize `REPORT.md`, and write `STOP`.

## Decision rule

The hypothesis is supported if:

- target concentration increases with \(k\);
- activation concentration also increases consistently with \(k\);
- the effect becomes stronger in deeper layers;
- the sharpest (\(k=320\)) model shows low activation movement away from \(b_0\) and concentrated movement near \(b_0\).

The hypothesis is not supported if activation movement remains approximately uniform, is unrelated to \(k\), or does not become more concentrated with depth, despite adequate target fitting and adequate training.

Do not force a monotonic interpretation if the observed result is mixed.

## Out of scope (do NOT)

- Do not train a digit classifier.
- Do not add digit classification as an auxiliary loss.
- Do not compare different model architectures.
- Do not compare cross-entropy with MSE.
- Do not introduce reconstruction targets.
- Do not interpolate between different digit identities.
- Do not add additional target families unless the ten-\(k\) experiment is complete.
- Do not tune hyperparameters separately for different \(k\) values.
- Do not claim that this experiment fully explains classification plateaus.
- Do not describe the activation path as a mathematical manifold unless that property is explicitly demonstrated.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> — <stage, % done, blocker if any>`

## Current status

**COMPLETE (2026-07-29), operator feedback #1 and #2 addressed.** S1-S4 done at the **extended ten-\(k\) scale**
(\(k\) up to 320, a step function at probe-grid resolution). All five success criteria are met: ten \(k\)
settings trained with identical inputs/hyperparameters, all 30 primary runs (10 \(k\) x 3 seeds) pass both
training-adequacy conditions, `REPORT.md` contains every required element, primary results use 3 seeds with
uncertainty across seeds and images, and every metric is defined in Methods with a saved, embedded figure.
The 10,000-image robustness grid (another 30 runs) is reported in full.

**Verdict (decision rule applied):** hypothesis **partially supported; the key prediction fails, now
decisively.** Target concentration rises with \(k\) to its ceiling (\(\Gamma\) 1.01 -> 5.00); activation
concentration rises with \(k\) and strengthens with depth (layer 1 flat at ~1.02, layer 3
1.094 ± 0.010 -> 1.491 ± 0.068) — but then **saturates**: \(\Gamma_3\) is flat at 1.45-1.49 from \(k=20\)
to \(k=320\) across a 16x further sharpening, and flank movement \(\Phi_3\) bottoms out at 0.265 and rises
back to 0.283 against a target \(\Phi\) of exactly 0. The decisive control: on the 10,000-image grid at
\(k=320\) the model **output** is a genuine switch (\(\Gamma\) 4.13 of a maximum 5.00, \(\Phi\) 0.005,
sweep \(R^2\) 0.848) while the deepest hidden layer stays at \(\Gamma_3 = 1.659 \pm 0.168\),
\(\Phi_3 = 0.279\) — output 78% of the way to a perfect plateau, representation 16%. So a switch-like
continuous target is **not sufficient** to produce classification-style plateaus, and the ceiling is a
property of the representation rather than a failure to fit.

**Operator feedback #2 (2026-07-29), addressed.** "The current plots do not show the most extreme
situation — show what d(t) during the transition looks like for different K." The 201-point probe grid
(spacing 0.003) is coarser than the \(k=320\) transition (width 0.0046), so the whole switch fell inside
one plotted step. Added a **6001-point dense probe** (`experiments/zoom.py`, spacing \(10^{-4}\)) over all
60 final checkpoints, two metrics beyond the plan — movement rate \(g_l(b)=(S-1)s_l(b)\), scale-resolved
\(\Gamma_l(w)\) and alignment-free \(\Lambda_l(w)\) — and three figures (`transition_zoom.png`,
`transition_zoom_n10k.png`, `transition_scale.png`) as REPORT.md §6. Findings: recomputing \(\Gamma_3\) at
30x resolution changes it by \(\le 0.006\) at every \(k\) (no hidden spike); at \(k=320\) the target's
movement rate peaks at 96x uniform while layer 3 reaches 1.5x and is flat; alignment-free, layer 3's best
0.005-wide stretch anywhere reaches \(\Lambda_3 = 2.43\) (1k images) / \(3.03\) (10k) against the output's
\(5.44\) / \(11.92\) and the target's \(79.7\). One correction to the previous status: measured
alignment-free, \(\Lambda_3\) does keep creeping up past \(k=20\) (1.89 -> 2.43), so part of the
\(\Gamma_3\) saturation is the model's transition drifting off \(b_0\) rather than a pure representational
ceiling — the verdict is unchanged because the output-to-representation gap widens over the same range.

Deviations from this plan, all logged in JOURNAL.md and CHANGELOG.md: primary training set is 1000
digit-balanced images rather than "all training images" (the two PLAN requirements conflict — a 1-D target
on 50k images shows no validation overfitting, failing the adequacy gate), with the 10,000-image grid
reported in full as the control that removes the fitting confound; a global cosine LR decay was added to
satisfy the smooth-convergence condition; one metric was added beyond the plan, the flank share
\(\Phi_l(k)\), plus \(g_l\), \(\Gamma_l(w)\) and \(\Lambda_l(w)\) for feedback #2; and the concentration score is written \(\Gamma_l(k)\) rather than \(R_l(k)\) per operator
feedback #1.

## Next step

None — the direction is complete, zero unaddressed feedback files remain, and `STOP` is written. If an
operator adds a `human_feedback*` / `*REVIEW*` file later, delete `STOP`, address every point, then
re-write `STOP` only when clean (CLAUDE.md rules 10-11).
