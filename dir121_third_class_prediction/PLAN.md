# PLAN — Direction 12.1: Stable third-class predictions on MNIST activation paths

> Working folder: `dir121_third_class_prediction`. This is a sub-direction of
> `dir12_plateau_during_training`, not direction 14. The agent rewrites “Current status” and “Next step” and
> ticks stages after each iteration. Disk (`PLAN.md`, `JOURNAL.md`, `RESULTS.md`, `CHANGELOG.md`,
> `../BUDGET.md`, and `../CLAUDE.md`) is the only memory.

## Research questions

When interpolating between the first-hidden-layer activations of real MNIST images from two different digit
classes:

1. Which digit-to-digit transitions reliably produce a third-class prediction, averaged over 100 fixed image
   pairs rather than inferred from one unusual pair?
2. Which transitions reliably show a sub-plateau in the mean relative-distance curve but do not reliably
   predict a third class?
3. For transitions with a stable third-class segment, are those interpolated activations close to the
   first-layer activations produced by real images of the predicted third digit?

The first two questions are Stage 1 and must be answered before selecting transitions for the activation-region
analysis. Do not begin with only the old 6→7 example, and do not select a pair or transition because it looks
especially clean.

## Definitions

- A **digit transition** is a pair of distinct endpoint labels. Evaluate all 45 unordered digit pairs. Present
  them in the natural low-digit→high-digit orientation; reversing a path is the same geometric path traversed
  backward, so do not spend compute duplicating all 90 directions.
- A **third-class segment** is a maximal continuous run of interpolation points whose predicted digit is neither
  endpoint digit.
- A **sub-plateau** is an intermediate shelf in the mean logit-space relative endpoint-distance curve, distinct
  from the two endpoint plateaus. Freeze a simple, quantitative shelf rule before inspecting the 45-transition
  screen, record it in `JOURNAL.md`, and show that the classification is not caused by one alpha point or one
  image pair. Report the mean curves as primary evidence even if the rule is needed for the summary table.
- The **activation region of digit z** means the distribution of first-hidden-layer activations produced by
  real MNIST images labeled z, summarized by its mean and spread. A prediction of z alone does not establish
  closeness to this region.

Do not call any one-pair effect stable. “Stable” here means reproducible across the frozen 100-pair bank. It
does not mean merely that a curve is visually striking or that a single pair has a long segment.

## Relationship to direction 12 and output ownership

Direction 12 is the implementation and artifact source. Read and reuse, or minimally extend:

- `../dir12_plateau_during_training/experiments/plateau_protocol.py` for the model, 50-point norm-rescaled
  spherical interpolation, patching at post-ReLU `h1`, downstream propagation, logits, predictions, and
  relative endpoint distance;
- `../dir12_plateau_during_training/experiments/avg_transition_curves.py` for the deterministic 100-pair bank
  pattern (rank-i image of class a paired with rank-i image of class b in the first 2,000 test images);
- `../dir12_plateau_during_training/results/full_mnist_from_scratch/seed_0/ckpts/step30000.pt` as the primary
  converged checkpoint;
- direction 12 manifests and saved pair indices when available, so endpoints and preprocessing remain exactly
  comparable.

Do not retrain the model and do not copy large checkpoints or existing result trees into this directory.
Import code and read immutable inputs from direction 12 by explicit relative or resolved paths. All new scripts,
numeric results, manifests, figures, logs, and reports belong under `dir121_third_class_prediction/` only:

- code in `experiments/`;
- numeric outputs in `results/`;
- figures in `plots/`;
- conclusions in this direction’s `RESULTS.md` and `REPORT.md`.

Never write new outputs into direction 12. Every saved result must record the source checkpoint path and a hash
or other unambiguous checkpoint identifier.

## Success criterion

`REPORT.md` and `RESULTS.md` contain:

- a complete 45-transition screen at the converged direction-12 seed-0 model;
- exactly 100 deterministic test-image pairs per transition and the same 50 alpha values used in direction 12;
- for every transition, the pointwise mean and standard deviation of `d(alpha)`, individual predicted-class
  paths, endpoint correctness, third-class frequencies, and third-digit identities;
- a matrix/table separating: stable third-class prediction, sub-plateau without stable third-class prediction,
  both, and neither;
- evidence that each positive finding is distributed across the 100 pairs rather than produced by an outlier;
- confirmation of the Stage-1 labels on direction-12 seeds 1 and 2, using the same image pairs and their final
  converged checkpoints;
- activation-region analysis for every transition classified as a stable third-class case (or a clearly stated,
  preregistered representative subset if resource limits make all cases impossible);
- a concise verdict distinguishing “activation-region match,” “prediction only,” and “mixed result.”

A null result is complete. When all required questions are answered, write an empty `STOP` file.

## Stage 1 protocol — stable transition census

### Fixed 100-pair bank

For every unordered endpoint-label pair `(a,b)`, pair the rank-i class-a test image with the rank-i class-b
test image for `i = 0,...,99`, following direction 12’s deterministic construction. Use only the same first
2,000 MNIST test images. Save the exact 4,500 endpoint-index pairs once and reuse them for every seed.

Evaluate all paths at the final converged full-60k MSE checkpoint. Use the same 50 alpha values and logit-space
`d(alpha)` as direction 12. For every path save:

- endpoint indices, true labels, endpoint predictions, and endpoint correctness;
- alpha and the raw `d(alpha)` curve;
- logits and predicted digit at every alpha;
- run-length-encoded predicted-class segments.

The primary analysis includes all 100 preregistered pairs. Also report the subset with both endpoints correctly
classified, but never silently discard incorrect endpoints.

### Decide whether a transition has a stable third-class prediction

For each of the 100 paths, record whether any third-class segment appears, which third digit(s) appear, and the
fraction of alpha points assigned to each. Then report per transition:

- the fraction of paths with any third-class segment;
- the fraction for each specific third digit z;
- the pointwise class histogram across alpha;
- the median and interquartile range of third-class segment width;
- the same quantities on the correctly classified-endpoint subset.

Freeze the threshold for the word “stable” before viewing the full transition results. The default rule is that
the same third digit z appears as a continuous segment in at least 50 of the 100 paths and that its median segment
contains at least 3 consecutive alpha points. Report raw counts beside the label and include a sensitivity table
for prevalence thresholds 25%, 50%, and 75%. The threshold is an organizational rule, not a significance test.

### Decide whether a transition has a stable sub-plateau

For every transition, average `d(alpha)` pointwise over all 100 paths and plot the mean with a ±1 standard
deviation band, exactly as in direction 12. Save individual curves as well. Apply the frozen shelf rule to the
mean curve, then verify that the shelf remains visible under both of these checks:

- leave-one-out or small-group removal does not change the classification;
- bootstrap resampling of the 100 pairs assigns the same classification in at least 90% of resamples.

This resampling is only a robustness check against an average dominated by unusual pairs; do not turn the project
into a search for statistical significance. Report the shelf’s alpha interval and `d(alpha)` level. A stable
third-class prediction and a stable sub-plateau are separate labels: neither implies the other.

### Required Stage 1 outputs

1. A 10×10 digit-transition matrix showing the four categories: both phenomena, stable third-class prediction
   only, stable sub-plateau only, and neither. Leave the diagonal blank.
2. A 5×9 grid (or comparably readable multipage figure) of all 45 mean curves with ±1 standard deviation bands,
   annotated with the dominant third digit and its prevalence.
3. For each positive transition, a class-composition plot over alpha and a distribution of per-pair segment
   widths.
4. A machine-readable table containing all classifications, raw counts, shelf intervals, robustness results,
   endpoint accuracy, and source checkpoint identifiers.

Only after these outputs exist may later stages choose transitions for closer study.

## Activation-region measurements

### Real digit reference sets

For every digit 0–9, collect equal-size sets of real MNIST activations at post-ReLU `h1` from the same model.
Use training images to calculate each digit’s activation mean and coordinate-wise variance. Use separate test
images to determine normal held-out distances. Save all image indices and never use interpolation points to
define a digit region.

For interpolation activation `u` and digit `c`, subtract digit c’s mean, divide each coordinate by that
coordinate’s standard deviation, and calculate the root-mean-square difference across all coordinates. On
held-out real digit-c images, calculate the same distance and save its 95th percentile. Divide the interpolation
distance by that percentile:

- below 1 means the point lies within the usual spread of real digit-c activations by this summary;
- above 1 means it is farther from the digit mean than 95% of held-out real digit-c activations.

For a segment predicted as z, call it close to the real-z activation region only when, for most points in the
segment, its normalized distance to z is below 1 and lower than its distance to every other digit. Report the
exact fraction of points and paths satisfying both conditions.

## Two complementary two-dimensional visualizations

Principal component analysis is not used for the main two-dimensional visualization. It finds directions of
largest overall variance, which need not be the directions that separate the three relevant classes or control
the model's decision near the interpolation path. Use both views below because they answer different questions.

### View A — real-class LDA plane

For each endpoint transition `a→b` with stable middle prediction `c`, fit three-class linear discriminant
analysis (LDA) to balanced real training-image `h1` activations from classes a, b, and c. LDA supplies at most
two discriminant axes for three classes. Fit the axes using real training activations only; then project held-out
real a/b/c activations and all interpolation paths without refitting.

Show faint held-out real points, class means, spread ellipses, and interpolation paths colored and marked by
their predicted class. State explicitly that LDA is supervised by the real class labels: it shows whether the
path approaches the directions that best separate real a, b, and c activations, but it does not necessarily
recover the model's local decision directions.

### View B — path-local margin-gradient/SVD plane

For a path whose middle segment predicts c, let `l_k(h)` be the downstream logit for digit k after patching
the first-hidden-layer activation h. At every interpolation point in the third-class segment, and enough adjacent
points on both sides to include the entry and exit boundaries, compute both gradients with respect to h:

- the a-versus-c margin gradient, `grad_h(l_a - l_c)`;
- the b-versus-c margin gradient, `grad_h(l_b - l_c)`.

Stack these gradient row vectors over the selected path points. Centering is not needed because these are
directions, not observations. Compute singular value decomposition and use the top two right-singular vectors as
orthonormal axes in `h1` space. Save the singular values and the fraction of squared gradient norm captured by
the two axes; a low captured fraction must be reported as a limitation of the plane.

Anchor the plane at the mean `h1` activation of the path's third-class segment. Project held-out real a/b/c
activations and the full interpolation path onto it. On a fixed two-dimensional grid in this plane, reconstruct
the corresponding `h1` activation, run it through the remaining network, and color each cell by predicted class.
Overlay the zero contours `l_a-l_c=0` and `l_b-l_c=0`, label the a/c and b/c boundaries, mark the path's entry
and exit from the c-decision region, and show the real a/b/c activations on the same axes.

The grid is a counterfactual slice through post-ReLU `h1` space. Report how much of the plotted grid has any
negative `h1` coordinate, because those points cannot be direct post-ReLU activations. Do not clamp grid points:
clamping would bend the plane and change the requested slice. Distinguish clearly between (1) entering a
c-decision region in this plane and (2) overlapping the distribution of real class-c activations. The former is
about the classifier; the latter is about activation-region similarity.

For aggregate findings, first show the 100-pair class composition and full-space distance results. Generate the
two planes for a frozen representative path: use the medoid path by `h1` distance among paths containing the
dominant stable c segment, not a hand-picked visually clean path. At minimum include every seed-0 stable
transition in an appendix/contact sheet and emphasize the cross-seed-stable `6→9→8` case in the main report.

## Required activation-region figures

For each analyzed stable transition:

1. **Prediction along interpolation.** Mean `d(alpha)` with its 100-pair spread, plus the predicted-class
   composition at every alpha. Show representative individual paths only after the aggregate.
2. **Real-class LDA view.** Show the held-out real a/b/c activation regions and the interpolation path on the two
   supervised discriminant axes, with the LDA fit restricted to real training activations.
3. **Margin-gradient/SVD decision slice.** Show the evaluated class-colored grid, projected real a/b/c
   activations, interpolation path, and explicitly labeled `a/c` and `b/c` zero-margin boundaries. Report the
   two-axis gradient-energy coverage and the fraction of off-ReLU-support grid points.
4. **Full-space distance view.** Across alpha, show the held-out-95th-percentile-normalized distance to both
   endpoint digits and each stable third digit, with a horizontal line at 1 and the third-class segments marked.

The scientific conclusion comes from the full `h1` coordinates and the 100-pair aggregates, not from a chosen
two-dimensional projection.

## Stages

- [x] **S1 — Census all digit transitions.** Reuse direction 12’s final seed-0 checkpoint and protocol; evaluate
  all 45 transitions with 100 fixed image pairs each; separately classify stable sub-plateaus and stable
  third-class predictions; save the transition matrix and all aggregate evidence.
- [x] **S2 — Confirm stability across seeds.** Repeat the frozen Stage-1 evaluation at the final converged
  direction-12 seed-1 and seed-2 checkpoints. Report agreements and disagreements without changing thresholds
  or pair selection.
- [x] **S3 — Collect real activations.** Save balanced training-reference and held-out-test `h1` activations for
  all ten digits and validate the distance calculation on held-out real images.
- [x] **S4b — Replace PCA with LDA and decision-aware planes.** Preserve the completed full-space comparison.
  For each seed-0 stable transition, select the representative path by the frozen medoid rule, generate both the
  real-class LDA plane and path-local margin-gradient/SVD decision slice, and report the required diagnostics.
  Put the complete set in an appendix/contact sheet and feature `6→9→8` in the main text.
- [x] **S5 — Controls and synthesis.** Check endpoint-predicted portions against endpoint activation regions and
  run fixed within-digit controls such as 6→6. Produce the three-way verdict for each analyzed transition.
- [x] **S6 — Finalize.** Curate `RESULTS.md` and `REPORT.md`, update `CHANGELOG.md`, verify manifests and figures,
  and write `STOP`.
- [x] **S7 — Finalize visualization revision.** Replace the old PCA figure and discussion with the verified LDA
  and margin-gradient/SVD outputs, update `RESULTS.md`, `REPORT.md`, and `CHANGELOG.md`, then restore `STOP`.

## Controls

- Held-out real images should usually be closest to their own digit’s activation region.
- Endpoint-predicted portions of a path should be close to the corresponding endpoint digit’s region.
- A fixed within-digit interpolation such as 6→6 should remain within that digit’s usual activation spread.
- Reversing a sampled path should reverse its alpha order without changing its category.

If these controls fail, report that the mean-and-variance activation summary is not reliable enough for the main
activation-region conclusion. Do not hide the failure by adding unrelated metrics.

## Optional later-layer follow-up

Only if stable third-class segments are predicted as z but are not close to real z activations at `h1`, repeat
the same reference-region and distance comparison at `h2` and `h3`. This asks whether similarity to real-z
activations appears only after later layers. Do not run it before the `h1` result is complete.

## Language rules

Use “activation region,” “third-class segment,” “stable third-class prediction,” and “sub-plateau” as defined
above. Do not use “manifold,” “submanifold,” “off-manifold,” “topology,” “phase transition,” or “circuit switch”
in the report. Do not infer an activation region from output predictions, an averaged curve, or a 2D plot alone.
Do not introduce clustering, density estimates, full Jacobian analyses, or unrelated plateau metrics. The only
new gradient analysis allowed here is the requested pair of path-local logit-margin gradients used to construct
the SVD visualization plane.

## Fallback if time runs short

Preserve the completed census and full-space results. At minimum, produce both LDA and margin-gradient/SVD views
for the cross-seed-stable `6→9→8` case using the frozen medoid path, including gradient-energy coverage,
off-ReLU-support grid fraction, real-class projections, and both zero-margin boundaries. State clearly that the
remaining seed-0 stable transitions have not yet received the revised visualization.

## Out of scope

- Retraining the model or changing its training schedule.
- Searching for a prettier image pair.
- Treating one path as evidence for a stable transition-level phenomenon.
- Copying or rewriting direction 12’s result artifacts.
- Proving that activation sets have any mathematical manifold structure.
- Adding measurements that do not answer the three research questions.

## On-track check

End every `JOURNAL.md` entry with:

`On track? <yes/no> — <stage, % done, blocker if any>`

## Current status

**Complete (S1–S7).** All stages are done, including the reopened visualization revision. The PCA view has
been removed and replaced by the two required planes, run on all 19 seed-0 stable transitions rather than a
subset. `STOP` is written and verified on disk; zero unaddressed feedback files exist.

- **S1/S2 census.** All 45 digit transitions × 100 fixed test-image pairs × 50 interpolation points at
  direction 12's seed-0/1/2 step-30000 checkpoints (4,500 paths per seed). Seed 0: **19 of 45**
  transitions stable third-class, **6** with a stable sub-plateau, **0** with a sub-plateau but no
  stable third-class prediction, 26 neither. Cross-seed the phenomenon replicates (12 and 18 of 45 at
  seeds 1 and 2) but the third digit's identity does not — 7/8 at seed 0, 1 at seed 1, 2/8 at seed 2;
  only **6→9 → z=8** is stable with the same z at all three seeds.
- **S3/S4/S5 activation regions at `h1`.** Ran on **all 19** stable transitions (no subset needed).
  Only **2.5%** of the 14,700 third-class segment points lie inside the real activation region of the
  digit they are predicted to be, and the median segment point is outside *every* digit's region
  (median nearest-digit ratio 1.23–2.09). Verdict **19 of 19 "prediction only"**, 0 match, 0 mixed.
  Controls C1–C4 all pass.
- **Later-layer follow-up.** Preregistered conditional follow-up, unlocked by the `h1` null and run:
  inside-region fraction is 2.5% (`h1`), 10.6% (`h2`), 0.2% (`h3`). The `h2` rise is not a match — a
  segment point is inside 5.8 of the 10 regions there, and the predicted digit is the *nearest* region
  for only 11.7% of points (chance is 12.5%).
- **S4b two 2-D views (replacing PCA).** Real-class LDA plane and path-local margin-gradient/SVD
  decision slice, on the frozen medoid path of **all 19** stable transitions. The supervised plane is
  *stricter* than the full-space test — **0.02%** of the 14,700 segment points inside the 2 s.d.
  real-*z* ellipse versus 2.5% in 200-d — so the null is not a dimensionality artifact. The slice
  shows a genuine third-digit *decision* region straddling the path (1.7%–37.7% of the window,
  two-axis gradient energy 96.2%–99.5%), while 100% of grid cells sit off the post-ReLU support.

Code: `experiments/s1_census.py`, `s1_analyze.py`, `s3_s4_regions.py`, `s4b_planes.py`,
`s6_later_layers.py`, `cvd_style.py` (shared colour-vision-deficiency-safe figure palette).
Numbers: `results/s1_census.npz`, `s1_classification.{json,csv}`, `s3_s4_regions.json`,
`s4b_planes.json`, `s6_later_layers.json`. Thirteen figures in `plots/`, all embedded as rendered
images in both RESULTS.md and REPORT.md; REPORT.md verified through the GitHub markdown API (16/16
display equations render, 0 KaTeX errors).

- **Figure accessibility (CLAUDE.md rule 13).** All figures are green-free with a second
  identity channel (hatch / linestyle / marker) on every series, and all captions and prose in both
  deliverables now name that channel instead of a colour. The three analysis scripts were re-run and
  every result JSON is identical to before, so no number changed.

The new visualizations corroborated the numerical activation-region verdict rather than contradicting it, so no
number changed. No model was retrained and no file in direction 12 was modified.

## Next step

None required — the plan is complete and `STOP` is on disk. If new feedback arrives, CLAUDE.md rule 11 applies:
delete `STOP`, address the feedback, re-write `STOP` only when clean.

Threads deliberately left out of scope, should anyone resume: (a) why each seed picks its own third digit, which
is a question about the between-cluster region of `h1` rather than about the prediction; (b) a region summary that
stays separable at `h2`, which would sharpen the later-layer result from "no evidence of a match" to a positive
test; (c) new, suggested by the S4b slices — each class's *decision* region in the between-cluster part of `h1` is
large (the third digit's covers a median 31.9% of the plotted window) while its *data* region is nowhere near, so
"does decision-region size predict which digit a seed defaults to?" is now well-posed.
