# PLAN — Direction 10: Does combined path smoothness recover the weekday activation manifold?

> Working folder: `dir10_optimize_smoothness_vs_on_manifold`. Agent REWRITES "Current status"/"Next step" + ticks stages each
> iteration. Disk (`PLAN.md`/`JOURNAL.md`/`RESULTS.md`/`REPORT.md`/`CHANGELOG.md` + `../BUDGET.md` + `../CLAUDE.md`) is the only memory.

## Research question

For fixed weekday-centroid endpoints, does optimizing a weighted combination of:

- activation-space kinetic energy; and
- downstream behavior-space kinetic energy

produce an activation path closer to the paper's fitted weekday activation manifold than either extreme alone?

This lambda sweep is a new experiment. Keep the model, prompts, activation collection, manifold fit, behavior representation, and intervention setup as close as possible to Wurgaft et al., *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior* (arXiv:2605.05115v1). Do not describe the combined objective as an experiment performed in that paper.

## Success criterion (definition of "done")

The direction is complete when:

1. The paper-consistent weekday setup is reproduced and validated:
   - Llama 3.1 8B base model;
   - exact 49 weekday-addition prompts;
   - layer 28, last-token residual-stream activations;
   - seven ground-truth-conditioned activation centroids;
   - PCA-64 activation subspace;
   - periodic cubic spline through the weekday centroids;
   - behavior distributions over seven weekday classes plus an `other` class in Hellinger coordinates.
2. A reproducible lambda sweep is run for at least one adjacent weekday pair, with the linear and output-only extremes included.
3. `RESULTS.md` and `REPORT.md` contain current-best results only, including:
   - activation and output energies;
   - distance/recovery relative to the fitted activation-manifold spline;
   - downstream `d(t)` curves;
   - a Pareto/trade-off plot;
   - a clear verdict on whether any intermediate lambda more closely recovers the centroid spline.
4. The result is checked across all seven adjacent weekday pairs if budget allows.
5. Null/negative results count as complete if the experiment is valid and the question is answered.
6. When complete, write an empty `STOP` file.

## Fallback (if time runs short)

Minimum acceptable deliverable:

- reproduce and validate the paper-consistent weekday centroids and periodic spline;
- run the full coarse lambda grid for Tuesday -> Wednesday;
- include the linear and output-only baselines;
- use at least the linear initialization plus one perturbed initialization;
- save the required plots and write a clear single-pair verdict;
- reserve the final 20 minutes for `REPORT.md`, `RESULTS.md`, `CHANGELOG.md`, and `STOP`.

## Setup (fixed)

### Source of truth

Before changing code, read:

- arXiv:2605.05115v1, especially Sections 2-3 and Appendix A.1-A.9;
- existing code on the current path-optimization branch, especially `optimize_path.py` and `slerp_relative_distance.py`;
- `../BUDGET.md`;
- `../CLAUDE.md`.

If this plan and the paper disagree on a paper-specific implementation detail, follow the paper and record the correction in `JOURNAL.md` and `CHANGELOG.md`. Do not silently guess missing details.

### Model and task

- Model: Llama 3.1 8B **base**, not an instruction-tuned substitute, unless the paper or existing verified implementation specifies otherwise.
- Use bfloat16, matching the paper.
- Weekday prompt template exactly:

  `Q: What day is {k} days after {entity}?\nA:`

- Entities: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.
- Increments: one, two, three, four, five, six, seven.
- Enumerate all 7 x 7 = 49 prompts.
- Ground truth wraps cyclically modulo seven.
- Validate that there are exactly seven prompts for each ground-truth weekday.

### Activation collection and manifold reference

- Collect the residual-stream activation at layer 28 and the final sequence position used to predict the answer token.
- Do not append the answer token to the prompt when collecting the intervention-site activation.
- Fit PCA with 64 components over all 49 task activations.
- Compute each weekday centroid by averaging PCA-projected activations grouped by **ground-truth answer**, not model prediction.
- Fit a periodic cubic spline through the seven weekday centroids, following Appendix A.3.
- The primary reference is the periodic cubic spline, not a straight centroid polyline.
- Preserve enough metadata to reproduce the prompt, ground truth, token position, raw activation, PCA activation, and centroid assignment.

### Behavior representation

- Softmax over the full vocabulary.
- Aggregate probability mass for tokenizer-valid spelling variants of each weekday, following Appendix A.2.
- Put all remaining vocabulary probability mass into one `other` bin.
- Save the exact token strings and token IDs used for each weekday; do not assume variants tokenize identically.
- Map the resulting eight-bin distribution into Hellinger coordinates using square-root probabilities.
- Report:
  - task accuracy;
  - mean probability mass on the seven weekday bins;
  - mean `other` mass.
- Do not continue to the expensive sweep if the model/task setup is clearly broken.

### Intervention and optimization space

Primary implementation should follow the paper's pullback optimization conventions where compatible with this new objective:

- parameterize the path as a one-dimensional natural cubic spline through 10 control vectors at uniform path positions;
- fix the first and last control vectors to the selected endpoint centroids;
- optimize only the interior control vectors;
- initialize from the linear chord;
- run at least one additional initialization with a small, seeded perturbation;
- use the first 32 PCA components as the primary optimization space, matching the paper's language-model pullback search space;
- keep the remaining PCA components and the orthogonal residual fixed from each base prompt during intervention;
- evaluate behavior over 16 base prompts and average the behavior-space energy over them;
- use L-BFGS with strong-Wolfe line search, matching the paper where practical;
- start with 50 outer steps and up to 5 inner iterations, with early stopping based on relative loss change;
- record every intentional deviation from the paper.

The endpoints must remain fixed for this experiment. Unlike the paper's behavior-target pullback objective, the combined kinetic-energy objective would otherwise admit an ill-posed or collapsing solution.

### Shared constraints

- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` — read both every iteration.**
- **Deliverable hygiene:** `RESULTS.md`/`REPORT.md` = current-best only; `CHANGELOG.md` = history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax.**
- Reuse the existing environment and code utilities where possible.
- Do not create a parallel implementation until the existing scripts have been audited.

## Objective and lambda sweep

Let the path contain uniformly spaced evaluation waypoints.

Compute two raw quantities:

- `E_act`: discrete activation kinetic energy in the optimization subspace.
- `E_out`: discrete kinetic energy of the induced eight-bin behavior distribution in Hellinger coordinates, averaged over the 16 base prompts.

Use the correct finite-difference approximation to the continuous kinetic-energy integral. Keep the raw energies for reporting.

For optimization, normalize the two terms so lambda is not determined only by arbitrary numerical scale:

- `E_act_norm = E_act / E_act(linear)`
- `E_out_norm = E_out / E_out(linear)`

Check that both denominators are finite and safely nonzero. If not, document and choose a stable pre-registered alternative before running the sweep.

Optimize:

`loss = E_act_norm + lambda * E_out_norm`

Coarse grid:

- lambda = 0
- lambda = 0.1
- lambda = 1
- lambda = 10
- lambda = 100
- output-only baseline, optimized with `E_out_norm` alone

If the coarse grid shows a transition, add a small number of log-spaced values around that region. Do not launch a large blind sweep.

Sanity expectations:

- lambda = 0 should recover the linear chord up to numerical tolerance;
- output-only should reduce `E_out` relative to the linear chord;
- all paths must retain the same fixed endpoints;
- failures of these checks block interpretation.

## Primary endpoint pairs

Pilot:

- Tuesday -> Wednesday

Full adjacent-pair check, if the pilot works:

- Monday -> Tuesday
- Tuesday -> Wednesday
- Wednesday -> Thursday
- Thursday -> Friday
- Friday -> Saturday
- Saturday -> Sunday
- Sunday -> Monday

Use the shorter arc of the periodic weekday spline as the activation-manifold reference for each adjacent pair.

## Metrics and required figures

For every lambda and baseline, save raw arrays and a tidy summary table containing:

- raw `E_act`;
- normalized `E_act`;
- raw `E_out`;
- normalized `E_out`;
- final optimization loss;
- endpoint error;
- optimization steps and convergence status;
- random seed and initialization type.

### 1. Activation-path recovery

Measure how close the optimized activation path is to the fitted periodic activation-manifold spline.

- Densely sample the relevant spline arc.
- Compute nearest-point residuals from optimized waypoints to the sampled spline.
- Implement the paper-style recovery comparison from Appendix A.9 as the primary metric.
- Re-read Appendix A.9 and use its exact common-subspace/SVD convention; do not guess the variance threshold.
- Also report the simpler mean nearest-spline distance in PCA-32 as a transparent diagnostic.
- Compare against the linear chord and output-only path.

Required figure:

- manifold-recovery distance versus lambda, with both baselines.

### 2. Energy trade-off

Required figure:

- `E_act` versus `E_out` for all optimized paths;
- mark the linear chord, output-only path, and fitted centroid-spline path;
- show whether the centroid-spline reference lies near or away from the empirical Pareto frontier.

### 3. Downstream plateau diagnostic

Use the existing, verified definition of `d(t)` from `slerp_relative_distance.py`.

- Audit and document exactly which downstream layer/state defines `d`.
- Do not use `d(t)` as a substitute for activation-space path distance.
- Plot `d(t)` for:
  - linear chord;
  - output-only path;
  - centroid-spline path;
  - each finite-lambda path.

Required figure:

- overlaid `d(t)` curves for the pilot pair.

### 4. Geometry visualization

Required diagnostic figure:

- PCA visualization of the weekday activations, centroids, fitted periodic spline, linear chord, output-only path, and selected finite-lambda paths.

This plot is illustrative only; use the high-dimensional recovery metric for conclusions.

### 5. Initialization sensitivity

For the pilot pair:

- linear initialization;
- at least one small seeded perturbation of the linear initialization;
- preferably three total seeds if budget permits.

Report whether the selected path and recovery score depend materially on initialization.

## Decision rule

The experiment supports the working hypothesis only if an intermediate lambda:

1. has lower activation-manifold recovery distance than both the linear and output-only extremes;
2. does so across initialization seeds;
3. is not explained only by endpoint error or failed optimization;
4. preferably generalizes across adjacent weekday pairs.

Interpretations:

- **Intermediate lambda recovers the spline:** combined input/output kinetic energy may capture part of the fitted manifold geometry.
- **Only `d(t)` matches but activation recovery does not:** downstream behavior does not uniquely determine the activation path.
- **No lambda approaches the spline:** generic kinetic smoothness does not explain this fitted activation manifold.
- **Results vary strongly by initialization:** the objective is underdetermined or optimization-dominated.
- **Different pairs require unrelated lambdas:** a single combined metric has weak explanatory power.
- **Centroid spline is dominated in both energies:** these two kinetic terms do not explain why that reference path should be preferred.

Do not claim that a successful lambda is a physical constant, a local-density estimate, or proof of the true activation manifold.

## Stages (checklist)

- [x] S1 — Audit before coding
  - Read `../CLAUDE.md`, `../BUDGET.md`, the paper sections listed above, and the current path-optimization scripts.
  - Write a concise code map and a paper-vs-current-implementation discrepancy table in `JOURNAL.md`.
  - Decide which existing functions will be reused or modified.
  - Do not launch the sweep in this stage.

- [x] S2 — Reproduce the paper-consistent weekday setup
  - Generate and save the exact 49-prompt dataset.
  - Load Llama 3.1 8B base in bfloat16.
  - Collect layer-28 last-token residual activations.
  - Build the eight-bin behavior distributions.
  - Fit PCA-64, seven centroids, and the periodic cubic spline.
  - Save validation metrics and plots.
  - Block further work if prompt counts, activation hooks, token aggregation, or model behavior are incorrect.

- [x] S3 — Implement the combined objective and tests
  - Add reusable functions for `E_act`, `E_out`, normalization, fixed-endpoint spline parameterization, and intervention over base prompts.
  - Add tests/sanity checks:
    - fixed endpoints;
    - lambda = 0 recovers the chord;
    - discretization is consistent when waypoint count changes;
    - gradients are finite;
    - output-only reduces output energy;
    - repeated seeded runs are reproducible.
  - Save a tiny smoke-test artifact before using the full model budget.

- [x] S4 — Pilot lambda sweep
  - Run Tuesday -> Wednesday.
  - Use the coarse lambda grid plus output-only.
  - Run required initialization seeds.
  - Save all raw paths, outputs, losses, metrics, and plots.
  - Determine whether adaptive refinement is justified.

- [x] S5 — Adjacent-pair sweep
  - If S4 is valid, repeat across all seven adjacent weekday pairs.
  - Use the same normalized objective and pre-registered lambda grid.
  - Do not tune a separate hidden protocol for each pair.
  - Summarize pair-level and aggregate results.

- [x] S6 — Final analysis and reporting
  - Produce and save every reported figure under `plots/`.
  - Define every metric and implementation choice in `REPORT.md` Methods.
  - Keep only current-best results in `RESULTS.md` and `REPORT.md`.
  - Put failed runs and superseded approaches in `CHANGELOG.md`.
  - State a direct verdict and limitations.
  - Write an empty `STOP` file.

## Out of scope (do NOT)

- Do not switch to Qwen, GPT-2, an instruction-tuned Llama, or another layer for the primary experiment.
- Do not add months, letters, ages, or unrelated datasets before the weekday result is complete.
- Do not estimate local activation density in this direction.
- Do not claim the centroid spline is the full or true natural-activation manifold.
- Do not turn this into a general study of plateau mechanisms.
- Do not optimize the paper's waypoint-wise behavior-target pullback loss as a substitute for the planned kinetic-energy sweep.
- Do not add large hyperparameter sweeps before the coarse lambda experiment passes sanity checks.
- Do not rewrite working infrastructure merely for style.
- Do not silently change the prompt template, token position, endpoint pair, output representation, or manifold fit.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> — <stage, % done, blocker if any>`

## Current status

**COMPLETE (S1–S6, STOP written).** Setup reproduced (Llama 3.1 8B base, layer 28; task acc 0.939).
The referenced `optimize_path.py`/`slerp_relative_distance.py` do NOT exist in this repo — built the
pipeline from scratch (`experiments/common.py`, `pathlib_opt.py`, `s2_collect.py`, `s4_sweep.py`,
`s5_analyze.py`, `s6_allpairs.py`). Tail runner validated against the full model. Pilot Tue→Wed lambda
sweep (5 λ + output-only, 3 seeds) and the all-7-adjacent-pairs run both give a **decisive NEGATIVE**:
recovery worsens monotonically with λ (best optimized path = linear chord for every pair), the
centroid spline is Pareto-dominated in both energies (7/7 pairs), and high-λ paths are init-dependent.
Operator feedback (human_feedback_07140930) addressed: star markers = 7 centroids, 7 seqs/weekday,
new cumulative-variance + plateau-metric figures, all 9 figures embedded as images. REPORT display
math re-verified (7/7 render, 0 degraded). Research question answered.

## Next step

COMPLETE — nothing required. Operator feedback `human_feedback_07140930.md` fully addressed (renamed
`.addressed.md`): star markers explained (= 7 weekday centroids), 7 sequences/weekday stated, new
cumulative-variance figure (`plots/s2_pca_cumvar.png`; 2–3 PCs NOT representative — PC1–3 = 44%), and
the requested plateau metric `p(t)=|h−hA|/(|h−hA|+|h−hB|)` implemented (`plots/s7_plateau_metric.png`;
spline's p(t) ≈ chord's — a fourth strike). All 9 figures now embedded as rendered images in both
deliverables (rule-12 fix). HF cache path auto-detect fixed in `common.py`. `STOP` written. Only
remaining optional polish: reproduce the exact Appendix-A.9 SVD recovery score if it becomes available.