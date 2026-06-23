# PLAN — Direction #6: What makes real activations real?

> Working folder: `dir6_jacobian_real_act_ddetector`. The agent REWRITES "Current status" and
> "Next step" and ticks the stage boxes every iteration. Disk (this file + JOURNAL.md + RESULTS.md
> + ../BUDGET.md) is the only memory.
>
> This file replaces the original scaffold TODOs with an implementation-ready research plan. The
> unresolved scaffold requirements are preserved as explicit execution rules: use GPT-2 small by
> default, stream data, respect `../BUDGET.md`, write results to RESULTS.md/REPORT.md, and create
> an empty `STOP` file only when the success criterion is met.

## Project thesis
Real residual-stream activations may have local statistical, geometric, and functional properties
that are not captured by matching their first two moments. Direction 6 asks whether such properties
exist, how well they generalize across hard synthetic/intervention-produced activations, and whether
they predict downstream validity. Direction 1 steering repair is a downstream causal validation, not
the primary scientific question.

The plan distinguishes five claims that must not be conflated:
- **Discrimination:** a score separates real activations from synthetic activations.
- **Generalization:** the score works on held-out negative generators and corruption families.
- **Prediction:** the score continuously predicts downstream degradation, beyond class labels.
- **Causality:** improving the score improves downstream behavior under external validation metrics.
- **Steering preservation:** repair can improve validity at matched steering strength or matched
  behavioral effect, without merely shrinking alpha or returning to the original activation.

## Current evidence from this folder and nearby work
- `JOURNAL.md` is empty except for the append-only format rule.
- `RESULTS.md` contains no Direction 6 empirical results yet.
- `experiments/` contains only `.gitkeep`; `results/` is empty. No Direction 6 code or cached results
  exist in this folder.
- `../BUDGET.md` is binding: 5 wall-clock hours for a full run, shared single RTX 3090, per-agent
  target of 0.45 VRAM fraction, about 14 GB RAM, 4 CPU threads, stream or memmap activation caches,
  halve batch/sample sizes after OOM, reserve final 20 minutes to finalize.
- Relevant nearby evidence from Direction 9: scalar plateau/Jacobian-style OOD signals were weak.
  A genuine output-Jacobian Frobenius metric did not beat baselines; MSP won random/shuffled-token OOD,
  and Mahalanobis/cupbearer RMD/QUE won code-domain shift. Therefore Direction 6 must not depend
  exclusively on Jacobian norm, and every proposed local-geometry score must be compared against
  strong norm/density/covariance baselines.
- Relevant nearby evidence from Direction 3: GPT-2 small layer-6 residual activations showed low local
  intrinsic-dimension estimates under some estimators, but reconstruction evidence was sensitive to a
  dominant activation dimension. Therefore manifold-style claims must control for dominant dimensions,
  preprocessing, token position, and density shortcuts.

## Research questions and hypotheses
- **RQ1:** Which measurable properties distinguish real residual-stream activations from synthetic,
  corrupted, or intervention-produced activations?
- **RQ2:** Which properties generalize to held-out negative families rather than exploiting generator
  shortcuts?
- **RQ3:** Do realness scores predict downstream validity continuously after controlling for norm,
  layer, token position, prompt identity, distance to original, and distance to the real activation set?
- **RQ4:** Can changing an activation to improve a realness score causally improve external downstream
  metrics?
- **RQ5:** In steering settings, can realness-improving correction preserve the intended intervention
  at matched achieved effect?

Hypotheses to evaluate:
- **H1:** Real activations possess local geometric or functional properties not captured by matching
  first two moments.
- **H2:** These properties distinguish real activations from multiple hard synthetic classes, not only
  obvious noise.
- **H3:** A learned or hand-designed realness score predicts downstream validity metrics such as KL,
  next-token loss, continuation stability, plateau behavior, and task accuracy.
- **H4:** Increasing the score of a corrupted or steered activation causally improves downstream
  behavior under metrics not directly optimized.
- **H5:** Such improvement can occur without erasing the intended semantic intervention.

## Scope and non-goals
In scope:
- GPT-2 small residual-stream activations by default, with layer sweep `{3, 6, 9}` and optional layer
  11 if budget allows.
- Last-token activations for the minimum viable study; mean-over-token and token-stratified variants
  as extensions.
- Real vs synthetic benchmark construction, baseline scores, richer local-geometry features, learned
  realness detectors, functional prediction, causal correction, and downstream steering validation.

Out of scope:
- Do not run expensive experiments unless a prior phase's decision gate justifies them.
- Do not treat a successful real-vs-fake classifier as sufficient evidence of activation validity.
- Do not make the plan depend exclusively on Jacobian norm or scalar plateau-ness.
- Do not drift into Direction 1 until Direction 6 produces a score with evidence for generalization
  and prediction.
- Do not install or downgrade `torch`, `torchvision`, `transformer_lens`, `cupbearer`, `jax`, or
  `flax`; use the existing environment. Pure-Python packages may be added with `--no-deps` only if
  necessary.

## Success criterion (definition of "done")
Direction 6 is complete when `RESULTS.md` and `REPORT.md` contain a clear verdict on whether any
activation-realness score satisfies the five-claim ladder:
1. beats strong statistical baselines on discrimination across easy and hard negatives,
2. retains useful AUROC/AUPRC/calibration on leave-one-corruption-family-out tests,
3. predicts downstream degradation with added value over norm, Mahalanobis, kNN/density, and
   distance-to-original controls,
4. improves external downstream metrics when optimized or projected in a causal intervention, and
5. preserves steering effect in the Direction 1 validation better than simply reducing alpha.

A null result is complete and acceptable if it identifies which notions of "realness" fail and why.
When done, write `REPORT.md`, update `RESULTS.md`, append the final JOURNAL entry, and create an
empty `STOP` file.

## Fallback (if time runs short)
Minimum acceptable deliverable: a self-contained benchmark on GPT-2 small layer 6 with real FineWeb
activations, at least four negative families (isotropic Gaussian, covariance-matched Gaussian,
shuffled-real, and norm-preserving perturbations), at least six scores (norm, Mahalanobis, PCA error,
kNN distance, Jacobian-Frobenius or directional sensitivity, and a small MLP detector), leave-one-family
held-out evaluation, and correlation with downstream KL/loss for corruption severity. Finalize
`RESULTS.md` + `REPORT.md` during the last 20 minutes, then create `STOP` only if the fallback question
is answered.

## Setup (fixed)
- Default model: GPT-2 small via HuggingFace `transformers`, using forward hooks. Do not require
  `transformer_lens`.
- Default data: stream FineWeb or an already available text sample for real activations; use local
  offline code text if a domain-shift control is needed.
- Hook point: residual stream after transformer blocks, named in code as `resid_post` or the closest
  HuggingFace equivalent. Record exact hook semantics in `experiments/ENV_NOTES.md`.
- Default layers: `{3, 6, 9}`; layer 6 is the minimum single-layer fallback.
- Default token position: last non-padding token; record absolute position and optionally token type.
- Cache activations as memmaps or sharded `.npy` under `results/acts/`; never keep large matrices in
  RAM.
- At process start: read `../BUDGET.md`, call `torch.cuda.set_per_process_memory_fraction(0.45)` if
  CUDA is available, use `torch.set_num_threads(2)`, and use DataLoader `num_workers <= 2`.

## Existing assets and code to reuse
- This folder currently has no reusable Direction 6 code.
- Reuse ideas and, if appropriate, copy small utilities from:
  - `../dir9_ood/experiments/plateau_v2.py` and `plateau_score.py` for output KL, perturbation, and
    Jacobian measurement patterns.
  - `../dir9_ood/experiments/extract_acts.py` for activation caching conventions.
  - `../dir9_ood/experiments/cupbearer_helpers.py` or the isolated `cupenv` only if already present
    and safe; do not alter the shared environment to install cupbearer.
  - `../dir3_manifold/experiments/id_estimate.py`, `pca_pr.py`, and AE scripts for PCA, kNN, local-ID,
    and reconstruction-score references.
- Any copied or adapted code must be placed under this folder's `experiments/` directory and logged in
  `JOURNAL.md`.

## Benchmark and data construction
Real positive examples:
- Sample prompts from FineWeb or the local text source used by prior directions.
- Split by prompt/document before activation extraction: `train`, `val`, `test`, and `ood_prompt_test`.
- Avoid sequence leakage: all tokens from one prompt/document belong to one split.
- Store metadata per activation: split, prompt/document ID, token index, absolute position, layer,
  norm, dtype, dataset source, model, hook name, and random seed.

Negative families, grouped by difficulty:
- **Easy negatives:** isotropic Gaussian; per-layer diagonal Gaussian; random directions at multiple
  radii; random-token activations if generated by forward pass.
- **Moment/density matched negatives:** full mean/covariance-matched Gaussian; shrinkage covariance
  Gaussian; per-coordinate quantile/marginal samples; PCA-subspace samples with matched spectrum.
- **Real-derived hard negatives:** shuffled coordinates within activation; permuted dimensions with
  fixed norm; convex interpolations between real activations; extrapolations beyond pairs of real
  activations; nearest-neighbor/local-density-preserving synthetic samples.
- **Perturbation hard negatives:** norm-preserving perturbations; random-direction perturbations;
  perturbations along data-tangent PCA directions; perturbations orthogonal to local PCA/kNN tangent;
  Jacobian-sensitive and Jacobian-insensitive directions.
- **Model/intervention negatives:** standard activation-steering outputs across alpha values; SAE
  reconstructions; SAE latent-combination samples; deep-autoencoder reconstructions and samples if
  built in Direction 3 style; outputs of any existing activation prior or meta-model if later found.
- **Adversarial negatives:** activations optimized to fool the detector while increasing downstream
  KL/loss or damaging continuation quality.

OOD split rule:
- Train detectors on real positives plus a selected training set of negative families.
- Hold out entire corruption families for OOD evaluation, e.g. train on Gaussian/shuffle/perturbation
  and test on SAE/steering/adversarial, then rotate.
- Report family-specific and macro-averaged results. Do not pool negatives in a way that hides failure
  on hard families.

Shortcut prevention:
- Match or stratify by layer, activation norm, token position, prompt source, dtype, and preprocessing.
- For intervention negatives derived from a real activation, retain `source_activation_id` and report
  distance to original.
- Use balanced splits and also report natural-prevalence calibration if a realistic mixture is defined.

## Candidate properties and models
For every score, record signal measured, cost per activation, confounds, differentiability, and whether
it can support Direction 1 correction.

Simple statistical baselines:
- `norm`: activation L2 norm. Cheap, differentiable; major shortcut risk.
- `mean_l2`: distance to layer mean. Cheap, differentiable; weak for anisotropic data.
- `mahalanobis`: shrinkage covariance Mahalanobis. Medium fit cost, cheap score, differentiable;
  must be well-powered and checked for dominant-dimension artifacts.
- `coordinate_quantile`: per-coordinate tail score. Cheap; assumes independent coordinates.
- `pca_recon`: PCA reconstruction error and tail energy. Medium; may track dominant PCs.
- `knn_distance` and `local_density`: kNN radius, local reachability, KDE approximations. Medium to
  high; non-differentiable unless approximated; memorization risk.

Local functional/geometric properties:
- `jac_frob`: Hutchinson-estimated Jacobian Frobenius norm of next-token log-probs w.r.t. activation.
  Medium/high; differentiable only with second-order work; known weak baseline from Direction 9.
- `jac_spectrum`: randomized SVD or power iteration for top singular values, stable rank, and spectral
  decay. Higher cost; richer than scalar norm.
- `directional_sensitivity`: KL/loss change under random, data-tangent, and normal directions. Medium;
  can reveal anisotropic local geometry.
- `local_lipschitz`: maximum observed output KL per activation-space radius. Medium/high; sensitive to
  radius choice.
- `plateau_width`: radius until output-distribution KL threshold. Medium; use as a feature, not a sole
  detector.
- `hvp_curvature`: Hessian-vector summaries for small validation subset only. Large; optional.
- `cross_layer_consistency`: replace activation, continue forward, compare downstream layer trajectory
  to real-activation neighborhoods. Medium/high; directly relevant to validity.

Learned detectors:
- `linear_probe`: logistic regression on raw or standardized activations plus controlled metadata-free
  features. Cheap; interpretable.
- `small_mlp`: low-capacity MLP with capacity sweep. Medium; must control overfitting and shortcuts.
- `contrastive_embedding`: real vs corruption contrastive representation with held-out families. Medium.
- `energy_score` or density-ratio estimator: train a scalar energy/logit; calibrate and test OOD.
- `feature_stack_model`: model over hand-designed features only, to test added value and ablations.

Generative/reconstruction scores:
- `ae_recon`: reconstruction error from shallow/deep autoencoder trained on real activations. Medium;
  differentiable; can support projection but risks learning identity.
- `denoising_error`: denoising autoencoder residual after controlled corruptions. Medium; differentiable.
- `activation_prior_likelihood`: if an activation prior/meta-model is discovered, use likelihood or score.
- `local_manifold_distance`: distance to local PCA/kNN tangent reconstruction. Medium; partially
  differentiable if approximated.

## Experiment phases

### Phase 0 — Repository and result audit
**Goal:** prevent duplicated work and lock down constraints.
- **D6-P0-audit**
  - Scientific question: what evidence, code, constraints, and TODOs already exist?
  - Inputs: `JOURNAL.md`, `PLAN.md`, `RESULTS.md`, `experiments/`, `results/`, `../BUDGET.md`, and
    nearby Direction 3/9 artifacts.
  - Method: list files, inspect result summaries, record exact reusable scripts and caveats.
  - Outputs: `JOURNAL.md` entry and `results/audit.json`.
  - Compute: tiny.
  - Decision gate: proceed only after confirming whether cached activations or previous detector outputs
    can be reused safely.

### Phase 1 — Dataset and benchmark construction
**Goal:** create a leakage-resistant activation-realness benchmark.
- **D6-P1-acts — Real activation cache**
  - Inputs: FineWeb/local text, GPT-2 small.
  - Layers: `{3, 6, 9}`; fallback layer 6.
  - Method: stream prompts, split by document, cache residual activations and metadata.
  - Baselines: none yet.
  - Metrics: counts by split/layer/token position, norm distributions, dtype consistency.
  - Controls: no prompt leakage, balanced token-position buckets, identical preprocessing.
  - Outputs: `results/acts/{split}__resid{layer}.npy`, `results/acts/metadata.parquet` or `.csv`,
    `results/dataset_summary.json`.
  - Compute: small.
- **D6-P1-negatives — Graduated negative suite**
  - Inputs: real activation cache, optional steering/SAE/AE sources when available.
  - Method: generate easy, moment-matched, real-derived, perturbation, model/intervention, and
    adversarial negatives according to the benchmark section.
  - Metrics: severity, norm, distance to original/nearest real, covariance match quality.
  - Controls: norm-matched variants for every family where feasible.
  - Outputs: `results/negatives/{family}__resid{layer}.npy`, `results/negatives/metadata.csv`,
    `results/negative_family_summary.csv`.
  - Compute: small to medium.
  - Decision gate: at least one hard negative family must be present before learned detector claims are
    allowed.

### Phase 2 — Baseline characterization
**Goal:** establish how far simple statistics and density baselines go.
- **D6-P2-baselines — Strong baseline table**
  - Positive/negative examples: all Phase 1 families.
  - Method: compute norm, mean distance, shrinkage Mahalanobis, coordinate quantile score, PCA
    reconstruction error, kNN/local density.
  - Metrics: AUROC, AUPRC, expected calibration error, Brier score, family macro-average, severity
    curves, compute cost per activation.
  - Controls: stratified and residualized analyses controlling for norm/layer/position/source.
  - Expected under H1/H2: moment-matched and local-density-preserving negatives should reduce baseline
    performance, leaving room for richer scores.
  - Failure interpretation: if baselines solve all hard negatives and predict degradation, Direction 6
    may conclude realness is mostly captured by density/covariance for this setup.
  - Outputs: `results/baseline_scores.csv`, `results/baseline_metrics.csv`, `results/plots/baselines/`.
  - Compute: small to medium.
  - Decision gate: future scores must show added value over the best baseline, not only over Gaussian
    negatives.

### Phase 3 — Jacobian and local-geometry study
**Goal:** test richer local functional geometry without relying on scalar Jacobian norm.
- **D6-P3-geometry — Directional and spectral local features**
  - Inputs: activation cache and negatives; GPT-2 continuation from hook point.
  - Method: compute `jac_frob`, top singular estimates, directional KL/loss sensitivity along random,
    data-tangent, normal, Jacobian-sensitive, and Jacobian-insensitive directions; estimate plateau width
    on a validation subset.
  - Baselines: all Phase 2 baselines plus Direction 9-style MSP/selfNLL controls.
  - Metrics: discrimination, leave-one-family-out, added predictive value in logistic/linear models,
    rank correlation with downstream degradation.
  - Controls: matched norm, matched Mahalanobis bins, matched perturbation radius, identical output
    metric orientation fixed before AUROC.
  - Expected under H1: directional/spectral summaries add value on hard negatives even when scalar
    Jacobian norm does not.
  - Failure interpretation: if local geometry adds no value after density controls, scalar and spectral
    local sensitivity are not the missing validity property.
  - Outputs: `results/geometry_features.csv`, `results/geometry_metrics.csv`,
    `results/plots/geometry/`.
  - Compute: medium; exact spectral/HVP variants are large and optional.
  - Decision gate: continue to learned detectors only if either baselines leave unsolved hard negatives
    or geometry features suggest useful residual signal.

### Phase 4 — Learned realness detector
**Goal:** test whether learned scores generalize rather than learning source-label shortcuts.
- **D6-P4-detectors — Controlled learned detectors**
  - Inputs: raw activations, feature stacks, train/val/test splits, negative family labels.
  - Models: logistic regression, linear probe, small MLP, feature-stack MLP, contrastive embedding, and
    energy/density-ratio score if time allows.
  - Method: train on selected negative families, validate on held-in families, evaluate on leave-one-
    corruption-family-out and cross-dataset splits.
  - Baselines: Phase 2/3 scores, family-prior classifier, metadata-only classifier as a leakage check.
  - Metrics: AUROC, AUPRC, calibration, macro-family performance, capacity curves, seed sensitivity,
    adversarial-fooling rate.
  - Controls: no layer/token/source metadata in primary detector; stratified batches; balanced classes;
    detector capacity sweep; nearest-neighbor memorization checks.
  - Expected under H2: useful detectors degrade gracefully on unseen families and beat feature baselines.
  - Failure interpretation: high held-in and poor held-out performance indicates generator memorization,
    not realness.
  - Outputs: `results/detector_metrics.csv`, `results/detector_calibration.csv`,
    `results/plots/detectors/`, model configs under `results/models/`.
  - Compute: small to medium.
  - Decision gate: no score is eligible for causal repair unless it has nontrivial held-out-family
    performance and calibration.

### Phase 5 — Functional validation
**Goal:** determine whether realness scores predict downstream model validity.
- **D6-P5-prediction — Score-to-degradation analysis**
  - Downstream metrics: next-token loss, KL from uncorrupted continuation distribution, continuation
    entropy, stable continuation behavior under repeated perturbations, task accuracy on small prompts,
    plateau width, and sensitivity to additional perturbation.
  - Method: create severity sweeps within each corruption family; compute realness scores and downstream
    metrics; fit regression/ranking models with controls.
  - Baselines: norm, Mahalanobis, kNN density, distance to original, corruption severity, alpha for
    steering.
  - Metrics: Spearman/Kendall rank correlation, partial correlation, incremental R^2/log-likelihood,
    calibration by severity bin, held-out-family prediction error.
  - Controls: matched distance and matched norm comparisons; prompt and layer fixed effects; family-held-
    out validation.
  - Expected under H3: realness score predicts degradation continuously beyond proximity and density.
  - Failure interpretation: if prediction vanishes after distance-to-original controls, the score is
    only a proximity proxy.
  - Outputs: `results/functional_metrics.csv`, `results/prediction_models.csv`,
    `results/plots/functional/`.
  - Compute: medium.
  - Decision gate: causal intervention may optimize only scores that predict at least one external
    downstream metric not used to train the score.

### Phase 6 — Causal intervention on corrupted activations
**Goal:** test whether increasing realness causally improves external behavior.
- **D6-P6-repair — Realness-improving projection/optimization**
  - Inputs: corrupted or synthetic activations, best eligible differentiable score(s), external metrics.
  - Method: optimize activation with penalty on distance moved; compare gradient ascent, denoising AE,
    local manifold projection, and score-matching style denoising if available.
  - Baselines/controls: optimize random frozen score; optimize norm/Mahalanobis only; shrink toward
    original activation by matched distance; random move of matched norm; detector-fooling adversarial
    negatives.
  - Metrics: score change, downstream KL/loss/coherence change, distance moved, nearest-real distance,
    external metric improvement not included in objective, reward-hacking diagnostics.
  - Controls: validation metrics excluded from optimization; multiple seeds; matched movement budget.
  - Expected under H4: realness-improving moves improve external metrics more than controls at the same
    movement distance.
  - Failure interpretation: improvement only under optimized detector metric is reward hacking.
  - Outputs: `results/repair_metrics.csv`, `results/repair_examples.jsonl`,
    `results/plots/repair/`.
  - Compute: medium to large; run only after Phase 5 gate.

### Phase 7 — Direction 1 steering application
**Goal:** use the best validated realness criterion as a downstream causal validation while keeping
Direction 6 primary.
- **D6-P7-steering — Preservation-aware steering correction**
  - Inputs: standard activation steering vectors and alpha sweeps, original activations, best repair
    method from Phase 6.
  - Method: apply steering, then constrained correction that improves realness while preserving a
    steering readout or matched achieved behavioral effect.
  - Baselines: uncorrected steering, smaller-alpha steering matched for downstream effect, projection
    toward original activation, random/density-only correction.
  - Metrics: Pareto frontier with x-axis steering strength or achieved effect and y-axis validity
    metric; steering retention; coherence/loss/task accuracy; distance from steered and original
    activations.
  - Controls: matched achieved steering effect, matched distance moved, alpha-shrinkage check, readout
    preservation constraint.
  - Expected under H5: correction improves validity at the same achieved steering effect and cannot be
    explained by moving back to the unsteered activation.
  - Failure interpretation: if correction only shrinks alpha or erases the readout, it is not useful
    for Direction 1 even if it improves generic validity.
  - Outputs: `results/steering_metrics.csv`, `results/plots/steering_pareto/`,
    `results/steering_examples.jsonl`.
  - Compute: large; optional unless previous gates succeed.

## Metrics
Primary metrics:
- AUROC and AUPRC for discrimination, reported by negative family and macro-averaged.
- Calibration: expected calibration error, Brier score, reliability curves.
- Leave-one-corruption-family-out AUROC/AUPRC/calibration.
- Rank correlation between realness score and downstream degradation.
- Added predictive value over norm, Mahalanobis, kNN/density, PCA error, and distance-to-original.
- Performance across corruption severity.
- Cross-layer and cross-dataset generalization; cross-model transfer only if budget permits.
- Compute cost per activation and total wall-clock.

Correction/steering metrics:
- Downstream KL, next-token loss/perplexity, continuation stability, task performance where feasible.
- Steering retention and achieved behavioral effect.
- Distance moved from corrupted/steered activation and distance to original activation.
- Pareto dominance over alpha-shrinkage and density-only controls.
- External validation metric improvement not optimized directly.

## Critical controls
- Activation norm as a shortcut: norm-match, stratify, and residualize.
- Distance to original activation: include as a baseline and match in repair analyses.
- Layer identity: evaluate per layer and prevent detectors from seeing layer labels unless explicitly
  testing cross-layer transfer.
- Token position: split or stratify by absolute position; avoid last-token-only overclaims.
- Prompt/source leakage: split by document and use metadata-only leakage tests.
- Dataset source mismatch: generate real and synthetic examples from the same prompt pool when possible.
- Dtype/preprocessing: save dtype and preprocessing metadata; compare standardized and raw features.
- Class imbalance: report balanced and prevalence-weighted metrics.
- Detector capacity: capacity sweeps and seed sensitivity.
- Train/test corruption overlap: leave out entire families and severity ranges.
- Nearest-neighbor memorization: kNN to train positives and duplicate-prompt checks.
- Correction magnitude: compare to shrinkage and random moves at matched distance.
- Detector overfitting/reward hacking: adversarial examples and external metrics not in objective.
- Multiple comparisons: pre-register primary scores per phase and report all tried variants.

## Decision gates
- **Gate 1:** benchmark quality. Do not train learned detectors until real activations and at least
  four negative families, including one hard family, are cached with metadata.
- **Gate 2:** baseline sufficiency. If simple baselines solve all hard negatives and predict degradation,
  write the null result rather than escalating to high-compute geometry.
- **Gate 3:** geometry value. Local-geometry features must add value over norm/Mahalanobis/kNN on at
  least one hard or held-out family to justify expensive spectral/HVP variants.
- **Gate 4:** detector generalization. A detector that only performs well on held-in families is not a
  successful realness detector.
- **Gate 5:** prediction. A score must predict external downstream degradation beyond distance/norm
  controls before it can be used for causal repair.
- **Gate 6:** causality. A repair method must improve metrics not optimized directly and beat matched
  shrinkage/random controls.
- **Gate 7:** steering preservation. Direction 1 validation succeeds only at matched achieved steering
  effect or under an explicit steering-readout preservation constraint.

## Expected artifacts and filenames
- `experiments/audit.py`
- `experiments/collect_acts.py`
- `experiments/make_negatives.py`
- `experiments/baselines.py`
- `experiments/geometry_features.py`
- `experiments/train_detectors.py`
- `experiments/functional_validation.py`
- `experiments/repair.py`
- `experiments/steering_validation.py`
- `experiments/make_plots.py`
- `experiments/ENV_NOTES.md`
- `results/audit.json`
- `results/dataset_summary.json`
- `results/negative_family_summary.csv`
- `results/baseline_metrics.csv`
- `results/geometry_metrics.csv`
- `results/detector_metrics.csv`
- `results/functional_metrics.csv`
- `results/repair_metrics.csv`
- `results/steering_metrics.csv`
- `results/plots/`
- `RESULTS.md` with headline tables and caveats
- `REPORT.md` with final verdict, limitations, and Direction 1 implications

## Compute-conscious execution order
1. Run Phase 0 audit and write `results/audit.json`.
2. Cache one-layer layer-6 activations and four cheap negative families.
3. Run Phase 2 baselines on layer 6.
4. Add layers 3 and 9 only after baseline pipeline works.
5. Add local-geometry features on a subsample; expand only if they add signal.
6. Train small learned detectors with leave-one-family-out evaluation.
7. Run functional validation on severity sweeps for the best 2-3 scores.
8. Attempt causal repair only for scores passing prediction gates.
9. Attempt steering validation only after causal repair beats controls.
10. Reserve final 20 minutes for `RESULTS.md`, `REPORT.md`, `JOURNAL.md`, and optional `STOP`.

## Risks and mitigations
- **Shortcut classifiers:** use metadata-only probes, stratification, matched controls, and held-out
  corruption families.
- **Jacobian overfocus:** treat scalar Jacobian norm as one baseline feature; prioritize added value
  over density/norm baselines.
- **Compute blowup:** use layer 6 first, small validation subsets for geometry, Hutchinson estimates,
  memmaps, and early decision gates.
- **Unstable conclusions from small N:** report uncertainty across seeds and bootstrap confidence
  intervals where practical.
- **Synthetic negatives too easy:** include moment-matched, local-density-preserving, steering, SAE/AE,
  and adversarial families before making strong claims.
- **Repair reward hacking:** evaluate on external metrics not optimized and use adversarial detector
  probes.
- **Steering erasure:** always compare at matched achieved steering effect and report distance to the
  original unsteered activation.

## Stages (checklist — update marks each iteration)
- [x] S1 — Audit repository, constraints, nearby evidence, and reusable code. (iter1)
- [x] S2 — Build real activation cache and metadata. (iter1 — REUSED dir3 acts_layer{3,6,9})
- [x] S3 — Build easy and hard negative families with held-out-family splits. (iter1: 4 families +
      LOFO in iter2; iter3 added interp/tangent_pert/orth_pert harder norm-matched families)
- [x] S4 — Run statistical/density baselines and decide whether richer scores are justified. (iter1)
- [x] S5 — Run local functional/geometric feature study beyond scalar Jacobian norm. (iter4:
      entropy/MSP/plateau-KL functional probe; uniquely catches interp ~0.61 where stats are chance)
- [x] S6 — Train and evaluate learned realness detectors under leave-one-family-out protocols.
      (iter2: logreg/MLP FAIL to generalize, LOFO macro 0.68/0.74 < kNN 0.913 → Gate 4 fails for
      discriminative detectors; realness = local density, a one-class property)
- [x] S7 — Test continuous prediction of downstream degradation. (iter6 + iter8 CORRECTION: genuinely
      in-context severity sweep via forward hook; functional plateau-KL/entropy predict true in-context
      KL beyond dist-to-orig, partial ρ +0.51/+0.35 noise +0.25/+0.19 interp; density scores are
      proximity proxies. Claim 3 = PARTIAL/YES for functional axis — verified in-context.)
- [x] S8 — Test causal realness-improving repair against matched controls. (iter7 + iter8 CORRECTION:
      NEGATIVE, in-context with per-descent move-matched controls — descent on Mahalanobis/plateau-KL
      makes downstream KL WORSE than corrupted & worse than its OWN matched random (3.33>1.99, 8.82>2.20);
      only oracle clean-direction recovers (0.009). Realness scores ≠ valid causal objectives.)
- [ ] S9 — Validate on steering while preserving achieved steering effect. (NOT done — blocked by S8's
      negative; a manifold/denoising-prior objective is needed first. Documented as future work.)
- [x] S10 — Finalize RESULTS.md, REPORT.md, JOURNAL.md, and STOP. (iter7; iter8 re-finalized after review)

## On-track check (required every iteration)
End each JOURNAL.md entry with one line:
`On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status (after iter8 — REVIEW CORRECTIONS applied, re-FINALIZED, STOP re-created)
Addressed external codex review (codex_review_20260623T024606Z.md). Its main valid finding: Phase-5/6
"in-context" continuation actually fed a SINGLE position through the late blocks (no prompt attention).
Re-ran BOTH phases genuinely in-context (forward hook overwrites only the last-token resid@L6 during a
FULL forward; sanity severity-0 KL = 0.0000 exactly): `context_validation_v2.py`, `causal_repair_v2.py`.
RESULT: both conclusions HOLD. (3) prediction — functional plateau-KL/entropy keep positive partial-ρ
beyond distance (+0.51/+0.35 noise, +0.25/+0.19 interp); raw Spearman even rises. (4) causality — both
descents still worse than their PER-DESCENT move-matched random controls (maha 3.33>1.99; func 8.82>2.20;
oracle 0.009), fixing review finding #2's move-mismatch. Also corrected wording for findings #3 (combined
LOFO uses held-out-label orientation — diagnostic), #4 (single-position probe phrasing), #5 (split is
contiguous+gap leakage-reduction, not verified doc-level). Verdict UNCHANGED: (1) YES multi-axis; (2)
PARTIAL; (3) PARTIAL/functional; (4) NO/reward-hacks; (5) untested. H1 supported. Project DONE.
Future work: manifold/denoising-prior causal objective, in-context DISCRIMINATION benchmark, cross-model,
token-position stratification (pos-0 sink confound), bootstrap CIs.

## (prior) Current status (after iter7 — FINALIZED, STOP created)
FULL 5-claim ladder addressed; REPORT.md + RESULTS.md complete & self-consistent. Verdict: (1)
discrimination YES via multi-axis score; (2) generalization PARTIAL; (3) prediction PARTIAL (functional
axis predicts downstream KL beyond distance); (4) causality NO — descent on realness scores reward-hacks
and worsens behavior (only oracle clean-direction recovers); (5) steering untested, blocked by (4). H1
supported. Project DONE (fallback + most of full criterion met, honest mixed/negative result). Future
work: manifold/denoising-prior causal objective, in-context discrimination benchmark, cross-model,
token-position stratification (pos-0 sink confound).

## (prior) Current status (after iter6)
Claims 1-3 of the 5-claim ladder are now evidenced and written up in REPORT.md (complete) + RESULTS.md.
Discrimination: multi-axis (density ⊕ two-sided covariance ⊕ functional), no single statistic. Learned
discriminative detectors memorize generators (LOFO < kNN). interp is anomalous by being too-CENTRAL
(two-sided Mahalanobis ~0.68), undetectable by one-sided/local scores. Prediction (Phase 5, context-
aware): functional plateau-KL/entropy predict true in-context downstream KL beyond dist-to-orig
(partial ρ up to +0.57); density scores are proximity proxies. Fallback deliverable fully met. Only
claims 4 (causal repair) and 5 (steering) remain — optional high-compute extensions.
Next substantive option: Phase 6 causal repair on the Phase-5 inject-and-continue machinery.

### (history) Iter1-3 done on GPT-2 layers {3,6,9} (real positives REUSED from dir3_manifold cache; pure-numpy).
Phase 2 (incl. cross-depth + harder negatives): kNN(density) ⊕ Mahalanobis(covariance) are
complementary and depth-stable; together they cover Gaussian/shuffle/orth/norm-pert families. BUT the
new `interp` family (convex combo of two reals, renormed) DEFEATS EVERY statistical baseline at every
layer (AUROC ≈ 0.44–0.54 ≈ chance) — decisive evidence (H1) that realness has structure beyond moments
and local density. Phase 4 LOFO: learned discriminative detectors do NOT generalize (< kNN). So the
generalizing statistical signal is one-class density, and it has a HARD CEILING at `interp`/`tangent_pert`.

## Next step (UPDATED iter4 — core verdict reached; remaining work is optional strengthening + REPORT)
Iter4 ran the functional probe: on `interp` (≈chance for all stats) entropy/plateau_kl reach ~0.61,
the only signal beating chance → functional component of realness confirmed (weak). RESULTS.md Headline
written. Remaining options before finalize: (a) incremental-value logistic over [kNN, Mahalanobis,
entropy, plateau_kl] to see if a combined score covers ALL families incl. interp; (b) write REPORT.md
with the full verdict, limitations (single-position probe, pooled positions incl. sink tokens, GPT-2
small only, no causal/steering phases), and Direction-1 implications. If <40 min left, do (b) only.

### (superseded) prior next step
Test a FUNCTIONAL probe exactly where statistics fail (interp, tangent_pert). Requires the model:
`pip install --no-deps transformers tokenizers safetensors regex huggingface_hub` (NEVER touch
torch/numpy/cuda; gpt2 is already in HF cache at /mars-vol/.cache/huggingface). Then write
`experiments/functional_features.py`: treat each cached activation as resid_post at the final
position, run GPT-2 blocks L+1..11 + ln_f + unembed -> logits; compute per-activation functional
features: output entropy, max-prob (MSP), and plateau-KL (output KL under a small random activation
perturbation, Hutchinson-style — reuse dir9 plateau_score.py). Compute AUROC real-vs-{interp,
tangent_pert,cov_gauss} for each functional feature and compare to the statistical ceiling. Cap VRAM
0.45, torch threads 2, batch small, GPU is A10. If functional features catch `interp` where all stats
fail → H1 confirmed via functional geometry and Phase 3/5 become the core contribution.

## Concrete TODO checklist for Claude
1. Read `../BUDGET.md`, set CUDA/CPU limits, and record environment details in `experiments/ENV_NOTES.md`.
2. Create `results/audit.json` summarizing this folder, nearby reusable scripts, and the fact that no
   Direction 6 empirical results exist yet.
3. Implement `experiments/collect_acts.py` for GPT-2 small residual activations at layer 6 first, then
   layers 3 and 9 after validation.
4. Implement `experiments/make_negatives.py` with isotropic Gaussian, covariance-matched Gaussian,
   shuffled-real, and norm-preserving perturbation families as the minimum viable suite.
5. Write metadata for every activation and negative example, including prompt ID, split, layer, norm,
   token position, family, severity, source activation ID, and seed.
6. Implement `experiments/baselines.py` for norm, mean distance, Mahalanobis, PCA error, quantile score,
   and kNN distance.
7. Produce `results/baseline_metrics.csv` with AUROC, AUPRC, calibration, family macro-average, and
   compute cost per activation.
8. Add at least one hard held-out family before training learned detectors.
9. Implement geometry features only after baselines are stable; include directional/spectral summaries,
   not only scalar Jacobian norm.
10. Train learned detectors with leave-one-corruption-family-out splits and capacity controls.
11. Run functional prediction analyses with downstream KL/loss and distance-to-original baselines.
12. Attempt causal repair and steering validation only after the relevant decision gates pass.
13. Update `RESULTS.md` after each completed phase; write `REPORT.md` and `STOP` only when the success
    criterion or fallback criterion is honestly met.
