# PLAN — Direction #8: Plateaus as a test for interpretability methods

> Working folder: `dir8_sae_act_pleateauness`. The agent rewrites "Current status" and
> "Next step" and ticks the stage boxes every iteration. Disk (this file + JOURNAL.md +
> RESULTS.md + ../BUDGET.md) is the only memory.
>
> Sole deliverable for this planning pass: this execution-ready plan. Do not edit
> `JOURNAL.md`, `RESULTS.md`, `experiments/`, or `results/` during planning.

## Success criterion (definition of "done")
Direction 8 is complete when `RESULTS.md` and `REPORT.md` contain a clear verdict on:

1. whether real activations and SAE reconstructions have stronger downstream-response
   plateaus than naive SAE-latent compositions;
2. whether any plateau gap survives matched controls for norm, distance to source,
   reconstruction error, latent sparsity, coefficient scale, decoder bias, direction
   family, local Jacobian/sensitivity, and encode-decode cycle error;
3. whether cycle-consistent or co-occurrence-aware synthetic SAE codes recover plateau
   behavior;
4. whether plateau metrics predict an independent downstream-validity target beyond
   simple baselines; and
5. whether the conclusion remains stable under one limited generalization check.

A null result is complete if it identifies which notion fails: synthetic-provenance
detection, off-distribution detection, local downstream-invalidity detection, or mere
local robustness. When done, Claude writes `REPORT.md`, updates `RESULTS.md`, appends a
final `JOURNAL.md` entry, and creates an empty `STOP` file.

## Fallback (if time runs short)
Minimum acceptable deliverable after implementation: one GPT-2 small layer-6 SAE study
on last-token FineWeb prompts with real activations, SAE reconstructions, naive latent
compositions, norm-matched random controls, and local perturbations. It must report the
primary plateau metric, norm, source distance, reconstruction error, sparsity,
coefficient scale, cycle error, local Jacobian/sensitivity, and output-KL validity on a
held-out split. It must include paired or matched analyses, not only group AUROC. Reserve
the final 20 minutes for `RESULTS.md`, `REPORT.md`, `JOURNAL.md`, and optional `STOP`.

## Setup (fixed)
- Default model: GPT-2 small via HuggingFace `transformers` and forward hooks.
- Default hook point for minimum viable experiments: `resid_post` after block 6 on the
  last non-padding token, matching the corrected Direction 6 in-context hook method.
- Default data: local FineWeb text/caches already present in neighboring directions.
- Read `../BUDGET.md` at every iteration. Respect: `torch.set_num_threads(2)`,
  DataLoader `num_workers <= 2`, CUDA memory fraction 0.45, memmap/shard large arrays,
  halve batch sizes on OOM, and reserve the final 20 minutes to finalize.
- Do not install or downgrade `torch`, `torchvision`, `transformer_lens`, `cupbearer`,
  `jax`, or `flax`. Current env has `numpy`, `torch`, `transformers`,
  `huggingface_hub`, `tokenizers`, and `safetensors`; it does not have `sae_lens`,
  `datasets`, `sklearn`, `scipy`, `pandas`, or `matplotlib`.
- Use pure `numpy`/`torch` implementations for metrics and statistics unless a missing
  pure-Python dependency can be added safely with `--no-deps`.

## 1. Repository status

### Established local facts
- This Direction 8 folder is a fresh scaffold:
  - `JOURNAL.md` only contains the append-only journal rule and required `On track?`
    final line.
  - `PLAN.md` was a TODO scaffold before this rewrite.
  - `RESULTS.md` is empty except for `Metrics` and `Headline` placeholders.
  - `experiments/` contains only `.gitkeep`; `results/` is empty.
- No Direction 8 code, local plateau implementation, activation-condition builder, SAE
  loader, result schema, plots, or previous artifacts exist yet.
- There are no supported local SAEs in this repo. No local SAE checkpoints were found in
  the workspace or HuggingFace cache. `sae_lens` is not installed.
- Nearby reusable infrastructure:
  - `../dir3_manifold/data/acts_layer{0,3,6,9,11}.npy`: GPT-2 small residual-stream
    activations, `[200000, 768]`, raw `float16`, all token positions pooled. Layers
    0/3/6/9 are interior block outputs; layer 11 in that cache is post-final-layernorm,
    not raw block-11 `resid_post`.
  - `../dir3_manifold/data/acts_layer6_pos.npy` and `acts_layer6_posidx.npy`: layer-6
    activations with absolute position metadata for token-position checks.
  - `../dir3_manifold/data/fineweb_texts.json`: FineWeb text cache used to collect the
    activations.
  - `../dir6_jacobian_real_act_detector/experiments/context_validation_v2.py`: canonical
    corrected in-context method. It runs a full GPT-2 forward and overwrites only the
    last-token `resid_post@L6` with a candidate activation via forward hook. Use this
    pattern for all downstream endpoint measurements.
  - `../dir6_jacobian_real_act_detector/experiments/mvp_benchmark.py`,
    `baselines_layers.py`, and `combined_score.py`: pure-numpy AUROC/AUPRC, shrinkage
    covariance, Mahalanobis, PCA reconstruction, kNN distance, norm-matched controls,
    tangent/orthogonal perturbations, and score-combination patterns.
  - `../dir9_ood/experiments/plateau_v2.py` and `plateau_score.py`: perturbation-score
    and Hutchinson Jacobian-Frobenius plateau proxies at configurable measurement
    points.
- Nearby results to treat as warnings, not Direction 8 findings:
  - Direction 9 found plateau-as-OOD weak against MSP/Mahalanobis/cupbearer baselines.
    Therefore this project must not validate plateau-ness by OOD or provenance
    detection alone.
  - Direction 6 found in-context `plateau_kl` predicted downstream KL beyond movement
    distance on noise/interpolation sweeps, but optimizing plateau flatness caused
    reward hacking. Therefore plateau may be predictive without being a safe causal
    objective.
  - Direction 6 also found norm, Mahalanobis, kNN, distance-to-source, and direction
    choices can dominate apparent "realness" signals.

### External SAE availability to verify before experiments
- Preferred target: a GPT-2 small residual-stream SAE matching the hook point, ideally a
  resid-post SAE for block 6. Public HuggingFace candidates include reformatted GPT-2
  small SAE repositories, but Claude must verify exact file paths and hook semantics
  with `huggingface_hub` before running.
- Concrete audit targets:
  - [`jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs`](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs):
    a public GPT-2 small resid-post SAE repository. Verify whether
    `blocks.6.hook_resid_post` or another layer needed here is present.
  - [`jbloom/GPT2-Small-SAEs-Reformatted/blocks.6.hook_resid_pre`](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted/tree/main/blocks.6.hook_resid_pre):
    public files include `cfg.json`, `sae_weights.safetensors`, and
    `sparsity.safetensors`, but this is `resid_pre`; use it only if the experiment
    hook is switched consistently to `resid_pre`.
- If a layer-6 `resid_post` SAE is unavailable, use a layer/hook that has an available
  SAE and change the model hook consistently. Do not mix `resid_pre` SAE decode vectors
  with `resid_post` activations.
- If direct checkpoint loading is impractical, Stage A may use an explicitly labeled
  "trained-local toy SAE" on cached layer-6 activations only as a fallback smoke test;
  it cannot support claims about public interpretability methods.

### Missing infrastructure
- SAE checkpoint loader and encode/decode functions.
- Prompt-aligned activation capture for real activations and SAE reconstructions.
- Activation-condition construction for edits, compositions, co-occurrence codes, and
  cycle-consistent codes.
- Plateau-curve sweeps with saved per-direction/per-epsilon curves.
- Baseline/covariate table for SAE-specific quantities.
- Statistical analysis and plotting scripts.

### Conflicts and constraints
- The requested output path includes a trailing `r`, but the actual repository directory
  is `dir8_sae_act_pleateauness`. Use the existing directory unless the operator creates
  the typo path explicitly.
- Current local activation caches pool token positions and lack prompt/document IDs.
  For paired downstream validity, Claude should re-capture a small prompt-aligned set
  rather than relying only on pooled cached rows.
- Layer 11 cached activations are post-final-layernorm; do not use them as raw block-11
  `resid_post` without re-collecting by hook.

## 2. Hypotheses and falsification criteria

| ID | Hypothesis | Supporting outcome | Falsifying or weakening outcome | Null interpretation |
|---|---|---|---|---|
| H1 | Real activations and accurate SAE reconstructions exhibit stronger plateau behavior than naive synthetic SAE-latent compositions. | Paired real and reconstruction conditions have larger primary plateau scores than naive compositions at matched layer/token/direction settings. | Gap disappears under matched norm/source-distance/sparsity/coefficient controls, or reconstructions are no stronger than naive compositions. | Plateau gap may reflect construction artifacts or SAE quality, not downstream validity. |
| H2 | Some or all of the difference is explained by simple covariates. | Norm, reconstruction error, source distance, sparsity, coefficient scale, decoder bias, or direction family strongly predicts plateau score and removes the group effect. | Group effect survives matched analyses and covariate adjustment. | If simple covariates explain the gap, plateau-ness is not an independent diagnostic in this setup. |
| H3 | Cycle-consistent or distribution-aware synthetic SAE codes exhibit stronger plateaus than independently composed codes. | Co-occurrence-aware or `encode(decode(z))≈z` codes move toward real/reconstruction plateau distributions at matched norm/sparsity/coefficient scale. | Improved synthetic codes remain as plateau-poor as independent compositions, or only improve because they are closer to a source real activation. | The missing ingredient may not be latent marginal realism; it may require higher-order model-computation compatibility or the metric is weak. |
| H4 | Plateau metrics predict an independently defined downstream-validity target beyond baseline metrics. | Plateau features improve held-out prediction of output KL, downstream hidden-state deviation, ranking stability, or edit collateral damage beyond baselines alone. | Added value vanishes after controlling for norm, source distance, reconstruction error, sparsity, cycle error, and local Jacobian/sensitivity. | Plateau-ness may detect provenance or robustness but not downstream validity. |
| H5 | Results depend substantially on model layer, token position, SAE, perturbation direction, or plateau definition. | Effect signs or sizes change across one additional layer/hook, token bucket, direction family, or metric definition. | Results stable across the limited robustness grid. | Dependence is not a failure by itself, but broad claims must be scoped to the stable subset. |

Project-level negative criterion: plateau-ness is not a useful independent diagnostic if
it loses predictive value after matching or controlling for norm, reconstruction error,
latent sparsity, coefficient scale, perturbation direction, cycle-consistency error, and
local sensitivity/Jacobian statistics.

## 3. Staged scope

- **Stage A — Basic reproduction, required.** Verify SAE availability, implement a small
  end-to-end pipeline, and reproduce the basic real/reconstruction versus naive synthetic
  plateau comparison on the smallest matching GPT-2 small layer/SAE setup.
- **Stage B — Core controls, required if Stage A works.** Test norm, source distance,
  cosine similarity, reconstruction error, sparsity, coefficient scale, decoder bias,
  cycle error, direction family, endpoint scale, and local sensitivity controls.
- **Stage C — Improved synthetic construction, required if Stage B leaves a plateau gap.**
  Add sparsity/coefficient-matched, co-occurrence-aware, and cycle-consistent codes.
- **Stage D — Independent downstream validity, required before any positive claim.**
  Evaluate whether plateau metrics predict output KL, hidden-state deviation, ranking
  stability, or edit collateral damage beyond baselines on held-out examples.
- **Stage E — Limited generalization, contingent.** Test one additional layer, SAE, or
  token-position setting only after Stages A-D produce interpretable artifacts. Do not run
  a broad benchmark until Stage B and D gates pass.

## 4. Activation conditions

| Condition | Construction | Pairing | Matched quantities | Comparison | Confounds | Status |
|---|---|---|---|---|---|---|
| Real activations | Capture real `resid_*` activation at the SAE hook point for selected prompts/tokens. | Source prompt/token is the unit. | Layer, token position, norm bins for matched analyses. | Reference plateau and downstream endpoint. | Positional sink tokens, prompt leakage, hook mismatch. | Required |
| SAE reconstructions | Encode real activation `x`, decode active code `z` to `x_hat`; include decoder bias exactly as loader defines it. | Paired with source `x`. | Same prompt/token; record `||x_hat||`, `||x_hat-x||`, cosine. | Tests whether the interpretability method preserves local validity. | Reconstruction error, decoder bias, SAE trained on different distribution. | Required |
| Small single-feature edits | For source code `z`, add/subtract one active or high-attribution feature coefficient by calibrated delta; decode. | Paired with source `x` and reconstruction. | Norm or edit-distance bins; same source. | Tests whether localized SAE edits preserve plateaus. | Edit semantic meaning may be unknown; coefficient scale. | Required if SAE loads |
| Multi-feature edits | Modify top-k active features or sampled feature sets at strengths `{0.25,0.5,1,2}` of source coefficient RMS; decode. | Paired with source. | Source distance, norm, sparsity, coefficient RMS. | Tests edit-strength degradation. | Strong edits may intentionally change behavior. | Required in Stage B |
| Naive independent compositions | Sample feature indices and coefficients independently from empirical marginals; decode from zero or decoder bias baseline. | Unpaired, with matched source assigned for controls. | Optional norm/sparsity matching post hoc. | Reproduces reported synthetic plateau weakness. | Unrealistic co-occurrence, coefficient mismatch, decoder bias dominance. | Required |
| Sparsity/coefficient-matched compositions | Sample k from empirical L0 distribution and coefficients from matched active-coefficient distribution; optionally match source L0 and coefficient RMS exactly. | Paired to source by target L0/RMS. | L0, coefficient RMS/quantiles, decoded norm. | Separates marginal mismatch from higher-order structure. | Still lacks co-occurrence. | Required Stage B/C |
| Co-occurrence-aware compositions | Sample active feature sets from empirical same-example sets, kNN in code space, pairwise co-occurrence model, or bootstrap real supports; resample coefficients conditionally. | Matched to source support size or nearest code-neighbor bucket. | L0, coefficient scale, support co-occurrence statistics. | Tests missing higher-order latent structure. | Needs enough encoded real examples; may copy real codes too closely. | Required if enough encoded cache |
| Cycle-consistent codes | Generate candidate z, decode to x, re-encode to z2, keep or project candidates with `||z2-z||/||z||` below validation quantile. | Unpaired or matched to source bins. | L0, coefficient scale, decoded norm, cycle error. | Tests encode-decode self-consistency as compatibility criterion. | Encoder nonlinearity and thresholding details. | Required Stage C |
| Moment-matched random activations | Sample Gaussian or PCA activations matching real mean/covariance or top PCs; no SAE decode. | Unpaired; optionally norm-matched to source. | Mean/covariance or PCA spectrum. | Baseline for non-SAE synthetic realism. | Known covariance shortcuts; no latent semantics. | Required control |
| Norm-matched random activations | Add isotropic noise to source or sample random direction, rescale to real/source norm. | Paired with source. | Norm exactly; optionally source distance. | Tests norm shortcut. | Direction distribution artificial. | Required control |
| Local perturbations around real activations | `x + r * ||x|| * d`, with controlled direction families and radius grid. | Paired with source. | Source, norm, radius, direction norm. | Calibrates plateau metric and downstream validity. | Not an interpretability method output. | Required |
| Existing generative activation baseline | None exists locally. If a prior activation generator is later found, add it with a separate condition label and config. | TBD | TBD | Tests non-SAE synthetic baseline. | Unknown training distribution. | Optional only |

## 5. Plateau metric

Primary metric: **normalized low-response plateau area** (`plateau_auc_low`), chosen
before condition-label analysis.

- Source model layer: minimum viable `GPT-2 small`, block-6 `resid_post` last
  non-padding token, unless SAE audit requires a different matching hook.
- Downstream endpoint: next-token log-probability distribution from a full in-context
  GPT-2 forward where only the target residual activation is overwritten by a forward
  hook. Do not use the superseded single-position late-block continuation as primary.
- Perturbation path: for activation `x`, direction `d` with `||d||_2=1`, evaluate
  `x(r)=x + r * ||x||_2 * d`.
- Primary direction distribution: isotropic Gaussian directions normalized to unit L2.
  It is primary because it is model-agnostic, available for all conditions, and matches
  prior Direction 6/9 random-direction probes. Direction-specific claims require Section
  7 robustness checks.
- Magnitude grid: `r = [0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]` for Stage A.
  Expand only if curves saturate too early or never cross threshold.
- Downstream-distance metric: `KL(p_x || p_x(r))`, where `p_x` is the output
  distribution from the unperturbed candidate activation in the same prompt context.
- Threshold selection: choose `tau` on a held-out calibration set of real activations
  only, before comparing conditions. Default `tau = median_real_KL(r=0.02)`. Record the
  chosen value and calibration split. Do not tune `tau` using group labels.
- Non-monotonic curves: compute `KL_monotone(r)` as the cumulative maximum over
  increasing `r` for threshold-derived quantities. Save raw curves too.
- Primary statistic:
  `plateau_auc_low = mean_d integral_r clip(1 - KL_monotone_d(r)/tau, 0, 1) / max(r)`.
  Higher means wider low-response plateau. This is continuous but still interpretable
  against a pre-registered threshold.
- Aggregation: first average over directions within activation, then analyze activation
  units. Do not treat directions or epsilon points as independent examples.
- Uncertainty: paired bootstrap over source prompts/activations for paired conditions;
  hierarchical bootstrap over prompt and seed for unpaired synthetic conditions.

Robustness metrics:
- crossing radius where `KL_monotone > tau`;
- initial log-KL slope over the first three nonzero radii;
- fixed-radius `plateau_kl` at `r=0.02` for comparability with Direction 6;
- Hutchinson `jacFrob` of log-probs with respect to activation;
- directional quantiles of crossing radius, especially 10th percentile.

## 6. Essential controls

Required baseline/covariate columns per activation:
- activation norm;
- distance to source real activation;
- cosine similarity to source real activation;
- SAE reconstruction error `||x_hat-x||` and relative error;
- latent sparsity L0 and active mass L1/L2;
- active coefficient mean, RMS, max, and quantiles;
- decoder bias contribution norm and fraction of decoded norm;
- encode-decode cycle error `||encode(decode(z))-z||` plus support overlap;
- token position and source prompt/document ID when available;
- model layer and hook point;
- downstream endpoint entropy/MSP/logit scale;
- perturbation direction family and radius grid;
- local sensitivity baseline: fixed-radius KL and/or `jacFrob`.

Analyses:
- Paired real vs reconstruction vs edit comparisons by source prompt/token.
- Matched synthetic comparisons by norm, L0, coefficient RMS, and decoded norm.
- Distance-matched comparisons for edited and random controls.
- Regression or residualized rank analysis with the above covariates.
- "Proxy test": report how much of plateau variance is explained by each simple
  baseline alone and whether adding plateau improves prediction of downstream validity.

## 7. Perturbation-direction robustness

Direction families to evaluate:
- isotropic random directions, primary;
- directions toward random real activations;
- directions toward moment-matched random activations;
- directions toward other real activations from the same norm/token bucket;
- SAE decoder-feature directions, both single decoder columns and sparse combinations;
- synthetic-minus-source directions for paired generated activations;
- all directions normalized to equal L2 perturbation norm.

Interpretation:
- If the plateau gap appears only for isotropic directions, it is a generic local
  sensitivity artifact, not evidence of SAE downstream validity.
- If the gap appears only along synthetic-minus-source directions, it may be detecting
  the edit/composition trajectory rather than local plateau geometry.
- If SAE decoder-feature directions reverse the conclusion, report the metric as
  direction-dependent and scope claims to the primary direction family.

## 8. Independent downstream-validity targets

Do not validate plateau-ness by predicting synthetic provenance alone. Required targets:

1. **Output KL validity.** For candidate activation `x_c` paired with source prompt,
   overwrite the activation in full context and compute `KL(p_real || p_candidate)`.
   Valid means KL is low relative to matched controls at the same source distance.
2. **Downstream hidden-state deviation.** Capture later-layer residuals and compare
   candidate-run hidden states to the real run using normalized L2/cosine at layers
   after the injection point. Valid means small deviation beyond the edited layer.
3. **Next-token ranking stability.** Compare top-k next-token sets/ranks between real
   and candidate runs. Valid means high overlap or low rank displacement when no
   semantic edit is intended.

Contingent target for SAE edits:
- **Semantic edit success vs collateral damage.** If edited features have known
  neuronpedia/dashboard semantics or interpretable prompts, define an edit readout
  before analysis. Valid edit means the intended readout changes while unrelated KL,
  hidden-state deviation, and ranking damage remain controlled.

Evaluation models:
- plateau metrics alone;
- baselines alone;
- plateau metrics plus baselines.

Use train/validation/test or cross-validation by source prompt. Threshold selection and
model fitting must not use the same examples used for final evaluation.

## 9. Statistical analysis

- Unit of analysis: source prompt/token activation. For unpaired synthetic examples,
  assign a source-match bin and use synthetic seed as a grouping factor.
- Paired structure: real, reconstruction, edits, local perturbations, and distance
  controls share a `source_activation_id`; analyze paired differences first.
- Minimum Stage A sample target: smoke test N=24 prompts, then N=200 prompts, 4 seeds,
  8 directions. Stage B target: N=400 prompts, 5 synthetic seeds, 16 directions.
  Increase only if bootstrap CIs are too wide.
- Precision target: report 95% CIs for median paired difference and Cliff's delta or
  standardized paired effect. Treat effect sizes below ~0.1 SD or AUROC differences
  below ~0.03 as not decision-grade at N around 200.
- Bootstrap: paired bootstrap over source activations; hierarchical bootstrap over
  source and generation seed for synthetic conditions.
- Repeated perturbation points: epsilon points and directions are repeated measures
  used to compute one activation-level plateau statistic, not independent samples.
- Covariate adjustment: use pure-numpy linear/logistic models or rank residualization;
  report matched analyses alongside regression.
- Multiple comparisons: one primary metric, one primary direction family, one primary
  layer/SAE. All other metrics/directions/layers are secondary; control false discovery
  by reporting all tried tests and emphasizing effect sizes/CIs over p-values.
- Exclusions: malformed prompts, tokenization failures, missing SAE encodes, NaN/Inf
  activations, failed hooks, out-of-memory aborted batches, and examples whose source
  and candidate hook semantics do not match. Save every exclusion record.
- Null and contradictory results must remain in `RESULTS.md` and `REPORT.md`.

## 10. Implementation design

Proposed modules; local Direction 8 has no code yet, so these are new files that should
reuse nearby utilities by copying small, attributed functions where appropriate.

| Path | Purpose | Inputs | Outputs | Reuse | Seed/cache behavior |
|---|---|---|---|---|---|
| `experiments/env_audit.py` | Record environment, repo status, available files, SAE availability. | repo paths, HF cache, optional HF repo IDs. | `results/audit.json`. | D6 audit style. | No randomness; save git commit and package versions. |
| `experiments/sae_io.py` | Load SAE config/weights, implement encode/decode/cycle error without requiring `sae_lens`. | `cfg.json`, `sae_weights.safetensors`, hook spec. | importable module plus `results/sae_audit.json`. | `safetensors`, `huggingface_hub`. | Cache downloaded files under HF cache; record revision. |
| `experiments/collect_prompt_acts.py` | Capture prompt-aligned real activations at matching hook. | FineWeb texts, tokenizer/model, layer/hook config. | `results/acts/prompt_acts.npz` with ids, masks, last positions, activations, metadata. | D3 collection and D6 hook capture. | Deterministic prompt selection seed; memmap/shard if large. |
| `experiments/build_conditions.py` | Construct activation conditions and baseline covariates. | prompt acts, SAE, config. | `results/conditions/*.npz`, `results/condition_metadata.csv`. | D6 negative-family builders. | Separate `sample_seed`, `edit_seed`, `condition_seed`; save source IDs. |
| `experiments/plateau_sweeps.py` | Run primary and robustness perturbation sweeps in full context. | conditions, prompts, model, config. | `results/plateau_curves/*.npz`, `results/plateau_metrics.csv`. | D6 `context_validation_v2.py`, D9 perturbation hooks. | Direction seeds saved; resumable per condition shard. |
| `experiments/baseline_metrics.py` | Compute norm, distance, cosine, Mahalanobis, kNN, SAE covariates, Jacobian/sensitivity. | conditions, train real activations, plateau curves. | `results/baseline_metrics.csv`. | D6 MVP/combined, D9 `jacfrob_score`. | Fit baselines on train split only; cache fit stats. |
| `experiments/validity_targets.py` | Compute output KL, hidden-state deviation, ranking stability, optional edit success. | conditions, prompts, model. | `results/validity_targets.csv`. | D6 in-context KL method. | Batch-resumable; save failed examples. |
| `experiments/analyze.py` | Group, matched, predictive, bootstrap, and decision-gate analyses. | plateau, baseline, validity CSVs. | `results/analysis_summary.json`, `results/tables/*.csv`. | Pure numpy statistics from D6. | Analysis seed for bootstraps. |
| `experiments/make_plots.py` | Optional lightweight plots if plotting library exists; otherwise write CSV-ready tables. | results tables. | `results/plots/` or skipped note. | none. | No required dependency. |

Configuration fields to standardize:
- model name/revision, SAE repo/revision/sae_id, layer, hook point;
- dataset source, prompt count, sequence length, token-selection rule;
- condition list, edit strengths, composition seeds;
- perturbation directions, radius grid, number of directions, threshold calibration split;
- baseline fit split, kNN subsample size, covariance shrinkage;
- output file prefix and resume policy.

## 11. Reproducibility and artifacts

Every run must record:
- model identifier and revision if available;
- SAE repository, revision, file paths, config, hook point, decoder bias convention;
- layer and hook semantics (`resid_pre`/`resid_post`);
- dataset source, prompt IDs, token indices, token-selection procedure;
- random seeds for prompt selection, condition construction, directions, bootstraps;
- complete configuration snapshot;
- git commit and dirty-state summary;
- dependency versions and CUDA device;
- result schemas and row counts;
- failed-example records with reason;
- per-run logs;
- aggregate CSV/JSON tables;
- figures or explicit note that plots were skipped due missing plotting dependency.

After each implemented experiment, Claude should append to `JOURNAL.md`:
- command/config;
- what artifacts were produced;
- what was learned without overclaiming;
- next step and gate status;
- final line exactly `On track? <yes/no> — <stage, % done, blocker>`.

`RESULTS.md` should receive only evidence: setup, tables, effect sizes/CIs, caveats,
and headline verdicts. Speculation and future-work brainstorming belong in `PLAN.md` or
`REPORT.md`, not in result tables.

## 12. Milestones and gates

| Milestone | Experiment | Deliverables | Pass criterion | Failure interpretation | Next after null | Cost |
|---|---|---|---|---|---|---|
| M0 | Validate plateau pipeline on real and random controls. | `audit.json`, smoke `plateau_metrics.csv`, sanity curves. | Real-vs-real severity 0 gives KL≈0; random/norm controls produce finite monotone-ish curves; hooks verified. | Pipeline or hook semantics invalid. | Fix hook/SAE alignment before any science. | Low |
| M1 | Reproduce real/reconstruction vs naive SAE synthetic difference. | condition table, plateau group table, paired effects. | Naive compositions have lower `plateau_auc_low` than real/reconstructions with nontrivial effect and CI excluding noise. | Motivating result not reproduced in this setup. | Report null; still run minimal covariate table to diagnose. | Low/Medium |
| M2 | Test core controls. | matched analyses and adjusted models. | Plateau gap survives norm, distance, sparsity, coefficient, cycle, Jacobian, and direction controls. | Gap is a proxy for simple covariates or direction choice. | Stop scaling; write negative/limited conclusion. | Medium |
| M3 | Test improved synthetic constructions. | co-occurrence/cycle-consistent condition results. | Improved codes recover plateau scores toward real/reconstruction at matched covariates. | Missing structure is not fixed by these code constraints, or metric is insensitive. | Proceed to validity prediction only if M2/M3 leave interpretable variation. | Medium |
| M4 | Predict independent downstream validity. | validity targets, predictive tables, held-out results. | Plateau + baselines beats baselines alone on held-out output KL/hidden deviation/ranking stability. | Plateau detects provenance/robustness but not downstream validity. | Report project-level negative if no added value. | Medium |
| M5 | Limited generalization. | one additional layer/SAE/token-position comparison. | Direction and effect size are similar enough to justify scoped claim. | Results are layer/SAE/token-specific. | Scope all claims; do not broad benchmark. | Medium/High |

## 13. Prioritized experiment queue

1. **Smoke test full pipeline.**
   - Purpose: verify model hook, SAE load or fallback, condition schema, and plateau metric.
   - Command: `python experiments/env_audit.py && python experiments/collect_prompt_acts.py --n_prompts 24 --layer 6 --seq_len 64 && python experiments/build_conditions.py --conditions real,norm_random --n_per_condition 24 && python experiments/plateau_sweeps.py --smoke --n_dirs 2 --r_grid 0,0.01,0.02`.
   - Inputs: FineWeb text cache, GPT-2, SAE audit if available.
   - Config: last-token, isotropic directions, `resid_post@6` unless SAE audit changes it.
   - Dependencies: `torch`, `transformers`, `numpy`; SAE optional for this smoke.
   - Expected outputs: `results/audit.json`, `results/acts/prompt_acts.npz`,
     `results/plateau_metrics.csv`.
   - Acceptance: severity-0 KL≈0, no NaNs, hook replacement changes logits for nonzero perturbation.
   - Compute: low.
   - Failure modes: wrong hook, missing prompt cache, CUDA OOM, SAE hook mismatch.

2. **SAE load and reconstruction audit.**
   - Purpose: establish supported SAE, layer/hook, encode/decode semantics, and reconstruction quality.
   - Command: `python experiments/sae_io.py --audit --preferred_layer 6 --preferred_hook resid_post`.
   - Inputs: verified SAE repo/files or local checkpoint.
   - Config: no `sae_lens` requirement; use direct `safetensors` load.
   - Dependencies: smoke test complete.
   - Expected outputs: `results/sae_audit.json`, reconstruction-error summary.
   - Acceptance: encode/decode runs on N>=24 real activations; hook semantics recorded.
   - Compute: low.
   - Failure modes: no matching SAE, incompatible checkpoint format, memory too large.

3. **Stage A reproduction.**
   - Purpose: compare real, SAE reconstructions, and naive independent compositions.
   - Command: `python experiments/build_conditions.py --conditions real,recon,naive_latent --n_prompts 200 --n_seeds 3 && python experiments/plateau_sweeps.py --n_dirs 8 && python experiments/analyze.py --stage A`.
   - Inputs: prompt acts, SAE.
   - Config: primary metric, isotropic directions, calibration `tau` from real train split.
   - Dependencies: experiments 1-2.
   - Expected outputs: `results/plateau_metrics.csv`, `results/tables/stage_A_groups.csv`.
   - Acceptance: interpretable group table with CIs and saved raw curves.
   - Compute: medium.
   - Failure modes: no plateau gap, noisy curves, synthetic codes decode to extreme norms.

4. **Core covariate controls.**
   - Purpose: test H2.
   - Command: `python experiments/baseline_metrics.py && python experiments/analyze.py --stage B --matched norm,l0,coef_rms,source_distance,cycle,jacfrob`.
   - Inputs: Stage A metrics and conditions.
   - Config: exact/nearest-neighbor matching plus rank residualization.
   - Dependencies: Stage A.
   - Expected outputs: `results/tables/stage_B_controls.csv`, `analysis_summary.json`.
   - Acceptance: each required confound has either a matched analysis or explicit missing-data note.
   - Compute: low/medium.
   - Failure modes: insufficient overlap for matching; report lack of common support.

5. **Direction-family robustness.**
   - Purpose: test whether conclusions depend on perturbation directions.
   - Command: `python experiments/plateau_sweeps.py --direction_families isotropic,real_target,moment_target,sae_decoder,synthetic_minus_source --n_dirs 8 && python experiments/analyze.py --stage B_direction`.
   - Inputs: Stage A/B conditions.
   - Config: equal-norm directions.
   - Dependencies: core conditions.
   - Expected outputs: `results/tables/direction_robustness.csv`.
   - Acceptance: direction-specific effect table with CIs.
   - Compute: medium.
   - Failure modes: SAE decoder directions unavailable for fallback SAE; mark missing.

6. **Improved synthetic codes.**
   - Purpose: test H3.
   - Command: `python experiments/build_conditions.py --conditions matched_latent,cooccurrence,cycle_consistent --n_prompts 400 --n_seeds 5 && python experiments/plateau_sweeps.py --n_dirs 16 && python experiments/analyze.py --stage C`.
   - Inputs: encoded real-code cache.
   - Config: L0/coefficient matching, cycle-error threshold from real validation quantile.
   - Dependencies: Stage B leaves interpretable gap and enough encoded examples.
   - Expected outputs: `results/tables/stage_C_synthetic.csv`.
   - Acceptance: co-occurrence and cycle conditions compared at matched norm/L0/RMS.
   - Compute: medium.
   - Failure modes: co-occurrence model copies real codes; cycle filter rejects nearly all candidates.

7. **Independent downstream-validity prediction.**
   - Purpose: test H4.
   - Command: `python experiments/validity_targets.py --targets output_kl,hidden_deviation,rank_stability && python experiments/analyze.py --stage D --heldout_by prompt`.
   - Inputs: all condition activations, prompt contexts, plateau/baseline metrics.
   - Config: train/validation/test by source prompt; baselines-only vs plateau-only vs combined.
   - Dependencies: Stage B metrics complete.
   - Expected outputs: `results/validity_targets.csv`, `results/tables/stage_D_prediction.csv`.
   - Acceptance: held-out added-value table with CIs or clear null.
   - Compute: medium.
   - Failure modes: target dominated by source distance; plateau adds no value.

8. **Limited generalization.**
   - Purpose: test H5 without broad benchmarking.
   - Command: `python experiments/collect_prompt_acts.py --layer <alt> ... && python experiments/analyze.py --stage E`.
   - Inputs: one additional matching SAE/layer or token-position bucket.
   - Config: choose only one axis based on availability: adjacent layer with SAE, or layer-6 token-position bucket.
   - Dependencies: Stage D positive or scientifically ambiguous.
   - Expected outputs: `results/tables/stage_E_generalization.csv`.
   - Acceptance: scoped comparison to primary setup.
   - Compute: medium/high.
   - Failure modes: no alternate SAE, too little token-position overlap.

9. **Finalize report.**
   - Purpose: make the result durable.
   - Command: manual writeup after all runnable experiments: update `RESULTS.md`, write `REPORT.md`, append `JOURNAL.md`, create `STOP` only if criterion met.
   - Inputs: all tables and summaries.
   - Config: include nulls and caveats.
   - Dependencies: at least fallback complete.
   - Expected outputs: final docs and `STOP`.
   - Acceptance: results answer provenance vs off-manifold vs downstream-validity distinctions.
   - Compute: low.
   - Failure modes: insufficient artifacts; finalize fallback instead.

## 14. Decision table

| Outcome | Interpretation for Direction 8 thesis |
|---|---|
| Plateau gap disappears after norm matching. | Plateau score is mostly a norm proxy in this setup; not an independent validity diagnostic. |
| Gap disappears after source-distance or reconstruction-error matching. | Plateau reflects closeness to the real activation, not special downstream compatibility. |
| Gap disappears after sparsity/coefficient matching. | Naive synthetic weakness was caused by marginal latent-code mismatch. |
| Gap disappears after Jacobian/fixed-radius sensitivity adjustment. | Primary metric adds little beyond local sensitivity; report as robustness, not interpretability validity. |
| Gap depends strongly on perturbation direction. | Plateau-ness is direction-family-specific; broad "activation validity" claims are unsupported. |
| Cycle-consistent synthetic activations recover plateaus. | Encode-decode compatibility is a plausible ingredient of downstream-valid synthetic activations. |
| Co-occurrence-aware codes recover plateaus. | Higher-order SAE latent structure matters beyond independent feature marginals. |
| Plateau predicts synthetic provenance but not downstream validity. | It may be a provenance or distribution detector, not a test of interpretability-method validity. |
| Plateau predicts downstream validity beyond all baselines. | Strong support that plateau-ness measures a useful part of local downstream validity. |
| Conclusions vary sharply by layer or SAE. | Thesis must be scoped; run no broad benchmark until mechanism is understood. |
| SAE reconstructions retain plateaus only when reconstruction error is tiny. | Plateau preservation may be a reconstruction-quality consequence, not evidence about interpretability. |
| Real and all synthetic conditions have similar plateaus. | The motivating result fails here; either synthetic construction is better than expected or plateau metric lacks power. |

## Stages (checklist — update marks each iteration)
- [x] S1 — Audit environment, repo status, nearby reusable code, and SAE availability.
- [x] S2 — Implement/load SAE and prompt-aligned activation capture.
- [x] S3 — Smoke-test full-context plateau pipeline on real and random controls.
- [x] S4 — Stage A: real/reconstruction/naive synthetic reproduction.
- [x] S5 — Stage B: distance-to-source matched control + sparsity/coef matching + held-out τ
      (direction-family control still pending within Stage B).
- [x] S6 — Stage C: improved synthetic latent constructions. (H3 negative: co-occurrence &
      cycle-consistent codes stay BELOW random; only genuine real-derived codes exceed it)
- [x] S7 — Stage D: independent downstream-validity prediction. (H4 negative beyond local sensitivity)
- [ ] S8 — Stage E: one limited generalization check. (skipped — null complete & named)
- [x] S9 — Finalize `RESULTS.md`, `REPORT.md`, `JOURNAL.md`, and `STOP`.

## Out of scope (do NOT)
- Do not claim plateau-ness is a manifold detector unless controls support that specific interpretation.
- Do not validate solely on real-vs-synthetic provenance labels.
- Do not run broad multi-model or multi-SAE benchmarks before Stage B and D gates pass.
- Do not use the superseded single-position late-block continuation as the primary downstream endpoint.
- Do not optimize activations to maximize plateau score as a repair objective in this project; Direction 6 found reward-hacking risk.
- Do not install heavyweight framework packages or downgrade the shared torch/CUDA stack.

## On-track check (required every iteration)
End each `JOURNAL.md` entry with one line:
`On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
**FINALIZED — project-level null complete, direction-robust AND improved-code-robust; STOP created.**
Stages A (M1) + B (M2) + B-dir + C (M3) + D (M4) done & decisive. SAE = jbloom resid_pre@6
(d_sae=24576), hook = block-5 output. Scripts: `experiments/smoke_plateau.py` (A),
`stageB_distance.py` (B), `stageB_directions.py` (B-dir), `stageC_synthetic.py` (C),
`stageD_validity.py` (D).
- Stage A (N=200, 8 dirs): plateau_auc_low real 0.200, recon 0.162, naive 0.066, norm_rand
  0.035; all paired gaps exclude 0; NOT a norm artifact (Spearman(plateau,norm)=+0.06).
- Stage B (N_eval=100, 6 dirs, held-out τ=1.33e-4): iso_displace random-displacement reference
  0.184/0.173/0.128/0.078 at dist 15/30/60/120. Distance-matched residual: recon −0.016, naive
  −0.058, sparse_match −0.063. **No SAE-decoded condition ABOVE the random curve.** H2 negative.
- Stage B-dir (N=200, N_eval=100, 8 dirs; `stageB_directions.py` full config): distance-matched
  residual under 3 perturbation-direction families. recon ≈0 on all; naive/sparse BELOW random
  everywhere — iso −0.061/−0.062, sae_single −0.066/−0.062, sae_sparse −0.077/−0.071. **Direction-
  family robust**; deficit slightly LARGER along SAE-decoder dirs. Closes the biggest open caveat.
- Stage D (1400 rows, split by source): held-out R² for log10 output_kl — baseline(dist,norm)
  0.795, +plateau 0.869, baseline+locsens 0.873, all 0.878. **ΔR² plateau beyond {dist,norm}
  =+0.073 (partial ρ −0.65) but beyond {dist,norm,locsens} =+0.005 (partial ρ −0.16).** Plateau's
  validity prediction = local sensitivity; marginal ρ: plateau −0.85 ≈ locsens +0.84.
- Stage C (N=200, N_eval=100, 6 dirs; `stageC_synthetic.py`): H3 — improved synthetic codes.
  Distance-matched residual ρ_c: cooc −0.044 [−0.049,−0.036], cycle_consistent −0.043
  [−0.049,−0.040] both BELOW random (only marginally above naive −0.054); cooc_full (genuine
  real-derived code) **+0.043** [+0.035,+0.056] ABOVE. **H3 negative** for constructible codes:
  neither support co-occurrence nor encode–decode cycle-consistency recovers plateau; the missing
  ingredient is real-activation manifold membership. cooc_full above = positive control Stage B
  lacked. Cycle filter: real-code p75 cycle err τ_cyc=0.342, naive pass rate 0.56%.
- **Project verdict (null):** plateau-ness = closeness-to-real (B) + local robustness (D), NOT
  an SAE interpretability-validity diagnostic, and NOT recoverable by improved SAE codes (C). Of
  {provenance, OOD, downstream-invalidity, mere local robustness} → **mere local robustness** +
  distance-to-real / real-manifold membership. Matches D9 & D6.
Env note: transformers/tokenizers/safetensors/huggingface_hub pip-installed `--no-deps`
(tokenizers 0.22.2); torch/CUDA untouched. matplotlib present.

## Next step
DONE. RESULTS.md + REPORT.md now include Stage C (improved synthetic codes, H3 negative) curated
to current-best; CHANGELOG + JOURNAL appended; STOP re-created. Only Stage E (one alternate-layer
generalization) remains, expected to scope not overturn the local-sensitivity null (which is now
direction-family robust AND improved-code robust).
