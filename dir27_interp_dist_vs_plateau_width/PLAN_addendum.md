# PLAN amendment â Direction 27: paired interpolation-path control

> Working folder: `dir27_interp_dist_vs_plateau_width`.
> Merge this amendment into the current local PLAN; do not replace the whole file,
> reset completed stages, or discard existing results. No new direction.
> The latest online PLAN was not retrievable when this amendment was drafted.
> Read the actual local PLAN, code, and run manifests before applying it.

## Goal and success criterion

Keep the block-0 `resid_post` **linear-interpolation experiment as the primary
analysis**. Add a small paired control using Matthew's interpolation function to
ask whether the distanceâtransition-width relationship depends on the choice of
path. A result showing agreement, disagreement, or insufficient coverage is a
valid outcome. Do not change the setup to recover the previous conclusion.

This control changes the interpolation path, not the intervention site. It is
not a full reproduction of Matthew's experiments.

## Fixed setup and scope

- Start with **GPT-2 Large only**, using the current `The house was big` setup.
  Keep the exact checkpoint, model-loading transformations, precision, token IDs,
  prefix/BOS handling, attention masks, interpolation grid, downstream readouts,
  width extraction, and flagging rules from the block-0 linear run.
- Obtain endpoint vectors `a` and `b` from the **last-position output of the first
  complete transformer block**, equivalent to `blocks.0.hook_resid_post`.
  Inject there; leave earlier positions and downstream computations unchanged.
- Use **256 partner token IDs**, sampled without replacement from the current
  sweep's eligible vocabulary with seed 0, independently of observed widths.
  Save the selected IDs before running the new path. Use the identical pairs
  for both methods; do not substitute new pairs after seeing their outcomes.
- Add up to six already reported representative pairs for illustration. Record
  their selection separately; do not pool them into random-sample summaries.
- Reuse saved linear results only when token IDs, intervention site, model,
  precision, grid, and readouts match. Otherwise rerun this subset, not the full
  vocabulary. Never substitute old embedding-run results.
- Do not automatically expand this control to all five models, the full
  vocabulary, or France/dir29. Report the pilot before proposing an expansion.
- Read `../BUDGET.md` and `../CLAUDE.md`. Reuse the current inference environment;
  do not install or upgrade model libraries.

## Two paths, identical endpoints

1. `linear`: `r(t) = (1 - t) * a + t * b` â retain the existing default.
2. `matthew_slerp_rescale`: port or reuse the reference `slerp_rescale(a, b, t)`
   from Matthew's `scripts/utils.py`; add it as an explicit opt-in method.

Reference:
https://github.com/MShinkle/activation_plateau_mechanisms/blob/main/scripts/utils.py

Record the source commit and retain an attributable copy of the reference
function for equivalence tests. The inspected function computes its angle using
`acos(abs(dot_product))`, not the usual signed-dot SLERP angle. **Do not silently
replace it with standard SLERP or silently remove the absolute value.** Compare
the port with the pinned reference on nondegenerate test vectors.

Check endpoint cosine similarities and realized vector norms. Flag negative
cosines: for these pairs the published implementation need not maintain the
nominal linearly interpolated norm. Record zero-norm or singular-angle cases and
any numerical handling explicitly. Do not quietly repair the reference and
still label it an exact implementation match.

## Measurements: separate path choice from normalization

For each pair and method, record:

```text
D_endpoint = ||b - a||_2
L_path = sum_i ||r(t[i+1]) - r(t[i])||_2
```

`D_endpoint` is identical for the two methods. It is the block-0 endpoint
separation, not token-embedding distance. `L_path` is the sampled length of the
**injected** path, not downstream trajectory length. It equals endpoint distance
for linear interpolation, up to numerical precision; do not assume this equality
for Matthew's path. Compute it from a dense path-only grid without additional
model forwards, checking convergence when the grid is doubled.

**Primary comparison:** use the current dir27 readout normalization for both
paths. Inspect and record its actual implementation first. The preceding plan
retained the fixed-endpoint-distance definition, for downstream readout `x(t)`:

```text
y_fixed(t) = ||x(t) - x_A||_2 / ||x_B - x_A||_2
```

If the current local implementation differs, report the discrepancy rather than
silently changing the baseline. Hold the verified baseline normalization fixed
across the primary path comparison.

**Secondary, offline comparison:** also compute Matthew's relative-to-both-
endpoints curve for both paths, without new model forwards:

```text
y_relative(t) = ||x(t) - x_A||_2 /
                (||x(t) - x_A||_2 + ||x(t) - x_B||_2)
```

Save the raw downstream readouts, or at least both per-point endpoint distances
and the endpoint separation. Keep the current first-crossing width extraction
and flagging rules. Do not introduce isotonic fitting, clipping, or new plateau
thresholds. Record undefined readout denominators as invalid, not as zero width.

For representative pairs, also show progress of the injected path under the
same distance normalization. A curved input path can itself yield a nonlinear
progress curve; do not treat `y(t) != t` under that path as sufficient evidence
of nonlinear model processing.

## Validation before the pilot

- Verify the actual hook and that only the final position is patched.
- Check each method's injected endpoints against `a` and `b`.
- Compare **raw downstream readouts** at `t=0` and `t=1` with independent,
  unpatched endpoint runs at precision-appropriate tolerances. Normalized curve
  endpoints alone are not a sufficient test.
- Verify the port against the reference, finite values, path-length convergence,
  and the cosine/norm diagnostics above. Document handling of exceptional pairs.
- Keep method-specific flags. For paired unflagged comparisons, use only pairs
  valid and unflagged under both methods; retain all original rows and report
  exclusions. Do not compare two differently filtered subsets without saying so.

## Deliverables and completion stages

- [ ] **P1 â Audit and freeze.** Preserve completed stages; record the current
  code/configuration, reference version, and pair list. Add the opt-in path.
- [ ] **P2 â Validate.** Complete the endpoint, reference-equivalence, and path
  checks before model inference for the pilot.
- [ ] **P3 â Paired pilot.** Run or reuse matched linear data and run Matthew's
  path on the frozen pairs. Save sufficient data for both normalizations.
- [ ] **P4 â Report.** Add a short interpolation-path control subsection to the
  existing REPORT/RESULTS, with:
  - distance-versus-width comparisons for both paths, distinguishing endpoint
    distance from actual path length;
  - a paired linear-versus-Matthew width comparison and a few matched curves;
  - a separately labeled offline normalization comparison;
  - actual sample counts, exclusions, and whether the distance interpretation
    changes. Say "gradual" rather than equating every broad curve with linearity.

Save under `results/block0_resid_post/path_control/` and
`plots/block0_resid_post/path_control/`, or a nonconflicting equivalent if the
current layout differs. Include method, normalization, hook, model/configuration,
reference version, seed, and pair IDs in metadata. Do not overwrite existing
linear or embedding results. Update CHANGELOG/JOURNAL using the existing format.

## Minimal edits to the existing PLAN

Retain completed work and the primary linear setup. Narrow any existing
"do not switch to SLERP" prohibition to: **do not change the main experiment's
default; the opt-in paired control above is allowed**. Likewise, permit only
this amendment's path-length diagnostic and offline normalization comparison;
keep unrelated metrics, prompts, models, and full replication out of scope.
Append P1âP4 without resetting earlier stage statuses. Update Current status and
Next step from the actual repository state, not from this amendment's assumptions.

If time is short, use the first 64 IDs of the frozen random sample plus the
illustrative pairs, and report the reduced coverage. Reserve the finalization
time required by the existing budget. The next action is to audit the local
baseline and freeze the paired sampleânot to restart a five-model full sweep.
