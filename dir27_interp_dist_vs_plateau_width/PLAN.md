# PLAN â Direction 27: distance vs. plateau width after the first transformer block

> Working folder: `dir27_interp_dist_vs_plateau_width`.
> Minimal intervention-site rerun of dir27; do not create a new direction.
> Agent updates Current status / Next step and ticks stages each iteration.
> Preserve the previous PLAN and results before applying this amendment. If already
> replaced, recover the previous PLAN from git. Inherit all unmentioned settings.

## Success criterion (definition of "done")

Repeat the existing dir27 distance-versus-transition-width experiment, changing
only the intervention site from input embeddings to the final-position residual
stream **after transformer block 0**. Recompute endpoint distance in this same
space. Regenerate the existing plots and width rankings from the new runs.
Report the observed outcome, whether or not it agrees with the embedding result.

This matches Matthew's **intervention location**, not his entire protocol:
keep dir27's linear interpolation and readout normalization unchanged.

## Setup (fixed)

- Reuse the exact checkpoints, precision, token IDs, prefix/BOS handling, attention
  masks, interpolation grid, readout blocks, width extraction, and flagging rules
  from the existing dir27 runs. Start with GPT-2 Large; then reuse the existing
  GPT-2 XL, Pythia-1.4B, Llama-3.1-8B, and Qwen3-8B-Base configurations.
- Keep the original `The house was big` vocabulary-sweep setup: construct endpoint
  B by replacing only the final token ID, without decoding and re-tokenizing.
  Keep each model's original vocabulary coverage; do not restrict to the old Top 50.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` â read both.**
- **Do not install or upgrade torch, torchvision, transformer_lens, cupbearer,
  jax, or flax. Reuse the existing model-loading and inference stack.**

### Change 1 â Move the interpolation hook

Run endpoints A and B normally and collect their last-position residual vectors
at the output of the first complete transformer block:

```text
r_A = resid_post(block=0, position=last; endpoint=A)
r_B = resid_post(block=0, position=last; endpoint=B)
r(t) = (1 - t) * r_A + t * r_B
```

The TransformerLens reference location is `blocks.0.hook_resid_post`.
In the existing implementation, use the equivalent full-block output, including
its residual additions: not the embedding, attention output, MLP output alone,
or a normalized input to the next block. Record the actual module/hook used for
each model; do not rely on the ambiguous label "layer 0".

For each interpolation point, run endpoint A with a forward hook that replaces
**only the last-position hidden state at block 0's output** with `r(t)`.
Leave prefix positions untouched and let all remaining blocks compute normally.
Do not feed `r(t)` back through block 0 or reuse downstream KV caches from another
t. Keep attention and MLP computations unfrozen. Preserve the block's other
returned values when replacing its hidden-state output.

If editing shared helpers, add an explicit intervention-site option and retain
the old default for existing callers, including other directions.

### Change 2 â Measure distance where interpolation happens

```text
D = ||r_B - r_A||_2
```

Use this unnormalized L2 distance as the scatter plot's horizontal coordinate.
For the retained straight-line interpolation, it is the length of the path being
injected. Do not pair new transition widths with old embedding distances.
Label the axis `Block-0 resid_post L2 distance from ' big'` (or the corresponding
anchor-token display). Do not call it token-embedding distance.

### Everything else stays unchanged

Keep the current downstream residual readouts, progress-curve normalization,
first-crossing width calculation, non-monotonicity flags, and plotting filters.
(Amended by PLAN_addendum.md: do not change the main experiment's default; the opt-in paired
path control, its path-length diagnostic and its offline normalization comparison are allowed.)
In particular, do not switch to SLERP, logits, relative-to-both-endpoints
normalization, or an isotonic fit as part of this rerun.

## Stages (checklist)

- [x] **S1 â Audit and patch.** Read the previous local PLAN, run configuration,
  and `experiments/common.py`. Preserve the previous run and reopen dir27's STOP
  sentinel for this rerun. Implement only the hook/distance changes above. Record
  any discrepancy between local code and REPORT.md before proceeding.
- [x] **S2 â Validate endpoints.** Reuse the existing small sanity-check token
  set. Verify that the injected vector equals `r(t)` and prefix positions are
  unchanged. At `t=0` and `t=1`, compare the **raw downstream readout vectors**
  against independent, unpatched A and B runs within precision-appropriate
  tolerances. Checking normalized curve endpoints alone is not sufficient.
- [x] **S3 â Rerun.** Recompute distances and full interpolation curves for the
  existing vocabulary sweeps. Old widths cannot be reused. Keep the same batch
  settings unless memory requires a smaller batch; record any such change.
- [x] **S4 â Refresh existing deliverables.** Rebuild distance-versus-width plots,
  final-width distributions, representative curves, and existing token rankings
  from the new results. Update methods and captions with the explicit hook and
  distance definition. Do not copy the old conclusions into the new report.

Path-control stages (PLAN_addendum.md):

- [x] **P1 — Audit and freeze.** Reference pinned (repo commit 4222c3e), verbatim copy kept, 256 seed-0
  pair IDs + 6 illustrative pairs saved before the new path was run, opt-in path added.
- [x] **P2 — Validate.** Port vs reference, hook, endpoints, raw readouts, path-length convergence.
- [x] **P3 — Paired pilot.** GPT-2 Large, both paths on the frozen pairs; linear subset rerun (saved sweep
  holds only normalized curves) and checked against the saved sweep.
- [x] **P4 — Report.** REPORT.md Finding 5 / Table 3; RESULTS.md R6 with Figures 18-22.

## Output handling

Save new numeric outputs under `results/block0_resid_post/` and plots under
`plots/block0_resid_post/`; never overwrite the embedding-run data. Record
`intervention_site=block0_resid_post` and `distance_space=block0_resid_post` in
run metadata. Preserve the old reports before updating RESULTS.md / REPORT.md
to describe the current run; keep the history in CHANGELOG.md. If shared output
paths are used by another direction, leave them untouched.

## Report budget

`REPORT_generalization.md` (feedback 3, approved) keeps the operator's earlier allowance: "The follow-up
report may contain up to 12 main figures" (PLAN.md as of 2026-09-28). `REPORT.md` stays at 8 or fewer.
The gate reads one limit for all `REPORT*.md`, so it is stated once here:

- Maximum main figures: 12

## Fallback (if time runs short)

Complete GPT-2 Large first. Report exactly which other model sweeps are complete
or partial; do not fill missing panels with embedding-run results. Reserve the
last 20 minutes to finalize the completed outputs and write STOP.

## Out of scope (do NOT)

No new models, prompts, interpolation-layer sweep, metrics, hypothesis tests,
feature analysis, France/dir29 rerun, or full reproduction of Matthew's protocol.
Do not tune the setup to recover the old conclusion.

## On-track check

End each JOURNAL.md entry with:
`On track? <yes/no> â <stage, % done, blocker if any>`.

## Current status

Complete, including the PLAN_addendum.md path control. All five models were rerun with the interpolation and
the distance at the last-position output of transformer block 0 (S1-S4). The GPT-2 Large paired path control
(P1-P4) ran 256 random pairs + 6 illustrative pairs under the linear path and Matthew's `slerp_rescale` path;
data in `results/block0_resid_post/path_control/`, plots in `plots/block0_resid_post/path_control/`.
Observed outcome: the middle-block rise of width with distance holds under both paths; single-pair widths
change little; the final-block rise is 0.02 (linear) vs 0.05 (Matthew's path) with dir27's normalization and
0.10-0.14 with the relative normalization. No pair had a negative endpoint cosine.
REPORT.md keeps 8 figures, so the path-control figures are in RESULTS.md R6 and REPORT.md carries Table 3.

## Next step

`human_feedback_7.txt` is `review_pending` again (2026-10-14, repair after rejection): cropped Llama panel in the
summary figure; per-model SLERP figures REPORT.md Figures 9-13, Table 4; RESULTS.md R7. Wait for the wrapper's review.

## Method references

- Original dir27 Methods:
  https://github.com/billy000400/marsv/blob/main/dir27_interp_dist_vs_plateau_width/REPORT.md
- Matthew's collection/injection code (`collect_original_activations`,
  `interpolate_resid_post_layer`):
  https://github.com/MShinkle/activation_plateau_mechanisms/blob/main/scripts/interpolate_and_record_activations.py
