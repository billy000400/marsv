# PLAN - Direction: Do next-token distributions correspond to transition sharpness?

> Working folder: `dir18_continuation_jsd_plateau`.
>
> **`PLAN.md` is operator-owned and read-only. The agent must not edit it, tick its checkboxes, or rewrite `Current status` or `Next step`.** Record work in `JOURNAL.md`, `RESULTS.md`, and `CHANGELOG.md`.

## Success criterion (definition of "done")

- The trained-model relationship between corpus next-token JSD and transition width is reported using both the 1,000-pair generality analysis and the controlled 60-pair analysis.
- Step 0 appears only as a baseline control; intermediate-checkpoint formation results do not appear in the main report.
- Every central quantity is defined in plain English before notation is introduced.
- `REPORT.md` clearly separates what the experiment shows from what it does not show.
- `RESULTS.md` and `REPORT.md` contain current-best results only; history remains in `CHANGELOG.md`.
- A null, mixed, or invalid-metric result is complete if it is reported clearly. When the revision is complete and no feedback remains, write an empty `STOP` file.

## Fallback (if time runs short)

At minimum, produce a short report containing: the 1,000-pair final-checkpoint plot, the controlled 60-pair estimate, the step-0 baseline, the model-output-JSD validation, and the main limitations. Move all construction and statistical detail to an Appendix. Reserve the final 20 minutes for current-best files and `STOP`.

## Setup (fixed)

- **Main model:** current `EleutherAI/pythia-1.4b-deduped`, revision `step143000`; never use `-v0`.
- **Baseline only:** the same model at `step0`.
- **Cross-scale check:** current `EleutherAI/pythia-410m-deduped`, revision `step143000`.
- **Corpus:** the existing two row-aligned samples from `EleutherAI/pile-deduped-pythia-preshuffled`. One sample is used to choose pairs; the other measures the JSD used in the reported association.
- **JSD:** compare the distributions of the single token immediately following endpoint tokens `u` and `v`. This is not a multi-token continuation metric.
- **Fixed sentence frames:** `The thing was`, `They said it was`, and `I thought it was`.
- **Interpolation:** patch the final-token residual stream after block 0; use norm-rescaled SLERP at 50 positions; record final-position logits after the remaining blocks.
- **Output-distance score:**

```math
d(t)=\frac{\lVert z(t)-z_u\rVert_2}
          {\lVert z(t)-z_u\rVert_2+\lVert z(t)-z_v\rVert_2}.
```

- **Transition width:**

```math
w=t(d=0.9)-t(d=0.1).
```

  Smaller `w` means that `d(t)` changes within a narrower part of the path. It does not by itself prove that the full logit vector is stationary on either side.
- **Pair sets:**
  - 1,000 pairs for the main visual/generalization result; tokens recur, so use endpoint-aware uncertainty.
  - 60 carefully matched pairs for the controlled analysis; no token is reused.
- Reuse the existing manifests, raw curves, and analyses. Do not rerun GPU experiments unless an artifact is missing or invalid.
- Read `../BUDGET.md` and `../CLAUDE.md` every iteration. Do not replace the installed CUDA/PyTorch stack or install TransformerLens, JAX, or Flax.

## Stages (checklist; operator updates only)

- [x] **S1 - Corpus metric:** count immediate successors in both corpus samples and verify that the JSD estimate is reliable.
- [x] **S2 - Pair sets and interpolation:** run the frozen 60-pair and 1,000-pair sets with curve-validity checks.
- [x] **S3 - Final-model analysis:** estimate the JSD-width relationship, validate corpus JSD against model-output JSD, and run the planned sensitivity analyses.
- [ ] **S4 - Correspondence-only report:** lead with the 1,000-pair result, follow with the controlled 60-pair result, retain only essential controls, and move technical detail to Methods/Appendix.

For the 1,000-pair analysis, do not treat pairs as independent. For the 60-pair analysis, do not treat the three sentence frames or the 50 interpolation positions as independent observations.

## Interpretation rules

- **Larger corpus JSD predicts smaller `w`, and corpus JSD predicts model-output JSD:** endpoint tokens with more different immediate-next-token roles tend to be separated by narrower trained-model transitions.
- **Corpus JSD predicts model-output JSD but not `w`:** the model distinguishes the endpoints, but the distinction is not generally expressed as a narrow transition.
- **Corpus JSD predicts neither quantity:** the context-averaged one-token statistic is too coarse for this question.
- **The same relationship appears at step 0:** initialization or token geometry is a likely confound.
- **The association disappears after accounting for model-output JSD and pair properties:** report the strong overall association, but do not claim an independent relationship.

## Required outputs

- Pair and corpus manifests.
- Raw 50-point `d(t)` curves and curve-validity summaries.
- A main 1,000-pair figure and a controlled 60-pair figure.
- Model-output-JSD validation and one concise adjusted analysis.
- A self-contained `REPORT.md`, current-best `RESULTS.md`, append-only `CHANGELOG.md`, and `STOP` when complete.

## Out of scope (do not)

- Intermediate checkpoints, onset timing, late-training reversal, or any claim about how training forms the relationship.
- Detailed learned `Î”w` analysis.
- Attention/MLP freezing, Jacobians, full layer scans, or other mechanism studies.
- Context-conditioned or multi-token continuation distributions.
- Claims that each plateau represents one continuation distribution, or that continuation distributions jump at the measured boundary. The experiment did not measure continuation distributions along the path.
- Claims of causality, semantic grouping, necessity for low loss, grokking, or universal replication.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> - <stage, % done, blocker if any>`

Do not record progress by editing this plan.

## Current status

The experiments are complete. The final 1.4B model gives `rho = -0.486` on the 1,000-pair analysis with endpoint-aware inference and `rho = -0.525` on the controlled 60-pair analysis. The corresponding step-0 estimates are near zero. Corpus JSD predicts model-output JSD (`rho = +0.751`); the final-checkpoint relationship is no longer significant after accounting for model-output JSD and all measured pair properties. The 410M model provides a consistent cross-scale check.

The existing intermediate-checkpoint results are valid artifacts, but they belong to the separate formation direction and must not drive this report's main story.

## Next step

Revise `REPORT.md` and `RESULTS.md` according to the current human-feedback file. Do not rerun completed experiments, delete formation artifacts, or edit `PLAN.md`.

## References

- Matthew Shinkle and StefanHex, â€œActivation Plateaus: Where and How They Emergeâ€: https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Pythia model suite: https://github.com/EleutherAI/pythia