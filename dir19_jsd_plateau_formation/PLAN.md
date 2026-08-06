# PLAN - Direction: When does the JSD-transition relationship form during training?

> Working folder: `dir19_jsd_plateau_formation`.
>
> **`PLAN.md` is operator-owned and read-only. The agent must not edit it, tick its checkboxes, or rewrite `Current status` or `Next step`.** Record work in `JOURNAL.md`, `RESULTS.md`, and `CHANGELOG.md`.

## Research question

When during Pythia training do two distinct phenomena appear?

1. **JSD-selective ordering:** pairs with larger corpus next-token JSD become the pairs with narrower transitions.
2. **Global plateau formation:** transition curves become narrower and their output movement becomes more concentrated near a boundary.

The goal is to learn whether these happen together or at different times. This plan characterizes one released Pythia training trajectory; it does not by itself establish a universal training law or a causal mechanism.

## Success criterion (definition of "done")

- Bound the onset of the negative JSD-width relationship more tightly than â€œpresent by step 1000.â€
- Determine whether JSD predicts the **change in width within each training interval**, rather than merely correlating with width at each checkpoint.
- Determine whether full-output movement becomes increasingly concentrated near the `d(t)` transition as training proceeds.
- Test whether the observed 64k-to-final widening is reproduced on the 1,000-pair set or should be treated as noise in the 60-pair set.
- `REPORT.md` gives a clear verdict even if no sharp onset, phase change, or late reversal is found. Null results are complete.
- `RESULTS.md` and `REPORT.md` contain current-best results only; history remains in `CHANGELOG.md`. When complete, write an empty `STOP` file.

## Fallback (if time runs short)

Run the 60-pair set at all early checkpoints from step 0 through step 1000, combine these with the existing 8k, 32k, 64k, and final results, and produce one figure separating correlation, median width, and width variability over time. This is sufficient to report an onset interval. Reserve the final 20 minutes for current-best files and `STOP`.

## Setup (fixed)

- Treat `../dir18_continuation_jsd_plateau` as read-only upstream work. Copy the exact pair manifests, sentence frames, corpus JSD values, metric code, and relevant existing checkpoint results into this directory; record file hashes before any new run.
- **Main model:** current `EleutherAI/pythia-1.4b-deduped`; never use `-v0`.
- **Primary time-series set:** the existing 60 carefully matched pairs in which no token is reused, evaluated at every required checkpoint.
- **Large validation set:** the existing 1,000 pairs, with endpoint-aware uncertainty, at `step0`, `step512`, `step1000`, `step64000`, and `step143000`. After the 60-pair scan identifies an onset bracket, also run the two checkpoints that define that bracket if they are not already in this list; label this as follow-up validation.
- Use the same three fixed sentence frames, post-block-0 patch point, norm-rescaled SLERP, 50 interpolation positions, valid output IDs, `d(t)`, `w`, and curve-validity rules as dir18. Never reselect pairs at a checkpoint.
- The inherited endpoints were filtered partly using final-checkpoint model plausibility. Therefore every timing claim is conditional on endpoints that the final model considers plausible; state this prominently in the report.
- **Required early checkpoints:** `step0`, `step1`, `step2`, `step4`, `step8`, `step16`, `step32`, `step64`, `step128`, `step256`, `step512`, `step1000`.
- **Required later checkpoints:** reuse `step8000`, `step32000`, `step64000`, and `step143000`. Add `step2000`, `step4000`, `step16000`, `step96000`, and `step128000` if budget permits. Add further checkpoints only to resolve a clearly stated interval; label such zoom-ins as follow-up resolution, not independent confirmation.
- Process one checkpoint at a time and delete only reproducible model-cache copies after its artifacts pass validation. Never delete upstream results.
- At every checkpoint, record the learning rate and evaluate next-token loss on one frozen sample of 256 unused corpus rows. This is only timing context; it is not a grokking test.
- Read `../BUDGET.md` and `../CLAUDE.md` every iteration. Do not replace the installed CUDA/PyTorch stack or install TransformerLens, JAX, or Flax.

## Primary measurements

At every checkpoint `s`, save raw curves and compute:

1. **Cross-sectional ordering**

```math
\rho_s=\operatorname{Spearman}(J_{\mathrm{corpus}},w_s).
```

2. **Global shape:** median and IQR of `w_s`, median edge drift, and median `w_s` within each of the five JSD groups.

3. **Training-induced change**

```math
\Delta w_s=w_s-w_0,
```

and, for adjacent measured checkpoints,

```math
\Delta w_{s_1\rightarrow s_2}=w_{s_2}-w_{s_1}.
```

Report `rho(J_corpus, Î”w)`. This directly tests whether higher-JSD pairs sharpen more during a particular interval. Compute each change only when the same pair and sentence frame is valid at both checkpoints, and bootstrap the changes as paired observations.

4. **Learned output separation:** model-output JSD for each pair and its correlation with corpus JSD. Treat this as a co-developing measurement, not proof of mediation or causality.

5. **Full-output movement along the path.** Let `q_s(t) = softmax(z_s(t))` over all valid tokenizer vocabulary IDs. For neighboring interpolation positions, compute

```math
m_{s,j}=JSD\!\left(q_s(t_j),q_s(t_{j+1})\right).
```

Compute adjacent JSD from stable log-softmax values and stream the calculation; do not save full-vocabulary logits or probabilities. Save total movement `T_s = sum_j m_{s,j}` before normalization. Flag paths with `T_s < 1e-8` bits and do not normalize them.

For the remaining paths, normalize `r_{s,j} = m_{s,j}/T_s`. Use normalized entropy `H(r_s)/log(49)` as the primary concentration metric; lower values mean more concentrated movement. As a location check, report the mass inside a fixed-width window of 0.2 in `t`, centered where `d(t)` crosses 0.5. Report the mass inside the variable 0.1-to-0.9 transition segment only as a secondary alignment metric, because it is mechanically related to `w`.

## Onset rules

- Bootstrap the entire checkpoint trajectory using the same resampled pairs at every checkpoint, and construct a simultaneous 95% band from the maximum bootstrap deviation across checkpoints. Keep width IQR visible as a restricted-range diagnostic; do not use an arbitrary IQR threshold as an onset gate.
- **JSD-ordering onset bracket:** after the last measured checkpoint whose simultaneous band includes zero and by the first of two consecutive checkpoints whose band lies below zero. Validate the two bracket checkpoints on the 1,000-pair set using endpoint-aware inference. If this rule never triggers, report no stable onset within the measured resolution.
- **Global sharpening onset bracket:** use paired trajectory bootstraps for `w_s - w_0`. Call the curves plateau-shaped only when, for two consecutive checkpoints, median `w` is below the straight-line value 0.8, median edge drift is below its straight-line reference, and both comparisons exclude their reference values.
- Do not call either event a â€œphase transitionâ€ unless an abrupt change is reproduced on an independent training run. With the present design, use â€œonset windowâ€ or â€œrapid change.â€

## Stages (checklist; operator updates only)

- [ ] **S1 - Transfer and freeze:** copy the exact dir18 manifests/code/results, verify hashes, and reproduce the existing step-0/final metrics without changing definitions.
- [ ] **S2 - Early formation scan:** run the 60-pair set at steps 1-512 and combine with step 0 and step 1000.
- [ ] **S3 - Separate ordering from global sharpening:** plot `rho_s`, median/IQR `w_s`, JSD-group trajectories, `Î”w_s`, interval-specific `rho(J, Î”w)`, and fixed-sample next-token loss.
- [ ] **S4 - Measure full-output redistribution:** compute neighboring-output JSD profiles and test whether movement concentrates near the transition over training.
- [ ] **S5 - Validate key observations:** run the 1,000-pair set at the fixed milestones and the onset-bracket checkpoints, using an endpoint-aware bootstrap and endpoint-label permutation; test the 64k-to-final comparison as a prespecified follow-up to the 60-pair observation.
- [ ] **S6 - Finalize:** produce current-best figures, a self-contained report, limitations, reproducibility details, and `STOP`.

Every reported metric must have a saved figure in `plots/` and a definition in `REPORT.md` Methods.

## Required outputs

- Frozen copies and hashes of the pair/corpus manifests inherited from dir18.
- Raw per-pair, per-context, 50-point curves for every checkpoint.
- `results/checkpoint_metrics.json` with `w`, edge drift, output JSD, total output movement, normalized movement entropy, fixed-window movement mass, fixed-sample loss, learning rate, and validity flags.
- `plots/formation_overview.png`: `rho_s`, median/IQR `w_s`, and edge drift versus training step.
- `plots/interval_sharpening.png`: JSD-group width trajectories and `rho(J, Î”w)` by interval.
- `plots/output_movement_formation.png`: concentration of neighboring full-output changes over training.
- `plots/large_bank_confirmation.png`: endpoint-aware milestone results for the 1,000-pair set.
- Current-best `RESULTS.md`, self-contained `REPORT.md`, append-only `CHANGELOG.md`, and `STOP` when complete.

## Prespecified verdicts

- **JSD ordering appears before global plateau shape:** corpus JSD is associated with pair ordering early; later training mainly sharpens many pairs together.
- **JSD ordering and plateau shape appear together:** selective separation and global sharpening are temporally coupled.
- **Global plateau shape appears without stable JSD ordering:** training creates plateaus, but context-averaged immediate-next-token JSD does not explain which pairs sharpen.
- **Cross-sectional `rho_s` persists but interval `rho(J, Î”w)` is near zero after early training:** the ordering was established earlier and is merely preserved later.
- **The late widening reproduces on 1,000 pairs:** report a late reversal in this Pythia run; do not infer a universal phase.
- **The late widening does not reproduce:** report that the reversal is unsupported on the large bank and may be specific to the 60-pair set.

## Out of scope (do not)

- Rewrite or delete files in dir18.
- Pair reselection at different checkpoints.
- Attention-versus-MLP freezing, Jacobians, spline/local-complexity estimates, or full layer scans. Those are a separate mechanism direction.
- Call the result grokking without separately showing delayed generalization or robustness after training performance has saturated.
- Use â€œregion migrationâ€ or â€œphase transitionâ€ as a synonym for non-monotonic training curves.
- Infer causality from temporal order.
- Claim that each plateau represents one continuation distribution; endpoint corpus JSD is still not measured along the interpolation path.
- Generalize onset timing beyond final-model-plausible endpoints or beyond this released Pythia trajectory without an independent selection rule and additional training runs.
- Add context-conditioned or multi-token continuation metrics in this direction.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> - <stage, % done, blocker if any>`

Do not record progress by editing this plan.

## Current status

Existing dir18 artifacts cover step 0, 1000, 8000, 32000, 64000, and 143000 on the 60-pair set. They show no JSD-width relationship at step 0 and a relationship already comparable to the final model at step 1000, but they cannot locate formation inside the 0-to-1000 interval. Median width decreases through step 64000 and then rises modestly at the final checkpoint; this late change has not yet been confirmed on the 1,000-pair set.

## Next step

Create the new directory, copy and hash the fixed manifests and existing formation artifacts from dir18, reproduce one existing checkpoint as a compatibility check, and then run the required early checkpoints from step 1 through step 512. Do not edit this `PLAN.md`.

## References

- Pythia's released early and evenly spaced checkpoints: https://huggingface.co/EleutherAI/pythia-1.4b-deduped
- Matthew Shinkle and StefanHex, â€œActivation Plateaus: Where and How They Emergeâ€: https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Ahmed Imtiaz Humayun, Randall Balestriero, and Richard Baraniuk, â€œDeep Networks Always Grok and Here is Whyâ€: https://arxiv.org/abs/2402.15555