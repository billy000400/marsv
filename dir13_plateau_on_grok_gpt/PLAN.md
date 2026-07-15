# PLAN - Direction: Does the 12-layer Shakespeare GPT show activation plateaus?

> Working folder: `dir12_small_gpt_plateau_sanity`. Agent REWRITES "Current status"/"Next step" + ticks stages each iteration. Disk (`PLAN.md`/`JOURNAL.md`/`RESULTS.md`/`REPORT.md`/`CHANGELOG.md` + `../BUDGET.md` + `../CLAUDE.md`) is the only memory.

## Research question and decision

Before attempting to map or interpret plateaus in the paper's small GPT, answer the cheaper gating question:

> Does the trained 12-layer, 12-head character-level GPT from *Deep Networks Always Grok and Here is Why* exhibit reproducible activation plateaus under the same perturbation-style assay used for the MNIST model?

This is a go/no-go sanity check. A positive result justifies a follow-up study of what the plateaus correspond to. A calibrated negative result is equally complete and means we should not spend time clustering or interpreting regions in this model.

Important source distinction: Figure 8 of the paper is a separate modular-addition Transformer. The 12-layer GPT of interest is Figure 9: a 12-layer, 12-head GPT trained for next-character prediction on Shakespeare text, with GeLU MLPs. Do not conflate the two experiments.

## Success criterion (definition of "done")

`RESULTS.md` and `REPORT.md` give one of the following evidence-backed verdicts for the late-trained 12-layer GPT:

1. **Plateaus detected:** in-distribution final-position activations have a delayed-then-steep downstream response that is reproducible across held-out contexts and perturbation seeds, stronger than the matched off-distribution control, and present at two or more consecutive intervention blocks; or
2. **No plateaus detected under the calibrated assay:** after checking all plausible intervention depths and a perturbation range large enough to change downstream predictions, the response is smooth/approximately linear or is no more plateau-like than the matched control; or
3. **Model-faithfulness blocker:** only if the paper's model cannot be faithfully reconstructed, the exact missing information is documented, a clearly labeled best-effort reconstruction has been tested, and the report separates conclusions about that reconstruction from conclusions about the paper's model.

Required artifacts:

- `MODEL_SPEC.md`: every architecture, data, optimizer, and checkpoint detail, each labeled **confirmed from source** or **reconstruction choice**.
- Reproducible training or checkpoint-loading code and a checkpoint/config/data hash.
- A saved tidy result table with one row per context x direction x radius x intervention block x basepoint type.
- `plots/training_curves.*`.
- `plots/response_by_layer.*`: raw downstream-distance curves for natural and matched-control basepoints, with bootstrap uncertainty.
- `plots/plateau_score_by_layer.*`: control-calibrated plateau score/effect size by intervention block.
- `plots/individual_curves.*`: representative individual rays so an average cannot manufacture an apparent plateau.
- `REPORT.md` with Methods, calibration checks, the verdict, limitations, and exact figure references.

Null/negative results are COMPLETE if the assay passed the calibration checks below. When complete, write an empty `STOP` file.

## Fallback (if time runs short)

Finish a defensible gate rather than starting plateau interpretation:

- Use one faithful late-training checkpoint, 64 held-out contexts, 8 fixed random directions per context, at least four intervention depths spanning early to late blocks, and both natural and matched-control basepoints.
- Produce the training/checkpoint provenance, response curves, plateau-score comparison, and a qualified yes/no verdict.
- Do not spend fallback time on clustering, character-level interpretation, local-complexity replication, or training-depth sweeps.

The wrapper reserves the final 20 minutes for `RESULTS.md`, `REPORT.md`, `CHANGELOG.md`, figure checks, and `STOP`.

## Setup (fixed unless source audit proves otherwise)

### Model and data

- Target: the paper's 12-block, 12-head causal GPT with GeLU MLPs, trained on next-character prediction on the Shakespeare Text Dataset.
- First preference: authors' exact released code, config, and checkpoint.
- The current public paper repository may not contain the GPT training code/checkpoint. Spend at most 30 minutes auditing the paper, supplement, linked repository history/branches/releases, and configs. Record findings in `MODEL_SPEC.md`.
- If exact artifacts remain unavailable, implement a minimal faithful reconstruction. Do **not** silently assume GPT-2-small defaults. Dataset split, context length, embedding width, dropout, optimizer, learning-rate schedule, weight decay, batch size, and number of updates must be explicitly marked confirmed or inferred.
- Save log-spaced checkpoints during training, including initialization/early/late states when affordable, so a later training-evolution study will not require retraining. This direction analyzes only the late checkpoint.
- Primary evaluation examples are held-out Shakespeare contexts on which the model's next character is correct. Pre-register a high-confidence subset (top confidence quartile) and a lower-confidence comparison subset before looking at plateau results.

### Activation intervention

- Unit of intervention: the residual-stream vector at the **final sequence position** after a transformer block. Keep all other sequence positions unchanged.
- Primary downstream measurement point: final-position residual stream immediately before the language-model head.
- Primary response metric:

  \[
  d_{\mathrm{hidden}}(\alpha)=\frac{\|z(h+\alpha u)-z(h)\|_2}{\sqrt{d_{\mathrm{model}}}}.
  \]

- Secondary functional metric: Jensen-Shannon divergence between the baseline and perturbed next-character distributions. Also record top-1 character flips and confidence.
- Primary directions: fixed random unit directions. Use the same direction/radius schedule for the natural basepoint and its matched control.
- Candidate intervention blocks must span the network. Pilot blocks `{0, 3, 6, 9, 10}`; if any signal is found or the result is ambiguous, sweep every block `0..10`. Block 11 is measurement-adjacent and may be retained only as an expected weak/negative control.

### Perturbation scale and plateau statistic

- At each layer define the natural scale `s_l` as the median L2 distance between randomly paired held-out final-position activations. Parameterize perturbations as `alpha = rho * s_l` so radii are comparable across layers.
- Pilot a monotone grid of at least 41 `rho` values starting at zero. Adapt `rho_max` using only pilot data until at least 80% of rays either flip the top-1 next-character prediction or clearly enter a large-response regime. Cap and report the search range; freeze it before the confirmatory run.
- Plot raw, unsmoothed curves. Smoothing/isotonic regression may be used only to compute a summary score and must never replace raw plots.
- For a curve normalized to `x = rho/rho_max` and `y = d(rho)/d(rho_max)`, define the preregistered plateau index

  \[
  PI=\int_0^1 [x-y(x)]\,dx.
  \]

  Positive `PI` means the response is delayed relative to a straight-line response. Report `PI` together with boundary sharpness (maximum finite-difference slope divided by mean slope); neither metric alone is sufficient.
- Primary comparison is `Delta PI = median(PI_natural) - median(PI_control)` with a hierarchical bootstrap over contexts, then directions. Report a 95% interval and Cliff's delta. Treat tiny but statistically significant differences as inconclusive.

### Matched controls and implementation checks

- Matched off-distribution basepoint: sample from the empirical per-layer diagonal Gaussian, then rescale to match the paired natural activation's norm. Keep the surrounding sequence activations fixed. Document any alternative if the model implementation makes this invalid.
- Required checks before interpreting a negative result:
  - `alpha=0` reproduces the unmodified forward pass within numerical tolerance.
  - Batched interventions agree with a single-example reference implementation.
  - Radius zero gives zero distance; the largest radius produces a substantial response and top-1 flips for at least 80% of rays.
  - Results are not an artifact of averaging: inspect individual curves and per-context scores.
  - Hidden-state and output-distribution metrics lead to compatible qualitative conclusions.
  - The same code can detect a synthetic delayed-response curve in a unit test.

### Reproducibility and operating rules

- Fix and record model seed, data seed, context IDs/offsets, direction seeds, package versions, device, precision, and git commit.
- Cache clean activations; stream perturbation batches to avoid storing full sequence activations for every radius.
- Keep raw results separate from plotting code. All figures must be exactly regenerable from saved result tables.
- Read shared limits in `../BUDGET.md` and operator rules in `../CLAUDE.md` every iteration.
- `RESULTS.md`/`REPORT.md` contain current-best results only; `CHANGELOG.md` contains history.
- Do not `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax; preserve the existing CUDA environment.

## Stages (checklist)

- [x] **S1 - Source and model audit.** Read the paper/supplement and official repository; create `MODEL_SPEC.md`; identify exact code/checkpoint if available. If details remain missing after 30 minutes, freeze and justify a reconstruction config rather than repeatedly searching.
- [x] **S2 - Reproduce/load the model.** Train or load the 12-layer GPT; verify next-character loss/accuracy and checkpoint provenance; save `plots/training_curves.*`. Do not start plateau interpretation until the model is demonstrably trained.
- [x] **S3 - Implement and validate the assay.** Add final-position residual hooks, perturbation batching, matched controls, raw tidy output, and unit/smoke tests. Pass all zero-radius and batching checks.
- [x] **S4 - Pilot and freeze calibration.** Run a small layer/radius pilot, choose `rho_max` without using confirmatory examples, and write the frozen context-selection, radius, and scoring config to disk.
- [x] **S5 - Confirmatory plateau test.** Run the frozen assay on held-out contexts and seeds; generate response, score, and individual-curve figures with bootstrap intervals. If positive/ambiguous, sweep every candidate block. If clearly negative and calibrated, stop rather than adding unrelated probes.
- [x] **S6 - Verdict and handoff.** Write `RESULTS.md`, `REPORT.md`, and `CHANGELOG.md`. State exactly which of the success-criterion verdicts is supported, what model was actually tested, and whether a plateau-mapping follow-up is warranted. Create `STOP`.

## Decision rule for follow-up

- **Go:** natural activations show a visually clear delayed-then-steep response, `Delta PI > 0` with a non-trivial effect size and 95% interval excluding zero, and the result appears at two or more consecutive intervention blocks in both downstream metrics.
- **No-go for this model:** calibrated curves are approximately linear/smooth, plateau scores are indistinguishable from matched controls, or any apparent signal occurs only in a single layer/seed or only after averaging.
- **Qualified result:** the reconstruction rather than the paper's exact checkpoint was tested. A positive result can motivate follow-up; a negative result cannot establish that the paper's exact model lacks plateaus.

## Out of scope (do NOT)

- Do not map, count, cluster, or assign semantics to plateaus yet.
- Do not study how plateaus evolve across training checkpoints yet; only save checkpoints for that later direction.
- Do not reproduce the paper's local-complexity or adversarial-grokking results except for a minimal training sanity check.
- Do not switch to the separate modular-addition Transformer, MNIST model, larger ResNet, or pretrained GPT-2.
- Do not launch width/depth/noise sweeps or steering experiments.
- Do not redefine a plateau after seeing the confirmatory curves.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with: `On track? <yes/no> - <stage, % done, blocker if any>`.

## Current status

**COMPLETE — verdict: NO plateaus detected (qualified reconstruction). `STOP` written.**

Source audit (S1) found the paper's GPT code/checkpoint are unreleased (repo has only MNIST-MLP +
CIFAR-ResNet), so a faithful 12L/12H GeLU char-GPT reconstruction was trained (val acc 0.560) and
probed. Across all 11 intervention blocks the natural final-position activations show a
**saturating/concave** downstream response (median PI −0.15…−0.30), not the flat-then-steep plateau
shape. ΔPI(nat−ctrl) is positive & significant everywhere (peak +0.096, Cliff's δ +0.91, JSD agrees)
but is a difference between two non-plateau shapes. All calibration checks passed (unit test, alpha=0
fidelity, ≥81% flips at max radius, individual rays, hidden↔JSD agreement). This is success-criterion
(2), qualified by (3). No-go for a plateau-mapping follow-up on this model.

## Next step

None — direction complete and `STOP` written. If ever revisited: probe a much longer-trained
(grokking-scale) checkpoint and/or learned directions, but that overlaps the "during training"
direction and is out of this gate's scope.

## Primary references

- Humayun, Balestriero, and Baraniuk, *Deep Networks Always Grok and Here is Why*: https://arxiv.org/abs/2402.15555
- Official project repository: https://github.com/AhmedImtiazPrio/grok-adversarial