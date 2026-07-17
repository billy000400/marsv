# PLAN - Direction: Does the 12-layer Shakespeare GPT show Matthew-style activation plateaus?

> Working folder: `dir12_small_gpt_plateau_sanity`. Agent REWRITES "Current status"/"Next step" + ticks stages each iteration. Disk (`PLAN.md`/`JOURNAL.md`/`RESULTS.md`/`REPORT.md`/`CHANGELOG.md` + `../BUDGET.md` + `../CLAUDE.md`) is the only memory.

## Research question

Before trying to map or interpret plateaus in the paper's small GPT, answer the gating question using the phenomenon defined in Matthew Shinkle and StefanHex's post:

> When we interpolate between the last-position activations of two sequences that are identical except for their final input character, does the downstream representation remain close to endpoint A, rapidly cross a boundary, and then remain close to endpoint B?

This direction must reproduce the **two-natural-endpoint interpolation experiment** from Matthew's post. A random ray `h + alpha*u` tests local perturbation robustness and does **not** answer this question.

Important source distinction: Figure 8 of *Deep Networks Always Grok and Here is Why* is a separate modular-addition Transformer. The model of interest is Figure 9: a 12-layer, 12-head GPT trained for next-character prediction on Shakespeare text.

## Authoritative plateau definition

Read Matthew's post and its linked code before changing the assay:

- Post: https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Code: https://github.com/MShinkle/activation_plateau_mechanisms

For two input sequences `A` and `B` that are identical except for the final character:

1. Run both sequences and collect their final-position activations `h_A` and `h_B` at interpolation layer `L`.
2. Spherically interpolate from `h_A` to `h_B`. As in Matthew's post, slerp the directions and linearly interpolate the vector norms.
3. Patch each interpolated activation `h(t)` into the final sequence position at layer `L`; leave every earlier position unchanged.
4. Record downstream activation or logit vector `x(t)` at recording point `R`.
5. Compute Matthew's relative distance to the two endpoint outputs:

   \[
   d(t)=\frac{\|x(t)-x_A\|_2}{\|x(t)-x_A\|_2+\|x(t)-x_B\|_2}.
   \]

A Matthew-style plateau curve stays near `d=0`, changes rapidly over a narrow interval, and then stays near `d=1`. A roughly diagonal curve is not a plateau.

## Success criterion (definition of "done")

`RESULTS.md` and `REPORT.md` give a clear verdict for the trained 12-layer GPT:

1. **Plateaus detected:** multiple preregistered prompt pairs show plateau-boundary-plateau curves in final logits, the curves are visible individually rather than created by averaging, and the boundary becomes at least as clear when more downstream layers are included; or
2. **No Matthew-style plateaus detected:** after testing a sufficiently varied frozen set of minimal pairs with interpolation early enough to leave most of the network downstream, no individual curves show the plateau-boundary-plateau structure; or
3. **Qualified reconstruction result:** the exact paper checkpoint is unavailable, so the report states that the verdict applies to the trained reconstruction, not necessarily the authors' exact GPT.

Required artifacts:

- `MODEL_SPEC.md` with confirmed paper details and reconstruction choices.
- `prompt_pairs.json` containing the frozen sequences, shared prefix, endpoint characters, selection source, and split.
- A tidy result table with one row per pair x interpolation layer x recording point x interpolation step.
- `plots/pair_curves_logits.*`: individual Matthew `d(t)` curves in final-logit space. Do not show only an average.
- `plots/layerwise_emergence.*`: for fixed early interpolation, show `d(t)` at successive downstream layers for representative fixed pair IDs.
- `plots/interpolation_layer_comparison.*`: final-output curves while varying the interpolation layer, if the primary result is positive or ambiguous.
- `REPORT.md` with the exact interpolation, patching, endpoint-fidelity checks, pair-selection procedure, figures, and verdict.

The previous random-direction result is not evidence for or against this success criterion. Archive it in `CHANGELOG.md` or a clearly labeled previous-results folder; do not use it as the current verdict.

Null/negative results are COMPLETE if the endpoint and pair-coverage checks below pass. When complete, write an empty `STOP` file.

## Fallback (if time runs short)

Run the direct test that answers the question:

- At least 20 frozen minimal pairs.
- Interpolate at `resid_post` of block 0, leaving 11 blocks downstream.
- Use at least 101 evenly spaced `t` values.
- Record final logits and compute Matthew's `d(t)`.
- Save every individual curve and report how many clearly show plateau-boundary-plateau behavior.

Do not spend fallback time on random directions, matched-random activations, clustering, local complexity, or new summary metrics.

The wrapper reserves the final 20 minutes for `RESULTS.md`, `REPORT.md`, `CHANGELOG.md`, figure checks, and `STOP`.

## Setup (fixed)

### Model and data

- Reuse the existing trained 12-block, 12-head GeLU character-level Shakespeare GPT reconstruction and its checkpoint/config provenance.
- The paper's GPT code/checkpoint were not found in the public repository. Keep the conclusion explicitly qualified unless exact artifacts become available.
- Use held-out Shakespeare text to construct shared prefixes. Do not train or tune the model further for this sanity check.

### Constructing natural minimal pairs

- Each pair must have equal-length sequences and differ at exactly the final input character: `prefix + char_A` versus `prefix + char_B`.
- Freeze all pair IDs before inspecting interpolation curves.
- Prefer endpoint characters that are both plausible after the shared prefix. Build candidates without looking at path shape, using either:
  1. two characters observed after the same prefix in held-out text, when available; or
  2. two high-probability endpoint characters under the model given the shared prefix.
- Include diverse endpoint-character combinations and shared-prefix contexts. Deduplicate prefixes.
- Record endpoint predictions and endpoint logit distance, but do not select or discard pairs based on whether their interpolation later looks plateau-like.
- Degenerate pairs for which `x_A` and `x_B` are numerically indistinguishable may be excluded only by a frozen numerical threshold documented before the full run.

### Spherical interpolation

For `t` in `[0,1]`, slerp the unit directions and linearly interpolate the norms:

\[
\hat h(t)=\frac{\sin((1-t)\theta)}{\sin\theta}\frac{h_A}{\|h_A\|}
+\frac{\sin(t\theta)}{\sin\theta}\frac{h_B}{\|h_B\|},
\]

\[
h(t)=\big[(1-t)\|h_A\|+t\|h_B\|\big]\hat h(t),
\qquad
\theta=\arccos\left(\frac{h_A^\top h_B}{\|h_A\|\|h_B\|}\right).
\]

- Clamp the cosine for numerical safety and implement a documented near-collinear fallback.
- Use the same interpolation grid for all pairs. Primary grid: 101 evenly spaced steps including both endpoints.
- Do not replace slerp with random perturbations, a line from one endpoint, or a path normalized by an arbitrary maximum radius.

### Interpolation and recording layers

- Primary test: interpolate the final-position `resid_post` activation after block 0 and record final logits. This leaves the largest practical number of downstream transformer blocks to form a plateau.
- Layerwise-emergence test: keep interpolation fixed after block 0 and record final-position `resid_post` after every later block, followed by final logits. This directly tests Matthew's observation that plateaus sharpen through successive layers.
- Complementary test, only after the primary run: record final logits while moving the interpolation point across blocks `{0, 2, 4, 6, 8, 10}`. Later interpolation should generally weaken a real plateau because fewer layers remain downstream.
- Keep all sequence positions except the final one unchanged. Because the two sequences share their entire prefix and the model is causal, earlier-position activations should be identical.

### Minimal, boundary-position-invariant summary

- The primary evidence is the raw individual `d(t)` curve.
- Plot the diagonal `d=t` only as a visual non-plateau reference.
- Use one optional scalar summary: transition width

  \[
  w_{10\rightarrow90}=t(d=0.9)-t(d=0.1).
  \]

  Compute crossings from an isotonic copy only for this scalar; always plot the raw curve.
- A candidate plateau has a narrow transition (`w_10->90 <= 0.25`) and at least 10% of the path visibly remaining near each endpoint before and after the transition. Report the count `n/N`; do not hide heterogeneity behind a mean curve.
- Non-monotone curves are reported separately and are not forced into the transition-width statistic.
- Do not use the previous `PI`, which confounds boundary location with plateau sharpness. Do not introduce additional scores unless the raw curves reveal a concrete unresolved ambiguity.

### Required implementation checks

- `t=0` exactly reproduces endpoint A at the patched layer and downstream recording point within numerical tolerance.
- `t=1` exactly reproduces endpoint B within numerical tolerance.
- Direct unpatched forwards for A and B agree with their corresponding patched endpoints.
- Sequences differ only in the final character, and all earlier-position activations match within numerical tolerance.
- Batched interpolation agrees with a single-example reference implementation.
- The relative-distance implementation returns `d(0)=0` and `d(1)=1` for every non-degenerate pair.
- A synthetic step-like path is recognized as narrow-transition and a synthetic linear path is not.

### Reproducibility and reporting rules

- Fix and record pair-generation seed, pair IDs, interpolation grid, model checkpoint, package versions, precision, device, and git commit.
- Save endpoint vectors or stable references sufficient to reproduce each curve.
- Keep raw results separate from plotting code; every figure must regenerate from the saved table.
- Read shared limits in `../BUDGET.md` and operator rules in `../CLAUDE.md` every iteration.
- `RESULTS.md`/`REPORT.md` contain current-best results only; `CHANGELOG.md` contains history.
- Do not `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax; preserve the existing CUDA environment.

## Stages (checklist)

- [x] **S1 - Source/model audit.** The paper model was identified and the absence of released GPT code/checkpoint documented in `MODEL_SPEC.md`.
- [x] **S2 - Model reconstruction.** The 12L/12H GeLU character GPT reconstruction was trained and validated. Retain its provenance; do not treat the previous random-ray assay as this direction's result.
- [x] **S3 - Freeze prompt pairs.** Read Matthew's post/code, generate natural minimal pairs without inspecting interpolation paths, validate exact one-character differences, and save `prompt_pairs.json`.
- [x] **S4 - Implement Matthew's assay.** Implement norm-preserving slerp, last-position patching, two-endpoint relative distance, tidy saving, and all endpoint/synthetic tests.
- [x] **S5 - Primary and layerwise runs.** Run block-0-to-logits curves for every frozen pair, then record layerwise emergence. Inspect and plot individual curves before computing any aggregate.
- [x] **S6 - Depth comparison if needed.** If plateaus appear or the primary result is ambiguous, vary the interpolation layer while keeping final logits as the recording point.
- [x] **S7 - Rewrite verdict.** Replace the previous random-ray conclusion in `RESULTS.md` and `REPORT.md` with the Matthew-style result; retain the old assay only as labeled history. State the qualified scope and create `STOP`.

## Decision rule

- **Plateaus present:** multiple frozen pairs show two endpoint plateaus separated by a narrow boundary in raw final-logit `d(t)` curves, and the same pairs show coherent sharpening across downstream recording layers. Report the exact count and pair IDs; one cherry-picked curve is insufficient.
- **No evidence in this reconstruction:** no frozen pairs show the structure despite valid, non-degenerate endpoints, exact endpoint reproduction, early interpolation, and enough downstream layers. Report `0/N` and the tested layer coverage.
- **Mixed:** only a small subset shows plateaus or curves are strongly non-monotone. Report the heterogeneity directly; do not force a binary conclusion or average it away.

## Out of scope (do NOT)

- Do not use random-direction rays as the primary plateau assay.
- Do not use matched Gaussian/random activations, `Delta PI`, Cliff's delta, or JSD to decide whether Matthew-style plateaus exist.
- Do not average curves before determining whether individual pairs have plateaus.
- Do not cluster, count, or assign semantics to plateau regions yet.
- Do not study checkpoint-to-checkpoint evolution, noise sweeps, steering, or local complexity.
- Do not switch to modular addition, MNIST, a ResNet, or pretrained GPT-2.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with: `On track? <yes/no> - <stage, % done, blocker if any>`.

## Current status

**COMPLETE — Matthew-style plateaus ARE present (decision rule: plateaus present, qualified reconstruction).**

The full two-endpoint assay was run 2026-07-17: 40 frozen minimal pairs (seed 20260717, 0 degenerate,
`results/prompt_pairs.json`), slerp at block-0 `resid_post`, 101-step grid, all implementation checks
passed (endpoint fidelity <1e-3, prefix invariance <1e-4, batched=single <1e-5, synthetic step
detected w=0.089 / line rejected w=0.800). Result: **14/40 pairs pass the frozen plateau rule** in raw
individual final-logit curves (median w=0.309 vs diagonal 0.8; 0 non-monotone); layerwise emergence is
strictly monotone (median w 0.777 at block 1 → 0.445 at block 11 → 0.309 at logits); depth comparison
shows the predicted weakening (median w 0.309 → 0.802 as interpolation moves from block 0 to 10).
`RESULTS.md`/`REPORT.md` rewritten around this verdict; old random-ray assay retained only in
CHANGELOG history. `STOP` written.

## Next step

None — direction complete. If revisited: map plateau boundaries semantically or track their emergence
over training checkpoints (overlaps the "during training" direction).

## Primary references

- Matthew Shinkle and StefanHex, *Activation Plateaus: Where and How They Emerge*: https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Matthew's experiment code: https://github.com/MShinkle/activation_plateau_mechanisms
- Humayun, Balestriero, and Baraniuk, *Deep Networks Always Grok and Here is Why*: https://arxiv.org/abs/2402.15555
- Official grokking-paper repository: https://github.com/AhmedImtiazPrio/grok-adversarial