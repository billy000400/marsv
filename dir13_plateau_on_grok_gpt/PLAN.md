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

**COMPLETE — Matthew-style plateaus ARE present (decision rule: plateaus present, qualified reconstruction). Independently re-verified 2026-07-17.**

The full two-endpoint assay was run 2026-07-17 and re-run end-to-end later the same day with a
bit-exact reproduction of every number: 40 frozen minimal pairs (seed 20260717, 0 degenerate,
`results/prompt_pairs.json`), slerp at block-0 `resid_post`, 101-step grid, all implementation checks
passed (endpoint fidelity <1e-3, prefix invariance <1e-4, batched=single <1e-5, synthetic step
detected w=0.089 / line rejected w=0.800). Result: **14/40 pairs pass the frozen plateau rule** in raw
individual final-logit curves (IDs 0,4,5,6,7,9,14,20,21,22,28,34,36,37; median w=0.309 vs diagonal
0.8; 0 non-monotone); layerwise emergence is strictly monotone (median w 0.777 at block 1 → 0.445 at
block 11 → 0.309 at logits); depth comparison shows the predicted weakening (median w 0.309 → 0.802
as interpolation moves from block 0 to 10). `RESULTS.md`/`REPORT.md` are rewritten around this
verdict; the old random-ray assay is retained only in CHANGELOG history. Operator feedback #1 and #2
addressed (both files `.addressed.md`). `STOP` written.

## Next step

None — direction complete. If revisited: map plateau boundaries semantically or track their emergence
over training checkpoints (overlaps the "during training" direction).

## Primary references

- Matthew Shinkle and StefanHex, *Activation Plateaus: Where and How They Emerge*: https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Matthew's experiment code: https://github.com/MShinkle/activation_plateau_mechanisms
- Humayun, Balestriero, and Baraniuk, *Deep Networks Always Grok and Here is Why*: https://arxiv.org/abs/2402.15555
- Official grokking-paper repository: https://github.com/AhmedImtiazPrio/grok-adversarial