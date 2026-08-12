# PLAN - Do internal feature differences explain transition width at matched successor JSD?

> This plan supersedes the previous exploratory `dir20` plan. Run it as a fresh confirmatory
> direction. Existing `dir20` pair banks and outcomes may be used to reuse code, but they must not
> enter the primary analysis.

## Question

Why can two final-token pairs have similar successor JSD but very different transition widths?

Primary hypothesis:

> Among pairs matched on successor JSD and basic endpoint geometry, a pair that engages more
> different downstream MLP features will have a sharper transition, i.e. a smaller width.

This experiment tests an association first. It attempts a causal intervention only if that
association replicates on fresh data.

## Success criterion (definition of done)

`REPORT.md` gives a clear supported / not supported / underpowered verdict using a fresh held-out
bank in GPT-2 Large. It must contain:

1. Matthew's `big`/`in` plateau and `big`/`large` smooth comparison as a sanity check;
2. at least 80 matched within-prefix contrasts locked before any interpolation outcome is computed;
3. a balance table showing that successor JSD and endpoint-geometry confounds are matched;
4. the paired difference in transition width between high- and low-feature-difference pairs;
5. raw curves for representative supporting cases and counterexamples.

A clean null result is complete. When complete, write an empty `STOP` file.

## Fallback

If fewer than 80 contrasts survive the single pre-specified caliper relaxation, run all surviving
contrasts only if there are at least 40, label the result underpowered, finalize the report, and stop.
Do not change the feature metric or continue relaxing the matching rules after seeing widths.

## Setup (fixed)

- Model: pretrained `gpt2-large`, evaluation mode.
- Hooking convention: TransformerLens `resid_post` / equivalent verified Hugging Face hooks.
- Interpolation site: the final token at block-0 `resid_post`.
- Interpolation: Matthew's rescaled SLERP - SLERP the direction and linearly interpolate the L2 norm.
- Grid: 101 equally spaced values of alpha in `[0, 1]`.
- Readout: full final-token logits after the remaining blocks.
- Fresh corpus bank: 300 eligible prefixes from the WikiText-103 **test** split, seed 31, using random
  20-40-token spans. Do not reuse the old validation prefixes or old low-JSD bank.
- Candidate final tokens: the top 24 printable, non-special next tokens under the shared prefix.
- Fixed seeds, `torch.no_grad()`, and float32 metric computation.
- Read `../BUDGET.md` and `../CLAUDE.md` every iteration. Keep `RESULTS.md` and `REPORT.md`
  current-best only; put history in `CHANGELOG.md`.
- Do not install or upgrade PyTorch, torchvision, TransformerLens, JAX, or Flax.

## Fixed measurements

### 1. Successor JSD

For a shared prefix `P` and candidate final tokens `A` and `B`, run the complete prompts `P+A` and
`P+B`. Let `p_A` and `p_B` be their full-vocabulary next-token distributions. Compute

\[
JSD(A,B)=\frac{1}{2}KL(p_A\|m)+\frac{1}{2}KL(p_B\|m),\qquad
m=\frac{p_A+p_B}{2},
\]

in natural-log units. This is inference-time successor JSD, not a training-corpus statistic.

### 2. Internal feature difference (primary independent variable)

Run each endpoint normally and record post-GELU MLP-neuron activations at the final token in blocks
1-35. Block 0 is excluded because its `resid_post` is the interpolation site.

For neuron `j` in block `l`, define its contribution score as

\[
s_{l,j}=|a_{l,j}|\,\|W^{out}_{l,j}\|_2.
\]

For each endpoint, keep the top 64 neurons per block by this score. Treat `(block, neuron)` as the
feature identity. The primary feature-difference score is the Jaccard distance

\[
F(A,B)=1-\frac{|S_A\cap S_B|}{|S_A\cup S_B|}.
\]

Freeze this definition before computing any interpolation widths. Call it an **MLP feature proxy**,
not proof that individual neurons are semantic features.

### 3. Transition width (primary outcome)

For each interpolated output `x_alpha` and endpoint outputs `x_A,x_B`, compute Matthew's relative
distance

\[
d(\alpha)=\frac{\|x_\alpha-x_A\|_2}
{\|x_\alpha-x_A\|_2+\|x_\alpha-x_B\|_2}.
\]

Use `w_TV` as the primary width because it remains defined for non-monotonic curves. Let cumulative
variation be

\[
c_k=\frac{\sum_{i=1}^{k}|d_i-d_{i-1}|}
{\sum_{i=1}^{100}|d_i-d_{i-1}|}.
\]

Then

\[
w_{TV}=\alpha(c=0.75)-\alpha(c=0.25).
\]

A linear response has `w_TV = 0.5`; a sharp step approaches zero. Report raw `d(alpha)`,
`w_10-90`, and a non-monotonicity score as secondary diagnostics. Do not classify pairs using an
arbitrary plateau/no-plateau threshold.

## Stages

- [x] **S1 - Validate the implementation with Matthew's contrast.**
  - Run exact prompts `The house was big` / `The house was in` and `The house was big` /
    `The house was large` in GPT-2 Large.
  - Save the two raw curves together with the linear reference.
  - Require endpoint reconstruction error below `1e-4` and `w_TV(big,in) < w_TV(big,large)`.
  - If this fails, debug and stop before mining new pairs.

- [x] **S2 - Build and lock matched contrasts without looking at width.**
  - For every fresh prefix, form all unordered pairs among the 24 candidate final tokens.
  - Before interpolation, compute successor JSD, `F`, final-logit L2 endpoint distance, block-0
    endpoint angle, block-0 log norm ratio, and mean token surprisal under the shared prefix.
  - Keep candidate pairs with `0.005 <= JSD <= 0.20` and final-logit distance above the bank's 10th
    percentile, preventing near-identical endpoints from making `d(alpha)` noise-dominated.
  - Within each prefix, find two candidate pairs using four distinct final tokens. Label the one with
    larger `F` as `high-F` and the other as `low-F`.
  - Primary calipers: `|Delta JSD| <= 0.01`, standardized Euclidean distance at most `0.50` across
    final-logit distance, block-0 angle, block-0 log norm ratio, and mean surprisal, and
    `Delta F >= 0.10`.
  - Select at most one contrast per prefix: maximize `Delta F`, then minimize confound distance.
  - If fewer than 80 contrasts survive, apply exactly one relaxation:
    `|Delta JSD| <= 0.02`, confound distance at most `0.75`, and `Delta F >= 0.08`.
  - Save the chosen prompts, endpoint metrics, and matching version to `results/matched_pairs.json`.
    Hash this file and record the hash in `JOURNAL.md`. Only then may S3 compute interpolation curves.

- [x] **S3 - Test the matched prediction.**
  - Run the identical block-0 interpolation for both members of every locked contrast.
  - For contrast `i`, compute
    `Delta w_i = w_TV(high-F)_i - w_TV(low-F)_i`.
    The prediction is `Delta w < 0`.
  - Primary summaries: median `Delta w`, its prefix bootstrap 95% CI, and the fraction of contrasts
    with `Delta w < 0`. A paired permutation p-value may be reported as a secondary summary.
  - Call the association supported only if there are at least 80 contrasts, median `Delta w <= -0.05`,
    at least 60% have the predicted sign, and the 95% CI lies below zero. Otherwise report the null or
    underpowered result and stop.
  - Required figures:
    - `plots/matching_balance.png`: high-F versus low-F balance for every matched variable;
    - `plots/matched_widths.png`: paired low-F to high-F width lines plus the `Delta w` distribution;
    - `plots/example_curves.png`: five strongest supporting contrasts and five strongest
      counterexamples, with prompt tokens, JSD, `F`, and `w_TV` shown.

- [x] **S4 - Conditional causal test; run only if S3 is supported.**
  - Keep the original block-0 endpoint activations and every SLERP vector fixed. Do not ablate block 0,
    rerun upstream blocks, or regenerate the path.
  - In blocks 1-35, take neurons in the symmetric difference of the two endpoint top-64 feature sets.
    At each alpha, replace only their post-GELU activations with the linear endpoint interpolation
    `a'_j(alpha) = (1-alpha)a_j(A) + alpha a_j(B)`.
  - This preserves both endpoints while removing nonlinear switching in the differential neurons.
  - Compare with an equal-size control set matched by block, mean contribution magnitude, endpoint
    activation difference, and output-weight norm.
  - The causal prediction is that differential-neuron linearization increases `w_TV` more than the
    matched control. Report paired effects and raw curves. If it does not, retain the S3 result as an
    association only.

## Required outputs

- `results/matched_pairs.json`: locked pre-outcome matched-pair manifest.
- `results/matched_sweeps.npz`: alpha grid and raw `d(alpha)` curves.
- `results/matched_metrics.json`: endpoint metrics, widths, diagnostics, and paired effects.
- `plots/matthew_sanity.png`.
- `plots/matching_balance.png`.
- `plots/matched_widths.png`.
- `plots/example_curves.png`.
- `RESULTS.md`: compact tables and numerical results.
- `REPORT.md`: Question, Methods, Results, Limitations, and one clear verdict.

## Out of scope (do not drift)

- No Pythia, OPT, GPT-2 Small/Medium, model-size comparison, depth sweep, training-time sweep, or
  continuation-offset study.
- No new feature metric after seeing widths; head, SAE, residual-distance, spline-density, and local-
  complexity measures are follow-up directions, not extra chances for a positive result.
- No causal localization unless S3 passes its frozen gate.
- No claim that successor JSD causes width, or that the neuron proxy identifies semantic features.
- The four earlier hand-written prompt pairs may appear only as qualitative examples, not evidence.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> - <stage, % done, blocker if any>`

## Current status

**COMPLETE — verdict: supported, confirmed by a pre-registered independent replication (S3R), with a
causal test behind it.** S1 passed its gate
($w_{TV}$ 0.012 vs 0.292, endpoint error 3.5e-7). S2 locked **101** matched contrasts under the single
pre-specified relaxation (manifest sha256 `2415f5ff6dfcf88fb9cc7a67b87c93d859434296310f4b8d406c6f545e23ff56`,
recorded before any sweep). S3 met all four gate clauses: median `Delta w = -0.0708`, 95% CI
`[-0.0866, -0.0582]`, 82.2% predicted sign, permutation p < 1e-4. S4 (conditional, unlocked by S3)
supported: median `w_TV` 0.144 -> 0.471 for differential-neuron linearization against 0.167 for the
matched control, 202/202 pairs.

**One deviation from this plan, recorded in REPORT.md Limitations, CHANGELOG.md and JOURNAL.md:** the
bank was enlarged from the specified 300 prefixes to all 1395 eligible WikiText-103 test paragraphs,
because 300 yielded only 21 contrasts — below this plan's own 40-contrast fallback floor. Every metric
definition, eligibility filter and caliper was unchanged and no interpolation width had been computed.
Because that stopping rule was frozen in advance, S1-S4 are labelled the **amended analysis**
throughout both deliverables (operator feedback #1).

**S3R — pre-registered independent replication: PASSED.** Protocol frozen in JOURNAL.md
2026-08-12T02:44Z, before any replication data was scored: WikiText-103 **train** split (untouched by
any analysis here), bank size fixed at exactly 1400 prefixes, run once with no enlargement, re-seeding,
re-drawing or second relaxation, everything else identical, same four-clause gate. Outcome: 5 contrasts
under the primary calipers -> 99 under the one pre-specified relaxation; median `Delta w = -0.0641`,
95% CI `[-0.0908, -0.0426]`, 78.8% predicted sign, permutation p < 1e-4 — all four clauses met.
Manifest sha256 `ed1df0866f012b6195521dcda0d81306c7c6cb9d00e5dca2b30cda62e9af6d6b`, recorded before its
first sweep. The confirmatory claim for the matched association now rests on S3R; S4's causal result
still rests on the amended bank only.

## Next step

None required — the success criterion is met and the association is confirmed by S3R. Three extensions
if the direction is continued:
1. a pre-registered replication of the **causal** experiment (S4) on the S3R bank, which would put the
   mechanism on the same footing as the association;
2. the **minimal sufficient differential set** (S4 shows 1.7% of neurons suffice, not that they are
   necessary at that size);
3. the same locked matched design with an **SAE-feature or attention-head** version of `F`, testing
   whether "different machinery" is a neuron-level or a feature-level fact.
