# PLAN - A visual test of feature-specific plateau geometry in a character-level GPT

> Working folder: create a new direction. Treat `dir13_plateau_on_grok_gpt` as read-only and reuse its trained model and utilities.

## Question

Does the character-level GPT treat meaningful linguistic directions differently from arbitrary directions?

The first-pass prediction is simple: changing an activation along one meaningful feature direction should affect the model sooner than spreading the same change across two feature directions. This would be visually consistent with feature-specific error correction.

This is an exploratory experiment. It does not need to establish a statistical result or prove that the model uses superposition.

## Success criterion

Produce a short `REPORT.md` that answers two questions:

1. Do the two constructed directions change the model in the expected linguistic way?
2. Do the single-feature directions leave the low-change region earlier than their equal mixture?

The report should contain no more than three figures and 900 words. A mixed or negative result is complete if the plots answer the questions clearly.

## Setup (fixed)

- Use the existing trained Shakespeare character-level GPT and the same checkpoint used for the main `dir13` results.
- Intervene only at the last character position, at one early residual-stream layer already used in `dir13`. Do not search over layers.
- Keep the activation length fixed while moving it toward each test direction.
- Measure the effect as JSD from the clean next-character distribution. The small vocabulary makes the full output change easy to inspect.
- Use held-out text examples for every evaluation. No example used to build a direction may be used to test it.
- Read `../CLAUDE.md` and `../BUDGET.md` every iteration.
- Keep only current-best results in `RESULTS.md` and `REPORT.md`; put failed attempts in `CHANGELOG.md`.

## The two candidate features

### 1. Speaker-label state

Compare positions inside an all-capital speaker label at the beginning of a line with positions inside ordinary capitalized text. Match each pair by its final character so that the direction cannot be explained only by character identity.

Expected behavior: moving toward the speaker-label side should increase probability on uppercase letters or a colon.

### 2. Word-continuation state

Compare contexts where the current word continues with another letter against contexts where the word ends next. Again, match each pair by its final character.

Expected behavior: moving toward the continuation side should increase total probability on letters and reduce probability on spaces or punctuation.

For each feature, collect 60 matched pairs. Use 30 pairs to build the direction by averaging the positive-minus-negative activation differences. Reserve the other 30 pairs for the behavioral check.

If either feature lacks enough examples or fails the behavioral check, replace it once with a line-ending direction: contexts whose next character is a newline versus matched contexts whose next character is not. If two usable directions still cannot be built, stop and report that the geometry test could not be grounded.

## Stages

- [ ] **S1 - Validate the feature directions.** For each direction, perturb held-out examples in both signs and check whether the relevant character probabilities move as expected. Make Figure 1 with two simple steering curves and a few example next-character distributions. Keep a direction only if the change is clear and has the correct meaning.

- [ ] **S2 - Compare single features with their mixture.** Use 30 new, ordinary-text activations as anchors. Sweep the same perturbation sizes along feature 1, feature 2, their equal mixture, and 10 random directions. Remove the overlap between the two feature directions before mixing them. Make Figure 2: output JSD versus perturbation size. Show the average curves and several individual anchors so that the reader can see whether the pattern is typical.

- [ ] **S3 - Show the geometry and report it.** Sweep a two-dimensional grid spanning the two feature directions. Make Figure 3 with side-by-side heatmaps for the feature-feature plane and a random-random plane. Color each point by output JSD. Rescale the axes only so their single-direction boundaries line up; do not fit a shape or introduce a new score. Then write and reader-test `REPORT.md` using `Motivation`, `Setup`, `Results`, and `What this does and does not show`.

## How to describe the result

- **Clear first-pass support:** both directions pass the behavioral check; the single-feature curves change earlier than the equal mixture for most anchors; and the feature-plane heatmap shows a visibly broader stable region between the axes than the random plane.
- **Mixed:** the behavioral checks pass, but the curve or heatmap comparison is inconsistent across anchors.
- **No visible support:** the mixture changes at about the same point as the single directions, or the random plane looks similar to the feature plane.
- **Not tested:** two directions with clear linguistic effects could not be constructed.

Use the phrase **"consistent with feature-specific error correction"** for a positive result. Do not claim that the experiment proves error correction, identifies the model's true features, or demonstrates computation in superposition.

## Out of scope

Do not fit a superellipse; report an exponent, correlation, p-value, confidence interval, or new score; train an SAE; scan many features, layers, checkpoints, or model widths; retrain the model; or investigate neurons, Jacobians, splines, or MLP mechanisms. Those are follow-ups only if the first-pass pictures are clear.

## Fallback if time runs short

Finish S1 and the one-dimensional curves in S2. Write the report and state that the two-dimensional geometry remains untested. Do not rush into a partial shape fit.

## On-track check

End each `JOURNAL.md` entry with:

`On track? <yes/no> - <stage, percent done, blocker if any>`

## Current status

S0 done. dir13's checkpoint and corpus lived under `/tmp` and have been wiped, so the same model is
being retrained here with dir13's exact recipe and seeds (`experiments/train_char_gpt.py`); the
reproduction matches dir13's logged loss curve step for step. Matched examples counted
(`results/feature_counts.json`) and ten readable examples of each feature saved
(`results/feature_examples.txt`): both features clear the 60-pair requirement by a wide margin
(speaker-label 23,010 train / 2,742 held-out matched pairs; word-continuation 152,467 / 16,683).
S1-S3 scripts are written and smoke-tested on an intermediate checkpoint.

## Next step

Run S1 on the finished checkpoint and keep or replace each direction on its behavioural check.

## Motivation and references

- The Shakespeare model comes from the character-level GPT experiment in [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555). This plan reuses the model but does not test grokking.
- [Activation Plateaus: Where and How They Emerge](https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge) motivates using an early intervention and a late readout so that plateau structure has room to emerge.
- [Evidence for Feature-Specific Error Correction in LLMs](https://arxiv.org/abs/2606.24964) motivates comparing single feature directions with their mixtures. This first pass keeps its intuitive plots but deliberately omits the fitted geometry and formal statistics.
