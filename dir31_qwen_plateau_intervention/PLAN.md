# PLAN â Does the plateau follow country identity or the later answer?

> Suggested folder: `qwen_plateau_intervention`. Update Current status / Next step and tick stages each iteration. Keep PLAN/JOURNAL/RESULTS/REPORT/CHANGELOG as working memory.

## Success criterion

Find an early intervention that changes the Tokyo/Berlin answer preference, then test whether the plateau boundary follows it while an independent country readout stays stable. Null or inseparable effects are complete results; do not increase strength indefinitely.

## Setup (fixed)

- Read `../BUDGET.md` and `../CLAUDE.md` every iteration. Do not reinstall torch, torchvision, transformer_lens, cupbearer, jax, or flax.
- Reuse dir30's Qwen3-0.6B, `mwhanna/qwen3-0.6b-transcoders-lowl0`, pinned revisions, loader, and settings. Keep native MLPs and normal attention; load dictionaries layerwise.
- Reuse dir30's correct LRE samples and currency/language controls. Existing feature labels are provisional. No large new dataset.
- Target pair: `The capital of Japan` / `The capital of Germany`. Teacher-force the identical bridge ` is` to measure the later city prediction. Verify tokenization and both baseline answers.
- Tune on one correct capital-template pair; reserve the original pair and remaining correct paraphrases for evaluation. These are held-out phrasings, not independent facts. Report all of them.

## Stages

### [x] S1 â Reproduce the plateau and establish readouts

- Reuse the original interpolation rule at the layer-0 block output, country position: Japan to Germany, 41 points. Recompute downstream states; no stale KV cache.
- Record country-position residual states across layers, immediate next-token distributions, and city probabilities after ` is`. Measure Tokyo-minus-Berlin logits, or complete-answer log probabilities for multi-token answers.
- Locate plateau-bearing layers using endpoint-distance curves and adjacent-step L2 changes. A top-token switch alone is not a plateau.
- Train a small frozen linear country readout on balanced, explicit-country, non-capital prompts (currency/language/identity); validate on held-out template families. Never train on capital answers or edited states. Failed validation means country preservation is unverified; successful decoding still does not prove all country information survives.

### [x] S2 â Obtain a directed, early intervention

- Rank features by output-gradient projection onto decoder directions, weighted by donor-minus-recipient activation. Include either endpoint's active features, not just dir30's sets. Validate with actual native-model interventions; no full attribution graph.
- First test at the final `is` position as a positive control. Then scan the country position, layers 0â26. Layer 27 has no later attention to carry a country-position MLP edit to `is`.
- At one layer, patch the top 1, 5, or 10 available directions toward the opposite-country donor:

  `MLP_new = MLP_native + beta * sum_i[(a_donor_i - a_current_i) * decoder_i]`

  Test both directions with `beta = 0, 0.5, 1, 2`; preserve reconstruction error. If needed, try two-adjacent-layer groups. Label extrapolation separately from natural donor replacement.
- Select the smallest successful edit on tuning prompts, then freeze sites/features/donors/strength. On reserved prompts, check city-preference reversal, greedy completions, country readout, and `is` preservation. A margin reversal need not change top-1.
- Compare three same-site, feature-count/norm-matched random controls. Check beta=0, hook removal, and collateral effects on country-context prompts.
- If only final-`is` edits work, or effective early edits also change country identity, stop with that limitation.

### [x] S3 â Does the boundary follow the edited answer?

Only with a usable early intervention: repeat the identical 41-point path with baseline, Tokyo-directed, Berlin-directed, and control edits. Recompute current activations, but never reselect features along the path.

Make one three-panel figure with a common interpolation axis:

1. Native hidden-state plateau curves and adjacent-step changes.
2. Frozen country-readout scores.
3. Tokyo/Berlin answer probabilities after the fixed bridge.

Read native states downstream of the edit, not clamped activations. Keep unedited endpoint references fixed; show absolute changes to expose collapse. An unchanged upstream plateau cannot support the input hypothesis.

| Observation | Interpretation |
|---|---|
| Country readout stays stable; answer preference and plateau boundary move together | Supports an answer-related contribution to that layer's plateau. |
| Answer changes but plateau does not | Separates that answer effect from the measured plateau; does not alone prove an input explanation. |
| One city remains preferred across the path, but the plateau still tracks the country switch | Supports a country-related component; inspect the full output distribution because the same top answer does not mean identical predictions. |
| Country and answer both change, or controls have comparable effects | Not disentangled / inconclusive. |

## Deliverables and fallback

- Concise current-best `REPORT.md` / `RESULTS.md`: intervention dose-response, the three-panel plateau comparison if qualified, example completions, and a plain-language verdict. Save figures, raw curves, chosen features, config, and one rerun command.
- If time is short or the early-intervention gate fails, finish S1âS2 and explain the blocker. Reserve the final 20 minutes for reporting; write an empty `STOP` when done.
- No new model/dictionary training, large sweeps, significance testing, or claims of pure features/planning. A layer-dependent or inconclusive answer is acceptable.

## On-track check

End each JOURNAL entry with `On track? <yes/no> â <stage, % done, blocker>`.

## Current status

Complete (2026-10-09). Success criterion met as an "inseparable" result.

- S1: plateau reproduced at the country token in layers 20-27 (layer 26: largest step 2.65x the mean, boundary t = 0.512); answer boundary t = 0.531; country readout validated leave-one-family-out (accuracy 1.00 at every layer).
- S2: no country-token or bridge-token edit reverses the margin at beta <= 1. The pre-declared rule selects layer 16, top-10 features, beta = 2 (extrapolation). It reverses the answer on all reserved prompts (controls do not), but it also flips the country readout and the currency/language answers on 6/6 mid-sentence context prompts. Gate outcome: effective early edit changes country identity.
- S3: path rerun under the frozen edit as descriptive evidence. Answer boundary and readout crossing vanish and the layer-26 largest step falls to 34-39% of baseline; six random controls keep all three. Plan verdict row: "country and answer both change: not disentangled".
- Deliverables: REPORT.md, RESULTS.md, plots/fig1-3, results/ (raw curves, frozen_edit.json, run_config.json), experiments/run_all.sh. STOP written.

## Next step

None. Do not reopen without new operator feedback. A decisive test would need an edit that moves the capital answer while currency and language answers stay correct; the plan's scan found none.

## Method references

- [Existing dir30 report](https://github.com/billy000400/marsv/blob/main/dir30_qwen_city_country_features/REPORT.md)
- [Anthropic: planning and feature-group interventions](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [Anthropic: intervention methods and limitations](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [Activation Plateaus: Where and How They Emerge](https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge)
