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

### [x] S4 - Operator addendum (2026-10-11): additive Berlin promotion that preserves Japan

Authority: `human_feedback_additive_berlin.md` and its manifest in `.tasks/`. That text is verbatim and wins over this summary.

Question: can an additive intervention at the country token promote the later Berlin answer while preserving Japan-related information? The S2 donor replacement changed country identity. It is not a validated Berlin-specific direction and is not reused.

Fixed rules

- Reuse dir31's model, transcoders, hooks, country readout and prompts. Do not reinstall the ML environment.
- Existing files under `results/`, `plots/` and `experiments/` stay unchanged. New outputs go to `results/additive/` and `plots/additive/`; new code goes in new files under `experiments/`.
- Tune on "The capital city of Japan is" only. The original capital prompt and the remaining correct paraphrase are reserved. Never retune on reserved results.
- Edit at the Japan token: `y_new = y_native + beta * sum_i(reference_amplitude_i * decoder_i)` with beta = 0, 0.5, 1, 2. No subtraction of current activations, no removal of Japan features, no replacement of the native MLP, no larger strengths. Downstream states are recomputed normally.

Sub-stages

- [x] S4.1 Candidates and amplitudes. dir30's Berlin candidates outside its Germany candidate set, layers 1-26 (a candidate filter, not proof of semantic independence). Reference amplitude = median positive activation on dir30's Berlin-answer discovery prompts at that layer, never from the Germany token of a capital prompt. Record sample support and extraction position.
- [x] S4.2 Screen on the tuning prompt. Single features first, ranked by the actual increase in logit(Berlin) - logit(Tokyo) after the fixed bridge " is". Apply the three fixed pilot tolerances during selection: layer-26 country readout changes by at most 10% of the unedited Japan-Germany readout gap for the matched capital template; each valid mid-sentence Japanese language/currency control keeps its original correct top-1 and loses at most 0.05 absolute probability; " is" stays top-1 at the capital-prefix country token and loses at most 0.05 absolute probability. First-token controls known to invalidate the readout are excluded. Then combine the top 3 and top 5 preservation-passing features within a layer, where available, and recheck every condition. No broader search.
- [x] S4.3 Freeze and evaluate. Freeze the smallest qualifying edit from tuning results only. Evaluate reserved prompts against baseline and three same-site, feature-count/norm-matched random controls. The pilot passes only if every reserved prompt gains at least +1 logit toward Berlin, the answer effect exceeds every matched random control, and all preservation checks still pass. Show probabilities, readout changes and greedy completions. Berlin need not become top-1. Outcome 2026-10-11: the rule declared before the screen (+1 on tuning) selected nothing. By the operator's resolution of "qualifying" (given after the tuning screen and before any reserved run) the largest preservation-passing setting was frozen (layer 17, candidates 15678 + 70371 + 150354, beta 2; `results/additive/frozen_edit.json`) and run once on both reserved prompts with three matched random controls: +0.381 and +0.293 logits against at most +0.201 for the controls. Criterion 1 (+1) not met, criterion 2 (above every control) met, criterion 3 (preservation) not met because P(" is") falls by 0.095. The pilot does not pass (`results/additive/pilot_verdict.json`).
- [x] S4.4 Conditional plateau test, only after the pilot passes. Apply the same fixed additive vector at every point of the original 41-point Japan-Germany path. Check the country readout against the unedited path, both endpoints included; if preservation fails along the path, label the plateau result confounded. Plot (1) hidden-state d(t) with each condition's own edited endpoints, (2) adjacent-step L2 changes divided by that condition's mean step, (3) country readout and Tokyo/Berlin probabilities. Also keep fixed-baseline-reference curves, raw step sizes and endpoint separation. Read native states downstream of the edit. Outcome 2026-10-11: NOT run, because the pilot did not pass (criteria 1 and 3); there is no plateau result to label.
- [x] S4.5 Deliverables. Clearly separated addendum sections in `RESULTS.md` and `REPORT.md`, JOURNAL and CHANGELOG entries, one rerun command. The existing S1-S3 results stay in place.

Stop rule: report the best answer effect and its country cost, failures included. If no setting passes, conclude: "No selective additive intervention found in this bounded candidate search." Do not conclude that disentanglement is impossible, enlarge strengths, or relabel country features as city features.

## Deliverables and fallback

- Concise current-best `REPORT.md` / `RESULTS.md`: intervention dose-response, the three-panel plateau comparison if qualified, example completions, and a plain-language verdict. Save figures, raw curves, chosen features, config, and one rerun command.
- If time is short or the early-intervention gate fails, finish S1âS2 and explain the blocker. Reserve the final 20 minutes for reporting; write an empty `STOP` when done.
- No new model/dictionary training, large sweeps, significance testing, or claims of pure features/planning. A layer-dependent or inconclusive answer is acceptable.

## On-track check

End each JOURNAL entry with `On track? <yes/no> â <stage, % done, blocker>`.

## Current status

S1-S3 complete (2026-10-09); success criterion met as an "inseparable" result. Reopened 2026-10-11 for stage S4 by operator feedback `human_feedback_additive_berlin.md`. The independent content review approved S4 on 2026-10-12 (UTC); the feedback file is now `human_feedback_additive_berlin.md.addressed.md` and STOP is restored.

- S1: plateau reproduced at the country token in layers 20-27 (layer 26: largest step 2.65x the mean, boundary t = 0.512); answer boundary t = 0.531; country readout validated leave-one-family-out (accuracy 1.00 at every layer).
- S2: no country-token or bridge-token edit reverses the margin at beta <= 1. The pre-declared rule selects layer 16, top-10 features, beta = 2 (extrapolation). It reverses the answer on all reserved prompts (controls do not), but it also flips the country readout and the currency/language answers on 6/6 mid-sentence context prompts. Gate outcome: effective early edit changes country identity.
- S3: path rerun under the frozen edit as descriptive evidence. Answer boundary and readout crossing vanish and the layer-26 largest step falls to 34-39% of baseline; six random controls keep all three. Plan verdict row: "country and answer both change: not disentangled".
- Deliverables (S1-S3): REPORT.md, RESULTS.md, plots/fig1-3, results/ (raw curves, frozen_edit.json, run_config.json), experiments/run_all.sh.
- S4 (2026-10-11): complete and approved on the third review (rounds 1 and 2 were rejected and repaired, round 2 on wording and disclosure only). S4.1: 69 candidates in layers 3-26, reference amplitudes 0.14-42.5 from 7 discovery prompts of 2 facts. S4.2: 256 settings on the tuning prompt (207 single-feature, 49 within-layer groups, beta 0.5/1/2); 250 pass all three preservation checks; answer effects range from -0.286 to +0.314 logits, so none reaches +1. S4.3: under the operator's resolution of "qualifying" the largest passing setting (layer 17, top-3 group, beta 2) was frozen and run once on both reserved prompts: +0.381 and +0.293 logits, above all three matched random controls (at most +0.201; weak evidence, since the three take their directions from a pool of 64 decoder rows and share two of them), far below +1, and P(" is") falls by 0.095 at the reserved prefix (tolerance 0.05). The pilot does not pass. S4.4 was not run. Conclusion: "No selective additive intervention found in this bounded candidate search." Outputs: `results/additive/`, `plots/additive/`, `experiments/run_additive.sh`. Deliverables: RESULTS.md Section 5 (5.1-5.11, Figures 4-7), REPORT.md section "Addendum: can an additive edit promote Berlin while preserving Japan?" (Figure 4; 4,992 of 5,000 words, 4 of 8 figures), CHANGELOG and JOURNAL entries. Task manifest state: review_pending.

## Next step

None. Do not reopen without new operator feedback. If it is reopened: REPORT.md has 8 words of budget left, so put detail in RESULTS.md. The reserved prompts have been run once with the frozen edit: do not rerun them with any other edit, do not redraw the random controls, do not retune, do not run the path (the pilot did not pass) and do not raise beta above 2.

## Method references

- [Existing dir30 report](https://github.com/billy000400/marsv/blob/main/dir30_qwen_city_country_features/REPORT.md)
- [Anthropic: planning and feature-group interventions](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [Anthropic: intervention methods and limitations](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [Activation Plateaus: Where and How They Emerge](https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge)
