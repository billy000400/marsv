# PLAN â Direction: Context Dependence of `big` â Vocabulary Transition Widths

> Working folder: choose a new direction folder under `marsv` (do not overwrite Direction 27). Agent REWRITES "Current status"/"Next step" + ticks stages each iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.

## Research question

Does the transition-width pattern for interpolation from the same anchor token (` big`) to other vocabulary tokens depend on the prefix context?

The experiment repeats the Direction 27 vocabulary sweep under four different fixed prefixes. For each prefix, interpolate from the final-token representation for ` big` to the corresponding representation for every usable GPT-2 token, plot the distribution of transition widths, and inspect the tokens with transition width greater than 0.3.

The goal is descriptive and exploratory: identify whether the wide-transition tokens show recognizable patterns within each context, and whether those patterns remain similar when only the prefix changes.

## Success criterion (definition of "done")

`REPORT.md` contains:

1. One section for each of the four prompts below.
2. For each prompt, a histogram of final transition widths over all usable vocabulary tokens.
3. For each prompt, the complete list of tokens with transition width `> 0.3`, sorted by width, plus a short qualitative description of any visible pattern.
4. A final comparison section describing which patterns are shared across contexts and which appear context-dependent.
5. A simple cross-context comparison of the `w > 0.3` token sets (for example: tokens shared by all four contexts, shared by some contexts, and unique to one context).

Null results are complete results. If the four contexts produce no interpretable difference, report that directly.

## Fallback (if time runs short)

Complete all four vocabulary sweeps and save the four transition-width histograms and the four `w > 0.3` token lists. The cross-context interpretation can remain brief.

## Setup (fixed; match Direction 27)

- **Model:** pretrained `gpt2-large`, evaluation mode.
- **Tokenizer:** standard GPT-2 tokenizer.
- **Anchor token:** the single token ` big`.
- **Partner tokens:** sweep over every usable vocabulary token, following the same token-validity rules as Direction 27. Do not retokenize decoded strings to create endpoints.
- **Interpolation site:** final token, block-0 `resid_post`, as in Direction 27.
- **Interpolation path:** use the same interpolation implementation and alpha grid as Direction 27. Do not introduce a new interpolation method for this direction.
- **Readout:** propagate the interpolated representation through the remaining model and measure the final-token output representation/logits exactly as in Direction 27.
- **Transition width:** use the same Direction 27 definition, `w = t_0.9 - t_0.1`, with the same threshold-crossing implementation. Keep the implementation identical across all four sections.
- **Primary threshold for inspection:** `w > 0.3`.
- Save at minimum: prompt/context ID, partner token ID, decoded token, transition width, and any validity/failure flag used in Direction 27.
- Use the existing Direction 27 code where possible rather than reimplementing the metric.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` â read both every iteration.**
- **Deliverable hygiene (see CLAUDE.md):** RESULTS.md/REPORT.md = current-best only, no history; CHANGELOG.md = the history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, flax.**

## Stage 0 â Reproduce the Direction 27 control

- [x] Locate the exact Direction 27 implementation used for the GPT-2 Large vocabulary sweep.
- [x] Confirm the model, hook point, interpolation path, alpha grid, transition-width calculation, and usable-token filtering are unchanged.
- [x] Run a small sanity check on the original Direction 27 setup before starting the four new contexts.
- [x] Do not continue if the Direction 27 control cannot be reproduced approximately; document the discrepancy first.

## Section 1 â `My house is big`

Prompt:

`My house is big`

Experiment:

- [ ] Keep the prefix `My house is` fixed.
- [ ] Use ` big` as endpoint A.
- [ ] For every usable vocabulary token `B`, construct the endpoint with the same fixed prefix and token `B` in the final position.
- [ ] Interpolate ` big â B` using the Direction 27 procedure.
- [ ] Compute the final transition width for every valid partner token.

Outputs:

- [ ] Plot and save a histogram of transition widths over the full usable-token sweep.
- [ ] Extract every token with `w > 0.3`.
- [ ] Save these tokens sorted from largest to smallest transition width.
- [ ] Inspect the decoded tokens and write a short qualitative note describing any obvious regularities. Do not introduce clustering or formal statistics.
- [ ] Save a small number of representative transition curves only when useful for understanding the pattern.

## Section 2 â `The opposite of small is big`

Prompt:

`The opposite of small is big`

Experiment:

- [ ] Keep the prefix `The opposite of small is` fixed.
- [ ] Use ` big` as endpoint A.
- [ ] Sweep the same usable vocabulary token set as Section 1.
- [ ] Interpolate ` big â B` with the identical Direction 27 procedure.
- [ ] Compute the final transition width for every valid partner token.

Outputs:

- [ ] Plot and save a histogram of transition widths.
- [ ] Extract every token with `w > 0.3`.
- [ ] Save the tokens sorted by transition width.
- [ ] Inspect them qualitatively for recognizable lexical, semantic, syntactic, or token-form patterns.
- [ ] Keep the analysis descriptive; do not add new statistical tests.

## Section 3 â `He dreamed big`

Prompt:

`He dreamed big`

Experiment:

- [ ] Keep the prefix `He dreamed` fixed.
- [ ] Use ` big` as endpoint A.
- [ ] Sweep the same usable vocabulary token set.
- [ ] Interpolate ` big â B` using the identical procedure.
- [ ] Compute the final transition width for every valid partner token.

Outputs:

- [ ] Plot and save a histogram of transition widths.
- [ ] Extract every token with `w > 0.3`.
- [ ] Save the tokens sorted by transition width.
- [ ] Inspect the high-width tokens qualitatively and record any obvious patterns.

## Section 4 â `The elephant was big`

Prompt:

`The elephant was big`

Experiment:

- [ ] Keep the prefix `The elephant was` fixed.
- [ ] Use ` big` as endpoint A.
- [ ] Sweep the same usable vocabulary token set.
- [ ] Interpolate ` big â B` using the identical procedure.
- [ ] Compute the final transition width for every valid partner token.

Outputs:

- [ ] Plot and save a histogram of transition widths.
- [ ] Extract every token with `w > 0.3`.
- [ ] Save the tokens sorted by transition width.
- [ ] Inspect the high-width tokens qualitatively and record any obvious patterns.

## Section 5 â Cross-context comparison

The main question is whether changing the prefix changes which `big â token` interpolations have wide transitions.

- [ ] Compare the four histograms side by side using identical bins and axis ranges.
- [ ] Compare the four sets of tokens satisfying `w > 0.3`.
- [ ] Report, in a simple table:
  - tokens with `w > 0.3` in all four contexts;
  - tokens with `w > 0.3` in 2â3 contexts;
  - tokens with `w > 0.3` in only one context.
- [ ] For tokens that cross the `0.3` threshold in some contexts but not others, inspect the actual widths before interpreting the difference. Distinguish large changes from borderline threshold crossings.
- [ ] Summarize the qualitative result using one of these descriptions, without forcing a stronger claim:
  - **mostly context-invariant:** roughly the same tokens have wide transitions across prefixes;
  - **partly context-dependent:** there is a stable core plus systematic context-specific changes;
  - **strongly context-dependent:** the identity of wide-transition tokens changes substantially across prefixes;
  - **mixed / unclear:** no simple pattern is visible.
- [ ] Note any recurring token categories that appear across several contexts, and any categories that appear only in a specific context.

## Figures / artifacts

Required:

- `plots/section1_width_histogram.png`
- `plots/section2_width_histogram.png`
- `plots/section3_width_histogram.png`
- `plots/section4_width_histogram.png`
- machine-readable per-token results for each section
- sorted `w > 0.3` token lists for each section
- one compact cross-context comparison table

Optional only if they clarify an observed pattern:

- representative `d(t)` curves for a few high-width and low-width tokens
- a four-context width comparison for a handful of especially informative tokens

## Out of scope (do NOT)

- Do not run all token-pair Ã token-pair combinations.
- Do not change the anchor away from ` big` in this direction.
- Do not change models; use GPT-2 Large only.
- Do not add Spearman/Kendall correlations, clustering, classifiers, embeddings, or other sophisticated analyses in the first pass.
- Do not introduce a new plateau definition or optimize the `0.3` threshold after seeing the results.
- Do not infer a causal mechanism from semantic-looking token groups. This experiment asks whether the pattern changes with context.
- Do not expand to additional prompts until the four requested sections and the cross-context comparison are complete.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> â <stage, % done, blocker if any>`

## Current status

Stage 0 done. s1 complete (692 tokens W_final > 0.3). s2 running (8/25 shards at 21:45; queue.sh; shards ~193 s while GPU shared with dir29,
~95 s alone; s2 expected ~22:45, s3 ~23:30, s4 ~00:15 on 2026-09-29/30). REPORT.md drafted: research question, Methods,
Section 1 qualitative note done; histograms, Sections 2-4, cross-context, Summary, Conclusion marked PENDING.

## Next step

When results/queue.done exists: python experiments/analyze.py && python experiments/report_lists.py; inspect s2-s4 lists;
fill the PENDING parts of REPORT.md (embed 4 histograms, paste fragments into Appendix A/B), write RESULTS.md,
run ../check_render.py. If still running, do nothing expensive (do not start parallel sweeps: GPU throughput is shared).
