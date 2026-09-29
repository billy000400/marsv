# PLAN â Direction: `France` â Vocabulary Sweep

> Working folder: create a new direction folder under `marsv` (do not overwrite Direction 27). Agent REWRITES "Current status"/"Next step" + ticks stages each iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.

## Research question

For a fixed factual prefix, which GPT-2 tokens produce a smooth / non-plateau transition when interpolating away from the token ` France`?

Use the prompt:

`The capital of France`

Keep the prefix `The capital of` fixed and use ` France` as endpoint A. Sweep endpoint B over the GPT-2 vocabulary, using the same interpolation and transition-width procedure as Direction 27.

The main motivation is exploratory. With examples such as `big â large`, a smooth transition can plausibly be related to similarity between the endpoint tokens. For `France`, it is much less obvious what kinds of tokens should behave similarly. This sweep asks whether the low-width / non-plateau partner tokens form any recognizable groups.

## Success criterion (definition of "done")

`REPORT.md` contains:

1. A histogram of transition widths for ` France â B` over all usable GPT-2 vocabulary tokens.
2. A complete list of non-plateau candidates using the same Direction 27 threshold, i.e. tokens with transition width `w <= 0.3`, sorted from smallest to largest width.
3. A compact qualitative inspection of the lowest-width tokens, asking whether they show recognizable lexical, semantic, syntactic, geographic, token-form, or other patterns.
4. Representative `d(t)` curves for a small number of especially informative low-width tokens, plus a few plateau-like controls with `w > 0.3`.
5. A short conclusion stating whether the non-plateau tokens appear structured or heterogeneous. Null / unclear results count as complete.

## Fallback (if time runs short)

Finish the full vocabulary sweep, save the transition-width histogram and sorted `w <= 0.3` token list, and manually inspect the lowest-width tokens. Do not add more analysis until these basic outputs are complete.

## Setup (fixed; match Direction 27)

- **Model:** pretrained `gpt2-large`, evaluation mode.
- **Tokenizer:** standard GPT-2 tokenizer.
- **Prompt:** `The capital of France`.
- **Fixed prefix:** `The capital of`.
- **Anchor token / endpoint A:** the single token ` France`.
- **Partner tokens / endpoint B:** every usable token in the GPT-2 vocabulary, following the same token-validity rules as Direction 27.
- Construct each endpoint as the same fixed prefix followed by exactly one final token. Do not retokenize decoded strings to create endpoints.
- **Interpolation site:** final token, block-0 `resid_post`, exactly as in Direction 27.
- **Interpolation path and alpha grid:** reuse the Direction 27 implementation unchanged.
- **Readout:** propagate the interpolated representation through the remaining model and compute `d(t)` / transition width exactly as in Direction 27.
- **Transition width:** use the same Direction 27 definition, `w = t_0.9 - t_0.1`, with the same threshold-crossing implementation.
- **Plateau threshold:** preserve the existing `0.3` cutoff. Treat `w > 0.3` as plateau-like and `w <= 0.3` as the primary non-plateau candidate set for this exploratory study.
- Save at minimum: partner token ID, decoded token, transition width, and any validity/failure flag used in Direction 27.
- Reuse Direction 27 code rather than reimplementing the metric.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` â read both every iteration.**
- **Deliverable hygiene:** RESULTS.md/REPORT.md = current-best only, no history; CHANGELOG.md = history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, flax.**

## Stage 0 â Verify Direction 27 compatibility

- [ ] Locate the exact Direction 27 GPT-2 Large vocabulary-sweep implementation.
- [ ] Confirm model, tokenizer, hook point, interpolation path, alpha grid, `d(t)`, transition-width calculation, threshold, and usable-token filtering.
- [ ] Run a small sanity check with ` France` and a handful of partner tokens before launching the full vocabulary sweep.
- [ ] If any setup differs from Direction 27, fix it before proceeding rather than silently changing the method.

## Section 1 â Full `France â vocabulary` sweep

For every usable GPT-2 vocabulary token `B`:

- [ ] Construct endpoint A: `The capital of France`.
- [ ] Construct endpoint B: `The capital of` + token `B`.
- [ ] Interpolate the final-token representation ` France â B` using the Direction 27 procedure.
- [ ] Compute `d(t)` and transition width `w`.
- [ ] Save one result row per partner token.

Required outputs:

- [ ] Plot a histogram of transition widths over the complete valid vocabulary sweep.
- [ ] Use the same histogram conventions / binning style as Direction 27 where possible.
- [ ] Mark `w = 0.3` on the plot so the non-plateau candidate region is visually clear.
- [ ] Report the number and fraction of valid tokens with `w <= 0.3` and `w > 0.3` as simple descriptive counts only.

## Section 2 â Inspect the non-plateau tokens

Primary object of interest: all partner tokens with `w <= 0.3`.

- [ ] Save the complete set sorted from smallest to largest transition width.
- [ ] Display at least the lowest-width tokens in a readable table containing token ID, decoded token, and transition width.
- [ ] Inspect decoded tokens directly for obvious regularities.
- [ ] Ask simple qualitative questions such as:
  - Are other country or place names enriched among the smoothest transitions?
  - Are there demonyms, languages, nationalities, cities, or geographic terms?
  - Are there semantically related words that are not geographic entities?
  - Are many tokens merely orthographically / tokenization-similar to ` France`?
  - Are punctuation, whitespace, fragments, capitalization variants, or other token-form effects common?
  - Are the low-width tokens heterogeneous with no obvious semantic relationship?
- [ ] Do not force tokens into categories when the pattern is unclear.

The main deliverable for this section is a human-readable description of what the smoothest `France â token` transitions actually look like.

## Section 3 â Representative transition curves

Choose a small number of examples after seeing the sweep.

- [ ] Plot `d(t)` for several of the lowest-width / clearest non-plateau examples.
- [ ] Include a few ordinary `w > 0.3` examples as controls.
- [ ] Prefer examples that illustrate genuinely different observations rather than many near-duplicates.
- [ ] Annotate each plot with the partner token and transition width.

This section is for visual confirmation only; do not introduce another plateau metric.

## Section 4 â Qualitative conclusion

Summarize what the vocabulary sweep suggests about which tokens do **not** form a plateau with ` France` in this context.

Distinguish between possibilities such as:

- **semantic structure:** smooth transitions disproportionately involve tokens related to `France` in meaning;
- **token-form structure:** smooth transitions are better explained by spelling, subword form, whitespace, capitalization, or related tokenizer effects;
- **multiple recognizable groups:** several distinct kinds of tokens produce smooth transitions;
- **heterogeneous:** low-width tokens do not show an obvious common pattern;
- **mixed / unclear:** some pattern is visible but does not provide a simple explanation.

Keep the conclusion descriptive. This experiment does not by itself establish why the geometry is smooth, or whether semantic similarity causally determines plateau formation.

## Figures / artifacts

Required:

- `plots/france_vocab_width_histogram.png`
- machine-readable full-vocabulary results
- sorted list of all tokens with `w <= 0.3`
- readable table of the lowest-width tokens
- representative `d(t)` curves for selected low-width and plateau-like examples

Optional only if immediately useful:

- a simple manually assigned category column for a small number of inspected low-width tokens
- separate small tables for obvious groups discovered during inspection

## Out of scope (do NOT)

- Do not run all vocabulary-token Ã vocabulary-token pairs.
- Do not change the anchor away from ` France` in this direction.
- Do not add additional contexts in the first pass; this direction intentionally fixes `The capital of`.
- Do not change models; use GPT-2 Large only.
- Do not add embedding-similarity correlations, Spearman/Kendall tests, clustering, classifiers, nearest-neighbor analyses, or automated semantic labeling in the first pass.
- Do not optimize the `0.3` threshold after observing the results.
- Do not interpret a few semantic-looking examples as evidence that semantic similarity generally causes smooth transitions.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> â <stage, % done, blocker if any>`

## Current status

Stage 0 done (iteration 1, 2026-09-29): dir27 code imported unchanged via `experiments/sweep.py`; control reproduces dir27 shard 0 to 7e-7. Full sweep running (`results/sweep.log`). Found that this PLAN's threshold labels are inverted relative to dir27 (dir27: small W = abrupt jump = plateau-like; W > 0.3 = smoother). Handling: report both sides of 0.3, literal `w <= 0.3` list ascending plus the `w > 0.3` list, and let d(t) curves show which side is smooth.

## Next step

When sweep finishes: `python experiments/analyze.py`, inspect both ends, pick tokens for `experiments/curves.py`, write RESULTS.md and REPORT.md, run `../check_render.py`.
