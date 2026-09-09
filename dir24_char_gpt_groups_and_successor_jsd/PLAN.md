# PLAN - What predicts transition width in the Shakespeare character GPT?

> Working folder: `dir24_char_gpt_groups_and_successor_jsd`. The agent rewrites "Current status" and
> "Next step" and ticks stages each iteration. Disk (`PLAN.md`, `JOURNAL.md`, `RESULTS.md`,
> `REPORT.md`, `CHANGELOG.md`, `../BUDGET.md`, and `../CLAUDE.md`) is the only memory.

## Core question

The comma sweep in `dir13_plateau_on_grok_gpt` suggested that transition widths differ systematically
across character types. This direction asks two linked follow-up questions:

1. Does the same grouping appear when the fixed endpoint is a **letter**, rather than punctuation?
2. Across character pairs, do endpoints with more different immediate-successor distributions have
   narrower transitions?

These questions belong in one direction because successor divergence is a plausible predictor of the
group differences. This is a descriptive follow-up on one trained character GPT. It does **not** test
whether the model groks, when the pattern emerges during training, or what mechanism produces it.

## Success criterion (definition of "done")

- `REPORT.md` gives a direct, self-contained answer to both questions, including null or mixed answers.
- The letter result is established across all eligible letter anchors, not from one selected letter.
- The successor-JSD result reports both the pooled relationship and the relationship within fixed
  letter anchors, so repeated endpoints cannot silently create or hide the trend.
- Every main result has an interpretable figure. A reader can understand the model, interpolation,
  transition width, character groups, and successor JSD without opening `dir13` or another report.
- `REPORT.md` is 1,600 words or fewer, contains at most three figures and one compact table, and follows
  `Summary -> Methods -> Results -> Conclusion`.
- `RESULTS.md` and `REPORT.md` contain current-best results only. Null results count as complete. When
  the checks below pass, write an empty `STOP` file.

## Fallback (if time runs short)

Reuse the stored step-30,000 widths, reproduce the across-letter group plot, compute corpus successor
JSD for the well-trained pairs, and make one two-panel JSD figure: pooled scatter with binned medians on
the left and the distribution of per-letter correlations on the right. Write the short report and
`STOP`; do not start a new training run.

## Setup (fixed)

### Model and interpolation

- **Model:** the step-30,000 character-level Shakespeare GPT from
  `dir13_plateau_on_grok_gpt`: 12 transformer blocks, 12 attention heads, width 240, context length
  128, dropout 0.2, approximately 8.6M parameters, vocabulary of 65 characters.
- **Corpus:** the same tinyshakespeare file and 90/10 split used for training. Verify SHA-256
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` before counting characters.
- **Prompt:** `"The house was "`, including the trailing space.
- **Endpoints:** append character `a` or `b` to that shared prompt. Interpolate the final-position
  residual stream after block 0 with the existing norm-rescaled spherical interpolation, using 50
  evenly spaced positions. Patch the final position only and measure the final-token logits.
- **Primary input:** reuse the 2,080 pairwise widths in
  `../dir13_plateau_on_grok_gpt/results/allpairs_summary.json`. Do not retrain the model or rerun the
  interpolation sweep unless this artifact fails validation.
- Treat `dir13_plateau_on_grok_gpt` as read-only source material. All new code, plots, results, and
  prose go in this direction.

### Primary sample

The original follow-up defined a character as well trained when it appears at least 1,000 times in the
training split. Keep that frozen threshold. The primary analysis uses the 53 well-trained characters
and their 1,378 unordered pairs. Show pairs touching any of the 12 rarer characters only as a labelled
sensitivity check; never mix them into the primary estimate, because their successor distributions
are estimated from little data and their transitions are already known to be unusually wide.

Letter anchors are the 43 well-trained alphabetic characters. Partner groups are fixed before analysis:

- lower-case vowels;
- lower-case consonants;
- upper-case vowels;
- upper-case consonants;
- punctuation and digits;
- whitespace (space and newline).

Call these **character classes**, not semantic features.

### Metrics

The output-position curve is inherited unchanged from `dir13`. If `z(t)` is the final-logit vector at
interpolation position `t`, and `z_a`, `z_b` are the endpoint logit vectors, define

```math
d(t)=\frac{\lVert z(t)-z_a\rVert_2}
          {\lVert z(t)-z_a\rVert_2+\lVert z(t)-z_b\rVert_2}.
```

`d(t)` near 0 means the output still resembles endpoint `a`; `d(t)` near 1 means it resembles endpoint
`b`. Fit the same isotonic copy used in `dir13` and define transition width

```math
w_{10\to90}=t(d=0.9)-t(d=0.1).
```

Smaller `w` means a narrower, sharper transition. A straight line has `w = 0.8`.

Here **successor JSD** means the corpus statistic used in
`dir18_continuation_jsd_plateau`, adapted to characters. For endpoint character `c`, estimate the
distribution `P(y | c)` of the single character immediately following `c` in the training split. For
a pair `(a,b)`, let `m=(P(.|a)+P(.|b))/2` and compute base-2 Jensen-Shannon divergence:

```math
J(a,b)=\frac{1}{2}D_{\mathrm{KL}}\!\left(P(\cdot|a)\,\Vert\,m\right)
      +\frac{1}{2}D_{\mathrm{KL}}\!\left(P(\cdot|b)\,\Vert\,m\right).
```

`J = 0` means the two characters tend to be followed by the same characters; larger `J` means their
immediate-successor distributions differ more. Do not call this the JSD "of a token": it is the JSD
**between two successor distributions**. Do not substitute endpoint plausibility
`P(a | "The house was ")`, which `dir13` already measured and which answers a different question.

## Stages (checklist)

- [x] **S1 - Validate and isolate the source data.** Read `dir13`'s model specification, all-pairs
  script, stored summary, and relevant follow-up analysis. Recompute `w` summaries from the stored
  pair rows and verify the pair count, vocabulary order, context, checkpoint, hook point, endpoint
  checks, and the 1,000-occurrence split. Record source paths and the source git commit in
  `RESULTS.md`. Do not copy unrelated `dir13` mechanism results.

- [x] **S2 - Test grouping from letter anchors.** For every one of the 43 well-trained letters, group
  its 52 well-trained partners by the six fixed classes and summarize `w` within each class. Report
  the class medians across anchors, how consistent the ordering is across anchors, and the variation
  between letters. Use the existing rank-based agreement analysis if it reproduces; do not invent a
  new grouping score. Make one clear two-panel figure: the left panel shows class medians for four
  non-cherry-picked anchors (the most frequent eligible member of each of the four letter classes),
  and the right panel shows the distribution across all 43 anchors. The result must say whether the
  grouping is consistent, partial, or absent. One letter alone cannot establish the claim.

- [x] **S3 - Compute and validate successor JSD.** Count training-split character bigrams and compute
  `P(.|c)` without smoothing. Repeat the estimates on the first and second halves of the training
  split. For the 53 well-trained characters, report split-half rank agreement of pairwise JSD and the
  median same-character split-half JSD as a sampling-noise floor. If reliability is poor, report the
  JSD analysis as inconclusive rather than tuning smoothing or the frequency threshold after seeing
  the answer.

- [x] **S4 - Relate successor JSD to transition width.** On the 1,378 well-trained pairs, plot `w`
  against `J(a,b)` with ten equal-count binned medians and report Spearman's rank correlation. Because
  pairs reuse characters, do not use the ordinary 2,080-pair correlation p-value as if the pairs were
  independent. Use a character-label permutation that preserves both pairwise matrices for the pooled
  null. Then hold the anchor identity fixed: for each of the 43 letter anchors, correlate JSD with
  width across its 52 partners and report the median, interquartile range, and number of negative
  correlations. Put the pooled result and the per-letter distribution in one two-panel figure. Repeat
  the pooled calculation on pairs touching rare characters only as a visibly separate sensitivity
  check. Do not create a composite predictor or a large regression model.

- [x] **S5 - Write the report around the answers.** The report has exactly two Results subsections:
  `1. Letter-to-character widths are/are not grouped by character class` and
  `2. Successor divergence does/does not predict transition width`. Start each with its plain-language
  verdict, then give the figure and the minimum statistics needed to support it. Explain any difference
  between the pooled and fixed-letter answers rather than selecting the more favorable one. State
  explicitly that correlation does not show that successor JSD causes the transition or fully explains
  the character-class pattern.

- [x] **S6 - Humanize and reader-test.** Remove research-process history, feedback references,
  implementation archaeology, repeated caveats, and unexplained statistical jargon from `REPORT.md`.
  Ask a fresh technical reader to answer four questions using only the report: (i) what states were
  interpolated, (ii) what smaller `w` means, (iii) which two distributions JSD compares, and (iv) the
  answer to each research question. Revise until all four answers are correct. Run
  `python3 ../check_render.py REPORT.md RESULTS.md` and write `STOP` only after it passes.

## Required report shape

### Summary

In 120-180 words: motivation, exact model/setup, one-sentence answer to question 1, one-sentence answer
to question 2, and the one-model/one-context limitation. Do not summarize the history of `dir13`.

### Methods

In 350-500 words: data and model, interpolation and `w`, character classes, successor JSD and its
reliability check. Define each quantity before interpreting it.

### Results

Two subsections only, in the same order as the two core questions. Each subsection must remain
understandable if its figure is hidden. Figures support the prose; they do not carry the argument by
themselves.

### Conclusion

In 150-250 words: answer both questions, distinguish strong from weak evidence, and state what the
experiment does not establish. Do not add a new mechanism hypothesis unless a reported measurement
directly distinguishes it.

## Out of scope (do not)

- Do not append these experiments to `dir13/PLAN.md`, `dir13/REPORT.md`, or
  `dir13/REPORT_followup.md`.
- Do not retrain the GPT, rerun all 2,080 interpolation curves, or launch a checkpoint/layer/context
  sweep unless the stored widths fail validation.
- Do not revisit Grokking, local complexity, adversarial robustness, MLP/attention mechanisms,
  Jacobians, frozen blocks, neuron probes, or spline codes.
- Do not broaden to GPT-2, Pythia, BPE tokens, multi-token continuations, or other corpora.
- Do not use the model's probability of an endpoint after the shared prompt as "successor JSD".
- Do not claim that character classes are semantic features, that JSD defines a plateau, or that an
  association is causal.
- Do not add an appendix, more than three figures, or more than one table to rescue an unclear main
  narrative. Rewrite the narrative instead.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> - <stage, % done, blocker if any>`

## Current status

Done, and revised once for `human_feedback.txt` (all 48 checklist items; manifest state
`review_pending`). `experiments/analysis.py` (CPU, no model loaded) validated the stored dir13 sweep,
reproduced the letter-anchor class analysis over all 43 well-trained letters, computed corpus successor
JSD with a split-half reliability check, and related JSD to width in fixed-width bins, pooled, and within
fixed anchors. Answers now stated from the raw pairwise data first: (1) the 43 x 53 width heatmap shows a
real but coarse class effect — the class median is representative for the small classes (within-anchor
raw IQR 0.021, 0.020) but not for the two consonant classes (0.051, 0.063) against a 0.087 between-class
spread; (2) no general monotonic JSD-width relationship — fixed-width bin means run 0.317-0.360 from
JSD 0.1 to 1.0, and the wide 0.0-0.1 bin (n = 12, mean 0.581) is 10/12 punctuation-punctuation pairs,
which average 0.531 at every JSD; within a fixed anchor the trend is negative but weak. The rank
statistics (W = 0.424, pooled rho = -0.064, per-anchor median rho = -0.205) now live in `RESULTS.md`
only. `REPORT.md` (1,600 words by wc; 3 figures, 1 table) and `RESULTS.md` are current-best;
`python3 ../check_render.py REPORT.md RESULTS.md` passes.

## Next step

None from the plan — the success criterion is met. Awaiting the wrapper's independent content review of
the feedback task; no `STOP` while `human_feedback.txt` is unaddressed. If reopened: the untested
question is whether the class ordering and the within-anchor JSD trend hold at other blocks, contexts, or
checkpoints.

## References

- Original model and all-pairs source: `../dir13_plateau_on_grok_gpt/`
- Successor-JSD definition and reporting precedent: `../dir18_continuation_jsd_plateau/`
- Matthew Shinkle and StefanHex, *Activation Plateaus: Where and How They Emerge*:
  https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Humayun, Balestriero, and Baraniuk, *Deep Networks Always Grok and Here is Why*:
  https://arxiv.org/abs/2402.15555
