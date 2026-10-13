# Human concern — dir31 addendum: promote Berlin while preserving Japan

## Question

Can an additive intervention at the country token promote the later
Berlin answer while preserving Japan-related information?

The previous donor replacement changed country identity. Do not reuse
its successful edit as if it were a validated Berlin-specific direction.

## Setup

- Reuse dir31's model, transcoders, hooks, country readout, and prompts.
- Read ../CLAUDE.md and ../BUDGET.md. Do not reinstall the ML environment.
- Preserve existing results. Save this experiment under results/additive/
  and plots/additive/, and add a clearly separated report section.
- Tune on "The capital city of Japan is". Reserve the original capital
  prompt and the remaining correct paraphrase for evaluation.

## 1. Define candidates and a strictly additive edit

Start with dir30's Berlin candidates outside its Germany candidate set,
at layers 1–26. Set exclusion is only a candidate filter, not proof of
semantic independence.

For each feature, obtain a reference amplitude from its median positive
activation on dir30's Berlin-answer discovery prompts at that layer.
Do not obtain amplitudes from the Germany token of a capital prompt.
Record sample support and extraction position.

At the Japan token, apply:

    y_new = y_native + beta * sum_i(reference_amplitude_i * decoder_i)

Test beta = 0, 0.5, 1, 2. Do not subtract current activations, remove Japan
features, or replace the native MLP. Recompute downstream states normally.

## 2. Screen for answer effects AND country preservation

Test features individually first. Rank by the actual increase in
logit(Berlin) - logit(Tokyo), measured after the fixed bridge " is".

Use these fixed pilot tolerances:
- Country readout at layer 26 changes by at most 10% of the unedited
  Japan–Germany readout gap for the matched capital template.
- On each valid mid-sentence Japanese language/currency control, the
  original correct top-1 is preserved and its answer probability falls
  by no more than 0.05 absolute.
- " is" remains top-1 at the capital-prefix country token, with its
  probability falling by no more than 0.05 absolute.

Exclude first-token controls already known to invalidate the readout.
Apply preservation checks during selection, not only after choosing
the strongest answer-changing edit.

Combine the top 3 and top 5 preservation-passing features within a layer,
where available. Recheck every condition: individual preservation does
not guarantee group preservation. No broader search in this pass.

## 3. Freeze and evaluate

Freeze the smallest qualifying edit using tuning results only.
Evaluate reserved prompts against baseline and three same-site,
feature-count/norm-matched random controls.

A useful pilot result requires:
- At least +1 logit toward Berlin on each reserved prompt.
- A larger answer effect than every matched random control.
- All preservation checks still pass.

Show probabilities, readout changes, and greedy completions.
Berlin need not become top-1. Do not retune on reserved results.

## 4. Conditional plateau test

Only after the pilot passes, apply the SAME fixed additive vector at
every point of dir31's original 41-point Japan–Germany path.

Check country readout against the unedited path, including both endpoints.
If country preservation fails along the path, label the plateau result
confounded.

Plot:
1. Hidden-state d(t), using EACH condition's OWN edited endpoints.
2. Adjacent-step L2 changes divided by that condition's mean step.
3. Country readout and Tokyo/Berlin probabilities.

Also retain fixed-baseline-reference curves, raw step sizes, and endpoint
separation. Near-collapsed endpoints make normalized curves unreliable.
Read native states downstream of the edit, not directly edited outputs.

## Completion / stop rule

Report the best answer effect and its country cost, including failures.
If no setting passes, conclude:
"No selective additive intervention found in this bounded candidate search."

Do not conclude that disentanglement is impossible, enlarge strengths,
or relabel country features as city features. Update JOURNAL/RESULTS/REPORT
and finish according to the existing agent workflow.
