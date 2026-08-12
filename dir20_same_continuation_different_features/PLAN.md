# PLAN — Intuitive Same-Continuation Examples: When Do Conceptually Different Endpoints Produce Plateaus?

> Working folder: `dir20_same_continuation_different_features`
>
> This plan **supersedes the current F-based matched-pair analysis as the next direction**.
> The goal is no longer to explain transition width with a newly invented feature score.
> The goal is to find **simple, human-readable examples** where two conceptually different endpoint tokens lead to essentially the same continuation, then directly inspect whether interpolation between them produces a plateau.

## Question

Consider two prompts that are identical except for the final token:

```text
P + A
P + B
```

where `A` and `B` represent the same underlying referent, quantity, entity, or information in noticeably different ways.

Examples:

```text
Mary and John went to the store. John gave a book to Mary
Mary and John went to the store. John gave a book to her
```

```text
Two plus two is four
Two plus two is 4
```

The central question is:

> **When two conceptually different endpoint tokens lead the model to make essentially the same next prediction, which endpoint pairs produce a plateau during activation interpolation, and which do not?**

A second question is:

> **Does the same example behave similarly in GPT-2 Large and Pythia, or can one model show a plateau while the other is smooth?**

The output of this direction should primarily be **concrete examples and raw interpolation curves**, not a new scalar explanation.

---

## Success criterion

`REPORT.md` is complete when it contains:

1. A bank of at least **12 intuitive endpoint pairs** that pass the continuation checks below, preferably spanning at least 4 human-interpretable relation types.
2. The exact natural-language prompt for every pair.
3. For every accepted pair, the actual endpoint next-token predictions and short greedy continuations in:

   * GPT-2 Large;
   * Pythia.
4. For every accepted pair, a **normalized-distance-vs-interpolation-position plot** for both models.
5. A plain-English grouping of examples into:

   * clear plateau-like behavior in both models;
   * smooth / approximately proportional behavior in both models;
   * GPT-2 Large / Pythia disagreement;
   * ambiguous or pathological curves.
6. Several particularly clean positive and negative examples discussed individually.

A null result is valid. For example, if almost every continuation-matched example is smooth, report that clearly.

Do **not** invent another feature score to rescue the direction.

---

## Fallback

If fewer than 12 examples survive the continuation requirements, use all surviving examples if there are at least 6.

If fewer than 6 survive, expand the set of **human-designed semantic templates**, not the statistical machinery.

Do not switch to random token-pair mining simply to increase sample size.

---

## Setup

### Models

Run every useful example independently on:

* pretrained `gpt2-large`;
* the same pretrained Pythia checkpoint already used by the earlier exploratory code if one exists; otherwise use `EleutherAI/pythia-410m`.

Use the final pretrained checkpoints in `eval()` mode.

The purpose is **cross-model replication of individual examples**, not a model-size scaling study.

### Tokenization requirement

For every candidate pair:

* the shared prefix must be identical within each model;
* `A` must be exactly one final token;
* `B` must be exactly one final token;
* both requirements must hold for **both GPT-2 Large and Pythia**.

Reject examples that cannot be represented as a single differing final-token position in both models.

### Interpolation

Use the same intervention geometry as the existing Matthew-style experiment:

* interpolation site: final-token `resid_post` after block 0;
* 101 equally spaced interpolation positions
  `alpha = 0.00, 0.01, ..., 1.00`;
* rescaled SLERP: interpolate direction with SLERP and interpolate activation norm linearly;
* run all remaining transformer blocks normally.

Verify that `alpha=0` and `alpha=1` reproduce the original endpoint outputs.

---

## What counts as an intuitive candidate?

Do **not** start from arbitrary token pairs and then try to explain them after the fact.

Start from a human-readable relationship between `A` and `B`.

Prioritize categories such as:

### 1. Same referent, different linguistic form

Example:

```text
Mary ↔ her
John ↔ him
```

The surrounding sentence must make the coreference unambiguous.

### 2. Same quantity, different representation

Examples:

```text
four ↔ 4
ten ↔ 10
```

### 3. Same entity or fact, different notation

Examples may involve:

```text
name ↔ abbreviation
name ↔ symbol
symbol ↔ numeric identifier
```

but only keep them when the resulting English prompt is natural and the relationship is obvious without specialist explanation.

### 4. Same lexical content, different surface form

Examples such as:

```text
four ↔ Four
```

Treat these as useful **controls**, not the main conceptual examples.

### 5. Other simple equivalences

The agent may propose additional categories, but every candidate must be explainable in **one ordinary English sentence**.

Do not introduce terms such as "feature identity", "latent semantics", or newly named categories unless they are genuinely necessary.

---

## Stage S1 — Reproduce the known interpolation behavior

Before searching for new examples, reproduce Matthew's basic sanity check in GPT-2 Large:

```text
The house was big
The house was in
```

and

```text
The house was big
The house was large
```

Save the normalized-distance curves with the linear reference.

This stage only verifies that the interpolation implementation is behaving as expected.

Do not use these examples as evidence for the new question.

---

## Stage S2 — Build intuitive candidates without looking at interpolation curves

Start with these four seed pairs:

```text
Mary and John went to the store. John gave a book to Mary
Mary and John went to the store. John gave a book to her
```

```text
Two plus two is four
Two plus two is 4
```

```text
The answer is four
The answer is Four
```

```text
Which chemical element does this clue identify? Au
Which chemical element does this clue identify? 79
```

Then construct additional simple examples from the categories above.

Aim for approximately **30–50 candidates before continuation filtering**.

For every candidate save:

* shared prefix `P`;
* endpoint token `A`;
* endpoint token `B`;
* one-sentence plain-English explanation of the relationship;
* token IDs in both models.

Do **not** compute interpolation curves during candidate construction.

---

## Stage S3 — Check that the continuation is actually the same

This stage happens **before interpolation**.

For `P+A` and `P+B`, in each model separately, save:

1. the top-5 next-token predictions and probabilities;
2. the greedy next token;
3. a 5-token greedy continuation;
4. the already-established inference-time successor JSD.

### Primary continuation requirement

A useful example must have the **same top-1 next-token prediction after A and B in both GPT-2 Large and Pythia**.

Prefer stronger examples where several subsequent greedy tokens are also identical.

### Avoid trivial punctuation matches

If the shared top-1 prediction is only punctuation or whitespace, do not treat that alone as evidence of a shared continuation.

In that case inspect the following greedy tokens and require that the continuation also agrees on meaningful content beyond the punctuation.

For example, this is weak:

```text
A → "."
B → "."
```

This is much stronger:

```text
A → ". The next ..."
B → ". The next ..."
```

### JSD

Successor JSD is a **sanity check only**.

Use it to detect cases where the two full next-token distributions are clearly different even though the argmax happens to match.

Do not:

* optimize examples for JSD;
* match pairs by JSD;
* regress plateau strength against JSD;
* turn JSD into the explanation of the result.

As a simple guard, flag examples with successor JSD above `0.15` in either model as weak continuation matches and normally exclude them from the main examples.

### Lock before interpolation

Save all surviving examples to:

```text
results/intuitive_pairs.json
```

before looking at any interpolation curves.

This prevents selecting examples only because their plateau looked interesting.

---

## Stage S4 — Plot the interpolation path for every surviving example

For every locked example, run the activation interpolation independently in GPT-2 Large and Pythia.

Let:

* `x_A` be the final-token output logit vector at endpoint A;
* `x_B` be the final-token output logit vector at endpoint B;
* `x_alpha` be the output produced by the interpolated activation.

Plot the existing normalized distance:

```math
d(\alpha)
=
\frac{
\|x_\alpha-x_A\|_2
}{
\|x_\alpha-x_A\|_2+\|x_\alpha-x_B\|_2
}.
```

A proportional response follows approximately:

```math
d(\alpha)=\alpha.
```

A plateau-like response stays near one endpoint for a substantial part of the path and then changes rapidly.

### Required figure for every useful example

Create one figure with two side-by-side panels:

```text
GPT-2 Large                       Pythia
d(alpha)                          d(alpha)
1 |                               1 |
  |        _____                    |       /
  |       /                         |      /
  |______/                          |_____/
0 +----------- alpha             0 +----------- alpha
  0           1                    0           1
```

Each panel must show:

* `d(alpha)`;
* the `d=alpha` linear reference;
* axes fixed to `[0,1]`;
* exact endpoint tokens;
* successor JSD;
* greedy continuation after A;
* greedy continuation after B.

Save as:

```text
plots/examples/<example_name>.png
```

The **plot is the primary evidence**.

Do not replace the curve with a width number.

If the existing code already computes `w_TV`, it may be stored as a secondary diagnostic, but:

* do not invent a new width metric;
* do not rank examples using `w_TV`;
* do not make the report depend on a threshold in `w_TV`;
* do not use a scalar metric instead of showing the raw curve.

Also flag examples where the two endpoint output vectors are so close that normalized distance becomes numerically unstable or visually meaningless. Keep those separate as "uninformative", rather than interpreting noise as a plateau.

---

## Stage S5 — Look for concrete positive, negative, and disagreement cases

After all locked examples have been swept, organize them by what the raw curves show.

### A. Plateau in both models

Find examples where GPT-2 Large and Pythia both show a clear sharp transition.

These are the strongest examples that the phenomenon is not model-specific.

### B. Smooth in both models

Find examples where both models move approximately proportionally despite the endpoints being conceptually different and having the same continuation.

These are equally important.

They show that:

> conceptual difference + same continuation is **not sufficient** to produce a plateau.

### C. Cross-model disagreement

Look specifically for examples like:

```text
GPT-2 Large: strong plateau
Pythia:      smooth
```

or the reverse.

These are particularly useful because the **textual relationship is held fixed**, but the learned model changes.

Do not explain the disagreement with a new score in this direction.

Simply document it cleanly as a target for a later mechanistic experiment.

### D. Ambiguous / pathological

Keep non-monotonic, noisy, or near-identical-output examples visible, but do not force them into plateau/no-plateau categories.

---

## Required summary figure

Create:

```text
plots/example_summary.png
```

showing a small set of the clearest examples, preferably:

* 2 plateau-in-both examples;
* 2 smooth-in-both examples;
* 2 cross-model disagreement examples, if they exist.

Every mini-panel must show the actual normalized-distance curve.

The reader should be able to understand the main result from this figure without reading a statistical methods section.

---

## REPORT.md structure

Keep the report short and example-driven.

### 1. Question

Explain in plain English:

> We hold the continuation approximately fixed and change how the final concept is represented. Does the internal path between those representations contain a plateau?

### 2. Experimental setup

Briefly explain:

* identical prefix;
* one-token endpoint replacement;
* same-next-prediction requirement;
* block-0 activation interpolation;
* normalized output distance;
* GPT-2 Large / Pythia cross-check.

### 3. The examples

For each important example show:

```text
Prompt A:
Prompt B:

Why A/B are conceptually related:

GPT-2 Large:
next prediction A:
next prediction B:
short continuation A:
short continuation B:
successor JSD:

Pythia:
next prediction A:
next prediction B:
short continuation A:
short continuation B:
successor JSD:
```

Then immediately show the normalized-distance plot.

### 4. What has a plateau?

Organize examples into:

* plateau in both;
* smooth in both;
* model disagreement;
* ambiguous.

Focus on concrete comparisons.

### 5. Takeaway

The report should answer:

> **Can two conceptually different representations with essentially the same continuation have either a plateau or no plateau?**

and, if the data permit:

> **Can the same textual example plateau in one model but not another?**

Do not claim to know *why* yet.

That becomes the next direction.

---

## Required outputs

```text
results/intuitive_candidates.json
results/intuitive_pairs.json
results/interpolation_curves.npz
plots/matthew_sanity.png
plots/example_summary.png
plots/examples/*.png
RESULTS.md
REPORT.md
```

`results/intuitive_pairs.json` must include the human-readable prompts and continuation outputs, not only token IDs.

---

## Out of scope — do not drift

Do **not**:

* define another feature-difference score;
* use the old MLP Jaccard `F` as the organizing variable;
* search for correlations between `F` and width;
* perform neuron linearization or causal neuron interventions;
* run regression, bootstrap, matching, or quantile analysis;
* mine thousands of arbitrary WikiText token pairs;
* introduce SAE scores, attention-head scores, local-complexity scores, or other proxies;
* claim that the experiment explains the mechanism producing the plateau;
* hide raw interpolation curves behind summary statistics.

The purpose of this direction is deliberately simpler:

> **Find understandable examples. Show the curves. Cross-check the same examples in two models. Establish what kinds of cases can have a plateau and what kinds can fail to have one.**

Mechanistic explanation comes later.

---

## On-track check

End each `JOURNAL.md` entry with:

```text
On track? <yes/no> — <stage, % done, blocker if any>
```

If the agent finds itself inventing a new scalar metric to summarize "conceptual difference", the answer is **no**.

---

## Current status

Fresh restart from the earlier four hand-written examples.

The previous F-based result may remain as historical work, but it is not the organizing principle of this direction.

---

## Next step

Implement S2 and S3 first:

1. verify tokenization of the four seed examples in GPT-2 Large and Pythia;
2. print their top-5 next-token predictions and 5-token greedy continuations;
3. construct additional human-readable candidates;
4. lock all candidates that genuinely have similar continuations;
5. **only then** run interpolation and generate the normalized-distance plots.
