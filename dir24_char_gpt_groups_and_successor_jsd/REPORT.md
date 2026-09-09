# What predicts how sharply a character GPT switches its output?

## Summary

Slide a trained model's internal state smoothly from one input to another and its output need not follow
smoothly: it can sit still and then flip. How abruptly it flips says when a small change to an activation
will change what the model does.

We measure this on one small character-level GPT trained on Shakespeare, at a single prompt and a single
interpolation point, and ask two questions. First: with one endpoint fixed to a letter, are the switch
widths grouped by the kind of character at the other endpoint? Yes, consistently across all 43 eligible
letters, though the effect is modest and partly tracks character frequency. Second: do endpoints followed
by more different characters in the training text give narrower switches? Mixed — nothing when pairs are
pooled, a weak but reliable trend in that direction once the letter endpoint is held fixed. All of it is
one model, one checkpoint, and correlational.

## Methods

**Data and model.** A 12-block, 12-head decoder-only GPT (width 240, context 128, dropout 0.2, about
8.4M parameters) trained for next-character prediction on Tiny Shakespeare (1,115,394 characters, first
90% used for training, 65-character vocabulary), at step 30,000. Widths come from a stored sweep in
`../dir13_plateau_on_grok_gpt` (commit `01d2501`), validated in `RESULTS.md`.

**Interpolation and transition width.** Measurements start from the prompt `"The house was "` (with its
trailing space) plus one more character. For characters `a` and `b` we run both 15-character inputs and
take the residual stream at the final position after block 0. Blending the two vectors along a
norm-preserving spherical path in 50 steps indexed by $t$ ($t=0$ gives the `a` state, $t=1$ the `b`
state), we patch the blend into the final position only and read the final-token logits $z(t)$, then
measure how far the output has moved:

```math
d(t)=\frac{\lVert z(t)-z_a\rVert_2}{\lVert z(t)-z_a\rVert_2+\lVert z(t)-z_b\rVert_2}.
```

$d=0$ means the output still matches endpoint `a`, $d=1$ that it matches `b`. The **transition width** is
the span of $t$ over which $d$ climbs from 0.1 to 0.9, read off a monotone (isotonic) fit of the curve:

```math
w_{10\to90}=t(d=0.9)-t(d=0.1).
```

Small $w$ means an abrupt switch; an output tracking the state proportionally would trace a straight line
and give $w=0.8$ (Figure 1).

**Well-trained characters and classes.** Rare characters have unusually wide transitions, so we keep the
earlier direction's frozen threshold of 1,000 training-split occurrences: 53 characters and their 1,378
pairs form the primary sample, the other 12 appearing only in a sensitivity check. Of the 53,
43 are letters, each used in turn as a fixed **anchor** against its 52 partners. Partners fall into six
classes fixed before analysis: lower-case vowels, lower-case consonants, upper-case vowels, upper-case
consonants, punctuation and digits, and whitespace. These are surface character classes, not learned
features.

**Successor divergence.** Question 2 needs a property of a character defined by the corpus, not the
model, so we use what follows it: counting character bigrams in the training split without smoothing
gives $P(y \mid c)$, the distribution of the character immediately after `c`. We compare a pair's
distributions with the base-2 Jensen–Shannon divergence, $m$ being their average:

```math
J(a,b)=\tfrac{1}{2}D_{\mathrm{KL}}\big(P(\cdot \mid a)\,\Vert\,m\big)
      +\tfrac{1}{2}D_{\mathrm{KL}}\big(P(\cdot \mid b)\,\Vert\,m\big).
```

$J=0$ bits means the two characters are followed by the same mix of characters, $J=1$ bit that their
successors never overlap. This is a divergence *between two endpoints*, not a property of one token.
Estimated on each half of the training split, the pair values agree at Spearman $\rho = 0.95$, well above
the 0.007-bit same-character noise floor.

![Three interpolation curves with their transition widths](plots/fig1_width_definition.png)

**Figure 1.** How the transition width is defined. x: interpolation position $t$ from endpoint `a` to
`b`; y: relative output distance $d(t)$. Curves: three single pairs (solid, dashed, dash-dotted; labelled
with their widths), the median over the 1,378 well-trained pairs (thick gray, median width 0.320), and
the straight-line reference $w=0.8$ (dotted).

## Results

### 1. Letter-to-character widths are grouped by character class

**Verdict: yes, consistently across letters — but the effect is modest and partly tracks how common the
partner character is.** The grouping is not a quirk of one special character such as the comma;
essentially every letter shows the same class ordering, so it is a general property of the model's
character layout.

For each of the 43 anchors we take the median width over its partners in each class (Figure 2). Averaged
over anchors, lower-case vowels give the narrowest transitions (0.270) and upper-case consonants the
widest (0.356), the rest in between (Table 1). Agreement across anchors, measured by Kendall's
coefficient of concordance $W$ — 0 if anchors rank the classes independently, 1 if identically — is
$W = 0.42$ (Friedman $\chi^2 = 91.2$, $p = 3.8\times10^{-18}$), and 38 of 43 anchors put lower-case
vowels below upper-case consonants. The effect is small beside the spread between anchors, whose overall
medians span 0.258–0.404, and one anchor's profile can invert the average ordering (Figure 2, left), so
the claim needs all 43.

Class membership is also confounded with frequency: vowels are common, upper-case consonants rare. But
repeating the concordance test after removing each anchor's partner-frequency trend still leaves a clear
pattern ($W = 0.27$, $p = 4.3\times10^{-11}$), so frequency explains part of the grouping, not all.

![Class medians for four anchors and across all 43 anchors](plots/fig2_class_widths.png)

**Figure 2.** Transition width by partner class, seen from letter anchors. Both panels: x: partner class,
ordered by mean rank over anchors (narrowest left); y: median $w_{10\to90}$ over the anchor's partners in
that class. Left: the most frequent letter of each letter class (`e`, `t`, `I`, `T`; own line
style and marker each). Right: one point per anchor; boxes give median and quartiles.

| partner class | mean median $w$ | mean rank | mean median $J$ (bits) |
|---|---|---|---|
| lower vowel | 0.270 | 1.79 | 0.60 |
| whitespace | 0.286 | 2.37 | 0.69 |
| upper vowel | 0.316 | 3.56 | 0.73 |
| lower consonant | 0.320 | 3.84 | 0.49 |
| punctuation and digits | 0.331 | 4.47 | 0.83 |
| upper consonant | 0.356 | 4.98 | 0.61 |

**Table 1.** Each column averages a per-anchor median over the 43 letter anchors; rows are ordered by
width and rank 1 is narrowest. Class sizes are 5, 2, 5, 17, 8 and 16 well-trained members in row order. The last column gives the successor
divergence of Result 2; the two orderings disagree — lower consonants have the lowest divergence but
middling widths, punctuation the highest but wide transitions.

### 2. Successor divergence predicts width only within a fixed letter, and weakly

**Verdict: mixed. Pooled over well-trained pairs there is no relationship; with the letter anchor held
fixed, more different successor distributions go with slightly narrower transitions in 36 of 43 anchors.**
Pooling mixes between- and within-anchor variation, so the fixed-anchor answer is the more informative.
Because pairs share characters, each aggregate below is tested by permuting the 53 character labels,
which leaves both pairwise matrices intact and breaks only their alignment.

Pooled over the 1,378 pairs, Spearman $\rho = -0.06$ (permutation $p = 0.32$): no effect, and the ten
binned medians in Figure 3 (left) stay between 0.307 and 0.341 across the full divergence range.

Holding the anchor fixed changes the picture. Correlating $J$ with $w$ across each anchor's 52 partners
gives a median $\rho$ of $-0.20$ (interquartile range $-0.40$ to $-0.08$), negative for 36 of 43 anchors
and individually significant at 0.05 for 18. The sign is the anticipated direction: higher divergence,
narrower transition. Under permutation the median $\rho$ stays within $[-0.12, +0.11]$ 95% of
the time, so $-0.20$ is unlikely under the null ($p = 0.002$, 1,000 permutations). Partialling out the
partner's log frequency strengthens it slightly (median $\rho = -0.32$, negative for 39 of 43), so this
is not the frequency effect again.

The gap between the two answers is visible in the data: the between-anchor component runs the other way.
Anchors with more divergent partners on average also have somewhat wider transitions on average
($\rho = +0.25$, $p = 0.10$, n = 43) — weak, but enough to cancel a within-anchor slope of $-0.20$ once
pairs are pooled. Table 1 shows the same tension, so successor divergence does not explain the class
ordering of Result 1.

![Pooled scatter and per-anchor correlations](plots/fig3_jsd_vs_width.png)

**Figure 3.** Successor divergence against transition width. Left: x: $J(a,b)$ in bits; y:
$w_{10\to90}$; one dot per well-trained pair, with ten equal-count bin medians on the diamond-marked
line. Right: x: the Spearman $\rho$ between $J$ and $w$ within one anchor across its 52 partners; y:
anchor count. Gray band: 95% range of the median $\rho$ under permutation; dashed line: observed
median; dash-dotted line: pooled value.

## Conclusion

Character-class grouping of transition width is general across letters: the 43 anchors rank the six
partner classes consistently ($W = 0.42$, $p = 3.8\times10^{-18}$, Figure 2), lower-case vowels narrowest
and upper-case consonants widest. This is the stronger result, and it survives removing the
partner-frequency trend, though weakened.

Successor divergence is weaker: pooled over pairs there is nothing ($\rho = -0.06$, $p = 0.32$), and only
with the anchor held fixed does a consistent negative trend appear (median $\rho = -0.20$, 36 of 43
anchors, permutation $p = 0.002$, Figure 3). It explains little of the variation in width.

Nothing here is causal: we did not intervene on successor statistics or class membership, so neither can
be said to *make* transitions sharper, and a shared cause such as how often and where a character appears
fits equally well. Everything comes from one 8.4M-parameter model at one checkpoint, prompt,
interpolation point (block 0) and patched position, so we cannot say how it varies with model, context or
depth, nor when during training the pattern appears.
