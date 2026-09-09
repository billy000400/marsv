# What predicts how sharply a character GPT switches its output?

## Summary

Freeze the prompt and slide an internal activation vector from the state a model holds for input X to
the state for input Y: the output sits still, then flips. How abruptly it flips says how much a nudge to
an activation changes what the model does.

On one small character-level GPT trained on Shakespeare we ask two questions. **First:** with one
endpoint fixed to a letter, is the switch width organised by what *kind* of character sits at the other
endpoint (vowel, consonant, punctuation, whitespace)? **Second:** do endpoints followed by more different
characters switch more narrowly? Both answers are read from the raw pairwise measurements, before
averaging.

## Methods

**Data and model.** A 12-block, 12-head decoder-only GPT (width 240, context 128, 8.4M parameters)
trained for next-character prediction on Tiny Shakespeare (1,115,394 characters, 90% train split), at
step 30,000. Widths come from a stored sweep in `../dir13_plateau_on_grok_gpt`, revalidated in
`RESULTS.md` section 1.

**Interpolation and transition width.** Each measurement compares two characters, $c_{\mathrm{anchor}}$
and $c_{\mathrm{partner}}$. For each we run the prompt `"The house was "` plus that character and take
the residual stream at the last position after block 0. The two vectors are blended along a
norm-preserving spherical path in 50 steps indexed by $t$ ($t=0$ anchor, $t=1$ partner) and patched back
in; the logits $z(t)$ say how far the output has moved:

```math
d(t)=\frac{\|z(t)-z(0)\|_2}{\|z(t)-z(0)\|_2+\|z(t)-z(1)\|_2}.
```

$d=0$ means the output still matches the anchor, $d=1$ the partner. The **transition width** is the span
of $t$ over which $d$ climbs from 0.1 to 0.9, read off a monotone (isotonic) fit:

```math
w(c_{\mathrm{anchor}},c_{\mathrm{partner}})=t(d=0.9)-t(d=0.1).
```

Small $w$ means an abrupt switch: a proportional output would give $w=0.8$, whereas the median here is
0.320 (Figure 1). $w$ is symmetric.

![interpolation curves](plots/fig1_width_definition.png)

**Figure 1.** The transition width. x: interpolation position $t$ (anchor 0, partner 1); y: output
distance $d(t)$. Three pairs (solid, dashed, dash-dotted, labelled with their widths), the all-pair
median (thick gray), the proportional line $w=0.8$ (dotted).

**Which characters we use: 65, then 53, then 43.** The vocabulary has 65 characters. Rare characters have
unusually wide transitions, so we keep the earlier threshold of 1,000 training-split occurrences;
the 53 that pass we call *well-trained*, giving the 1,378 pairs for question 2. Question 1 needs a letter
endpoint: 43 of the 53 are letters, each in turn a fixed **anchor** against the other 52 well-trained
characters, its **partners**.

**Anchors and class medians, with a worked example.** Partners are sorted into six surface classes of the
character — fixed before analysis, not learned: lower-case vowels, lower-case consonants, upper-case
vowels, upper-case consonants, punctuation and digits, whitespace. A *class-level median* summarises
several separate measurements. With `t` as the anchor, its value for the lower-case-vowel
class is

```math
\mathrm{median}\lbrace w(\texttt{t},\texttt{a}),\, w(\texttt{t},\texttt{e}),\, w(\texttt{t},\texttt{i}),\, w(\texttt{t},\texttt{o}),\, w(\texttt{t},\texttt{u})\rbrace.
```

The model never interpolates from `t` towards a "lower-case-vowel representation". It performs five
separate character-to-character interpolations — `t`→`a`, `t`→`e`, `t`→`i`, `t`→`o`, `t`→`u` — whose
widths we summarise with a median, one number per class.

**Successor divergence.** Question 2 needs a character property defined by the corpus alone, so we use
what follows each character. Unsmoothed character-bigram counts on the training split give
$P_c$, the distribution of the character right after `c`. We compare two such distributions with the
base-2 Jensen–Shannon divergence, with $m$ their average and $D_{\mathrm{KL}}$ the Kullback–Leibler
divergence:

```math
J(c_{\mathrm{anchor}},c_{\mathrm{partner}})=\tfrac{1}{2}D_{\mathrm{KL}}(P_{c_{\mathrm{anchor}}}\Vert m)+\tfrac{1}{2}D_{\mathrm{KL}}(P_{c_{\mathrm{partner}}}\Vert m).
```

$J=0$ bits means the two characters are followed by the same mix, $J=1$ bit that their successors never
overlap. Estimated separately on each half of the training split, pair values agree at rank correlation
0.95 (`RESULTS.md` section 3).

## Results

### 1. Raw widths show class structure; the class median is only sometimes representative

Figure 2 shows every measurement behind question 1: one cell per pair, no averaging, columns grouped by
class.

![pairwise width heatmap](plots/fig2_width_heatmap.png)

**Figure 2.** Left: raw transition width for every measured pair. x: the 53 well-trained partners,
grouped into the six classes (black separators, names above), alphabetical within class; y: the 43
letter anchors, grouped likewise; colour: $w$, dark blue = abrupt, yellow = gradual, white = self-pair.
Right, summarised: x: an anchor's median width in a class; y: the six classes; one point per anchor,
boxes give quartiles.

Lower-vowel and whitespace columns are dark for almost every anchor, the upper-consonant block
brightest, so the right panel's ordering is visible in the raw cells: lower-case vowels narrowest (0.270
averaged over anchors), upper-case consonants widest (0.356), a gap of 0.087 (Table 1). The tendency is
broad — 38 of 43 anchors put lower-case vowels below upper-case consonants.

The heatmap also shows where that median stops being fair. Within one anchor, raw widths span an
interquartile range of only 0.021 (lower vowels) and 0.020 (whitespace), but 0.051 and 0.063 in the two
consonant blocks — as large as the 0.087 gap between the extreme class averages. **For the consonant
classes the median hides substantial pair-to-pair variation and should not be read as a typical pair.**
The variation runs down columns, striping those blocks: `s` averages 0.260 over the anchors, `v` 0.386.

| partner class | members | mean class median $w$ | mean within-anchor IQR of raw $w$ |
|---|---|---|---|
| lower vowel | 5 | 0.270 | 0.021 |
| whitespace | 2 | 0.286 | 0.020 |
| upper vowel | 5 | 0.316 | 0.043 |
| lower consonant | 17 | 0.320 | 0.051 |
| punctuation & digits | 8 | 0.331 | 0.050 |
| upper consonant | 16 | 0.356 | 0.063 |

**Table 1.** Column 3 averages each anchor's class median over the 43 anchors, narrowest first; column 4
averages the interquartile range of the *raw* widths in that block, so small values mean the median
represents its cells. Agreement statistics and a frequency control: `RESULTS.md` section 2.

### 2. Successor divergence: flat except for a punctuation cluster at the bottom

Figure 3 plots all 1,378 pairs in fixed-width divergence bins of 0.1, so the few near-zero-divergence
pairs keep a bin of their own.

![width against successor divergence](plots/fig3_jsd_vs_width.png)

**Figure 3.** x: successor divergence $J$ between endpoints (bits), fixed bins of 0.1; y: transition
width $w$. Faint dots: pairs with at least one non-punctuation endpoint (n = 1,350); diamonds: both
endpoints punctuation (n = 28). Solid line, squares: bin mean width, all pairs; dashed line, triangles:
the same with punctuation–punctuation pairs removed, leaving 2 pairs in the lowest bin (annotated).
Rotated text: pairs per bin, with the punctuation–punctuation count ("p-p").

From 0.1 upward the bin means barely move: each lies between 0.317 and 0.360, a range of 0.04 against a
pair-to-pair spread near 0.2. The lowest bin is the exception — below $J=0.1$ sit 12 pairs of mean width
0.581, nearly double the rest. (Recomputed here, bins 1–3 give n = 12 / 31 / 78 and means
0.581 / 0.360 / 0.343, matching the review.)

Ten of those 12 pairs are punctuation-to-punctuation, drawn from `! , . : ; ?` — expected, since
punctuation marks are followed by similar things and so sit at low $J$ by construction. They are wide
*wherever* they fall on the x-axis: mean width 0.531 over all 28, against 0.327 for the other 1,350. The
bin's two other pairs, `g`–`t` and `h`–`m`, have ordinary widths (0.302, 0.379), and removing the
punctuation–punctuation pairs flattens the bin to 0.340 — an average over two pairs, so weak by itself.
The wide lowest bin may therefore be a punctuation cluster rather than a general effect of low
divergence.

Holding the letter anchor fixed removes between-anchor differences. Done that way, 36 of the 43 anchors
show the expected direction — more different successors, slightly narrower transitions — but the trend
is weak and explains little of an anchor's variation. Pooled over all pairs it disappears; a weak,
non-significant trend of opposite sign between anchors may offset it. Per-anchor values and rank
statistics are in `RESULTS.md` section 4.2 and its Figure 3.

## Conclusion

**Character classes (question 1).** Figure 2 shows a genuine class tendency, consistent across anchors,
but the class median summarises fairly only the small, homogeneous classes: in the consonant blocks
pair-to-pair variation within an anchor matches the gap between the extreme class averages (Table 1), so
the partner character carries as much structure as its class.

**Successor divergence (question 2).** Across most of the divergence range, transition width changes
little. Pairs with divergence below 0.1 have substantially wider transitions, but this small group of 12
is dominated by punctuation–punctuation pairs, which are wide at every divergence level. We therefore do
not find a general monotonic relationship between successor divergence and transition width. Within a
fixed letter anchor, larger successor divergence is weakly associated with narrower transitions, but the
trend explains little of the variation.

Nothing here is causal — we intervened on neither successor statistics nor class membership — and this is
one model at one checkpoint, prompt, block and position.
