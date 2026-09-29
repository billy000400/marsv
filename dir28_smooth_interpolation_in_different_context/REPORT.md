# Does the prefix change which tokens GPT-2 Large blends smoothly with " big"?

## Summary

PENDING — written when all four sweeps are complete (Sections 2–4 are still running).

## Research question

When GPT-2 Large reads the word " big" at the end of a sentence, and we slowly replace the input
embedding of " big" with the embedding of some other token B, the model's final-layer output usually
does not move gradually. For most tokens it stays near the " big" output and then jumps to the B output
over a short stretch of the path. A few tokens instead give a slow, ramp-like change. Direction 27
measured this for one sentence, "The house was big".

This report asks: **does the set of tokens that give a slow (wide) transition from " big" depend on the
words that come before " big"?** We repeat the Direction 27 vocabulary sweep under four prefixes, list
every token whose transition width is above 0.3 in each, and compare the four lists. The study is
descriptive. It asks whether the pattern changes with context, not why.

## Methods

**Data & Model.** We use pretrained GPT-2 Large (774M parameters, 36 transformer blocks) in float32,
with the standard GPT-2 tokenizer. We use four prompts, each ending in the single token " big":

| Section | Prompt | Token IDs |
|---|---|---|
| 1 | My house is big | 3666, 2156, 318, 1263 |
| 2 | The opposite of small is big | 464, 6697, 286, 1402, 318, 1263 |
| 3 | He dreamed big | 1544, 27947, 1263 |
| 4 | The elephant was big | 464, 20950, 373, 1263 |

Endpoint A is the prompt as written. Endpoint B replaces the last token ID (" big", ID 1263) with another
vocabulary ID and keeps the prefix fixed. The text is never decoded and re-tokenized, so A and B differ
only in the last token. In every section we sweep all 50,256 other vocabulary tokens; Direction 27
excluded none, and neither do we. All code for the model run and the width measurement is imported
unchanged from Direction 27; only the prompt string differs (`experiments/sweep.py`). Re-running the
Direction 27 prompt through this code reproduced Direction 27's stored widths for its first 256 tokens
to within 7×10⁻⁷.

**Interpolation.** We interpolate at layer 0, the input embedding of the last position. Both endpoints
share the same position embedding, so this is a straight-line mix of the two token embeddings, called
h_A (" big") and h_B. We feed the model this mix at 101 evenly spaced values of t:

```math
h(t) = (1-t)\,h_A + t\,h_B, \qquad t \in \lbrace 0, 0.01, \dots, 1 \rbrace
```

**Readout and progress curve.** We read the last-position output of block 35 (the final transformer
block, before the last layer norm), written h_35(t). To put every token on the same scale we measure how
far that output has moved at step t, as a fraction of the full distance between the two endpoints. We
call this the progress curve y(t):

```math
y(t) = \frac{\lVert h_{35}(t) - h_{35}(0) \rVert_2}{\lVert h_{35}(1) - h_{35}(0) \rVert_2}
```

By construction y = 0 at the " big" endpoint and y = 1 at the B endpoint.

**Transition width W.** To summarize how gradual the change is in one number, we measure how much of the
path the model needs to go from 10% to 90% of the way to B:

```math
W = t_{0.9} - t_{0.1}
```

Here t_0.1 and t_0.9 are the first values of t at which y(t) reaches 0.1 and 0.9, found by straight-line
interpolation between neighbouring grid points. A small W (for example 0.05) means an abrupt jump. W
near 0.8 would mean a straight ramp. The inspection threshold W > 0.3 is fixed by the plan and was not
tuned. We call tokens above it "wide-transition tokens".

**Input distance D** is the Euclidean distance between the two embeddings, D = ‖h_A − h_B‖. We use it
only to recognize one group of tokens: rarely trained "glitch" tokens (control bytes, broken byte
fragments, and scraped identifiers such as ` externalToEVA`) all sit at nearly the same distance,
D ≈ 5.16, because their embeddings are close to one shared point.

**Flag.** Following Direction 27, a curve is flagged when y crosses 0.1 or 0.9 more than once, or falls by
more than 0.05 between neighbouring steps. Flagged tokens stay in the lists and are marked `[flagged]`.

## Section 1 — "My house is big"

In this context 692 of 50,256 tokens (1.4%) have W > 0.3. The median width is 0.106, and the widest token
reaches 0.66. So most substitutions are sharp jumps, and the wide-transition tokens form a thin tail.

Reading the complete list (Appendix A) shows three visible groups:

- **Size and magnitude words dominate the top.** Of the 40 widest tokens, about 25 are words for large
  size or amount: " HUGE", " tremendous", " whopping", " huge", " monumental", " giant", " substantial",
  " colossal", " vast", " sizable", " hefty", " Massive", " large", " gigantic". Word pieces of these
  words ("UGE", "ossal", "stantial", "hemoth") also appear. Further down, words for small size
  (" small", " tiny", " little", " microscopic", " miniature") and other gradable adjectives, many
  describing a place or object (" spacious", " cramped", " crowded", " luxurious", " scenic"), fill most
  of the list.
- **Code identifiers and scraped strings are mixed in throughout** (" guiName", "isSpecialOrderable",
  "PsyNetMessage", " externalToEVAOnly", "ItemTracker").
- **A block of 64 glitch tokens sits just above the threshold** at W ≈ 0.32 and D ≈ 5.16: control
  characters ("\x00" to "\x1f"), invalid byte fragments, and identifiers such as " TheNitrome" and
  "quickShip". They have nearly identical widths because their embeddings nearly coincide.

PENDING: histogram (Figure 1) is drawn after all four sweeps so the four panels share bins and axes.

## Section 2 — "The opposite of small is big"

PENDING — sweep running.

## Section 3 — "He dreamed big"

PENDING — sweep not started.

## Section 4 — "The elephant was big"

PENDING — sweep not started.

## Cross-context comparison

PENDING.

## Conclusion

PENDING.
