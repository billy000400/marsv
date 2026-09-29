# REPORT — Which tokens make a smooth transition away from ` France`?

## Summary

**Question.** We take the prompt "The capital of France" in GPT-2 Large and slowly blend the last input
token, ` France`, into a different token B. For each of the other 50,256 tokens B in the vocabulary we ask:
does the model's final-layer output switch from the ` France` output to the B output in one sudden
jump, or does it change gradually? Which tokens B give the gradual, smooth kind of switch, and do they
form recognizable groups?

This matters for the "plateau" experiments in this project. A plateau is a flat stretch of the output
followed by a sudden jump. Those experiments read plateaus as a sign of how the model separates inputs.
If the tokens that avoid a plateau form a meaningful group, that says something about which inputs the
model treats as close.

**Answer.** Almost every token (47,073 of 50,256, 93.7%) switches sharply, with width W ≤ 0.3 (Figure 1).
The sharpest switches come from short, capitalized word starts such as ` Ent`, ` Spe` and ` Tem`
(Table 1). The d(t) curves in Figure 2 show that these tokens produce the flat-then-jump plateau shape.
The smooth, gradual switches are on the other side of the threshold, the 3,183 tokens with W > 0.3.
Among the 100 smoothest, 53 are, by our reading, place names: French regions and cities, neighbouring
countries, and former French colonies (Table 2). Another 17 are rarely seen scraped or code-like strings.
So the smooth side looks structured, and its biggest visible group is geographic. We only observed this
pattern. We did not test whether similarity in meaning causes it.

## Methods

### Data & model

**Model.** Pretrained GPT-2 Large (774M parameters, 36 transformer blocks), evaluation mode, standard
GPT-2 tokenizer.

**Prompt and endpoints.** The prompt is "The capital of France", which tokenizes as
`[464, 3139, 286, 4881]`. Endpoint A is the last token ` France` (id 4881). Endpoint B keeps the same prefix
"The capital of" and replaces the last token ID with B. We never decode and retokenize a string. B runs
over all 50,256 other GPT-2 token IDs. We excluded none, following Direction 27.

**Where we interpolate.** Everything reuses Direction 27's code without changes (`experiments/sweep.py`
imports it). Interpolation happens at layer 0, the input embedding of the last position. The position
embedding is shared by A and B, so blending the two token embeddings blends the layer-0 hidden states:

```math
h(t) = (1-t)\,e_{\text{France}} + t\,e_B, \qquad t \in \lbrace 0, 0.01, \dots, 1 \rbrace
```

Here $e$ is a row of GPT-2's token-embedding matrix, and t takes 101 evenly spaced values. The PLAN names
the site "block-0 resid_post". Direction 27's code actually interpolates at the embedding layer, so we
followed the code. As a check, this code reproduced Direction 27's own first 256 results to within
7e-7.

**Readout.** We run each blended input through the model. We read the last-position output of block 35,
the final block, taken before the final layer norm. We call it $h_{35}(t)$. We also save the input distance
$D = \lVert e_B - e_{\text{France}} \rVert_2$. It is the straight-line (L2) distance between the two token
embeddings, and we use it only to pick out one group of far-away tokens in the Results.

### Metrics

**Progress curve d(t).** We need to compare tokens B whose endpoint outputs lie at very different distances
apart. So we measure how far the readout has moved at step t, as a fraction of the full distance between
the two endpoint readouts. Direction 27 calls this y(t), and the PLAN calls it d(t):

```math
d(t) = \frac{\lVert h_{35}(t) - h_{35}(0) \rVert_2}{\lVert h_{35}(1) - h_{35}(0) \rVert_2}
```

d(t) starts at 0 at ` France` and ends at 1 at B. A plateau curve stays near 0, jumps quickly to near 1,
and stays there. A smooth curve rises gradually across the whole range. Figure 2 plots d(t).

**Transition width W.** To summarize each curve with one number, we take the stretch of t over which d
goes from 10% to 90% of the way to B:

```math
W = t_{0.9} - t_{0.1}
```

$t_{0.1}$ and $t_{0.9}$ are the first values of t where d(t) crosses 0.1 and 0.9, found by linear
interpolation between grid points. A small W means a sudden jump, the plateau shape. A large W means a
gradual switch. W is the x-axis of Figure 1 and the sort key of Tables 1–2.

**Threshold.** We keep Direction 27's cutoff of W = 0.3 and did not tune it. The PLAN calls the W ≤ 0.3
side "non-plateau candidates" and the W > 0.3 side "plateau-like". Given the definition of W above, a
small W is the sudden jump. So the PLAN's two labels appear swapped relative to its own question, which
asks about smooth transitions. We did not settle this by choosing one side. We report the requested
W ≤ 0.3 list in full, report the W > 0.3 list as well, and use the d(t) curves (Figure 2) to show which
side is actually smooth.

**Flag.** Following Direction 27, a curve is flagged when it crosses 0.1 or 0.9 more than once, or when d
falls by more than 0.05 between neighbouring steps. Flagged curves stay in all counts. 537 tokens are
flagged: 532 with W ≤ 0.3 and 5 with W > 0.3.

**Reading the token lists.** The inspection is manual. We read the decoded tokens and, for the 100
widest, assigned each one to one of three labels by hand. "Place" means a place name or an unambiguous
piece of one, such as `agascar`. "Scraped" means a code-like, username-like, byte or foreign-script token
of the rarely seen kind described in Direction 27. "Other" covers everything else. We used no clustering
and no automated labeling (see the PLAN's out-of-scope list).

## Results

### The width distribution

To see how many tokens fall on each side of the threshold, we plot the width of every one of the 50,256
tokens:

![Histogram of final transition widths](plots/france_vocab_width_histogram.png)

**Figure 1.** Final transition width for ` France` → B, over all 50,256 tokens B. x: W at block 35;
y: number of tokens per bin (log scale; 100 equal bins). The dashed line marks W = 0.3.

The distribution has one broad peak and a long tail toward wide switches. The median W is 0.188, and the
middle half of tokens lies between 0.151 and 0.229. The range runs from 0.046 to 0.655. There are 47,073
tokens (93.7%) with W ≤ 0.3 and 3,183 tokens (6.3%) with W > 0.3 (`results/summary.json`).

### The W ≤ 0.3 list and its narrowest end

The complete W ≤ 0.3 list has 47,073 tokens. It is saved, sorted from smallest to largest W, in
`results/france_W_le_0.3.csv` (columns: rank, token ID, decoded token, W, input distance, flag). That is
too long to print here. The table below shows the 40 narrowest, i.e. the sharpest switches in the whole
vocabulary:

| rank | id | token | W | rank | id | token | W |
|---:|---:|---|---:|---:|---:|---|---:|
| 1 | 7232 | ` Ent` | 0.046 | 21 | 347 | ` B` | 0.058 |
| 2 | 2531 | ` Spe` | 0.049 | 22 | 5187 | ` Wil` | 0.059 |
| 3 | 5825 | ` Tem` | 0.052 | 23 | 1253 | ` incre` | 0.059 |
| 4 | 3125 | ` More` | 0.053 | 24 | 412 | ` E` | 0.059 |
| 5 | 4700 | ` Tre` | 0.054 | 25 | 2275 | ` Ab` | 0.059 |
| 6 | 2874 | ` stru` | 0.054 | 26 | 17175 | ` Ble` | 0.059 |
| 7 | 2448 | ` Per` | 0.054 | 27 | 12483 | ` dispos` | 0.060 |
| 8 | 4990 | ` Ret` | 0.055 | 28 | 5564 | ` Ro` | 0.060 |
| 9 | 4415 | ` Prof` | 0.055 | 29 | 4759 | ` Tra` | 0.061 |
| 10 | 1665 | ` Te` | 0.056 | 30 | 990 | ` produ` | 0.061 |
| 11 | 3169 | ` Ne` | 0.056 | 31 | 13794 | ` Und` | 0.061 |
| 12 | 2039 | ` En` | 0.056 | 32 | 2345 | ` Cont` | 0.061 |
| 13 | 1125 | ` che` | 0.057 | 33 | 943 | ` Ar` | 0.061 |
| 14 | 12830 | ` Fif` | 0.057 | 34 | 21783 | ` Ner` | 0.061 |
| 15 | 30558 | ` Repe` | 0.057 | 35 | 7164 | ` Gar` | 0.061 |
| 16 | 406 | ` L` | 0.057 | 36 | 7845 | ` displ` | 0.062 |
| 17 | 1114 | ` For` | 0.058 | 37 | 16319 | ` Esc` | 0.062 |
| 18 | 376 | ` F` | 0.058 | 38 | 1052 | ` An` | 0.062 |
| 19 | 5576 | ` uns` | 0.058 | 39 | 4377 | ` Any` | 0.062 |
| 20 | 3184 | ` Sim` | 0.058 | 40 | 2750 | ` By` | 0.062 |

**Table 1.** The 40 smallest final widths. Tokens are shown with their leading space.

The narrowest end has a clear token-form pattern. Almost all of these tokens are word starts: a leading
space followed by a short fragment that needs another token to finish a word (` Ent`, ` Spe`, ` stru`,
` incre`, ` produ`). Some are single capital letters (` L`, ` F`, ` B`, ` E`) and a few are short function
words (` More`, ` For`, ` By`, ` Any`). As a simple count, 82 of the 100 narrowest tokens are a space plus
a capitalized fragment. Only 21% of the whole vocabulary has that form, and 90 of the 100 are at most four
characters long. By our reading, none of the 100 narrowest is a place, a nationality or a language. We
see no link in meaning to France. Two orthographic neighbours of ` France` are also sharp. The same word
without the leading space, `France`, has W = 0.072. ` Fran` and ` Franc` have W = 0.158 and 0.153, close
to the vocabulary median.

### Which side is smooth: the d(t) curves

The width number alone does not show the shape of the curve. To check directly which side of 0.3 has the
flat-then-jump plateau shape, we plot d(t) for five tokens from each side. Panel (a) takes the two
narrowest word starts from Table 1 (` Ent`, ` More`), a very common word (` the`), the no-space spelling
`France`, and the related word ` French`, which sits just below the threshold. Panel (b) takes the widest
token overall (` Brittany`), three familiar places (` Spain`, ` Paris`, ` Germany`), and one rarely seen
scraped token (` SolidGoldMagikarp`).

![Representative d(t) curves on both sides of W = 0.3](plots/france_representative_curves.png)

**Figure 2.** Progress curves d(t) at block 35 for ` France` → B. x: interpolation position t (0 = ` France`,
1 = token B); y: progress d(t) (0 = ` France` readout, 1 = B readout). Dotted horizontal lines mark 0.1 and
0.9, the levels that define W. (a) Five tokens with W ≤ 0.3. (b) Five tokens with W > 0.3. Each token
has its own line style and marker; the legend gives each token and its W.

Panel (a) shows the plateau shape. For ` Ent`, ` More`, ` the` and `France`, d(t) stays below 0.2 until
t ≈ 0.35–0.47, then jumps to about 1 within 0.05–0.08 of t. ` French` (W = 0.266) is gentler but still
S-shaped. Panel (b) shows the gradual kind of switch. For ` Paris`, ` Spain` and ` Germany`, d(t) climbs
steadily from t ≈ 0.2 to t ≈ 0.7, and for ` Brittany` from t ≈ 0.23 to t ≈ 0.88. So under Direction 27's
definition, the smooth, non-plateau transitions are the W > 0.3 tokens. The W ≤ 0.3 tokens, and above
all the narrowest ones in Table 1, are the plateau-like ones.

### The W > 0.3 side: the smooth transitions

The full W > 0.3 list (3,183 tokens) is in `results/france_W_gt_0.3.csv`, sorted from widest to
narrowest. Because Figure 2 shows that these are the smooth switches, we read the 100 widest by hand. The
labels are in `results/france_widest100_manual.csv`. The table below shows the 30 widest:

| rank | token | W | label | rank | token | W | label |
|---:|---|---:|---|---:|---|---:|---|
| 1 | ` Brittany` | 0.655 | place | 16 | ` Adinida` | 0.584 | scraped |
| 2 | ` TheNitromeFan` | 0.649 | scraped | 17 | ` Algeria` | 0.584 | place |
| 3 | ` Marino` | 0.640 | place | 18 | ` Alger` | 0.583 | place |
| 4 | ` Rica` | 0.634 | place | 19 | ` pandemonium` | 0.582 | other |
| 5 | ` dstg` | 0.630 | scraped | 20 | ` Lyons` | 0.578 | place |
| 6 | ` Rico` | 0.621 | place | 21 | ` Palestin` | 0.576 | place |
| 7 | ` Lucia` | 0.618 | place | 22 | `quished` | 0.575 | other |
| 8 | ` Normandy` | 0.614 | place | 23 | `uyomi` | 0.567 | other |
| 9 | `SourceFile` | 0.612 | scraped | 24 | ` Leone` | 0.567 | place |
| 10 | ` Lyon` | 0.607 | place | 25 | `eden` | 0.565 | other |
| 11 | ` Guerrero` | 0.605 | other | 26 | ` Luxembourg` | 0.561 | place |
| 12 | ` Gaul` | 0.603 | place | 27 | ` Lydia` | 0.561 | other |
| 13 | ` Monaco` | 0.594 | place | 28 | ` Smartstocks` | 0.560 | scraped |
| 14 | (byte token 13945) | 0.590 | scraped | 29 | ` Aviv` | 0.560 | place |
| 15 | ` Cannes` | 0.586 | place | 30 | ` Belgium` | 0.560 | place |

**Table 2.** The 30 largest final widths, with our manual label.

Across the 100 widest tokens, 53 are places, 17 are scraped strings and 30 are other. The places fall
into a few recognizable kinds:

- **Parts of France and French cities:** ` Brittany`, ` Normandy`, ` Lyon`, ` Lyons`, ` Cannes`, ` Calais`,
  ` Paris`, and the historical name ` Gaul`.
- **Neighbouring countries and regions:** ` Monaco`, ` Luxembourg`, ` Belgium`, ` Switzerland`,
  ` Spain`, ` Portugal`, ` Catalonia`, ` Alps`.
- **North and West African countries with French colonial history:** ` Algeria`, ` Alger`, ` Tunisia`,
  ` Tunis`, ` Morocco`, ` Senegal`, ` Cameroon`.
- **Second pieces of multi-token place names:** ` Marino` (San Marino), ` Rica`, ` Rico`, ` Leone`,
  ` Aires`, ` Janeiro`, `agascar`, `raltar`.

Other large countries also switch more smoothly than most tokens. Their W values and percentiles among
all 50,256 tokens are ` Italy` 0.481 (99.7th), ` Germany` 0.423 (99.3rd), ` Britain` 0.355 (97.5th) and
` Japan` 0.324 (95.7th). The adjective ` French` is at 0.266 (87.9th).

The 17 scraped strings are rarely seen tokens such as ` SolidGoldMagikarp` and ` TheNitromeFan`.
Direction 27 found the same kind of token among its widest switches for ` big`, so this group appears
for two different anchors. A separate group of 61 byte and garbled tokens sits far from ` France` in the
input embedding (L2 distance 4.5–5.0; 90% of tokens lie between 2.3 and 2.9) and has W between 0.33 and 0.43. These are the
same kind of token that formed the tight far-away cluster in Direction 27. The "other" label covers people's names, word fragments and words like
` pandemonium`, with no shared theme that we could see.

## Conclusion

**The literal W ≤ 0.3 set is heterogeneous in meaning but structured in form.** It holds 94% of the
vocabulary (Figure 1). Its narrowest end is dominated by short, capitalized word starts, with no visible
link to France (Table 1). Figure 2(a) shows that these tokens give the sharpest plateau-shaped switches.

**The smooth transitions are the W > 0.3 tokens, and they look structured.** Figure 2(b) confirms that
these curves are gradual. Among the 100 smoothest, 53 are place names (Table 2). Many of them are tied to
France: its regions and cities, its neighbours, and its former colonies. A second, smaller group is the
rarely seen scraped strings that Direction 27 also found. The rest are mixed. So the answer is "multiple
recognizable groups": a large geographic group, a scraped-token group, and a heterogeneous remainder.

**Limits.** These are observations from one prompt, one anchor and one model. The labels are our reading
of each token. We did not compare against how common place names are in the whole vocabulary, and we
did not test why these tokens switch smoothly. The geographic group is consistent with the idea that
tokens close in meaning to ` France` give smooth switches. It does not show that closeness in meaning
causes them. The PLAN's threshold labels appear swapped relative to its question (see Methods).
Figure 2 settles which side is smooth under Direction 27's definition.
