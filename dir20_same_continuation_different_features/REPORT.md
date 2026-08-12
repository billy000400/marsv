# Why do two prompts with the same prediction switch at different speeds? Matched-pair evidence that internal feature differences set transition width

## Summary

Take a prompt, replace its last token with two different tokens, and interpolate the internal
activation of that position from one to the other while running the rest of the network normally.
Sometimes the output flips from answer A to answer B inside a narrow window — it behaves like a switch;
sometimes it drifts across proportionally. Both happen for pairs whose *outputs* are about equally far
apart, so output divergence does not explain the difference. Something internal does. It matters
because interpretability treats a walk between two internal states as a walk between two
interpretations; if the shape of that walk is set by geometry, interpolation experiments are a mirage,
and so are safety arguments built on them.

**Hypothesis under test.** Among pairs matched on how much their predictions differ and on endpoint
geometry, the pair that engages more *different* downstream MLP neurons switches more sharply.

**What we did.** In GPT-2 Large we scored all 385020 candidate final-token pairs from 1395 held-out
WikiText-103 **test** paragraphs and locked 101 within-prefix contrasts — pairs of prompt-pairs sharing
a prefix, using four distinct final tokens, matched on successor divergence and four endpoint-geometry
confounds and differing only in the feature-difference score $F$ — writing the manifest and its SHA-256
to disk before computing a single interpolation curve. We call this the **amended analysis**, for the
reason below, and then ran a fully pre-registered **independent replication** on a different corpus
split.

**Why "amended".** The written plan fixed the bank at 300 prefixes and said to stop and report an
underpowered verdict if fewer than 40 matched contrasts survived. At 300 prefixes only 21 survived, and
the bank was enlarged to 1395 instead of stopping. No transition width had been computed and nothing
else changed, so the enlargement could not have been steered by the outcome — but it broke a rule
frozen in advance, and a result whose sample size was chosen after looking at the data is not a clean
pre-registered test. The 101-contrast result is therefore an amended analysis throughout, and the
confirmatory claim rests on the replication.

**What we found.** The prediction holds in both banks, with a large effect and a mechanism behind it.

- *Amended analysis (101 contrasts).* The higher-feature-difference member has the sharper switch in
  **83 of 101** contrasts; median width drops from $w_{TV} = 0.203$ to $0.098$, a median paired
  difference of $\Delta w = -0.071$ (bootstrap 95% CI $[-0.087, -0.058]$, permutation $p < 10^{-4}$).
- *Independent replication (99 contrasts, bank size and stopping rule frozen before any of its data
  was seen).* The effect reappears at nearly the same size: median $\Delta w = -0.064$, 95% CI
  $[-0.091, -0.043]$, 78.8% with the predicted sign, permutation $p < 10^{-4}$. All four clauses of its
  pre-registered gate are met, so the association is a confirmed result.
- *The neurons are causal (amended bank).* Forcing the ~1.7% of downstream MLP neurons that
  distinguish the two endpoints to interpolate linearly — leaving both endpoints bit-identical — takes
  the median width from $0.144$ to $0.471$, essentially a proportional response, while a size- and
  statistics-matched control set moves it to $0.167$. All 202 pairs show the gap.

**Why this matters for safety.** To claim that two inputs put a model in genuinely different internal
configurations, the sharpness of the switch between them is cheap evidence, and this report shows what
it is made of: how disjoint the machinery producing the two outputs is. The caution is that a sharp
switch says a *different set of neurons* is engaged, not that either set is interpretable.

**Scope.** One model (GPT-2 Large), one patch site (block-0 `resid_post` at the final token), one
interpolation rule, and a neuron-level proxy for "feature". In both banks the primary calipers yielded
very few contrasts (4 and 5), so the single pre-specified relaxation was applied, producing the 101 and
the 99. "Independent" here means independent data and a frozen protocol: the replication draws from a
corpus split no analysis in this direction has touched, and its sample size and stopping rule were
written down before any of its data was scored. Same code, same authors, so replication by another
group remains outstanding. The causal experiment was run on the amended bank only. `RESULTS.md` holds the
supporting analyses: the harness check, the robustness checks on the residual confound imbalance, and
raw curves of the strongest supporting and contradicting contrasts.

---

## Methods

### Data & Model

**Model.** `gpt2-large` (774M parameters, 36 transformer blocks, $d_{model} = 1280$, 5120 MLP neurons
per block), the pretrained Hugging Face checkpoint, frozen, float32, no sampling — the model the
phenomenon was originally reported in.

**Corpus (amended analysis).** The **test** split of WikiText-103 (raw), disjoint from earlier
exploratory work here. Every paragraph of at least 400 characters not beginning with a section marker
qualifies; there are 1395. From each we take one random 20–40 token span (generator seed 31) as a
shared **prefix**.

**Corpus (independent replication).** The **train** split of the same corpus, which no analysis in
this direction has ever touched: 80000 rows drawn at random and scanned under the identical paragraph
filter until **exactly 1400** prefixes are collected. That bank size was fixed in writing before any of
these paragraphs was scored (see *The independent replication and its frozen protocol* below).

**Candidate final tokens.** For each prefix we take the model's top 24 printable, non-special next
tokens. A **pair** is the prefix plus one such token against the prefix plus another; building inputs
as `prefix_ids + [token_id]` makes "identical prefix, one differing final token" exact. All
$\binom{24}{2} = 276$ pairs per prefix are scored: 385020 candidates in the amended bank, 386400 in the
replication bank.

**Hook point.** The patch site is `resid_post` after block 0 — the residual stream immediately after
the first transformer block — at the **final token position only**; the prefix is shared and attention
is causal, so every earlier position is bit-identical. Blocks 1–35 are the "downstream" blocks; block 0
is excluded from the feature score because its output is what is interpolated.

**Sample sizes.** 806 sweeps of 101 forward evaluations each: 2 harness check, 202 amended matched
test, 198 replication, 404 causal test. Endpoint reproduction error never exceeded
$9.2 \times 10^{-7}$ (relative L2 on final-token logits).

### Metrics

The chain is: define "the two prompts predict the same thing", then "they use different machinery",
then "it switched sharply" — match on the first, contrast on the second, measure the third.

**Interpolation path.** A straight line between two activations passes through vectors much shorter
than either endpoint, confounding "the switch" with "the norm collapsed", so we interpolate direction
along the sphere and length linearly. With $h_A, h_B$ the block-0 activations,
$\hat h = h / \lVert h \rVert$ and $\Omega = \arccos(\hat h_A \cdot \hat h_B)$:

```math
h_\alpha \;=\; \big[(1-\alpha)\lVert h_A\rVert + \alpha\lVert h_B\rVert\big] \cdot
\frac{\sin\!\big((1-\alpha)\Omega\big)\,\hat h_A + \sin(\alpha\Omega)\,\hat h_B}{\sin \Omega},
\qquad \alpha \in \{0, 0.01, \ldots, 1\}.
```

At $\alpha = 0$ this is exactly $h_A$ and at $\alpha = 1$ exactly $h_B$, so the endpoint error is a real
check on the harness.

**Successor JSD — the matching variable.** "Same continuation" has to become a number: a symmetric,
bounded measure of how differently the two prompts predict the *next* token. With $p_A, p_B$ their
full-vocabulary next-token distributions and $m = (p_A + p_B)/2$:

```math
\mathrm{JSD}(A,B) \;=\; \tfrac{1}{2} D_{KL}(p_A \Vert m) \;+\; \tfrac{1}{2} D_{KL}(p_B \Vert m),
\qquad D_{KL}(p \Vert q) = \sum_v p_v \log \frac{p_v}{q_v},
```

in natural-log units, from 0 (identical predictions) to $\log 2 \approx 0.693$ (disjoint support). Low
JSD is what makes a contrast interesting: if two prompts already predict different things, of course
the model treats them differently. JSD drives the eligibility filter and the balance table.

**Feature difference $F$ — the independent variable.** "These two prompts light up different machinery"
has to be one number, computed from the endpoints alone. Raw activation magnitude will not do: a large
post-GELU activation on a neuron with tiny output weights changes nothing downstream, so we weight each
neuron's activation by how much it can write into the residual stream. For neuron $j$ in block $l$ with
post-GELU activation $a_{l,j}$ and output-weight row $W^{out}_{l,j}$:

```math
s_{l,j} \;=\; \lvert a_{l,j} \rvert \cdot \lVert W^{out}_{l,j} \rVert_2 .
```

For each endpoint we keep the top 64 neurons by $s$ **within each of blocks 1–35**, treat $(l, j)$ as a
feature identity, and call the resulting 2240-element set $S_A$ (resp. $S_B$). The score is the Jaccard
distance between them:

```math
F(A,B) \;=\; 1 - \frac{\lvert S_A \cap S_B \rvert}{\lvert S_A \cup S_B \rvert} \;\in\; [0, 1].
```

$F = 0$ means an identical top set, $F = 1$ completely disjoint sets; per-block selection stops the
blocks with the largest activations from dominating. This is an **MLP feature proxy**: a top-scoring
neuron is a unit of computation the prompt engages, and nothing here shows it is human-interpretable.

**Relative distance $d(\alpha)$ — the response curve.** To say where along the path the output "is", we
place each interpolated final-token logit vector $x_\alpha$ on the segment between $x_A$ and $x_B$:

```math
d(\alpha) \;=\; \frac{\lVert x_\alpha - x_A \rVert_2}{\lVert x_\alpha - x_A \rVert_2 + \lVert x_\alpha - x_B \rVert_2}.
```

$d(0) = 0$, $d(1) = 1$, and $d$ is invariant to logit scale. A proportional response traces
$d(\alpha) = \alpha$; a switch stays near 0, jumps, and stays near 1.

**Transition width $w_{TV}$ — the outcome.** The obvious sharpness statistic, the $\alpha$-distance
between the first crossings of $d = 0.1$ and $d = 0.9$, is undefined whenever a curve doubles back, and
mining a corpus guarantees some do, so the outcome is built on cumulative variation, always defined.
With $d_i = d(\alpha_i)$ over the 101 grid points:

```math
c_k \;=\; \frac{\sum_{i=1}^{k} \lvert d_i - d_{i-1} \rvert}{\sum_{i=1}^{100} \lvert d_i - d_{i-1} \rvert},
\qquad
w_{TV} \;=\; \alpha(c = 0.75) - \alpha(c = 0.25),
```

with $\alpha(c)$ from linear interpolation of the $c$ grid. In words: the width of the $\alpha$-window
in which the middle half of the curve's total movement happens. **Smaller is sharper**; a proportional
response gives exactly $0.5$ and a perfect step approaches $0$.

**Paired effect $\Delta w$ — the test statistic.** Each locked contrast $i$ contributes one number,

```math
\Delta w_i \;=\; w_{TV}(\text{high-}F)_i - w_{TV}(\text{low-}F)_i ,
```

and the hypothesis predicts $\Delta w < 0$. Sharing a prefix removes prefix-level width variation,
which is large. We summarise with the median (robust to the heavy right tail of widths), a bootstrap
95% CI over contrasts (10000 resamples), the fraction with the predicted sign, and a paired permutation
$p$-value (10000 sign flips).

**Balance — standardized mean difference (SMD).** To show the contrast isolates $F$, we report for each
matched variable $v$ the gap between the groups in units of the pooled spread:

```math
\mathrm{SMD}(v) \;=\; \frac{\bar v_{\text{high-}F} - \bar v_{\text{low-}F}}{\mathrm{sd}\big(v_{\text{high-}F} \cup v_{\text{low-}F}\big)} .
```

Values near 0 mean matched; the conventional threshold for "well balanced" is $0.25$.

**Causal intervention — differential-neuron linearization.** An association between $F$ and width could
run either way, so the intervention removes exactly the nonlinearity the hypothesis blames. Let $D$ be
the symmetric difference $S_A \triangle S_B$. At every $\alpha$, for every $j \in D$, we overwrite that
neuron's post-GELU activation at the final token with the straight line between its endpoint values:

```math
a'_j(\alpha) \;=\; (1-\alpha)\, a_j(A) \;+\; \alpha\, a_j(B), \qquad j \in D,
```

while the block-0 patch and every other neuron run normally. At $\alpha = 0$ the forced values equal
$a_j(A)$ and the block-0 activation is $h_A$, so the run is bit-identical to prompt A's clean run (and
likewise at $\alpha = 1$): both endpoints are preserved, so $w_{TV}$ still describes that condition's
switch. The statistic is the per-pair gap against the control below,

```math
g \;=\; \big[w_{TV}(\text{differential}) - w_{TV}(\text{unablated})\big] - \big[w_{TV}(\text{control}) - w_{TV}(\text{unablated})\big],
```

with the prediction $g > 0$, summarised exactly as $\Delta w$ is.

### Baselines

**The linear response.** The reference for "no compression" is an output moving proportionally to
$\alpha$: $d(\alpha) = \alpha$, giving $w_{TV} = 0.5$. Every width is read against it.

**The low-$F$ member — the matched baseline.** The comparison that carries the primary result is
internal to each contrast: two pairs share a prefix, use four distinct final tokens, and sit within
calipers of each other on successor JSD and the confound space. The low-$F$ member is the baseline for
its partner's width, so prefix-level width variation cancels and the only systematic difference left is
$F$.

**Matching rule and its single relaxation.** The confound distance between candidate pairs $a, b$ is
the standardized Euclidean distance over four endpoint-geometry variables — final-logit L2 distance,
block-0 endpoint angle, block-0 absolute log norm ratio, and mean surprisal of the final tokens — each
scaled by its standard deviation $\sigma_c$ over the eligible bank:

```math
\mathrm{dist}(a,b) \;=\; \sqrt{\sum_{c \in \mathcal{C}} \left(\frac{a_c - b_c}{\sigma_c}\right)^{2}},
\qquad \mathcal{C} = \{\text{logit dist},\ \text{angle}_0,\ \lvert \log \text{norm ratio} \rvert,\ \text{surprisal}\}.
```

Eligibility, fixed in advance: $0.005 \le \mathrm{JSD} \le 0.20$ and final-logit distance above the
bank's 10th percentile (which stops near-identical endpoints from making $d(\alpha)$ noise-dominated).
Primary calipers: $\lvert \Delta \mathrm{JSD} \rvert \le 0.01$, $\mathrm{dist} \le 0.50$,
$\Delta F \ge 0.10$; one pre-specified relaxation if fewer than 80 contrasts survive:
$\lvert \Delta \mathrm{JSD} \rvert \le 0.02$, $\mathrm{dist} \le 0.75$, $\Delta F \ge 0.08$. At most one
contrast per prefix, maximising $\Delta F$ then minimising $\mathrm{dist}$.

**The matched control neuron set.** For the causal test the baseline is an equal-size set of neurons
that is *not* differential: for each block, $\lvert D_l \rvert$ neurons from outside $S_A \cup S_B$,
each the nearest unused neuron to a differential neuron in the standardized 3-D space of (mean
contribution magnitude $\tfrac{1}{2}(\lvert a_j(A)\rvert + \lvert a_j(B)\rvert)\lVert W^{out}_j \rVert$,
endpoint activation gap $\lvert a_j(A) - a_j(B) \rvert$, output-weight norm $\lVert W^{out}_j \rVert$).
Same count, same blocks, same activation statistics, so any effect difference is
attributable to which neurons were chosen.

**Pre-registration, locking, and the amendment.** Pre-registration here means writing the metrics,
filters, sample size and decision rule to disk before any of the data to be analysed has been scored,
so the analysis cannot be tuned to the answer. Every metric, caliper and success criterion above was
written into `PLAN.md` before the bank was mined, and the chosen contrasts were written to
`results/matched_pairs.json` and hashed
(`2415f5ff6dfcf88fb9cc7a67b87c93d859434296310f4b8d406c6f545e23ff56`) before any interpolation curve
existed. The success gate — $n \ge 80$, median $\Delta w \le -0.05$, at least 60% with the predicted
sign, and a 95% CI entirely below zero — was likewise fixed in advance.

One clause was not honoured. The plan fixed the bank at 300 prefixes and instructed us to stop with an
underpowered verdict if fewer than 40 contrasts survived; 21 survived, and the bank was extended to all
1395 eligible test paragraphs. All definitions, filters and calipers were unchanged and no
interpolation curve had been computed, so the enlargement was blind to the outcome, but the stopping
rule was frozen and we departed from it. Everything computed on that bank is labelled the **amended
analysis** for this reason.

**The independent replication and its frozen protocol.** To recover a clean pre-registered test, a
second bank was built, and this was written to `JOURNAL.md` before a single replication prefix was
scored: corpus = the WikiText-103 **train** split (the only split untouched by any analysis in this
direction; the amended analysis used *test*, earlier exploratory work *validation*), 80000 rows with
generator seed 132, same paragraph filter, one 20–40 token span per paragraph with generator seed 131;
**bank size fixed at exactly 1400 prefixes**; the bank to be run **once**, with no enlargement,
re-seeding or re-drawing for any reason and no second relaxation of the calipers; every other element
of the protocol identical to the amended analysis; and the same four-clause gate as the decision rule,
with $40 \le n < 80$ to be reported as underpowered and $n < 40$ as a failure to power. The 1400
follows from the amended yield of 101 contrasts per 1395 prefixes, putting the expected yield just
above the gate's $n \ge 80$. The bank was locked to `results/matched_pairs_rep.json` and hashed
(`ed1df0866f012b6195521dcda0d81306c7c6cb9d00e5dca2b30cda62e9af6d6b`) before its first interpolation
curve was computed. No replication width existed while these choices were being made.

---

## Results

### 101 contrasts, matched on everything except the feature difference (amended analysis)

The harness first reproduced the observation on the two prompt pairs it was first reported with
(`RESULTS.md`, S1): they differ 24-fold in $w_{TV}$ but also 13-fold in successor JSD, so alone they
cannot separate "different predictions" from "different features" — hence the matched design.

The contrasts were chosen without any knowledge of the outcome, so we report what the locking produced
before any width. Across 1395 prefixes, 385020 candidate pairs were scored; 26275 passed the
eligibility window (only 6.8% of pairs predict similarly enough to qualify); the primary calipers left
4 contrasts, so the single pre-specified relaxation was applied and 101 survived, with a median
$\Delta F$ of 0.095. The 1395 prefixes are themselves the amendment: the plan had fixed 300, which
yielded 21, below its own floor of 40.

The balance table is the design's own audit: if the confounds are matched and $F$ is not, a difference
in width is attributable to $F$.

| Matched variable | high-$F$ mean | low-$F$ mean | standardized mean difference |
|---|---|---|---|
| successor JSD (nats) | 0.0967 | 0.0951 | **+0.030** |
| block-0 \|log norm ratio\| | 0.1147 | 0.1143 | **+0.005** |
| mean final-token surprisal (nats) | 5.218 | 5.175 | **+0.025** |
| final-logit L2 distance | 288.0 | 279.1 | +0.231 |
| block-0 endpoint angle (rad) | 1.0614 | 1.0267 | +0.252 |
| feature difference $F$ (the variable under test) | 0.8652 | 0.7622 | **+1.506** |

Successor JSD — the variable the question is posed at fixed value of — is balanced to 3% of a standard
deviation, while the two groups differ by 1.5 standard deviations on $F$. That 50-fold ratio is what
lets a width difference be read as a feature effect. Final-logit distance and block-0 angle retain
about a quarter of a standard deviation of imbalance, at the conventional boundary for "well balanced"
and both in the direction that would flatter the hypothesis; the robustness checks in `RESULTS.md`
remove them and the effect survives.

### Pairs that engage more different features switch more sharply (amended analysis)

Every locked contrast was swept identically — 101 interpolation points at block-0 `resid_post` of the
final token, readout at the final-token logits — with endpoint reproduction exact to
$9.2 \times 10^{-7}$, so the widths describe the model and not patching error. Because this bank's size
came from breaking the plan's stopping rule, the gate column records that the amended analysis clears
the same four thresholds the plan set, not that a pre-registered test was passed; that test is the
replication in the next subsection.

| Summary | Value | Gate | Met |
|---|---|---|---|
| Contrasts $n$ | 101 | $\ge 80$ | yes |
| Median $\Delta w = w_{TV}(\text{high-}F) - w_{TV}(\text{low-}F)$ | **$-0.0708$** | $\le -0.05$ | yes |
| Bootstrap 95% CI on the median | $[-0.0866, -0.0582]$ | below 0 | yes |
| Fraction with the predicted sign ($\Delta w < 0$) | **82.2%** (83/101) | $\ge 60$% | yes |
| Paired permutation $p$ (10000 sign flips) | $< 10^{-4}$ | — | — |
| Median $w_{TV}$, low-$F$ → high-$F$ | $0.203 \rightarrow 0.098$ | — | — |
| **Verdict** | **supported** | | |

The effect is large in the units that matter. The median low-$F$ pair completes the middle half of its
output movement across 20% of the interpolation range; its matched high-$F$ partner does it across 10%.
That factor of two comes from a median $\Delta F$ of 0.095 — about a 10-percentage-point difference in
how much of their top-neuron sets the two endpoints share. The classical first-crossing width moves the
same way (`RESULTS.md`), so this is a property of the curves, not of the total-variation definition.
The design's strength is that successor divergence is held fixed: divergence and feature overlap move together in a
free-range bank, which makes a cross-pair correlation between them uninformative about mechanism, while
inside a matched contrast they are decoupled and the feature term survives. Post-hoc checks in `RESULTS.md`
confirm the residual imbalance is not producing the effect: the covariate-adjusted estimate is slightly
*larger* ($-0.085 \pm 0.013$) than the raw one, and the effect holds in the subsets where the imbalance
points the other way. Figure 1 shows the paired widths and the distribution of $\Delta w$.

![Paired low-F to high-F width lines and the distribution of their differences](plots/matched_widths.png)

**Figure 1.** The primary result of the amended analysis. Left — x: which member of the contrast
(low-$F$ at left, high-$F$ at right); y: transition width $w_{TV}$ (smaller = sharper). Each thin line
is one contrast: solid lines fall (the predicted direction), dashed rise; the heavy black line with
circles joins the medians, $0.203 \rightarrow 0.098$. Right — x: the paired difference $\Delta w$; y:
number of contrasts; dotted vertical = zero, dashed = the median $-0.071$, shaded band = its bootstrap
95% CI.

### The independent replication passes its pre-registered gate

Everything above rests on a bank whose size was chosen after seeing that a smaller bank was too small.
However blind that enlargement was, it makes the amended analysis hypothesis-generating: with the
stopping rule broken once, a reader has no guarantee it would not have been broken again had the
numbers come out otherwise.

The commitment held. The replication bank produced 1400 prefixes, 386400 candidate pairs and 25321
eligible pairs; the primary calipers left 5 contrasts, so the one pre-specified relaxation was applied
and 99 survived — not enlarged, re-seeded or re-drawn. Confound balance
matched the amended bank: SMDs of $+0.03$ on successor JSD, $-0.05$ on the log norm ratio and $+0.09$
on surprisal against $+1.63$ on $F$, with the same residual imbalance on final-logit distance ($+0.20$)
and block-0 angle ($+0.29$).

| Summary (replication, train split) | Value | Gate | Met |
|---|---|---|---|
| Contrasts $n$ | 99 | $\ge 80$ | yes |
| Median $\Delta w$ | **$-0.0641$** | $\le -0.05$ | yes |
| Bootstrap 95% CI on the median | $[-0.0908, -0.0426]$ | below 0 | yes |
| Fraction with the predicted sign | **78.8%** (78/99) | $\ge 60$% | yes |
| Paired permutation $p$ (10000 sign flips) | $< 10^{-4}$ | — | — |
| Median $w_{TV}$, low-$F$ → high-$F$ | $0.173 \rightarrow 0.095$ | — | — |
| **Verdict** | **supported (pre-registered)** | | |

The replication reproduces the amended analysis closely: $-0.064$ against $-0.071$, each estimate
inside the other's confidence interval, and the same near-halving of the median width. Its interval is
wider ($0.048$ across against $0.028$), as 99 contrasts from a different corpus split should be, and
its upper end sits at $-0.043$, still clear of zero but closer to it than the amended $-0.058$.
Figure 2 puts the two side by side.

![Two horizontal confidence intervals and paired bars comparing the amended analysis with the replication](plots/replication_forest.png)

**Figure 2.** The amended analysis against the pre-registered independent replication. Left — x: the
median paired difference $\Delta w$, negative meaning the high-$F$ member switches more sharply; each
horizontal bar is one bank's bootstrap 95% CI with its median marked (circle = amended, test split,
$n = 101$; square = replication, train split, $n = 99$); dashed vertical = the gate threshold $-0.05$,
dotted = zero. Right — x: which member of the contrast; y: that group's median $w_{TV}$; bars hatched
`//` are the amended analysis, `xx` the replication; the dotted horizontal at $0.5$ marks a
proportional response.

The association between feature difference and transition sharpness is therefore confirmed: predicted
in advance, tested once on data chosen in advance, against a decision rule written before that data was
scored. The amended contrasts still supply the better-powered estimate and the pairs the causal
experiment used, but the existence of the effect no longer depends on them.

### The differential neurons cause the switch

The matched design supports the association but cannot order the causation, so the final experiment
intervenes on the quantity the hypothesis names. It was run on the amended bank, before the replication
existed, and has no pre-registered counterpart. For all 202 amended pairs the sweep was re-run twice:
once with the symmetric-difference neurons forced to interpolate linearly, once with the matched
control set. Both reproduce both endpoints bit-identically (to $8.9 \times 10^{-7}$), so all three
widths describe switches between the same two states.

| Condition | median $w_{TV}$ | median change from unablated |
|---|---|---|
| Unablated (the primary sweeps) | 0.144 | — |
| Control set linearized (1.7% of neurons) | 0.167 | $+0.019$ |
| Differential set linearized (1.7% of neurons) | **0.471** | $+0.308$ |
| Gap (differential $-$ control), median | | **$+0.275$**, 95% CI $[0.251, 0.298]$ |
| Pairs with the predicted sign | 202/202 (**100%**) | permutation $p < 10^{-4}$ |

Linearizing a median of 3063 neurons — 1.7% of the 179200 MLP neurons below the patch — takes the
median pair from a switch completing in 14% of the interpolation range to within 0.03 of proportional.
The control touches the same number of neurons, in the same blocks, with matched activation statistics,
and moves the median by 0.019: a sixteenth of the effect. The mechanism is specific to which neurons
were selected, not to the amount of intervention.

The most informative number is the convergence. Under the intervention the high-$F$ member goes
$0.098 \rightarrow 0.467$ and the low-$F$ member $0.203 \rightarrow 0.474$: the two groups that differed
by a factor of two land within 0.007 of each other. The width difference is carried by the neurons that
distinguish the endpoints, and neutralising their nonlinearity removes the switch and the group
difference at once — which makes the claim mechanistic and gives a practitioner a 1.7%-of-neurons
handle on why an interpolation snaps. Figure 3 shows all 202 pairs move the same way, so this is not a
story about the median.

![Per-pair widths under three conditions and the distribution of the differential-minus-control gap](plots/causal_linearization.png)

**Figure 3.** The causal test on the amended bank. Left — x: condition (unablated, control linearized,
differential linearized); y: transition width $w_{TV}$; each thin line is one of the 202 pairs, the
heavy black line with circles joins the medians, and the dotted horizontal marks the linear response
$w_{TV} = 0.5$. Right — x: the per-pair gap $g$, how much more the differential linearization widened
the switch than the control did; y: number of pairs; dotted vertical = zero, dashed vertical = the
median $+0.275$, shaded band = its bootstrap 95% CI. The entire distribution lies above zero.

### Limitations

**One model, one site, one path — and "feature" is a proxy.** Everything is GPT-2 Large, patched at
block-0 `resid_post` of the final token, along a rescaled-SLERP path; other depths, architectures and
interpolation rules are untested here. $F$ counts top-scoring MLP neurons weighted by output-weight
norm, and nothing here shows they are monosemantic or human-interpretable, so the honest reading of the
causal result is that the *set of neurons that differ between the endpoints* carries the switch.

**The relaxation was used in both banks.** The primary calipers produced 4 contrasts in the amended
bank and 5 in the replication bank, so both reported samples come from the relaxed rule
($\lvert \Delta \mathrm{JSD} \rvert \le 0.02$, confound distance $\le 0.75$, $\Delta F \ge 0.08$), fixed
in advance and applied once in each. The resulting contrasts are matched less tightly than intended,
visible in the residual SMDs of $+0.20$ to $+0.29$ on final-logit distance and block-0 angle. The
`RESULTS.md` robustness checks address that in the amended bank; they were not repeated in the
replication, whose role is the single pre-registered decision.

**The amended analysis broke the plan's stopping rule, and only the replication is confirmatory.** The
plan required stopping with an underpowered verdict below 40 contrasts at 300 prefixes; 21 survived and
the bank was extended to 1395. The change was blind to the outcome — no width computed, no definition
changed — but it departs from a rule frozen in advance, so that bank's results are labelled an amended
analysis and the confirmatory claim rests on the replication.

**The replication is independent in data, not in personnel.** Its corpus split, sample size and
stopping rule were fixed before any of its data was scored, and it was run once. It used the same code,
patch site and authors, so it cannot detect a systematic error shared by both banks — a bug in the
width metric or the patching harness, for instance. Replication by another group, and on another model,
remains outstanding. The causal experiment has no pre-registered replication at all: run on the amended
bank only, it is the amended analysis's strongest evidence rather than a confirmed result.

**The intervention is blunt.** Forcing 3063 activations to a straight line is a strong manipulation:
the control set shows that the *choice* of neurons matters, but the minimal sufficient set is
unmeasured.

---

## Conclusion

Among GPT-2 Large prompt pairs matched on successor JSD and endpoint geometry, the pair that engages
more different downstream MLP features has the sharper block-0 transition. Two banks say so, with
different weight. The amended analysis (101 contrasts, test split) gives the better-powered
estimate, median $\Delta w = -0.071$ (Figure 1), but its sample size was set after seeing that the
planned 300 prefixes were too few. The pre-registered replication (99 contrasts, train split, bank size
and single-run stopping rule fixed before any of its data was scored) gives $-0.064$ and meets all four
gate clauses (Figure 2), so the association is a confirmed result.

The causal finding rests on the amended bank alone: forcing the 1.7% of neurons that distinguish the
two endpoints to interpolate linearly collapses the switch from $w_{TV} = 0.144$ to $0.471$, an
essentially proportional response, while a matched control set leaves it at $0.167$, in all 202 pairs
(Figure 3).

**Verdict: supported — the matched association is confirmed by a pre-registered independent
replication; the causal mechanism rests on the amended analysis and awaits one.**
