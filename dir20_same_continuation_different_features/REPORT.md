# A plateau in a single-token activation interpolation signals *dissimilar* continuations, not shared ones

> Final, presentable, current-best only (history is in CHANGELOG.md).

## Summary

A common move in interpretability is to take the internal activation vector a language model computes
for prompt A, the one it computes for prompt B, and walk continuously from one to the other, watching
how the model's output changes. When the output stays put for a while and then switches abruptly — a
**plateau** followed by a jump — it is tempting to read that as a discrete internal feature flipping
state, and in particular as a sign that the two prompts share a continuation while differing in some
hidden property. This report tests that reading directly: if two prompts differ in one final token
but predict *nearly the same next token*, does interpolating that token's activation give a sharper
plateau?

It gives a **wider** one, in every model we tried. Working in three model families — GPT-2 medium
(355M), OPT-350m (331M) and Pythia-410m-deduped, each 24 blocks deep with a 1024-dimensional residual
stream — with a bank of 200 prompt pairs per model mined from WikiText-103, the rank correlation
between how differently the two prompts predict the next token and how sharp the transition is comes
out negative and strongly significant in all three. Away from the divergence ceiling, where the
divergence measure can still order pairs, Spearman $\rho = -0.61$ (GPT-2 medium, $n = 142$),
$-0.57$ (OPT-350m, $n = 129$) and $-0.45$ (Pythia-410m, $n = 127$), all $p < 10^{-7}$. More divergent
endpoints give sharper plateaus. Three separate sharpness statistics agree on the sign, and the effect
survives controlling for how far apart the two patched activations are geometrically.

Two supporting facts complete the picture. First, plateaus are the *default*: 82% of the mined GPT-2
medium pairs, 61% of the OPT pairs and 48% of the Pythia pairs cross our sharpness threshold, and a
deliberately dissimilar hand-picked control pair plateaus as hard as four hand-picked "same
continuation, different feature" pairs — in OPT-350m the control is the sharpest of the five. Second,
the sharpening is supplied by the depth that processes the edit, and we show this causally by moving
the patch site up the stack. Re-running the same 200 pairs with the interpolated vector inserted after
block 12 and after block 20 — leaving 11 and 3 blocks below it instead of 23 — walks the plateau away:
the share of sharp pairs falls from 82% to 50% to 10% in GPT-2 medium, from 61% to 37% to 1% in OPT,
and from 48% to 2.5% to 0% in Pythia, whose median response at block 20 is proportional to the edit to
within 2% of the linear baseline. The divergence effect is separable from this: in GPT-2 medium the
correlation holds at $-0.61$, $-0.53$, $-0.53$ across the three patch sites and in OPT at $-0.57$,
$-0.54$, $-0.55$, so depth sets how much the response is compressed while endpoint divergence sets
which pairs compress more.

For practice, this inverts a diagnostic. An observed plateau is not evidence that two prompts share a
continuation; the sharpest plateaus come from the pairs whose predictions agree least. Any claim
resting on plateau shape needs an interpolation between prompts of known dissimilarity to calibrate
against, and the *difference* between the two, not the absolute sharpness, is the only part that
carries information.

## Methods

### Data & Model

**Models.** Three final pretrained checkpoints, all frozen and in evaluation mode, float32, no
sampling: `gpt2-medium` (355M parameters, 24 blocks, $d_{model}=1024$),
`EleutherAI/pythia-410m-deduped` at `revision="step143000"` (24 blocks, $d_{model}=1024$) and
`facebook/opt-350m` (331M parameters, 24 blocks, $d_{model}=1024$). The three are matched on the two
structural quantities that the depth experiment below shows to matter — number of blocks and residual
width — and differ in tokenizer, architecture details and training corpus. OPT-350m is the informative
third point because its vocabulary is exactly GPT-2's 50257 token strings plus 8 special tokens and it
segments our prompts identically, while Pythia uses the GPT-NeoX vocabulary; a quantity that tracks
the tokenizer should put OPT with GPT-2.

**Hand-picked pairs (Experiment 1).** Five hand-written prompt pairs, each a shared prefix plus one
differing final token. Four are test pairs, chosen so the two versions of the sentence plausibly
continue the same way while differing in some internal property (identity vs. pronoun, word-form vs.
numeral, lower vs. upper case, chemical symbol vs. atomic number):

1. `Mary and John went to the store. John gave a book to` + ` Mary` / ` her`
2. `Two plus two is` + ` four` / ` 4`
3. `The answer is` + ` four` / ` Four`
4. `Which chemical element does this clue identify?` + ` Au` / ` 79`

The fifth is the **control**: `The house was` + ` big` / ` in`, an adjective against a preposition,
whose continuations are genuinely different. It plays the role of a negative example — under the
hypothesis it should be the one pair that does *not* plateau.

**Mined pair bank (Experiment 2).** Five hand-written pairs cannot measure an association: the 15
model-pair cells they produce only have the power to detect a correlation above about $\rho = 0.51$.
We therefore mine a bank of pairs automatically. We take the first 40 paragraphs of at least 400
characters from the WikiText-103 validation split (natural English, not written by us), truncate each
to a prefix of $L$ tokens with $L$ drawn uniformly from $[10, 40]$, and run the model on the prefix.
Final token A is always the model's **top-1** next token; final token B is the token at rank $r$, with
five values of $r$ drawn log-uniformly from $[1, 5000]$ per prefix. A rank-1 partner produces a
near-tie between two plausible continuations, a rank-5000 partner an implausible one, so the endpoint
divergence defined below spans its whole range by construction. That gives **200 pairs per model**
from 40 prefixes (the same prefixes and ranks in all three models; the tokens themselves are each
model's own). Building the inputs as `prefix_ids + [token_id]` makes the "identical prefix, one
differing single final token" condition exact by construction.

**Patch-depth replication (Experiment 4).** Reading the interpolation out at an earlier block shows
where sharpness accumulates but cannot show that the downstream blocks *cause* it, since reading
earlier is not the same as computing less. To get a causal handle we re-run the entire mined bank —
the same 200 pairs per model, the same tokens, the same 101 interpolation points — with the patch
applied after block 12 and after block 20 instead of block 0, so the interpolated vector is processed
by 11 or 3 remaining blocks rather than 23. The endpoint activations $h_A, h_B$ are then read at that
same block. Nothing else changes, so any difference in the sharpness statistics is attributable to the
amount of computation below the patch.

**Validity check (Experiment 1).** For the hand-written pairs we require, per model, that the two
prompts tokenize to an identical prefix and exactly one differing single final token; a pair failing
this in a model would be dropped for that model. All 5 pairs passed in all three models (prefix
lengths 3–14 tokens), so all 15 model-pair cells are reported and no multi-token interpolation was
performed.

**Hook point and sample sizes.** The default patch site is `resid_post` after block 0 — the residual
stream immediately after the first transformer block — at the **final token position only**; Experiment
4 repeats the bank at blocks 12 and 20. Because the prefix is identical and attention is causal, every
earlier position is bit-identical between the two prompts, so one forward pass per interpolation point
fully determines the run. For the hand-picked pairs, downstream `resid_post` is also recorded at the
final token of every later block (blocks 1–23) plus the final logits; for the mined bank only the final
logits are recorded. Every sweep uses 101 evenly spaced interpolation values on $[0,1]$, under
`torch.no_grad()` with fixed seeds. Total: $(5 + 200 \times 3)$ pairs $\times$ 3 models $\times$ 101
points.

### Metrics

The whole study depends on making "the model's output moved from A to B" a number, so we start there
and build up to the sharpness statistics and the association tests that consume them.

**Interpolation.** We need a path between the two activation vectors that does not shrink toward zero
in the middle, as a straight line between two high-dimensional vectors does. We therefore interpolate
the direction along the sphere (SLERP) and the length linearly. With $h_A, h_B$ the patched-layer
activations, $\hat h = h / \lVert h \rVert$ and $\Omega = \arccos(\hat h_A \cdot \hat h_B)$:

```math
h_\alpha \;=\; \Big[(1-\alpha)\lVert h_A\rVert + \alpha\lVert h_B\rVert\Big]\cdot
\frac{\sin\!\big((1-\alpha)\Omega\big)\,\hat h_A + \sin\!\big(\alpha\Omega\big)\,\hat h_B}{\sin\Omega}
```

$h_\alpha$ replaces the patched block's output at the final token and is run forward through the rest
of the model. At $\alpha=0$ and $\alpha=1$ this is the identity, which gives a free correctness check on the
harness (reported in Results). The angle $\Omega$ is also recorded per pair and reused later as a
control variable: it measures how far apart the two endpoints are geometrically, independently of what
they predict.

**Endpoint divergence (JSD)** — how differently the two complete prompts predict the next token, and
so the independent variable of the whole study. It is measured at inference from the full-vocabulary
softmax distributions $P_A, P_B$ at the final position, never from corpus counts. Jensen–Shannon
divergence is symmetric and stays finite when one distribution puts near-zero mass where the other
does not, where a plain KL divergence diverges. Units are nats; 0 means identical predictions:

```math
\mathrm{JSD}(P_A, P_B) \;=\; \tfrac{1}{2} D_{KL}\!\big(P_A \,\Vert\, M\big) + \tfrac{1}{2} D_{KL}\!\big(P_B \,\Vert\, M\big),
\qquad M = \tfrac{1}{2}\big(P_A + P_B\big)
```

JSD is bounded above by $\ln 2 \approx 0.693$ nats, attained when the two distributions have disjoint
support. That ceiling matters for the analysis: pairs that both predict completely different tokens
all pile up at the same $x$ value and can no longer be ordered, so we report the association both on
the full bank and on the **unsaturated** subset with $\mathrm{JSD} < 0.65$.

**Relative distance $d(\alpha)$** — where the model's output sits on the way from A to B. A raw
distance is not comparable across pairs whose endpoints are far apart to different degrees, so we
normalize by the total: $d=0$ means the output is exactly A's, $d=1$ exactly B's. For a vector
$x_\alpha$ read at any hook point (a downstream `resid_post`, or the final logit vector):

```math
d(\alpha) \;=\; \frac{\lVert x_\alpha - x_A\rVert_2}{\lVert x_\alpha - x_A\rVert_2 + \lVert x_\alpha - x_B\rVert_2}
```

This is the quantity plotted in Figure 1. A model that responds proportionally to the input edit gives
$d(\alpha)=\alpha$, the diagonal; a plateau is any large flat stretch followed by a fast rise.

**Transition width $w_{10-90}$** (primary sharpness statistic, fixed by the plan) — how much of the
sweep the output spends actually moving. It is the $\alpha$-distance between the first upward
crossings of $d=0.1$ and $d=0.9$, so a small value means the output ignored most of the interpolation
and then switched:

```math
w_{10\text{-}90} \;=\; \alpha(d=0.9) - \alpha(d=0.1)
```

Crossings are linearly interpolated on the 101-point grid. Following the plan, $w_{10-90} < 0.5$ is
called a clear plateau, against $0.8$ for the linear response. It drives the verdict column of Table 1
and the top row of Figure 4, and is the statistic tracked across depth in Figure 5.

**Total-variation width $w_{TV}$** (threshold-free sharpness statistic) — the same idea without the
fixed crossing levels. Non-monotonic curves are common (Table 1; only 7.5% of the mined GPT-2 medium
curves are monotonic): they dip and re-cross, which pushes the $d=0.1$ crossing far to the left and
makes a visibly sharp curve score as wide. Let $C(\alpha)$ be the fraction of the curve's total
variation accumulated by position $\alpha$; then $w_{TV}$ is the $\alpha$-span carrying the middle
half of all the movement:

```math
C(\alpha) = \frac{\int_0^{\alpha} \lvert d'(u)\rvert\,du}{\int_0^{1} \lvert d'(u)\rvert\,du},
\qquad w_{TV} \;=\; C^{-1}(0.75) - C^{-1}(0.25)
```

It is $0.5$ for a linear response and tends to $0$ for a step, so we call $w_{TV} < 0.25$ sharp.
Because it survives non-monotonicity it is the statistic used for the prevalence count in Figure 2 and
for the headline correlation.

**Plateau fraction PF** (second robustness statistic) — how much of the sweep sits pinned at an
endpoint, computed directly from the grid without any crossing logic, so it is immune both to
non-monotonicity and to the choice of which crossing counts. Over the $N=101$ grid points:

```math
\mathrm{PF} \;=\; \frac{1}{N}\,\#\big\lbrace \alpha_i : d(\alpha_i) < 0.1 \ \ \text{or} \ \ d(\alpha_i) > 0.9 \big\rbrace
```

Higher is more plateau-like; the linear response gives $0.2$. Its role in Results is to check that all
three sharpness statistics agree on the *sign* of the JSD relationship — with the sign convention
flipped, since large PF and small $w$ both mean "sharp".

**Association test.** The hypothesis predicts that low endpoint JSD goes with low width, so we score
it with the Spearman rank correlation $\rho$ between JSD and each sharpness statistic. Rank
correlation is used because the prediction is about ordering and the relationship need not be linear.
Pairs that share a prefix are not independent, so all confidence intervals come from a **cluster
bootstrap**: resample the 40 prefixes with replacement, take all pairs of the drawn prefixes, and
report the 2.5th and 97.5th percentiles of $\rho$ over 2000 resamples. We also report the ordinary
least-squares slope of each width on JSD (units: width per nat) with the same bootstrap, which gives
the effect an interpretable size.

**Divergence-matched model comparison.** The three models plateau at different rates, and a raw
prevalence count cannot say whether that is a fact about the models or about their banks landing at
different divergences. To separate the two we cut each bank into four fixed JSD bins
($[0, 0.2)$, $[0.2, 0.4)$, $[0.4, 0.65)$, $[0.65, \ln 2]$) and compare the median $w_{TV}$ of the
models within each bin. A gap that persists in every bin is a property of the model. This feeds
Figure 3 and is what lets us ask whether the sharing of a tokenizer predicts sharing a prevalence.

**Partial correlation (confound control).** A pair whose prompts predict different next tokens might
simply have more distant activations at the patch site, and distance alone could drive sharpness. To
separate the two we compute the partial Spearman correlation of JSD with $w_{TV}$ controlling for the
block-0 angle $\Omega$: rank-transform all three variables, linearly regress the JSD and $w_{TV}$ ranks
on the $\Omega$ ranks, and correlate the residuals.

### Baselines

**Linear response** — the null shape, the behavior of a model whose output moves in proportion to the
activation edit:

```math
d(\alpha) = \alpha \quad\Longrightarrow\quad w_{10\text{-}90} = 0.8,\quad w_{TV} = 0.5,\quad \mathrm{PF} = 0.2
```

It appears as the gray dashed diagonal in Figure 1 and the gray dashed line in Figures 2–6. Any value
below these is a plateau of some strength.

**Dissimilar-continuation control** — the `The house was` + ` big` / ` in` pair, run through identical
machinery. It fixes the value the plateau statistics take when two prompts emphatically do not share a
continuation, and is drawn with a thick frame in Figure 1 and a thick marker edge in Figures 2 and 4.

**Mined bank as a null distribution** — the 200 corpus-derived pairs per model give the distribution of
plateau sharpness for prompt pairs picked with no regard to continuation similarity at all. A
hand-picked pair is only interesting insofar as it sits away from this distribution; Figure 2 places
the five of them inside it.

**Smallest detectable correlation** — the reference against which a null claim is judged. For a
two-sided test at $\alpha = 0.05$ using the Fisher $z$ transform, the smallest $|\rho|$ that would
reach significance at sample size $n$ is:

```math
\rho_{\min}(n) \;=\; \tanh\!\Big(\frac{1.96}{\sqrt{n-3}}\Big)
```

This is $0.51$ at $n=15$ (the hand-picked cells, which is why they can only be suggestive) and $0.14$
at $n=200$ (the mined bank).

**Harness identity check** — patching $h_0$ and $h_1$ must reproduce the unpatched runs, giving
$d(0)=0$ and $d(1)=1$ exactly. Deviation from this measures implementation error, reported below.

## Results

**The harness is correct.** All 5 hand-written pairs tokenized validly in all three models, and across
all 1815 model-pair sweeps the patched runs at the endpoints reproduced the clean forward passes to
$|d(0)| \le 4 \times 10^{-4}$ and $|d(1) - 1| \le 4 \times 10^{-4}$. The numbers below are therefore
about the model, not about patching artifacts.

### Plateaus appear everywhere, including where they should not

**Table 1 — endpoint divergence and plateau strength for the five hand-picked pairs.** Reading the
table: JSD is the independent variable (small = the two prompts predict nearly the same next token);
$w_{10-90}$ and $w_{TV}$ are smaller when the transition is sharper; PF is larger when more of the
sweep is pinned at an endpoint. Bold marks the sharpest cell per model. The hypothesis predicts the
small-JSD rows should carry the small widths.

| Model | Prompt pair (final tokens) | endpoint JSD (nats) | $w_{10-90}$ | $w_{TV}$ | PF | monotonic |
|---|---|---|---|---|---|---|
| gpt2-medium | gave a book to ` Mary` / ` her` | 0.068 | 0.586 | 0.114 | 0.51 | no |
| gpt2-medium | Two plus two is ` four` / ` 4` | 0.138 | 0.454 | 0.232 | 0.55 | yes |
| gpt2-medium | The answer is ` four` / ` Four` | 0.377 | **0.120** | **0.058** | 0.88 | no |
| gpt2-medium | clue identify? ` Au` / ` 79` | 0.342 | 0.358 | 0.117 | 0.64 | no |
| gpt2-medium | *control:* The house was ` big` / ` in` | 0.659 | 0.516 | 0.272 | 0.50 | no |
| opt-350m | gave a book to ` Mary` / ` her` | 0.038 | 0.734 | 0.356 | 0.28 | no |
| opt-350m | Two plus two is ` four` / ` 4` | 0.027 | 0.907 | 0.680 | 0.11 | yes |
| opt-350m | The answer is ` four` / ` Four` | 0.472 | 0.530 | 0.293 | 0.48 | no |
| opt-350m | clue identify? ` Au` / ` 79` | 0.296 | 0.705 | 0.177 | 0.31 | no |
| opt-350m | *control:* The house was ` big` / ` in` | 0.646 | **0.143** | **0.068** | 0.85 | no |
| pythia-410m | gave a book to ` Mary` / ` her` | 0.033 | 0.582 | 0.268 | 0.43 | yes |
| pythia-410m | Two plus two is ` four` / ` 4` | 0.056 | 0.758 | 0.451 | 0.25 | yes |
| pythia-410m | The answer is ` four` / ` Four` | 0.271 | **0.340** | **0.135** | 0.66 | yes |
| pythia-410m | clue identify? ` Au` / ` 79` | 0.385 | 0.598 | 0.254 | 0.41 | yes |
| pythia-410m | *control:* The house was ` big` / ` in` | 0.665 | 0.425 | 0.137 | 0.57 | yes |

Thirteen of the fifteen cells land below the linear-response value of $w_{TV} = 0.5$ and eight cross
the sharp threshold of $0.25$, so this style of single-token interpolation does induce plateaus, at
strengths from mild to near-step (gpt2-medium `four`/`Four` moves through 80% of the gap in $0.12$ of
the sweep). The two exceptions are both `four`/`4`, one of the two *most* similar pairs in the study.
The control — an adjective against a preposition, at the largest divergence measured — is sharper than
both low-JSD pairs in gpt2-medium, sharper than three of four test pairs in pythia-410m, and in
opt-350m it is the sharpest cell of all five at $w_{TV} = 0.068$, a near-step. A diagnostic that fires
hardest on its own negative control is not yet measuring what it was supposed to measure.

The verdict rests first on the shape of the raw sweeps, so we show all fifteen before any summary
statistic. If plateaus tracked continuation similarity, the top two rows (lowest JSD) would be the
flattest and the bottom row of each model (the control) the most linear.

![Relative distance versus interpolation position for five prompt pairs in three models](plots/final_logit_curves.png)

**Figure 1.** Final-logit response to interpolating one token's block-0 activation. x: interpolation
position $\alpha$ from prompt A (0) to prompt B (1); y: relative distance $d$ (0 = at A's logits,
1 = at B's logits). Solid curve with circles = measured $d(\alpha)$; gray dashed = the linear reference
$d = \alpha$. Rows are prompt pairs, columns are models; the bottom row (thick frame) is the control
pair, whose continuations differ most. Most panels bend well away from the diagonal into a
flat-then-jump shape, the `four`/`4` panels stay closest to it, and the control bends at least as much
as the test pairs in every model. The wiggles in the gpt2-medium and opt-350m columns are the
non-monotonicity flagged in Table 1 and are why $w_{TV}$ exists.

### In a corpus-mined bank, the plateau is the default response

Five pairs cannot say whether a plateau is rare enough to be informative. To find out we ask how often
an arbitrary prompt pair plateaus, using the 200 WikiText-derived pairs per model, and where the
hand-picked pairs fall inside that distribution.

![Distribution of transition sharpness over 200 mined prompt pairs per model, with the five hand-picked pairs marked](plots/bank_prevalence.png)

**Figure 2.** Plateaus are the norm for arbitrary prompt pairs. x: $w_{TV}$ at the final logits
(smaller = sharper); y: number of mined pairs in each bin (gray hatched histogram, $n=200$ per model).
Gray dashed = the linear-response value ($0.5$), dotted = our sharpness threshold ($0.25$). The five
markers in the strip above each histogram are the hand-picked pairs of Table 1 at their $w_{TV}$
values (marker shape and color per the legend; the control has a thick black edge) — they are placed
on a separate strip because their y position carries no meaning. In gpt2-medium the mined mass piles
up near zero; in opt-350m and pythia-410m it centers near the threshold.

In gpt2-medium, **82% of arbitrary mined pairs are sharp** by our threshold (median $w_{TV} = 0.080$,
median $w_{10-90} = 0.241$); in opt-350m 61% are ($0.221$, $0.511$) and in pythia-410m 48% are
($0.266$, $0.593$). Every one of the hand-picked pairs sits inside the bulk of its model's
distribution — none is an outlier in the sharp direction. Observing a plateau for a hand-chosen pair
in gpt2-medium therefore carries almost no information: four in five random pairs do the same thing.

### The prevalence gap between models is real, and it does not follow the tokenizer

The three models sit 21 and 13 percentage points apart on that prevalence, which raises two questions
a practitioner would want answered before transferring any of this between models. Is the gap merely
an artifact of the three banks landing at different endpoint divergences, given that divergence is
about to be shown to drive sharpness? And does it track the tokenizer, since OPT-350m and GPT-2 medium
share a vocabulary while Pythia does not? Comparing the models inside fixed JSD bins answers both.

![Median transition width per endpoint-divergence bin for three models](plots/jsd_matched.png)

**Figure 3.** The prevalence gap between models survives matching on endpoint divergence, and it does
not follow the tokenizer. x: endpoint JSD bin in nats (the last bin is the $\ln 2$ ceiling), annotated
with the number of mined pairs per model in that bin; y: median $w_{TV}$ at the final logits over the
pairs in the bin (smaller = sharper). gpt2-medium = circles with a solid line; pythia-410m = squares
with a dashed line; opt-350m = triangles with a dotted line. Gray dashed = linear response ($0.5$),
dotted = sharp threshold ($0.25$).

**Table 2 — median $w_{TV}$ per model within fixed endpoint-divergence bins**, with the number of
mined pairs each model contributes to the bin.

| JSD bin (nats) | median $w_{TV}$ gpt2-medium | opt-350m | pythia-410m | $n$ per model |
|---|---|---|---|---|
| 0.00–0.20 | 0.263 | 0.496 | 0.421 | 44 / 43 / 23 |
| 0.20–0.40 | 0.103 | 0.317 | 0.276 | 23 / 15 / 22 |
| 0.40–0.65 | 0.043 | 0.147 | 0.220 | 75 / 71 / 82 |
| 0.65–0.69 | 0.047 | 0.166 | 0.274 | 58 / 71 / 73 |

gpt2-medium is the sharpest of the three in all four bins, by a factor of 2–4 on the median, so its
higher prevalence is a property of the model and not of how its bank happened to be distributed. The
tokenizer explanation fails: opt-350m tokenizes our prompts exactly as gpt2-medium does and still
plateaus 21 points less often, and its ordering against pythia-410m flips across the range — wider at
low divergence, sharper at high. What a practitioner should take from this is that the *rate* at which
plateaus appear has to be measured per model, while the *direction* of the divergence effect, which
every line in Figure 3 shows by falling from left to right, transfers across all three.

### Sharper plateaus go with *less* similar continuations

With $n=200$ per model the association test finally has power: it can detect $|\rho| \ge 0.14$, where
the 15 hand-picked cells could only have detected $|\rho| \ge 0.51$. Figure 4 plots every mined pair.

![Endpoint divergence against two sharpness statistics for 200 mined pairs per model, with fits](plots/bank_regression.png)

**Figure 4.** Endpoint divergence predicts sharpness — with the sign opposite to the hypothesis.
x (all panels): endpoint JSD in nats, larger = the two prompts predict more different next tokens;
the dash-dot vertical line is the $\ln 2$ ceiling that JSD attains for disjoint predictions.
y: $w_{10-90}$ (top row) and $w_{TV}$ (bottom row) at the final logits, smaller = sharper. Columns are
models. Light circles are the 200 mined pairs, the solid line is the OLS fit, the dashed line with
squares joins quintile means of JSD with $\pm 1$ standard error, and the stars are the five
hand-picked pairs (thick black edge = control). Gray dashed = linear response, dotted = plateau
threshold. All three models trend downward; pythia-410m's quintile means turn back up in the last two
quintiles, which sit at the JSD ceiling.

The correlations are collected below. All are Spearman $\rho$ between endpoint JSD and the named
statistic, with 95% cluster-bootstrap intervals over the 40 prefixes. Remember the sign convention:
negative $\rho$ for a width and positive $\rho$ for PF both mean "more divergent endpoints, sharper
plateau" — the *opposite* of the hypothesis, which predicts positive $\rho$ for widths.

**Table 3 — association between endpoint divergence and plateau sharpness, mined bank.**

| Model | statistic | $\rho$ | 95% CI | $p$ | OLS slope (per nat) |
|---|---|---|---|---|---|
| gpt2-medium | $w_{10-90}$ | $-0.47$ | $[-0.59, -0.33]$ | $1.4\times10^{-12}$ | $-0.64\ [-0.78, -0.49]$ |
| gpt2-medium | $w_{TV}$ | $-0.55$ | $[-0.66, -0.41]$ | $6.2\times10^{-17}$ | $-0.42\ [-0.54, -0.31]$ |
| gpt2-medium | PF | $+0.44$ | $[+0.31, +0.56]$ | $5.7\times10^{-11}$ | — |
| opt-350m | $w_{10-90}$ | $-0.43$ | $[-0.56, -0.27]$ | $3.5\times10^{-10}$ | $-0.53\ [-0.66, -0.39]$ |
| opt-350m | $w_{TV}$ | $-0.39$ | $[-0.55, -0.21]$ | $1.3\times10^{-8}$ | $-0.41\ [-0.55, -0.26]$ |
| opt-350m | PF | $+0.43$ | $[+0.28, +0.56]$ | $3.4\times10^{-10}$ | — |
| pythia-410m | $w_{10-90}$ | $-0.12$ | $[-0.30, +0.05]$ | $0.090$ | $-0.24\ [-0.35, -0.11]$ |
| pythia-410m | $w_{TV}$ | $-0.11$ | $[-0.30, +0.07]$ | $0.123$ | $-0.20\ [-0.31, -0.07]$ |
| pythia-410m | PF | $+0.12$ | $[-0.06, +0.31]$ | $0.090$ | — |

In gpt2-medium the effect is large and unambiguous, and its size is easy to read off the slope: moving
from two prompts that predict the same token to two that predict disjoint tokens ($0 \to 0.69$ nats)
shortens the transition by $0.29$ of the sweep on $w_{TV}$ — more than half of the $0.5$ that a linear
response would spend. The quintile means in Figure 4 fall monotonically ($w_{TV}$: $0.35 \to 0.13 \to
0.09 \to 0.07 \to 0.08$), so this is not one extreme group doing the work. All three sharpness
statistics agree on the direction in all three models, which is what separates a real effect from the
noise-driven sign disagreement five hand-picked pairs produce.

Pythia-410m looks weak in Table 3, and the reason is visible in Figure 4: 36.5% of its mined pairs sit
at or above JSD $0.65$, essentially at the $\ln 2$ ceiling, where the divergence measure can no longer
order them and they contribute a vertical stripe of noise (29.0% of gpt2-medium's pairs and 35.5% of
opt-350m's are similarly saturated). Removing the saturated pairs sharpens all three models and brings
them into agreement:

**Table 4 — the same test restricted to pairs below the JSD ceiling ($\mathrm{JSD} < 0.65$).**

| Model | $n$ | $\rho$ ($w_{TV}$) | $p$ | $\rho$ ($w_{10-90}$) | $p$ |
|---|---|---|---|---|---|
| gpt2-medium | 142 | $-0.61$ | $1.5\times10^{-15}$ | $-0.54$ | $4.9\times10^{-12}$ |
| opt-350m | 129 | $-0.57$ | $1.3\times10^{-12}$ | $-0.59$ | $3.1\times10^{-13}$ |
| pythia-410m | 127 | $-0.45$ | $9.0\times10^{-8}$ | $-0.47$ | $2.3\times10^{-8}$ |

All three models show a moderate-to-strong negative association at $p < 10^{-7}$ on both width
statistics. The finding is therefore not a GPT-2 idiosyncrasy, and it is not confined to one tokenizer
or one architecture family: in the regime where the independent variable is measurable at all, prompts
that predict more different continuations produce sharper plateaus everywhere we looked.

**The effect is not just endpoint geometry.** Pairs that disagree about the next token might simply
have activations further apart at the patch site, and larger separation might mechanically produce a
sharper crossover. Controlling for the block-0 angle $\Omega$ between the two patched vectors leaves
the association intact or strengthens it: partial $\rho = -0.55$ in gpt2-medium against a raw $-0.55$
(there $\Omega$ barely tracks JSD, $\rho = 0.03$), $-0.44$ against $-0.39$ in opt-350m, and $-0.16$
against $-0.11$ in pythia-410m, the last two with $\rho(\Omega, \mathrm{JSD}) \approx 0.30$. In all
three models $\Omega$ correlates only weakly with sharpness ($\rho = 0.13$–$0.16$). Whatever produces
the effect is carried by what the prompts predict, not by how far apart their activations start.

### Depth, not prompt content, manufactures the plateau shape

If sharpness is not evidence about shared continuations, it needs an explanation of its own. To locate
where it arises we recompute $w_{10-90}$ at every block's residual stream between the patch site and
the output, for the five hand-picked pairs.

![Transition width versus recording block for five prompt pairs in three models](plots/layerwise_widths.png)

**Figure 5.** The plateau is built up gradually across depth, not created at the patch site. x: the
block whose `resid_post` is read out at the final token (the patch is applied after block 0; the last
x value is the final logits); y: $w_{10-90}$ at that read-out point. One line per prompt pair, with
color, line style and marker all varying together (see legend). Gray dashed = linear response (0.8),
dotted = plateau threshold (0.5). Every pair starts near 0.8 immediately after the patch and narrows
with depth in all three models; the control (triangles, dash-dot) is among the fastest to sharpen, and
in opt-350m most of the sharpening arrives in the last two blocks.

One block after the patch, all fifteen cells sit at $w_{10-90} = 0.78$–$0.83$: the residual stream
still responds almost exactly proportionally to the edit. The width then falls across the following 23
blocks, reaching $0.12$–$0.91$ at the logits. A deep stack of nonlinear layers repeatedly compresses an
interpolated direction toward whichever endpoint dominates, and it does so for the control just as
readily as for the test pairs.

### Removing the downstream blocks removes the plateau

Figure 5 is only a description of where sharpness accumulates. Reading the residual stream out at an
earlier block is not the same as making the model compute less, so it cannot establish that those
blocks are what *builds* the plateau. The causal version of the experiment moves the patch site: we
re-run the entire mined bank with the interpolated vector inserted after block 12 and after block 20,
leaving 11 and 3 blocks below it instead of 23, with the pairs, the tokens and the interpolation grid
held fixed. If depth is doing the work, sharpness should decay as blocks are taken away.

![Median transition width and JSD-sharpness correlation against patch site for three models](plots/depth_effect.png)

**Figure 6.** Removing downstream blocks removes the plateau, but not the divergence effect in
gpt2-medium or opt-350m. x (both panels): the patch site — the block whose `resid_post` at the final
token is replaced by the interpolated vector — labelled with the number of blocks remaining below it.
Left y: median $w_{TV}$ at the final logits over the 200 mined pairs (smaller = sharper), shaded band =
interquartile range, gray dashed = linear response ($0.5$), dotted = sharp threshold ($0.25$). Right y:
Spearman $\rho$ between endpoint JSD and $w_{TV}$ over pairs below the $\ln 2$ ceiling (JSD $< 0.65$;
$n = 142$ gpt2-medium, $129$ opt-350m, $127$ pythia-410m), error bars = 95% cluster bootstrap over the
40 prefixes, gray dashed = no association. gpt2-medium = circles with a solid line; pythia-410m =
squares with a dashed line; opt-350m = triangles with a dotted line.

**Table 5 — plateau strength and the divergence association at three patch sites**, same 200 mined
pairs per model in every row. "% sharp" is the share of pairs with $w_{TV} < 0.25$; "monotonic" is the
share of $d(\alpha)$ curves that never decrease.

| Model | patch site | blocks below | median $w_{TV}$ | % sharp | median $w_{10-90}$ | monotonic | $\rho$(JSD, $w_{TV}$), JSD $<0.65$ | 95% CI | $p$ |
|---|---|---|---|---|---|---|---|---|---|
| gpt2-medium | block 0 | 23 | 0.080 | 82.0% | 0.241 | 7.5% | $-0.61$ | $[-0.70,-0.46]$ | $1.5\times10^{-15}$ |
| gpt2-medium | block 12 | 11 | 0.250 | 50.5% | 0.556 | 33.0% | $-0.53$ | $[-0.64,-0.39]$ | $1.1\times10^{-11}$ |
| gpt2-medium | block 20 | 3 | 0.383 | 10.0% | 0.701 | 72.0% | $-0.53$ | $[-0.66,-0.38]$ | $8.2\times10^{-12}$ |
| opt-350m | block 0 | 23 | 0.221 | 61.0% | 0.511 | 41.0% | $-0.57$ | $[-0.72,-0.37]$ | $1.3\times10^{-12}$ |
| opt-350m | block 12 | 11 | 0.307 | 36.5% | 0.641 | 77.5% | $-0.54$ | $[-0.67,-0.37]$ | $6.5\times10^{-11}$ |
| opt-350m | block 20 | 3 | 0.420 | 1.0% | 0.741 | 99.5% | $-0.55$ | $[-0.67,-0.39]$ | $1.6\times10^{-11}$ |
| pythia-410m | block 0 | 23 | 0.266 | 47.5% | 0.593 | 98.0% | $-0.45$ | $[-0.63,-0.24]$ | $9.0\times10^{-8}$ |
| pythia-410m | block 12 | 11 | 0.419 | 2.5% | 0.749 | 100% | $-0.44$ | $[-0.62,-0.23]$ | $2.4\times10^{-7}$ |
| pythia-410m | block 20 | 3 | 0.509 | 0.0% | 0.808 | 100% | $+0.04$ | $[-0.11,+0.22]$ | $0.62$ |

Depth is what makes the plateau. Sharpness decays monotonically as blocks are removed, in all three
models and on both width statistics, and the endpoint of that decay is the linear baseline itself:
with 3 blocks below the patch, pythia-410m has median $w_{TV} = 0.509$ and median $w_{10-90} = 0.808$
against the linear response's $0.5$ and $0.8$, and **not one of its 200 pairs is sharp**. opt-350m
reaches 1% sharp at the same site and gpt2-medium resists longest at 10%, but all three travel the same
path. The non-monotonic wiggles that made $w_{TV}$ necessary are a deep-stack product too: the share of
monotonic curves rises from 7.5% to 72.0% in gpt2-medium and from 41.0% to 99.5% in opt-350m as depth
is removed. This is the strongest form of the report's central warning: an experimenter who patches
early and sees a plateau is looking at a property of the 23 blocks below the patch, which are the same
23 blocks whatever the two prompts were.

The divergence effect is not the same phenomenon wearing a different hat. If more divergent pairs were
sharper only because a deep stack compresses them harder, the correlation should fade along with the
plateau. In gpt2-medium it does not move at all — $\rho = -0.61$, $-0.53$, $-0.53$ at 23, 11 and 3
remaining blocks, flat within the bootstrap intervals even where 90% of pairs no longer plateau — and
opt-350m behaves identically ($-0.57$, $-0.54$, $-0.55$) even where 99% no longer plateau. In
pythia-410m it holds at $-0.45$ and $-0.44$ and then vanishes ($+0.04$, $p = 0.62$) exactly at the
patch site where the response has gone linear and there is no transition shape left to modulate. The
two effects are therefore separable: the depth below the patch sets *how much* the response is
compressed, while endpoint divergence sets *which* pairs compress more, and in two of the three models
the latter is already fully expressed by the last three blocks. That is consistent with the competition
account — when the two endpoints predict disjoint token sets the two candidate outputs are well
separated and the winner flips abruptly, whereas near-identical predictions differ only in small logit
components that get carried across smoothly — with the refinement that the competition is resolved
close to the output, not accumulated over the whole stack.

## Conclusion

Interpolating a single token's early-layer activation between two prompts reliably produces a
plateau-then-jump logit response — in 82% (gpt2-medium), 61% (opt-350m) and 48% (pythia-410m) of 200
corpus-mined prompt pairs, and in 13 of 15 hand-picked model-pair cells. The plateau does not indicate
that the two prompts predict the same continuation. Across the mined bank the relationship runs the
other way and is statistically strong in all three model families: $\rho = -0.61$, $-0.57$ and $-0.45$
once pairs at the JSD ceiling are excluded, all with $p < 10^{-7}$, agreeing across three sharpness
statistics and surviving a control for activation geometry. The hand-picked negative control, chosen so
its two continuations clearly differ, plateaus as sharply as the four test pairs — and in opt-350m more
sharply than all of them.

For anyone using interpolation as an interpretability probe, the actionable point is that plateau shape
needs calibration before it can be evidence for anything. Report an interpolation between prompts of
known dissimilarity alongside the pair of interest and treat the difference as the signal; reading a
sharp plateau as "these two prompts share a continuation" gets the sign backwards, since in our banks
the sharp end of the distribution is where the *most* divergent pairs live. Figures 5 and 6 explain why
the bar is that high: the sharpening is supplied by the blocks downstream of the patch, which are the
same blocks in every condition, and taking those blocks away by patching at block 20 removes the
plateau outright — 0% of pythia-410m's pairs and 1% of opt-350m's stay sharp, at a median response
within 2% of the linear baseline for pythia. Two corollaries for experimental design follow. The patch
depth is itself a knob: a plateau seen with a late patch is far more informative than the same plateau
seen with an early one. And the calibration has to be redone per model: at matched endpoint divergence
gpt2-medium plateaus 2–4 times more sharply than the other two (Table 2), a gap that persists in every
divergence bin and does not follow the tokenizer, since opt-350m tokenizes identically to gpt2-medium
and behaves less like it than pythia-410m does in some bins.

**Limitations.** The pairs are constructed by swapping the final token for a lower-ranked alternative,
which spans endpoint divergence well but leaves the two prompts differing only at one position and only
in a way the model itself considered plausible; pairs differing in earlier tokens, or in more than one,
are untested. JSD saturates at $\ln 2$ and 29–37% of mined pairs sit near that ceiling, so the
full-bank correlations in Table 3 understate the effect and Table 4 is the cleaner estimate. All
results are for one patched position (the final token) in three models of similar size and identical
depth and width (24 blocks, $d_{model} = 1024$), at three patch sites; a 5-block model or a 60-block
model could sit anywhere on the depth curve of Figure 6. Table 2 rules the tokenizer out as the
explanation of the cross-model prevalence gap but does not identify what does explain it —
architecture, training corpus and pretraining length remain confounded across our three models.
Finally, $d(\alpha)$ measures movement in raw logit space, so a pair could hold its logit vector still
while reordering low-probability tokens and this metric would not see it.
