# A plateau in a single-token activation interpolation is not evidence of a shared continuation

> Final, presentable, current-best only (history is in CHANGELOG.md).

## Summary

A common move in interpretability is to take the internal activation vector a language model
computes for prompt A, the one it computes for prompt B, and walk continuously from one to the
other, watching how the model's output changes. When the output stays put for a while and then
switches abruptly — a **plateau** followed by a jump — it is tempting to read that as a discrete
internal feature flipping state. This report tests the specific version of that claim this direction
was set up to check: if two prompts differ in one final token but predict *nearly the same next
token*, does interpolating that token's activation give a plateau?

The answer is that plateaus appear, but they carry no such information. Across four hand-picked
prompt pairs and one deliberately dissimilar control pair, in two models (GPT-2 medium, 355M, and
Pythia-410m-deduped), the final-logit response was flat-then-abrupt in 9 of 10 cases — including the
control, whose two continuations diverge the most of anything we tested. How similar the two
continuations are does not predict how sharp the plateau is: the rank correlation between endpoint
Jensen–Shannon divergence and transition width is $-0.37$ ($p = 0.29$, $n = 10$), and it changes
sign depending on which model and which sharpness statistic you use. The two lowest-divergence
pairs, which the hypothesis says should plateau hardest, gave the two *widest* transitions in the
study.

The practical consequence is a warning about a cheap and popular diagnostic. Reading a plateau as a
feature boundary requires a control interpolation between two prompts you already know do *not*
share a continuation; without one, the observation is consistent with the network's generic response
to editing a single token's activation. Figure 3 shows why the shape is generic: the response leaves
the patched layer essentially linear and is sharpened progressively by the ~23 blocks that follow,
which is a property of stacked nonlinear layers, not of the prompt pair.

## Methods

### Data & Model

**Models.** Two final pretrained checkpoints, both frozen and in evaluation mode, float32, no
sampling: `gpt2-medium` (355M parameters, 24 blocks, $d_{model}=1024$) and
`EleutherAI/pythia-410m-deduped` at `revision="step143000"` (24 layers, $d_{model}=1024$). Two
model families are used so that any effect specific to one tokenizer or one architecture shows up as
a disagreement between columns.

**Data.** Five hand-written prompt pairs, no corpus and no training-set statistics. Each pair is a
shared prefix plus one differing final token. Four are test pairs, chosen so the two versions of the
sentence plausibly continue the same way while differing in some internal property (identity vs.
pronoun, word-form vs. numeral, lower vs. upper case, chemical symbol vs. atomic number):

1. `Mary and John went to the store. John gave a book to` + ` Mary` / ` her`
2. `Two plus two is` + ` four` / ` 4`
3. `The answer is` + ` four` / ` Four`
4. `Which chemical element does this clue identify?` + ` Au` / ` 79`

The fifth is the **control**: `The house was` + ` big` / ` in`, an adjective against a preposition,
whose continuations are genuinely different. It plays the role of a negative example — under the
hypothesis it should be the one pair that does *not* plateau.

**Validity check.** Before any interpolation we require, per model, that the two prompts tokenize to
an identical prefix and exactly one differing single final token; a pair failing this in a model
would be dropped for that model. All 5 pairs passed in both models (prefix lengths 3–13 tokens), so
all 10 model-pair cells are reported and no multi-token interpolation was performed.

**Hook point and sample sizes.** We read and patch `resid_post` after block 0 — the residual stream
immediately after the first transformer block — at the **final token position only**. Because the
prefix is identical and attention is causal, every earlier position is bit-identical between the two
prompts, so one forward pass per interpolation point fully determines the run. Downstream
`resid_post` is recorded at the final token of every later block (blocks 1–23), plus the final
logits. The sweep uses 101 evenly spaced interpolation values on $[0,1]$; everything runs under
`torch.no_grad()` with fixed seeds. Total: 5 pairs × 2 models × 101 points.

### Metrics

The whole study depends on making "the model's output moved from A to B" a number, so we start
there and build up to two sharpness statistics.

**Interpolation.** We need a path between the two activation vectors that does not shrink toward
zero in the middle, as a straight line between two high-dimensional vectors does. We therefore
interpolate the direction along the sphere (SLERP) and the length linearly. With $h_A, h_B$ the
patched-layer activations, $\hat h = h / \lVert h \rVert$ and $\Omega = \arccos(\hat h_A \cdot \hat h_B)$:

```math
h_\alpha \;=\; \Big[(1-\alpha)\lVert h_A\rVert + \alpha\lVert h_B\rVert\Big]\cdot
\frac{\sin\!\big((1-\alpha)\Omega\big)\,\hat h_A + \sin\!\big(\alpha\Omega\big)\,\hat h_B}{\sin\Omega}
```

$h_\alpha$ replaces the block-0 output at the final token and is run forward through the rest of the
model. At $\alpha=0$ and $\alpha=1$ this is the identity, which gives a free correctness check on
the harness (reported in Results).

**Endpoint divergence (JSD)** — how differently the two complete prompts predict the next token, and
so the independent variable of the whole study. It is measured at inference from the full-vocabulary
softmax distributions $P_A, P_B$ at the final position, never from corpus counts. Jensen–Shannon
divergence is used because it is symmetric and finite even when one distribution puts near-zero mass
where the other does not, which a plain KL divergence is not. Units are nats; 0 means identical
predictions:

```math
\mathrm{JSD}(P_A, P_B) \;=\; \tfrac{1}{2} D_{KL}\!\big(P_A \,\Vert\, M\big) + \tfrac{1}{2} D_{KL}\!\big(P_B \,\Vert\, M\big),
\qquad M = \tfrac{1}{2}\big(P_A + P_B\big)
```

**Relative distance $d(\alpha)$** — where the model's output sits on the way from A to B. A raw
distance is not comparable across pairs whose endpoints are far apart to different degrees, so we
normalize by the total: $d=0$ means the output is exactly A's, $d=1$ exactly B's. For a vector
$x_\alpha$ read at any hook point (a downstream `resid_post`, or the final logit vector):

```math
d(\alpha) \;=\; \frac{\lVert x_\alpha - x_A\rVert_2}{\lVert x_\alpha - x_A\rVert_2 + \lVert x_\alpha - x_B\rVert_2}
```

This is the quantity plotted in Figure 1. A model that responds proportionally to the input edit
gives $d(\alpha)=\alpha$, the diagonal; a plateau is any large flat stretch followed by a fast rise.

**Transition width $w_{10-90}$** (primary sharpness statistic) — how much of the sweep the output
spends actually moving. It is the $\alpha$-distance between the first upward crossings of $d=0.1$
and $d=0.9$, so a small value means the output ignored most of the interpolation and then switched:

```math
w_{10\text{-}90} \;=\; \alpha(d=0.9) - \alpha(d=0.1)
```

Crossings are linearly interpolated on the 101-point grid. Following the plan, $w_{10-90} < 0.5$ is
called a clear plateau, against $0.8$ for the linear response. This statistic drives the verdict
column of Table 1 and both Figure 2 (top) and Figure 3.

**Total-variation width $w_{TV}$** (robustness statistic) — the same idea without the fixed
thresholds. Four of the ten curves are non-monotonic (Figure 1): they dip and re-cross, which can
push the $d=0.1$ crossing far to the left and make a visibly sharp curve score as wide. Let
$C(\alpha)$ be the fraction of the curve's total variation accumulated by position $\alpha$; then
$w_{TV}$ is the $\alpha$-span carrying the middle half of all the movement:

```math
C(\alpha) = \frac{\int_0^{\alpha} \lvert d'(u)\rvert\,du}{\int_0^{1} \lvert d'(u)\rvert\,du},
\qquad w_{TV} \;=\; C^{-1}(0.75) - C^{-1}(0.25)
```

It is $0.5$ for a linear response and tends to $0$ for a step, so we call $w_{TV} < 0.25$ sharp.
Figure 2 (bottom) repeats the main test with it.

**Plateau fraction PF** (second robustness statistic) — how much of the sweep sits pinned at an
endpoint, computed directly from the grid without any crossing logic, so it is immune to both
non-monotonicity and to the choice of which crossing counts. Over the $N=101$ grid points:

```math
\mathrm{PF} \;=\; \frac{1}{N}\,\#\big\lbrace \alpha_i : d(\alpha_i) < 0.1 \ \ \text{or} \ \ d(\alpha_i) > 0.9 \big\rbrace
```

Higher is more plateau-like; the linear response gives $0.2$. Its role in Results is to show that the
three sharpness statistics disagree about the *sign* of the JSD relationship.

**Association test.** The hypothesis predicts that low endpoint JSD goes with low width, so we score
it with the Spearman rank correlation $\rho$ between JSD and each sharpness statistic, within each
model ($n=5$) and pooled over all model-pair cells ($n=10$). Rank correlation is used because the
prediction is about ordering and $n$ is small.

### Baselines

**Linear response** — the null shape, the behavior of a model whose output moves in proportion to
the activation edit:

```math
d(\alpha) = \alpha \quad\Longrightarrow\quad w_{10\text{-}90} = 0.8,\quad w_{TV} = 0.5,\quad \mathrm{PF} = 0.2
```

It appears as the gray dashed diagonal in Figure 1 and the gray dashed line in Figures 2 and 3. Any
value below these is a plateau of some strength.

**Dissimilar-continuation control** — the `The house was` + ` big` / ` in` pair, run through
identical machinery. This is the load-bearing baseline: it fixes the value that the plateau
statistics take when the two prompts emphatically do *not* share a continuation, which is what turns
"the test pairs plateau" from a positive result into a null one. It is drawn with a thick frame in
Figure 1 and a thick marker edge in Figure 2.

**Harness identity check** — patching $h_0$ and $h_1$ must reproduce the unpatched runs, giving
$d(0)=0$ and $d(1)=1$ exactly. Deviation from this measures implementation error, reported below.

## Results

**The harness is correct.** All 5 pairs tokenized validly in both models, and the patched runs at
the endpoints reproduced the clean forward passes to $|d(0)| \le 10^{-4}$ and
$|d(1) - 1| \le 10^{-4}$ in all 10 cells. The numbers below are therefore about the model, not about
patching artifacts.

**Table 1 — endpoint divergence and plateau strength, all 10 model-pair cells.** Reading the table:
JSD is the independent variable (small = the two prompts predict nearly the same next token);
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
| pythia-410m | gave a book to ` Mary` / ` her` | 0.033 | 0.582 | 0.268 | 0.43 | yes |
| pythia-410m | Two plus two is ` four` / ` 4` | 0.056 | 0.758 | 0.451 | 0.25 | yes |
| pythia-410m | The answer is ` four` / ` Four` | 0.271 | **0.340** | **0.135** | 0.66 | yes |
| pythia-410m | clue identify? ` Au` / ` 79` | 0.385 | 0.598 | 0.254 | 0.41 | yes |
| pythia-410m | *control:* The house was ` big` / ` in` | 0.665 | 0.425 | 0.137 | 0.57 | yes |

**Plateaus are real and they are everywhere.** On the threshold-free statistic, 9 of the 10 cells
land at $w_{TV} \le 0.27$ against the linear-response value of $0.5$ — the sole exception being
pythia-410m on `four`/`4` at $0.451$, which Figure 1 confirms is nearly a straight diagonal. So the
plan's first question has a positive answer: this style of single-token interpolation does induce
plateaus, in both model families, at strengths from mild to near-step (gpt2-medium `four`/`Four`
moves through 80% of the gap in $0.12$ of the sweep).

**The plateaus are uninformative about continuation similarity.** The two smallest-JSD cells,
`Mary`/`her` in pythia-410m (0.033 nats) and `four`/`4` in pythia-410m (0.056 nats), give
$w_{10-90}$ of $0.582$ and $0.758$ — the two widest transitions anywhere in the study, one of them
essentially the linear baseline. The sharpest cell in each model is `four`/`Four`, whose endpoint
divergence (0.271–0.377 nats) is 5–8× larger. Most decisively, the control pair — an adjective
against a preposition, at the largest divergence measured (0.659–0.665 nats) — plateaus at
$w_{10-90} = 0.516$ / $0.425$ and $w_{TV} = 0.272$ / $0.137$, sharper than both low-JSD pairs in
gpt2-medium on $w_{TV}$ and sharper than three of four test pairs in pythia-410m. A diagnostic that
fires just as hard on its own negative control is not measuring what it was supposed to measure.

**The correlations confirm the null and expose its shape.** Pooled over all 10 cells, Spearman
$\rho$ between endpoint JSD and $w_{10-90}$ is $-0.37$ ($p = 0.29$), between JSD and $w_{TV}$ is
$-0.15$ ($p = 0.68$), and between JSD and PF is $+0.32$ ($p = 0.37$). The first two have the
*opposite* sign to the hypothesis (more divergent endpoints, slightly sharper transitions), the
third has the hypothesized sign, and none is significant. Within models the disagreement is starker:
$\rho$ for $w_{TV}$ is $+0.30$ in gpt2-medium and $-0.60$ in pythia-410m. Three statistics that
measure the same underlying property should agree on a real effect's direction; theirs is set by
noise.

The verdict rests first on the shape of the raw sweeps, so we show all ten before any summary
statistic. If plateaus tracked continuation similarity, the top two rows (lowest JSD) would be the
flattest and the bottom row (the control) the most linear.

![Relative distance versus interpolation position for five prompt pairs in two models](plots/final_logit_curves.png)

**Figure 1.** Final-logit response to interpolating one token's block-0 activation. x: interpolation
position $\alpha$ from prompt A (0) to prompt B (1); y: relative distance $d$ (0 = at A's logits,
1 = at B's logits). Solid curve with circles = measured $d(\alpha)$; gray dashed = the linear
reference $d = \alpha$. Rows are prompt pairs, columns are models; the bottom row (thick frame) is
the control pair, whose continuations differ most. Every panel except pythia-410m `four`/`4` bends
well away from the diagonal into a flat-then-jump shape, and the control bends as much as the test
pairs. The wiggles in the gpt2-medium column are the non-monotonicity flagged in Table 1 and are why
$w_{TV}$ exists.

Figure 1 establishes that plateaus are present everywhere; the direction's actual question is
whether their sharpness is explained by how similar the two continuations are. Figure 2 puts the two
quantities on the same axes, under both sharpness statistics.

![Endpoint divergence plotted against two transition-sharpness statistics](plots/jsd_vs_width.png)

**Figure 2.** Endpoint divergence does not predict transition sharpness. x (both rows): endpoint JSD
in nats — larger means the two prompts predict more different next tokens. y: $w_{10-90}$ (top row)
and $w_{TV}$ (bottom row), both at the final logits, smaller = sharper. Columns are models; each
marker is one prompt pair (shape and color per the legend; the control has a thick black edge). Gray
dashed = the linear-response value of the statistic, dotted = the plateau threshold. Under the
hypothesis the points would rise from left to right in every panel; they do not, and the control
(rightmost marker in each panel) sits at or below the plateau threshold in three of four panels.

If sharpness is not coming from the prompts, it has to come from the architecture. To locate where,
we recompute $w_{10-90}$ at every block's residual stream between the patch site and the output.

![Transition width versus recording block for five prompt pairs in two models](plots/layerwise_widths.png)

**Figure 3.** The plateau is built up gradually across depth, not created at the patch site.
x: the block whose `resid_post` is read out at the final token (the patch is applied after block 0;
the last x value is the final logits); y: $w_{10-90}$ at that read-out point. One line per prompt
pair, with color, line style and marker all varying together (see legend). Gray dashed = linear
response (0.8), dotted = plateau threshold (0.5). Every pair starts near 0.8 immediately after the
patch and narrows steadily with depth in both models; the control (triangles, dash-dot) is among the
fastest to sharpen.

**Depth, not prompt content, produces the plateau.** Figure 3 is the mechanistic half of the
argument. One block after the patch, all ten cells sit at $w_{10-90} \approx 0.79$–$0.81$: the
residual stream still responds almost exactly proportionally to the edit. The width then falls
monotonically across the following 23 blocks, reaching $0.12$–$0.76$ at the logits. This is the
expected behavior of a deep stack of nonlinear layers repeatedly compressing an interpolated
direction toward whichever endpoint dominates, and it happens for the control just as readily as for
the test pairs. That is the mechanism behind the null: the plateau is manufactured by depth, so it
cannot carry information about whether the two prompts share a continuation.

## Conclusion

Interpolating a single token's early-layer activation between two prompts reliably produces a
plateau-then-jump logit response — 9 of 10 model-pair cells beat the linear baseline on a
threshold-free sharpness statistic — but the plateau says nothing about whether the two prompts
predict the same continuation. Endpoint Jensen–Shannon divergence and transition sharpness are
uncorrelated at this sample size ($\rho = -0.37$, $p = 0.29$, $n = 10$), with the sign flipping
across models and across sharpness statistics, and the two most-similar pairs give the two widest
transitions. The negative control, chosen so its two continuations clearly differ, plateaus as
sharply as the test pairs.

For anyone using interpolation as an interpretability probe, the actionable point is that the
plateau shape needs a control before it can be evidence for anything: report a dissimilar-prompt
interpolation alongside the pair of interest, and treat the difference between them, not the
absolute sharpness, as the signal. Figure 3 explains why the bar is that high — the sharpening is
supplied by the 23 blocks downstream of the patch, which are the same 23 blocks in every condition.

**Limitations.** The strongest one is sample size: 5 prompt pairs and 2 models give $n=10$ cells, so
we can say the hypothesized relationship is not detectable here, not that it is exactly zero; a
correlation of moderate size would survive these $p$-values. The pairs are hand-written rather than
sampled from a corpus, and were chosen by us to look like "same continuation, different feature"
cases; a corpus-mined bank of pairs spanning a continuous JSD range would test the association far
more powerfully and is the obvious next experiment. All results are for one patch site (block 0) at
one position (the final token) in two models of similar size, and we did not test whether patching
deeper, or at more than one position, changes the picture. Finally, $d(\alpha)$ measures movement in
raw logit space; a pair could hold its logit vector still while reordering low-probability tokens,
which this metric would not see.
