# Activation-interpolation plateaus: what reproduces, what depth explains, and which heads cause the switch

> Final, presentable, current-best only (history is in CHANGELOG.md).

## Summary

A common move in interpretability is to take the internal activation vector a language model computes
for prompt A, the one it computes for prompt B, and walk continuously from one to the other, watching
how the model's output changes. When the output stays put for a while and then switches abruptly — a
**plateau** followed by a jump — it is tempting to read that as a discrete internal state flipping.
Matthew reports exactly this in GPT-2 Large for the prompt `The house was` completed with ` big`
versus ` in`, and reports a smooth response for ` big` versus ` large`. This report asks three things:
does that contrast reproduce, what controls it, and does it support the hypothesis it is being used to
motivate?

**It reproduces, in his model.** Interpolating the final token's block-0 activation in GPT-2 Large
gives a near-perfect step for `big`/`in` — the output moves through 80% of the gap in 4% of the sweep
— and a near-linear response for `big`/`large`. **It does not reproduce in GPT-2 Medium**, where
`big`/`in` gives a transition width of 0.516, above the plateau criterion, and in GPT-2 Small it is
wider still. A study that swaps GPT-2 Large for GPT-2 Medium is not testing the same phenomenon.

**What controls it is relative depth.** Moving the patch site up the stack removes the plateau, and the
variable that governs how much is removed is the *fraction* of the network below the patch, not the
number of blocks. GPT-2 Large patched at block 12 and GPT-2 Medium patched at block 0 both have 23
blocks below the patch and differ 3.2-fold in median transition width; plotted against the fraction,
three GPT-2 models of 12, 24 and 36 blocks nearly superimpose. Depth is necessary but not sufficient:
`big`/`large` has 35 blocks below the patch in GPT-2 Large and still does not sharpen, so whether a
given interpolation sharpens depends on the pair as well as on the depth available to process it.

**The base rate is the practical caution.** In a bank of 200 corpus-mined pairs per model, 83.5% of
GPT-2 Large pairs and 73.0% of GPT-2 Medium pairs plateau under the predefined criterion. A plateau
observed for one chosen pair is therefore weak evidence by itself; what carries information is where
that pair sits relative to the distribution.

**The motivating hypothesis splits in two, and the halves come out differently.** The claim is that
*holding output divergence low, different circuits or features may occupy different plateaus*. Read as
a prediction about **intermediate** resting points — the curve pausing partway between A and B — it
fails a well-powered test: across 1120 mined pairs whose next-token predictions nearly agree, no
measure of circuit or feature difference predicts intermediate plateaus in any of three models
(Spearman $\rho$ between $-0.11$ and $+0.12$, none surviving correction, at sample sizes where
$|\rho| \ge 0.10$ would have been detected). Read as a prediction that **each prompt sits on its own
plateau** and the interpolation snaps between them, it holds: at matched output predictions, pairs that
engage more different attention heads, MLP neurons or sparse-autoencoder features have measurably
sharper transitions, in every model and on every instrument we built ($\rho$ from $-0.11$ to $-0.36$,
14 of 14 negative). Measuring the machinery rather than the geometry is what surfaces this — the
residual-stream proxy used before gives $\rho = -0.13$ in GPT-2 Medium where the head-level measurement
gives $-0.36$.

**And in GPT-2 Large that second reading is causal.** Mean-ablating the 3% of attention heads that
write most differently for the two prompts widens the median transition by 81% ($w_{TV}$ $0.198 \to
0.358$); at 10% of heads the model stops switching altogether and responds proportionally
($0.484$ against the linear response's $0.5$). Deleting an equal-sized set of heads matched on how much
they write, but chosen to write *similarly* for the two prompts, changes nothing ($0.198 \to 0.200$).
So the sharp switch in the model where this phenomenon was reported is produced by a small set of
heads, identifiable in advance from the two clean forward passes.

**Those heads are shared across pairs, and they act from above the patch.** A single fixed set of 22
GPT-2 Large heads, ranked on one half of the prefixes and ablated on the other half, widens the median
switch to $0.485$ — more than the per-pair sets manage, so tailoring the selection to each pair was
adding noise rather than precision. But the set does its work through five heads in **block 0**, which
sit above the patch site and therefore cannot process the interpolated vector; they can only decide what
the two interpolated vectors contain. Rebuild the same fixed set from blocks 1–35 and the effect drops
by 94%, to a small but solid $+0.012$. Interpolation sharpness is thus about the endpoints as much as
about the depth that processes them — which matters for anyone reading a plateau as evidence about
downstream mechanism.

## Methods

### Data & Model

**Models.** Five final pretrained checkpoints, all frozen and in evaluation mode, float32, no sampling:
`gpt2-large` (774M parameters, 36 blocks, $d_{model}=1280$), `gpt2-medium` (355M, 24 blocks,
$d_{model}=1024$), `gpt2` (124M, 12 blocks, $d_{model}=768$), `facebook/opt-350m` (331M, 24 blocks,
$d_{model}=1024$) and `EleutherAI/pythia-410m-deduped` at `revision="step143000"` (24 blocks,
$d_{model}=1024$). GPT-2 Large is the model in which the phenomenon was originally reported, so it is
the primary model here. The three GPT-2 checkpoints share a tokenizer, architecture and pretraining
corpus and differ in depth, which is what makes the depth-scaling comparison of Experiment 5
interpretable; their residual width rises with depth, a confound we state rather than remove. OPT-350m
and Pythia-410m are other families at the same depth as GPT-2 Medium: OPT's vocabulary is exactly
GPT-2's 50257 token strings plus 8 specials and it segments our prompts identically, while Pythia uses
the GPT-NeoX vocabulary.

**Hand-picked pairs (Experiment 1).** Six prompt pairs, each a shared prefix plus one differing final
token. Two are Matthew's own, and their roles are asymmetric:

1. `The house was` + ` big` / ` in` — his **plateau case**, the pair he reports as showing a sharp
   transition. It is a positive example, not a negative control.
2. `The house was` + ` big` / ` large` — his **smooth case**, the comparison pair he reports as *not*
   plateauing.

The other four are ours, chosen so the two versions of the sentence plausibly continue the same way
while differing in some internal property (identity vs. pronoun, word-form vs. numeral, lower vs. upper
case, chemical symbol vs. atomic number):

3. `Mary and John went to the store. John gave a book to` + ` Mary` / ` her`
4. `Two plus two is` + ` four` / ` 4`
5. `The answer is` + ` four` / ` Four`
6. `Which chemical element does this clue identify?` + ` Au` / ` 79`

**Mined pair bank (Experiments 2, 3, 5).** Six hand-written pairs cannot measure a base rate or an
association, so we mine a bank automatically. We take the first 40 paragraphs of at least 400
characters from the WikiText-103 validation split (natural English, not written by us), truncate each
to a prefix of $L$ tokens with $L$ drawn uniformly from $[10, 40]$, and run the model on the prefix.
Final token A is always the model's **top-1** next token; final token B is the token at rank $r$, with
five values of $r$ drawn log-uniformly from $[1, 5000]$ per prefix. A rank-1 partner produces a
near-tie between two plausible continuations, a rank-5000 partner an implausible one, so endpoint
divergence spans its whole range by construction. That gives **200 pairs per model** from 40 prefixes
(the same prefixes and rank draws in every model; the tokens are each model's own). Building the inputs
as `prefix_ids + [token_id]` makes the "identical prefix, one differing single final token" condition
exact rather than something to check.

**Low-JSD pair bank (Experiment 6).** The hypothesis under test needs pairs whose next-token
predictions nearly agree, and those are rare in the general bank — filtering it left only a few dozen
pairs per model, too few to detect anything but a large effect. So we mine for them directly. We take
200 WikiText-103 prefixes built exactly as above; for each, token A is again the model's top-1 next
token, and the candidate partners are the tokens at ranks 1–20 plus ten more ranks drawn log-uniformly
from $[21, 2000]$. One batched forward pass gives every candidate prompt's next-token distribution, we
compute each candidate's endpoint divergence against prompt A, and we keep up to six candidates with
$\mathrm{JSD} < 0.1$, spread across the qualifying ranks so the bank is not made entirely of rank-2
partners. Screening before sweeping is what buys the power: nearly every swept pair is usable. The kept
pairs get the same block-0 sweep as everywhere else.

**Patch-depth intervention (Experiment 4).** Reading the interpolation out at an earlier block shows
where sharpness accumulates but cannot show that the downstream blocks *cause* it, since reading
earlier is not the same as computing less. We therefore re-run the entire mined bank — same pairs, same
tokens, same 101 interpolation points — with the patch applied after block 12 and after block 20
instead of block 0, so the interpolated vector is processed by 11 or 3 remaining blocks rather than 23.
The endpoint activations are then read at that same block.

**Depth-scaling comparison (Experiment 5).** Experiment 4 moves the patch inside three models that all
have 24 blocks, so "11 blocks below the patch" and "just under half the stack below the patch" name the
same runs and cannot be told apart. To separate them we mine a fresh 200-pair bank for `gpt2` and for
`gpt2-large` and sweep each at sites picked so the three GPT-2 models line up under one reading or the
other: blocks 0/6/8/10 for the 12-block model and blocks 0/12/18/24/31 for the 36-block model, against
blocks 0/12/20 for the 24-block model.

**Head-ablation intervention (Experiment 7).** Experiment 6 is correlational, so it cannot say whether
the differentially-engaged heads *make* the switch sharp. Experiment 7 removes them and looks. For each
low-JSD pair in all three GPT-2 models we mean-ablate a set of attention heads at the final token
only, then re-run the entire sweep — both endpoints and all 101 interpolation points — with the
ablation active, so $d(0)=0$ and $d(1)=1$ still hold and $w_{TV}$ describes the ablated model's own
switch. Ablating at the final token only keeps every earlier position, and therefore the whole prefix
computation, untouched. The replacement value for an ablated head is its mean output over the final
tokens of 100 bank prompts, which keeps the model on-distribution in a way zeroing does not. Three
doses are run — 3%, 6% and 10% of all heads (4/9/14 of GPT-2 Small's 144; 12/23/38 of GPT-2 Medium's
384; 22/43/72 of GPT-2 Large's 720) — because a single dose cannot distinguish "no effect" from "too
small an intervention". The doses were fixed before any of them was run.

**Localisation and the fixed cross-pair set (Experiment 8).** Experiment 7 chooses a fresh head set for
every pair, which leaves open whether there is one circuit to name. Experiment 8 splits each low-JSD
bank into two folds by the parity of the prefix index, ranks all heads by how often they enter a pair's
top-$k$ differential set within one fold, and ablates the resulting single **fixed set** — the same $k$
heads for every pair, at the 3% dose — on the pairs of the *other* fold. Both folds are run and pooled,
so every pair is scored under a set it had no part in choosing. Because the patch replaces the final
token's residual stream after block 0, a block-0 head can only affect the sweep through the two
endpoint activations that are interpolated, never through the computation applied to them; we therefore
repeat the whole construction in GPT-2 Large with block 0 struck from the ranking, so every ablated
head is downstream of the patch. GPT-2 Small is included in the recurrence counts but not the fixed-set
ablation, whose $k=4$ is too small for a ranking to mean much.

**Validity check.** For the hand-written pairs we require, per model, that the two prompts tokenize to
an identical prefix and exactly one differing single final token. All 6 pairs passed in all five models
(prefix lengths 3–14 tokens), so all 30 model-pair cells are reported and no multi-token interpolation
was performed.

**Hook point and sample sizes.** The default patch site is `resid_post` after block 0 — the residual
stream immediately after the first transformer block — at the **final token position only**. Because
the prefix is identical and attention is causal, every earlier position is bit-identical between the
two prompts, so one forward pass per interpolation point fully determines the run. For the hand-picked
pairs, downstream `resid_post` is also recorded at the final token of every later block plus the final
logits; for the mined bank only the final logits are recorded. Every sweep uses 101 evenly spaced
interpolation values on $[0,1]$, under `torch.no_grad()` with fixed seeds. That comes to 4750 unablated
sweeps: 30 hand-picked model-pair cells, 1000 mined-bank sweeps at block 0 across five models, 1200
more at blocks 12 and 20 in the three 24-block models, 1400 at the extra Experiment 5 sites, and 1120
on the low-JSD banks of Experiment 6 (365 pairs from 102 prefixes in GPT-2 Small, 399 from 119 in GPT-2
Medium, 356 from 113 in GPT-2 Large). Experiment 7 re-runs all 1120 low-JSD sweeps under six ablation
conditions each, for 6720 more, and Experiment 8 adds 1111 held-out fixed-set sweeps — 12581 in
total.

### Metrics

The study depends on making "the model's output moved from A to B" a number, so we start there and
build up to the sharpness statistics, then to the two quantities Experiment 6 needs.

**Interpolation.** We need a path between the two activation vectors that does not shrink toward zero
in the middle, as a straight line between two high-dimensional vectors does. We interpolate the
direction along the sphere (SLERP) and the length linearly. With $h_A, h_B$ the patched-layer
activations, $\hat h = h / \lVert h \rVert$ and $\Omega = \arccos(\hat h_A \cdot \hat h_B)$:

```math
h_\alpha \;=\; \Big[(1-\alpha)\lVert h_A\rVert + \alpha\lVert h_B\rVert\Big]\cdot
\frac{\sin\!\big((1-\alpha)\Omega\big)\,\hat h_A + \sin\!\big(\alpha\Omega\big)\,\hat h_B}{\sin\Omega}
```

$h_\alpha$ replaces the patched block's output at the final token and is run forward through the rest
of the model. At $\alpha=0$ and $\alpha=1$ this is the identity, which gives a free correctness check
on the harness (reported in Results).

**Relative distance $d(\alpha)$** — where the model's output sits on the way from A to B. A raw
distance is not comparable across pairs whose endpoints are far apart to different degrees, so we
normalize by the total: $d=0$ means the output is exactly A's, $d=1$ exactly B's. For a vector
$x_\alpha$ read at any hook point:

```math
d(\alpha) \;=\; \frac{\lVert x_\alpha - x_A\rVert_2}{\lVert x_\alpha - x_A\rVert_2 + \lVert x_\alpha - x_B\rVert_2}
```

This is the quantity plotted in Figure 1. A model that responds proportionally to the input edit gives
$d(\alpha)=\alpha$, the diagonal; a plateau is a large flat stretch followed by a fast rise.

**Transition width $w_{10-90}$** (primary sharpness statistic, fixed by the plan in advance) — how much
of the sweep the output spends actually moving. It is the $\alpha$-distance between the first upward
crossings of $d=0.1$ and $d=0.9$, so a small value means the output ignored most of the interpolation
and then switched:

```math
w_{10\text{-}90} \;=\; \alpha(d=0.9) - \alpha(d=0.1)
```

Crossings are linearly interpolated on the 101-point grid. The predefined criterion is
$w_{10-90} < 0.5$ for a clear plateau, against $0.8$ for the linear response. All prevalence counts in
Results are quoted under this criterion first.

**Total-variation width $w_{TV}$** (threshold-free sharpness statistic) — the same idea without fixed
crossing levels. Non-monotonic curves are common: they dip and re-cross, which pushes the $d=0.1$
crossing far to the left and makes a visibly sharp curve score as wide. Let $C(\alpha)$ be the fraction
of the curve's total variation accumulated by position $\alpha$; then $w_{TV}$ is the $\alpha$-span
carrying the middle half of all the movement:

```math
C(\alpha) = \frac{\int_0^{\alpha} \lvert d'(u)\rvert\,du}{\int_0^{1} \lvert d'(u)\rvert\,du},
\qquad w_{TV} \;=\; C^{-1}(0.75) - C^{-1}(0.25)
```

It is $0.5$ for a linear response and tends to $0$ for a step; $w_{TV} < 0.25$ is called sharp. It is
the statistic used for the model comparisons, where non-monotonicity would otherwise dominate.

**Plateau fraction PF** (robustness statistic) — how much of the sweep sits pinned at an endpoint,
computed directly from the grid without crossing logic. Over the $N=101$ grid points:

```math
\mathrm{PF} \;=\; \frac{1}{N}\,\#\big\lbrace \alpha_i : d(\alpha_i) < 0.1 \ \ \text{or} \ \ d(\alpha_i) > 0.9 \big\rbrace
```

Higher is more plateau-like; the linear response gives $0.2$.

**Endpoint divergence (JSD)** — how differently the two complete prompts predict the next token. In
Experiments 3–5 it is a descriptive variable; in Experiment 6 it is a **control held low**. It is
measured at inference from the full-vocabulary softmax distributions $P_A, P_B$ at the final position.
Jensen–Shannon divergence is symmetric and stays finite when one distribution puts near-zero mass where
the other does not, where a plain KL divergence diverges. Units are nats; 0 means identical
predictions:

```math
\mathrm{JSD}(P_A, P_B) \;=\; \tfrac{1}{2} D_{KL}\!\big(P_A \,\Vert\, M\big) + \tfrac{1}{2} D_{KL}\!\big(P_B \,\Vert\, M\big),
\qquad M = \tfrac{1}{2}\big(P_A + P_B\big)
```

JSD is bounded above by $\ln 2 \approx 0.693$ nats, attained when the two distributions have disjoint
support. Pairs at that ceiling can no longer be ordered, so Experiment 3 reports the association on the
**unsaturated** subset with $\mathrm{JSD} < 0.65$.

**Relative depth $f$** — the quantity Experiment 5 exists to test. A patch site can be described by how
many blocks sit below it or by what fraction of the stack sits below it, and the two coincide whenever
every model has the same depth. With $L$ the 0-indexed block whose output is patched and $N$ the number
of blocks:

```math
f \;=\; \frac{N-1-L}{N-1} \;\in\; (0, 1]
```

$f = 1$ is a patch after block 0, with the whole stack below it. To decide which description governs
plateau strength, we pick levels at which the three GPT-2 models are matched under one description or
the other and measure how much they still disagree, using the across-model spread of the median
$w_{TV}$, written $\tilde w_{TV}(m, \ell)$ for model $m$ at matched level $\ell$:

```math
S(\ell) \;=\; \max_{m} \tilde w_{TV}(m, \ell) \;-\; \min_{m} \tilde w_{TV}(m, \ell)
```

A description that captures what drives the plateau makes $S$ small; the wrong description leaves the
models spread out.

**Measuring "different circuits or features".** This is the independent variable the hypothesis needs,
and the hard part of testing it. Once output divergence is held low, the question is whether the two
prompts nevertheless engage different machinery inside the model. Distance between residual streams is
the easy answer but the wrong construct — two prompts can route through different features and still
land at a similar vector — so we measure at the level of the objects interpretability actually calls
features and circuits, and report every instrument rather than picking one after the fact. Each measure
below is 0 when the two prompts use identical machinery and approaches 1 when they share none of it.
Experiment 6 correlates each of them against the plateau statistics.

**SFD, SAE feature-set disjointness** — the most direct instrument, and the one the operator's review
asked for. A *sparse autoencoder* (SAE) is a dictionary trained on a model's activations to rewrite
each activation vector as a sparse non-negative sum of learned directions; each direction is a
candidate interpretable feature, and typically only a few dozen of the 24576 are active at any token.
We use the publicly released SAEs of Bloom (2024), `jbloom/GPT2-Small-SAEs-Reformatted`, which cover
every residual-stream location in GPT-2 Small, so this measure exists for that model only. With
$F^l(x)$ the vector of feature activations at hook point $l$ for the final token of prompt $x$, and
$A(x) = \lbrace (l,i) : F_i^l(x) > 0 \rbrace$ the set of features that fire anywhere in the stack:

```math
\mathrm{SFD} \;=\; 1 - \frac{\lvert A(x_A) \cap A(x_B)\rvert}{\lvert A(x_A) \cup A(x_B)\rvert}
```

That is a Jaccard distance between two feature sets: it asks what fraction of the features involved in
either prompt are not shared. Hook-point correspondence: the SAE trained on `blocks.`$l+1$`.hook_resid_pre`
takes the residual stream after block $l$, and the released SAEs were trained on activations centred
over $d_{model}$, so we centre ours the same way before encoding. Results reports the reconstruction
quality and sparsity we measure on this bank's own prompts, as a check that the correspondence is right.

**SFC, SAE feature-profile distance** — the threshold-free version of the same measurement, in case
binarising "active vs not" throws away the signal. It compares the concatenated activation vectors
$F(x) = [F^1(x), \dots, F^N(x)]$ by angle:

```math
\mathrm{SFC} \;=\; 1 - \frac{F(x_A) \cdot F(x_B)}{\lVert F(x_A)\rVert\,\lVert F(x_B)\rVert}
```

**HCD, head-contribution distance** — a circuit-level instrument that exists in every model, unlike the
SAEs. Each attention head writes a vector into the final token's residual stream: with $z_h$ the head's
output before the output projection and $W_O^h$ its slice of that projection, head $h$ contributes
$c_h = z_h W_O^h$. HCD asks how differently the heads write for the two prompts, averaging the angle
between their contributions and weighting each head by how much it writes, so idle heads do not
dominate:

```math
\mathrm{HCD} \;=\; \frac{\sum_h \big(\lVert c_h^A\rVert + \lVert c_h^B\rVert\big)\Big(1 - \frac{c_h^A \cdot c_h^B}{\lVert c_h^A\rVert \lVert c_h^B\rVert}\Big)}{\sum_h \big(\lVert c_h^A\rVert + \lVert c_h^B\rVert\big)}
```

**HSD, head-set disjointness** — the set-based reading of the same idea, closest to how circuit work
usually talks ("which heads are doing the work"). With $T_k(x)$ the $k$ heads that write most to the
final token of prompt $x$, taken over all blocks at once, $k$ = 10% of all heads:

```math
\mathrm{HSD} \;=\; 1 - \frac{\lvert T_k(x_A) \cap T_k(x_B)\rvert}{k}
```

**NSD, neuron-set disjointness** — the MLP counterpart, since attention is only half the network. Let
$T_{64}^l(x)$ be the 64 largest post-GELU activations in block $l$'s MLP at the final token and
$U(x) = \bigcup_l T_{64}^l(x)$; NSD is again a Jaccard distance, $1 - \lvert U(x_A) \cap U(x_B)\rvert /
\lvert U(x_A) \cup U(x_B)\rvert$. Individual neurons are polysemantic, which is what the SAE fixes, so
NSD is a coarse instrument reported alongside the sharper ones.

**IRD, internal representational distance** — the residual-stream proxy used before the feature-level
instruments existed, kept as a comparison so the report can say what the upgrade bought:

```math
\mathrm{IRD} \;=\; \frac{1}{N}\sum_{l=1}^{N}\Big(1 - \frac{h_A^l \cdot h_B^l}{\lVert h_A^l\rVert\,\lVert h_B^l\rVert}\Big)
```

An instrument that does not vary across pairs cannot detect anything, so Results reports each measure's
interquartile range next to its correlations.

**Intermediate-plateau width IPW** — the dependent variable of Experiment 6. The hypothesis is about
different features occupying different *plateaus*, which predicts a resting point partway between A and
B, not a narrow A-to-B transition. So we score the longest stretch of the sweep over which the output
holds still at a level that is neither A's nor B's. With $\mathcal{W}$ the set of index intervals
$[i,j]$ on the $\alpha$ grid whose $d$ values span at most $0.10$ and whose mean level lies in
$[0.15, 0.85]$:

```math
\mathrm{IPW} \;=\; \max_{[i,j]\,\in\,\mathcal{W}} \big(\alpha_j - \alpha_i\big)
```

A linear response $d(\alpha)=\alpha$ gives $\mathrm{IPW} = 0.10$ exactly, because a window spanning
$0.10$ in $d$ spans $0.10$ in $\alpha$. We therefore call $\mathrm{IPW} > 0.20$ — twice the linear
baseline — an intermediate plateau. IPW is the dependent variable of Experiment 6's primary test.

Experiment 6 also reads the hypothesis a second way. "Different circuits occupy different plateaus" can
mean a resting point *between* A and B, which is IPW, or it can mean that each prompt sits on its own
plateau and the interpolation snaps between them — which is a narrow transition, $w_{TV}$, measured
here with output divergence held low so it is not the Table 3 regularity in disguise. Both readings are
tested against every instrument.

**Differential engagement $\delta_h$, and the control it defines** — which heads Experiment 7 removes,
and what it removes instead. HCD asks how differently the heads write *on average*; to intervene we
need the same question per head, so $\delta_h$ is simply HCD's own numerator term for head $h$, large
when the head writes a lot and writes in different directions for the two prompts. Writing
$m_h = \lVert c_h^A\rVert + \lVert c_h^B\rVert$ for how much head $h$ writes in total:

```math
\delta_h \;=\; m_h\Big(1 - \frac{c_h^A \cdot c_h^B}{\lVert c_h^A\rVert\,\lVert c_h^B\rVert}\Big)
```

The **differential** condition ablates the $k$ heads with the largest $\delta_h$. Removing those heads
also removes a chunk of ordinary attention output, and that alone could change the curve, so the
**matched-control** condition removes an equal-sized set chosen for the same $m_h$ and the smallest
$\delta_h$: for each differential head we take the 24 heads whose $m_h$ is closest to it and keep the
one that writes most similarly for the two prompts. The two conditions therefore delete about the same
amount of head output and differ in whether that output was prompt-discriminating. Results reports the
achieved ratio of removed $\sum_h m_h$ as a check on the matching, and the drop in HCD as the
manipulation check: an ablated head writes the identical mean vector for both prompts, so it can no
longer contribute any circuit difference.

**Head-set overlap $J$** — how Experiment 8 asks whether the same heads keep being selected. Counting
how often a single head is chosen says little on its own, because with $k$ heads drawn from $H$ a head
is chosen $k/H$ of the time by luck; what we want is whether two *pairs* pick the same heads. For the
top-$k$ differential sets $S_1$ and $S_2$ of two prompt pairs:

```math
J(S_1, S_2) \;=\; \frac{\lvert S_1 \cap S_2\rvert}{\lvert S_1 \cup S_2\rvert}
```

$J = 1$ means the two pairs chose identical heads and $J = 0$ that they share none. We report the mean
of $J$ over random pairs of prompt pairs drawn from *different* WikiText prefixes, since pairs built on
the same prefix share most of their input and would inflate the number. Two references make it
readable: sets drawn uniformly at random (the null) and the top-$k$ heads ranked by write magnitude
$m_h$ instead of $\delta_h$, a set that recurs almost by definition because the same heads are always
the loudest. High $J$ against the random null with low $J$ against the magnitude set is the signature of
a recurring core with a pair-specific tail.

**Recovery fraction** — what Experiment 8's fixed set is worth, in units of the per-pair effect it is
trying to replace. With $\tilde w$ a median over the held-out pairs:

```math
\text{recovery} \;=\; \frac{\tilde w(\text{fixed set}) - \tilde w(\text{no ablation})}
                            {\tilde w(\text{per-pair set}) - \tilde w(\text{no ablation})}
```

A recovery of 1 means one head set chosen without seeing the pair does exactly what a set tailored to
that pair does, so nothing is lost by naming a single circuit; near 0 means the effect is genuinely
pair-specific. Values above 1 are possible and are informative — they say the per-pair selection was
adding noise rather than precision. The same ratio, computed for the fixed set with block-0 heads
struck out, is how we split the effect into what acts below the patch and what acts on the interpolated
vector itself.

**Association tests.** All correlations are Spearman rank correlations, because the predictions are
about ordering and the relationships need not be linear. Pairs that share a prefix are not independent,
so confidence intervals come from a **cluster bootstrap**: resample the prefixes with replacement, take
all pairs of the drawn prefixes, and report the 2.5th and 97.5th percentiles over 2000 resamples. Two
further guards apply in Experiment 6. Residual divergence inside the low-JSD band, and the block-0
angle $\Omega$ between the two patched activations, could each drive a correlation on their own, so
every test is repeated as a **partial Spearman correlation**: rank all three variables, regress the
ranks of the independent and dependent variable on the ranks of $\mathrm{JSD}$ and $\Omega$, and
correlate the residuals. And because six instruments are tested against two statistics in three models,
we name the primary test in advance — the most direct set-of-features instrument available in that
model (SFD where SAEs exist, its head-level analogue HSD otherwise) against IPW — and Holm-correct
across the three models' primary
tests. Everything else is exploratory and reported with uncorrected $p$-values.

### Baselines

**Linear response** — the null shape, the behavior of a model whose output moves in proportion to the
activation edit:

```math
d(\alpha) = \alpha \quad\Longrightarrow\quad w_{10\text{-}90} = 0.8,\quad w_{TV} = 0.5,\quad \mathrm{PF} = 0.2,\quad \mathrm{IPW} = 0.10
```

It appears as the gray dashed diagonal in Figure 1 and a gray dashed line in Figures 2–10.

**Matthew's smooth case as the negative example** — `The house was` + ` big` / ` large`, run through
identical machinery. This is the pair reported as *not* plateauing, so it is the reference for what a
non-plateau looks like in this setup. It is drawn as the bottom row of Figure 1.

**Mined bank as a base rate** — the 200 corpus-derived pairs per model give the distribution of plateau
sharpness for pairs picked with no regard to continuation similarity. A hand-picked pair is only
informative insofar as it sits away from this distribution.

**Smallest detectable correlation** — the reference against which a null claim is judged. For a
two-sided test at $\alpha = 0.05$ using the Fisher $z$ transform:

```math
\rho_{\min}(n) \;=\; \tanh\!\Big(\frac{1.96}{\sqrt{n-3}}\Big)
```

This is $0.14$ at $n=200$ (the mined bank) and $0.10$ at $n \approx 370$ (Experiment 6's low-JSD
banks), so Experiment 6's null covers everything but small associations. Filtering the general bank down
to low divergence, instead of mining for it, would have left roughly 35 pairs per model and
$\rho_{\min} = 0.32$ — three times coarser, which is why the dedicated bank is worth its cost.

**Engagement-matched head ablation** — the baseline the causal claim of Experiment 7 rests on. Deleting
the differential heads must be compared against deleting *something*, or any change is just the cost of
losing attention output. The control condition removes the same number of heads with the same total
write magnitude $\sum_h m_h$ (achieved ratio $1.01$–$1.12$), chosen for the smallest $\delta_h$, so it
holds everything fixed except whether the removed heads discriminated between the two prompts. Its
effect on $w_{TV}$ is the null against which the differential condition is read.

**Harness identity check** — patching $h_0$ and $h_1$ must reproduce the unpatched runs, giving
$d(0)=0$ and $d(1)=1$ exactly. Deviation from this measures implementation error.

## Results

**The harness is correct.** All 6 hand-written pairs tokenized validly in all five models, and across
all 12581 sweeps the patched runs at the endpoints reproduced the clean forward passes to
$|d(0)| \le 3.6 \times 10^{-4}$ and $|d(1) - 1| \le 3.6 \times 10^{-4}$. The numbers below are about
the models, not about patching artifacts.

### The reported contrast reproduces in GPT-2 Large and fails in GPT-2 Medium

**Table 1 — endpoint divergence and plateau strength for the six hand-picked pairs in five models.**
Reading the table: $w_{10-90}$ and $w_{TV}$ are smaller when the transition is sharper; PF is larger
when more of the sweep is pinned at an endpoint. The "plateau?" column applies the predefined criterion
$w_{10-90} < 0.5$. Bold marks the sharpest test-pair cell per model. Models are ordered by depth.

| Model | Prompt pair (final tokens) | endpoint JSD (nats) | $w_{10-90}$ | $w_{TV}$ | PF | plateau? |
|---|---|---|---|---|---|---|
| gpt2-large | *M. plateau case:* The house was ` big` / ` in` | 0.663 | 0.044 | 0.012 | 0.95 | yes |
| gpt2-large | *M. smooth case:* The house was ` big` / ` large` | 0.053 | 0.592 | 0.292 | 0.42 | no |
| gpt2-large | The answer is ` four` / ` Four` | 0.283 | **0.133** | **0.020** | 0.86 | yes |
| gpt2-large | gave a book to ` Mary` / ` her` | 0.051 | 0.288 | 0.144 | 0.71 | yes |
| gpt2-large | clue identify? ` Au` / ` 79` | 0.312 | 0.448 | 0.139 | 0.55 | yes |
| gpt2-large | Two plus two is ` four` / ` 4` | 0.048 | 0.485 | 0.216 | 0.52 | yes |
| gpt2-medium | *M. plateau case:* The house was ` big` / ` in` | 0.659 | 0.516 | 0.272 | 0.50 | no |
| gpt2-medium | *M. smooth case:* The house was ` big` / ` large` | 0.042 | 0.719 | 0.398 | 0.29 | no |
| gpt2-medium | The answer is ` four` / ` Four` | 0.377 | **0.120** | **0.058** | 0.88 | yes |
| gpt2-medium | gave a book to ` Mary` / ` her` | 0.068 | 0.586 | 0.114 | 0.51 | no |
| gpt2-medium | clue identify? ` Au` / ` 79` | 0.342 | 0.358 | 0.117 | 0.64 | yes |
| gpt2-medium | Two plus two is ` four` / ` 4` | 0.138 | 0.454 | 0.232 | 0.55 | yes |
| gpt2-small | *M. plateau case:* The house was ` big` / ` in` | 0.658 | 0.691 | 0.254 | 0.32 | no |
| gpt2-small | *M. smooth case:* The house was ` big` / ` large` | 0.053 | 0.760 | 0.456 | 0.25 | no |
| gpt2-small | The answer is ` four` / ` Four` | 0.358 | **0.548** | **0.225** | 0.46 | no |
| gpt2-small | gave a book to ` Mary` / ` her` | 0.030 | 0.556 | 0.276 | 0.45 | no |
| gpt2-small | clue identify? ` Au` / ` 79` | 0.355 | 0.906 | 0.781 | 0.10 | no |
| gpt2-small | Two plus two is ` four` / ` 4` | 0.173 | 0.607 | 0.352 | 0.40 | no |
| opt-350m | *M. plateau case:* The house was ` big` / ` in` | 0.646 | 0.143 | 0.068 | 0.85 | yes |
| opt-350m | *M. smooth case:* The house was ` big` / ` large` | 0.042 | 0.831 | 0.598 | 0.18 | no |
| opt-350m | The answer is ` four` / ` Four` | 0.472 | 0.530 | 0.293 | 0.48 | no |
| opt-350m | gave a book to ` Mary` / ` her` | 0.038 | 0.734 | 0.356 | 0.28 | no |
| opt-350m | clue identify? ` Au` / ` 79` | 0.296 | **0.705** | **0.177** | 0.31 | no |
| opt-350m | Two plus two is ` four` / ` 4` | 0.027 | 0.907 | 0.680 | 0.11 | no |
| pythia-410m | *M. plateau case:* The house was ` big` / ` in` | 0.665 | 0.425 | 0.137 | 0.57 | yes |
| pythia-410m | *M. smooth case:* The house was ` big` / ` large` | 0.042 | 0.802 | 0.505 | 0.21 | no |
| pythia-410m | The answer is ` four` / ` Four` | 0.271 | **0.340** | **0.135** | 0.66 | yes |
| pythia-410m | gave a book to ` Mary` / ` her` | 0.033 | 0.582 | 0.268 | 0.43 | no |
| pythia-410m | clue identify? ` Au` / ` 79` | 0.385 | 0.598 | 0.254 | 0.41 | no |
| pythia-410m | Two plus two is ` four` / ` 4` | 0.056 | 0.758 | 0.451 | 0.25 | no |

In GPT-2 Large the reported contrast comes out clean and large. `big`/`in` moves through 80% of the gap
in 4% of the sweep, with 95% of the grid pinned at one endpoint or the other — about as close to a step
function as a 101-point sweep can show. `big`/`large` takes 59% of the sweep and tracks the diagonal.
The ratio between the two widths is 13-fold, so this is not a marginal difference that a different
statistic would erase: $w_{TV}$ gives $0.012$ against $0.292$ and PF gives $0.95$ against $0.42$.

The same `big`/`in` pair in GPT-2 Medium gives $w_{10-90} = 0.516$ — it fails the predefined criterion
— and in GPT-2 Small $0.691$. The pair is not intrinsically a plateau pair; it is a plateau pair in a
36-block model. Any conclusion drawn from its curve in a 24-block model is about a different
phenomenon, and the depth results below say why.

`big`/`large` is the useful fixed point in the other direction: it is the widest or second-widest
transition in **every** model, from $0.592$ in GPT-2 Large to $0.831$ in OPT-350m. GPT-2 Large has more
blocks below the patch than any other model here and still does not sharpen this pair.

Under the predefined criterion, **11 of the 30** model-pair cells plateau (14 of 30 under
$w_{TV} < 0.25$) — a minority, concentrated in the deeper models: 5/6 in GPT-2 Large, 3/6 in GPT-2
Medium, 2/6 in Pythia-410m, 1/6 in OPT-350m, 0/6 in GPT-2 Small. Statistics summarise; the shapes are
what the claim rests on, so Figure 1 shows all thirty sweeps.

![Relative distance versus interpolation position for six prompt pairs in five models](plots/final_logit_curves.png)

**Figure 1.** The `big`/`in` versus `big`/`large` contrast reproduces in GPT-2 Large and weakens as
model depth falls. x: interpolation position $\alpha$ from prompt A (0) to prompt B (1); y: relative
distance $d$ (0 = at A's logits, 1 = at B's logits). Solid curve with circles = measured $d(\alpha)$;
gray dashed = the linear reference $d=\alpha$. Columns are models, ordered by depth (GPT-2 Large, 36
blocks, leftmost). Rows are prompt pairs; the two bottom rows (thick frames) are Matthew's pair — row 5
his plateau case `big`/`in`, row 6 his smooth case `big`/`large`. Panel titles give that cell's
endpoint JSD and $w_{10-90}$, and flag non-monotonic curves. Row 5 is a near-vertical step in the GPT-2
Large panel and drifts toward the diagonal as depth falls; row 6 stays near the diagonal everywhere.

### For an arbitrary prompt pair, a plateau is the common case

A single pair's plateau means little without a base rate. The mined bank supplies one, and Figure 2
shows where the hand-picked pairs fall inside it.

![Distribution of transition sharpness over 200 mined prompt pairs per model, with the hand-picked pairs marked](plots/bank_prevalence.png)

**Figure 2.** Plateaus are common for arbitrary prompt pairs, increasingly so in deeper models. x:
$w_{TV}$ at the final logits (smaller = sharper); y: number of mined pairs per bin (gray hatched
histogram, $n=200$ per model). Gray dashed = linear response (0.5), dotted = sharpness threshold
(0.25). The markers on the strip above each histogram are the hand-picked pairs of Table 1 at their
$w_{TV}$ values (shape and color per the legend, Matthew's pairs with a thick black edge); their y
position carries no meaning.

**Table 2 — how often an arbitrary mined pair plateaus, under both criteria.**

| Model (blocks) | % with $w_{10-90}<0.5$ (predefined) | % with $w_{TV}<0.25$ | median $w_{10-90}$ | median $w_{TV}$ |
|---|---|---|---|---|
| gpt2-large (36) | 83.5% | 89.5% | 0.155 | 0.047 |
| gpt2-medium (24) | 73.0% | 82.0% | 0.241 | 0.080 |
| gpt2-small (12) | 60.5% | 74.0% | 0.417 | 0.153 |
| opt-350m (24) | 47.0% | 61.0% | 0.511 | 0.221 |
| pythia-410m (24) | 30.0% | 47.5% | 0.593 | 0.266 |

In GPT-2 Large, five random pairs in six plateau. That is the number to hold in mind when a single
chosen pair plateaus: the observation is consistent with the pair being special and equally consistent
with it being typical. What carries information is the pair's position in this distribution, or better,
a matched comparison pair run through the same machinery — which is exactly the role `big`/`large`
plays for `big`/`in`, and why reporting the two together is the right design.

The hand-picked set runs the other way: only 11/30 of those cells plateau, so our four test pairs are
*smoother* than random pairs. Hand-chosen prompts are not a neutral sample.

### Endpoint divergence tracks transition width — a descriptive regularity, not a test of the hypothesis

This section reports a relationship in the mined bank between how differently the two prompts predict
the next token and how sharp the transition is. It is worth knowing, because anyone sweeping pairs will
encounter it, and because its direction is the opposite of what one might guess. It is **not** a test
of the motivating hypothesis, which concerns matched-output pairs and intermediate plateaus; that test
is Experiment 6. Figure 3 plots the relationship in all five models.

![Endpoint divergence against two sharpness statistics for 200 mined pairs per model, with fits](plots/bank_regression.png)

**Figure 3.** Across mined pairs, more divergent endpoints go with sharper transitions in all five
models. x (all panels): endpoint JSD in nats; the dash-dot vertical line is the $\ln 2$ ceiling JSD
attains for disjoint predictions. y: $w_{10-90}$ (top) and $w_{TV}$ (bottom) at the final logits,
smaller = sharper. Columns are models. Light circles = the 200 mined pairs; solid line = OLS fit;
dashed line with squares = quintile means of JSD with $\pm1$ SE; stars = the hand-picked pairs (thick
black edge = Matthew's pairs). Gray dashed = linear response, dotted = plateau threshold.

**Table 3 — rank correlation between endpoint divergence and sharpness, unsaturated pairs only**
($\mathrm{JSD} < 0.65$, where JSD can still order pairs). Negative means more divergent endpoints give
sharper transitions.

| Model | $n$ | $\rho$ ($w_{TV}$) | $p$ | $\rho$ ($w_{10-90}$) | $p$ |
|---|---|---|---|---|---|
| gpt2-large | 137 | $-0.64$ | $6.1\times10^{-17}$ | $-0.61$ | $2.3\times10^{-15}$ |
| gpt2-medium | 142 | $-0.61$ | $1.5\times10^{-15}$ | $-0.54$ | $4.9\times10^{-12}$ |
| opt-350m | 129 | $-0.57$ | $1.3\times10^{-12}$ | $-0.59$ | $3.1\times10^{-13}$ |
| pythia-410m | 127 | $-0.45$ | $9.0\times10^{-8}$ | $-0.47$ | $2.3\times10^{-8}$ |
| gpt2-small | 147 | $-0.44$ | $2.7\times10^{-8}$ | $-0.36$ | $7.8\times10^{-6}$ |

The effect is consistent across five models and two statistics. A plausible mechanism is competition
between two well-separated output modes: when the endpoints predict disjoint token sets the winner
flips abruptly, whereas near-identical predictions differ only in small logit components that get
carried across smoothly. Note what this does and does not license. It says the sharp end of the width
distribution is populated by divergent pairs, so sharpness alone should not be read as evidence of
shared continuation. It says nothing about whether two prompts with matched predictions differ
internally.

Figure 4 asks whether the cross-model differences in Table 2 survive matching on divergence. They do,
so they are properties of the models rather than of how each bank happened to be distributed.

![Median transition width per endpoint-divergence bin for five models](plots/jsd_matched.png)

**Figure 4.** The cross-model sharpness ordering survives matching on endpoint divergence. x: endpoint
JSD bin in nats (the last bin is the $\ln 2$ ceiling), annotated with the number of mined pairs per
model in that bin; y: median $w_{TV}$ at the final logits over the pairs in the bin (smaller = sharper).
One line per model, each with its own color, line style and marker (see legend). Gray dashed = linear
response (0.5), dotted = sharp threshold (0.25). The two deepest GPT-2 models are sharpest in every
bin, and all lines fall from left to right.

### Depth below the patch is what allows a plateau to form

Sharpness has to originate somewhere. Figure 5 locates it by recomputing the width at every block
between the patch site and the output.

![Transition width versus recording block for six prompt pairs in five models](plots/layerwise_widths.png)

**Figure 5.** The plateau is built up gradually across depth, not created at the patch site. x: block
whose `resid_post` is read out (the patch is applied after block 0; the last x value is the final
logits); y: $w_{10-90}$ at that read-out point. One line per prompt pair (color, line style and marker
all vary together; see legend); panels are models. Gray dashed = linear response (0.8), dotted =
plateau threshold (0.5). Every pair starts near 0.8 just after the patch; some narrow steeply with
depth, and `big`/`large` stays near the top in every model.

Reading out earlier is not the same as computing less, so the causal version moves the patch instead;
Figure 6 gives the result.

![Median transition width and JSD-sharpness correlation against patch site for three models](plots/depth_effect.png)

**Figure 6.** Removing downstream blocks removes the plateau. x (both panels): the patch site — the
block whose `resid_post` at the final token is replaced by the interpolated vector — labelled with the
number of blocks remaining below it. Left y: median $w_{TV}$ at the final logits over the 200 mined
pairs (smaller = sharper), shaded band = interquartile range, gray dashed = linear response (0.5),
dotted = sharp threshold (0.25). Right y: Spearman $\rho$ between endpoint JSD and $w_{TV}$ over the
pairs below the $\ln 2$ ceiling, error bars = 95% cluster bootstrap over the 40 prefixes, gray dashed =
no association. gpt2-medium = circles, solid; pythia-410m = squares, dashed; opt-350m = triangles,
dotted.

**Table 4 — plateau strength at three patch sites**, same 200 mined pairs per model in every row.

| Model | patch site | blocks below | median $w_{TV}$ | % sharp | median $w_{10-90}$ |
|---|---|---|---|---|---|
| gpt2-medium | block 0 | 23 | 0.080 | 82.0% | 0.241 |
| gpt2-medium | block 12 | 11 | 0.250 | 50.5% | 0.556 |
| gpt2-medium | block 20 | 3 | 0.383 | 10.0% | 0.701 |
| opt-350m | block 0 | 23 | 0.221 | 61.0% | 0.511 |
| opt-350m | block 12 | 11 | 0.307 | 36.5% | 0.641 |
| opt-350m | block 20 | 3 | 0.420 | 1.0% | 0.741 |
| pythia-410m | block 0 | 23 | 0.266 | 47.5% | 0.593 |
| pythia-410m | block 12 | 11 | 0.419 | 2.5% | 0.749 |
| pythia-410m | block 20 | 3 | 0.509 | 0.0% | 0.808 |

Sharpness decays monotonically as depth is removed, in all three models, and the endpoint of that decay
is the linear baseline itself: with 3 blocks below the patch, Pythia-410m has median $w_{TV} = 0.509$
against $0.5$, and **not one of its 200 pairs is sharp**.

The right statement of this result is that downstream depth is *necessary* for the plateau, not that it
is sufficient. Table 1 supplies the counterexample: `big`/`large` has 35 blocks below the patch in
GPT-2 Large — more than any condition in Table 4 — and stays at $w_{10-90} = 0.592$. Depth sets how
sharp a transition *can* become; which pairs actually sharpen depends on the interpolation path.

### The depth that matters is relative depth

Table 4 is ambiguous about units in a way that matters for anyone applying it. All three models there
have 24 blocks, so "11 blocks below the patch" and "just under half the stack below the patch" pick out
the same runs, and the two readings give opposite advice about any other model. Experiment 5 separates
them inside the GPT-2 family, where tokenizer, architecture and corpus are fixed and only depth
changes.

**Table 5 — plateau strength at every patch site in three GPT-2 models of different depth.** Rows with
the same "blocks below" are matched under the absolute reading; rows with the same $f$ under the
relative reading.

| Model (blocks) | patch site | blocks below | $f$ | median $w_{TV}$ | % sharp | median $w_{10-90}$ |
|---|---|---|---|---|---|---|
| gpt2-small (12) | block 0 | 11 | 1.000 | 0.153 | 74.0% | 0.417 |
| gpt2-small (12) | block 6 | 5 | 0.455 | 0.289 | 35.5% | 0.632 |
| gpt2-small (12) | block 8 | 3 | 0.273 | 0.363 | 12.0% | 0.703 |
| gpt2-small (12) | block 10 | 1 | 0.091 | 0.456 | 3.5% | 0.768 |
| gpt2-medium (24) | block 0 | 23 | 1.000 | 0.080 | 82.0% | 0.241 |
| gpt2-medium (24) | block 12 | 11 | 0.478 | 0.250 | 50.5% | 0.556 |
| gpt2-medium (24) | block 20 | 3 | 0.130 | 0.383 | 10.0% | 0.701 |
| gpt2-large (36) | block 0 | 35 | 1.000 | 0.047 | 89.5% | 0.155 |
| gpt2-large (36) | block 12 | 23 | 0.657 | 0.255 | 47.0% | 0.570 |
| gpt2-large (36) | block 18 | 17 | 0.486 | 0.342 | 22.5% | 0.673 |
| gpt2-large (36) | block 24 | 11 | 0.314 | 0.444 | 1.5% | 0.754 |
| gpt2-large (36) | block 31 | 4 | 0.114 | 0.495 | 0.0% | 0.796 |

**One comparison settles it.** GPT-2 Large patched at block 12 and GPT-2 Medium patched at block 0 have
exactly 23 blocks below the patch, the same tokenizer and the same training corpus. Under the absolute
reading they should behave alike. They do not: median $w_{TV}$ is $0.255$ against $0.080$, a factor of
3.2, and 47.0% of pairs sharp against 82.0%. Those 23 blocks are almost the whole of the 24-block model
and two thirds of the 36-block one. At 11 blocks below the disagreement is worse and the ordering
inverts the absolute reading outright — the 12-block model is the sharpest (0.153) and the 36-block
model the flattest (0.444). Figure 7 puts the two readings of depth side by side.

![Median transition width against blocks below the patch and against fraction of the stack below the patch, for three GPT-2 models of different depth](plots/depth_scaling.png)

**Figure 7.** Relative depth, not absolute depth, organises the plateau. Both panels: y = median
$w_{TV}$ at the final logits over that model's 200 mined pairs (smaller = sharper); gray dashed =
linear response (0.5), dotted = sharp threshold (0.25). x, left panel: number of blocks below the patch
site; x, right panel: the same runs against the relative depth $f$ defined in Methods. gpt2-small =
circles, solid; gpt2-medium = squares, dashed; gpt2-large = triangles, dotted. The annotation in each
panel is the mean across-model spread $S$ at matched levels (Table 6); smaller means the three models
agree better under that reading. On the left the curves are separated and ordered by model size; on the
right they nearly superimpose.

**Table 6 — how far apart the three models stay once matched, under each reading.** The spread column
is $S(\ell)$, the range of median $w_{TV}$ across the three models at that level.

| Matched on | level | gpt2-small | gpt2-medium | gpt2-large | spread $S$ |
|---|---|---|---|---|---|
| blocks below | 11 blocks | 0.153 | 0.250 | 0.444 | **0.291** |
| blocks below | 3–4 blocks | 0.363 | 0.383 | 0.495 | **0.133** |
| relative depth $f$ | $f=1.00$ | 0.153 | 0.080 | 0.047 | **0.106** |
| relative depth $f$ | $f=0.46$–$0.49$ | 0.289 | 0.250 | 0.342 | **0.093** |
| relative depth $f$ | $f=0.09$–$0.13$ | 0.456 | 0.383 | 0.495 | **0.112** |

Averaged over levels, matching on relative depth halves the residual disagreement, $S = 0.212$ against
$0.104$. The leftover spread has a readable sign: at matched $f$ the deeper model is somewhat sharper
($0.153 \to 0.080 \to 0.047$ at $f=1$), so absolute depth contributes a second-order effect. Since
depth and width rise together in this family, that second-order term could be either.

This is what makes the depth result portable. An experimenter can estimate how much plateau a patch
site will manufacture in an untested model from $f$ alone, and a patch site quoted as "block 20" is not
comparable across models — it is near-linear in a 24-block model and a strong plateau in a 60-block
one. It also explains Table 1: the same `big`/`in` pair plateaus in a 36-block model and not in a
24-block one because $f = 1$ buys more compression when the stack is longer.

### Testing the motivating hypothesis with real features: no intermediate plateaus, but different circuits do sharpen the switch

The hypothesis is that *holding output JSD low, different circuits or features may occupy different
plateaus*. Output divergence is a **control to be held low**, not a variable to correlate against, so
Table 3 does not bear on it. What it predicts about the curve admits two readings, and Experiment 6
tests both against every instrument: a resting point **between** A and B (measured by IPW), or **each
prompt on its own plateau** with a snap between them (measured by $w_{TV}$, with divergence held low so
this is not Table 3 in disguise).

Both readings need an independent variable that says *which machinery* the two prompts use, not how far
apart their representations sit. We therefore identify the features and heads directly: sparse-autoencoder
feature sets (SFD, SFC) in GPT-2 Small, where public SAEs cover every residual-stream location, and
attention-head contributions (HCD, HSD) and MLP neuron sets (NSD) in all three models. The old
residual-stream proxy IRD is kept in the table so the upgrade can be priced.

The SAEs behave as they should on this bank's prompts, which is what licenses reading SFD as a feature
measurement at all: reconstruction explains 77–91% of the variance at every hook point, with 20–77 of
the 24576 features active per token. Figure 8 shows that check alongside the two readings of the
hypothesis in the one model where features are directly available.

![SAE feature-set disjointness against intermediate-plateau width and against transition width in GPT-2 Small, with autoencoder validation](plots/sae_features.png)

**Figure 8.** In GPT-2 Small, prompts that fire more disjoint SAE feature sets do not rest at
intermediate levels (left) but do switch more sharply (middle). Left and middle: one point per low-JSD
mined pair ($n = 365$); x = SFD, the Jaccard distance between the two prompts' active feature sets
(0 = identical features, 1 = no shared feature); y = IPW, the intermediate-plateau width (left) and
$w_{TV}$, the transition width (middle). Dashed = the linear-response value of that statistic (0.10 and
0.5); dotted = the threshold for calling it an intermediate plateau (0.20) or sharp (0.25). Right: the
autoencoder validation — x = the block after which the residual stream is read; solid circles, left y
= fraction of activation variance the SAE reconstructs; dashed squares, right y = active features per
token ($L_0$).

**Table 7 — the low-JSD banks and the pre-specified primary test.** Every pair here has
$\mathrm{JSD} < 0.1$: the two prompts predict nearly the same next token. "% intermediate" is the share
of pairs with $\mathrm{IPW} > 0.20$; "% sharp" the share with $w_{TV} < 0.25$. The primary test is the
most direct set-of-features instrument available in that model against IPW, Holm-corrected across the
three models.

| Model | $n$ | prefixes | median JSD | % intermediate | % sharp | primary test | $\rho$ | $p$ | Holm $p$ | detectable $\lvert\rho\rvert$ |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2-small | 365 | 102 | 0.035 | 40.0% | 31.5% | SFD → IPW | $+0.08$ | 0.12 | 0.24 | 0.10 |
| gpt2-medium | 399 | 119 | 0.035 | 37.3% | 47.1% | HSD → IPW | $-0.04$ | 0.40 | 0.40 | 0.10 |
| gpt2-large | 356 | 113 | 0.039 | 2.0% | 64.0% | HSD → IPW | $-0.11$ | 0.04 | 0.13 | 0.10 |

**The intermediate-plateau reading fails, and now it fails with power.** No primary test survives
correction, and the exploratory ones agree: across all fourteen instrument-model combinations the
correlation with IPW runs from $-0.11$ to $+0.12$, inside the band that these sample sizes cannot
distinguish from zero. This is not the earlier under-powered null — at $n \approx 370$ any association
of $\lvert\rho\rvert \ge 0.10$ would have shown up, so the finding rules out everything but a weak
effect. GPT-2 Large makes the point most starkly: 64% of its low-JSD pairs switch sharply, yet only
2.0% pause anywhere in between. Its curves step once; they do not climb a staircase.

**The endpoint-plateau reading holds, on every instrument and in every model.** Table 8 gives all
fourteen tests against $w_{TV}$, and all fourteen are negative — more different machinery, sharper
switch — with thirteen significant at $p < 0.05$ and eleven with cluster-bootstrap intervals excluding
zero.

**Table 8 — every circuit-difference measure against both readings.** IQR is the measure's own
interquartile range, included because an instrument that does not vary cannot detect anything. The
partial column repeats the $w_{TV}$ test with the residual JSD inside the low band and the block-0
angle $\Omega$ regressed out.

| Model | measure | IQR | $\rho$ → IPW | $\rho$ → $w_{TV}$ | 95% CI ($w_{TV}$) | partial $\rho$ → $w_{TV}$ |
|---|---|---|---|---|---|---|
| gpt2-small | SFD (SAE features) | 0.52–0.71 | $+0.08$ | $-0.21$ | $[-0.34, -0.07]$ | $-0.17$ |
| gpt2-small | SFC (SAE profile) | 0.34–0.64 | $+0.04$ | $-0.12$ | $[-0.25, +0.01]$ | $-0.04$ |
| gpt2-small | HCD (head contributions) | 0.022–0.047 | $+0.07$ | $-0.21$ | $[-0.34, -0.08]$ | $-0.21$ |
| gpt2-small | HSD (head sets) | 0.07–0.21 | $+0.00$ | $-0.18$ | $[-0.31, -0.05]$ | $-0.13$ |
| gpt2-small | NSD (neuron sets) | 0.58–0.73 | $+0.12$ | $-0.20$ | $[-0.32, -0.08]$ | $-0.16$ |
| gpt2-small | IRD (representation geometry) | 0.09–0.18 | $+0.10$ | $-0.17$ | $[-0.30, -0.04]$ | $-0.15$ |
| gpt2-medium | HCD (head contributions) | 0.021–0.054 | $-0.10$ | $\mathbf{-0.36}$ | $[-0.47, -0.25]$ | $-0.43$ |
| gpt2-medium | HSD (head sets) | 0.11–0.21 | $-0.04$ | $-0.23$ | $[-0.34, -0.09]$ | $-0.25$ |
| gpt2-medium | NSD (neuron sets) | 0.60–0.77 | $+0.03$ | $-0.23$ | $[-0.35, -0.10]$ | $-0.32$ |
| gpt2-medium | IRD (representation geometry) | 0.046–0.106 | $+0.05$ | $-0.13$ | $[-0.26, +0.00]$ | $-0.28$ |
| gpt2-large | HCD (head contributions) | 0.045–0.096 | $-0.03$ | $-0.29$ | $[-0.44, -0.12]$ | $-0.38$ |
| gpt2-large | HSD (head sets) | 0.15–0.29 | $-0.11$ | $\mathbf{-0.31}$ | $[-0.45, -0.17]$ | $-0.28$ |
| gpt2-large | NSD (neuron sets) | 0.64–0.81 | $-0.03$ | $-0.26$ | $[-0.42, -0.09]$ | $-0.34$ |
| gpt2-large | IRD (representation geometry) | 0.11–0.25 | $+0.10$ | $-0.11$ | $[-0.29, +0.07]$ | $-0.19$ |

**Where this is strongest, and why it matters.** The effect is largest in the two deeper GPT-2 models
and on the head-level instruments — $\rho = -0.36$ in GPT-2 Medium for HCD and $-0.31$ in GPT-2 Large
for HSD — and it survives partialling out the two obvious alternative explanations, residual divergence
inside the low band and the block-0 geometry of the patched vectors; for HCD the partial correlation is
*larger* than the raw one ($-0.43$ and $-0.38$). This is the first quantity in this report that predicts
plateau strength **from the pair itself**. Everything up to Experiment 5 said sharpness is manufactured
by depth, with output divergence sorting which pairs sharpen most; here depth and outputs are both held
fixed and the pair's internal machinery still orders the curves. Practically, that gives an experimenter
a check they can run: if a chosen pair shows a sharp switch, ask whether it also engages unusually
disjoint heads or features, because that is the part of the effect not supplied by the architecture.

**How much the feature-level measurement bought.** IRD, the residual-stream proxy this study used
before, is the weakest instrument in both deep models — $-0.13$ in GPT-2 Medium and $-0.11$ in GPT-2
Large, the only two rows whose confidence interval touches zero — while head-level measurement on the
identical pairs reaches $-0.36$ and $-0.29$. Distance between representations and difference of
machinery are not the same quantity, and the earlier proxy was diluting the signal by roughly a factor
of three. In GPT-2 Small the SAE instrument matches the head-level ones ($-0.21$ against $-0.21$),
which is the reassuring case: two very different ways of asking "are different features involved" agree.

**What the correlations alone do not license.** These are rank correlations of $0.2$–$0.4$, so circuit
difference orders the curves without determining them — it accounts for something like 4–13% of the
rank variance in $w_{TV}$, against the far larger effect of relative depth in Experiment 5. The
instruments are also correlated with each other by construction, so fourteen tests are not fourteen
independent replications; the honest summary is one effect, seen from six angles in three models.

Figure 9 puts all fourteen tests on one axis, which is the clearest way to see the two readings of the
hypothesis come apart.

![Spearman correlations between six circuit-difference measures and two plateau statistics in three models](plots/circuit_forest.png)

**Figure 9.** The two readings of the hypothesis separate cleanly: everything pointed at intermediate
plateaus lands in the undetectable band, everything pointed at transition width lands left of zero. x:
Spearman $\rho$ between the circuit-difference measure and the plateau statistic named in the row label
(measures defined in Methods; IPW = intermediate-plateau width, WTV = transition width $w_{TV}$); bars
= 95% cluster-bootstrap intervals over prefixes. y: one row per model-measure-statistic combination,
grouped by model (gpt2-small circles, gpt2-medium squares, gpt2-large triangles, separated by
horizontal rules). The gray band marks $\lvert\rho\rvert < 0.10$, the smallest correlation these
sample sizes can detect; the dashed vertical line is no association. Large markers with thick edges are
the three pre-specified primary tests.

### Removing the differentially-engaged heads destroys the sharp switch in GPT-2 Large

An association between circuit difference and sharpness has an obvious innocent explanation: some third
property of a prompt pair could produce both. The way to rule that out is to delete the machinery and
see whether the switch survives. Experiment 7 mean-ablates, at the final token only, the $k$ heads that
write most differently for the two prompts, and compares that against deleting an equal-sized set of
heads matched on how much they write but chosen to write *similarly* for the two prompts. Both
conditions remove about the same quantity of attention output — the median ratio of removed
$\sum_h m_h$ is $1.01$–$1.02$ in GPT-2 Large and $1.08$–$1.12$ in GPT-2 Medium — so the comparison
isolates the discriminating part. Both endpoints are re-run under the ablation, and the identity check
still holds ($|d(0)|, |d(1)-1| \le 3.5\times10^{-4}$), so these are the ablated models' own switches.

**Table 9 — the intervention, at three doses.** $k$ is the number of heads ablated. "median $w_{TV}$"
is over that model's whole low-JSD bank; $\Delta$ is the *paired* median of
$w_{TV}(\text{differential}) - w_{TV}(\text{control})$ with a 95% cluster bootstrap over prefixes and a
Wilcoxon signed-rank $p$; "HCD left" is the median head-contribution distance after the differential
ablation as a fraction of its unablated value, the manipulation check.

| Model | dose ($k$) | median $w_{TV}$: none | control | differential | $\Delta$ | 95% CI | $p$ | pairs with $\Delta>0$ | HCD left |
|---|---|---|---|---|---|---|---|---|---|
| gpt2-large | 3% (22) | 0.198 | 0.198 | **0.358** | $+0.097$ | $[+0.054, +0.146]$ | $1.4\times10^{-43}$ | 83% | 0.76 |
| gpt2-large | 6% (43) | 0.198 | 0.196 | **0.441** | $+0.145$ | $[+0.093, +0.201]$ | $1.8\times10^{-48}$ | 87% | 0.65 |
| gpt2-large | 10% (72) | 0.198 | 0.200 | **0.484** | $+0.199$ | $[+0.125, +0.268]$ | $3.3\times10^{-47}$ | 87% | 0.54 |
| gpt2-medium | 3% (12) | 0.257 | 0.251 | 0.264 | $+0.009$ | $[+0.000, +0.014]$ | $0.019$ | 55% | 0.71 |
| gpt2-medium | 6% (23) | 0.257 | 0.248 | 0.258 | $+0.009$ | $[+0.001, +0.016]$ | $0.010$ | 56% | 0.61 |
| gpt2-medium | 10% (38) | 0.257 | 0.250 | 0.263 | $+0.010$ | $[+0.002, +0.018]$ | $0.014$ | 56% | 0.52 |
| gpt2-small | 3% (4) | 0.315 | 0.312 | 0.345 | $+0.014$ | $[+0.008, +0.025]$ | $1.7\times10^{-4}$ | 63% | 0.73 |
| gpt2-small | 6% (9) | 0.315 | 0.315 | 0.325 | $+0.019$ | $[+0.006, +0.030]$ | $1.6\times10^{-3}$ | 59% | 0.57 |
| gpt2-small | 10% (14) | 0.315 | 0.315 | 0.350 | $+0.025$ | $[+0.010, +0.041]$ | $6.5\times10^{-4}$ | 59% | 0.47 |

**In GPT-2 Large the sharp switch is caused by these heads.** Deleting 22 of 720 heads — 3% — takes the
median transition width from $0.198$ to $0.358$, an 81% widening, and 10% of heads takes it to $0.484$,
which is the linear response ($0.5$) to within 3%: the model has stopped switching and started
responding proportionally. The matched control does nothing at any dose ($0.198 \to 0.198, 0.196,
0.200$), so this is not the generic effect of removing attention output. The effect is present in 83–87%
of individual pairs, it grows monotonically with dose, and it tracks the manipulation check — each dose
removes more of the measured circuit difference and widens the switch further. Since GPT-2 Large is the
model in which the phenomenon was reported, this identifies what produces it there: a small set of
attention heads that write in different directions for the two prompts.

**In the two smaller GPT-2 models the same intervention barely moves the curve.** It replicates in
both, at every dose, with every interval excluding zero — GPT-2 Medium $+0.009$, $+0.009$, $+0.010$ and
GPT-2 Small $+0.014$, $+0.019$, $+0.025$ — but it is 4 to 15 times smaller than in GPT-2 Large, and
56–63% of pairs move in the predicted direction against a 50% coin flip, rather than 83–87%. The
manipulation was not weaker in the small models: it removed *more* of the circuit difference than in
GPT-2 Large (HCD down to 0.52 and 0.47 at the top dose, against 0.54). Two things follow. The effect is
not ordered by model size, since GPT-2 Small sits above GPT-2 Medium, so this is a property of GPT-2
Large and not a trend in depth. And GPT-2 Medium is the model with the *stronger* correlation in
Experiment 6 ($\rho = -0.36$ for HCD, against $-0.29$ in GPT-2 Large), so the size of an association was
a poor guide to what an intervention would do — an argument for running the intervention rather than
inferring it. GPT-2 Small's dose curve appears with the others in Figure 11D.

Figure 10 shows the dose-response for the two deeper models together with the manipulation check.

![Transition width against ablation dose for differential and matched-control head sets in two models, with the paired effect and a manipulation check](plots/ablation_causal.png)

**Figure 10.** Removing the heads that write differently for the two prompts flattens the switch in
GPT-2 Large and hardly touches GPT-2 Medium. x in all four panels: the ablation dose, as a percentage of
all attention heads in the model (3%, 6%, 10%). Top row, y: median $w_{TV}$ over that model's whole
low-JSD bank (smaller = sharper); circles with a solid line = no ablation, squares dashed = matched
control heads, triangles dotted = differential heads; gray dashed = linear response (0.5), dotted =
sharp threshold (0.25). The top-left panel is scaled to the same range as the top-right, which is why
GPT-2 Medium's three conditions nearly coincide — its effect is real but 15 times smaller, and the
bottom-left panel is where to read it. Bottom left, y (symmetric log scale): the paired median of
$w_{TV}$(differential) $-$ $w_{TV}$(control), bars = 95% cluster bootstrap over prefixes, gray dashed =
no effect; gpt2-medium squares dashed, gpt2-large triangles dotted. Bottom right, y: median HCD after
ablation as a fraction of its unablated value — the manipulation check — with the matched-control set (squares) and the
differential set (triangles) for both models; gray dashed = no change.

**What this does and does not establish.** The heads are selected per pair by the same quantity that
defines HCD, so the intervention confirms that the construct Experiment 6 measures is causally load-bearing
in GPT-2 Large, not that some independently-discovered circuit is. Mean-ablation at one position is also
a blunt instrument: it holds the head's output at a bank average rather than removing it from the
computation graph, and a large enough dose would eventually degrade any behavior — which is why the
matched control, at the identical dose, is the comparison that carries the claim. Experiment 8 takes the
per-pair selection away and asks where in the stack the heads act.

### One fixed set of heads works better than per-pair selection — and it works from above the patch

Experiment 7 leaves two questions open. If a fresh head set is chosen for every pair, is there a circuit
here at all, or only a construct that happens to be load-bearing? And wherever these heads are, do they
shape the sweep by processing the interpolated vector or by deciding what that vector contains?

The counting version of the first question is head-set overlap $J$ between pairs built on *different*
prefixes, read against the random-selection null and against the magnitude-ranked set.

**Table 10 — how much the per-pair head sets recur.** $k$ is the 3% dose. "Most-selected head" is the
head entering the largest share of pairs' top-$k$ sets. The last column is the share of all selections
that fall on the $k$ most frequently selected heads; if every pair chose the same set it would be 100%.

| Model | $k$ (3% of heads) | most-selected head, and its rate | $J$, different prefixes: differential | by magnitude | random | selections in the top $k$ heads |
|---|---|---|---|---|---|---|
| gpt2-large | 22 of 720 | block 0, head 14 — 78.9% | 0.090 | 0.160 | 0.016 | 30.7% |
| gpt2-medium | 12 of 384 | block 1, head 4 — 46.1% | 0.064 | 0.412 | 0.016 | 24.7% |
| gpt2-small | 4 of 144 | block 0, head 1 — 85.8% | 0.280 | not run | 0.016 | 58.7% |

**There is a recurring core with a pair-specific tail.** Overlap between pairs from different prefixes
runs 4× (GPT-2 Medium), 6× (GPT-2 Large) and 18× (GPT-2 Small) the random rate, and GPT-2 Large's single
most-selected head enters four pairs in five — far from the pair-specific picture Experiment 7 assumed.
It is not one fixed list either: GPT-2 Large's overlap is well below its magnitude-ranked set's, and its
22 most frequent heads carry only 30.7% of all selections. The core is tighter in the smallest model,
which has fewer heads to choose from.

The causal version of the question is the decisive one, so we ran it: rank heads on one half of the
prefixes, then ablate that single fixed set — the same $k$ heads for every pair — on the held-out half.

**Table 11 — a fixed cross-pair head set, ablated on pairs that had no say in choosing it.** "Fixed
$\Delta$ vs none" is the paired median change in $w_{TV}$ with a 95% cluster bootstrap over prefixes;
$p$ is a Wilcoxon signed-rank test of the fixed set against the per-pair matched control at the same
dose; the last column is the recovery fraction defined in Methods.

| Model | $n$ (held out) | median $w_{TV}$: none | matched control | per-pair set | fixed set | fixed $\Delta$ vs none | 95% CI | $p$ vs control | recovery |
|---|---|---|---|---|---|---|---|---|---|
| gpt2-large | 356 | 0.198 | 0.198 | 0.358 | **0.485** | $+0.189$ | $[+0.140, +0.249]$ | $4\times10^{-51}$ | 198% |
| gpt2-medium | 399 | 0.257 | 0.251 | 0.264 | 0.254 | $+0.004$ | $[+0.000, +0.007]$ | $0.033$ | 70% |

**A fixed set is not merely as good as per-pair selection in GPT-2 Large — it is better.** Ablating the
same 22 heads for every held-out pair takes the median transition width to $0.485$, which is the linear
response to within 3%, against $0.358$ for sets tailored pair by pair ($p = 1\times10^{-17}$ for the
difference), even though the fixed set shares only 29.4% of its heads with the average pair's own
top-22. Recovery above 1 has a clean reading: choosing heads per pair was adding noise, and the shared
core is what carries the effect. GPT-2 Medium shows the same structure in miniature, recovering 70% of
its much smaller per-pair effect. So there is a circuit to name in both models, not just a construct.

Naming it exposes a problem for the obvious interpretation. The most frequently selected heads sit in
**block 0**, and the interpolated vector replaces the final token's residual stream *after* block 0. A
block-0 head cannot process the interpolated vector; it can only change the two endpoint activations
that get interpolated. So we rebuilt the fixed set from the same held-out ranking with block 0 removed.

**Table 12 — splitting the fixed-set effect by where the heads sit, in GPT-2 Large.** Both rows ablate
22 heads at the 3% dose on the same held-out pairs; the second draws them only from blocks 1–35, so
every ablated head is downstream of the patch.

| GPT-2 Large, held-out fixed set | median $w_{TV}$ | $\Delta$ vs none | 95% CI | $p$ vs control | recovery |
|---|---|---|---|---|---|
| all blocks (22 heads) | 0.485 | $+0.189$ | $[+0.140, +0.249]$ | $4\times10^{-51}$ | 198% |
| block 0 excluded (22 heads) | 0.217 | $+0.012$ | $[+0.009, +0.017]$ | $5\times10^{-24}$ | 13% |

**Most of the effect is upstream of the patch.** Striking block 0 costs 94% of the widening. What
remains is small but unambiguous — $+0.012$, interval far from zero, $p = 5\times10^{-24}$ against the
matched control — so heads below the patch genuinely contribute, an order of magnitude less than the
handful of block-0 heads that decide what the interpolated vector contains. Figure 11 collects the
recurrence, the depth profile, the held-out ablation and the three-model dose response.

![Head selection frequency, depth profile of selected heads, held-out fixed-set ablation, and dose response in three GPT-2 models](plots/localization.png)

**Figure 11.** The differential heads are a shared core dominated by block 0, and a single fixed set
transfers to held-out pairs. **A** — x: head rank after sorting all heads by how often they enter a
pair's top-$k$ differential set (log scale); y: that fraction. Dotted horizontals mark the rate expected
if pairs chose heads at random, $k/H$ with $H$ the model's total head count. gpt2-small circles solid,
gpt2-medium squares dashed, gpt2-large triangles dotted, in every panel. **B** — x: relative depth, the
block index divided by (blocks $-$ 1), so models of different depth share one axis; y: the share of all
selected heads sitting in that block; the legend gives each model's block-0 share. **C** — y: median
$w_{TV}$ over the held-out pairs (smaller = sharper) under no ablation, the per-pair matched control,
the fixed cross-pair set, the per-pair differential set, and — GPT-2 Large only — the fixed set with
block-0 heads excluded; gray dashed = linear response (0.5). **D** — x: ablation dose as a percentage of
all heads; y (symmetric log): the paired median of $w_{TV}$(differential) $-$ $w_{TV}$(control) from
Experiment 7, bars = 95% cluster bootstrap over prefixes, gray dashed = no effect.

**What this changes for a user of the probe, and what it leaves open.** Sharpness has two sources that
the curve itself cannot separate. Depth below the patch supplies the capacity to compress a change
(Experiments 4–5), but *what* is available to be compressed is fixed before the patch: a few early heads
write the discriminating part of the activation, and a sweep between two vectors differing in that part
snaps. Reporting a plateau therefore describes the endpoints as much as the mechanism, and moving the
patch site one block earlier or later changes which of the two you are measuring. Two things remain
unattributed. The block-0 share does not explain the cross-model gap — GPT-2 Small draws 62.6% of its
differential heads from block 0 against GPT-2 Large's 16.7%, and its intervention effect is 4–7 times
smaller — so what makes GPT-2 Large special is still open. And the fixed set is not magnitude-matched
pair by pair the way Experiment 7's control is, so its claim rests on the per-pair control and on the
block-0-excluded variant, both run at the identical dose.

## Conclusion

The reported plateau is real in the model it was reported in. In GPT-2 Large, interpolating one token's
block-0 activation gives `The house was big`/`in` a near-step response ($w_{10-90} = 0.044$, 95% of the
sweep pinned at an endpoint) and `big`/`large` a near-linear one ($0.592$), a 13-fold difference that
holds across three sharpness statistics. In GPT-2 Medium the same pair does not plateau under the
predefined criterion, so model depth is not a free choice when reproducing this effect.

What governs the effect is how much of the network sits below the patch, measured as a fraction rather
than a count. Three GPT-2 models of 12, 24 and 36 blocks nearly superimpose when plotted against the
fraction below the patch and separate by up to $0.29$ in median $w_{TV}$ when plotted against the block
count; GPT-2 Large at block 12 and GPT-2 Medium at block 0 share 23 blocks below and differ 3.2-fold.
Removing that depth removes the plateau — Pythia-410m with 3 blocks below the patch has zero sharp
pairs out of 200 and a median response within 2% of proportional. But depth only sets the ceiling:
`big`/`large` has 35 blocks below it in GPT-2 Large and stays smooth, so the interpolation path decides
whether that ceiling is reached.

Three things follow for anyone using interpolation as a probe. Report the patch site as a fraction of
the stack, since a block number is not comparable across models. Report a base rate or a matched
comparison pair, since 83.5% of arbitrary GPT-2 Large pairs plateau under the predefined criterion and
a single sharp curve is therefore weak evidence — pairing `big`/`in` with `big`/`large` is the right
design, and it is the part worth copying. And keep the claim matched to the measurement: a narrow
A-to-B transition is not evidence about intermediate feature states, which is what the motivating
hypothesis is about.

On the motivating hypothesis the two readings come apart. The intermediate-plateau reading is dead as
a large effect: across 1120 low-divergence pairs in three models, no feature- or circuit-level measure
predicts a resting point between A and B ($\rho$ from $-0.11$ to $+0.12$, nothing surviving Holm
correction, at sample sizes that would have caught $\lvert\rho\rvert \ge 0.10$), and in GPT-2 Large
only 2.0% of such pairs pause anywhere in the middle. The endpoint-plateau reading holds, and in GPT-2
Large it holds causally: with outputs matched, pairs engaging more disjoint heads, neurons or
sparse-autoencoder features switch more sharply on all 14 instrument-model combinations, and deleting
3% of heads chosen for writing differently for the two prompts widens the median switch by 81% while an
engagement-matched control set does nothing.

Those heads are a shared core rather than a per-pair accident, and locating them changes what the
intervention means. One fixed set of 22 heads, chosen on half the prefixes, beats per-pair selection on
the other half ($0.485$ against $0.358$), so there is a circuit to name; but its most-selected members
sit in block 0, above the patch, and excluding them costs 94% of the effect. The useful reframing for
anyone using this probe is that a plateau is two things at once: depth below the patch supplies the
capacity to compress a change into a small stretch of the interpolation, and a handful of early
prompt-discriminating heads decides how much difference there is to compress. A sharp curve is evidence
about both, and the probe cannot tell them apart on its own — moving the patch site is what separates
them.

**Limitations.** The intervention's heads are selected per pair by the same statistic that defines HCD,
so it shows the measured construct is load-bearing rather than validating an independently-discovered
circuit; Experiment 8's fixed set is derived from the same statistic and is not magnitude-matched pair
by pair, so it leans on the per-pair control run at the identical dose. The three GPT-2 models differ
up to 15-fold in how much the intervention does, and the difference is not ordered by size and not
explained by where the heads sit, so it is described rather than attributed. All results are for one
patched position, the final token, and one
interpolation scheme; pairs differing at an earlier position, or in more than one token, are untested. Mined pairs are built by swapping the
final token for a lower-ranked alternative, so both continuations are ones the model itself considered
plausible. JSD saturates at $\ln 2$ and a third of mined pairs sit near that ceiling, which is why
Table 3 restricts to the unsaturated subset. The depth-scaling result rests on one family, where
residual width rises with depth (768, 1024, 1280), so the residual spread in Table 6 cannot be assigned
to depth or width separately, and relative depth has not been checked outside GPT-2 or outside the
12–36 block range. The cross-model differences in Table 2 are not attributed: architecture, corpus and
pretraining length are confounded across families. Finally, $d(\alpha)$ measures movement in raw logit
space, so a pair could hold its logit vector still while reordering low-probability tokens and this
metric would not see it.
