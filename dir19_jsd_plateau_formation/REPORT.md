# When do activation plateaus form, and when does corpus statistics start to predict them?

> Direction `dir19_jsd_plateau_formation`. Final, presentable, current-best only. History lives in
> `CHANGELOG.md`.

## Summary

Large language models appear to compute in a piecewise way: you can move a hidden activation a long
distance and the model's output barely changes, then cross an invisible boundary and the output
flips. The flat parts are called **activation plateaus**. They matter for safety because most
interpretability tooling works by perturbing activations — activation steering, activation patching,
feature attribution. Inside a plateau those perturbations return nothing; near a boundary they return
everything. So the shape of this landscape controls what our tools can see.

Earlier work in this project established a *correlate* of the boundaries in one model: for a pair of
single-token continuations, the more the two tokens are followed by different words in the training
corpus, the **sharper** the transition between them. This report asks a timing question: during
training, does the model first learn *which* pairs get sharp boundaries, or does it first make
*all* transitions sharp? We scan 20 released checkpoints of Pythia-1.4B-deduped, from step 0 to step
143000, running one frozen 60-pair bank at every checkpoint.

The two events are separated by roughly two orders of magnitude in training steps, and they come in
the surprising order:

1. **Corpus statistics start sorting the pairs between step 8 and step 32** — within the first 32 of
   143,000 steps, at a learning rate still in warmup. At step 32 the correlation is already
   $\rho = -0.428$, two-thirds of its final value ($-0.525$), and it never returns to zero.
2. **At that moment there are no plateaus at all.** Median transition width at step 32 is 0.827,
   statistically indistinguishable from the straight-line reference value of 0.8 and from the
   untrained model's 0.831. The model has ranked the pairs correctly across a total spread of
   0.006 in width.
3. **Plateau shape appears much later, between step 1000 and step 2000**, and the single interval
   that produces the largest global sharpening (step 512 → 1000) does not sort pairs by corpus
   statistics at all ($\rho = +0.035$, 95% CI $[-0.241, +0.307]$).

So the ordering is not a by-product of sharpening: it is laid down first, in a regime where the
quantity it orders is barely varying, and later training sharpens nearly every pair together while
keeping corpus divergence pointing the same way throughout.

A third measurement says how much of the *final* answer the model holds at step 32, and the answer
is: only the divergence-aligned part of it. The per-pair ranking of widths at step 32 agrees with the
final model's ranking at just $\pi = 0.161$ — inside the chance envelope — and once corpus divergence
is partialled out, the agreement is nothing at all ($-0.082$). The pair-specific detail of the final
ranking arrives between step 64 and step 128, a third clock sitting between the other two. This is
not a measurement-noise artefact: the three carrier sentences agree on each pair's width at step 32
at $\bar r = 0.83$, so $\pi$ could have reached 0.94 had the rankings matched.

Because the step-32 ordering sits on such a tiny spread, we
also check it against chance directly: under 20,000 relabellings it gives $p = 0.0007$, and
$p = 0.0072$ after paying for having examined all 19 checkpoints. On the 1,000-pair bank, where
permuting the 123 endpoint labels prices the token reuse into the null, the same bracket holds
($p = 0.64$ at step 8, $p = 0.0031$ at step 32). We also confirm a **late reversal** — transitions get
*blunter* over the last third of training — on an independent 1,000-pair bank with inference that
accounts for reuse of endpoint tokens (median $\Delta w = +0.0158$, 95% CI $[+0.0081, +0.0224]$).

A by-product of the scan: the artefact that Hugging Face serves as revision `step16` of
`EleutherAI/pythia-1.4b-deduped` **is not a step-16 model**. Its held-out loss is 2.320 nats where
its neighbours are near 9, and its 9,000 measured curve values are bit-identical to the final
checkpoint's. Anyone using Pythia's early checkpoints should check this.

---

## Methods

### Data & Model

**Model.** `EleutherAI/pythia-1.4b-deduped` (1.4B parameters, GPT-NeoX architecture, 24 blocks,
hidden size 2048), in `float32`. We use 20 released revisions: steps 0, 1, 2, 4, 8, 32, 64, 128,
256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 96000, 128000, 143000. Revision `step16` was
assayed and then **excluded**; the evidence is in Result 9. The tokenizer for each revision is loaded
from that same revision.

**Hook point.** The final token position of the residual stream immediately after **block 0**. This
is the same patch point used by the upstream work this direction continues, so all numbers are
directly comparable.

**Pair bank (primary).** 60 pairs of single-token continuations, frozen once and never reselected.
Each pair is two lowercase alphabetic word-start tokens that are among the model's top-256
continuations in all three carrier sentences, occur at least 20,000 times per corpus split, differ
in corpus frequency by at most a factor of two, and are balanced across five corpus-divergence
quintiles on log-frequency and surprisal. The bank is **endpoint-disjoint**: no token appears in two
pairs, so the 60 pairs are independent observations. Each pair is measured in three carrier
sentences — `"The thing was"`, `"They said it was"`, `"I thought it was"` — giving 180 curves per
checkpoint and 3,600 curves in total across the scan.

**Pair bank (validation).** 1,000 pairs built from 123 endpoint tokens under the same corpus rules.
Endpoint tokens recur across pairs here, so these 1,000 observations are not independent and need
clustered inference (defined below). This bank is measured at steps 0, 8, 32, 64000 and 143000.

**Corpus statistics.** The Pile-deduped sample used upstream: two disjoint 500,000-row splits of
2,049 tokens each. Split A selected the pairs; split B is the held-out split all reported
divergences come from. The split-half reliability of the divergence estimate is high
(Spearman between splits $= 0.9998$; median split-half divergence is 7.2% of the median
between-token divergence).

**Timing context.** At every checkpoint we also evaluate next-token cross-entropy on one frozen
held-out sample — the last 256 rows of split B, truncated to 512 tokens, never used for pair
selection — and record the published learning rate.

**Interpolation.** Between the two endpoint activations we take 50 evenly spaced positions
$t \in [0, 1]$ using norm-rescaled spherical linear interpolation (SLERP): directions are
interpolated on the unit sphere and the norm is interpolated linearly, so intermediate states keep a
plausible magnitude. Only the final token position of block 0's output is replaced; the remaining
23 blocks run normally.

**Validity.** A curve is used only if it starts at or below 0.1, ends at or above 0.9, never
backslides by more than 0.02, and crosses each of the 0.1 and 0.9 levels exactly once. **All 3,600
curves passed** at every checkpoint, so no result below is affected by curve rejection. Patching at
$t=0$ and $t=1$ reproduces the unpatched endpoint logits to a maximum relative error of
$4.6 \times 10^{-5}$ across the entire scan.

### Metrics

The scan needs one number per pair that says *how sharp* the transition is, one that says *how flat
the ends are*, one that says *how selectively* training acts, and one that says *where along the path
the model's output actually changes*. We build them in that order.

**Relative logit distance $d(t)$.** Reading a plateau off raw logits is hard because logit scale
changes enormously during training — comparing step 0 to step 143000 in absolute units is
meaningless. Normalising by the distance between the two endpoints removes the scale, so $d$ always
runs from 0 to 1 and is comparable across checkpoints. With $z(t)$ the final-position logits
restricted to the 50,060 tokenizer IDs that occur in the corpus sample, and $z_A = z(0)$,
$z_B = z(1)$:

```math
d(t) \;=\; \frac{\lVert z(t) - z_A\rVert_2}{\lVert z(t) - z_A\rVert_2 \;+\; \lVert z(t) - z_B\rVert_2}
```

$d = 0$ means the output is still exactly what endpoint $A$ produced; $d = 1$ means it has fully
become endpoint $B$. A plateau shows up as a curve that stays near 0, then rises steeply, then stays
near 1.

**Transition width $w$ (primary shape metric).** The steepness of that rise, measured as the
fraction of the path spent between the 10% and 90% levels:

```math
w \;=\; t(d = 0.9) \;-\; t(d = 0.1)
```

Smaller $w$ means a sharper boundary. A model with no plateau structure at all, whose output moves
uniformly along the path ($d(t) = t$), gives $w = 0.8$; this is the **straight-line reference** we
compare against throughout. Per pair we take the median $w$ over the three carrier sentences.
Results 2, 3 and 5 consume this metric.

**Edge drift $E$.** Width alone can be fooled: a curve could have a steep middle and still creep
steadily at the ends, which is not a plateau. $E$ measures exactly that creep — how far $d$ moves
away from its endpoint values inside the outer 20% of the path:

```math
E \;=\; \frac{1}{|L|}\sum_{t \in L}\bigl(d(t) - d(0)\bigr) \;+\; \frac{1}{|R|}\sum_{t \in R}\bigl(d(1) - d(t)\bigr),
\qquad L = \{t \le 0.2\},\; R = \{t \ge 0.8\}
```

$E \approx 0$ means genuinely flat ends. The straight line $d(t) = t$ gives $E = 0.184$. Result 2
uses $E$ as the independent confirmation that narrow transitions really are plateaus.

**Corpus next-token divergence $J$.** The property of a token pair we test as a predictor: how
differently the corpus continues after token $u$ versus token $v$. With $P_u$ the empirical
distribution of the token that follows $u$ in held-out split B, this is the base-2 Jensen–Shannon
divergence (JSD) — a symmetric, bounded measure of how far apart two distributions are, in bits:

```math
J(u,v) \;=\; \tfrac{1}{2}\mathrm{KL}\!\left(P_u \,\Vert\, M\right) \;+\; \tfrac{1}{2}\mathrm{KL}\!\left(P_v \,\Vert\, M\right),
\qquad M = \tfrac{1}{2}\left(P_u + P_v\right)
```

$J = 0$ means the two tokens are followed by identical word distributions; $J = 1$ bit means their
continuations share no support. $J$ is a fixed property of the corpus and does not change with the
checkpoint.

**Cross-sectional ordering $\rho_s$ (primary selectivity metric).** At checkpoint $s$, how well does
corpus divergence rank the pairs by sharpness? We use Spearman rank correlation, which cares only
about ordering, because the *scale* of $w$ changes drastically over training while the question is
about rank:

```math
\rho_s \;=\; \mathrm{Spearman}\bigl(J,\; w_s\bigr)
```

A **negative** $\rho_s$ is the effect of interest: higher corpus divergence goes with *smaller*
width, i.e. sharper boundaries. $\rho_s = 0$ means corpus statistics carry no information about which
pairs are sharp. Result 1 is a trajectory of this quantity.

**Simultaneous 95% band.** Reading an onset off 20 checkpoints means asking 20 questions at once, so
pointwise intervals would find a "first significant checkpoint" by chance alone. We resample the 60
pairs with replacement 4,000 times, recompute the *whole* trajectory from each resample (the same
resampled pairs at every checkpoint, so the trajectory stays paired), and take the constant $c$ that
covers the maximum deviation across all checkpoints 95% of the time:

```math
c \;=\; Q_{0.95}\Bigl(\max_{s}\bigl|\rho_s^{(b)} - \rho_s\bigr|\Bigr), \qquad \text{band} = \rho_s \pm c
```

Here $c = 0.324$. The band is wider than a pointwise interval by construction; an onset call that
survives it is not a multiple-comparison artefact.

**Interval-specific selectivity.** A cross-sectional correlation at checkpoint $s$ can persist purely
because it was established earlier and never undone. To separate *establishing* the ordering from
*inheriting* it, we correlate $J$ with the change in width produced inside each interval:

```math
\Delta w_{s_1 \rightarrow s_2} \;=\; w_{s_2} - w_{s_1}, \qquad \rho^{\Delta}_{s_1 \rightarrow s_2} \;=\; \mathrm{Spearman}\bigl(J,\; \Delta w_{s_1 \rightarrow s_2}\bigr)
```

A negative $\rho^{\Delta}$ says that in *that interval* the high-divergence pairs sharpened more than
the low-divergence ones. A $\rho^{\Delta}$ near zero says the interval's sharpening was blind to
corpus statistics, whatever the cross-sectional correlation looks like. Result 3 is built on this,
and it is what makes the timing claim more than a restatement of Result 1.

**Ranking persistence $\pi(s)$.** Knowing that corpus divergence ranks the pairs at step 32 still
leaves open whether that ranking is the one the trained model ends up with. It could be a
divergence-shaped ordering the model later discards and re-derives. So we score the per-pair width
ranking at checkpoint $s$ against the final model's ranking directly:

```math
\pi(s) \;=\; \mathrm{Spearman}\bigl(w_s,\; w_{143000}\bigr)
```

$\pi = 1$ means the model already ranks the 60 pairs exactly as it finally will; $\pi = 0$ means the
current ranking tells you nothing about the final one. Because $J$ correlates with both ends, some
agreement is guaranteed by the ordering itself, so we also report the part that is *not* explained by
corpus divergence — the partial Spearman, computed by ranking $w_s$, $w_{143000}$ and $J$, regressing
the first two on $\mathrm{rank}(J)$, and correlating the residuals:

```math
\pi^{\perp}(s) \;=\; \mathrm{Spearman}\bigl(w_s,\; w_{143000} \;\big|\; J\bigr)
```

$\pi^{\perp}$ asks whether the early widths carry pair-specific information about the final ranking
beyond what $J$ already supplies. Result 8 consumes both, under the same paired bootstrap and the same
single-permutation-per-trajectory null defined below (relabelling the 60 pairs of $w_s$ against
$w_{143000}$).

**Reliability of $w$, and the ceiling it puts on $\pi$.** A near-zero $\pi$ early in training has a
boring explanation: at step 32 the entire spread of $w$ is 0.006, so $w$ might simply be too noisy
there to correlate with anything. The three carrier sentences are three independent measurements of
the same pair, so their agreement measures that noise. With $\bar r$ the mean pairwise Spearman
between per-sentence widths, the Spearman–Brown formula gives the reliability of the 3-sentence
median we actually use:

```math
\mathrm{rel}(s) \;=\; \frac{3\,\bar r_s}{1 + 2\,\bar r_s}, \qquad
\pi_{\max}(s) \;=\; \sqrt{\mathrm{rel}(s)\cdot \mathrm{rel}(143000)}
```

$\pi_{\max}$ is the largest $\pi$ attainable even if the underlying rankings were identical. Result 8
reports it next to the observed $\pi$, so a low value can be read as a real disagreement rather than
as measurement noise.

**Model output divergence.** As a co-developing check, the base-2 JSD between the model's own
next-token distributions at the two endpoints, $\mathrm{JSD}(\mathrm{softmax}(z_A),
\mathrm{softmax}(z_B))$. We track $\mathrm{Spearman}(J, \text{output JSD})$ to see when the model's
own predictions start to reflect the corpus divergence. Result 3 reports it; it is a temporal
observation, not evidence of mediation.

**Output movement along the path.** $d(t)$ compresses the whole output into one number, so it can
look sharp without the model's actual predictions being concentrated anywhere. We therefore measure
movement directly: between neighbouring interpolation positions, the JSD in bits between the model's
full next-token distributions $q(t) = \mathrm{softmax}(z(t))$ over the 50,060 valid IDs:

```math
m_j \;=\; \mathrm{JSD}\bigl(q(t_j),\, q(t_{j+1})\bigr), \qquad j = 1,\dots,49
```

The total $T = \sum_j m_j$ is how far the output travels in bits, before any normalisation. Paths
with $T < 10^{-8}$ bits would be normalised by nothing; none occurred. For the rest we normalise,
$r_j = m_j / T$, and summarise concentration by the **normalised entropy**:

```math
H(r) \;=\; -\frac{1}{\log 49}\sum_{j} r_j \log r_j
```

$H = 1$ means the output moves by the same amount at every step along the path — no concentration.
Smaller values mean the movement is bunched into a few steps. As a **location** check we also report
the movement mass inside a fixed window of width 0.2 in $t$, centred on the position where $d$
crosses 0.5. Its width does not depend on $w$, so a narrowing transition cannot inflate it
mechanically; under uniform movement it equals 0.2. Result 4 consumes both.

### Baselines and references

**Straight-line (no-plateau) reference.** The curve $d(t) = t$, i.e. output moving uniformly along
the interpolation path. It gives $w = 0.8$ and $E = 0.184$, and is the reference the "has plateau
shape yet?" test is run against.

**Untrained model (step 0).** The same 60 pairs in the randomly initialised network. It gives
median $w = 0.831$, $E = 0.213$, $\rho_0 = -0.056$, and movement entropy $1.000$ — no plateau shape
and no ordering. Every claim about training creating something is measured against this.

**Uniform-movement reference.** $H(r) = 1$ and fixed-window mass $= 0.2$, the values obtained when
output movement is spread evenly along the path.

**Endpoint-clustered inference (1,000-pair bank).** Because endpoint tokens recur across pairs there,
a bootstrap over pairs would understate uncertainty. We resample the **123 endpoint tokens** with
replacement and weight each pair by the product of its two endpoints' multiplicities (a dyadic
bootstrap), then take weighted medians and weighted Spearman correlations. Results 5 and 6 use this.

**Label-permutation null (chance reference).** Every interval above is a bootstrap: it asks how much
the statistic would wobble if we redrew the pairs. That leaves the sceptic's question unanswered at
the earliest checkpoints, where the entire ordered spread in width is 0.006 — how large a $|\rho|$
does this design produce when corpus divergence carries *no* information about width at all? A
permutation test answers exactly that, by destroying the link between divergence and width while
keeping every measured curve intact. On the 60-pair bank the pairs are endpoint-disjoint, so the
exact null relabels pairs: draw $\pi$ uniformly from the permutations of the 60 pairs and recompute

```math
\rho^{\pi}_s \;=\; \mathrm{Spearman}\bigl(J_{\pi(i)},\; w_{s,i}\bigr)
```

over $B = 20{,}000$ draws. The two-sided $p$-value counts how often chance beats the observation, with
the usual $+1$ so it can never be exactly zero:

```math
p_s \;=\; \frac{1 + \bigl|\{\, b \,:\, |\rho^{\pi_b}_s| \ge |\rho_s| \,\}\bigr|}{1 + B}
```

The same $\pi$ is applied at *every* checkpoint, so the null trajectory keeps
the across-checkpoint dependence of the real one, and the maximum of $|\rho^{\pi}_s|$ over checkpoints
gives a family-wise $p$-value that already pays for having looked at 19 of them:

```math
p^{\mathrm{fw}}_s \;=\; \Pr\Bigl(\max_{s'}\bigl|\rho^{\pi}_{s'}\bigr| \;\ge\; |\rho_s|\Bigr)
```

The same permutations, applied to $\Delta w$, test the interval statistic across the 18 intervals.

**Endpoint-label permutation for the 1,000-pair bank.** Permuting pairs there would be invalid,
because its 1,000 pairs are built from only 123 endpoint tokens and pairs sharing a token are
dependent. Instead we permute the **endpoint labels**: draw $\sigma$ over the 123 tokens and look each
pair's divergence up in the frozen $123 \times 123$ held-out divergence matrix at the relabelled
position, leaving its measured width, the pairing graph and the per-endpoint use counts untouched:

```math
\rho^{\sigma} \;=\; \mathrm{Spearman}\Bigl(J\bigl[\sigma(u_p),\, \sigma(v_p)\bigr],\; w_p\Bigr)
```

Only the correspondence between token identity and divergence is broken, so the null inherits the
bank's clustering. This is the standard quadratic-assignment permutation for dyadic data. Result 7
consumes both nulls.

**Onset rules (fixed before reading the trajectories).** The *ordering onset* bracket opens after the
last checkpoint whose simultaneous band contains zero and closes at the first of **two consecutive**
checkpoints whose band lies entirely below zero. The *shape onset* bracket uses the same two-in-a-row
requirement, with the condition that median $w$ and its band lie below 0.8 **and** median $E$ and its
band lie below 0.184.

---

## Results

All numbers below are on the 60-pair endpoint-disjoint bank at post-block-0, unless the 1,000-pair
bank is named. Brackets are 95% intervals; $\rho$ intervals in Result 1 are the simultaneous band.

### Result 1 — Corpus divergence starts ordering the pairs between step 8 and step 32

This is the report's main finding, and it is far tighter than the previous bound of "present by step
1000." To locate the onset we plot the cross-sectional correlation $\rho_s$ with its simultaneous
band, alongside the two global shape metrics, so the reader can see selectivity and shape on the same
time axis.

![Three panels: correlation, width and edge drift against training step](plots/formation_overview.png)

**Figure 1.** Onset of ordering (A) long before onset of plateau shape (B, C). x in all panels:
training step on a symmetric-log axis, so step 0 sits at the left edge next to step 1. **A** y:
Spearman $\rho$ between corpus next-token divergence $J$ and transition width $w$; the shaded band
(hatched `//`) is the simultaneous 95% band over all 20 checkpoints, the horizontal dashed line is
$\rho = 0$, and the vertical hatched stripe marks the step 8 → 32 onset bracket. **B** y: transition
width; solid-circle series is the median $w$ over the 60 pairs with its simultaneous band, the
dotted-triangle series is the interquartile range of $w$, and the dashed horizontal line is the
straight-line reference $w = 0.8$. **C** y: edge drift $E$ (dashed squares, with simultaneous band)
against its straight-line reference $E = 0.184$.

The correlation is flat and centred on zero for the first five measured checkpoints —
$\rho_0 = \rho_1 = \rho_2 = -0.056$, $\rho_4 = -0.070$, $\rho_8 = -0.060$, all with bands comfortably
spanning zero — and then drops to $\rho_{32} = -0.428$ $[-0.753, -0.104]$, the first checkpoint whose
band excludes zero. It stays below zero at every one of the remaining 14 checkpoints, ending at
$\rho_{143000} = -0.525$ $[-0.849, -0.200]$, which reproduces the upstream final-checkpoint value
exactly. By step 32 the effect has reached 82% of its final magnitude.

Panels B and C show what makes this surprising: at step 32 **nothing has sharpened**. Median
$w = 0.827$, against 0.831 in the untrained model and 0.8 for a straight line; median $E = 0.209$,
*above* the straight-line 0.184. The model has no plateaus, and it has already sorted the pairs.

The ordering at that moment lives in an extremely small dynamic range, and we show it rather than
smoothing it away: the interquartile range of $w$ at step 32 is **0.008**, and the five
corpus-divergence quintile medians are 0.8298, 0.8279, 0.8275, 0.8267, 0.8241 — perfectly monotone in
$J$, spanning 0.0057 in total. A rank statistic is the right tool for exactly this situation, but the
honest reading is that the ordering is real and reliable while the thing being ordered is, at step 32,
a set of near-identical curves. What Result 3 adds is that this tiny early ordering is not a
coincidence of the ranking: the *change* in width during that interval is itself divergence-ordered.
Result 8 adds the complementary limit — the step-32 ranking is not yet the final ranking in detail,
only in its divergence-aligned part.

### Result 2 — Global plateau shape appears between step 1000 and step 2000, ~60× later

Figure 1B and 1C also carry the second onset. Median $w$ sits within 0.02 of the untrained value
through step 512 (0.831, 0.831, 0.831, 0.831, 0.829, 0.827, 0.827, 0.837, 0.834, 0.814), then falls
to 0.753 at step 1000, 0.680 at 2000, 0.639 at 4000, and on down to 0.512 at 64000. Edge drift
tracks it, falling from 0.217 at step 512 to 0.153 at 1000, 0.117 at 2000 and 0.069 at 64000 —
well below the straight-line 0.184, confirming that the narrowing transitions really do have flat
ends and are not just steep-in-the-middle ramps.

Applying the prespecified shape rule: at step 1000 the median width's simultaneous band still reaches
0.805, just failing the 0.8 threshold; at steps 2000 and 4000 both $w$ (bands up to 0.732 and 0.691)
and $E$ (bands up to 0.147 and 0.131) clear their references. The shape onset bracket is therefore
**after step 1000, by step 2000**, against **after step 8, by step 32** for the ordering — a
separation of roughly two orders of magnitude in training steps.

### Result 3 — The largest single sharpening event is blind to corpus statistics

A cross-sectional correlation cannot tell an ordering that is being *created* from one that is merely
being *carried forward*. To separate them we ask, for each interval between adjacent checkpoints,
whether corpus divergence predicts the width change produced *inside that interval*.

![Three panels: width by divergence quintile, interval correlations, cumulative correlations](plots/interval_sharpening.png)

**Figure 2.** Where in training the divergence-selectivity is actually generated. x in all panels:
training step (symmetric-log). **A** y: median transition width within each corpus-divergence
quintile, Q1 (lowest $J$, solid circles) through Q5 (highest $J$, dashed triangles); the legend gives
each quintile's median $J$ in bits. **B** y: Spearman correlation between $J$ and the width change
$\Delta w$ produced within the interval ending at that step, with pointwise 95% bootstrap bars;
dashed line at zero. **C** y: Spearman correlation of $J$ with the cumulative change since step 0
(solid circles) and with the model's own endpoint output divergence (dashed squares).

The interval test independently confirms the step 8 → 32 onset: in that interval
$\rho^{\Delta} = -0.466$ $[-0.663, -0.223]$, even though the median width change is only $-0.0011$.
Corpus divergence predicts which pairs move, at a point when almost nothing moves.

The dissociation appears at the other end. The single largest global sharpening happens between step
512 and step 1000 — median $\Delta w = -0.0618$ $[-0.0721, -0.0537]$, Wilcoxon $p = 1.9\times10^{-11}$,
a change 56× larger than the step 8 → 32 interval produced — and in that interval corpus divergence
predicts nothing: $\rho^{\Delta} = +0.035$ $[-0.241, +0.307]$. The same pattern holds for step
4000 → 8000 ($\rho^{\Delta} = +0.258$, median $\Delta w = -0.033$; not distinguishable from chance
once all 18 intervals are accounted for, see Result 7) and step 16000 → 32000
($\rho^{\Delta} = +0.119$, median $\Delta w = -0.024$). Selectivity does reappear in some later
intervals — step 256 → 512 ($-0.439$), step 1000 → 2000 ($-0.267$), step 32000 → 64000 ($-0.540$) —
so the process is not confined to the first 32 steps, but the bulk sharpening events themselves are
divergence-blind.

Figure 2A makes the same point visually: all five quintile trajectories start on top of each other
near 0.831, fan out slightly and in the correct order by step 32, and then all five fall together
through the step-1000 sharpening while keeping their relative order.

Figure 2C adds the timing of a third quantity. The correlation between corpus divergence and the
model's *own* endpoint output divergence is 0.145 at step 0, still only 0.253 at step 32 where the
ordering onset occurs, and reaches 0.720 by step 512 and 0.798 by step 1000. The model's predictions
come to reflect corpus divergence on the same timescale as global sharpening, well after the width
ordering is in place. Temporal order alone does not establish a mechanism, and we do not claim one.

### Result 4 — Output movement concentrates onto the boundary, on the shape timescale

Narrow $d(t)$ could in principle be an artefact of the distance summary. The direct question is
whether the model's full next-token distribution actually stops changing away from the boundary. To
answer it we track how the total output movement is distributed along the path.

![Three panels: movement entropy, window mass, total movement and loss](plots/output_movement_formation.png)

**Figure 3.** Movement concentration develops with plateau shape, not with the ordering. x in all
panels: training step (symmetric-log). **A** y: normalised entropy $H(r)$ of the movement profile
with its simultaneous band; the dashed line at 1 is the uniform-movement reference and lower means
more concentrated. **B** y: fraction of total movement inside the fixed 0.2-wide window centred on the
$d = 0.5$ crossing (dashed squares, with band); the dashed line at 0.2 is the uniform expectation.
**C** left y (solid circles, log scale): median total movement $T$ in bits; right y (dotted
triangles): held-out next-token loss in nats.

Movement is exactly uniform in the untrained model ($H = 1.000$, window mass $0.200$) and stays that
way through step 512 ($H = 0.971$, mass $0.272$). It then concentrates steadily: $H = 0.917$ and mass
$0.382$ at step 1000, $H = 0.824$ and mass $0.583$ at 2000, $H = 0.690$ and mass $0.828$ at 8000,
reaching $H = 0.630$ and mass $0.900$ at the final checkpoint. By the end of training, **90% of the
model's entire output change along the path happens in the middle 20% of it.** This answers the
question the plan posed with a clear yes, and it puts movement concentration on the *shape* timeline
(onset around step 1000–2000), not the ordering timeline.

Figure 3C shows that concentration is not the model simply moving less: total movement *grows* by two
orders of magnitude, from 0.0016 bits at step 0 to 0.135 bits at the end. The model moves much
further and does it in a much smaller region. The held-out loss on the same axis falls monotonically
from 11.010 to 2.245 nats and gives the training-progress context for every timing claim here.

The shape of the profile itself, in Figure 4, is the most direct picture of the phenomenon.

![Median movement profile against position relative to the d=0.5 crossing, at five checkpoints](plots/movement_profiles.png)

**Figure 4.** From a flat profile to a spike. x: interpolation position relative to the $d = 0.5$
crossing, $t - t_{50}$, so 0 is the boundary. y: median normalised movement $r_j$ across all 60 pairs
and 3 carrier sentences. The five series are checkpoints — step 0 (solid circles), 128 (dashed
squares), 1000 (dotted triangles), 8000 (dash-dot diamonds) and 143000 (long-dash triangles). At steps
0 and 128 the profile is flat at $1/49 \approx 0.020$; by step 143000 a single step at the crossing
carries 0.17 of the total movement, roughly 8× the uniform value.

### Result 5 — The late widening is real, and reproduces on 1,000 independent pairs

Median width does not decrease monotonically to the end: it bottoms out at 0.512 at step 64000 and
rises to 0.517, 0.528 and 0.541 at steps 96000, 128000 and 143000. A median over 60 pairs is easy to
move by chance, so this was flagged upstream as possibly noise. We re-test it on the frozen
1,000-pair bank (Figure 5), with inference that accounts for the reuse of endpoint tokens.

![Two panels: median width at two checkpoints for both banks, and the paired change with CIs](plots/large_bank_confirmation.png)

**Figure 5.** The late reversal survives on the larger bank. **A** x: the two checkpoints, step 64000
and step 143000; y: median transition width $w$. Solid circles are the 60-pair controlled set, dashed
squares the 1,000-pair set. **B** x: median paired change $\Delta w$ from step 64000 to step 143000,
with 95% intervals; positive means blunter at the end of training; the dashed vertical line is zero.
y lists the two banks and the resampling unit used for each — pairs for the controlled set, the 123
endpoint tokens for the 1,000-pair set.

On the 60-pair set, median $\Delta w = +0.0121$ $[+0.0016, +0.0259]$, with 38 of 60 pairs blunter
(paired Wilcoxon $p = 0.0052$). On the 1,000-pair set, median $\Delta w = +0.0158$
$[+0.0081, +0.0224]$ with 65.1% of pairs blunter (95% CI 57.6% to 71.8%) — a tighter interval, in the same
direction, under the clustered bootstrap that is the conservative choice for that bank. The reversal
is a property of this training run, not of the small bank.

Two things coincide with the late widening and are worth recording without over-reading. The learning
rate is in the tail of its cosine decay ($1.26\times10^{-4}$ at step 64000 down to $2.0\times10^{-5}$
at the end), and our held-out loss also stops improving over the same window, rising from 2.245 nats
at step 96000 to 2.314 at 128000 and 2.322 at the end. Whether these are causally linked is not
something a single training run can answer.

### Result 6 — The step 8 → 32 onset bracket replicates on the 1,000-pair bank

The onset in Result 1 rests on 60 pairs, so the plan required re-running the two checkpoints that
*define* the bracket on the larger bank. We assayed all 1,000 pairs at step 8 and step 32 and applied
the same endpoint-clustered bootstrap; Figure 6 puts both banks on one axis.

![Correlation with 95% intervals at five checkpoints for both banks](plots/large_bank_onset.png)

**Figure 6.** The bracket survives on an independent, 17× larger bank. x: the five checkpoints
measured on both banks; y: Spearman $\rho$ between corpus divergence and transition width, with 95%
intervals. Circles are the 1,000-pair set (interval from the dyadic bootstrap over its 123 endpoint
tokens); squares are the 60-pair controlled set (bootstrap over pairs, shown pointwise here). The
hatched stripe marks the step 8 → 32 onset bracket; the dashed line is $\rho = 0$.

The large bank places the onset in the same interval: $\rho = -0.021$ $[-0.132, +0.104]$ at step 8,
which contains zero, and $\rho = -0.149$ $[-0.286, -0.011]$ at step 32, which excludes it. It also
reproduces the endpoints of the trajectory — $-0.008$ $[-0.117, +0.115]$ at step 0, $-0.563$
$[-0.668, -0.438]$ at step 64000, $-0.486$ $[-0.617, -0.354]$ at step 143000 — and shows no
sharpening at step 32 either (median $w = 0.828$, IQR 0.007, against 0.831 at step 0).

The effect size at step 32 is markedly smaller on the large bank ($-0.149$ vs $-0.428$), while by the
final checkpoint the two banks nearly agree ($-0.486$ vs $-0.525$). That gap is worth stating
plainly: the 1,000-pair bank is not matched on frequency or surprisal, reuses endpoint tokens, and
fills the crowded middle of the divergence range, so it is the harder test and it dilutes a weak
early signal. The bracket replicates; the *magnitude* of the step-32 ordering should be read from the
controlled bank, with the large bank establishing that the timing is not an artefact of 60 pairs.

### Result 7 — Chance never produces this ordering, on either bank

Every interval so far is a bootstrap, which measures wobble rather than chance. The claim that most
needs a chance reference is the earliest one: at step 32 the ordering lives on a width spread of
0.006, and a reader is entitled to ask whether rank structure that fine is simply what *any* labelling
of 60 pairs would give. We therefore recomputed each statistic under 20,000 label permutations, and
under the endpoint-label permutation for the clustered bank; Figure 7 shows both nulls.

![Three panels: observed correlations against permutation null envelopes for both banks](plots/permutation_null.png)

**Figure 7.** The observed ordering sits far outside what relabelling produces, and the two
divergence-blind intervals sit inside it. x in all panels: training step (symmetric-log). y in **A**
and **C**: Spearman $\rho$ between corpus divergence $J$ and transition width $w$; y in **B**:
$\rho$ between $J$ and the within-interval width change $\Delta w$, plotted at the interval's end
step. In every panel the hatched band bounded by dotted lines is the pointwise 95% envelope of
$|\rho|$ under the null, and the solid-circle series is the observed value. **A** (60-pair bank) adds
the dashed horizontal lines at $\pm 0.353$: the *simultaneous* null envelope covering all 19
checkpoints at once. The vertical hatched stripe in A and C marks the step 8 → 32 onset bracket;
**C** (1,000-pair bank, endpoint-label null) is annotated with each checkpoint's two-sided $p$.

The onset survives the strictest form of the test. Through step 8 the observed correlation lies well
inside the null envelope ($p = 0.67$, $0.67$, $0.67$, $0.60$, $0.65$ at steps 0, 1, 2, 4, 8); at step
32 it is outside it, $p = 0.0007$, and the family-wise value that pays for all 19 checkpoints at once
is $p^{\mathrm{fw}} = 0.0072$. Every later checkpoint has $p^{\mathrm{fw}} \le 0.013$. Chance
labellings of 60 pairs reach $|\rho| = 0.26$ pointwise and $0.35$ simultaneously, against an observed
$0.428$ — so the step-32 ordering is not what fine-grained noise looks like, even though the curves it
orders are nearly identical.

The interval statistic separates the two regimes just as sharply, and this is the test the
dissociation rests on. Step 8 → 32 gives $\rho^{\Delta} = -0.466$ with $p = 0.0003$ and
$p^{\mathrm{fw}} = 0.0035$ across all 18 intervals, while the largest sharpening event, step
512 → 1000, gives $p = 0.78$ — indistinguishable from a random relabelling. The interval that moves
width the most is, by this test, exactly as divergence-selective as chance.

The permutation also corrects one reading of Result 3. The positive $\rho^{\Delta} = +0.258$ at step
4000 → 8000 has a bootstrap interval that just excludes zero, but its permutation $p = 0.045$ does not
survive the 18-interval correction ($p^{\mathrm{fw}} = 0.55$). That interval is best described as
divergence-blind rather than as reversed selectivity; the four intervals that do survive correction
are step 8 → 32, 256 → 512, 1000 → 2000 and 32000 → 64000, all negative.

On the 1,000-pair bank the endpoint-label null does something the bootstrap could not: it prices the
clustering directly. Because those 1,000 pairs come from only 123 tokens, relabelling produces
$|\rho|$ up to 0.09 by chance — half again the 0.062 that 1,000 genuinely independent pairs would
give. Measured against that inflated bar, step 0 ($p = 0.87$) and step 8 ($p = 0.64$) are null, step
32 is not ($\rho = -0.149$, $p = 0.0031$, $p^{\mathrm{fw}} = 0.0082$), and both late checkpoints have
$p < 0.001$. The bracket that Result 6 established with a clustered bootstrap therefore holds under a
second, independent form of inference that makes no distributional assumption at all — which matters
because $-0.149$ was the weakest number in this report.

### Result 8 — At step 32 the model holds only the divergence-aligned part of the final ranking

Results 1 and 7 establish that corpus divergence orders the pairs at step 32. They do not establish
that this is the *same* ordering the trained model ends up with, and a reader should not assume it:
the step-32 widths span 0.006, and an ordering that fine could be discarded and re-derived later
without leaving a trace in $\rho_s$. Figure 8 tests it directly, by scoring every checkpoint's
per-pair ranking against the final model's.

![Two panels: persistence of the width ranking against training step, and the checkpoint-by-checkpoint agreement matrix](plots/ranking_persistence.png)

**Figure 8.** The width ranking locks in between step 64 and step 128 — after the ordering, before
the shape. **A** x: training step (symmetric-log). y: rank agreement with the final checkpoint's
ranking of the 60 pairs. Solid circles are $\pi(s)$, dashed squares are the partial version
$\pi^{\perp}(s)$ with corpus divergence $J$ removed, both with pointwise 95% bootstrap bars; the
hatched band between dotted lines is the pointwise 95% envelope of $|\pi|$ under 20,000 pair
relabellings; the dash-dot gray line is the attenuation ceiling $\pi_{\max}$ set by the reliability
of $w$. The left vertical stripe (`\\` hatch) is the step 8 → 32 divergence-ordering bracket, the
right one (`xx` hatch) the step 64 → 128 ranking bracket. **B** x and y: the 20 checkpoints in
training order (index spacing, not to scale); colour: Spearman $\rho$ between the per-pair widths at
those two checkpoints, on the `cividis` scale at right. The dashed white lines mark step 128.

Persistence is flat and inside the chance envelope for every checkpoint through step 64:
$\pi = +0.109$ $[-0.169, +0.373]$ at step 0, $+0.121$ at step 8, $+0.161$ $[-0.089, +0.405]$ at step
32 ($p = 0.21$), $+0.207$ at step 64 ($p = 0.11$). It then jumps to $+0.437$ $[+0.202, +0.623]$ at
step 128 — the first checkpoint outside the envelope, $p = 0.0007$ and $p^{\mathrm{fw}} = 0.0053$
across all 19 checkpoints — and climbs steadily after that: $0.532$ at 256, $0.696$ at 512, $0.788$
at 1000, and above $0.85$ from step 4000 on.

The partial version is what makes this a statement about the *ordering* rather than about widths in
general. At step 32, $\pi^{\perp} = -0.082$ $[-0.317, +0.177]$, $p = 0.53$: once corpus divergence is
removed, the step-32 ranking and the final ranking have nothing in common. The observed $\pi = 0.161$
is close to the $(-0.428)\times(-0.525) = 0.225$ that the two divergence correlations alone imply. So
the model at step 32 holds the divergence-aligned component of the final ordering and no more.
Pair-specific detail beyond $J$ first clears the family-wise bar at step 256
($\pi^{\perp} = +0.380$, $p^{\mathrm{fw}} = 0.0275$; at step 128 it is $+0.238$ with
$p^{\mathrm{fw}} = 0.39$).

The obvious objection is attenuation — that $w$ at step 32 is too noisy to agree with anything — and
it does not hold. Across three unrelated carrier sentences, per-pair widths agree at
$\bar r = 0.830$ at step 32, giving a reliability of 0.936 and a ceiling of $\pi_{\max} = 0.935$.
Width at step 32 is a reliable measurement of a stable pair property, it correlates with corpus
divergence at $-0.428$, and it still shares only 0.161 of its ranking with the final model. The same
holds at step 0, where reliability is already 0.872: even the untrained network ranks the pairs
consistently across sentences, and that ranking is not the final one.

Figure 8B shows the same thing as structure rather than as a trajectory. Checkpoints from step 0
through step 64 form a block that agrees with itself ($\rho(w_0, w_{32}) = 0.435$) and disagrees with
everything later; from step 128 on, every checkpoint agrees with every later one, with the agreement
rising smoothly towards the diagonal. Step 128 is where the model stops rearranging which pairs are
sharp relative to each other.

This adds a third clock, and it tightens rather than weakens the report's main claim. The three
events are ordered: divergence starts selecting pairs (step 8 → 32), the pair ranking becomes the
final one (step 64 → 128), and only much later do the transitions actually become plateaus
(step 1000 → 2000). Corpus divergence is not merely present before the shape — it is the *first*
component of the final ordering to appear, ahead of the pair-specific detail that fills in around it.

### Result 9 — The released `step16` revision of Pythia-1.4B-deduped is not a step-16 model

The scan surfaced a data-integrity problem that anyone studying Pythia's early checkpoints should
know about. We report it because it would silently corrupt exactly the kind of early-training
analysis this report performs; Figure 9 shows how it stands out.

![Held-out loss against training step with step16 marked as an excluded outlier](plots/checkpoint_qc.png)

**Figure 9.** One released revision breaks the loss trajectory. x: training step (symmetric-log);
y: held-out next-token loss in nats on the frozen 256-row sample. The connected circles are the 19
checkpoints we keep; the large cross is revision `step16`.

The evidence is threefold. Its held-out loss is 2.320 nats, while step 8 gives 9.889 and step 32
gives 8.824 — a 6.5-nat improvement and 6.5-nat regression inside 24 steps at a learning rate of
$2\times10^{-6}$, which no optimiser trajectory produces. Its 60 pairs × 3 sentences × 50 positions
= 9,000 measured $d(t)$ values are **bit-identical** to those of `step143000` (maximum absolute
difference exactly $0$), as are its 8,820 movement values. And its `model.safetensors` is 2,829,329,888
bytes where all 19 other revisions we queried are 2,829,329,920. The artefact served under `step16`
is the final model. We excluded it from every trajectory in this report; its assay output is retained
in `results/assay_step16.json` and the checks in `results/ckpt_qc.json`.

### Summary of the onsets

The table below collects the timing verdicts. Each row is an event, the bracket the prespecified rule
returned, the statistic that moved, and what the *other* phenomena were doing at the same moment —
which is the comparison the whole report turns on. Read down the first column: the three onsets are
separated, and they run in the order divergence-selection → pair ranking → plateau shape.

| Event | Onset bracket | Statistic at onset | State of the other phenomena |
|---|---|---|---|
| Divergence-selective ordering | after step 8, by step 32 | $\rho_{32} = -0.428$ $[-0.753, -0.104]$, permutation $p^{\mathrm{fw}} = 0.0072$; interval $\rho^{\Delta}_{8\to32} = -0.466$ $[-0.663, -0.223]$, $p^{\mathrm{fw}} = 0.0035$ | no sharpening: median $w = 0.827$ vs 0.831 untrained, IQR 0.008, $E = 0.209$ above the straight line; ranking not yet final ($\pi = 0.161$) |
| Pair ranking becomes final | after step 64, by step 128 | $\pi = +0.437$ $[+0.202, +0.623]$, $p^{\mathrm{fw}} = 0.0053$, against a ceiling of 0.95 | still no sharpening: median $w = 0.837$, $E = 0.222$, both above the straight-line reference |
| Global plateau shape | after step 1000, by step 2000 | median $w = 0.680$ $[\text{band} \le 0.732]$, $E = 0.117$ $[\text{band} \le 0.147]$ | ordering already 2,000 steps old and near its final value; ranking already at $\pi = 0.82$ |
| Movement concentration | with the shape (step 1000–2000) | $H = 0.824$, window mass 0.583 at step 2000 | uniform ($H = 1.000$, mass 0.200) at step 32 when ordering appeared |
| Late widening | step 64000 → 143000 | 60-pair $+0.0121$ $[+0.0016, +0.0259]$; 1,000-pair $+0.0158$ $[+0.0081, +0.0224]$ | ordering persists ($\rho = -0.525$) |

The same bracket on the 1,000-pair bank, under the endpoint-label null that prices in its token
reuse: $p = 0.87$ at step 0, $p = 0.64$ at step 8, $p = 0.0031$ at step 32, $p < 0.001$ at both late
checkpoints.

---

## Conclusion

In this Pythia-1.4B-deduped run, learning *which* activation-space boundaries should be sharp and
learning *to make them sharp* are separate processes that happen ~60× apart in training time. Corpus
next-token divergence ranks the pairs correctly by step 32 — inside the learning-rate warmup, when
median transition width is 0.827 against 0.831 in the untrained model and the whole ordered spread is
0.006. Plateau shape arrives between step 1000 and step 2000, and the concentration of the model's
full output movement onto the boundary arrives with it, reaching 90% of movement inside 20% of the
path by the end of training. The interval that does the most sharpening (step 512 → 1000) sorts
pairs by corpus divergence not at all.

Between those two events sits a third. The per-pair ranking of widths only becomes the final ranking
between step 64 and step 128; at step 32 it agrees with the final model at $\pi = 0.161$, and at
nothing once corpus divergence is partialled out. So the order of assembly is: corpus divergence
selects first, pair-specific detail fills in around it, and the geometry that makes any of it visible
as a plateau comes last.

Read together, these say the corpus-divergence correlate reported upstream is not a side effect of
plateaus forming. It is present before there is anything to be a side effect of, it is the first part
of the final ordering to appear, and it survives a sharpening process that is itself largely
indifferent to it. For interpretability practice, the useful consequence is that the *ranking* of
which pairs will end up with sharp boundaries is available very early in training — by step 128, a
thousandth of the way through — while the *magnitude* of the boundary is not, so an intervention
budget aimed at plateau structure can be targeted long before the structure exists.

**What this does not show.** (i) One training run of one model. An onset bracket measured here is not
a universal training law, and we deliberately avoid the phrase "phase transition": the prespecified
rule for that language requires reproducing an abrupt change on an independent run, which we have
not done. (ii) The endpoint pairs were filtered partly by what the *final* model considers a
plausible continuation, so every timing claim is conditional on endpoints the fully trained model
finds plausible; a bank selected by an early checkpoint could behave differently. (iii) Temporal
order is not causality. The model's own output divergence catches up to corpus divergence on the
shape timescale, but we do not claim it mediates anything. (iv) The corpus divergence is an
immediate-next-token, context-averaged quantity, and it is not measured along the interpolation path;
it does not license the claim that each plateau corresponds to one continuation distribution.
(v) The step-32 ordering, though it clears both a simultaneous bootstrap band and a family-wise
permutation test on both banks, sits on a width spread of 0.006 — a real rank effect on nearly
identical curves, and it should be described that way.

**Reproducibility.** All code is in `experiments/`; `scan.py` drives the checkpoint scan (one
checkpoint on disk at a time), `run_assay.py` measures one checkpoint, `analyze.py` produces
`results/checkpoint_metrics.json`, `ckpt_qc.py` produces `results/ckpt_qc.json`, `large_late.py`
produces `results/large_late.json`, `permtest.py` produces `results/permutation.json`,
`persistence.py` produces `results/persistence.json`, and `plot_formation.py`, `plot_perm.py` and
`plot_persistence.py` produce every figure above. The frozen
pair manifests, corpus manifests and inherited upstream results were copied unmodified from
`dir18_continuation_jsd_plateau` and their SHA-256 hashes are recorded in
`results/INHERITED_HASHES.txt`. Re-running the assay at step 0 reproduced the upstream curves
bit-for-bit (maximum absolute difference $0$ over all 9,000 values), and our final-checkpoint
$\rho = -0.525$ matches the upstream value to four decimals. Raw 50-point $d(t)$ curves and 49-point
movement profiles are saved per checkpoint in `results/`.
