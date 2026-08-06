# RESULTS — When does the corpus-divergence / plateau relationship form during training?

> CURRENT-BEST ONLY. One row per experiment. No history (that lives in `CHANGELOG.md`).
> Full write-up with all definitions: `REPORT.md`.

## Headline

Scanning 20 released checkpoints of **Pythia-1.4B-deduped** (steps 0 → 143000) with one frozen
60-pair bank at the post-block-0 residual stream, the phenomena are **separated by ~60× in
training time, and in the counter-intuitive order**:

- **Corpus next-token divergence starts ranking pairs by sharpness between step 8 and step 32**, when
  the model has *no plateaus at all* (median transition width 0.827 vs 0.831 untrained, IQR 0.008).
- **The per-pair ranking becomes the final one between step 64 and step 128** — a third clock. At
  step 32 the model holds only the divergence-aligned part of the final ordering.
- **Plateau shape appears between step 1000 and step 2000**, and the single largest sharpening event
  (step 512 → 1000) is completely blind to corpus divergence.

Notation: $J$ = corpus next-token Jensen–Shannon divergence of a token pair (bits, held-out split);
$w$ = transition width, $t(d{=}0.9) - t(d{=}0.1)$, **smaller = sharper**; $E$ = edge drift, **smaller
= flatter ends**; $\rho$ = Spearman rank correlation, **negative = higher divergence gives sharper
boundaries**; $\pi$ = rank agreement of a checkpoint's per-pair widths with the final checkpoint's.
A no-plateau straight line $d(t)=t$ gives $w = 0.8$, $E = 0.184$.

## Metrics — current best

The scan's primary output is the pair of onset brackets below. Each row names the event, the bracket
the prespecified rule returned, the statistic at onset, and what the other phenomenon was doing at
that moment — the last column is the contribution.

| Event | Onset bracket | Statistic at onset | State of the other phenomena |
|---|---|---|---|
| Divergence-selective ordering | after step 8, **by step 32** | $\rho_{32} = -0.428$, simultaneous band $[-0.753, -0.104]$, label-permutation $p^{\mathrm{fw}} = 0.0072$ | no sharpening: median $w = 0.827$ (untrained 0.831), IQR($w$) 0.008, $E = 0.209$ *above* the straight line; ranking not yet final ($\pi = 0.161$) |
| Pair ranking becomes final | after step 64, **by step 128** | $\pi = +0.437$ $[+0.202, +0.623]$, $p^{\mathrm{fw}} = 0.0053$, ceiling 0.95 | still no sharpening: median $w = 0.837$, $E = 0.222$ |
| Global plateau shape | after step 1000, **by step 2000** | median $w = 0.680$ (band $\le 0.732$), $E = 0.117$ (band $\le 0.147$) | ordering established ~2,000 steps earlier; ranking already final ($\pi = 0.82$) |
| Output-movement concentration | with the shape (step 1000–2000) | $H = 0.824$, fixed-window mass 0.583 at step 2000 | uniform ($H = 1.000$, mass 0.200) at step 32 when ordering appeared |
| Late widening (reversal) | step 64000 → 143000 | 60-pair median $\Delta w = +0.0121$ $[+0.0016, +0.0259]$ | ordering persists, $\rho = -0.525$ |

The interval test separates *creating* the ordering from *inheriting* it, and it is what makes the
timing claim more than a restatement of the cross-sectional correlation.

| Interval | median $\Delta w$ (change in width) | $\rho(J, \Delta w)$ | permutation $p$ / $p^{\mathrm{fw}}$ | reading |
|---|---|---|---|---|
| step 8 → 32 | $-0.0011$ | $-0.466$ $[-0.663, -0.223]$ | 0.0003 / 0.0035 | almost no sharpening, but strongly divergence-selective |
| step 512 → 1000 | $-0.0618$ $[-0.0721, -0.0537]$ | $+0.035$ $[-0.241, +0.307]$ | 0.78 / 1.00 | the largest sharpening event, entirely divergence-blind |
| step 4000 → 8000 | $-0.0328$ | $+0.258$ $[+0.002, +0.483]$ | 0.045 / 0.55 | large sharpening; does not survive the 18-interval correction |
| step 32000 → 64000 | $-0.0358$ | $-0.540$ $[-0.697, -0.332]$ | <0.0001 / 0.0001 | selectivity does reappear later |

$p^{\mathrm{fw}}$ is the family-wise permutation $p$-value, which pays for having examined all 18
intervals. It changes one reading: the positive $\rho$ at step 4000 → 8000 has a bootstrap interval
that just excludes zero but is not distinguishable from a random relabelling, so that interval is
divergence-**blind** rather than reversed.

Validation on the frozen **1,000-pair** bank (123 endpoint tokens, so uncertainty comes from a dyadic
bootstrap over endpoints, not pairs) reproduces both the ordering and the late reversal.

| 1,000-pair bank | step 0 | step 8 | step 32 | step 64000 | step 143000 |
|---|---|---|---|---|---|
| $\rho(J, w)$, endpoint-clustered 95% CI | $-0.008$ $[-0.117, +0.115]$ | $-0.021$ $[-0.132, +0.104]$ | $-0.149$ $[-0.286, -0.011]$ | $-0.563$ $[-0.668, -0.438]$ | $-0.486$ $[-0.617, -0.354]$ |
| endpoint-label permutation $p$ | 0.87 | 0.64 | **0.0031** | <0.0001 | <0.0001 |
| median $w$ | 0.831 | 0.830 | 0.828 | 0.537 | 0.555 |

The onset bracket replicates: the interval containing zero closes and the interval excluding zero
opens between the same two checkpoints, step 8 and step 32, on a bank 17× larger. The effect size at
step 32 is smaller there ($-0.149$ vs $-0.428$) because that bank is unmatched on frequency and
surprisal, reuses endpoint tokens, and fills the crowded middle of the divergence range — it is the
harder test, and it dilutes a weak early signal. Read the timing from both banks and the magnitude
from the controlled one.

Paired across those last two checkpoints, the 1,000-pair bank gives median $\Delta w = +0.0158$
$[+0.0081, +0.0224]$ with 65.1% of pairs blunter (95% CI 57.6% to 71.8%) — the same direction as the
60-pair set, with a tighter interval.

**How much of the final answer is present at step 32?** Only its divergence-aligned part. Rank
agreement between a checkpoint's per-pair widths and the final checkpoint's, $\pi$, stays inside the
chance envelope through step 64 and jumps at step 128.

| Persistence | step 0 | step 8 | step 32 | step 64 | step 128 | step 256 | step 1000 |
|---|---|---|---|---|---|---|---|
| $\pi$ (agreement with final ranking) | $+0.109$ | $+0.121$ | $+0.161$ | $+0.207$ | $\mathbf{+0.437}$ | $+0.532$ | $+0.788$ |
| permutation $p$ / $p^{\mathrm{fw}}$ | 0.40 / 0.97 | 0.35 / 0.95 | 0.21 / 0.80 | 0.11 / 0.55 | **0.0007 / 0.0053** | <0.0001 / 0.0001 | <0.0001 / <0.0001 |
| $\pi^{\perp}$, corpus divergence removed | $+0.094$ | $+0.105$ | $-0.082$ | $-0.023$ | $+0.238$ | $+0.380$ | $+0.698$ |
| reliability ceiling $\pi_{\max}$ | 0.902 | 0.898 | 0.935 | 0.955 | 0.953 | 0.941 | 0.951 |

At step 32 the ranking shares nothing with the final ranking once $J$ is partialled out
($\pi^{\perp} = -0.082$, $p = 0.53$), and the observed $\pi = 0.161$ is what the two divergence
correlations alone imply ($(-0.428)\times(-0.525) = 0.225$). This is not measurement noise: the three
carrier sentences agree on each pair's width at step 32 at $\bar r = 0.830$, so $\pi$ could have
reached 0.935.

**Robustness to the reference checkpoint.** "Final" here means step 143000, which is where the
released trajectory stops rather than where the model settles — and Result 5 shows the widths still
move late. Rescoring the whole persistence analysis against step 8000, 32000, 64000 and 128000
returns the **same step 64 → 128 bracket in every case**: $\pi_{128}$ is $+0.447$, $+0.394$, $+0.430$,
$+0.410$ against those four references (family-wise $p$ 0.0045 to 0.018) while step 64 stays
non-significant ($p^{\mathrm{fw}}$ 0.47 to 0.89), and $\pi_{32}$ stays inside the chance envelope
($+0.077$ to $+0.200$, $p \ge 0.13$) with $\pi^{\perp}_{32} \le 0$ throughout.

**Data-integrity finding.** The artefact Hugging Face serves as revision **`step16`** of
`EleutherAI/pythia-1.4b-deduped` is **not a step-16 model — it is `step143000`**. Held-out loss is
2.320 nats against 9.889 at step 8 and 8.824 at step 32; its 9,000 measured $d(t)$ values are
bit-identical to `step143000`'s; and streaming the 2.63 GiB tensor payload from the Hub gives the
identical SHA-256 (`fbd54ccec4e0f5ee…`), with all 10 individually hashed tensors matching
`step143000` byte for byte and none matching `step8` or `step32`. Only the packaging differs: its
header omits the `__metadata__` field, making it 32 bytes shorter. Auditing all 21 revisions used
here, the other 20 share one byte-identical header layout and have 20 distinct payload digests, so
this is the only affected revision. It is excluded from every trajectory here, which is also why the
ordering bracket cannot be narrowed below step 8 → 32: no genuine step-16 weights are published.

**Robustness to the width definition.** Every number here uses $w = t(0.9) - t(0.1)$, whose levels
are a convention. Recomputing the whole scan with $w_a = t(1-a) - t(a)$ for
$a \in \lbrace 0.10, 0.15, 0.20, 0.25, 0.30\rbrace$ (straight-line reference $1-2a$; curve validity
held fixed at the original rules so the same curves enter every trajectory) leaves the ordering
bracket at **step 8 → 32 for all five**, with $\rho_{32}$ between $-0.428$ and $-0.385$ and
$\rho(J, \Delta w)$ over step 8 → 32 between $-0.466$ and $-0.452$. The shape bracket moves one
checkpoint earlier (step 512 → 1000) for $a \ge 0.20$, which weight only the steep middle of the
curve, so the separation is 31× rather than 62× there. Over step 512 → 1000, $\rho(J, \Delta w)$ is
$+0.035$, $+0.142$, $+0.229$, $+0.275$, $+0.312$ — never negative, so the largest sharpening event is
divergence-blind under every definition.

**Robustness to the carrier sentences.** Every width is a median over three fixed sentence frames,
which could turn one frame's quirk into an apparent training-time fact. Recomputing each pair's width
from a single frame, with no averaging, and re-running all three onset rules on each frame separately
returns the **step 8 → 32 ordering bracket and the step 1000 → 2000 shape bracket in all three**
($\rho_{32} = -0.363, -0.442, -0.359$, each with a simultaneous band excluding zero; median $w$ at
step 32 = 0.826, 0.827, 0.828). Two of the three also return the step 64 → 128 ranking bracket
($\pi_{128} = +0.504$ and $+0.388$); the third closes one checkpoint later at step 256, which is what
attenuation predicts — a single frame can reach at most $\pi_{\max} = 0.871$ at step 128 against
0.953 for the median of three. No frame changes the order of the three events.

**Quality controls.** All 3,600 curves across the 20-checkpoint scan passed the strict validity
criteria (valid-curve rate 1.000 at every checkpoint). Endpoint patching reproduces the unpatched
logits to a maximum relative error of $4.6\times10^{-5}$. Re-running step 0 reproduced the upstream
dir18 curves **bit-for-bit**, and our final-checkpoint $\rho = -0.525$ matches upstream to four
decimals. No path had total output movement below the $10^{-8}$ bit floor.

## Figures

To locate both onsets on one time axis, Figure 1 plots the cross-sectional correlation with its
simultaneous 95% band next to the two global shape metrics.

![Three panels: correlation, width and edge drift against training step](plots/formation_overview.png)

**Figure 1.** Ordering (A) appears ~60× earlier than plateau shape (B, C). x in all panels: training
step, symmetric-log so step 0 sits at the left edge. **A** y: Spearman $\rho(J, w)$; hatched band =
simultaneous 95% band over all 20 checkpoints; dashed horizontal line = 0; vertical hatched stripe =
the step 8 → 32 onset bracket. **B** y: transition width — solid circles are median $w$ with its
band, dotted triangles are IQR($w$), dashed line is the straight-line reference $w = 0.8$. **C** y:
edge drift $E$ (dashed squares, with band) against its straight-line reference 0.184.

A cross-sectional correlation cannot distinguish an ordering being created from one being carried
forward, so Figure 2 also correlates divergence with the width change produced inside each interval.

![Three panels: width by divergence quintile, interval correlations, cumulative correlations](plots/interval_sharpening.png)

**Figure 2.** The bulk sharpening events are divergence-blind. x: training step (symmetric-log).
**A** y: median $w$ within each corpus-divergence quintile, Q1 (lowest $J$, solid circles) to Q5
(highest $J$, dashed triangles); legend gives each quintile's median $J$ in bits. **B** y:
$\rho(J, \Delta w)$ for the interval ending at that step, pointwise 95% bars, dashed line at zero.
**C** y: $\rho$ of $J$ with the cumulative change since step 0 (solid circles) and with the model's
own endpoint output divergence (dashed squares).

Narrow $d(t)$ could be an artefact of the distance summary, so Figure 3 checks whether the model's
full next-token distribution really stops moving away from the boundary.

![Three panels: movement entropy, window mass, total movement and loss](plots/output_movement_formation.png)

**Figure 3.** Movement concentration develops on the *shape* timeline. x: training step
(symmetric-log). **A** y: normalised entropy $H(r)$ of the movement profile with band; dashed line at
1 is uniform movement, lower = more concentrated. **B** y: movement mass inside the fixed 0.2-wide
window centred on the $d = 0.5$ crossing (dashed squares, with band); dashed line at 0.2 is the
uniform expectation. **C** left y (solid circles, log): median total movement $T$ in bits; right y
(dotted triangles): held-out next-token loss in nats.

The profile itself, in Figure 4, is the most direct picture of what "plateau" means for the full output.

![Median movement profile against position relative to the d=0.5 crossing, at five checkpoints](plots/movement_profiles.png)

**Figure 4.** From a flat profile to a spike. x: position relative to the $d = 0.5$ crossing,
$t - t_{50}$ (0 = boundary). y: median normalised movement $r_j$ over 60 pairs × 3 carrier sentences.
Series are checkpoints — step 0 (solid circles), 128 (dashed squares), 1000 (dotted triangles), 8000
(dash-dot diamonds), 143000 (long-dash triangles). Flat at $1/49 \approx 0.020$ early; by the end one
step at the crossing carries 0.17 of all movement.

A median over 60 pairs is easy to move by chance, so the late reversal was re-tested on the frozen
1,000-pair bank with endpoint-clustered inference (Figure 5).

![Two panels: median width at two checkpoints for both banks, and the paired change with CIs](plots/large_bank_confirmation.png)

**Figure 5.** The late widening reproduces at scale. **A** x: the two checkpoints; y: median $w$;
solid circles = 60-pair controlled set, dashed squares = 1,000-pair set. **B** x: median paired
$\Delta w$ from step 64000 to 143000 with 95% intervals, positive = blunter at the end; dashed
vertical line at zero; y names each bank and its resampling unit.

The onset itself rests on 60 pairs, so the two checkpoints defining the bracket were re-run on the
1,000-pair bank as well, which Figure 6 compares against the controlled bank.

![Correlation with 95% intervals at five checkpoints for both banks](plots/large_bank_onset.png)

**Figure 6.** The step 8 → 32 bracket survives on the 17× larger bank. x: the five checkpoints
measured on both banks; y: Spearman $\rho(J, w)$ with 95% intervals. Circles = 1,000-pair set
(dyadic bootstrap over its 123 endpoint tokens); squares = 60-pair controlled set (bootstrap over
pairs). Hatched stripe = the onset bracket; dashed line = zero.

Bootstrap intervals measure wobble, not chance, and the step-32 ordering lives on a width spread of
0.006 — so we also ask what a random relabelling of this design produces (Figure 7).

![Three panels: observed correlations against permutation null envelopes for both banks](plots/permutation_null.png)

**Figure 7.** The observed ordering lies outside the chance envelope; the divergence-blind intervals
lie inside it. x in all panels: training step (symmetric-log). y in **A** and **C**: Spearman
$\rho(J, w)$; y in **B**: $\rho(J, \Delta w)$, plotted at the interval's end step. Hatched band
between dotted lines = pointwise 95% envelope of $|\rho|$ under 20,000 label permutations; solid
circles = observed. **A** (60-pair bank) adds dashed lines at $\pm 0.353$, the *simultaneous* null
envelope covering all 19 checkpoints. Hatched vertical stripe in A and C = the step 8 → 32 bracket;
**C** (1,000-pair bank, endpoint-label null) is annotated with each checkpoint's two-sided $p$.

Chance labellings of the 60-pair bank reach $|\rho| = 0.26$ pointwise and 0.35 simultaneously against
the observed 0.428 at step 32. On the 1,000-pair bank, relabelling its 123 endpoint tokens reaches
$|\rho| = 0.09$ — half again the 0.062 that 1,000 independent pairs would give — which prices the
token reuse into the null and still leaves step 32 significant. That matters because $-0.149$ was the
weakest number in this report.

A correlation with $J$ at step 32 does not mean the model already ranks the pairs the way it finally
will, so Figure 8 scores every checkpoint's ranking against the final one.

![Two panels: persistence of the width ranking against training step, and the checkpoint-by-checkpoint agreement matrix](plots/ranking_persistence.png)

**Figure 8.** The ranking locks in between step 64 and step 128 — after the divergence ordering,
before the shape. **A** x: training step (symmetric-log); y: rank agreement with the final ranking.
Solid circles = $\pi(s)$, dashed squares = the partial $\pi^{\perp}(s)$ with $J$ removed, both with
95% bootstrap bars; hatched band between dotted lines = pointwise 95% envelope of $|\pi|$ under
20,000 pair relabellings; dash-dot gray = the attenuation ceiling $\pi_{\max}$ from the reliability of
$w$. Left vertical stripe (`\\`) = the step 8 → 32 ordering bracket, right stripe (`xx`) = the step
64 → 128 ranking bracket. **B** x and y: the 20 checkpoints in training order (index spacing, not to
scale); colour (`cividis`): Spearman $\rho$ between per-pair widths at those two checkpoints. Dashed
white lines mark step 128; checkpoints up to step 64 form a block that agrees with itself and not
with anything later.

The scan's data-integrity finding needs its own evidence, since it would silently corrupt any
early-training analysis of this model; Figure 9 is that evidence.

![Three panels: loss trajectory outlier, byte-level tensor match, and header audit of all revisions](plots/checkpoint_qc.png)

**Figure 9.** Revision `step16` ships the final model's weights, and it is the only revision that
does. **A** x: training step (symmetric-log); y: held-out next-token loss (nats) on the frozen
256-row sample; connected circles are the 20 checkpoints kept, the large cross is `step16`. **B**
x: the revision compared against `step143000`; y: how many of 10 sampled tensors are byte-identical
to it (embeddings, unembedding, final layer norm, and the attention output weight and bias of blocks
0, 11 and 23); the label above each bar says whether the whole 2.63 GiB payload digest matches.
**C** x: training step (symmetric-log); y: safetensors JSON header length in bytes for each of the
21 revisions; circles have a `__metadata__` field, the cross does not.

Finally, the whole timing claim rests on one definition of "width", so Figure 10 re-runs both onset
rules on five of them.

![Three panels: correlation trajectories, sharpening curves, and onset brackets for five width definitions](plots/threshold_robustness.png)

**Figure 10.** The separation survives every width definition. Series in **A** and **B** are the five
level pairs defining $w_a$: 10%/90% (solid circles), 15%/85% (dashed squares), 20%/80% (dotted
up-triangles), 25%/75% (dash-dot diamonds), 30%/70% (long-dash down-triangles). **A** x: training step
(symmetric-log); y: Spearman $\rho(J, w_a)$; hatched stripe = the step 8 → 32 bracket. **B** x:
training step (symmetric-log); y: median $w_a$ divided by its own straight-line reference $1-2a$, so
all five share one scale and 1.0 (dashed line) is the no-plateau value. **C** x: training step (log);
y: the five definitions, each row showing the divergence-ordering bracket (left bar) and the
plateau-shape bracket (right bar), labelled with the ratio between the two closing checkpoints.

The third clock is scored against one arbitrary endpoint — the last released checkpoint — so Figure 11
asks whether the bracket survives scoring against four earlier mature checkpoints instead.

![Two panels: persistence trajectories under five reference checkpoints, and the bracket checkpoints per reference](plots/reference_robustness.png)

**Figure 11.** The step 64 → 128 bracket is the same under every reference. **A** x: training step
(symmetric-log); y: rank agreement $\pi_{\mathrm{ref}}(s)$ between the widths at step $s$ and at the
reference. Series are the references — step 8000 (solid circles), 32000 (dashed squares), 64000
(dotted up-triangles), 128000 (dash-dot diamonds), 143000 (long-dash down-triangles); each omits the
point where it scores against itself. Hatched horizontal band = pointwise 95% envelope of $|\pi|$
under 20,000 pair relabellings; hatched vertical stripe (`xx`) = the step 64 → 128 bracket. **B** x:
rank agreement with that row's reference; y: the five references. Filled circles = $\pi$ at step 32,
filled squares = $\pi$ at step 128, with 95% bootstrap intervals; open triangles = $\pi^{\perp}$ at
step 32 with $J$ removed; hatched band = the pointwise 95% chance envelope at step 32.

Averaging the three carrier sentences is what makes $w$ reliable enough to correlate at step 32, but
it could also hide one frame carrying the whole result, so Figure 12 re-runs every onset rule inside
each frame alone.

![Four panels: correlation, median width, ranking persistence and bracket summary, one series per carrier sentence](plots/sentence_jackknife.png)

**Figure 12.** No single sentence frame is carrying the result. Series in **A**–**C**: sentence 1
`"The thing was"` (solid circles), sentence 2 `"They said it was"` (dashed squares), sentence 3
`"I thought it was"` (dotted triangles), and the primary median of all three (dash-dot diamonds).
**A** x: training step (symmetric-log); y: $\rho(J, w)$; stripe = the step 8 → 32 bracket. **B** y:
median $w$, dashed line = the straight-line reference 0.8; stripe = the step 1000 → 2000 bracket.
**C** y: rank agreement $\pi(s)$ with that series' own final ranking; hatched horizontal band =
pointwise 95% envelope of $|\pi|$ under 20,000 relabellings; stripe = the step 64 → 128 bracket.
**D** x: training step (log); y: the four width definitions, each row showing all three brackets as
bars labelled with their opening and closing checkpoint.
