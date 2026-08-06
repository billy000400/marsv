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

**Data-integrity finding.** The artefact Hugging Face serves as revision **`step16`** of
`EleutherAI/pythia-1.4b-deduped` is **not a step-16 model**: held-out loss 2.320 nats against 9.889
at step 8 and 8.824 at step 32; its 9,000 measured $d(t)$ values are bit-identical to
`step143000`'s; its `model.safetensors` is 32 bytes smaller than all 19 other revisions queried. It
is excluded from every trajectory here.

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

![Held-out loss against training step with step16 marked as an excluded outlier](plots/checkpoint_qc.png)

**Figure 9.** One released revision breaks the loss trajectory. x: training step (symmetric-log);
y: held-out next-token loss (nats) on the frozen 256-row sample. Connected circles are the 19
checkpoints kept; the large cross is revision `step16`, whose curves are bit-identical to
`step143000`'s.
