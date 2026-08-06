# RESULTS — When does the corpus-divergence / plateau relationship form during training?

> CURRENT-BEST ONLY. One row per experiment. No history (that lives in `CHANGELOG.md`).
> Full write-up with all definitions: `REPORT.md`.

## Headline

Scanning 20 released checkpoints of **Pythia-1.4B-deduped** (steps 0 → 143000) with one frozen
60-pair bank at the post-block-0 residual stream, the phenomena are **separated by ~60× in
training time, and in the counter-intuitive order**:

- **Corpus next-token divergence starts ranking pairs by sharpness between step 8 and step 32**, when
  the model has *no plateaus at all* (median transition width 0.827 vs 0.831 untrained, IQR 0.008).
  At that moment the effect is the top of the divergence range separating, not a graded axis:
  deleting the highest-divergence quintile removes it, deleting any other quintile does not.
- **The rest of the divergence range fills in between step 32 and step 128** — and still before any
  plateau. At step 128 the 1,000-pair median width is 0.832 (untrained 0.831) while its top
  divergence quintile sits at 0.806 and the other four at 0.837: the bank separates around an
  unchanged median.
- **The per-pair ranking becomes the final one between step 64 and step 128** (step 32 → 64 on the
  1,000-pair bank) — at step 32 the model holds only the divergence-aligned part of the final
  ordering. Measured on one bank at one set of checkpoints, this clock and the graded ordering are
  one checkpoint apart, so they are one early episode rather than two.
- **Plateau shape appears between step 1000 and step 2000**, and the single largest sharpening event
  (step 512 → 1000) is completely blind to corpus divergence.

Notation: $J$ = corpus next-token Jensen–Shannon divergence of a token pair (bits, held-out split);
$w$ = transition width, $t(d{=}0.9) - t(d{=}0.1)$, **smaller = sharper**; $E$ = edge drift, **smaller
= flatter ends**; $\rho$ = Spearman rank correlation, **negative = higher divergence gives sharper
boundaries**; $\pi$ = rank agreement of a checkpoint's per-pair widths with the final checkpoint's.
A no-plateau straight line $d(t)=t$ gives $w = 0.8$, $E = 0.184$.

## Metrics — current best

The scan's primary output is the set of onset brackets below. Each row names the event, the bracket
the prespecified rule returned, the statistic at onset, and what the other phenomenon was doing at
that moment — the last column is the contribution.

| Event | Onset bracket | Statistic at onset | State of the other phenomena |
|---|---|---|---|
| Divergence-selective ordering | after step 8, **by step 32** | $\rho_{32} = -0.428$, simultaneous band $[-0.753, -0.104]$, label-permutation $p^{\mathrm{fw}} = 0.0072$ | no sharpening: median $w = 0.827$ (untrained 0.831), IQR($w$) 0.008, $E = 0.209$ *above* the straight line; ranking not yet final ($\pi = 0.161$) |
| Graded ordering across the whole divergence range | after step 32, **by step 128** | 600 middle-range pairs of the 1,000-pair bank: $\rho = -0.055$ ($p = 0.34$) → $-0.257$ $[-0.409, -0.106]$, $p^{\mathrm{fw}} = 0.0004$; group gap $G$ $-0.0018 \to -0.0308$ | still no sharpening: 1,000-pair median $w = 0.832$ at step 128 vs 0.831 untrained |
| Pair ranking becomes final | after step 64, **by step 128** (60 pairs); after step 32, **by step 64** (1,000 pairs) | $\pi = +0.437$ $[+0.202, +0.623]$, $p^{\mathrm{fw}} = 0.0053$, ceiling 0.95; $\Delta\pi = +0.389$ $[+0.187, +0.592]$, $\pi^{\perp}_{\mathrm{L}} = +0.184$ $[+0.028, +0.329]$ | still no sharpening: median $w = 0.837$ (60 pairs), 0.826 (1,000 pairs), $E = 0.222$ |
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

| 1,000-pair bank | step 0 | step 8 | step 32 | step 64 | step 128 | step 256 | step 1000 | step 8000 | step 64000 | step 143000 |
|---|---|---|---|---|---|---|---|---|---|---|
| $\rho(J, w)$, endpoint-clustered 95% CI | $-0.008$ $[-0.118, +0.113]$ | $-0.021$ $[-0.136, +0.100]$ | $-0.149$ $[-0.287, -0.015]$ | $-0.351$ $[-0.496, -0.208]$ | $-0.478$ $[-0.618, -0.330]$ | $-0.548$ $[-0.661, -0.428]$ | $-0.604$ $[-0.705, -0.487]$ | $-0.537$ $[-0.650, -0.411]$ | $-0.563$ $[-0.661, -0.443]$ | $-0.486$ $[-0.603, -0.356]$ |
| endpoint-label permutation $p$ | 0.87 | 0.65 | **0.0019** | <0.0001 | <0.0001 | <0.0001 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| median $w$ | 0.831 | 0.830 | 0.828 | 0.826 | 0.832 | 0.829 | 0.750 | 0.609 | 0.537 | 0.555 |
| $\Delta\pi$ (ranking acquired since step 0) | $0$ | $-0.005$ | $+0.150$ | $\mathbf{+0.389}$ | $+0.541$ | $+0.571$ | $+0.789$ | $+0.892$ | $+0.959$ | $+1.068$ |

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

**Which pairs carry the early ordering.** $\rho_{32}$ is a rank correlation over a wide divergence
range, and such a number can come entirely from the range's extremes. Deleting one divergence
quintile at a time shows it does. Dropping the *lowest* quintile leaves $\rho_{32} = -0.426$, exactly
the median of 4,000 random 46-pair subsets; Q2, Q3 and Q4 likewise leave it between $-0.46$ and
$-0.48$ with the step 8 → 32 bracket intact. Dropping the *highest* quintile collapses it to
$-0.191$ (band spanning zero, $p^{\mathrm{fw}} = 0.77$) and moves the bracket to step 64 → 128 —
and that is not the price of 12 fewer pairs, since all 4,000 random 48-pair subsets gave a more
negative value. Over step 8 → 32 only the top quintile sharpens at all (median $\Delta w = -0.0057$
$[-0.0094, -0.0026]$; Q1–Q4 all cover zero). The 1,000-pair bank, which fills the crowded middle of
the range, rules out a power explanation: its 600 middle-range pairs give $\rho = -0.055$
($p = 0.35$) at step 32 and $-0.300$ ($p < 0.0001$) at step 143000. The timing claim stands, but what
is dated is the separation of the most distinguishable pairs; the graded ordering across the middle
of the range arrives later.

**When the graded ordering arrives.** Running the 1,000-pair bank at steps 64, 128, 256, 1000 and
8000 dates it: the same 600 middle-range pairs go from $\rho = -0.055$ ($p = 0.34$) at step 32
through $-0.157$ at step 64 ($p^{\mathrm{fw}} = 0.088$, not significant) to $-0.257$
$[-0.409, -0.106]$ at step 128 ($p^{\mathrm{fw}} = 0.0004$), then hold for the remaining 142,872
steps ($-0.315$, $-0.379$, $-0.319$, $-0.330$, $-0.300$). The prespecified rule brackets the graded
ordering at **after step 32, by step 128** — still ~8× before the shape bracket opens. The widths
have not moved at that checkpoint: 1,000-pair median $w$ is 0.832 against 0.831 untrained, because
the top divergence quintile has gone to 0.806 while the other four have gone to 0.837. The group gap
$G$ = median $w$(Q5) − median $w$(Q1–Q4) dates the same two events without a correlation: $0.0000$
(step 0), $-0.0002$ (step 8), $-0.0018$ $[-0.0037, -0.0001]$, $p = 0.0040$ (step 32), $-0.0149$
(step 64), $-0.0308$ (step 128), $-0.0794$ (step 1000).

**Are the graded ordering and the ranking lock-in one event?** Yes, as far as the released checkpoint
spacing can tell. Scoring the 1,000-pair widths against the final ones puts both clocks on the same
pairs at the same checkpoints. The agreement training adds, $\Delta\pi(s) = \pi(s) - \pi(0)$, is
$+0.150$ at step 32 with a simultaneous band $[-0.053, +0.352]$ that includes zero and
$+0.389$ $[+0.187, +0.592]$ at step 64 — bracket **after step 32, by step 64**, one checkpoint ahead
of the graded ordering. The divergence-independent part moves with it ($\pi^{\perp}_{\mathrm{L}}$
$+0.011$ $[-0.135, +0.154]$ at step 32, $+0.184$ $[+0.028, +0.329]$ at step 64), so this is
pair-specific structure being acquired, not the divergence axis being inherited — the 60-pair
Result above, reproduced on 17× more pairs and one checkpoint earlier. Both clocks therefore run
inside step 32 → 128, at median widths of 0.826 and 0.832 against 0.831 untrained.

**Robustness to the reference checkpoint, on the large bank too.** Rescoring the 1,000-pair ranking
against step 8000 and step 64000 instead of step 143000 returns the **same step 32 → 64 bracket in
all three cases**: $\Delta\pi$ at step 32 is $+0.148$, $+0.155$, $+0.150$ with simultaneous bands of
half-width 0.201–0.206 covering zero, and at step 64 it is $+0.391$, $+0.365$, $+0.389$, excluding
zero. One qualification: the divergence-independent part at step 64,
$\pi^{\perp}_{\mathrm{L}} = +0.202$ $[+0.043, +0.345]$ against step 8000 and $+0.184$
$[+0.028, +0.329]$ against step 143000, is $+0.135$ $[-0.033, +0.283]$ against step 64000 — the
"pair-specific structure, not just the divergence axis" reading holds against two of three references,
while the bracket itself holds against all three.

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

A correlation across the whole divergence range can be produced by its extremes alone, so Figure 13
deletes one divergence quintile at a time and re-runs the ordering rule on what is left.

![Three panels: correlation at step 32 per divergence subset against a size-matched random-drop envelope, per-quintile width change, and the same subsets on the large bank at two checkpoints](plots/quintile_dependence.png)

**Figure 13.** The step-32 ordering lives in the highest-divergence quintile. **A** x: Spearman
$\rho(J, w)$ at step 32 on the 60-pair bank; y: the subset used. Circles with solid bars = subsets
whose simultaneous 95% band still excludes zero, open squares with dashed bars = bands including
zero; the gray bar above each row is the 2.5–97.5% envelope of $\rho_{32}$ over 4,000 random subsets
of the same size (median tick), so a point outside it is about the deleted quintile rather than the
lost pairs. **B** x: divergence quintile Q1 (lowest $J$) to Q5 (highest), labelled with median $J$ in
bits; y: median $\Delta w$ over step 8 → 32 in units of $10^{-3}$, 95% bootstrap intervals, dotted
line at zero; the open square marks Q5, the only quintile that moves. **C** x: $\rho(J, w)$ on the
1,000-pair bank; y: the same subsets with the pairs each retains. Circles (solid) = step 32, squares
(dashed) = step 143000; hatched band = 95% envelope of $|\rho|$ under 20,000 endpoint-label
permutations.

That leaves the graded relation across the rest of the range undated, so Figure 14 tracks it on the
five checkpoints added between step 32 and step 64000.

![Three panels: correlation trajectories for the full bank and its middle three quintiles, the top-quintile width gap over training, and the four onset brackets on one timeline](plots/bulk_onset.png)

**Figure 14.** The graded ordering completes by step 128, where the widths have not moved. **A** x:
training step (symmetric-log); y: Spearman $\rho(J, w)$ on the 1,000-pair bank with 95% dyadic
endpoint-bootstrap bars; solid circles = all 1,000 pairs, dashed squares = the 600 pairs in
divergence quintiles 2–4. Dotted horizontal band = simultaneous 95% chance envelope for the full
bank; the two dotted lines are each series' own one-sided simultaneous threshold. Vertical stripes =
onset brackets, `\\` for the full bank (step 8 → 32) and `xx` for the middle three quintiles
(step 64 → 128). **B** x: training step (symmetric-log); y: group gap $G$, median $w$ of the top
divergence quintile minus median $w$ of the other four (symmetric-log scale, 95% bootstrap bars);
dashed line at 0 = no separation, negative = top quintile sharper. **C** x: training step (log); y:
the four dated events, each a bar spanning its onset bracket.

Two of those events were dated on different banks with overlapping windows, so Figure 15 re-measures
both on the 1,000-pair bank at the same ten checkpoints.

![Three panels: ranking agreement with the final widths on the 1,000-pair bank, the acquired agreement with its simultaneous band, and both clocks as a fraction of their final value](plots/large_persistence.png)

**Figure 15.** The ranking lock-in and the graded ordering are one episode inside step 32 → 128.
**A** x: training step (symmetric-log); y: rank agreement between the 1,000 per-pair widths at that
step and at step 143000 — solid circles $\pi_{\mathrm{L}}(s)$, dashed squares
$\pi^{\perp}_{\mathrm{L}}(s)$ with corpus divergence partialled out, 95% dyadic endpoint-bootstrap
bars. **B** x: training step (symmetric-log); y: $\Delta\pi(s) = \pi_{\mathrm{L}}(s) -
\pi_{\mathrm{L}}(0)$, the agreement training has added, with its simultaneous 95% band over all ten
checkpoints; dashed line at 0. **C** x: training step (symmetric-log); y: each clock as a fraction of
its own step-143000 value — graded ordering $\rho(J, w)$ over the middle 600 pairs (dashed squares)
and ranking $\Delta\pi$ (dash-dot diamonds). Dotted vertical stripe = the step 32 → 64 ranking
bracket; `xx` stripe in **C** = the step 64 → 128 graded-ordering bracket.

That bracket is scored against one arbitrary endpoint — the last released checkpoint, at which the
widths are still moving — so Figure 16 rebuilds it against two earlier mature references.

![Two panels: ranking-agreement trajectories on the 1,000-pair bank under three reference checkpoints, and the acquired agreement at the two bracket checkpoints for each reference](plots/large_persistence_ref.png)

**Figure 16.** The step 32 → 64 bracket survives every reference. **A** x: training step
(symmetric-log); y: rank agreement $\pi_{\mathrm{L,ref}}(s)$ between the 1,000 per-pair widths at step
$s$ and at the reference. Series are the references — step 8000 (solid circles), step 64000 (dashed
squares), step 143000 (dash-dot diamonds); each omits the point where it scores against itself. Dotted
vertical stripe = the step 32 → 64 bracket; dashed line at 0. **B** x: $\Delta\pi$ with its
simultaneous 95% band; y: the reference checkpoint. Circles = step 32, squares = step 64; dashed
vertical line at 0.
