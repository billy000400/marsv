# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-08-10 — first deliverables: the leftover after corpus JSD is a per-token additive effect

Direction created. RESULTS.md and REPORT.md written from placeholder to a full current-best report
(nothing superseded — this is the first content).

**What is new.** Re-analysis of `dir18`'s stored 1,000-pair artifacts (`pythia-1.4b-deduped`
@ `step143000`, residual after block 0), no new model inference.

- **S3 endpoint-movement gate.** `min`-over-frames model-output JSD >= 0.2 bits keeps 929/1000 pairs.
  Gated-out pairs are not noisier (median across-frame `w` spread 0.02-0.04, same as the rest) but are
  systematically wide (median `w` 0.687 vs 0.545) and low-JSD (0.401 vs 0.665 bits). Headline
  correlation on the gated subset: rho(J,w) = -0.409 (p = 1.0e-38), vs -0.486 (p = 2.6e-60) on all
  1,000 pairs.
- **S2 matched contrasts.** 1,529 narrow-vs-wide contrasts matched on corpus JSD (<= 0.02 bits) and on
  endpoint output movement (<= 0.05 bits) with dw >= 0.15 consistent in all three frames; only 21 share
  a token. Largest: ` her`/` when` w=0.34 vs ` kind`/` wrong` w=0.77 at J = 0.70 bits.
- **Main result (new).** Held-out R^2 for `w` (5-fold over pairs): corpus JSD 0.149; quadratic 0.165;
  model-output JSD 0.187; 5 pair covariates + J 0.399; **token-additive 0.365; token-additive + J
  0.578**; + model-output JSD 0.648; + block-0 geometry 0.723. Reproducibility ceiling 0.934
  (Spearman-Brown on across-frame agreement, mean r = 0.825).
- **What the token effect is not.** a_u vs corpus log-frequency rho = -0.33 (p = 2.9e-4), vs
  continuation entropy -0.24 (p = 0.008), vs model surprisal +0.26 (p = 0.004).
- **Refuted alternative.** Path-length normalisation: CV(w) = 0.158 but CV(w * d0) = 0.216, and
  rho(d0, w) = +0.17 — the wrong sign for a fixed-size transition on a longer path.
- **Residual pair structure.** Additive+J residuals agree across sentence frames at r = 0.67
  (variance share 0.86 of what remains); adding output JSD and block-0 geometry lowers it to r = 0.54.

**Figures added (all embedded in both deliverables):** `scatter_and_gate.png`, `contrast_curves.png`,
`cv_r2.png`, `prediction.png`, `token_effects.png`.

`check_render.py REPORT.md RESULTS.md` passes (10 display eqs, 5 embeds + captions each, 0 problems).

## 2026-08-10 (same iteration, second step) — the per-token effect measured, not just fitted

Two new GPU probes on `pythia-1.4b-deduped` @ `step143000` at the same block-0 hook point, using six
**anchor tokens** (` and`, ` significant`, ` close`, ` playing`, ` bigger`, ` buried`) that appear in
none of the 1,000 pairs. ~220k forward passes, ~10 min.

- **Anchor width** (`experiments/anchor_width.py`): dir18's interpolation protocol between each of the
  123 endpoint tokens and each anchor; per-token median over 6 anchors x 3 frames (18/18 curves valid
  for every token). **rho(anchor width, fitted a_u) = +0.70 (p = 4.6e-19)**, +0.67 with output entropy
  partialled out. At the pair level, 2 free parameters on the measured sum give held-out R^2 = 0.350 vs
  0.365 for the 123-parameter fitted model, and 0.452 with corpus JSD added.
- **Basin radius** (`experiments/basin_probe.py`): angle of great-circle travel from a token's block-0
  state before its output distribution moves tau bits. Random directions: rho = -0.02 (p = 0.87);
  anchor directions (tau = 0.2): rho = +0.39 (p = 1.1e-5) but with the sign OPPOSITE to the basin
  prediction. Pair-level radius sum + J = 0.299. The basin mechanism is therefore not supported and has
  been dropped from the hypotheses.

**Deliverable changes.** REPORT.md gains a Methods subsection for the two probes (with equations for
the great-circle sweep, the output-movement M_u(theta) and the anchor width), Results patterns 6 and 7,
and Figure 6 (`plots/transfer.png`); old patterns 6-7 renumbered to 8-9. Hypotheses rewritten: H1 is now
"width is a per-token trait" (supported by the transfer test) instead of the basin-radius version;
H2 folded output entropy and logit norm into the predictability story; H3 unchanged. Recommended next
experiment changed from the basin probe (now run) to a **forward screen on 40 unseen tokens / 200 fresh
pairs**. Summary and Conclusion rewritten around the transfer result. RESULTS.md gains the measured-
predictor table, the transfer/basin rows, and Figure 6.

No previously reported number was superseded — all 2026-08-10 first-step numbers stand unchanged.
`check_render.py` passes (13 display eqs, 6 embeds + captions per file, 0 problems).

## 2026-08-10 (same iteration, third step) — the forward screen: prediction on unseen tokens

Ran the experiment the second step recommended (`experiments/forward_screen.py`), so the report's
"recommended next experiment" advanced again.

- **Setup.** 40 tokens that appear in none of the 1,000 bank pairs and are not anchors
  (` re`, ` do`, ` time`, ` life`, ` maybe`, ` delicious`, ` extraordinary`, ` awkward`, ...). Measured
  their anchor widths only, then predicted all 780 pairs among them with
  w_hat = -0.0275 + 0.5242 * (w_hat_u + w_hat_v), slope and intercept fitted on the 929 bank pairs and
  frozen. Then ran the 780 pairs through the full protocol and scored. 718 survive the curve-validity
  criteria and the 0.2-bit gate.
- **Result. Forward R^2 = 0.397, Spearman rho = +0.656 (p = 1.5e-89), MAE = 0.047** width units on an
  observed range 0.34-0.78. Median observed w by predicted tercile: 0.50 / 0.57 / 0.62. Baseline on the
  same pairs: model-output JSD rho = -0.507 (and it needs both endpoints of every pair, where the screen
  needs 40 per-token measurements to cover 780 pairings). Corpus JSD could not be scored here — dir18's
  corpus count arrays are no longer on disk — stated in the deliverables.

**Deliverable changes.** REPORT.md: new Methods subsection "The forward screen" (frozen-coefficient
protocol + forward R^2 equation), new Results pattern 7 with **Figure 7** (`plots/forward_screen.png`);
patterns 7-9 renumbered to 8-10 and all cross-references updated; Summary and Conclusion now carry the
forward result; Limitations rewritten (the "screen untested outside the 123 tokens" caveat is retired,
the anchor-set-composition caveat added). Recommended next experiment changed from the forward screen
(now run) to **H1's anchor-set swap**, with a layer sweep as follow-on. RESULTS.md: new forward-screen
table, Figure 7, headline paragraph, and next-experiment section.

No earlier number superseded. `check_render.py` passes (15 display eqs, 7 embeds + captions per file).

## 2026-08-10 (same iteration, fourth step) — anchor-set swap: the trait is real, the measuring stick is not neutral

Ran the control the third step recommended (`experiments/anchor_swap.py`): the anchor width of all 123
endpoint tokens recomputed against two further anchor sets, disjoint from each other and from the
original — six function words (` he`, ` it`, ` we`, ` but`, ` they`, ` them`) and six rare content words
(` surreal`, ` creepy`, ` unbelievable`, ` disgusting`, ` ironic`, ` tempting`, chosen as the highest-id
alphabetic pool tokens, GPT-NeoX BPE ids being frequency-ordered).

- **Result.** The two disjoint sets rank the 123 tokens at rho = +0.46 (p = 1.0e-7) — above chance, well
  below identity. Each still recovers the fitted token effect: rho = +0.57 (function) and +0.61 (rare
  content) against +0.70 for the original mixed set. As pair-level predictors: held-out R^2 = 0.146
  (function), 0.265 (rare content), 0.350 (mixed), 0.318 (function + rare content together).
  Conclusion: a common per-token trait exists that any anchor set finds, plus an anchor-specific
  component, so the anchor set is part of the method and a mixed set of common words is the better
  measuring stick.
- A first run of this control used only ONE valid function-word anchor (the intended list barely
  intersected the eligible pool); it was rerun with a corrected six-token set and only that arm was
  recomputed. The one-anchor numbers are not reported anywhere.

**Deliverable changes.** REPORT.md: Methods subsection "The anchor-set swap", Results pattern 8 with
**Figure 8** (`plots/anchor_swap.png`), patterns 8-10 renumbered to 9-11 with cross-references updated,
Summary now states three qualifications instead of two, H1's alternative reading updated (partly
adjudicated) and its remaining test changed to a single-anchor-many-frames measurement, Conclusion and
Limitations updated (the "anchor-set swap not run" caveat retired, replaced by "w_hat_u means width
against this anchor set"). Recommended next experiment changed from the anchor-set swap (now run) to a
**layer sweep of the anchor width** at blocks 6/12/18. Also added a fold-seed stability note to Methods
(five fold seeds move each held-out R^2 by <= 0.01). RESULTS.md: anchor-set table, Figure 8, headline
paragraph, next-experiment section.

No earlier number superseded. `check_render.py` passes (15 display eqs, 8 embeds + captions per file).

## 2026-08-10 (same iteration, fifth step) — layer sweep: the ranking is early, the sharpening is deep

Ran the experiment the fourth step recommended (`experiments/layer_sweep.py`): the anchor-width
measurement repeated with the interpolation site at blocks 6, 12 and 18 (same 6 mixed anchors, same
protocol, 100% curve validity at every site).

| site | rho vs block-0 w_hat | rho vs fitted a_u | median w_hat | IQR across tokens | held-out R^2 for pair w |
|---|---|---|---|---|---|
| block 0 | 1.00 | +0.70 | 0.553 | 0.102 | 0.350 |
| block 6 | +0.92 | +0.59 | 0.621 | 0.086 | 0.284 |
| block 12 | +0.84 | +0.52 | 0.728 | 0.065 | 0.214 |
| block 18 | +0.72 | +0.35 | 0.800 | 0.020 | 0.146 |

The token ranking survives the move (rho = +0.72 at block 18) while the transitions themselves flatten
to the proportional-response value w = 0.8 and the token-to-token spread collapses fivefold. So which
tokens are narrow is settled early, and how sharp any transition gets is produced by the blocks below
the interpolation site — the sharpening-by-downstream-MLPs picture from the plateau literature.

**Deliverable changes.** REPORT.md: Methods subsection "The layer sweep", Results pattern 9 with
**Figure 9** (`plots/layer_sweep.png`), patterns 9-11 renumbered to 10-12 with cross-references updated,
a new Summary paragraph and Conclusion paragraph on depth, and a Limitations note that the block-18
correlations are attenuated by an IQR of 0.02. Recommended next experiment changed from the layer sweep
(now run) to **an embedding-level probe** (interpolate at the input embedding; fit a linear probe from
static embedding to w_hat_u on 80 tokens, test on 43). RESULTS.md: layer-sweep table, Figure 9, and the
new next-experiment section.

No earlier number superseded. `check_render.py` passes (15 display eqs, 9 embeds + captions per file).

## 2026-08-10 (iteration 2) — the embedding probe: the per-token number can be looked up, not measured

The previous iteration's session ended immediately after `experiments/embed_probe.py` wrote
`results/embed.json` (21:55), so that experiment had never been plotted or folded into the
deliverables. This iteration completed it and added the lookup's forward test.

- **Embedding site.** Anchor widths measured with the interpolation site at the *input embedding*
  (below block 0; 123 tokens x 3 frames x 6 anchors, 99.9% curve validity) agree with the block-0
  values at rho = +0.79 (p = 2.0e-27) and with the fitted token effect at rho = +0.60.
- **Probe.** A ridge probe from the static embedding row W_E[u] to the block-0 anchor width, fitted on
  80 of the 123 tokens and tested on 43 over 50 random splits: held-out rho = +0.764 +- 0.045,
  R^2 = 0.514 +- 0.073, positive in 50/50 splits. Shuffled-target control rho = -0.201. Probe to the
  fitted effect a_u: rho = +0.505.
- **New baseline run this iteration** (`experiments/embed_forward.py`): embedding norm alone, same
  splits, rho = +0.597 +- 0.071, R^2 = 0.190. So norm (a frequency proxy) carries a large part of the
  probe's signal and the rest of the embedding carries more. A first attempt added this baseline inside
  `embed_probe.py`, but re-running that script means redoing six 2048-feature ridge probes (>15 min
  under 4-way CPU contention), so the baseline was moved into `embed_forward.py`, which already loads
  the embedding matrix; `embed_probe.py` is back to its original form and `results/embed.json` is the
  file it wrote at 21:55, unmodified.
- **Zero-forward-pass screen** (`experiments/embed_forward.py`, new): probe fitted on the 123 bank
  tokens, applied to the 40 tokens outside the bank (their measured anchor widths recovered at
  rho = +0.66, p = 3.4e-6), pair-level slope/intercept re-estimated on bank pairs from out-of-fold probe
  features and frozen (w = -0.1019 + 0.5977 * (w_u + w_v)). On the same 718 gated pairs as the measured
  screen: R^2 = 0.213, rho = +0.526, MAE 0.055, tercile medians 0.508 / 0.567 / 0.610 — against the
  measured screen's 0.397 / +0.656 / 0.047. The gap is ridge shrinkage of the predicted range; the
  lookup still ranks unseen pairs about as well as model-output JSD (rho = -0.51) with no forward pass.

**Deliverable changes.** REPORT.md: new Methods subsection "The embedding probe: can the per-token
number be looked up instead of measured?" (probe equation, the two controls, the lookup screen); new
Results pattern 10 with **Figure 10** (`plots/embed_probe.png`, 4 panels); old patterns 10-12 renumbered
11-13 with all cross-references updated; a new Summary paragraph and Conclusion paragraph on the
lookup; forward-pass count 900k -> ~1M; Limitations gained the pool-scope caveat (every token used
anywhere here is a common single-token alphabetic word) and "four sites" -> "five sites". Recommended
next experiment changed from the embedding probe (now run) to **testing the lookup on the vocabulary at
large** (~30 tokens spanning the probe's predicted range, including subword fragments, punctuation,
numerals, capitalised names). RESULTS.md: headline paragraph, two new tables (probe vs its baselines;
measured screen vs lookup screen), Figure 10, and the new next-experiment section.

No earlier number superseded. `check_render.py` passes (16 display eqs, 10 embeds + captions per file).

## 2026-08-10 (iteration 2, second step) — the lookup tested on the vocabulary at large

Ran the experiment the first step recommended (`experiments/vocab_probe.py`): the probe fitted on the
123 bank tokens applied to all 50,304 embedding rows, then 32 tokens selected from four classes
`dir18`'s pool excludes or under-samples — ordinary words outside the pool, subword fragments,
punctuation/numerals, capitalised names — eight per class, spaced evenly over that class's predicted
range, and their anchor widths measured at block 0 (same six anchors, same three frames, 576/576 curves
valid).

| class (8 tokens each) | rho(predicted, measured) | median measured w_hat |
|---|---|---|
| all 32 together | +0.60 (p = 3.0e-4) | - |
| ordinary words outside the pool | +0.57 | 0.632 |
| subword fragments | +0.31 | 0.569 |
| punctuation and numerals | +0.24 | 0.529 |
| capitalised names | +0.83 | 0.527 |
| (reference: the 123 pool tokens) | - | 0.549 |

MAE 0.046; measured widths span 0.367-0.686 against the pool's 0.361-0.660; predicted sd 0.047 vs
measured sd 0.073 (ridge shrinkage, same as pattern 10). The ranking transfers outside the pool and no
class inverts it, though per-class estimates at n = 8 are indicative only.

**Deliverable changes.** REPORT.md: new Methods subsection "The vocabulary-wide test"; new Results
pattern 11 with **Figure 11** (`plots/vocab_probe.png`); old patterns 11-13 renumbered 12-14 with
cross-references updated; Conclusion paragraph extended with the vocabulary check; Limitations reworded
(the "validated only on common alphabetic words" caveat is replaced by the weaker, accurate "eight
tokens per class, indicative"); forward-pass accounting extended. Recommended next experiment changed
from the vocabulary test (now run) to **re-measuring anchor widths in structurally different contexts**
(mid-sentence, interrogative, list, code-like), since every result so far shares one frame shape.
RESULTS.md: vocabulary table, Figure 11, headline sentence, and the new next-experiment section.

No earlier number superseded. `check_render.py` passes (16 display eqs, 11 embeds + captions per file).

## 2026-08-10 (iteration 2, third step) — frame-shape control: the ranking is the token's, the level is the context's

Ran the experiment the second step recommended (`experiments/frame_control.py`): the anchor-width
measurement for the same 123 tokens and the same six anchors repeated in four contexts of deliberately
different shape, and each context's token ranking correlated with the original one. The reference is
the agreement among the three ORIGINAL frames, each summarised the same way (median over 6 anchors
within one frame), which is what two measurements of the same shape achieve.

| context | rho with the original ranking | median w_hat | IQR | curve validity |
|---|---|---|---|---|
| (reference: two of the three original frames) | +0.822 | 0.549 | 0.102 | - |
| `She kept walking because everything felt` | +0.844 | 0.599 | 0.123 | 1.000 |
| `Is it really` | +0.770 | 0.623 | 0.107 | 0.996 |
| `The report mentions the following:` | +0.735 | 0.530 | 0.118 | 1.000 |
| `def solve(x): / return` (code) | +0.501 | 0.705 | 0.049 | 1.000 |

The nearest context matches the within-shape reference (+0.844 vs +0.822), and even the code prefix
stays above the two-disjoint-anchor-set agreement (+0.46). The level moves a lot (median 0.530-0.705),
and the code context also compresses the token spread (IQR 0.049). Mean agreement among the four new
contexts: +0.624.

**Deliverable changes.** REPORT.md: new Methods subsection "The frame-shape control"; new Results
pattern 12 with **Figure 12** (`plots/frame_control.png`); old patterns 12-14 renumbered 13-15 with
cross-references updated; new Summary paragraph; Conclusion extended; **Limitations rewritten** — the
"three sentence frames shared by every measurement" caveat is now narrowed to the pair-level results,
since the per-token measurement has been checked in four other context shapes. Forward-pass accounting
1M -> 1.1M. Recommended next experiment changed from the frame-shape control (now run) to an
**embedding-space intervention**: add +-epsilon times the probe's unit direction to a token's embedding
row, re-measure its anchor width, with a matched-norm random-direction control — the first test of
whether the probe direction is causal rather than correlational. RESULTS.md: frame-control table,
Figure 12, headline paragraph, setting line, and the new next-experiment section.

No earlier number superseded. `check_render.py` passes (16 display eqs, 12 embeds + captions per file).

## 2026-08-10 (iteration 2, fourth step) — the embedding intervention: a null, and the first causal test

Ran the experiment the third step recommended (`experiments/embed_intervene.py`). The probe's gradient
with respect to the raw embedding row, g_j = beta_j / sd_j, gives a step delta = (Delta / ||g||^2) g
that changes the probe's OWN prediction by exactly Delta. For 16 tokens spread over the measured-width
range, that step was added to the token's embedding row for Delta in {-0.05, -0.025, +0.025, +0.05}
width units, the token's anchor width re-measured, the row restored; controls were a random direction
of the same step norm and the JSD shift of the token's next-token distribution.

| edit along | slope of measured vs requested dw | mean abs dw | sign agreement | output shift |
|---|---|---|---|---|
| (a causal direction would give) | 1.0 | 0.0375 | 1.00 | - |
| probe direction | -0.023 | 0.0027 | 0.39 | 0.0001 bits |
| random direction, same norm | +0.000 | 0.0008 | 0.50 | 0.0000 bits |

Per-token slopes scatter from -0.13 to +0.15 with no relation to the token's base width
(rho = -0.115, p = 0.67), so the "perturbation compresses width toward the middle" reading was checked
and is not supported either. Step norm 0.053 against a median embedding-row norm of 0.984 (~5%).
**Conclusion reported:** the embedding direction that predicts width does not set it — but the edits
shifted the model's output by only 0.0001 bits, so this establishes that the probe direction is not an
efficient lever, not that no lever exists.

**Deliverable changes.** REPORT.md: new Methods subsection "The embedding intervention" (with the step
equation); new Results pattern 13 with **Figure 13** (`plots/intervene.png`); old patterns 13-15
renumbered 14-16 with cross-references updated; Conclusion extended with the null and its caveat;
forward-pass accounting 1.1M -> 1.2M. Recommended next experiment changed from the intervention (now
run) to **repeating it with the step calibrated on the model's behaviour** (grow the step until the
token's output moves 0.05 / 0.1 / 0.2 bits), with the fallback target named (block 0's attention and
MLP response) if both directions then move width equally. RESULTS.md: intervention table, Figure 13,
setting line, and the new next-experiment section.

No earlier number superseded. `check_render.py` passes (17 display eqs, 13 embeds + captions per file).

## 2026-08-11 (iteration 3) — the behaviour-calibrated intervention: the trait is fragile, not steerable

Ran the experiment the previous iteration recommended (`experiments/embed_intervene2.py`,
`experiments/plot_intervene2.py`). For 12 tokens spanning the measured-width range, the embedding edit
is no longer sized by the probe's own prediction but grown along a unit direction until the token's
next-token distribution has moved a fixed number of bits (0.05 / 0.1 / 0.2), in BOTH signs, with a
random unit direction calibrated to the SAME output movement as the control — 144 re-measurements.
Calibration fidelity: achieved / requested output movement median 1.00 (IQR 0.91-1.05).

| edit along | mean abs dw at 0.05/0.1/0.2 bits | signed dw, + / - step (0.05 bits) | median step norm | edits that widen |
|---|---|---|---|---|
| probe direction | 0.103 / 0.130 / 0.148 | +0.118 / +0.088 | 1.01 | 72/72 |
| random, matched on output movement | 0.109 / 0.125 / 0.135 | +0.109 / +0.109 | 1.62 | 72/72 |

Probe vs random on matched edits 0.127 vs 0.123 (Wilcoxon p = 0.47, probe larger in 53%); regression of
measured on probe-predicted dw: slope -0.002, rho = +0.06 (p = 0.61). After a 0.2-bit edit the 12
tokens land at mean w_hat 0.691 (probe) / 0.678 (random) with sd across tokens 0.022 / 0.015, against
mean 0.543 sd 0.083 before; rho(base w_hat, dw) = -0.78 (probe) / -0.94 (random). Reaching a given
output movement takes a smaller step along the probe direction (norm ratio 1.54 / 1.66 / 1.76), and the
larger random displacements (median edited row norm 1.90 vs 1.40) still land at the same width, so the
collapse is indexed by output movement rather than by how far the row moved.

**Conclusion reported.** The previous iteration's null WAS a step-size null — width moves fifty times
more once the edit is behaviourally real (0.003 -> 0.10-0.15 width units) — but the specificity test
fails completely: no single embedding direction is a lever, and every behaviourally sized edit destroys
the token's width trait, leaving it at a generic ~0.68. Narrow transitions are a fragile property of the
exact embedding training produced.

**Deliverable changes.** REPORT.md: new Methods subsection "The behaviour-calibrated intervention"
(with the calibration equation); new Results pattern 14 with **Figure 14** (`plots/intervene2.png`); old
patterns 14-16 renumbered 15-17 with cross-references updated; Summary paragraph on the open mechanism
rewritten around the collapse result; Conclusion rewritten in the same place; forward-pass accounting
1.2M -> 1.3M. Recommended next experiment changed from the behaviour-calibrated intervention (now run)
to a **behaviour-preserving displacement**: step the embedding row by the norm a 0.2-bit edit needed but
along directions whose output shift stays under 0.005 bits, and see whether w_hat survives — deciding
whether the trait is behavioural or geometric, which also decides what the free vocabulary lookup is
reading. RESULTS.md: two new tables (calibrated intervention, and where the edited tokens land),
Figure 14, a new headline paragraph, the setting line, and the new next-experiment section.

No earlier number superseded; the probe-calibrated intervention is retained as the test of the probe's
own quantitative claim, explicitly framed as the loophole this experiment closes. `check_render.py`
passes (18 display eqs, 14 embeds + captions per file).

## 2026-08-11 (iteration 3, second step) — the fixed-displacement test: the collapse follows the move, not the model's response

Ran `experiments/embed_quiet.py` + `experiments/plot_quiet.py` (Figure 15) to separate the two readings
of the collapse. For each of the same 12 tokens, 48 random directions were probed at a step of 0.05
(the linear regime) and the logit responses stacked into a matrix whose left singular vectors give the
loudest and quietest combinations; the token was then displaced by the SAME norm its own 0.2-bit edit
required (median 1.84) along the quietest, the loudest and one plain random direction.

| direction, same displacement norm per token | output movement produced | mean dw | mean w_hat after | sd across tokens |
|---|---|---|---|---|
| (before any edit) | - | - | 0.543 | 0.083 |
| quietest of 48 combinations | 0.181 bits | +0.132 | 0.675 | 0.019 |
| loudest of 48 combinations | 0.261 bits | +0.105 | 0.648 | 0.039 |
| plain random direction | 0.165 bits | +0.132 | 0.675 | 0.026 |

Across the 36 edits, rho(output movement, dw) = +0.074 (p = 0.67) over a 0.03-0.77 bit range, while the
collapse reproduces in all three directions. Post-edit widths still correlate with the originals at
rho = +0.62 (p = 0.03). **Reported limitation:** the "quiet" construction is not quiet at this
displacement (0.181 vs 0.165 bits for a random direction) — the linear response at eps = 0.05 does not
survive a step of norm 1.84 — so the behaviour-preserving version of the test was not achieved.

**Superseded numbers.** The previous entry's claim that the collapse "is indexed by output movement
rather than by how far the row moved" (inferred from probe steps of norm 1.40 landing at 0.691 against
random steps of norm 1.90 landing at 0.678) is **withdrawn**: at fixed displacement norm the width
change does not track output movement at all (rho = +0.07, p = 0.67). The corrected statement is that
the landing width is insensitive to what the edit does to the model over this range, and what matters
is that a displacement of this size happened. Also corrected: "destroys the trait" -> "compresses the
trait", since the post-edit ranking still agrees with the original (rho = +0.73 / +0.85 after a
0.05-bit edit, +0.57 / +0.36 after a 0.2-bit one, +0.62 in this run).

**Deliverable changes.** REPORT.md: new Methods subsection "The fixed-displacement test" (with the
response-matrix equation); new Results pattern 15 with **Figure 15** (`plots/quiet.png`); patterns
15-17 renumbered 16-18 with cross-references updated; pattern 14, the Summary and the Conclusion
corrected as above; Limitations extended with the 12-token sample size and the failed quiet
construction; forward-pass accounting 1.3M -> 1.4M. Recommended next experiment changed to **a ladder
of displacement norms (0.1 to 1.0) with the quiet/loud combinations rebuilt at each**, to find the
norm at which the two separate. RESULTS.md: fixed-displacement table, Figure 15, headline paragraph
and next-experiment section updated the same way. `check_render.py` passes (19 display eqs, 15 embeds
+ captions per file).

## 2026-08-11 (iteration 4) — the displacement-norm ladder: behaviour destroys the trait, displacement only compresses it

Ran `experiments/norm_ladder.py` + `experiments/plot_ladder.py` (Figure 15, `plots/ladder.png`), the
experiment the previous entry recommended. For each of the same 12 tokens and each rung of a
displacement ladder (norms 0.15 / 0.4 / 0.9 / 1.8, against a median embedding-row norm of 0.98), 24
random unit directions were displaced by that norm and their **actual** output movement measured there;
the argmin became the quiet direction and the argmax the loud one, with a fixed random direction as
control. Anchor width was re-measured for all three at every rung (144 re-measurements).

| displacement norm | quiet: bits / mean w_hat / rho(before, after) | loud: bits / mean w_hat / rho(before, after) | paired p |
|---|---|---|---|
| (before any edit) | - / 0.543 / - | - / 0.543 / - | - |
| 0.15 | 0.0001 / 0.544 / +1.00 | 0.0003 / 0.546 / +1.00 | 0.09 |
| 0.40 | 0.0006 / 0.552 / +0.99 | 0.0027 / 0.562 / +0.99 | 0.02 |
| 0.90 | 0.0053 / 0.589 / +0.91 | 0.0221 / 0.620 / +0.87 | 0.0005 |
| 1.80 | 0.0489 / 0.656 / +0.94 | 0.4023 / 0.683 / +0.08 | 0.09 |

**Superseded result and reversed conclusion.** This replaces the fixed-displacement test (previous
entry, `plots/quiet.png`, 12 tokens at per-token norm ~1.84 with quiet/loud built from the SVD of
linear-regime logit responses), whose reported limitation was that the construction produced no
genuinely quiet direction there (0.181 bits quiet vs 0.165 random). Selecting by measured response at
each rung fixes that: 0.049 vs 0.402 bits at norm 1.8, an 8x separation. With a real contrast the
earlier reading **reverses**. Old: "the width change is flat in output movement (rho = +0.07, p = 0.67)
... what erases the trait is the displacement itself rather than what the displacement does to the
model's output." New: at identical displacement the quiet edit preserves the token ordering
(rho = +0.94, p = 4e-6) and the loud edit erases it (rho = +0.08, p = 0.80); the quiet direction widens
less than the loud one in the paired test at every rung (p = 0.0005 at norm 0.9, p = 0.02 at 0.4). The
*level* still follows the displacement (0.543 -> 0.656 even for the quietest direction at norm 1.8, sd
across tokens 0.083 -> 0.038), so the corrected statement is: displacement compresses the level,
behaviour destroys the ordering. Consequence for the deliverable's story: the vocabulary-wide lookup
reads a behavioural property, not a geometric accident — the caveat the previous entry flagged as
unresolved is resolved in the favourable direction.

**Deliverable changes.** REPORT.md: Methods subsection "The fixed-displacement test" replaced by "The
displacement-norm ladder" (with the per-rung response equation and the level/ordering split stated as
the two discriminating quantities); pattern 15 rewritten with the rung table and **Figure 15** now
`plots/ladder.png` (`plots/quiet.png` retired from both deliverables, PNG kept on disk); Summary and
Conclusion paragraphs on fragility rewritten from "displacement erases it" to "behaviour erases the
ordering, displacement compresses the level"; Limitations updated (quiet = quietest of 24 draws, and
0.049 bits is quiet-relative-to-loud, not silent); forward-pass accounting 1.4M -> 1.5M. Recommended
next experiment changed from the ladder itself to **decomposing the loud edit's output change by
successor token** (top-mass successors vs tail, at matched total output movement), which asks which
part of the token's behaviour carries the trait and whether it reconnects to corpus successor JSD.
RESULTS.md: headline fragility paragraph, ladder table, Figure 15 and next-experiment section updated
the same way. `check_render.py` passes (19 display eqs, 15 embeds + captions per file).

## 2026-08-11 — iteration 5: the mode split (which successors a disruptive edit moves)

**New experiment.** `experiments/mode_split.py` (+ `experiments/plot_mode.py`, `results/mode_split.json`,
`results/mode_split.log`) runs the experiment the previous entry recommended. For the same 12 tokens as
the ladder, every edit's output change is partitioned by successor token (JSD is a sum of non-negative
per-successor terms) into the share $S$ landing on the token's 32 most likely successors and the rest.
24 random directions per token at displacement norm 1.8 give the descriptive half; the most top-heavy
and most tail-heavy of them, each rescaled by a log-log calibration scan to the same 0.4 bits of output
movement, give the causal half. 12 tokens x (24 scan directions + 2 calibrated re-measurements).

**New numbers.** Base mass in the top 32 successors 0.707, but the loudest random direction puts only
**0.389** of its divergence there — the disturbance is tail-weighted relative to mass — and louder
directions are more tail-weighted still (median-token $\rho(B_j, S_j) = -0.36$). At matched movement:
top-heavy 0.410 bits, $S = 0.408$, mean $\hat w_u$ 0.666, $\rho$(before, after) $= -0.08$ ($p = 0.81$);
tail-heavy 0.453 bits, $S = 0.355$, mean $\hat w_u$ 0.651, $\rho = -0.37$ ($p = 0.24$); pre-edit
0.543 +- 0.083. Both destroy the ordering. One paired difference survives: the top-heavy edit widens
more (mean $\Delta\hat w_u$ +0.124 vs +0.108, Wilcoxon $p = 0.009$) while moving the output less.

**Nothing superseded.** This is a new result, not a replacement; no earlier number changed. The
steering half is a null with a stated limit (random directions span only $S = 0.36$–$0.56$ and never
reach the mass-proportional 0.71), reported as such.

**Deliverable changes.** REPORT.md: new Methods subsection "The mode split: which successors does a
disruptive edit move?" (per-successor JSD partition and top-mass share $S$ defined in a display
equation, with the base-mass reference point and the matched-movement rescaling); new Results
**pattern 16** with **Figure 16** = `plots/mode_split.png`; previous patterns 16/17/18 renumbered to
17/18/19 and the two "(pattern 16)" cross-references to the basin result updated to "(pattern 17)";
"Recommended next experiment" rewritten from "decompose the loud edit by successor token" (now done) to
**constructing** top-heavy and tail-heavy directions from the top successors' unembedding rows or a
subspace search, since random draws give too little contrast in $S$. RESULTS.md: mode-split table and
Figure 16 added, headline fragility paragraph extended with the tail-weighting number, next-experiment
section rewritten to match, forward-pass accounting and setting line updated. `check_render.py` passes
(REPORT 20 display eqs / 16 embeds, RESULTS 16 embeds, 0 problems).

## 2026-08-11 — iteration 6: constructing the top-heavy / tail-heavy directions

**New experiment.** `experiments/mode_construct.py` (+ `experiments/plot_construct.py`,
`results/mode_construct.json`, `results/mode_construct.log`) runs the experiment the previous entry
recommended, in its cheap form. For each of the same 12 tokens, 24 random probe directions are applied
at displacement norm 0.6 and their logit responses are turned into a 24x24 generalised eigenproblem
whose Rayleigh quotient IS the top-mass share $S$; the extreme eigenvectors give the $S$-maximising and
$S$-minimising combinations in closed form (no Jacobian, ~40x cheaper than the projection version). Both
are rescaled to the same 0.4 bits and anchor width is re-measured.

**New numbers.** Predicted $S$ (small-step regime) 0.856 vs 0.179 — a 0.68 separation, 3x what 24 random
draws span (0.21), with the top-heavy end past the mass-proportional 0.71. Achieved $S$ once rescaled to
0.4 bits: **0.369 vs 0.390** — indistinguishable, both below the base mass, paired difference
$p = 0.09$ and in the *wrong* direction. Output movement 0.422 vs 0.419 bits; mean $\hat w_u$ 0.666 vs
0.672 (pre-edit 0.543 +- 0.083), sd across tokens 0.023 / 0.020; $\rho$(before, after) $= -0.16$
($p = 0.62$) and $-0.28$ ($p = 0.38$); paired $\Delta\hat w_u$ difference $p = 0.38$.

**Nothing superseded.** New result; no earlier number changed. It converts pattern 16's underpowered
steering null into a *bounded* one — the instrument now exists and still cannot separate the arms —
and adds the mechanistic point that the tail-weighting of a large embedding edit is set by the step
size, not by the direction.

**Deliverable changes.** REPORT.md: new Methods subsection "Can the split be steered on purpose?" (the
Rayleigh-quotient construction with $A$/$B$ defined in a display equation, probe norm 0.6, predicted-vs-
achieved $S$ as a test of the method); new Results **pattern 17** with **Figure 17** =
`plots/mode_construct.png`; previous patterns 17/18/19 renumbered to 18/19/20 and the four cross-
references updated; "Recommended next experiment" rewritten from "build the two directions" (now done,
negative) to **component ablation in blocks 0–5** scored by how much of the across-token spread in
$\hat w_u$ each component destroys — the first intervention that is not an embedding edit; forward-pass
accounting 1.5M -> 1.6M and runtime 4h -> 4.5h. RESULTS.md: headline fragility paragraph extended with
the constructed-direction result, Figure 17 and its table added, next-experiment section rewritten to
match, setting line updated. `check_render.py` passes (REPORT 21 display eqs / 17 embeds, RESULTS 17
embeds, 0 problems).

## 2026-08-11 — iteration 7: component ablation localises the trait to the block-0 MLP (Figure 18)

**What ran.** New `experiments/ablate.py` + `experiments/plot_ablate.py` (outputs `results/ablate.json`,
`results/ablate_summary.json`, `results/ablate.log`, `plots/ablate.png`) runs the experiment the previous
entry recommended: the first intervention in this direction that is not an embedding edit. Each of the
102 early components — the 16 attention heads and the MLP of every block 0–5 — is mean-ablated one at a
time at the final token position (replacement = that component's mean output at that position over the
18 endpoint prompts, measured unablated), endpoints and interpolation bank are recomputed with the
ablation live, and the per-token anchor width is re-measured for the 12 intervention tokens against the
6 anchors in the first frame. 470 s on one GPU.

**New numbers.** Unablated: mean $\hat w_u$ 0.565, sd across tokens 0.084. **Block-0 MLP ablated: mean
0.822, sd 0.018, $\rho$(before, after) $= -0.10$, output movement 0.451 bits.** Every other component:
median $\rho = +0.99$ and median sd 0.084 across the 102; worst of the 96 attention heads $\rho = +0.97$
(sd 0.076, $\le 0.0004$ bits); worst of the five MLPs above block 0 $\rho = +0.90$ (sd 0.091,
$\le 0.007$ bits). $\rho$(output movement, $\rho_c$) $= -0.46$ across components.

**Nothing superseded.** New result; no earlier number changed. It is the first positive mechanistic
localisation in the direction, and it is reported with its confound stated: 0.451 bits is essentially
the 0.4-bit rung at which pattern 15's ladder showed any disturbance flattens the ordering, so ablation
alone cannot say whether the block-0 MLP computes the trait or is merely the only early component large
enough to reach that regime.

**Deliverable changes.** REPORT.md: new Methods subsection "Component ablation: which early computation
carries the trait?" (mean-ablation protocol, head-slice vs MLP-output hook points, and a display
equation defining the three per-component scores $\mathrm{sd}_c$, $\rho_c$, $B_c$); new Results
**pattern 21** with **Figure 18** = `plots/ablate.png` and its table, appended after pattern 20 so
figure order stays sequential; Summary gains a paragraph on the localisation and its confound;
"Recommended next experiment" rewritten from component ablation (now done) to the **block-0 MLP
dose–response curve against an output-movement-matched random control**. RESULTS.md: headline gains the
localisation paragraph, new ablation table + Figure 18 with the flat-profile reading spelled out, next-
experiment section rewritten to match. `check_render.py` passes (REPORT 22 display eqs / 18 embeds,
RESULTS 18 embeds, 0 problems).

## 2026-08-11 — iteration 8: dose–response breaks the block-0 MLP confound (pattern 22)

**New experiment.** `experiments/dose.py` + `experiments/plot_dose.py` → `results/dose.json`,
`plots/dose.png`. The block-0 MLP's final-position output is blended toward its mean with weight
alpha = 0.1 … 1; at each dose a fixed random direction added to the same residual stream is
bisection-scaled to move the model's output by the same number of bits. Both arms are scored by
rho(before, after) of the per-token anchor width and by the across-token sd, for the same 12 tokens,
6 anchors and 1 frame as the ablation sweep. 56 s.

**Result (new, supersedes nothing).** The arms separate where it matters. Over 0.007–0.103 bits the MLP
dose is below its matched control at every rung (rho +0.84/+0.99, +0.64/+0.91, +0.62/+0.79,
+0.25/+0.61); the MLP arm crosses rho = 0.6 at ~0.03 bits and the control at ~0.10, i.e. a random
disturbance needs ~3.5x more output movement for the same damage. The across-token sd collapses
identically in both arms (0.069/0.067, 0.055/0.053, 0.027/0.026). Above 0.25 bits both arms are at
noise (+0.74/−0.32, −0.10/−0.76; SE(rho) ≈ 0.3 at n = 12) and are reported but not interpreted.

**Deliverable changes.**
- RESULTS.md — summary bullet on the ablation updated: the confound sentence ("which is both the
  finding and its caveat") replaced by the dose–response verdict. New subsection "Dose–response: is the
  block-0 MLP special, or merely loud?" with the matched-bits table and **Figure 19**
  (`plots/dose.png`). "Next experiment" rewritten: old ask (run the dose–response) → new ask (probe and
  transplant the block-0 MLP output vector m_u).
- REPORT.md — Summary paragraph on the ablation updated with the dose–response verdict; new Methods
  subsection "Separating a carrier from a loud component: the matched-bits dose–response" defining
  m_u^mlp(alpha), m_u^ctrl(c) and the bisection condition B(ctrl) = B(mlp) in a ```math fence; new
  Results pattern 22 with the table and **Figure 19**; "Recommended next experiment" rewritten from the
  dose–response ask to the probe/transplant ask.
- Figure count 18 → 19 in both files. `check_render.py` passes on both (23 display eqs, 19 embeds).

**Interpretation recorded.** The direction's first positive mechanistic localisation: the per-token
width trait is realised in the block-0 MLP's contribution to the final-position residual stream, and
level (spread) versus ordering are separate channels — disturbance of any kind compresses the level,
only the block-0 MLP erases the ordering.

## 2026-08-11 — iteration 9 (final): Conclusion brought up to the current-best story; direction closed

**No new experiment.** Time budget exhausted; this iteration is finalization only. All plan stages
S1–S5 are complete and every experiment S5 recommended has been run through the block-0 MLP
dose–response (iteration 8). Zero unaddressed `human_feedback*.md` / `*REVIEW*` files in this
direction (checked at iteration start), so writing `STOP` is permitted under CLAUDE.md rule 11.

**Deliverable changes (REPORT.md only; RESULTS.md was already current-best and is unchanged).**
- REPORT.md Conclusion contained a stale claim that iteration 7–8's results had superseded:
  "What each token carries is the shape of its transition, and we cannot yet say what in the network
  produces it." The Summary and Results already carried the ablation + dose–response localisation, so
  the Conclusion contradicted the rest of the report. Old → new: that clause is cut, and a new
  Conclusion paragraph states the localisation — 101 of 102 early components leave the token ordering
  intact (median rho = +0.99, every head >= +0.97); only the block-0 MLP collapses the spread
  (0.084 → 0.018) and erases the ordering (rho = −0.10); the matched-bits dose–response breaks the
  loudness confound (at 0.014 bits, rho = +0.64 MLP vs +0.91 control; ~3.5x more output movement
  needed by the control for equal damage), with the across-token spread collapsing identically in both
  arms. Level and ordering named as separate channels; the probe/transplant follow-up named.
- REPORT.md Limitations gains the dose–response caveat: 12 tokens, one frame, one random-control seed,
  and both arms at noise above 0.25 bits (SE(rho) ≈ 0.3 at n = 12), so the localisation rests on the
  four rungs between 0.007 and 0.103 bits.
- No figure added or removed; figure count stays 19 in both files, numbering unchanged.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 23 display eqs / 522 inline eqs / 19 embeds / 0 problems; RESULTS 319 inline eqs / 19 embeds /
0 problems).

**Final state of the deliverable.** Headline chain, current-best: corpus successor JSD explains
held-out R^2 = 0.149 of transition width against a 0.934 reproducibility ceiling; a per-token additive
term takes it to 0.578 and alone reaches 0.365; the per-token number is measurable from six unpaired
anchors (rho = +0.70 with the fitted effect, 2 parameters matching 123); frozen into a screen it
predicts 718 pairs of 40 unseen tokens at R^2 = 0.397 / rho = +0.66 / MAE 0.047; it is readable from
the static embedding (probe rho = +0.76, zero-forward-pass screen R^2 = 0.213) and holds outside the
curated pool (rho = +0.60 on 32 tokens from four excluded classes) and across four context shapes
(rho +0.84 … +0.50 vs +0.82 for two original frames); the trait is behavioural, not positional
(quiet vs loud edits at displacement 1.8: rho +0.94 vs +0.08); and it is realised in the block-0 MLP's
contribution to the final-position residual stream (dose–response vs an output-matched control).
Negative results retained: basin radius refuted, path-length normalisation refuted, probe direction is
not a width lever, top-mass steering untestable by embedding edits.

**STOP written.** Plan complete, no unaddressed feedback.

## 2026-08-12 — iteration 10: operator feedback (`human_feedback.txt`), per-token-matched dose–response

**Feedback addressed.** `human_feedback.txt`: *"the MLP and random controls are matched only on the
mean output JSD across the 12 tokens, while the conclusion depends on the ordering of those individual
tokens … match or normalize output JSD separately for each token before concluding that the block-0
MLP specifically carries the transition-width ordering."* Renamed to `human_feedback.addressed.md`.

**What was run.** New `experiments/dose2.py`: the dose–response rerun with the random control's scale
binary-searched **separately for each of the 18 endpoint prompts** (12 tokens + 6 anchors) so every
token's own output moves exactly as many bits as the block-0 MLP dose moved it, repeated for three
random seeds (the previous run used one). Added a paired per-token statistic (Wilcoxon over the 12
tokens on |Δŵ_u|, and on the level-free |Δŵ_u − mean Δŵ|), and reran the old mean-matched control
alongside purely to measure how mismatched it had been. `experiments/plot_dose2.py` replaces
`plots/dose.png` with a four-panel figure. Results in `results/dose2.json` / `dose2.log` (~3 min GPU).

**The feedback was right, and the numbers change.** The mean-matched control gave individual tokens
between **0.08× and 8.5×** the output movement the MLP dose gave them (the dose itself is uneven: at
full ablation per-token movement spans 0.254–0.710 bits, a factor of 2.8). Superseded numbers, old →
new, in RESULTS.md and REPORT.md:

- Control's rank agreement, mean-matched → per-token-matched, at the matched MLP rungs:
  0.007 bits +0.99 → +0.98; 0.014 bits +0.91 → +0.91; 0.029 bits +0.79 → +0.76; **0.103 bits +0.61 →
  +0.15** (the old control was under-dosed at 0.078 bits there); 0.265 bits −0.32 → +0.24; 0.451 bits
  −0.76 → −0.06. MLP arm unchanged (+0.84 / +0.64 / +0.62 / +0.25 / +0.74 / −0.10).
- Headline margin: **"a random disturbance needs ~3.5× more output movement to do the same damage" →
  "~1.3×"** (rho = 0.6 crossing: MLP 0.031 bits, matched control 0.041 bits; the loose control's 0.086
  bits gives the discarded 2.8–3.5× figure).
- Band of the claim: "four rungs 0.007–0.103 bits" → **"five rungs up to 0.03 bits"**; above 0.1 bits
  the two arms cross and are not interpreted (SE(rho) ≈ 0.3 at n = 12).
- Seeds: one random-control seed → three (control rho now reported as mean ± sd across seeds).
- New primary statistic (level-free, per token, paired): the dose moves each token's width ~2× as far
  as that token's own matched control — 0.074 vs 0.036 width units at 0.0068 bits (Wilcoxon
  p = 0.0010); after subtracting each arm's mean shift, 0.034 vs 0.014 (p = 0.034) and 0.047 vs 0.022
  at 0.0143 bits (p = 0.016); null once the ordering is dead (p ≥ 0.47 above 0.1 bits).
- Retracted claim: "the across-token spread collapses **identically** in the two arms". True through
  0.014 bits (0.070/0.074, 0.069/0.068), false above it — the dose compresses harder (0.027 vs 0.055 at
  0.103 bits). The level/ordering split now rests on the rungs where an ordering still exists.
- Localisation itself **stands**: the MLP arm is below its matched control in 15/15 rung × seed
  comparisons in the live band. Its strength is downgraded from "3.5× cheaper" to "1.3× in bits, ~2×
  in per-token width change, below 0.03 bits".

**Where it changed.** REPORT.md Summary (matched-bits paragraph), Methods (the dose–response
subsection now defines per-prompt matching with $B_p$, the three seeds, and both paired statistics),
Results pattern 22 (rewritten, new table, new Figure 19 caption, mismatch diagnostic), Conclusion and
Limitations. RESULTS.md headline paragraph and the dose–response section (new table + paired-test
table). Figure count stays 19 in both files; `plots/dose.png` regenerated (2 panels → 4: ordering,
spread, control-matching ratio, per-token change).

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 25 display eqs / 551 inline eqs / 19 embeds / 0 problems; RESULTS 339 inline eqs / 19 embeds /
0 problems).

## 2026-08-12 — iteration 10 (part 2): the recommended follow-up run — block-0 MLP probe + transplant

With the feedback addressed and budget left, the experiment both deliverables named as the single most
informative next step was run: `experiments/mlp_read.py` (probe + transplant) and
`experiments/mlp_geom.py` (how large the transplant is). Results in `results/mlp_read.json`,
`results/mlp_geom.json`; new figure `plots/mlp_read.png` embedded as **Figure 20** in both
deliverables (figure count 19 → 20).

**Added to RESULTS.md and REPORT.md (new results, nothing superseded).**
- **Transplant — a strong positive.** Overwriting one token's block-0 MLP final-position output $m_u$
  with another's (anchors untouched) transports the width: per-recipient rho(donor width, resulting
  width) = **+0.968** (min +0.95, Wilcoxon p = 5e-4), slope **+0.913** on the donor's own width, while
  the recipient's remaining state contributes nothing (per-donor rho = **−0.104**, p = 0.64;
  between-donor variance 66× between-recipient). Self-transplant reproduces the baseline exactly
  (rho = +1.000, max diff 0.0000) — the pipeline's sanity check.
- **Context-free.** $m_u$ has cosine **1.0000** across the three sentence frames: Pythia's parallel
  residual means block 0's MLP reads the token embedding before attention writes, so a token's width is
  fixed before any context is read. This is now the stated reason the static-embedding lookup works.
- **Probe — a null, reported as one.** Ridge probe from $m_u$ to measured $\hat w_u$ (embedding
  probe's protocol): rho = **+0.748 ± 0.049**, R^2 = 0.511, against **+0.764** for the static embedding
  row and **+0.772** for the full post-block-0 state; shuffled targets −0.234. All within 1 sd, so the
  block-0 MLP carries the trait without making it more linearly readable. Practical consequence stated:
  the free embedding lookup gives up nothing to a deeper probe.
- **Scale caveat stated with numbers** (`mlp_geom.py`): a cross transplant moves the output by a median
  0.738 bits; $m_u$ is 0.79 of the post-block-0 state's norm and 0.76 of its across-token spread, and
  the hybrid state sits ~0.75 of the way from recipient to donor. Claim limited accordingly: the
  width-relevant content of the block-0 state lives in the MLP's contribution, not "a small edit
  suffices".
- REPORT.md Methods gains a subsection defining the probe references, the parallel-residual split
  $x_u = \text{rest}_u + m_u$, the transplant, and the two statistics $\rho_{\mathrm{donor}}$ /
  $\rho_{\mathrm{recip}}$ with equations; Results gains patterns 23 and 24; Summary, Conclusion and
  Limitations updated.
- **Recommended next experiment replaced** (old → new): "probe and transplant the block-0 MLP output"
  (now done) → "**how compressible is $m_u$?** — project $m_d - m_r$ onto the top $k$ principal
  components of $m$ across the 123 tokens, transplant only that, sweep $k$; a handful of directions
  reproducing slope +0.913 would make the trait a low-dimensional, monitorable feature."

**Implementation note.** `mlp_read.py` solves the ridge probe in its dual form (80×80 instead of
2049×2049 per ridge strength); the script asserts at run time that it matches
`embed_probe.probe` on the same splits, so the four probes are the same estimator as pattern 10's.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 28 display eqs / 630 inline eqs / 20 embeds / 0 problems; RESULTS 394 inline eqs / 20 embeds /
0 problems).

## 2026-08-12 — iteration 10 (part 3): how many directions of $m_u$ carry the width? None few enough

`experiments/mlp_rank.py` + `plot_mlp_rank.py`; results in `results/mlp_rank.json` / `.log`; new
figure `plots/mlp_rank.png` embedded as **Figure 21** in both deliverables (figure count 20 → 21).
This runs the experiment part 2 had named as next, so RESULTS.md's and REPORT.md's "next experiment"
sections are replaced again (see below).

**Added (new result, nothing superseded).** Partial transplant: project the donor–recipient difference
$m_d - m_r$ onto the top $k$ principal components of $m$ across the 123 tokens, transplant only that,
sweep $k$; controls = the bottom $k$ components and a random $k$-dimensional subspace.
- top 8 / 32 / 64 components (0.24 / 0.55 / 0.79 of the across-token variance) → transfer slope
  **+0.256 / +0.298 / +0.274**, against **+0.913** for the complete vector ($k = 122$);
- the discarded tail (bottom 58, 0.21 of the variance) transfers nothing (**−0.022**) and is nearly
  behaviourally inert (0.016 bits); a random 64-dim subspace gives **+0.000**;
- top-64 plus bottom-58 would give +0.25 if additive; intact gives +0.913;
- dissociation between damage and transfer: a top-64 transplant already causes 95% of the full
  transplant's output movement (0.713 of 0.750 bits) and inflates mean $\hat w$ to 0.613 (sd 0.060)
  from the unedited 0.565 (sd 0.084), while the complete transplant returns 0.573 (sd 0.076) — it
  exchanges widths instead of disturbing them.
- Conclusion recorded: the width trait is a property of the whole block-0 MLP output vector, not of a
  low-dimensional readable subspace — consistent with patterns 16–17 on the output side. Caveat stated:
  a truncated $m$ is off-manifold.
- REPORT.md Methods gains the partial-transplant definition with $m_{\mathrm{write}} = m_r + P_k(m_d -
  m_r)$; Results gains pattern 25; Summary, Conclusion and Limitations updated.

**Recommended next experiment replaced** (old → new): "how compressible is $m_u$?" (now answered: not
at all) → "**a second model**: repeat the cheap end of the pipeline on Pythia 410M or 2.8B — anchor
widths for ~60 tokens, embedding probe, block-0 MLP ablation — and compare the probe's held-out rho
(+0.76 here), the cross-model rank agreement of measured widths, and whether the block-0 MLP is again
the single carrier. This tests whether the free screen is a property of tokens or a per-model
calibration." Cost ~20 min GPU.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 29 display eqs / 669 inline eqs / 21 embeds / 0 problems; RESULTS 428 inline eqs / 21 embeds /
0 problems); 42 figure captions for 42 embeds across the two files.

## 2026-08-12 — iteration 11: the second model (and third, and fourth) — the trait is the token's

The experiment both deliverables named as next was run, and widened from one extra model to three:
`experiments/second_model.py` (per-model pipeline: anchor widths, embedding probe, block-level ablation)
on `pythia-160m/410m/1b-deduped` @ `step143000`, `experiments/second_ctrl.py` (per-token
movement-matched dose–response rerun on 410M, reusing `dose2.py`'s code unchanged),
`experiments/second_analysis.py` (reliability, disattenuated agreement, lookup transfer) and
`experiments/plot_second.py`. Results in `results/second_{160m,410m,1b}.json`,
`results/second_ctrl_410m.json`, `results/second_summary.json` (+ `.log`s). Three new figures —
`plots/cross_model.png`, `plots/second_repl.png`, `plots/second_ctrl.png` — embedded as **Figures
22–24** in both deliverables (figure count 21 → 24).

**Added to RESULTS.md and REPORT.md (new results; nothing superseded).**
- **Cross-model agreement, the headline.** Same 123 tokens, same 6 anchors, same 3 frames, same block-0
  site in each model. Pythia-410M / 1B / 1.4B rank the tokens at $\rho$ = **+0.884 / +0.898 / +0.890**
  (410M–1B) and, divided by each model's split-half (Spearman–Brown) reliability ceiling,
  **+0.995 / +0.989 / +0.977** — identical to within measurement noise. Reliabilities: 0.734 (160M),
  0.891 (410M), 0.932 (1B), 0.885 (1.4B).
- **Level vs ordering.** Median $\hat w_u$ falls 0.749 → 0.658 → 0.620 → 0.549 with size: transitions
  sharpen with scale while the ordering is preserved.
- **The free lookup transfers.** The probe read off 1.4B's embedding matrix ranks 410M's measured widths
  at **+0.760** and 1B's at **+0.745**, against **+0.765** on 1.4B itself. Refitting inside each model
  gives +0.774 (410M), +0.755 (1B) against +0.764 (1.4B).
- **160M is a genuine exception, not noise.** $\rho$ = +0.207 with 1.4B against a ceiling of 0.806
  (+0.256 corrected); the 1.4B lookup gives +0.043 (p = 0.63) and the refitted probe only
  +0.233 ± 0.104 (R² = −0.02). The trait is acquired between 160M and 410M.
- **Localisation replicates.** Mean-ablating every MLP and whole attention block in blocks 0–5: only the
  block-0 MLP collapses the across-token spread (0.169 → 0.023, 0.071 → 0.021, 0.096 → 0.019 for
  160M/410M/1B) and erases the ordering (+0.55 / −0.06 / −0.14), and it is again the only early
  component the model feels (0.40–0.45 bits vs ≤ 0.030 for every other).
- **But the matched-control margin does NOT replicate, and this is stated as a partial failure.** The
  410M rerun (9 doses × 3 seeds) reproduces the raw paired per-token effect (|Δŵ| 0.016 vs 0.008 at
  0.0010 bits, 0.049 vs 0.032 at 0.0074, 0.062 vs 0.048 at 0.0117; Wilcoxon p = 0.002 / 0.005 / 0.012)
  and the harder spread compression (sd 0.038 vs 0.051 at 0.026 bits), but the **level-free** paired
  test is null at all nine rungs (p ≥ 0.62 in the live band, vs p = 0.034 / 0.016 at 1.4B), the MLP arm
  is below its matched control in **9/18** rung × seed comparisons below 0.05 bits (chance), and the
  ρ = 0.6 crossing runs backwards (control 0.023 bits vs MLP 0.035, i.e. **0.66×** against 1.3× at
  1.4B). Both deliverables now say the *site* replicates and the *per-bit specificity* does not, and
  that the durable positive evidence for the component is the transplant (which needs no matched
  control).
- REPORT.md Methods gains a cross-model subsection defining the reliability correction
  $R_M = 2\rho_{\mathrm{half}}/(1+\rho_{\mathrm{half}})$, the disattenuated agreement
  $\rho^{*}_{AB}$ and the level-free per-token statistic $\Delta^{\mathrm{free}}_u$; Data & model names
  the three extra models; Results gains patterns 26–28; Summary, Conclusion and Limitations updated.

**Recommended next experiment replaced** (old → new): "test the lookup on a second model" (now done for
three) → "**where does the trait come from?** — measure anchor widths for the 123 tokens in Pythia-410M
at `step1000` / `step8000` / `step32000` / `step143000` and correlate each checkpoint's ranking with the
final one and with the token's unigram frequency and successor entropy. Early-and-sharpening ⇒ the
lookup is reading a corpus statistic computable with no model; late-and-gradual ⇒ it is reading what the
network learned about that token's successors." ~15 min GPU.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 32 display eqs / 769 inline eqs / 24 embeds / 0 problems; RESULTS 501 inline eqs / 24 embeds /
0 problems).

## 2026-08-12 — iteration 12: where the trait comes from — 17 training checkpoints

The experiment both deliverables named as next was run, and widened from 4 checkpoints to 17:
`experiments/checkpoints.py` (per-checkpoint anchor widths for the same 123 tokens × 6 anchors ×
3 frames at block 0, plus an embedding probe refitted inside each checkpoint) on
`pythia-410m-deduped` at `step0/2/8/16/32/64/128/256/512/1000/2000/4000/8000/16000/32000/64000/143000`,
`experiments/checkpoints_analysis.py` (split-half reliability, disattenuated agreement with the final
ranking, correlations with two corpus statistics, partial Spearman, rank $R^2$) and
`experiments/plot_checkpoints.py`. Results in `results/checkpoints.json`,
`results/checkpoints_summary.json` (+ `.log`s). Two new figures — `plots/ckpt_emergence.png`,
`plots/ckpt_source.png` — embedded as **Figures 25–26** in both deliverables (figure count 24 → 26).

**Added to RESULTS.md and REPORT.md (new results; nothing superseded).**
- **When: the ordering is learned in the first 512 of 143,000 steps.** At `step0` there is no ordering
  at all (across-token sd **0.003** vs 0.060 at the end, reliability 0.570, $\rho$ = **+0.015** with the
  final ranking), still true at `step16`. Agreement then runs +0.17 / +0.29 / +0.44 / +0.66 / **+0.79**
  at `step32/64/128/256/512` (**+0.87** after dividing by the noise ceiling), +0.94 by `step2000`, and
  does not change for the remaining 98.6% of training (+0.94 … +0.99).
- **Level and ordering separate in training too.** Median $\hat w_u$ 0.833 (`step256`) → 0.595
  (`step64000`) — sharpening continues two orders of magnitude after the ordering is fixed; the final
  checkpoint's 0.658 is the sweep's one non-monotone point.
- **What: two stages, and only the first is frequency.** $\rho(\hat w_u, \log_{10} N_u)$ = −0.39 /
  −0.63 / **−0.72** at `step32/64/128` — stronger than the finished model's −0.53 — while the agreement
  with the final ranking net of unigram count and successor entropy is zero there (−0.05 / −0.08 /
  +0.15). From `step256` the non-corpus component appears: partial agreement +0.45 → +0.60 (`step512`)
  → +0.75 (`step2000`) → +0.79–0.82. In the finished model the two corpus statistics explain
  $R^2_{\mathrm{corpus}}$ = **0.375** of the ranking's rank variance (0.378 in 1.4B).
- **The fixed 1.4B lookup reads a young checkpoint before that checkpoint's own embedding does.**
  +0.21 / +0.40 / +0.54 / +0.71 / +0.81 at `step32/64/128/256/512`, +0.77–0.84 later (best at
  `step2000`, +0.836, above the finished model's +0.760), while a probe refitted inside the checkpoint
  is at its shuffled-control level through `step256` and reaches its final +0.77–0.81 only from
  `step4000`.
- **Consistency check.** This sweep's `step143000` reproduces iteration 11's independent 410M run at
  $\rho$ = +1.0000 over the 123 tokens.
- Clarified an apparent inconsistency in the deliverables: the −0.33 frequency correlation reported in
  pattern 5 is for the *fitted* token effect $a_u$; the *measured* $\hat w_u$ tracks $\log_{10} N_u$ at
  −0.52 (1.4B) / −0.53 (410M) and successor entropy at −0.48 / −0.46. RESULTS.md's supporting-quantities
  table gains these three rows.
- REPORT.md Methods gains a checkpoint-sweep subsection defining $H_u$, the partial Spearman
  $\rho^{\mathrm{part}}$ and $R^2_{\mathrm{corpus}}$, and names the 17 revisions in Data & model;
  Results gains patterns 29–31; Summary, Conclusion and Limitations updated.

**Recommended next experiment replaced** (old → new): "when during training does the trait appear, and
is it a corpus statistic?" (now answered: by `step512`, and only partly) → "**a different tokenizer and
corpus**: measure anchor widths in `gpt2` for the token strings that are single tokens in both
vocabularies, same frames and anchor protocol, and compare the ranking with Pythia's against each
model's split-half reliability. Everything here holds token inventory and training data fixed, so this
is the last untested generalisation the deliverable rests on." ~10 min GPU. Cheaper mechanistic
follow-up named: transplant the final checkpoint's $m_u$ into the `step128` model and ask whether the
ordering appears.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 35 display eqs / 887 inline eqs / 26 embeds / 0 problems; RESULTS 633 inline eqs / 26 embeds /
0 problems); 52 figure captions for 52 embeds across the two files.

## 2026-08-12 — iteration 13: GPT-2 cross-tokenizer test (Figures 27–28); the ordering does NOT port

**Ran the experiment both deliverables named as next** (`experiments/envwidth.py`, `xmodel_width.py`,
`xmodel_analysis.py`, `gpt2_sites.py`, `xcurve_examples.py`, `plot_xmodel.py`; new
`results/xwidth_{gpt2,410m,1.4b}.json`, `xmodel_summary.json`, `gpt2_sites.json`, `xcurves.json`;
new `plots/xmodel_agreement.png`, `plots/gpt2_sites.png`). All 123 endpoint strings and all 6 anchor
strings are single tokens in GPT-2's vocabulary, so the comparison uses the same strings, anchors,
frames and block-0 site with no substitution.

**Result — negative, and it narrows a claim the report has been making since iteration 11.**
- GPT-2 ranks the 123 tokens at $\rho = -0.219$ with Pythia-1.4B and $-0.189$ with 410M (ceiling 0.53),
  against $+0.884$ (ceiling 0.888, disattenuated $+0.995$) between the two Pythias. Partialling out
  unigram count and successor entropy leaves it at $-0.211$.
- The fixed 1.4B embedding lookup ranks GPT-2's widths at $-0.200$, against $+0.760$/$+0.765$ on the
  Pythias. A probe refitted inside GPT-2 gives $+0.295$ against a shuffled control of $+0.275$ (null),
  against $+0.774$ (control $+0.032$) at 410M. GPT-2's widths do not even track $\log_{10} N_u$
  ($-0.038$ vs $-0.52$ in Pythia).
- Measurement failure comes first: 88.8% of GPT-2's block-0 curves are non-monotone (median backslide
  0.107, strict-validity 0.112 against 1.000 in both Pythias), and its split-half reliability is 0.319
  against 0.885/0.891.
- Site sweep (blocks 0/1/2/4/6/8) rules out the obvious confound: validity climbs 0.112 → 0.801 and
  the level 0.442 → 0.671, but reliability peaks at 0.462 and agreement with Pythia never exceeds
  $+0.141$ ($p = 0.12$); the negative sign at block 0 does not reproduce at any other site, so the
  reading is *no relationship*, not a reversed one.
- One replication: GPT-2's block-0 MLP is again the only early component the model registers
  (0.228 bits vs $\le 0.011$) and the only one that inflates the spread (0.116 → 0.201) and erases the
  ordering ($\rho = +0.06$) — suggestive at n = 12 and reliability 0.32, not established.

**New metric.** `w` is undefined for most GPT-2 curves, so both deliverables now define an **envelope
width** $\hat w^{\mathrm{env}}$ on the running maximum $e(t) = \max_{s\le t} d(s)$, which exists for
every curve and equals `w` exactly on monotone ones. Validated inside Pythia before use: rank
correlation with `w` = 1.0000 per curve and per token in both Pythia models (0.999998 per curve on
GPT-2's valid subset). Re-measuring 1.4B and 410M with it reproduces the existing numbers exactly
(median 0.549 / 0.658, probe $\rho = +0.764$ / $+0.774$, cross-model $\rho = +0.884$), so no previously
reported number changes.

**Deliverable changes.** RESULTS.md gains the section "GPT-2: the ordering is a property of the token
*in a training corpus*, not of the string" with Figures 27–28 and a six-row comparison table, a new
Headline paragraph stating the scope limit, and `gpt2` in the Setting line. REPORT.md gains a Methods
subsection (GPT-2 protocol, the envelope-width equations, the site and corpus-statistic controls), the
same Results section as patterns 32–34 with Figures 27–28, and a Summary paragraph. Both files now
carry 28 figures each.

**Claim narrowed** (old → new): "the ordering belongs to the token, the level to the network" (stated
without qualification) → "the ordering belongs to a token **as trained in a particular corpus**; the
practical screen is per-model, and the split-half reliability check — 0.89 where it works, 0.32 where
it does not, computable with no reference model — is the go/no-go test for porting it." The 160M floor
is re-read the same way: a fact about that training run rather than about parameter count.

**Recommended next experiment replaced** (old → new): "a different tokenizer and corpus" (now done and
negative) → "**separate two ways GPT-2 could fail**: compute edge drift `E` (movement of `d(t)` in the
outer 20% of the path; ~0 for a plateau, ~0.18 for a straight line) on the 2,214 curves already stored
per model and per site for GPT-2, 410M and 1.4B. Straight-line-like `E` in GPT-2 ⇒ it has no plateau to
measure and the negative is about plateau structure (predicting Pythia-160M looks the same);
Pythia-like `E` ⇒ genuine plateaus in a different token order, and the question becomes what that order
correlates with." No new forward passes.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 37 display eqs / 955 inline eqs / 28 embeds / 0 problems; RESULTS 684 inline eqs / 28 embeds /
0 problems); 56 figure captions for 56 embeds across the two files.

## 2026-08-12 — iteration 14: the two negatives split (edge drift), Figure 29 added

**What changed.** Ran the experiment both deliverables named as next: edge drift
`E = d(0.1) + (1 - d(0.9))` on the 2,214 stored curves of six configurations (GPT-2 blocks 0/4/8;
Pythia-160M/410M/1.4B at block 0), plus a curve-level filter that re-derives each token's width from
only its plateau-shaped curves. New code: `experiments/edgedrift.py` (run this iteration; also extended
to store per-curve widths), `edgedrift_analysis.py`, `plot_edgedrift.py`. New results:
`results/edgedrift.json`, `edgedrift_summary.json`. New figure: `plots/edgedrift.png` (Figure 29).
RESULTS.md gains a Headline paragraph and a new section with the `E` definition, Figure 29 and the
six-configuration table; REPORT.md gains a Methods subsection ("Is there a plateau to measure at all?
Edge drift"), a Results section with patterns 35–36 and Figure 29, a Summary paragraph, a Conclusion
paragraph (GPT-2 was previously absent from the Conclusion) and three new limitation clauses. Both
files now carry 29 figures each.

**Result superseded** (old → new): iteration 13's reading of the GPT-2 negative, "its measurement
reliability is 0.319, which caps any correlation at 0.53, so the correct reading is *no relationship*"
→ "scoring only its plateau-shaped curves (`E` ≤ 0.1, 56% of them) raises reliability **0.319 → 0.661**
and the ceiling **0.53 → 0.77**, while agreement with Pythia-1.4B stays at **−0.219 → −0.185**: GPT-2
has a reproducible width ordering of its own and it is unrelated to Pythia's." The GPT-2 section's
"What this costs the report" paragraph changed accordingly: the go/no-go reliability check must be
computed on plateau-shaped curves, and it is no longer the only check recommended.

**Prediction refuted** (old → new): iteration 13 predicted that if GPT-2 lacked plateau structure,
Pythia-160M — the size without the trait — would look the same. Neither half holds. GPT-2's curves are
plateau-shaped (median `E` 0.087 against Pythia-1.4B's 0.081, straight line 0.2), and **Pythia-160M is
the least plateau-shaped configuration measured** (0.183, 87% of curves above the 0.1 cut against 22%
of 1.4B's). The 160M floor is therefore re-read once more: not only a fact about that training run, but
one about a model whose transitions are close to straight ramps. Stated in both deliverables as a
correspondence between two measurements at one checkpoint each, not as a cause.

**Caveat added.** Within each Pythia, a token's edge drift and its width rank the tokens almost
identically (ρ = +0.93 / +0.96 / +0.97 at 160M / 410M / 1.4B) and transfer between 410M and 1.4B to the
same degree (+0.887 vs +0.884). Both deliverables now say the trait can equally be described as how
long the output stays put near the endpoints, and that this is one measurement rather than two.

**Recommended next experiment replaced** (old → new): "separate two ways GPT-2 could fail" (done) →
"**refit the ridge embedding probe inside GPT-2 against the plateau-filtered widths** (target
reliability 0.661 rather than 0.32), same 80/43 splits and shuffled-target control, read against the
new ceiling of 0.77 — the earlier probe null there was fitted against an unreliable target and said
little." No forward passes.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 38 display eqs / 1020 inline eqs / 29 embeds / 0 problems; RESULTS 1 display eq / 735 inline
eqs / 29 embeds / 0 problems). Determinism check: re-running `edgedrift.py` for Pythia-160M reproduced
its previously stored summary exactly (median `E` 0.1833, 0.868 of curves above 0.1).

---

## 2026-08-12 — iteration 15: the free lookup does not port to GPT-2; a control retracted

**New experiment.** `experiments/gpt2_embed_probe.py` (+ `plot_gpt2_probe.py`, `results/gpt2_probe.json`,
`results/gpt2_probe.log`, `plots/gpt2_probe.png`) refits the ridge embedding probe inside GPT-2 against
the plateau-filtered widths, as both deliverables named as the next step. Zero forward passes: stored
curves plus GPT-2's embedding matrix on CPU. **Figure 30** added to REPORT.md and RESULTS.md (both files
now 30 embeds, 30 captions).

**A control is retracted, and the claim it supported with it** (old → new). Old, in both deliverables:
"a probe refitted inside GPT-2 sits on its shuffled-target control (+0.295 against +0.275)". The +0.275
came from a SINGLE permuted target reused across all 50 splits. Re-run with 50 independent permutations,
the null is centred at −0.002 (sd 0.093, range −0.274 to +0.157) and the stored +0.275 — which
reproduces exactly — is the largest of the 50 draws. New: GPT-2's probe is above chance, +0.295 with
permutation p = 0.020. The cross-model table row in both files drops its parenthetical control values
(GPT-2 +0.295 / 410M +0.774 / 1.4B +0.764 now shown alone), and the Summary, Headline, pattern-32 prose
and Conclusion sentences that said "sits on its shuffled control" are rewritten. The other single-draw
controls (+0.032 at 410M, −0.201 at 1.4B) are unchanged in substance — those probes sit at +0.77 — and
are now labelled as one draw in RESULTS.md's Metrics table.

**New result, replacing the open question the previous iteration posed.** Against the plateau-filtered
target (reliability 0.661, ceiling √R = 0.813) GPT-2's probe reaches +0.244 ± 0.122, i.e. 0.30 of its
ceiling against Pythia-1.4B's 0.81, with held-out R² = −0.021 (permutation p = 0.020). Two supporting
measurements: the more reliable target scores 0.051 LOWER than the all-curve one (paired over the 50
shared splits, filtered ahead in 16/50, Wilcoxon p = 0.023), and a probe on two free corpus statistics
(log₁₀ N_u, successor entropy H_u) reaches +0.176 (p = 0.039), which 768 embedding dimensions beat by
only 0.067 ± 0.164 (34/50 splits, Wilcoxon p = 0.009). Cross-model: GPT-2's out-of-fold lookup ranks its
own filtered widths at +0.196 and Pythia-1.4B's at −0.174; the two models' lookups agree at −0.204.

**Where this landed in the deliverables.** REPORT.md gains a Results subsection "Does GPT-2's embedding
hold GPT-2's own widths?" (patterns 37 and 38, one table, Figure 30) after the edge-drift section, a
Methods paragraph defining the 50-permutation null and the two refits, and a new Conclusion paragraph.
RESULTS.md gains the condensed version of the same section plus the Metrics note. The auditor
recommendation grows from two checks to three: edge-drift distribution, split-half reliability on
plateau-shaped curves, and — only if a free lookup is wanted — a probe against a 50-draw permutation
null benchmarked on log₁₀ N_u and H_u.

**Recommended next experiment replaced** (old → new): "refit the probe inside GPT-2 against the
plateau-filtered widths" (done, above) → "**write the final checkpoint's block-0 MLP output m_u into the
`step128` model** and see whether the ordering appears — the one remaining test of direction rather than
correlation", with a cheap GPT-2 side experiment (probe edge drift E and filtered width separately, to
find which curve property its embedding holds).

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 38 display eqs / 1101 inline eqs / 30 embeds / 0 problems; RESULTS 1 display eq / 798 inline eqs
/ 30 embeds / 0 problems). The stored single-draw control was reproduced bit-for-bit (+0.275) before
being reinterpreted, which doubles as a determinism check on the probe code.

---

## 2026-08-12 — iteration 16: the cross-checkpoint transplant (Figure 31, patterns 39–40)

**New experiment** (`experiments/ckpt_transplant.py`, `ckpt_transplant_analysis.py`,
`ckpt_transplant_geom.py`, `plot_ckpt_transplant.py`; `results/ckpt_transplant{,_summary,_geom}.json`;
`plots/ckpt_transplant.png`). Pythia-410M-deduped, 123 tokens × 6 anchors × 3 frames, block-0 site.
Twelve full anchor-width sweeps: six write conditions (none / own m_u / donor m_u / donor norm-matched /
identity-shuffled / shuffled norm-matched) in each of two directions (step143000 → step128 and the
reverse). The `base` sweeps reproduce the stored checkpoint-sweep widths (median 0.819 sd 0.022 at
step128; 0.658 / 0.060 at step143000) and the `self` sweeps reproduce them exactly at 0.000 bits, which
is the hook sanity check.

**Added to REPORT.md:** a Methods subsection ("Can the block-0 MLP output install the ordering? The
cross-checkpoint transplant", after the checkpoint-sweep methods) defining the six conditions, κ, the
two scored agreements, the two partial correlations, the paired bootstrap, and the three geometry
measurements; a Results subsection with **Figure 31** carrying patterns 39 and 40; a Summary paragraph;
a Conclusion paragraph. **Added to RESULTS.md:** the corresponding section with Figure 31 and the
geometry table, plus the transplant in the Setting paragraph.

**Findings recorded.** Donor m_u into step128: agreement with the final ordering +0.329 (as measured)
and +0.189 (norm-matched, κ = 0.176) against −0.030 and −0.141 for the identity-shuffled write at equal
or larger output shift; gaps +0.357 [+0.151, +0.553] and +0.324 [+0.075, +0.572] (2,000-resample paired
bootstrap over tokens). Norm-matched erases the recipient's own ordering (−0.009) while keeping +0.189
with the final one; partial +0.240 removing the recipient's baseline and +0.272 also removing the donor
vector's length (‖m_u‖ ranks the final widths at −0.098). No transplant beats leaving step128 alone
(+0.443). Geometry: same-token cosine +0.178 (+0.198 centred), pairwise-arrangement agreement +0.031
(+0.096 centred), length ordering −0.043, median ‖m_u‖ 1.94 vs 11.06. Reverse direction reported as
uninformative: +1.000 → +0.148 own ordering, +0.027 with step128's, gap over control +0.088
[−0.091, +0.263], at 0.64 bits.

**Framing corrected (old → new).** Both deliverables and PLAN.md previously described `step128` as the
checkpoint "where the ordering does not yet exist". It is where the ordering is *half* present
(ρ = +0.443 with the final checkpoint, 0.50 of the 0.883 ceiling its own reliability allows) — the
number was already in the iteration-12 checkpoint table and the framing did not match it. Every
occurrence is now stated as the half-present figure.

**Claim narrowed (old → new).** "The transplant is decisive on sufficiency: one vector, m_u, carries
the whole trait" → unchanged as a statement about token-to-token substitution *inside one network*, and
now explicitly scoped that way, because the same substitution across two moments of training transfers
only +0.19 to +0.33 of the ordering and never beats doing nothing.

**Recommended next experiment replaced** (old → new): "write the final checkpoint's m_u into the
`step128` model" (done, above) → "**inside GPT-2, fit the same probe to edge drift E and to the
plateau-filtered width separately**", with the note that the Pythia causal line is closed because the
two checkpoints' m_u geometries are too different for any further cross-checkpoint edit to do better.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 40 display eqs / 1188 inline eqs / 31 embeds / 0 problems; RESULTS 1 display eq / 841 inline eqs
/ 31 embeds / 0 problems). Embed/caption counts match at 31 each; no bare `(plots/*.png)` references.

## 2026-08-12 — iteration 17: what the embedding lookup ranks (Figure 32, patterns 41–42)

**What was added.** `experiments/gpt2_shape_probe.py` and `experiments/plot_gpt2_shape.py`, writing
`results/gpt2_shape.json`, `results/gpt2_shape.log` and `plots/gpt2_shape.png` (Figure 32). Zero forward
passes: stored curves plus the two models' embedding matrices on CPU. This is the experiment both
deliverables named as next at iteration 16 — fit the same ridge probe to a token's edge drift E and to
its plateau-filtered width, on the same tokens and the same 50 splits — extended with the two residual
targets that actually separate the properties, and run in Pythia-1.4B as well as GPT-2.

**New in REPORT.md.** Methods subsection "Shape or width? Separating the two curve properties a probe
could be fitting" (targets E_u and w_u, the rank residuals, reliability of a residual target computed
inside each half, 2,000-token bootstrap intervals on every reliability, shared splits, 50-permutation
null with its p floor of 0.020). Results subsection "What does the embedding lookup actually rank — the
crossing, or the flat start?" with Figure 32, two tables and patterns 41–42.

**New in RESULTS.md.** Subsection "Shape or width: what the free lookup actually ranks" with Figure 32,
the target definitions, one combined table and the two findings.

**Numbers (all new).** Pythia-1.4B: shape probe +0.783 ± 0.046 (R = 0.859 [0.798, 0.901], 0.84 of
ceiling) vs width probe +0.658 ± 0.067 (R = 0.734 [0.598, 0.834], 0.77); paired +0.125, 47/50 splits,
p = 3.4e−14. Width with shape removed +0.072 ± 0.125, permutation p = 0.255 — not above chance — against
a reliably measured residual (R = 0.397 [0.098, 0.591]); shape with width removed +0.243 ± 0.102
(0.33 of ceiling, p = 0.020); the two residual probes differ by +0.171, 45/50 splits, p = 2.6e−11.
GPT-2: shape +0.216 ± 0.125 (R = 0.099 [−0.317, 0.374] — no ceiling quoted), width +0.244 ± 0.122
(0.30 of ceiling), width|shape +0.280 ± 0.107 (0.38 of ceiling, R = 0.543), shape|width +0.335 ± 0.106
(R = −0.155, no ceiling defined); all four permutation p = 0.020; shape − width paired −0.028, p = 0.40;
shape|width − width|shape +0.055, 31/50, p = 0.011. Targets rank the 123 tokens at ρ = +0.809
(Pythia-1.4B) and +0.537 (GPT-2) — lower than pattern 35's +0.967 and +0.770 because those use the
all-curve width, and both values are now stated side by side so the two tables cannot be read as
contradicting each other.

**Interpretation qualified, not retracted (old → new).** "The per-token number is largely readable from
the static embedding" (patterns 10, 38) → the practical claims are untouched (the lookup still ranks
measured widths at +0.76 and still predicts 718 unseen pairs at R² = 0.213), but what the probe reads is
now stated as curve *shape*: in Pythia-1.4B the width ordering is predicted only through its overlap
with shape, and the width-specific residual — a reliably measured quantity — is at chance. The
qualifier was added to the REPORT.md Summary, the REPORT.md Conclusion and the RESULTS.md Headline so no
part of either deliverable still describes the lookup as reading crossing width directly.

**Recommended next experiment replaced (old → new).** "Inside GPT-2, fit the same probe to edge drift E
and to the plateau-filtered width separately" (done, above) → "**repeat the within-model token-to-token
transplant of m_u and score the recipient's edge drift alongside its width**", because the two
transplant results (patterns 23, 39) were both scored on width alone and the property the embedding
carries one block earlier is shape.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 42 display eqs / 1300 inline eqs / 32 embeds / 0 problems; RESULTS 3 display eqs / 913 inline
eqs / 32 embeds / 0 problems). Embed/caption counts match at 32 each; no bare `(plots/*.png)`
references.
