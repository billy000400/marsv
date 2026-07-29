# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-27 — iteration 1: whole experiment (S1→S4) run end to end

**Feedback check first (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` /
`*REVIEW*` without a `.addressed.md` suffix — none exist. Proceeded with the plan.

**What I did.**
- `experiments/common.py`: brightness dataset (L2-normalize MNIST, scale to $b\sim U(0.4,1)$), the five
  normalized tanh targets, the canonical 4-layer/width-200 ReLU MLP with a scalar head, the 201-point
  brightness sweep, and the movement/concentration reductions.
- `experiments/s1_dataset.py`: numerically verified $\lVert x_b\rVert_2 = b$ (max abs error
  $1.8\times10^{-7}$ across all splits), digit-balanced splits, and saved `plots/target_functions.png`.
- `experiments/train.py`: 5 $k$ × 3 seeds, identical data/init/batch-order across $k$; saves final and
  min-val checkpoints plus full loss histories, and computes the adequacy ratios.
- `experiments/analyze.py`: sweeps 100 held-out digit-balanced test images × 201 brightness values for
  all 15 models × 2 checkpoints; `experiments/plots.py`: all seven figures + summary tables.
- Wrote RESULTS.md and REPORT.md from the templates; `check_render.py` passes (10 display eqs, 7
  embedded figures each, 0 problems).

**What I learned.**
1. **The result.** Activation concentration does rise with target sharpness, monotonically and more
   steeply in deeper layers, but saturates far below the target: $R_3$ 1.094 → 1.455 while the target
   goes 1.01 → 2.70, and flank movement $F_3$ only 0.356 → 0.265 vs the target's 0.397 → 0.048. No
   plateau-transition-plateau structure at any $k$. Layer 1 is flat (1.006 → 1.015) at every $k$.
2. **Why the ceiling is partly a fitting limit.** The 1000-image models answer a near-binary target with
   a soft sigmoid (output concentration 1.80 vs target 2.70). Retraining on 10,000 images (same step
   count) pushes the output to 2.42 and $R_3$ to 1.823 ± 0.222 — a much larger effect. But $F_3$ is
   still 0.204, half the uniform baseline, so the verdict survives. This was worth the extra 5 minutes
   of compute: without it the report would have overclaimed a hard representational ceiling.
3. **Adequacy needed a schedule change, not a data change.** Constant-LR AdamW drives train loss to
   ~1e-8 and then oscillates over 5 orders of magnitude, failing the smooth-convergence condition
   ($\rho_{train}$ up to 8995). A cosine decay applied globally to all $k$ fixes it ($\rho_{train}$
   1.00–1.13) without touching anything else.
4. **Even the $k=0.5$ control is not perfectly uniform** — $s_3(b)$ peaks near $b\approx0.6$. That is a
   ReLU-under-rescaling baseline, not a target effect, which is why the report reads *changes* in
   $R_l$/$F_l$ with $k$ rather than absolute values.

**Assumptions logged (loop mode — could not ask).**
- *Training-set size 1000, not "all training images".* PLAN.md recommends all images but also gates on
  slight validation overfitting; a 1-D brightness target on 50k images does not overfit at all
  ($\rho_{val}\approx1.005$, confirmed empirically by the 10k run). I took the adequacy gate as binding
  since PLAN.md says "Do not report plateau results from an inadequately trained model", and 1000 is
  this project's canonical MNIST plateau setting. Rejected alternatives: (a) all 50k images as primary —
  fails the gate; (b) drop the gate — contradicts an explicit PLAN.md requirement. Mitigation: the 10k
  grid is run and reported in full as a secondary check, so a reader gets both.
- *Added metric $F_l(k)$* (flank movement fraction) beyond PLAN.md's list, because $R_l$ alone cannot
  distinguish "more concentrated" from "flat at the ends", and the decision rule asks specifically
  whether $k{=}10$ shows *low movement away from* $b_0$.
- *Probe images fixed across seeds* (seed only varies init, batch order, and brightness assignments), so
  the seed-level CI is a pure model-variation interval; per-image SD reported separately.

**Next step.** None — PLAN.md's five success criteria are all met (five settings trained, all 15 runs
adequate, REPORT.md contains every required element, 3 seeds with uncertainty, every metric defined and
plotted), and the decision rule returns a clear, documented verdict. Wrote `STOP`. If an operator drops
feedback later, delete `STOP`, address it, and re-write `STOP` only when clean again (CLAUDE.md rule 11).

On track? yes — S1–S4 complete, 100% done, no blocker; direction finished with a clear negative verdict.

## 2026-07-29 — iteration 2: operator feedback #1 — extend k to a step function, fix notation & tables

**Feedback check first (CLAUDE.md Part C).** Found `human_feedback_1.txt` (no `.addressed.md` suffix) —
addressing it WAS this iteration. Two asks: (1) the five $k$ values were too small, three of them
near-linear; add ~5 bigger ones, the largest looking like a step function, and rewrite the report;
(2) the tables are confusing, with duplicated column names, and the concentration score $R_l$ collides
visually with the linear-regression $R^2$ — use better notation.

**What I did.**
- Extended `C.K_VALUES` to $\{0.5,1,2,5,10,20,40,80,160,320\}$ and made `train.py --ks` / `analyze.py`
  (env `K_LIST`) merge into the existing JSON, so the five original settings did not need retraining.
- Trained 15 new models on the primary 1000-image grid and 15 on the 10,000-image grid (3 seeds each),
  then ran the full 201-point sweep on 100 held-out test images for both checkpoints of all 30. Total
  ~25 min of compute.
- Renamed the metrics throughout code and deliverables: $R_l \to \Gamma_l$ (concentration gain),
  $F_l \to \Phi_l$ (flank share). $R^2$ is now reserved for sweep goodness-of-fit alone.
- Split the one wide table into four single-metric tables with fully spelled column names.
- Rewrote `plots.py`: `cividis` for the 10 ordered $k$ levels (the 5-hue categorical palette can't carry
  10 series), 2x5 prediction panel grid, a transition-zoom panel on the target figure, and a new
  `saturation_and_fit.png`. Rewrote REPORT.md end to end and re-curated RESULTS.md.
- `check_render.py` passes: 12 display eqs, 8 embedded+captioned figures per deliverable, 0 problems.

**What I learned.**
1. **The operator's hypothesis was worth testing and it is now cleanly refuted.** With $k \le 10$ the
   deepest-layer curve was still rising, so "it would keep rising with sharper targets" was a live
   reading. Extending to $k=320$ shows it flat: $\Gamma_3$ = 1.491, 1.483, 1.468, 1.451, 1.458 across
   $k$ = 20→320 (16x sharpening), each inside the others' CI, while the target reference goes 4.17→5.00
   (its ceiling). $\Phi_3$ even *rises* slightly, 0.265→0.283.
2. **The 10k-image control became the headline, not a footnote.** At 1000 images the sharpest models
   only reach $R^2 = 0.61$, so the primary grid alone cannot separate "representation won't plateau"
   from "function never learned". At 10k images the *output* is a genuine switch ($\Gamma$ 4.13/5.00,
   $\Phi$ 0.005, $R^2$ 0.848) while layer 3 sits at 1.659/0.279. Output 78% of the way to a perfect
   plateau, representation 16%. That single comparison is the strongest thing this direction has.
3. **Stating the metric's ceiling changed how legible the result is.** $\Gamma_{\max}=5$ was always
   implied by the middle-20% window but never written down; once stated, "layer 3 is at 11% of the range"
   replaces a vague "far below the target".
4. **Very sharp targets break the adequacy gate's spirit, not its letter.** At $k=160,320$ the validation
   minimum lands at epoch 15 of 2000 — the model overfits immediately because 1000 images can't pin a
   step. $\rho_{\text{val}} \le 1.11$ so the gate passes, but the min-val checkpoint is untrained, which
   is why the Figure 7 curves diverge there. Flagged in both deliverables rather than buried.
5. **10 ordered series needed a colormap, not a categorical palette.** CLAUDE.md rule 13 caps the
   5-hue palette at 5 series; `cividis` + cycling line styles is the right call for an ordinal variable
   and stays CVD-safe and grayscale-readable.

**Assumptions logged (loop mode — could not ask).**
- *Chose $k \in \{20,40,80,160,320\}$ (geometric, 2x steps).* The operator asked for "another 5 bigger k,
  the biggest looking like a step function". Geometric spacing continues the existing 0.5→10 pattern and
  $k=320$ gives a transition width (0.0046) below the probe grid spacing (0.003), which is the strongest
  "step" this measurement can resolve. Rejected: (a) stopping at $k=100$ — the target's own $\Gamma$ is
  already at its 5.00 ceiling by $k=80$, so a couple of settings past the ceiling are needed to show the
  *model* saturating independently; (b) going past $k=320$ — the target would be narrower than one grid
  step, making the target reference a measurement artifact rather than a curve.
- *Kept 1000 images as the primary grid* (same reasoning as iteration 1: the pre-registered adequacy gate
  is binding), but promoted the 10k grid from "secondary check" to a named Results section, because at
  large $k$ it is what rules out the fitting confound. Both are reported in full.
- *Renamed to $\Gamma$/$\Phi$ rather than dropping subscripts.* Layer indices are needed; the collision
  the operator saw was specifically $R_2$ (layer 2) reading as $R^2$. Greek letters remove it entirely.

**Next step.** None required by PLAN.md — S1–S4 remain complete and the success criteria are met at the
extended 10-setting scale. Zero unaddressed feedback files remain, so `STOP` is written. If an operator
drops another feedback file, delete `STOP`, address it, and re-write `STOP` only when clean (rule 11).
The natural scientific follow-up, noted in REPORT.md, belongs to a new direction: reintroduce the
removed ingredients one at a time (softmax head, cross-entropy, multi-class competition) and see which
drives $\Phi_3$ toward zero.

On track? yes — S1–S4 complete at 10 sharpness settings, 100% done, no blocker; feedback #1 fully
addressed and the negative verdict is now much better supported.

## 2026-07-29 — iteration 3: operator feedback #2 — the transition at 30x resolution

**Feedback check first (CLAUDE.md Part C).** Listed the direction root: `human_feedback_2.txt` had no
`.addressed.md` suffix, so addressing it WAS this iteration. Ask: *"The current plots in report does not
show the most extreme situation, can you show what d(t) during digit transition looks like for
different K"*.

**Interpretation (logged because it required a judgment call).** This direction has no digit-to-digit
interpolation — PLAN.md explicitly puts that out of scope — so "d(t) during the transition" reads as the
local-movement curve $s_l(b)$ *inside the target transition*, the analogue of dir13's $d(t)$. And the
operator is right that no existing figure shows it: the probe grid's spacing (0.003) is coarser than the
$k=320$ target's transition width (0.0046), so the entire switch happens inside one plotted step.
Rejected alternatives: (a) doing a literal digit-identity interpolation — out of scope, and not what the
sharpness sweep is about; (b) simply re-cropping the existing Figure 4 x-axis — it would show a straight
line between two grid points and answer nothing.

**What I did.**
- `experiments/zoom.py`: dense 6001-point brightness sweep (spacing $10^{-4}$) of all 60 final
  checkpoints on the same 100 held-out probe images, streaming activations chunk-by-chunk so only
  movement norms are stored (~0.3 s/model, 7 MB RAM). Metrics: movement rate $g_l(b)$, scale-resolved
  $\Gamma_l(w)$ for $w \in \{0.06,0.03,0.01,0.005,0.0025\}$, and alignment-free $\Lambda_l(w)$.
- Retrained the 15 missing `ckpt10k` models ($k \le 10$, never saved in iteration 1) and re-ran
  `analyze.py` for them; they reproduce the published numbers exactly, so no table was superseded.
- `experiments/zoom_plots.py`: Figures 7-9. Curated RESULTS.md and REPORT.md (new §6, new table, updated
  Summary/Conclusion/Limitation 4, figure+table renumbering), appended CHANGELOG.

**What I learned.**
1. **Nothing is hiding under the coarse grid.** $\Gamma_3$ recomputed at 30x resolution moves by
   $\le 0.006$ at every $k$. The 201-point probe was not smearing a spike — a real possibility a priori,
   since chord lengths under-count a curved path.
2. **The alignment metric was the more valuable half, and I nearly missed it.** My first pass measured
   the peak of the *image-averaged* curve; the 10k output peaked at only 5.5x uniform there despite
   $\Gamma_{\text{out}} = 4.13$, which flagged that averaging over images whose switches sit at slightly
   different brightnesses flattens peaks. $\Lambda$ (best window anywhere, per image, then average) fixes
   it: the output goes to 11.9x, layer 3 only to 3.0x. Without this the zoom figures alone would have
   been open to "your averaging destroyed the spike".
3. **An honest correction to the previous iteration's headline.** Alignment-free, layer 3 does *not*
   stop responding past $k=20$ — $\Lambda_3$ creeps $1.887 \to 2.431$. Some of the $\Gamma_3$ saturation
   is transition drift off $b_0$. Deliverables now say "sharpens very slowly, two orders of magnitude
   below the target" instead of "stops responding entirely"; the verdict is unchanged because the
   output-to-representation gap *widens* over exactly that range.
4. **CLAUDE.md 8b bit again in a new spot:** `$2.5\%$` inline. GitHub strips the backslash, `%` opens a
   KaTeX comment and eats the closing `$`. `check_render.py` caught all six instances; percentages now
   live outside math.

**Next step.** None required by PLAN.md — S1-S4 remain complete, and feedback #2 is fully addressed and
renamed to `human_feedback_2.addressed.md`, leaving zero unaddressed feedback files, so `STOP` is
written. If another feedback file appears, delete `STOP`, address it, re-write `STOP` only when clean
(rules 10-11). The scientific follow-up is unchanged and belongs to a new direction: reintroduce the
removed ingredients (softmax head, cross-entropy, multi-class competition) one at a time and see which
drives $\Phi_3$ toward zero.

On track? yes — S1-S4 complete, 100% done, no blocker; feedback #2 addressed with a 30x-finer probe that
closes both the resolution and the alignment objection to the negative verdict.
