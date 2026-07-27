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
