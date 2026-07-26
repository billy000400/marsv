# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-26 — iteration 1: S1+S2+S3 complete in one pass

**Feedback check:** listed the direction root for `human_feedback*.md` / `*REVIEW*` without the
`.addressed.md` suffix — **none present**, so this iteration advanced the plan.

**What I did.**
- `experiments/common.py`: shared corrupted-MNIST dataset (fixed-seed sigma=0.3 Gaussian noise,
  clipped; clean 7x7 avg-pooled target), generic-head 4-layer MLP (dir12 config), dir12's SLERP +
  relative-distance code, and the frozen 90-pair bank (2 pairs per unordered digit pair; replica 0
  reproduces dir12's exact pair so `6 -> 7` etc. are the same images).
- `experiments/train.py`: trains classifier and regressor per seed with bit-identical init in the
  three shared layers (asserted at runtime) and identical batch order. Seeds 0/1/2, ~90 s per model.
- `experiments/probe.py`: runs the frozen probe (101-point SLERP at h1) for both models, all seeds.
- `experiments/analyze.py`: aggregates, bootstraps, writes all 8 PNGs.
- `experiments/control_earlystop.py`: the early-stopped-classifier control.

**What I learned.**
- The effect is much larger and cleaner than expected: 4.4-5.9x lower linearity deviation for the
  regressor, significant at every layer, 90/90 pairs, tight across seeds. No pair-level heterogeneity
  to report — the plan's "results vary strongly by pair or seed" branch does not apply.
- The `6 -> 7` output-space figure is the most legible evidence: the regressor morphs the digit
  continuously while the classifier snaps 6 -> 9 -> 7 with flat regions between.
- The regressor does **not** overfit under matched training (validation loss falls monotonically to a
  flat floor), so the plan's "slight overfitting" adequacy criterion is met by the classifier only.
  **Assumption logged:** I kept the training budget matched (same 30,000 steps) rather than training
  the regressor longer to force overfitting, because matched optimization is the more important
  control for this comparison. *Rejected alternatives:* (a) train the regressor to overfitting on a
  reduced subset — breaks the "same inputs, same data order" requirement; (b) train the regressor for
  more steps — breaks matched step count and would confound the comparison. Reported plainly as
  Limitation 2.
- The confound a reviewer would raise first — the classifier memorizes (train MSE ~1e-7) while the
  regressor does not — is now ruled out by the early-stop control rather than only discussed.
- Two metric-definition choices logged: PLAN.md's endpoint-normalized `d(alpha)` is reported as
  primary (it is what the plan literally specifies); dir12's fraction form is computed alongside as a
  robustness check and gives the same verdict.

**Next step.** The plan's success criterion is met (both models trained and verified, hand-selected
and aggregate d(alpha) results, reconstructions along the path, 3 seeds, clear verdict). Remaining
optional polish only: nothing is blocking. Since S1-S3 are all complete and no feedback file is
outstanding, this direction writes `STOP`.

On track? yes — S3 (aggregate & report) complete, 100% done, no blocker.
