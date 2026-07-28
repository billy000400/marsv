# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-28 — S1 through S4 in one iteration; direction complete

**Feedback check.** Listed the direction root: no `human_feedback*.md` / `*REVIEW*` files at all, so
nothing to address; proceeded with the plan.

**What I did.** Ported dir16's five-file pipeline to the low-to-high task and ran the whole frozen
protocol end to end. Changes from dir16 were confined to what the plan asked for: input is now
`z = D(y)` (49 values, no Gaussian corruption) for *both* models, trunk is `49-200-200-200`, the
continuous head is 784 (clean image) instead of 49 (pooled image), and I added the `D`/`U`/`P`
operators, the removed-detail audit, and four frozen baselines. The SLERP probe, pair bank, metrics,
bootstrap, checkpoint rule and decision rule are dir16's, untouched.

- `audit.py` (S1): operator identities pass to <=2.4e-07, removed detail carries 39.1% of pixel
  energy, manifest + checksums frozen before training.
- `train.py` (S2): 3 seeds x 2 models, ~35 s each; bit-identical trunk init asserted every seed.
- `evaluate.py` (S2b): both task gates pass. Classifier 95.8/96.3/96.8%. Predictor `R2_detail`
  0.660 [0.654, 0.666].
- `probe.py` (S3): endpoint reproduction <=1.4e-06, every rerun bit-identical.
- `analyze.py` + `export_pairs.py` (S4): bootstrap, 10 figures, committed per-pair CSV.

**What I learned.**
1. The result replicates *and strengthens* under the harder task: classifier − predictor linearity
   deviation at hidden 3 is 0.1335 (4.9x) here versus 0.0990 (4.3x) in dir16, and the predictor is
   smoother on 90/90 pairs rather than 89/90. So the earlier finding was not an artifact of the
   continuous target being a downsampled copy of the input — the smoothing tracks target continuity,
   not information discarding. (Cross-direction numbers are context only, not a controlled contrast;
   the inputs differ by design. I kept that caveat in REPORT.md.)
2. The predictor beats the *privileged* digit template (given the true label) on detail MSE by a
   factor of 2.5. That was the check I was least sure would pass, and it is what lets the report say
   "instance-specific detail", not "learned ten prototypes". Worth having frozen it beforehand.
3. Figure 9 is the most informative single artifact: mid-path the predictor still emits sharp
   high-frequency detail, so its smoothness is not the model collapsing to a blur.

**Assumptions logged (loop mode, no human to ask).**
- Model naming `pre` (predictor) replaces dir16's `reg`; checkpoint/plot file names follow.
- For the pooled model-vs-baseline comparison I used the seed-averaged per-image error (matching how
  the interpolation aggregate averages seeds before bootstrapping pairs); per-seed values are in
  `results/task_quality.json`. Rejected: bootstrapping seeds as well (3 seeds is too few to
  resample) and reporting seed 0 only (throws away information).
- `.npz`/`.pt` are gitignored repo-wide, so the raw `d(alpha)` arrays stay local; I exported
  `results/per_pair_metrics.csv` (1,620 rows) so the per-pair LD/MJ behind every aggregate is
  committed. Rejected: force-adding 92 MB of probe arrays to git.

**Next step.** None — every success criterion in PLAN.md is satisfied, the render checker exits 0,
and no feedback file exists, so I wrote `STOP`. If an operator drops feedback later, the wrapper
re-enters: delete `STOP`, address the file, rename it `.addressed.md`, and only then re-write `STOP`.

On track? yes — S4 complete, 100% done, no blocker; verdict robust positive.
